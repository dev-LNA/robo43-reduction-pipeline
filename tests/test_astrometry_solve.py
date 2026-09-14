"""Tests for robo43.astrometry.solve.

Builds its own synthetic data and fake ``astrometry.Match``-shaped objects;
does not use tests/conftest.py. The real solver needs local astrometry.net
index files, so the one end-to-end test that exercises it is gated behind a
skip when the index directory is absent (mirroring how the rest of the suite
skips tests that need an external tool or dataset it cannot assume is
present).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from robo43.astrometry import solve as solve_mod
from robo43.astrometry.centroids import DetectedStar
from robo43.config import AstrometrySettings

REAL_INDEX_DIR = Path("/home/herpich/Documents/.astrometry")
REAL_PROC_DIR = Path("/home/herpich/Documents/data/ROBO43/mini_out/proc")


def _settings(**overrides) -> AstrometrySettings:
    base = {
        "index_dir": None,
        "search_radius_deg": 0.5,
        "scale_tolerance": 0.03,
        "frame_budget_s": 300.0,
        "max_sources": 150,
        "min_sources": 5,
        "rms_max_arcsec": 0.3,
        "sip_order": 2,
    }
    base.update(overrides)
    return AstrometrySettings(**base)


# --------------------------------------------------------------------------
# derive_index_scales
# --------------------------------------------------------------------------


def test_derive_index_scales_mini_sample_field_includes_4_5_6():
    # 4788 x 3194 px at 0.5277"/px, per the ROBO43 mini-sample _proc.fits.
    scales = solve_mod.derive_index_scales(42.1, 28.1)
    assert {4, 5, 6}.issubset(set(scales))
    assert 0 not in scales


def test_derive_index_scales_uses_the_smaller_dimension():
    # A field 100' x 10': only scales whose quad size overlaps [1.0, 10.0]'
    # (10% to 100% of the *smaller*, 10' dimension) should appear, regardless
    # of the much larger other dimension.
    scales = solve_mod.derive_index_scales(100.0, 10.0)
    for scale in scales:
        lo, hi = solve_mod._scale_range_arcmin(scale)
        assert hi > 1.0
        assert lo < 10.0


def test_derive_index_scales_small_field_excludes_large_scales():
    scales = solve_mod.derive_index_scales(3.0, 3.0)
    assert 15 not in scales
    assert 19 not in scales


def test_scale_range_doubles_every_two_steps():
    lo8, hi8 = solve_mod._scale_range_arcmin(8)
    lo10, hi10 = solve_mod._scale_range_arcmin(10)
    assert lo10 == pytest.approx(lo8 * 2.0)
    assert hi10 == pytest.approx(hi8 * 2.0)


# --------------------------------------------------------------------------
# solve_frame graceful-degradation paths (no real solver involved)
# --------------------------------------------------------------------------


def test_solve_frame_unsolved_when_index_dir_not_configured():
    sci = np.zeros((100, 100), dtype=np.float32)
    dq = np.zeros((100, 100), dtype=np.int32)
    solution = solve_mod.solve_frame(
        sci,
        dq,
        ra_deg=150.0,
        dec_deg=-10.0,
        pixel_scale_arcsec=0.5,
        seeing_fwhm_arcsec=4.0,
        settings=_settings(index_dir=None),
    )
    assert solution.solved is False
    assert "index_dir" in solution.failure_reason


def test_solve_frame_unsolved_when_no_index_files_cached(tmp_path):
    sci = np.zeros((100, 100), dtype=np.float32)
    dq = np.zeros((100, 100), dtype=np.int32)
    solution = solve_mod.solve_frame(
        sci,
        dq,
        ra_deg=150.0,
        dec_deg=-10.0,
        pixel_scale_arcsec=0.5,
        seeing_fwhm_arcsec=4.0,
        settings=_settings(index_dir=tmp_path),
    )
    assert solution.solved is False
    assert "no astrometry index files" in solution.failure_reason


def test_solve_frame_unsolved_for_invalid_pixel_scale():
    sci = np.zeros((100, 100), dtype=np.float32)
    dq = np.zeros((100, 100), dtype=np.int32)
    solution = solve_mod.solve_frame(
        sci,
        dq,
        ra_deg=150.0,
        dec_deg=-10.0,
        pixel_scale_arcsec=0.0,
        seeing_fwhm_arcsec=4.0,
        settings=_settings(index_dir="/tmp"),
    )
    assert solution.solved is False
    assert "pixel scale" in solution.failure_reason


# --------------------------------------------------------------------------
# _build_solution's residual-RMS / minimum-matched-sources quality gate
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeStar:
    ra_deg: float
    dec_deg: float


@dataclass(frozen=True)
class _FakeMatch:
    wcs_fields: dict
    stars: list
    logodds: float = 100.0
    scale_arcsec_per_pixel: float = 0.5
    center_ra_deg: float = 150.0
    center_dec_deg: float = -10.0


def _tan_wcs_fields(*, crval1: float, crval2: float, scale_arcsec: float) -> dict:
    scale_deg = scale_arcsec / 3600.0
    return {
        "WCSAXES": (2, ""),
        "CTYPE1": ("RA---TAN", ""),
        "CTYPE2": ("DEC--TAN", ""),
        "CUNIT1": ("deg", ""),
        "CUNIT2": ("deg", ""),
        "CRVAL1": (crval1, ""),
        "CRVAL2": (crval2, ""),
        "CRPIX1": (50.0, ""),
        "CRPIX2": (50.0, ""),
        "CD1_1": (-scale_deg, ""),
        "CD1_2": (0.0, ""),
        "CD2_1": (0.0, ""),
        "CD2_2": (scale_deg, ""),
    }


def _grid_stars(n: int) -> tuple[DetectedStar, ...]:
    return tuple(
        DetectedStar(
            x=30.0 + 10.0 * i,
            y=30.0 + 10.0 * i,
            flux=100.0,
            peak=100.0,
            sharpness=0.5,
            roundness=0.0,
        )
        for i in range(n)
    )


def _wcs_from_fields(wcs_fields: dict):
    from astropy.wcs import WCS

    header = fits.Header()
    for key, value in wcs_fields.items():
        header[key] = value
    return WCS(header)


def test_build_solution_accepts_a_perfect_match():
    wcs_fields = _tan_wcs_fields(crval1=150.0, crval2=-10.0, scale_arcsec=0.5)
    wcs = _wcs_from_fields(wcs_fields)
    stars = _grid_stars(6)
    ref_sky = wcs.pixel_to_world(
        np.array([s.x - 1.0 for s in stars]), np.array([s.y - 1.0 for s in stars])
    )
    match = _FakeMatch(
        wcs_fields=wcs_fields,
        stars=[
            _FakeStar(ra_deg=float(r), dec_deg=float(d))
            for r, d in zip(ref_sky.ra.deg, ref_sky.dec.deg, strict=True)
        ],
    )

    solution = solve_mod._build_solution(
        match,
        stars,
        rung=0,
        blind=False,
        series_name="5200",
        scales_used=(4, 5, 6),
        sip_order=2,
        settings=_settings(min_sources=5, rms_max_arcsec=0.3),
    )

    assert solution.solved is True
    assert solution.n_matched == 6
    assert solution.rms_arcsec == pytest.approx(0.0, abs=1e-6)
    assert solution.failure_reason is None
    assert len(solution.wcs_cards) > 0


def test_build_solution_rejects_when_rms_exceeds_gate():
    wcs_fields = _tan_wcs_fields(crval1=150.0, crval2=-10.0, scale_arcsec=0.5)
    wcs = _wcs_from_fields(wcs_fields)
    stars = _grid_stars(6)
    ref_sky = wcs.pixel_to_world(
        np.array([s.x - 1.0 for s in stars]), np.array([s.y - 1.0 for s in stars])
    )
    # Shift every reference star by ~1 arcsec in Dec: still within the 2.0"
    # match radius (so they still match), but well outside a 0.3" RMS gate.
    match = _FakeMatch(
        wcs_fields=wcs_fields,
        stars=[
            _FakeStar(ra_deg=float(r), dec_deg=float(d) + 1.0 / 3600.0)
            for r, d in zip(ref_sky.ra.deg, ref_sky.dec.deg, strict=True)
        ],
    )

    solution = solve_mod._build_solution(
        match,
        stars,
        rung=0,
        blind=False,
        series_name="5200",
        scales_used=(4, 5, 6),
        sip_order=2,
        settings=_settings(min_sources=5, rms_max_arcsec=0.3),
    )

    assert solution.solved is False
    assert solution.rms_arcsec is not None
    assert solution.rms_arcsec > 0.3
    assert "rms" in solution.failure_reason


def test_build_solution_rejects_when_matched_below_minimum():
    wcs_fields = _tan_wcs_fields(crval1=150.0, crval2=-10.0, scale_arcsec=0.5)
    stars = _grid_stars(6)
    # Reference stars nowhere near the detected stars: nothing matches.
    match = _FakeMatch(
        wcs_fields=wcs_fields,
        stars=[_FakeStar(ra_deg=10.0, dec_deg=10.0)],
    )

    solution = solve_mod._build_solution(
        match,
        stars,
        rung=0,
        blind=False,
        series_name="5200",
        scales_used=(4, 5, 6),
        sip_order=2,
        settings=_settings(min_sources=5, rms_max_arcsec=0.3),
    )

    assert solution.solved is False
    assert solution.n_matched == 0
    assert solution.rms_arcsec is None
    assert "no detected star matched" in solution.failure_reason


# --------------------------------------------------------------------------
# End-to-end real solve, gated on the local index cache being present.
# --------------------------------------------------------------------------


@pytest.mark.needs_astrometry_index
@pytest.mark.skipif(
    not REAL_INDEX_DIR.is_dir(), reason="local astrometry.net index cache not present"
)
def test_solve_frame_end_to_end_on_mini_sample():
    paths = sorted(REAL_PROC_DIR.glob("*_proc.fits"))
    if not paths:
        pytest.skip("ROBO43 mini-sample _proc.fits files not present on this machine")

    with fits.open(paths[0]) as hdul:
        sci = np.array(hdul["SCI"].data, dtype=np.float32)
        dq = np.array(hdul["DQ"].data, dtype=np.int32)
        header = hdul["SCI"].header

    # XPIXSZ is already the binned pixel size for this instrument (verified
    # against the documented 0.5277"/px); do not also multiply by XBINNING.
    pixel_scale_arcsec = 206.265 * header["XPIXSZ"] / header["FOCALLEN"]
    settings = _settings(
        index_dir=REAL_INDEX_DIR,
        min_sources=20,
        rms_max_arcsec=0.3,
        sip_order=3,
    )

    solution = solve_mod.solve_frame(
        sci,
        dq,
        ra_deg=header["RA"],
        dec_deg=header["DEC"],
        pixel_scale_arcsec=pixel_scale_arcsec,
        seeing_fwhm_arcsec=4.0,
        settings=settings,
    )

    assert solution.solved is True
    assert solution.rms_arcsec is not None
    assert solution.rms_arcsec <= 0.3
    assert solution.index_series == "5200"
