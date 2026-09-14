"""Tests for robo43.photometry.quality.

Builds its own in-memory numpy/astropy fixtures; does not use tests/conftest.py.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from robo43.config import CalibSettings, PhotometrySettings, Settings
from robo43.exceptions import PhotometryError
from robo43.instruments.profile import InstrumentLayout, InstrumentProfile
from robo43.io.meta import FrameMeta
from robo43.photometry.quality import Reject, assess, reference_sample


def _profile(**overrides) -> InstrumentProfile:
    defaults = {
        "name": "test",
        "matches": {},
        "pixel_scale_arcsec": 1.0,
        "gain_e_per_adu": 2.0,
        "read_noise_e": 5.0,
        "saturation_adu": 50000.0,
        "nonlinearity_adu": 40000.0,
        "binning_keyword": "XBINNING",
        "layout": InstrumentLayout(),
        "valid_filters": (),
        "filter_map": {},
        "keywords": {},
        "imagetyp_map": {},
        "site_lat_deg": None,
        "site_lon_deg": None,
        "site_elevation_m": None,
        "index_scales": None,
        "search_radius_deg": 0.5,
    }
    defaults.update(overrides)
    return InstrumentProfile(**defaults)


def _meta(*, naxis1=512, naxis2=512, pixel_scale=1.0, saturation=50000.0,
          **overrides) -> FrameMeta:
    defaults = {
        "path": Path("test.fits"),
        "naxis1": naxis1, "naxis2": naxis2,
        "frame_type": "science", "exptime": 30.0, "date_obs": None,
        "filter": "V", "ra_deg": None, "dec_deg": None,
        "pixel_scale": pixel_scale, "ccd_temp": None,
        "gain_setting": None, "read_noise": 5.0, "saturation": saturation, "airmass": None,
        "object": "test", "telescope": "robo43", "instrument": "robo43", "observer": None,
        "binning": 1, "readout_mode": None, "focal_length_mm": None,
        "site_lat_deg": None, "site_lon_deg": None, "site_elevation_m": None,
        "mjd": None, "night": None, "is_valid": True, "profile_name": "test", "origins": {},
    }
    defaults.update(overrides)
    return FrameMeta(**defaults)


def _settings(*, apply_gain=True, **photometry_overrides) -> Settings:
    return Settings(
        photometry=PhotometrySettings(**photometry_overrides),
        calib=CalibSettings(apply_gain=apply_gain),
    )


def _row(**overrides):
    base = {
        "X_IMAGE": 256.0, "Y_IMAGE": 256.0,
        "FLUX_AUTO": 10000.0, "FLUXERR_AUTO": 50.0,   # SNR=200
        "FLAGS": 0,
        "FWHM_IMAGE": 3.0,                          # 3.0 arcsec at pixel_scale=1.0
        "IMAFLAGS_ISO": 0,
        "FLUX_MAX": 1000.0,                         # far below both thresholds
        "ELLIPTICITY": 0.1,
        "CLASS_STAR": 0.95,
    }
    base.update(overrides)
    return base


def test_one_row_per_class():
    # Rows are spread out in Y (50px apart, well beyond the 3px isolation
    # radius) so that none of them accidentally trips BLENDED against
    # another row in this same table.
    rows = [
        _row(Y_IMAGE=100.0),
        _row(Y_IMAGE=150.0, FLAGS=1),
        _row(Y_IMAGE=200.0, IMAFLAGS_ISO=1),
        _row(Y_IMAGE=250.0, X_IMAGE=10.0),
        _row(Y_IMAGE=300.0, FWHM_IMAGE=0.1),
        _row(Y_IMAGE=350.0, FWHM_IMAGE=50.0),
        _row(Y_IMAGE=400.0, ELLIPTICITY=0.5),
        _row(Y_IMAGE=430.0, FLUX_AUTO=10000.0, FLUXERR_AUTO=5000.0),
        _row(Y_IMAGE=460.0, CLASS_STAR=0.1),
    ]
    expected = [
        Reject.NONE,
        Reject.FLAGS,
        Reject.BADPIX,
        Reject.EDGE,
        Reject.FWHM_LOW,
        Reject.FWHM_HIGH,
        Reject.ELONGATED,
        Reject.LOW_SNR,
        Reject.NOT_STELLAR,
    ]
    table = Table(rows)
    report = assess(table, settings=_settings(), profile=_profile(), meta=_meta())
    for i, exp in enumerate(expected):
        assert report.mask[i] == int(exp), f"row {i} expected {exp!r}, got {report.mask[i]!r}"

    assert report.n_clean == 1
    new_total = sum(v for k, v in report.counts.items() if k.endswith("_new"))
    assert new_total + report.n_clean == report.n_input


def test_saturated_only():
    profile = _profile(nonlinearity_adu=None)
    table = Table([_row(FLUX_MAX=95000.0)])
    report = assess(
        table, settings=_settings(), profile=profile, meta=_meta(), gain_e_per_adu=2.0
    )
    assert report.mask[0] == int(Reject.SATURATED)
    assert "NONLINEAR" in report.skipped


def test_nonlinear_only():
    table = Table([_row(FLUX_MAX=70000.0)])
    report = assess(
        table, settings=_settings(), profile=_profile(), meta=_meta(), gain_e_per_adu=2.0
    )
    assert report.mask[0] == int(Reject.NONLINEAR)
    assert not (report.mask[0] & int(Reject.SATURATED))


def test_saturated_and_nonlinear_skipped_without_a_measured_gain():
    """apply_gain=True but no gain_e_per_adu supplied (e.g. unmeasured setting):
    the cuts are skipped, never guessed at via the profile or the GAIN header."""
    profile = _profile(saturation_adu=50000.0, nonlinearity_adu=40000.0)
    table = Table([_row(FLUX_MAX=95000.0)])
    report = assess(table, settings=_settings(apply_gain=True), profile=profile, meta=_meta())
    assert not (report.mask[0] & int(Reject.SATURATED))
    assert not (report.mask[0] & int(Reject.NONLINEAR))
    assert "SATURATED" in report.skipped
    assert "NONLINEAR" in report.skipped


def test_saturation_both_unit_conventions_agree():
    profile = _profile(saturation_adu=50000.0, gain_e_per_adu=2.0)

    settings_e = _settings(apply_gain=True)
    meta_e = _meta(saturation=50000.0)
    table_e = Table([_row(FLUX_MAX=95000.0)])
    report_e = assess(
        table_e, settings=settings_e, profile=profile, meta=meta_e, gain_e_per_adu=2.0
    )

    settings_adu = _settings(apply_gain=False)
    meta_adu = _meta(saturation=50000.0)
    table_adu = Table([_row(FLUX_MAX=47500.0)])
    report_adu = assess(table_adu, settings=settings_adu, profile=profile, meta=meta_adu)

    assert report_e.mask[0] & int(Reject.SATURATED)
    assert report_adu.mask[0] & int(Reject.SATURATED)

    table_e_safe = Table([_row(FLUX_MAX=95000.0 * 0.5)])
    report_e_safe = assess(
        table_e_safe, settings=settings_e, profile=profile, meta=meta_e, gain_e_per_adu=2.0
    )
    table_adu_safe = Table([_row(FLUX_MAX=47500.0 * 0.5)])
    report_adu_safe = assess(table_adu_safe, settings=settings_adu, profile=profile, meta=meta_adu)

    assert not (report_e_safe.mask[0] & int(Reject.SATURATED))
    assert not (report_adu_safe.mask[0] & int(Reject.SATURATED))


def test_blended():
    rows = [
        _row(X_IMAGE=300.0, Y_IMAGE=300.0),
        _row(X_IMAGE=301.0, Y_IMAGE=300.0),
        _row(X_IMAGE=256.0, Y_IMAGE=256.0),
    ]
    table = Table(rows)
    report = assess(table, settings=_settings(), profile=_profile(), meta=_meta())
    assert report.mask[0] == int(Reject.BLENDED)
    assert report.mask[1] == int(Reject.BLENDED)
    assert report.mask[2] == int(Reject.NONE)


def test_large_table_performance_and_correctness():
    # NOTE: the brief's suggested (50, 4000) range on a 4096x4096 frame is
    # NOT sparse enough -- at that density a Poisson field of 20000 points
    # has an *expected* ~700 pairs closer than the 3px isolation radius
    # (mean nearest-neighbour distance ~14px, but the near-neighbour tail is
    # what matters here), confirmed empirically. A much larger coordinate
    # range is needed to make accidental collisions genuinely rare; (50,
    # 100000) with seed 0 gives zero accidental collisions in practice.
    rng = np.random.default_rng(0)
    n = 20000
    x = rng.uniform(50, 100000, size=n)
    y = rng.uniform(50, 100000, size=n)

    # sanity check the field is sparse enough for random collisions to be rare
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([x, y]))
    dist, _ = tree.query(np.column_stack([x, y]), k=2)
    nearest = dist[:, 1]
    assert np.count_nonzero(nearest < 3.0) == 0

    # inject one deliberate close pair
    x[0], y[0] = 50000.0, 50000.0
    x[1], y[1] = 50001.0, 50000.0

    rows = [_row(X_IMAGE=float(xi), Y_IMAGE=float(yi)) for xi, yi in zip(x, y, strict=False)]
    table = Table(rows)

    meta = _meta(naxis1=100050, naxis2=100050)
    start = time.perf_counter()
    report = assess(table, settings=_settings(), profile=_profile(), meta=meta)
    elapsed = time.perf_counter() - start

    assert elapsed < 15.0
    assert report.mask[0] & int(Reject.BLENDED)
    assert report.mask[1] & int(Reject.BLENDED)


def test_absolute_fwhm_window_legacy_pixel_scale():
    # The brief's original example used FWHM_IMAGE=1.0 px to test the low-side
    # rejection at pixel_scale=0.529, but 1*0.529=0.529 > 0.5 (fwhm_min_arcsec)
    # so it does NOT trip FWHM_LOW. Using 0.5 px instead: 0.5*0.529=0.2645 < 0.5.
    assert 1 * 0.529 > 0.5
    assert 0.5 * 0.529 < 0.5

    meta = _meta(pixel_scale=0.529)
    settings = _settings()
    profile = _profile()

    table_low = Table([_row(FWHM_IMAGE=0.5)])
    report_low = assess(table_low, settings=settings, profile=profile, meta=meta)
    assert report_low.mask[0] & int(Reject.FWHM_LOW)

    table_high = Table([_row(FWHM_IMAGE=20.0)])
    report_high = assess(table_high, settings=settings, profile=profile, meta=meta)
    assert report_high.mask[0] & int(Reject.FWHM_HIGH)

    table_ok = Table([_row(FWHM_IMAGE=5.0)])
    report_ok = assess(table_ok, settings=settings, profile=profile, meta=meta)
    assert not (report_ok.mask[0] & (int(Reject.FWHM_LOW) | int(Reject.FWHM_HIGH)))


def test_no_gaia_match_and_spurious_rate():
    rows = [
        _row(X_IMAGE=100.0, Y_IMAGE=100.0),
        _row(X_IMAGE=200.0, Y_IMAGE=100.0),
        _row(X_IMAGE=300.0, Y_IMAGE=100.0),
        _row(X_IMAGE=400.0, Y_IMAGE=100.0),
    ]
    table = Table(rows)
    table["ALPHA_J2000"] = [10.0, 10.001, 10.5, 10.6]
    table["DELTA_J2000"] = [20.0, 20.0, 20.0, 20.0]

    reference = Table()
    reference["RA"] = [10.0, 10.001]
    reference["DEC"] = [20.0, 20.0]

    report = assess(
        table, settings=_settings(), profile=_profile(), meta=_meta(), reference=reference
    )
    assert not (report.mask[0] & int(Reject.NO_GAIA_MATCH))
    assert not (report.mask[1] & int(Reject.NO_GAIA_MATCH))
    assert report.mask[2] & int(Reject.NO_GAIA_MATCH)
    assert report.mask[3] & int(Reject.NO_GAIA_MATCH)
    assert report.spurious_rate == pytest.approx(2 / 4)


def test_two_classes_both_bits_survive():
    table = Table([_row(FLAGS=1, CLASS_STAR=0.1)])
    report = assess(table, settings=_settings(), profile=_profile(), meta=_meta())
    assert report.mask[0] == int(Reject.FLAGS | Reject.NOT_STELLAR)


def test_excellent_catalogue_rejects_nobody():
    rng = np.random.default_rng(2)
    n = 50
    ncols = 10
    nrows = 5
    xs, ys = np.meshgrid(
        50 + 50 * np.arange(ncols), 50 + 50 * np.arange(nrows)
    )
    xs = xs.ravel().astype(float)[:n]
    ys = ys.ravel().astype(float)[:n]

    rows = []
    for xi, yi in zip(xs, ys, strict=False):
        rows.append(
            _row(
                X_IMAGE=xi,
                Y_IMAGE=yi,
                FLUX_AUTO=10000.0 * (1 + rng.uniform(-0.1, 0.1)),
                FWHM_IMAGE=3.0 + rng.uniform(-0.2, 0.2),
                ELLIPTICITY=0.1 + rng.uniform(-0.05, 0.05),
                CLASS_STAR=0.95 - rng.uniform(0, 0.1),
            )
        )
    table = Table(rows)
    meta = _meta(naxis1=2000, naxis2=2000)
    report = assess(table, settings=_settings(), profile=_profile(), meta=meta)

    # The legacy FLUX_AUTO > percentile(80) cut would have discarded 40 of
    # these 50 excellent sources regardless of quality; this module's
    # absolute config-driven cuts do not.
    assert report.n_clean == 50
    assert np.all(report.mask == int(Reject.NONE))


def test_ellipticity_and_elongation_agree():
    rows = [
        _row(X_IMAGE=100.0, Y_IMAGE=100.0, ELLIPTICITY=0.1),
        _row(X_IMAGE=200.0, Y_IMAGE=100.0, ELLIPTICITY=0.5),
        _row(X_IMAGE=300.0, Y_IMAGE=100.0, ELLIPTICITY=0.29),
    ]
    table_ellip = Table(rows)

    rows_elong = []
    for r in rows:
        ellipticity = r.pop("ELLIPTICITY")
        r["ELONGATION"] = 1 / (1 - ellipticity)
        rows_elong.append(r)
    table_elong = Table(rows_elong)

    settings = _settings()
    profile = _profile()
    meta = _meta()

    report_ellip = assess(table_ellip, settings=settings, profile=profile, meta=meta)
    report_elong = assess(table_elong, settings=settings, profile=profile, meta=meta)

    for i in range(len(rows)):
        assert (report_ellip.mask[i] & int(Reject.ELONGATED)) == (
            report_elong.mask[i] & int(Reject.ELONGATED)
        )
    assert report_ellip.median_ellipticity == pytest.approx(
        report_elong.median_ellipticity, abs=1e-9
    )


def test_skipped_cuts_recorded():
    row = {
        "X_IMAGE": 256.0, "Y_IMAGE": 256.0,
        "FLUX_AUTO": 10000.0, "FLUXERR_AUTO": 50.0,
        "FLAGS": 0,
        "FWHM_IMAGE": 3.0,
    }
    table = Table([row])
    meta = _meta(pixel_scale=None)
    report = assess(table, settings=_settings(), profile=_profile(), meta=meta, reference=None)

    optional_cuts = {
        "BADPIX", "SATURATED", "NONLINEAR", "FWHM_LOW", "FWHM_HIGH",
        "ELONGATED", "NOT_STELLAR", "BLENDED", "NO_GAIA_MATCH",
    }
    for name in optional_cuts:
        assert name in report.skipped
        assert name not in report.counts
        assert f"{name}_new" not in report.counts

    for name in ("FLAGS", "EDGE", "LOW_SNR"):
        assert name in report.counts

    summary = report.text_summary()
    assert isinstance(summary, str)
    assert summary != ""


def test_empty_table():
    table = Table(
        names=["X_IMAGE", "Y_IMAGE", "FLUX_AUTO", "FLUXERR_AUTO", "FLAGS", "FWHM_IMAGE"],
        dtype=[float, float, float, float, int, float],
    )
    report = assess(table, settings=_settings(), profile=_profile(), meta=_meta())
    assert report.n_input == 0
    assert report.n_clean == 0
    assert report.mask.shape == (0,)


def test_missing_column_raises():
    row = _row()
    del row["FWHM_IMAGE"]
    table = Table([row])
    with pytest.raises(PhotometryError, match="FWHM_IMAGE"):
        assess(table, settings=_settings(), profile=_profile(), meta=_meta())


def test_reference_sample_filters_flags_and_mask():
    rows = [
        _row(X_IMAGE=100.0, Y_IMAGE=100.0),
        _row(X_IMAGE=200.0, Y_IMAGE=100.0, CLASS_STAR=0.1),
        _row(X_IMAGE=300.0, Y_IMAGE=100.0, FLAGS=1),
    ]
    table = Table(rows)
    settings = _settings()
    report = assess(table, settings=settings, profile=_profile(), meta=_meta())
    sample = reference_sample(table, report, settings=settings)
    assert len(sample) == 1
    assert sample["X_IMAGE"][0] == 100.0


def test_non_square_frame_edge_rejection():
    meta = _meta(naxis1=200, naxis2=100)
    rows = [
        _row(X_IMAGE=190.0, Y_IMAGE=50.0),
        _row(X_IMAGE=100.0, Y_IMAGE=95.0),
        _row(X_IMAGE=100.0, Y_IMAGE=50.0),
    ]
    table = Table(rows)
    report = assess(table, settings=_settings(), profile=_profile(), meta=meta)
    assert report.mask[0] == int(Reject.EDGE)
    assert report.mask[1] == int(Reject.EDGE)
    assert report.mask[2] == int(Reject.NONE)
