"""Tests for robo43.photometry.catalog.

Builds its own synthetic frames and hand-built Settings/InstrumentProfile
objects, following the tests/test_gain.py / tests/test_mef.py convention:
this repo has no tests/conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from robo43.config import CalibSettings, PhotometrySettings, Settings
from robo43.exceptions import PhotometryError
from robo43.instruments.profile import InstrumentLayout, InstrumentProfile
from robo43.photometry.catalog import (
    FLAG_DEBLENDED,
    FLAG_NEIGHBOR,
    FLAG_SATURATED,
    PhotometryResult,
    measure_sources,
    pixel_scale_arcsec,
)
from robo43.products.catalog import CATALOG_COLUMNS
from robo43.products.dq import DQ, DQ_DTYPE

SHAPE = (300, 300)
TRUE_FLUX = 200000.0
TRUE_FWHM_PX = 4.0
STAR_X0 = 100.0  # 0-based injection coordinate
STAR_Y0 = 200.0  # 0-based injection coordinate
BACKGROUND_LEVEL = 500.0
READ_NOISE_E = 7.0

ALIAS_TO_CANONICAL = {
    "RA": "ALPHA_J2000",
    "DEC": "DELTA_J2000",
    "FLUX": "FLUX_AUTO",
    "FLUX_ERR": "FLUXERR_AUTO",
    "MAG": "MAG_AUTO",
    "MAG_ERR": "MAGERR_AUTO",
}


def _profile(**overrides) -> InstrumentProfile:
    defaults = dict(
        name="test-instrument",
        matches={},
        pixel_scale_arcsec=0.5,
        gain_e_per_adu=0.7,
        read_noise_e=READ_NOISE_E,
        saturation_adu=55000.0,
        nonlinearity_adu=50000.0,
        binning_keyword="XBINNING",
        layout=InstrumentLayout(),
        valid_filters=(),
        filter_map={},
        keywords={},
        imagetyp_map={},
        site_lat_deg=None,
        site_lon_deg=None,
        site_elevation_m=None,
        index_scales=None,
        search_radius_deg=0.5,
    )
    defaults.update(overrides)
    return InstrumentProfile(**defaults)


def _settings(**photometry_overrides) -> Settings:
    defaults = dict(
        back_size=32,
        back_filtersize=3,
        detect_thresh=3.0,
        detect_minarea=5,
        deblend_nthresh=32,
        deblend_mincont=0.005,
        seeing_fwhm_arcsec=2.0,
        aperture_radii_arcsec=(1.0, 1.5, 2.0),
        instrumental_zp=25.0,
        edge_margin_px=10,
    )
    defaults.update(photometry_overrides)
    return Settings(photometry=PhotometrySettings(**defaults), calib=CalibSettings())


def _gaussian(total_flux: float, fwhm_px: float, x0: float, y0: float) -> np.ndarray:
    yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    sigma = fwhm_px / 2.3548
    amp = total_flux / (2 * np.pi * sigma**2)
    return amp * np.exp(-(((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma**2)))


def _header() -> fits.Header:
    header = fits.Header()
    header["ASTRSCL"] = 0.5
    return header


def _blank_frame(*, seed: int = 1):
    rng = np.random.default_rng(seed)
    sci = rng.normal(BACKGROUND_LEVEL, READ_NOISE_E, size=SHAPE).astype(np.float32)
    err = np.full(SHAPE, READ_NOISE_E, dtype=np.float32)
    dq = np.zeros(SHAPE, dtype=DQ_DTYPE)
    return sci, err, dq


def _frame_with_star(*, seed: int = 42, dq: np.ndarray | None = None):
    sci, err, base_dq = _blank_frame(seed=seed)
    if dq is None:
        dq = base_dq
    star = _gaussian(TRUE_FLUX, TRUE_FWHM_PX, STAR_X0, STAR_Y0)
    sci = (sci + star).astype(np.float32)
    return sci, err, dq


def _brightest_row(result: PhotometryResult) -> int:
    return int(np.argmax(result.table["FLUX_AUTO"]))


def test_flux_auto_recovers_known_flux_for_isolated_star():
    sci, err, dq = _frame_with_star()
    result = measure_sources(
        sci, err, dq, header=_header(), settings=_settings(), profile=_profile()
    )
    assert result.n_sources >= 1
    i = _brightest_row(result)
    assert result.table["FLUX_AUTO"][i] == pytest.approx(TRUE_FLUX, rel=0.05)


def test_mag_auto_matches_zeropoint_formula_exactly():
    sci, err, dq = _frame_with_star()
    settings = _settings()
    result = measure_sources(
        sci, err, dq, header=_header(), settings=settings, profile=_profile()
    )
    i = _brightest_row(result)
    flux = float(result.table["FLUX_AUTO"][i])
    expected_mag = settings.photometry.instrumental_zp - 2.5 * np.log10(flux)
    assert float(result.table["MAG_AUTO"][i]) == pytest.approx(expected_mag, abs=1e-4)


def test_fwhm_image_recovers_known_fwhm():
    sci, err, dq = _frame_with_star()
    result = measure_sources(
        sci, err, dq, header=_header(), settings=_settings(), profile=_profile()
    )
    i = _brightest_row(result)
    assert result.table["FWHM_IMAGE"][i] == pytest.approx(TRUE_FWHM_PX, rel=0.15)


def test_x_image_y_image_are_one_based():
    sci, err, dq = _frame_with_star()
    result = measure_sources(
        sci, err, dq, header=_header(), settings=_settings(), profile=_profile()
    )
    i = _brightest_row(result)
    assert result.table["X_IMAGE"][i] == pytest.approx(STAR_X0 + 1.0, abs=1.0)
    assert result.table["Y_IMAGE"][i] == pytest.approx(STAR_Y0 + 1.0, abs=1.0)


def test_every_emitted_column_is_a_known_catalog_column():
    sci, err, dq = _frame_with_star()
    result = measure_sources(
        sci, err, dq, header=_header(), settings=_settings(), profile=_profile()
    )
    assert len(result.table.colnames) > 0
    for name in result.table.colnames:
        assert name in CATALOG_COLUMNS, f"{name!r} is not in CATALOG_COLUMNS"


def test_alias_columns_are_exact_copies_of_canonical_columns():
    sci, err, dq = _frame_with_star()
    result = measure_sources(
        sci, err, dq, header=_header(), settings=_settings(), profile=_profile()
    )
    table = result.table
    for alias, canonical in ALIAS_TO_CANONICAL.items():
        np.testing.assert_array_equal(np.asarray(table[alias]), np.asarray(table[canonical]))
    np.testing.assert_allclose(
        np.asarray(table["FWHM"]),
        np.asarray(table["FWHM_IMAGE"]) * pixel_scale_arcsec(_header(), None, _profile()),
        rtol=1e-5,
    )


def test_aperture_columns_have_one_column_per_configured_radius():
    sci, err, dq = _frame_with_star()
    settings = _settings(aperture_radii_arcsec=(1.0, 1.5, 2.0, 3.0))
    result = measure_sources(sci, err, dq, header=_header(), settings=settings, profile=_profile())
    assert result.table["FLUX_APER"].shape == (result.n_sources, 4)
    assert result.table["MAG_APER"].shape == (result.n_sources, 4)


def test_blank_frame_returns_zero_sources_without_raising():
    sci, err, dq = _blank_frame()
    result = measure_sources(
        sci, err, dq, header=_header(), settings=_settings(), profile=_profile()
    )
    assert result.n_sources == 0
    assert len(result.table) == 0
    assert result.metrics.n_sources == 0
    for name in result.table.colnames:
        assert name in CATALOG_COLUMNS


def test_saturated_pixel_inside_source_sets_flag_bit_4():
    dq = np.zeros(SHAPE, dtype=DQ_DTYPE)
    dq[int(STAR_Y0), int(STAR_X0)] |= int(DQ.SATURATED)
    sci, err, dq = _frame_with_star(dq=dq)
    result = measure_sources(
        sci, err, dq, header=_header(), settings=_settings(), profile=_profile()
    )
    i = _brightest_row(result)
    assert int(result.table["FLAGS"][i]) & FLAG_SATURATED


def test_blended_pair_is_deblended_into_two_flagged_sources():
    """A genuinely blended pair -- this is the shape of frame that exposed
    deblend_sources' lazy scikit-image import on real data (see the module
    docstring / PhotometryResult.deblended); the isolated-star tests above
    never reach the watershed code path at all.

    Separation is tuned empirically against the real photutils deblender for
    this module's settings (deblend_mincont=0.005 by default): 1.0x FWHM
    does not split into two sources here, 1.75x FWHM reliably does.
    """
    rng = np.random.default_rng(5)
    sci = rng.normal(BACKGROUND_LEVEL, READ_NOISE_E, size=SHAPE).astype(np.float32)
    err = np.full(SHAPE, READ_NOISE_E, dtype=np.float32)
    dq = np.zeros(SHAPE, dtype=DQ_DTYPE)
    sep_px = 1.75 * TRUE_FWHM_PX
    x1, y1 = 150.0 - sep_px / 2.0, 150.0
    x2, y2 = 150.0 + sep_px / 2.0, 150.0
    star = _gaussian(150000.0, TRUE_FWHM_PX, x1, y1) + _gaussian(150000.0, TRUE_FWHM_PX, x2, y2)
    sci = (sci + star).astype(np.float32)

    result = measure_sources(
        sci, err, dq, header=_header(), settings=_settings(), profile=_profile()
    )
    assert result.deblended is True
    assert result.n_sources == 2
    for flag in result.table["FLAGS"]:
        assert int(flag) & FLAG_NEIGHBOR
        assert int(flag) & FLAG_DEBLENDED


def test_deblend_import_error_degrades_gracefully(monkeypatch):
    """A missing scikit-image (photutils' deblend_sources raises ImportError/
    ModuleNotFoundError only when a source actually needs watershed
    splitting) must not abort the frame -- this is the failure mode a real
    ROBO43 crop hit when scikit-image was not installed."""
    import robo43.photometry.catalog as catalog_module

    def _raise_missing_skimage(*args, **kwargs):
        raise ImportError("No module named 'skimage'")

    monkeypatch.setattr(catalog_module, "deblend_sources", _raise_missing_skimage)

    sci, err, dq = _frame_with_star()
    result = measure_sources(
        sci, err, dq, header=_header(), settings=_settings(), profile=_profile()
    )
    assert result.n_sources >= 1
    assert result.deblended is False
    assert result.metrics.debl_nth == 0
    assert result.metrics.debl_minc is None


def test_measure_sources_raises_on_shape_mismatch():
    sci, err, dq = _frame_with_star()
    with pytest.raises(PhotometryError):
        measure_sources(
            sci, err[:-1, :], dq, header=_header(), settings=_settings(), profile=_profile()
        )


def test_pixel_scale_prefers_header_astrscl():
    header = fits.Header()
    header["ASTRSCL"] = 0.789
    scale = pixel_scale_arcsec(header, None, _profile(pixel_scale_arcsec=0.5))
    assert scale == pytest.approx(0.789)


def test_pixel_scale_falls_back_to_wcs_then_profile():
    header = fits.Header()
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [150.5, 150.5]
    wcs.wcs.crval = [10.0, -20.0]
    wcs.wcs.cdelt = [-0.6 / 3600.0, 0.6 / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    scale = pixel_scale_arcsec(header, wcs, _profile(pixel_scale_arcsec=0.5))
    assert scale == pytest.approx(0.6, rel=1e-3)

    scale_profile_only = pixel_scale_arcsec(header, None, _profile(pixel_scale_arcsec=0.42))
    assert scale_profile_only == pytest.approx(0.42)


def test_pixel_scale_raises_when_nothing_usable():
    header = fits.Header()
    with pytest.raises(PhotometryError):
        pixel_scale_arcsec(header, None, _profile(pixel_scale_arcsec=None))
