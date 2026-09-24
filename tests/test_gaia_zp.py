"""Tests for robo43.photometry.gaia_zp.

No network access: the Gaia query layer (``query_fn``/``xp_generate_fn``)
is always a fake injected by the test, per the module's own design for
offline testability.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from robo43.config import PhotometrySettings, Settings
from robo43.instruments.profile import InstrumentLayout, InstrumentProfile
from robo43.photometry import gaia_zp


def _profile(*, gaia_xp=None, gaia_transform=None) -> InstrumentProfile:
    return InstrumentProfile(
        name="test-t80s",
        matches={},
        pixel_scale_arcsec=0.55,
        gain_e_per_adu=None,
        read_noise_e=None,
        saturation_adu=65000.0,
        nonlinearity_adu=None,
        binning_keyword="XBINNING",
        layout=InstrumentLayout(science_hdu=0, n_amplifiers=1),
        valid_filters=("R",),
        filter_map={},
        filter_gaia_xp=gaia_xp or {},
        filter_gaia_transform=gaia_transform or {},
        keywords={},
        imagetyp_map={},
        site_lat_deg=None,
        site_lon_deg=None,
        site_elevation_m=None,
        index_scales=None,
        search_radius_deg=0.5,
    )


def _settings(**overrides) -> Settings:
    photometry = PhotometrySettings(**overrides)
    return Settings(photometry=photometry)


def _sci_header(*, naxis1=800, naxis2=800, tiles=None) -> fits.Header:
    header = fits.Header()
    header["NAXIS1"] = naxis1
    header["NAXIS2"] = naxis2
    header["MJD-OBS"] = 60000.0
    if tiles:
        for idx, region in enumerate(tiles):
            header[f"TILE{idx:02d}"] = region
    return header


def _synthetic_catalog(
    n,
    *,
    ra0=100.0,
    dec0=-30.0,
    mag0=15.0,
    seed=0,
    naxis1=800,
    naxis2=800,
) -> Table:
    rng = np.random.default_rng(seed)
    ra = ra0 + rng.uniform(-0.1, 0.1, n)
    dec = dec0 + rng.uniform(-0.1, 0.1, n)
    x = rng.uniform(50, naxis1 - 50, n)
    y = rng.uniform(50, naxis2 - 50, n)
    mag_auto = mag0 + rng.uniform(-2, 2, n)
    flux = 10 ** (-0.4 * (mag_auto - 20.0))
    return Table(
        {
            "NUMBER": np.arange(1, n + 1, dtype=np.int32),
            "X_IMAGE": x,
            "Y_IMAGE": y,
            "ALPHA_J2000": ra,
            "DELTA_J2000": dec,
            "FLUX_AUTO": flux.astype(np.float32),
            "FLUXERR_AUTO": (flux * 0.01).astype(np.float32),
            "MAG_AUTO": mag_auto.astype(np.float32),
            "MAGERR_AUTO": np.full(n, 0.01, dtype=np.float32),
            "MAG_APER": np.tile(mag_auto[:, None], (1, 2)).astype(np.float32),
            "MAGERR_APER": np.full((n, 2), 0.01, dtype=np.float32),
            "FLUX_MAX": (flux * 3).astype(np.float32),
            "FLAGS": np.zeros(n, dtype=np.int32),
            "IMAFLAGS_ISO": np.zeros(n, dtype=np.int32),
            "NIMAFLAGS_ISO": np.zeros(n, dtype=np.int32),
            "FWHM_IMAGE": np.full(n, 3.5, dtype=np.float32),
            "ELLIPTICITY": np.full(n, 0.05, dtype=np.float32),
        }
    )


def _fake_gaia_table(catalog: Table, *, true_zp, ruwe=1.0, excess=1.1, noise=0.0, seed=99) -> Table:
    """A Gaia table whose G magnitudes reproduce ``catalog``'s MAG_AUTO
    offset by ``true_zp`` plus independent Gaussian scatter ``noise`` (mag)
    (m_ref - m_inst == true_zp + noise for every star), at the catalogue's
    own RA/Dec, with zero proper motion."""
    n = len(catalog)
    rng = np.random.default_rng(seed)
    g = np.asarray(catalog["MAG_AUTO"]) + true_zp + rng.normal(0, noise, n)
    return Table(
        {
            "source_id": np.arange(1000, 1000 + n, dtype=np.int64),
            "ra": np.asarray(catalog["ALPHA_J2000"], dtype=float),
            "dec": np.asarray(catalog["DELTA_J2000"], dtype=float),
            "ref_epoch": np.full(n, 2016.0),
            "pmra": np.zeros(n),
            "pmdec": np.zeros(n),
            "phot_g_mean_mag": g.astype(float),
            "phot_bp_mean_mag": (g + 0.3).astype(float),
            "phot_rp_mean_mag": (g - 0.3).astype(float),
            "ruwe": np.full(n, ruwe),
            "phot_bp_rp_excess_factor": np.full(n, excess),
            "duplicated_source": np.zeros(n, dtype=bool),
        }
    )


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_transform_mag_validity_range():
    xf_name = "sdss_r"
    g_mag = np.array([15.0, 15.0])
    bp_rp = np.array([1.0, 10.0])  # second is far outside the validity range
    _mag, valid = gaia_zp.transform_mag(g_mag, bp_rp, xf_name)
    assert valid[0]
    assert not valid[1]


def test_fit_zero_point_recovers_known_offset():
    rng = np.random.default_rng(1)
    diffs = 25.0 + rng.normal(0, 0.01, 200)
    fit = gaia_zp.fit_zero_point(diffs, sigma=3.0)
    assert fit.zp == pytest.approx(25.0, abs=0.01)
    assert fit.n >= 190  # a few points legitimately clipped from 200 Gaussian samples
    assert fit.rms is not None and fit.rms < 0.05


def test_fit_zero_point_clips_outliers():
    rng = np.random.default_rng(2)
    diffs = np.concatenate([25.0 + rng.normal(0, 0.01, 100), [25.0 + 5.0, 25.0 - 5.0]])
    fit = gaia_zp.fit_zero_point(diffs, sigma=3.0)
    assert fit.zp == pytest.approx(25.0, abs=0.02)
    assert fit.n == 100  # the two 5-mag outliers were clipped


def test_fit_zero_point_empty_input():
    fit = gaia_zp.fit_zero_point(np.array([]), sigma=3.0)
    assert fit.zp is None
    assert fit.n == 0


def test_parse_tile_slices_round_trips_header_cards():
    header = _sci_header(naxis1=200, naxis2=100, tiles=["[1:100,1:100]", "[101:200,1:100]"])
    tiles = gaia_zp.parse_tile_slices(header, shape=(100, 200))
    assert tiles == [(slice(0, 100), slice(0, 100)), (slice(0, 100), slice(100, 200))]


def test_parse_tile_slices_falls_back_to_whole_frame():
    header = _sci_header(naxis1=200, naxis2=100)
    tiles = gaia_zp.parse_tile_slices(header, shape=(100, 200))
    assert tiles == [(slice(0, 100), slice(0, 200))]


def test_per_tile_offsets_single_tile_returns_empty():
    x = np.array([10.0, 20.0, 30.0])
    y = np.array([10.0, 20.0, 30.0])
    diffs = np.array([25.0, 25.0, 25.0])
    offsets = gaia_zp.per_tile_offsets(
        x, y, diffs, [(slice(0, 100), slice(0, 100))], zp_global=25.0, sigma=3.0
    )
    assert offsets == {}


def test_per_tile_offsets_detects_amplifier_deviation():
    tiles = [(slice(0, 100), slice(0, 100)), (slice(0, 100), slice(100, 200))]
    x = np.array([10.0, 20.0, 30.0, 110.0, 120.0, 130.0])
    y = np.array([10.0, 20.0, 30.0, 10.0, 20.0, 30.0])
    diffs = np.array([25.0, 25.0, 25.0, 25.2, 25.2, 25.2])  # tile 1 is 0.2 mag off
    offsets = gaia_zp.per_tile_offsets(x, y, diffs, tiles, zp_global=25.0, sigma=3.0, min_stars=3)
    assert offsets[0] == pytest.approx(0.0, abs=1e-6)
    assert offsets[1] == pytest.approx(0.2, abs=1e-6)


def test_gaia_quality_mask_rejects_bad_ruwe_and_duplicates():
    gaia = Table(
        {
            "ruwe": [1.0, 2.0, 1.1],
            "duplicated_source": [False, False, True],
        }
    )
    mask = gaia_zp.gaia_quality_mask(gaia, ruwe_max=1.4)
    assert mask.tolist() == [True, False, False]


# ---------------------------------------------------------------------------
# query_gaia_cone caching
# ---------------------------------------------------------------------------


def test_query_gaia_cone_reuses_cache(tmp_path):
    calls = {"n": 0}

    def fake_query(_ra, _dec, _radius):
        calls["n"] += 1
        return Table({"source_id": [1, 2], "ra": [_ra, _ra], "dec": [_dec, _dec]})

    t1 = gaia_zp.query_gaia_cone(10.0, -20.0, 0.5, cache_dir=tmp_path, query_fn=fake_query)
    t2 = gaia_zp.query_gaia_cone(10.0, -20.0, 0.5, cache_dir=tmp_path, query_fn=fake_query)
    assert calls["n"] == 1
    assert len(t1) == len(t2) == 2


# ---------------------------------------------------------------------------
# resolve_reference_mags: XP path, fallback, unmapped filter
# ---------------------------------------------------------------------------


def test_resolve_reference_mags_unmapped_filter_is_unavailable(tmp_path):
    profile = _profile()  # no gaia_xp, no gaia_transform
    gaia = _fake_gaia_table(_synthetic_catalog(5), true_zp=25.0)
    mag, _err, _valid, source = gaia_zp.resolve_reference_mags(
        gaia,
        filter_name="R",
        profile=profile,
        cache_dir=tmp_path,
        cache_key="k",
    )
    assert mag is None
    assert source == "unavailable"


def test_resolve_reference_mags_xp_missing_falls_back_to_transform(tmp_path):
    profile = _profile(gaia_xp={"R": "rJPLUS"}, gaia_transform={"R": "sdss_r"})
    catalog = _synthetic_catalog(10)
    gaia = _fake_gaia_table(catalog, true_zp=25.0)

    def xp_unavailable(_ids):
        raise ImportError("gaiaxpy not installed")

    mag, _err, _valid, source = gaia_zp.resolve_reference_mags(
        gaia,
        filter_name="R",
        profile=profile,
        cache_dir=tmp_path,
        cache_key="k2",
        xp_generate_fn=xp_unavailable,
    )
    assert source == "gaia_gbrp_sdss_r"
    assert mag is not None


def test_resolve_reference_mags_uses_xp_when_available(tmp_path):
    profile = _profile(gaia_xp={"R": "rJPLUS"}, gaia_transform={"R": "sdss_r"})
    catalog = _synthetic_catalog(10)
    gaia = _fake_gaia_table(catalog, true_zp=25.0)

    def fake_generate(_ids):
        import pandas as pd

        return pd.DataFrame(
            {
                "source_id": list(_ids),
                "Jplus_mag_rJPLUS": [20.0] * len(_ids),
                "Jplus_mag_error_rJPLUS": [0.01] * len(_ids),
            }
        )

    mag, _err, _valid, source = gaia_zp.resolve_reference_mags(
        gaia,
        filter_name="R",
        profile=profile,
        cache_dir=tmp_path,
        cache_key="k3",
        xp_generate_fn=fake_generate,
    )
    assert source == "gaia_xp_jplus"
    assert np.all(mag == 20.0)


# ---------------------------------------------------------------------------
# calibrate_frame: end-to-end with a fake Gaia query, no network
# ---------------------------------------------------------------------------


def test_calibrate_frame_recovers_zero_point(tmp_path):
    profile = _profile(gaia_transform={"R": "sdss_r"})
    settings = _settings(zp_min_stars=5, zp_rms_max=0.1, zp_sigma_clip=3.0)
    catalog = _synthetic_catalog(60, seed=3)
    gaia = _fake_gaia_table(catalog, true_zp=24.0, noise=0.02, seed=30)
    header = _sci_header()

    def fake_query(_ra, _dec, _radius):
        return gaia

    gzp, out = gaia_zp.calibrate_frame(
        catalog,
        header=header,
        wcs=None,
        settings=settings,
        profile=profile,
        gain_e_per_adu=None,
        scale=0.55,
        filter_name="R",
        cache_dir=tmp_path,
        query_fn=fake_query,
    )
    assert gzp.calibrated
    assert gzp.zp == pytest.approx(24.0, abs=0.05)
    assert gzp.source == "gaia_gbrp_sdss_r"
    assert "MAG_CALIB" in out.colnames
    assert "REFMAG" in out.colnames
    assert np.isfinite(np.asarray(out["REFMAG"])).sum() > 0


def test_calibrate_frame_rejects_when_too_few_stars(tmp_path):
    profile = _profile(gaia_transform={"R": "sdss_r"})
    settings = _settings(zp_min_stars=1000, zp_rms_max=0.1, zp_sigma_clip=3.0)
    catalog = _synthetic_catalog(20, seed=4)
    gaia = _fake_gaia_table(catalog, true_zp=24.0)
    header = _sci_header()

    gzp, out = gaia_zp.calibrate_frame(
        catalog,
        header=header,
        wcs=None,
        settings=settings,
        profile=profile,
        gain_e_per_adu=None,
        scale=0.55,
        filter_name="R",
        cache_dir=tmp_path,
        query_fn=lambda *_args: gaia,
    )
    assert not gzp.calibrated
    assert "MAG_CALIB" not in out.colnames  # not added at all without a fit


def test_calibrate_frame_rejects_on_high_rms(tmp_path):
    profile = _profile(gaia_transform={"R": "sdss_r"})
    settings = _settings(zp_min_stars=5, zp_rms_max=0.001, zp_sigma_clip=3.0)
    catalog = _synthetic_catalog(60, seed=5)
    gaia = _fake_gaia_table(catalog, true_zp=24.0, noise=0.02, seed=50)
    header = _sci_header()

    gzp, out = gaia_zp.calibrate_frame(
        catalog,
        header=header,
        wcs=None,
        settings=settings,
        profile=profile,
        gain_e_per_adu=None,
        scale=0.55,
        filter_name="R",
        cache_dir=tmp_path,
        query_fn=lambda *_args: gaia,
    )
    assert not gzp.calibrated
    assert gzp.zp is not None  # still reported, just flagged uncalibrated
    assert "MAG_CALIB" not in out.colnames  # uncalibrated fit must not populate calibrated columns


def test_calibrate_frame_per_tile_offsets(tmp_path):
    profile = _profile(gaia_transform={"R": "sdss_r"})
    settings = _settings(zp_min_stars=5, zp_rms_max=1.0, zp_sigma_clip=5.0)
    catalog = _synthetic_catalog(80, seed=6, naxis1=800, naxis2=100)
    gaia = _fake_gaia_table(catalog, true_zp=24.0)
    header = _sci_header(
        naxis1=800,
        naxis2=100,
        tiles=["[1:400,1:100]", "[401:800,1:100]"],
    )

    gzp, _out = gaia_zp.calibrate_frame(
        catalog,
        header=header,
        wcs=None,
        settings=settings,
        profile=profile,
        gain_e_per_adu=None,
        scale=0.55,
        filter_name="R",
        cache_dir=tmp_path,
        query_fn=lambda *_args: gaia,
    )
    assert gzp.calibrated
    # Both tiles are populated from the same synthetic distribution with no
    # injected offset, so any recovered per-tile offset should be small.
    for offset in gzp.tile_offsets.values():
        assert abs(offset) < 0.1


def test_header_cards_uses_none_sentinel_when_uncalibrated():
    gzp = gaia_zp.GaiaZeroPoint(
        zp=None,
        zp_err=None,
        zp_rms=None,
        n=0,
        source="unavailable",
        calibrated=False,
    )
    cards = {k: v for k, v, _c in gaia_zp.header_cards(gzp)}
    assert cards["ZPABS"] == "NONE"
    assert cards["ZPABSCAL"] is False
