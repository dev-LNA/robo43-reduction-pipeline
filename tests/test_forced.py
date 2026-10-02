"""Tests for robo43.photometry.forced and the ``robo43 forced`` subcommand.

Synthetic frames only: Gaussian stars rendered at sky positions through a TAN WCS, so
a shifted/rotated second frame has the same stars at different pixels.
"""

from __future__ import annotations

import csv
import logging
import math

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from robo43.cli.main import main
from robo43.config import ConfigError, ForcedSettings, PhotometrySettings, Settings, load_settings
from robo43.instruments.profile import InstrumentLayout, InstrumentProfile
from robo43.logging_setup import ROOT_LOGGER_NAME
from robo43.photometry.catalog import (
    FLAG_APERTURE_INCOMPLETE,
    FLAG_BAD_PHOTOMETRY,
    FLAG_DEBLENDED,
    FLAG_NEIGHBOR,
    FLAG_SATURATED,
    FLAG_TRUNCATED,
)
from robo43.photometry.forced import (
    FrameGeometry,
    MasterCatalog,
    build_master,
    measure_forced,
    measure_transparency,
    read_master,
    score_frames,
    select_centroid_frames,
    summarize_frame,
    write_master,
)
from robo43.products.dq import DQ, DQ_DTYPE

SHAPE = (300, 300)
SCALE = 0.5  # arcsec / pixel
FWHM_PX = 4.0
BKG = 500.0
NOISE = 7.0
RADII = (1.0, 2.0, 4.0)  # arcsec -> 2, 4, 8 px
TRUE_FLUX = 100000.0


@pytest.fixture(autouse=True)
def _reset_logging():
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    handlers = list(logger.handlers)
    level, propagate = logger.level, logger.propagate
    yield
    for handler in logger.handlers:
        if handler not in handlers:
            handler.close()
    logger.handlers = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def _profile() -> InstrumentProfile:
    return InstrumentProfile(
        name="test-instrument", matches={}, pixel_scale_arcsec=SCALE, gain_e_per_adu=0.7,
        read_noise_e=NOISE, saturation_adu=55000.0, nonlinearity_adu=50000.0,
        binning_keyword="XBINNING", layout=InstrumentLayout(), valid_filters=(), filter_map={},
        keywords={}, imagetyp_map={}, site_lat_deg=None, site_lon_deg=None,
        site_elevation_m=None, index_scales=None, search_radius_deg=0.5,
    )


def _settings(**forced) -> Settings:
    return Settings(
        photometry=PhotometrySettings(
            back_size=32, back_filtersize=3, detect_thresh=3.0, seeing_fwhm_arcsec=2.0,
            aperture_radii_arcsec=RADII, edge_margin_px=10,
        ),
        forced=ForcedSettings(**forced),
    )


def _wcs(*, shift=(0.0, 0.0), rot_deg=0.0, shape=SHAPE) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [150.0, -10.0]
    w.wcs.crpix = [shape[1] / 2.0 + shift[0], shape[0] / 2.0 + shift[1]]
    c, s = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
    d = SCALE / 3600.0
    w.wcs.cd = [[-d * c, d * s], [d * s, d * c]]
    return w


def _header(wcs: WCS, **extra) -> fits.Header:
    header = wcs.to_header()
    header["ASTRSCL"] = SCALE
    header.update(extra)
    return header


def _star_pixels(n_side: int = 4) -> np.ndarray:
    """Well separated stars on a jittered grid of frame A, as (x, y), 0-based."""
    rng = np.random.default_rng(3)
    grid = np.linspace(70, 230, n_side)
    xx, yy = np.meshgrid(grid, grid)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    return pts + rng.uniform(-6, 6, pts.shape)


def _render(wcs, ra, dec, fluxes, *, seed=1, noise=NOISE):
    yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    rng = np.random.default_rng(seed)
    sci = rng.normal(BKG, noise, size=SHAPE)
    px, py = wcs.all_world2pix(ra, dec, 0)
    sigma = FWHM_PX / 2.3548
    for x0, y0, flux in zip(px, py, fluxes, strict=True):
        sci += flux / (2 * np.pi * sigma**2) * np.exp(
            -(((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma**2))
        )
    err = np.full(SHAPE, NOISE, dtype=np.float32)
    return sci.astype(np.float32), err, np.zeros(SHAPE, dtype=DQ_DTYPE)


def _master_from(ra, dec, flags=None) -> MasterCatalog:
    n = len(ra)
    return MasterCatalog(
        number=np.arange(1, n + 1, dtype=np.int32), ra=np.asarray(ra, float),
        dec=np.asarray(dec, float), n_det=np.full(n, 3, np.int32),
        n_cover=np.full(n, 3, np.int32), snr=np.full(n, 100.0, np.float32),
        flags=np.zeros(n, np.int32) if flags is None else np.asarray(flags, np.int32),
    )


# --------------------------------------------------------------------------- measurement


def test_sky_fixed_apertures_follow_the_stars_through_a_shifted_rotated_wcs():
    wcs_a = _wcs()
    wcs_b = _wcs(shift=(7.3, -5.1), rot_deg=12.0)
    xy = _star_pixels()
    ra, dec = wcs_a.all_pix2world(xy[:, 0], xy[:, 1], 0)
    master = _master_from(ra, dec)
    fluxes = np.full(len(ra), TRUE_FLUX)
    sci, err, dq = _render(wcs_b, ra, dec, fluxes)

    res = measure_forced(
        sci, err, dq, master=master, header=_header(wcs_b), settings=_settings(),
        profile=_profile(), wcs=wcs_b,
    )
    assert len(res.table) == len(ra)
    assert list(res.table["NUMBER"]) == list(master.number)
    flux = np.asarray(res.table["FLUX_APER"])[:, 2]  # 4 arcsec = 8 px, encloses the star
    np.testing.assert_allclose(flux, TRUE_FLUX, rtol=0.02)
    # the measured position is the projection, not a recentred centroid
    px, py = wcs_b.all_world2pix(ra, dec, 0)
    np.testing.assert_allclose(res.table["X_IMAGE"], px + 1.0, atol=1e-6)
    np.testing.assert_allclose(res.table["Y_IMAGE"], py + 1.0, atol=1e-6)
    assert np.all(np.asarray(res.table["FLAGS"]) == 0)


def test_pixel_fixed_control_misses_the_stars():
    """Same frame, apertures at frame A's *pixels*: shows the test above is sensitive to the WCS."""
    wcs_a = _wcs()
    wcs_b = _wcs(shift=(7.3, -5.1), rot_deg=12.0)
    xy = _star_pixels()
    ra, dec = wcs_a.all_pix2world(xy[:, 0], xy[:, 1], 0)
    sci, err, dq = _render(wcs_b, ra, dec, np.full(len(ra), TRUE_FLUX))
    ra_px, dec_px = wcs_b.all_pix2world(xy[:, 0], xy[:, 1], 0)  # pixel-fixed
    res = measure_forced(
        sci, err, dq, master=_master_from(ra_px, dec_px), header=_header(wcs_b),
        settings=_settings(), profile=_profile(), wcs=wcs_b,
    )
    flux = np.asarray(res.table["FLUX_APER"])[:, 1]  # 2 arcsec = 4 px
    assert np.median(flux) < 0.5 * TRUE_FLUX


def test_local_background_ignores_masked_neighbours_and_bad_pixels():
    from robo43.photometry.forced import _local_background, _source_mask

    rng = np.random.default_rng(5)
    sub = rng.normal(3.0, 1.0, SHAPE)
    sub[110:130, 100:120] += 5000.0  # bright neighbour inside the annulus of (100, 100)
    bad = np.zeros(SHAPE, dtype=bool)
    bad[90:95, 90:100] = True
    sub[bad] = 1.0e6
    x = np.array([100.0, 250.0])
    y = np.array([100.0, 250.0])
    exclude = bad | _source_mask(SHAPE, np.array([110.0]), np.array([120.0]), 14.0)
    est = _local_background(sub, exclude, x, y, 10.0, 26.0)
    np.testing.assert_allclose(est, 3.0, atol=0.3)
    # a position whose annulus lies entirely off the frame has no usable pixels -> NaN
    far = _local_background(sub, exclude, np.array([-100.0]), np.array([-100.0]), 10.0, 26.0)
    assert np.isnan(far[0])


def test_offframe_edge_saturation_and_inherited_flags():
    wcs = _wcs()
    xy = np.array([[150.0, 150.0], [150.0, 220.0], [4.0, 150.0], [150.0, 6.0], [400.0, 150.0]])
    ra, dec = wcs.all_pix2world(xy[:, 0], xy[:, 1], 0)
    sci, err, dq = _render(wcs, ra, dec, np.full(len(ra), TRUE_FLUX))
    dq[220, 150] = int(DQ.SATURATED)  # centre of the star at (150, 220)
    flags_master = [FLAG_NEIGHBOR | FLAG_DEBLENDED, 0, 0, 0, 0]
    master = _master_from(ra, dec, flags_master)

    res = measure_forced(
        sci, err, dq, master=master, header=_header(wcs), settings=_settings(),
        profile=_profile(), wcs=wcs,
    )
    # off-frame source (x = 400) is omitted by default
    assert list(res.table["NUMBER"]) == [1, 2, 3, 4]
    assert res.n_offframe == 1
    flags = np.asarray(res.table["FLAGS"])
    assert flags[0] == FLAG_NEIGHBOR | FLAG_DEBLENDED
    assert flags[1] & FLAG_SATURATED
    assert flags[2] & FLAG_TRUNCATED
    assert flags[2] & FLAG_APERTURE_INCOMPLETE
    assert flags[3] & FLAG_TRUNCATED

    res_all = measure_forced(
        sci, err, dq, master=master, header=_header(wcs), settings=_settings(keep_offframe=True),
        profile=_profile(), wcs=wcs,
    )
    assert list(res_all.table["NUMBER"]) == [1, 2, 3, 4, 5]
    last = res_all.table[-1]
    assert np.all(np.isnan(np.asarray(last["FLUX_APER"])))
    assert int(last["FLAGS"]) & FLAG_BAD_PHOTOMETRY
    assert int(last["FLAGS"]) & FLAG_APERTURE_INCOMPLETE
    assert np.isfinite(float(last["RA"]))


def test_local_annulus_is_smoothed_and_can_be_switched_off():
    wcs = _wcs()
    xy = _star_pixels(3)
    ra, dec = wcs.all_pix2world(xy[:, 0], xy[:, 1], 0)
    sci, err, dq = _render(wcs, ra, dec, np.full(len(ra), TRUE_FLUX))
    kw = {
        "master": _master_from(ra, dec), "header": _header(wcs), "profile": _profile(), "wcs": wcs,
    }
    default = ForcedSettings()
    assert default.local_bkg_width_arcsec > 0 and default.local_bkg_smooth_k > 1
    # a flat +20 ADU pedestal that the 2-D model absorbs: the annulus must find ~0 residual
    on = measure_forced(sci + 20.0, err, dq, settings=_settings(), **kw)
    assert np.all(np.isfinite(np.asarray(on.table["BACKGROUND"])))
    assert np.max(np.abs(np.asarray(on.table["BACKGROUND"]))) < 1.0
    np.testing.assert_allclose(np.asarray(on.table["FLUX_APER"])[:, 2], TRUE_FLUX, rtol=0.02)
    # smoothing makes the local sky identical for sources that share their k nearest neighbours
    smooth = measure_forced(
        sci, err, dq, settings=_settings(local_bkg_smooth_k=len(ra)), **kw
    )
    assert len(set(np.round(np.asarray(smooth.table["BACKGROUND"]), 5))) == 1
    off = measure_forced(sci, err, dq, settings=_settings(local_bkg_width_arcsec=0.0), **kw)
    assert np.all(np.asarray(off.table["BACKGROUND"]) == 0.0)


def test_smooth_local_background_takes_the_neighbour_median_and_ignores_nan():
    from robo43.photometry.forced import _smooth_local_background

    x = np.arange(10.0) * 10.0
    y = np.zeros(10)
    raw = np.array([1.0, 1.0, 100.0, 1.0, np.nan, 1.0, 1.0, np.nan, 1.0, 1.0])
    out = _smooth_local_background(raw, x, y, 5)
    np.testing.assert_allclose(out, 1.0)  # the outlier and the NaNs do not leak in
    np.testing.assert_array_equal(
        _smooth_local_background(raw, x, y, 1), np.where(np.isfinite(raw), raw, 0.0)
    )
    assert _smooth_local_background(np.array([np.nan, np.nan]), x[:2], y[:2], 2).tolist() == [0, 0]


def test_table_has_exactly_the_measure_sources_columns():
    from robo43.photometry.catalog import COLUMN_NAMES

    wcs = _wcs()
    xy = _star_pixels(2)
    ra, dec = wcs.all_pix2world(xy[:, 0], xy[:, 1], 0)
    sci, err, dq = _render(wcs, ra, dec, np.full(len(ra), TRUE_FLUX))
    res = measure_forced(
        sci, err, dq, master=_master_from(ra, dec), header=_header(wcs), settings=_settings(),
        profile=_profile(), wcs=wcs,
    )
    assert tuple(res.table.colnames) == COLUMN_NAMES
    assert res.table["FLUX_APER"].shape == (len(ra), len(RADII))


def test_shape_columns_come_from_the_nearest_detection_and_are_nan_otherwise():
    wcs = _wcs()
    xy = np.array([[100.0, 100.0], [200.0, 200.0]])
    ra, dec = wcs.all_pix2world(xy[:, 0], xy[:, 1], 0)
    sci, err, dq = _render(wcs, ra, dec, np.full(2, TRUE_FLUX))
    n = 1
    detections = Table(
        {
            "RA": ra[:1] + 0.2 / 3600.0, "DEC": dec[:1],  # 0.2 arcsec from source 1; none near 2
            "FWHM_IMAGE": np.full(n, 3.7), "FLUX_RADIUS": np.full(n, 1.9),
            "A_IMAGE": np.full(n, 1.5), "B_IMAGE": np.full(n, 1.4),
            "THETA_IMAGE": np.full(n, 10.0), "ELONGATION": np.full(n, 1.07),
            "ELLIPTICITY": np.full(n, 0.06),
        }
    )
    res = measure_forced(
        sci, err, dq, master=_master_from(ra, dec), header=_header(wcs), settings=_settings(),
        profile=_profile(), wcs=wcs, frame_table=detections,
    )
    t = res.table
    assert float(t["FWHM_IMAGE"][0]) == pytest.approx(3.7)
    assert float(t["FWHM"][0]) == pytest.approx(3.7 * SCALE)
    assert float(t["FLUX_RADIUS"][0]) == pytest.approx(1.9)
    assert np.isnan(float(t["FWHM"][1])) and np.isnan(float(t["A_IMAGE"][1]))
    res0 = measure_forced(
        sci, err, dq, master=_master_from(ra, dec), header=_header(wcs), settings=_settings(),
        profile=_profile(), wcs=wcs,
    )
    assert np.isnan(np.asarray(res0.table["FWHM"])).all()
    assert res0.fwhm_arcsec == pytest.approx(res0.fwhm_px * SCALE)


def test_forced_requires_a_celestial_wcs():
    from robo43.exceptions import PhotometryError

    sci, err, dq = _render(_wcs(), [150.0], [-10.0], [TRUE_FLUX])
    with pytest.raises(PhotometryError):
        measure_forced(
            sci, err, dq, master=_master_from([150.0], [-10.0]), header=_header(_wcs()),
            settings=_settings(), profile=_profile(), wcs=WCS(naxis=2),
        )


# --------------------------------------------------------------------------- selection


def _info_table(ra, dec, flux, snr=100.0, flags=0) -> Table:
    n = len(ra)
    flux_ap = np.column_stack([np.asarray(flux) * f for f in (0.4, 0.7, 0.9, 0.97, 1.0)])
    return Table(
        {
            "RA": np.asarray(ra, float), "DEC": np.asarray(dec, float),
            "FLAGS": np.full(n, flags, dtype=np.int32), "SNR": np.full(n, snr),
            "FLUX_APER": flux_ap,
        }
    )


def _info(table, **kw):
    base = {
        "path": "f", "fwhm_arcsec": 3.0, "fwhm_px": 6.0, "exptime": 90.0,
        "astrometry_ok": True, "astrometry_rms": 0.3,
        "aperture_radii_arcsec": (2.0, 3.0, 4.0, 6.0, 8.0), "settings": Settings(),
        **kw,
    }
    return summarize_frame(table, **base)


def test_transparency_is_the_flux_ratio_normalised_to_the_clear_frames():
    rng = np.random.default_rng(0)
    n = 80
    ra = 150.0 + rng.uniform(0, 0.05, n)
    dec = -10.0 + rng.uniform(0, 0.05, n)
    flux = rng.uniform(1e4, 1e5, n)
    settings = Settings(forced=ForcedSettings(transparency_min_stars=10))
    scales = [1.0, 1.0, 1.0, 1.0, 0.8, 0.5]
    infos = [
        _info(_info_table(ra, dec, flux * s), settings=settings, path=f"f{i}")
        for i, s in enumerate(scales)
    ]
    t, used, ref = measure_transparency(infos, settings)
    np.testing.assert_allclose(t, scales, rtol=1e-6)
    assert used.min() >= 10
    assert 0 <= ref < len(infos)


def test_transparency_is_nan_without_enough_matched_bright_stars():
    ra = np.array([150.0, 150.01, 150.02])
    dec = np.array([-10.0, -10.0, -10.0])
    infos = [_info(_info_table(ra, dec, [1e4, 2e4, 3e4])) for _ in range(2)]
    t, _, _ = measure_transparency(infos, Settings())  # needs >= 20 stars
    assert np.isnan(t).all()


def test_score_prefers_many_useful_sources_when_seeing_is_close_and_respects_transparency():
    fs = ForcedSettings()
    fwhm = np.array([3.0, 3.1, 3.05, 3.0])
    n_useful = np.array([50000, 90000, 88000, 95000])
    t = np.array([1.0, 0.99, 0.98, 0.7])  # the best-populated frame is cloudy
    ok = np.array([True, True, True, True])
    eligible, score, relaxed = score_frames(fwhm, n_useful, t, ok, fs)
    assert not relaxed
    assert list(eligible) == [True, True, True, False]
    assert np.isneginf(score[3])
    best = select_centroid_frames(score, fwhm, 2)
    assert sorted(best) == [1, 2]  # not the sharpest frame (0), which has half the sources


def test_score_penalises_poor_seeing_with_the_configured_power():
    fwhm = np.array([2.5, 5.0])
    n_useful = np.array([60000, 100000])
    t = np.ones(2)
    ok = np.ones(2, dtype=bool)
    for power, winner in ((2.0, 0), (0.0, 1)):
        _, score, _ = score_frames(fwhm, n_useful, t, ok, ForcedSettings(seeing_power=power))
        assert select_centroid_frames(score, fwhm, 1) == [winner]


def test_unsolved_astrometry_is_never_selected_and_cut_is_relaxed_when_all_are_cloudy():
    fs = ForcedSettings()
    fwhm = np.array([3.0, 3.0, 3.0])
    n_useful = np.array([100, 90, 80])
    ok = np.array([False, True, True])
    eligible, score, relaxed = score_frames(fwhm, n_useful, np.array([1.0, 0.5, 0.6]), ok, fs)
    assert relaxed
    assert list(eligible) == [False, True, True]
    assert select_centroid_frames(score, fwhm, 3) == [1, 2]


# --------------------------------------------------------------------------- master


def _det_table(ra, dec, flags=None, snr=50.0) -> Table:
    n = len(ra)
    return Table(
        {
            "RA": np.asarray(ra, float), "DEC": np.asarray(dec, float),
            "FLAGS": np.zeros(n, dtype=np.int32) if flags is None else np.asarray(flags, np.int32),
            "SNR": np.full(n, snr),
        }
    )


def test_build_master_medians_detection_requirement_dedupe_and_numbering():
    wcs = _wcs()
    far = _wcs(shift=(400.0, 0.0))  # footprint that excludes the right-hand source
    geoms = [
        FrameGeometry(wcs, 300, 300), FrameGeometry(far, 300, 300), FrameGeometry(wcs, 300, 300)
    ]
    px = np.array([100.0, 200.0, 50.0, 260.0, 150.0, 150.0 + 0.3 / SCALE])
    py = np.array([100.0, 150.0, 220.0, 100.0, 40.0, 40.0])
    ra, dec = wcs.all_pix2world(px, py, 0)
    jit = 0.1 / 3600.0
    # frame 0: everything; frame 1: stars 0..2 + 4 (3 is outside its footprint); frame 2: 0, 1, 2
    t0 = _det_table(ra, dec, flags=[0, 0, 0, FLAG_NEIGHBOR, 0, 0])
    t1 = _det_table(ra[[0, 1, 2, 4]] + jit, dec[[0, 1, 2, 4]])
    t2 = _det_table(ra[[0, 1, 2]] - jit, dec[[0, 1, 2]] + jit)
    master = build_master([t0, t1, t2], geoms, ForcedSettings(min_frames=2))

    # star 3 is covered by frames 0 and 2 only (n_cover 2) and seen once -> dropped;
    # star 2 (x=50) is outside frame 1's footprint (n_cover 2) but seen in 3 -> kept;
    # star 4 is seen in frames 0 and 1; its twin 5 (0.3") is a duplicate -> one survives
    assert len(master) == 4
    assert list(master.number) == [1, 2, 3, 4]
    assert np.all(np.diff(master.ra) >= 0)  # numbered in RA order
    for s in (0, 1, 2):
        sep = np.hypot(
            (master.ra - ra[s]) * np.cos(np.radians(dec[s])), master.dec - dec[s]
        ) * 3600
        assert sep.min() < 0.12
    # the median of (x, x + j, x - j) is x itself
    j = int(np.argmin(np.abs(master.ra - ra[0]) + np.abs(master.dec - dec[0])))
    assert abs(master.ra[j] - ra[0]) < 1e-9
    assert master.n_det[j] == 3


def test_build_master_inherits_blend_flags_and_caps_requirement_by_coverage():
    wcs = _wcs()
    off = _wcs(shift=(500.0, 0.0))
    geoms = [FrameGeometry(wcs, 300, 300), FrameGeometry(off, 300, 300)]
    ra, dec = wcs.all_pix2world(np.array([100.0, 120.0]), np.array([100.0, 100.0]), 0)
    t0 = _det_table(ra, dec, flags=[FLAG_NEIGHBOR | FLAG_DEBLENDED | FLAG_SATURATED, 0])
    t1 = _det_table(ra[:0], dec[:0])
    master = build_master([t0, t1], geoms, ForcedSettings(min_frames=2))
    assert len(master) == 2  # covered by one frame only, detected there
    first = int(np.argmin(np.abs(master.ra - ra[0])))
    assert int(master.flags[first]) == FLAG_NEIGHBOR | FLAG_DEBLENDED  # saturation is per frame
    assert int(master.flags[1 - first]) == 0


def test_master_csv_roundtrip(tmp_path):
    m = _master_from([150.123456789012, 150.5], [-10.987654321098, -10.1], [3, 0])
    write_master(m, tmp_path / "forced_reference.csv")
    back = read_master(tmp_path / "forced_reference.csv")
    np.testing.assert_array_equal(back.number, m.number)
    np.testing.assert_allclose(back.ra, m.ra, rtol=0, atol=1e-12)
    np.testing.assert_allclose(back.dec, m.dec, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(back.flags, m.flags)


# --------------------------------------------------------------------------- config


def test_forced_settings_are_validated_and_outside_the_config_hash():
    base = load_settings(use_user_config=False, env={})
    assert base.forced == ForcedSettings()
    changed = load_settings(
        use_user_config=False, env={}, cli={"forced.seeing_power": 3.0}
    )
    assert changed.forced.seeing_power == 3.0
    assert changed.config_hash() == base.config_hash()
    with pytest.raises(ConfigError):
        load_settings(use_user_config=False, env={}, cli={"forced.n_centroid_frames": 0})
    with pytest.raises(ConfigError):
        load_settings(use_user_config=False, env={}, cli={"forced.transparency_min": 1.5})


# --------------------------------------------------------------------------- CLI


def _write_night(tmp_path, shifts):
    """Synthetic night: the same stars on every frame, different pointing and rotation."""
    night = tmp_path / "night"
    night.mkdir()
    wcs_a = _wcs()
    xy = _star_pixels(5)
    ra, dec = wcs_a.all_pix2world(xy[:, 0], xy[:, 1], 0)
    paths = []
    for i, (shift, rot, scale_flux) in enumerate(shifts):
        wcs = _wcs(shift=shift, rot_deg=rot)
        sci, err, dq = _render(
            wcs, ra, dec, np.full(len(ra), TRUE_FLUX * scale_flux), seed=10 + i
        )
        header = _header(
            wcs, EXPTIME=90.0, ASTRSOLV=True, ASTRRMS=0.3, BUNIT="electron", EXTNAME="SCI",
            RADESYS="ICRS", GAINVAL=0.7,
        )
        path = night / f"f{i}_proc.fits"
        fits.HDUList(
            [
                fits.PrimaryHDU(data=sci, header=header),
                fits.ImageHDU(data=err, name="ERR"),
                fits.ImageHDU(data=dq, name="DQ"),
            ]
        ).writeto(path)
        paths.append(path)
    config = tmp_path / "forced.toml"
    config.write_text(
        "[photometry]\nback_size = 32\ndetect_thresh = 3.0\nseeing_fwhm_arcsec = 2.0\n"
        "aperture_radii_arcsec = [1.0, 2.0, 4.0]\nedge_margin_px = 10\n"
        "[forced]\ntransparency_min_stars = 5\nsnr_useful = 10.0\ntransparency_snr_min = 10.0\n"
    )
    return paths, config, ra, dec


SHIFTS = [((0.0, 0.0), 0.0, 1.0), ((6.0, -4.0), 8.0, 0.97), ((-5.0, 7.0), -6.0, 0.6)]


def test_forced_cli_end_to_end(tmp_path):
    paths, config, ra, _ = _write_night(tmp_path, SHIFTS)
    assert main(["photometry", "--config", str(config), "--in-place", "--catalog-csv",
                 *map(str, paths)]) == 0
    out = tmp_path / "out"
    code = main(["forced", "--config", str(config), "--np", "1", "--n-frames", "2",
                 "--outdir", str(out), *map(str, paths)])
    assert code == 0

    status = list(csv.DictReader((out / "forced_status.csv").open()))
    assert len(status) == 3 and all(not r["error"] for r in status)
    chosen = sorted(r["raw_name"] for r in status if r["centroid_frame"])
    assert chosen == ["f0_proc.fits", "f1_proc.fits"]  # the cloudy third frame is not eligible
    assert float(status[2]["transparency"]) < 0.75

    master = read_master(out / "forced_reference.csv")
    assert len(master) >= len(ra) - 2

    # one CSV per frame, same layout as the photometry CSV, same master rows
    prod_cols = list(csv.DictReader((paths[0].parent / "f0_proc_catalog.csv").open()).fieldnames)
    for p in paths:
        rows = list(csv.DictReader((out / f"{p.stem}_forced_catalog.csv").open()))
        assert list(rows[0].keys()) == prod_cols
        assert len(rows) == len(master)
    # sky-fixed: the flux of the same star is the same on the rotated frames (cloudy one scaled)
    f0 = Table.read(out / "f0_proc_forced_catalog.csv", format="ascii.csv")
    f1 = Table.read(out / "f1_proc_forced_catalog.csv", format="ascii.csv")
    f2 = Table.read(out / "f2_proc_forced_catalog.csv", format="ascii.csv")
    clean = (np.asarray(f0["FLAGS"]) == 0) & (np.asarray(f1["FLAGS"]) == 0)
    assert clean.sum() >= 15
    r01 = np.asarray(f1["FLUX_APER_3"])[clean] / np.asarray(f0["FLUX_APER_3"])[clean]
    assert np.median(r01) == pytest.approx(0.97, abs=0.02)
    r02 = np.asarray(f2["FLUX_APER_3"])[clean] / np.asarray(f0["FLUX_APER_3"])[clean]
    assert np.median(r02) == pytest.approx(0.6, abs=0.02)
    assert (out / "forced_params.json").is_file()


@pytest.mark.parametrize("breakage", ["false", "missing", "rms_missing", "rms_high"])
def test_forced_cli_skips_frames_with_unsolved_astrometry(tmp_path, breakage):
    paths, config, _, _ = _write_night(tmp_path, SHIFTS[:2])
    assert main(["photometry", "--config", str(config), "--in-place", *map(str, paths)]) == 0
    if breakage == "false":
        fits.setval(paths[1], "ASTRSOLV", value=False, ext=0)
    elif breakage == "missing":
        with fits.open(paths[1], mode="update") as hdul:
            del hdul[0].header["ASTRSOLV"]
    elif breakage == "rms_missing":  # no ASTRRMS: NaN residual
        with fits.open(paths[1], mode="update") as hdul:
            del hdul[0].header["ASTRRMS"]
    else:
        fits.setval(paths[1], "ASTRRMS", value=2.5, ext=0)
    out = tmp_path / "out"
    assert main(["forced", "--config", str(config), "--np", "1", "--outdir", str(out),
                 *map(str, paths)]) == 0
    status = {r["raw_name"]: r for r in csv.DictReader((out / "forced_status.csv").open())}
    assert "astrometry not solved" in status["f1_proc.fits"]["error"]
    assert not status["f0_proc.fits"]["error"]
    assert (out / "f0_proc_forced_catalog.csv").is_file()
    assert not (out / "f1_proc_forced_catalog.csv").exists()
    config_all = tmp_path / "forced_all.toml"  # opt-out: the frame is measured anyway
    config_all.write_text(config.read_text() + "skip_unsolved = false\n")
    out2 = tmp_path / "out2"
    assert main(["forced", "--config", str(config_all), "--np", "1", "--outdir", str(out2),
                 *map(str, paths)]) == 0
    assert (out2 / "f1_proc_forced_catalog.csv").is_file()


def test_forced_cli_select_only_and_master_reuse(tmp_path):
    paths, config, _, _ = _write_night(tmp_path, SHIFTS[:2])
    assert main(["photometry", "--config", str(config), "--in-place", *map(str, paths)]) == 0
    out = tmp_path / "out"
    assert main(["forced", "--config", str(config), "--select-only", "--outdir", str(out),
                 *map(str, paths)]) == 0
    assert (out / "forced_reference.csv").is_file()
    assert not list(out.glob("*_forced_catalog.csv"))

    out2 = tmp_path / "out2"
    assert main(["forced", "--config", str(config), "--np", "1", "--outdir", str(out2),
                 "--master", str(out / "forced_reference.csv"), *map(str, paths)]) == 0
    assert len(list(out2.glob("*_forced_catalog.csv"))) == 2


def test_forced_cli_fits_extension(tmp_path):
    paths, config, _, _ = _write_night(tmp_path, SHIFTS[:2])
    assert main(["photometry", "--config", str(config), "--in-place", *map(str, paths)]) == 0
    # refuses to overwrite the inputs unless told to
    assert main(["forced", "--config", str(config), "--fits-ext", *map(str, paths)]) == 2
    out = tmp_path / "out"
    assert main(["forced", "--config", str(config), "--np", "1", "--fits-ext", "--outdir",
                 str(out), *map(str, paths)]) == 0
    with fits.open(out / "f0_proc.fits") as hdul:
        names = [h.name for h in hdul]
        assert "CATALOG_FORCED" in names and "CATALOG" in names
        forced = Table(hdul["CATALOG_FORCED"].data)
        assert forced["FLUX_APER"].shape[1] == 3
    with fits.open(paths[0]) as hdul:  # the input was untouched
        assert "CATALOG_FORCED" not in [h.name for h in hdul]
    assert main(["forced", "--config", str(config), "--np", "1", "--fits-ext", "--in-place",
                 *map(str, paths)]) == 0
    with fits.open(paths[0]) as hdul:
        assert "CATALOG_FORCED" in [h.name for h in hdul]


def test_forced_cli_dry_run_and_no_inputs(tmp_path):
    paths, config, _, _ = _write_night(tmp_path, SHIFTS[:2])
    assert main(["photometry", "--config", str(config), "--in-place", *map(str, paths)]) == 0
    out = tmp_path / "out"
    assert main(["forced", "--config", str(config), "--dry-run", "--outdir", str(out),
                 *map(str, paths)]) == 0
    assert not list(out.glob("*_forced_catalog.csv"))
    assert main(["forced", str(tmp_path / "nothing_here")]) == 2
