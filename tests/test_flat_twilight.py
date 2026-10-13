"""Tests for robo43.calib.twilight and its use in robo43.calib.flat.

Synthetic data only: twilight flats at very different levels with stars at
different positions (the telescope moves between twilight exposures), checking
that stars are removed from the master -- also stars that repeat within a small
dither in every frame -- that static structure survives, that levels and
gradients are handled, that the in-memory and streaming paths agree, that
dome flats are unchanged, and the classification / QC / reuse safeguards.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from robo43.calib.combine import combine_stack
from robo43.calib.flat import build_master_flats, group_flats_by_filter, normalise_flat
from robo43.calib.twilight import (
    TwilightParams,
    central_region,
    combine_twilight,
    flat_star_qc,
    frame_level,
    is_twilight_flat,
    raw_imagetyp,
    scales_from_levels,
)
from robo43.cli.reduce import _warn_if_flat_master_predates_flatkind
from robo43.config import CalibSettings, RuntimeSettings, Settings
from robo43.exceptions import CalibrationError
from robo43.instruments.profile import InstrumentLayout, InstrumentProfile, resolve_profile
from robo43.io.headers import read_meta, read_raw_imagetyp
from robo43.io.meta import FieldOrigin, FrameMeta, Source

SHAPE = (192, 192)
REL_NOISE = 0.004  # per-pixel noise, relative to the frame level


def _profile() -> InstrumentProfile:
    return InstrumentProfile(
        name="test-instrument",
        matches={},
        pixel_scale_arcsec=0.5,
        gain_e_per_adu=2.0,
        read_noise_e=5.0,
        saturation_adu=60000.0,
        nonlinearity_adu=None,
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


def _settings(**calib_overrides) -> Settings:
    return Settings(calib=CalibSettings(**calib_overrides), runtime=RuntimeSettings())


def _meta(path: Path, *, object=None, origins=None, filter="R") -> FrameMeta:
    return FrameMeta(
        path=path,
        naxis1=SHAPE[1],
        naxis2=SHAPE[0],
        frame_type="flat",
        exptime=1.0,
        date_obs=None,
        filter=filter,
        ra_deg=None,
        dec_deg=None,
        pixel_scale=None,
        ccd_temp=None,
        gain_setting=10.0,
        read_noise=None,
        saturation=None,
        airmass=None,
        object=object,
        telescope=None,
        instrument=None,
        observer=None,
        binning=None,
        readout_mode=None,
        focal_length_mm=None,
        site_lat_deg=None,
        site_lon_deg=None,
        site_elevation_m=None,
        mjd=None,
        night=None,
        is_valid=True,
        profile_name="test-instrument",
        origins=origins or {},
    )


def _truth(shape=SHAPE, seed=1) -> np.ndarray:
    """Flat truth: gentle vignetting (smooth on the 61 px QC scale) plus a 0.3 % pixel pattern."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    r2 = ((yy - shape[0] / 2) / shape[0]) ** 2 + ((xx - shape[1] / 2) / shape[1]) ** 2
    return (1.0 - 0.12 * r2) * (1.0 + 0.003 * rng.standard_normal(shape))


def _stars(rng, n_stars, amp_range, shape=SHAPE, sigma_px=1.5) -> np.ndarray:
    """Relative star image (fractions of the local level), Gaussian PSFs."""
    img = np.zeros(shape)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    for _ in range(n_stars):
        y0 = rng.uniform(12, shape[0] - 12)
        x0 = rng.uniform(12, shape[1] - 12)
        amp = rng.uniform(*amp_range)
        img += amp * np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / (2 * sigma_px**2))
    return img


def _make_frames(
    *,
    levels,
    amp_range=(0.3, 2.0),
    n_stars=6,
    gradient=0.0,
    seed=7,
    fixed_bump=False,
    with_gains=False,
):
    """Frames = level * truth * (1 + gradient) * (1 + stars) + noise.

    Returns ``(frames, truth)``, or ``(frames, truth, gains)`` with ``with_gains`` where
    ``gains`` are the per-frame illumination gradients ``1 + g_i * x``.
    """
    rng = np.random.default_rng(seed)
    truth = _truth()
    xx = np.linspace(-0.5, 0.5, SHAPE[1])[None, :]
    frames = []
    gains = []
    for level in levels:
        g = 1.0 + gradient * rng.uniform(-1, 1) * xx
        gains.append(np.broadcast_to(g, SHAPE))
        img = level * truth * g * (1.0 + _stars(rng, n_stars, amp_range))
        if fixed_bump:
            img[90:93, 100:103] *= 1.10  # real structure, positive in every frame
        img = img + rng.normal(0.0, REL_NOISE * level, SHAPE)
        frames.append(img.astype(np.float32))
    return (frames, truth, gains) if with_gains else (frames, truth)


def _write_group(tmp_path, frames, *, prefix, object=None):
    metas = []
    for i, frame in enumerate(frames):
        path = tmp_path / f"{prefix}-{i:02d}.fits"
        fits.PrimaryHDU(data=frame).writeto(path, overwrite=True)
        metas.append(_meta(path, object=object))
    return metas


def _loaders(frames):
    def make(frame):
        return lambda slices: frame[slices[0], slices[1]]

    return [make(f) for f in frames]


LEVELS = np.linspace(14000.0, 40000.0, 10)  # strongly different twilight levels


# ---------------------------------------------------------------- classification


def test_is_twilight_flat_object():
    assert is_twilight_flat(_meta(Path("a.fits"), object="SkyFlat"))
    assert is_twilight_flat(_meta(Path("a.fits"), object="Twilight R"))
    assert not is_twilight_flat(_meta(Path("a.fits"), object="DomeFlat"))


def test_is_twilight_flat_raw_imagetyp_origin():
    origins = {"imagetyp": FieldOrigin(Source.HEADER, "sky-flat")}
    assert is_twilight_flat(_meta(Path("frame.fits"), origins=origins))
    origins = {"imagetyp": FieldOrigin(Source.HEADER, "Flat Field")}
    assert not is_twilight_flat(_meta(Path("frame.fits"), origins=origins))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("skyflat-20251224-001234.fits.fz", True),
        ("SkyFlat_03.fits", True),
        ("sky-flat-03.fits", True),
        ("twilight-R-03.fits", True),
        ("domeflat-03.fits", False),
        ("flat-lamp-03.fits", False),
    ],
)
def test_is_twilight_flat_filename(name, expected):
    assert is_twilight_flat(_meta(Path(name))) is expected


# (OBJECT, raw IMAGETYP, file name, is a twilight flat)
CLASSIFICATION_CASES = [
    # probe cases of the review: no loose "sky" substring
    (None, None, "dome_sky_flat_01.fits", False),
    ("Husky Dome", None, "frame.fits", False),
    ("Dome flat (no sky)", None, "frame.fits", False),
    ("Husky", None, "frame.fits", False),
    ("TwiFlat", None, "frame.fits", True),
    ("FLAT-DUSK", None, "frame.fits", True),
    ("dawn flat", None, "frame.fits", True),
    (None, None, "TwiFlat.fits", True),
    (None, None, "FLAT-DUSK.fits", True),
    (None, None, "dawn flat.fits", True),
    # T80S: IMAGETYP 'sky-flat', no OBJECT; ROBO43: OBJECT 'SkyFlat'
    (None, "sky-flat", "frame.fits", True),
    ("SkyFlat", None, "frame.fits", True),
    ("SkyFlat", "Flat Field", "frame.fits", True),
    # priority IMAGETYP > OBJECT > file name; a dome word wins in the deciding source
    ("SkyFlat", "dome-flat", "skyflat.fits", False),
    ("DomeFlat", "sky-flat", "frame.fits", True),
    ("Dome", None, "skyflat.fits", False),
    ("Target", None, "skyflat.fits", True),
    ("sky", "Sky Dome Flat", "frame.fits", False),
    # camelCase, digits, plurals, compounds
    (None, None, "SkyFlat_03.fits", True),
    (None, None, "skyflat01.fits", True),
    (None, None, "skyflats-R.fits.fz", True),
    ("TwilightFlat", None, "frame.fits", True),
    (None, None, "domeflat-03.fits", False),
    (None, None, "flat-lamp-03.fits", False),
    (None, "Flat Field", "frame.fits", False),
]


@pytest.mark.parametrize(("obj", "imagetyp", "name", "expected"), CLASSIFICATION_CASES)
def test_is_twilight_flat_table(obj, imagetyp, name, expected):
    origins = {"imagetyp": FieldOrigin(Source.HEADER, imagetyp)} if imagetyp else {}
    meta = _meta(Path(name), object=obj, origins=origins)
    assert is_twilight_flat(meta) is expected


def _write_header_only_fits(path: Path, imagetyp: str | None) -> Path:
    hdu = fits.PrimaryHDU(data=np.zeros((4, 4), dtype=np.float32))
    if imagetyp is not None:
        hdu.header["IMAGETYP"] = imagetyp
    hdu.writeto(path, overwrite=True)
    return path


def test_missing_imagetyp_origin_falls_back_to_the_fits_header(tmp_path):
    """Inventories built before the origin was recorded: re-read IMAGETYP from the file."""
    sky = _write_header_only_fits(tmp_path / "frame_a.fits", "sky-flat")
    dome_named_sky = _write_header_only_fits(tmp_path / "skyflat_b.fits", "dome-flat")
    unlabelled = _write_header_only_fits(tmp_path / "skyflat_c.fits", None)
    assert is_twilight_flat(_meta(sky))
    assert not is_twilight_flat(_meta(dome_named_sky))  # the header outranks the file name
    assert is_twilight_flat(_meta(unlabelled))  # no IMAGETYP anywhere: the file name decides
    assert read_raw_imagetyp(sky) == "sky-flat"
    assert read_raw_imagetyp(unlabelled) is None
    assert raw_imagetyp(_meta(tmp_path / "missing.fits")) is None
    groups = group_flats_by_filter(
        [_meta(sky), _meta(dome_named_sky)], separate_dome_and_sky=True
    )
    assert sorted(groups) == ["R:dome", "R:sky"]
    # an origin, when present, is used as it is
    origin = {"imagetyp": FieldOrigin(Source.HEADER, "Flat Field")}
    assert raw_imagetyp(_meta(sky, origins=origin)) == "Flat Field"


def test_t80s_style_flats_group_as_sky():
    """No OBJECT, T80S file names: used to be mislabelled R:dome."""
    metas = [_meta(Path(f"skyflat-20251224-00{i}.fits.fz")) for i in range(3)]
    metas.append(_meta(Path("domeflat-1.fits")))
    groups = group_flats_by_filter(metas, separate_dome_and_sky=True)
    assert sorted(groups) == ["R:dome", "R:sky"]
    assert len(groups["R:sky"]) == 3


def test_read_meta_records_raw_imagetyp_for_t80s_skyflat():
    profile = resolve_profile(name="t80s")
    header = fits.Header()
    header["IMAGETYP"] = "sky-flat"
    header["EXPTIME"] = 49.0
    header["FILTER"] = "R"
    header["DATE-OBS"] = "2025-12-25T00:28:28"
    header["NAXIS1"] = 9216
    header["NAXIS2"] = 9232
    meta = read_meta(Path("renamed-frame.fits"), profile, header=header)
    assert meta.frame_type == "flat"
    assert meta.object is None
    assert meta.origins["imagetyp"].detail == "sky-flat"
    assert is_twilight_flat(meta)


# ---------------------------------------------------------------- scaling helpers


def test_scales_from_levels_common_level_is_median():
    l_ref, scales = scales_from_levels([10000.0, 20000.0, 40000.0])
    assert l_ref == 20000.0
    np.testing.assert_allclose(scales, [2.0, 1.0, 0.5])


@pytest.mark.parametrize("bad", [[1.0, 0.0], [1.0, -2.0], [1.0, float("nan")], []])
def test_scales_from_levels_rejects_bad_levels(bad):
    with pytest.raises(CalibrationError):
        scales_from_levels(bad)


def test_frame_level_matches_normalise_flat_region():
    rng = np.random.default_rng(3)
    data = rng.normal(1000.0, 10.0, (64, 80))
    settings = CalibSettings()
    _, cards = normalise_flat(data, settings)
    assert frame_level(data, settings.flat_norm_region_frac) == cards[0][1]
    rows, cols = central_region(data.shape, settings.flat_norm_region_frac)
    assert (rows.stop - rows.start, cols.stop - cols.start) == (32, 40)


def test_frame_level_ignores_nan_and_flags_an_all_nan_region():
    frame = np.full((64, 80), 100.0)
    frame[30:34, 38:42] = np.nan
    assert frame_level(frame, 0.5) == 100.0
    frame[central_region(frame.shape, 0.5)] = np.nan
    assert np.isnan(frame_level(frame, 0.5))  # no raise: the build drops such a frame


def test_twilight_params_validation():
    with pytest.raises(CalibrationError):
        TwilightParams(star_smooth_blocks=4)
    with pytest.raises(CalibrationError):
        TwilightParams(qc_threshold=0.9)


# ---------------------------------------------------------------- the combine


def _qc(master_like):
    return flat_star_qc(master_like / np.median(master_like)).n_peaks


def test_stars_removed_old_method_leaves_them():
    """Unscaled 3-sigma clipping keeps every star; the twilight combine removes them."""
    frames, _truth_img = _make_frames(levels=LEVELS)
    settings = CalibSettings()

    old = combine_stack(frames, settings)  # what build_master_flats used to do
    old_peaks = _qc(old.data.astype(np.float64))
    assert old_peaks >= 20  # ~60 stars, all still there

    levels = [float(np.median(f[48:144, 48:144])) for f in frames]
    result, info = combine_twilight(_loaders(frames), SHAPE, settings, levels=levels)
    assert _qc(result.data.astype(np.float64)) == 0
    assert info.masked_fraction > 0.0
    assert info.n_filled_px == 0
    assert int(result.n_used.min()) >= 3


def test_master_matches_truth_with_different_levels_and_gradients():
    frames, truth, gains = _make_frames(levels=LEVELS, gradient=0.04, seed=11, with_gains=True)
    levels = [frame_level(f, 0.5) for f in frames]
    result, info = combine_twilight(_loaders(frames), SHAPE, CalibSettings(), levels=levels)
    master = result.data.astype(np.float64)
    # Every scaled frame is l_ref * truth * (1 + g_i * x); the mean gradient is what remains.
    expected = info.l_ref * truth * np.mean(gains, axis=0)
    resid = master / expected - 1.0
    assert np.abs(np.median(resid)) < 5e-3
    assert np.abs(resid).max() < 0.02  # 10 frames x 0.4 % noise; unmasked stars would give >> 2 %
    assert _qc(master) == 0


def test_levels_are_equalised_before_combining():
    """Asymmetric levels: an unscaled mean would sit ~19 % above the common level."""
    levels = [14000.0] * 9 + [40000.0]
    frames, truth = _make_frames(levels=levels, n_stars=2, seed=17)
    measured = [frame_level(f, 0.5) for f in frames]
    result, info = combine_twilight(_loaders(frames), SHAPE, CalibSettings(), levels=measured)
    assert info.l_ref == pytest.approx(14000.0, rel=0.01)
    ratio = result.data.astype(np.float64) / (info.l_ref * truth)
    assert abs(np.median(ratio) - 1.0) < 5e-3
    np.testing.assert_allclose(info.scales[-1], info.l_ref / measured[-1])


def test_variance_is_at_the_scaled_noise_level():
    frames, _truth_img = _make_frames(levels=LEVELS, seed=5)
    levels = [frame_level(f, 0.5) for f in frames]
    result, info = combine_twilight(
        _loaders(frames), SHAPE, CalibSettings(), levels=levels
    )
    rel_var = np.median(result.variance[result.n_used >= 9]) / info.l_ref**2
    expected = REL_NOISE**2 / 10.0
    assert 0.6 * expected < rel_var < 1.6 * expected


def test_memmap_and_memory_loaders_identical_and_blocks_still_clean(tmp_path):
    frames, _truth_img = _make_frames(levels=LEVELS, seed=9)
    levels = [frame_level(f, 0.5) for f in frames]
    mm = np.memmap(tmp_path / "stack.dat", mode="w+", dtype=np.float32, shape=(len(frames), *SHAPE))
    for i, f in enumerate(frames):
        mm[i] = f
    mm.flush()

    def mm_loader(idx):
        return lambda s: np.asarray(mm[idx, s[0], s[1]])

    settings = CalibSettings(max_memory_mb=2.0)  # forces several row blocks
    res_mem, _ = combine_twilight(_loaders(frames), SHAPE, settings, levels=levels)
    res_mm, _ = combine_twilight(
        [mm_loader(i) for i in range(len(frames))], SHAPE, settings, levels=levels
    )
    np.testing.assert_array_equal(res_mem.data, res_mm.data)
    np.testing.assert_array_equal(res_mem.n_used, res_mm.n_used)
    assert _qc(res_mem.data.astype(np.float64)) == 0

    res_one, _ = combine_twilight(
        _loaders(frames), SHAPE, CalibSettings(max_memory_mb=1024.0), levels=levels
    )
    assert _qc(res_one.data.astype(np.float64)) == 0
    # Block size only changes the per-block noise estimate: masters agree to the noise.
    diff = res_mem.data.astype(np.float64) / res_one.data - 1.0
    assert np.abs(diff).max() < 5e-3


def test_static_structure_in_every_frame_survives():
    """A +10 % bump at one place in every frame is real structure, not a star."""
    frames, truth = _make_frames(levels=LEVELS, n_stars=3, fixed_bump=True, seed=13)
    levels = [frame_level(f, 0.5) for f in frames]
    result, info = combine_twilight(_loaders(frames), SHAPE, CalibSettings(), levels=levels)
    assert info.n_static_px > 0  # the bump is masked in every frame by pass 1, then recognised
    ratio = result.data[90:93, 100:103] / (info.l_ref * truth[90:93, 100:103])
    assert ratio.min() > 1.07  # the +10 % structure survives
    assert int(result.n_used[90:93, 100:103].min()) == 10  # pass 2 divides it out: nothing masked
    assert int(result.n_used.min()) >= 1
    cards = {c[0]: c[1] for c in info.header_cards()}
    assert cards["STARNFB"] == info.n_static_px
    assert cards["STARNFL"] == info.n_filled_px
    assert cards["STARMINF"] == 3
    assert cards["STARPASS"] == 2
    assert cards["STARSTAT"] == 3.0
    assert "STARPERS" not in cards


def _make_dithered_frames(*, levels, seed, n_stars=15, shift=2.0, n_hot=25, hot=0.10):
    """Stars at fixed base positions, shifted by up to ``shift`` px in every frame,
    plus static hot pixels (``hot`` above the truth) far from every star.

    Returns ``(frames, truth_norm, (hot_y, hot_x))``; ``truth_norm`` is the true flat
    (hot pixels included) normalised to the median of its central region.
    """
    rng = np.random.default_rng(seed)
    truth = _truth(seed=seed)
    yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    base = [
        (rng.uniform(15, SHAPE[0] - 15), rng.uniform(15, SHAPE[1] - 15), rng.uniform(0.3, 2.0))
        for _ in range(n_stars)
    ]
    hot_pix: list[tuple[int, int]] = []
    while len(hot_pix) < n_hot:
        y, x = int(rng.integers(5, SHAPE[0] - 5)), int(rng.integers(5, SHAPE[1] - 5))
        if all((y - by) ** 2 + (x - bx) ** 2 > 18**2 for by, bx, _ in base):
            hot_pix.append((y, x))
    hot_y, hot_x = np.array(hot_pix).T
    truth_hot = truth.copy()
    truth_hot[hot_y, hot_x] *= 1.0 + hot
    frames = []
    for level in levels:
        stars = np.zeros(SHAPE)
        for by, bx, amp in base:
            dy, dx = rng.uniform(-shift, shift, 2)
            stars += amp * np.exp(-((yy - by - dy) ** 2 + (xx - bx - dx) ** 2) / (2 * 1.5**2))
        img = level * truth_hot * (1.0 + stars) + rng.normal(0.0, REL_NOISE * level, SHAPE)
        frames.append(img.astype(np.float32))
    centre = (slice(48, 144), slice(48, 144))
    return frames, truth_hot / np.median(truth_hot[centre]), (hot_y, hot_x)


def test_stars_repeating_within_a_small_dither_are_removed_and_hot_pixels_kept():
    """Regression (review F1): a star on the same pixels in (nearly) every frame.

    The old persistence fallback combined any pixel masked in half the frames back
    in, so 15 stars shifted by <= 2 px left 15 QC peaks and a +100 % error.  Pass 2
    (frame / preliminary master) tells them from static structure, which survives.
    """
    frames, truth_norm, (hot_y, hot_x) = _make_dithered_frames(levels=LEVELS, seed=8)
    levels = [frame_level(f, 0.5) for f in frames]
    result, info = combine_twilight(_loaders(frames), SHAPE, CalibSettings(), levels=levels)
    master = result.data.astype(np.float64)
    master /= np.median(master[48:144, 48:144])
    assert flat_star_qc(master).n_peaks == 0
    err = master / truth_norm - 1.0
    measured = result.weight > 0  # filled pixels (below) are flagged, not measured
    assert np.abs(err[measured]).max() < 0.01
    # the +10 % hot pixels, masked in every frame by pass 1, are still there
    assert (master[hot_y, hot_x] / truth_norm[hot_y, hot_x]).min() > 0.98
    assert master[hot_y, hot_x].min() > 1.07 * np.median(master)
    assert info.n_static_px >= 25
    # pixels under a star in every frame: smooth fill, zero weight, flagged, bounded error
    filled = result.n_used == 0
    assert info.n_filled_px == int(filled.sum()) > 0
    assert np.all(result.weight[filled] == 0.0) and np.all(result.mask[filled])
    assert np.abs(err[filled]).max() < 0.03


def test_dithered_stars_with_hot_pixels_through_memmap_blocks(tmp_path):
    """Several row blocks (halo, static test and fill across block edges) give the same answer."""
    frames, truth_norm, _hot = _make_dithered_frames(levels=LEVELS, seed=8)
    levels = [frame_level(f, 0.5) for f in frames]
    mm = np.memmap(tmp_path / "stack.dat", mode="w+", dtype=np.float32, shape=(len(frames), *SHAPE))
    for i, f in enumerate(frames):
        mm[i] = f
    mm.flush()
    loaders = [(lambda s, i=i: np.asarray(mm[i, s[0], s[1]])) for i in range(len(frames))]
    result, _info = combine_twilight(
        loaders, SHAPE, CalibSettings(max_memory_mb=2.0), levels=levels
    )
    master = result.data.astype(np.float64)
    master /= np.median(master[48:144, 48:144])
    assert flat_star_qc(master).n_peaks == 0
    assert np.abs((master / truth_norm - 1.0)[result.weight > 0]).max() < 0.012


def test_single_contributing_frame_has_zero_weight_in_groups_of_three_or_more():
    """Review F9: one sample has no variance; its weight must not come from the noise floor."""
    frames, _truth_img = _make_frames(levels=LEVELS[:4], n_stars=0, seed=12)
    for f in frames[:3]:
        f[50, 60] = np.nan  # pixel (50, 60) is finite in the last frame only
    levels = [frame_level(f, 0.5) for f in frames]
    result, _info = combine_twilight(
        _loaders(frames),
        SHAPE,
        CalibSettings(),
        levels=levels,
        read_noise_e=5.0,
        gain_e_per_adu=1.0,
    )
    assert int(result.n_used[50, 60]) == 1
    assert result.weight[50, 60] == 0.0 and bool(result.mask[50, 60])
    assert np.isfinite(result.data[50, 60]) and result.data[50, 60] > 0
    assert result.weight[50, 61] > 0 and not bool(result.mask[50, 61])
    # two frames only: the rule applies from three frames on
    result2, _ = combine_twilight(
        _loaders(frames[2:]), SHAPE, CalibSettings(), levels=levels[2:],
        read_noise_e=5.0, gain_e_per_adu=1.0,
    )
    assert int(result2.n_used[50, 60]) == 1
    assert result2.weight[50, 60] > 0


def test_small_stack_caps_min_frames():
    frames, _truth_img = _make_frames(levels=LEVELS[:2], n_stars=2, seed=2)
    levels = [frame_level(f, 0.5) for f in frames]
    result, info = combine_twilight(_loaders(frames), SHAPE, CalibSettings(), levels=levels)
    assert info.min_frames == 1
    assert int(result.n_used.min()) >= 1


def test_combine_twilight_needs_two_frames():
    frames, _truth_img = _make_frames(levels=LEVELS[:1])
    with pytest.raises(CalibrationError):
        combine_twilight(_loaders(frames), SHAPE, CalibSettings(), levels=[20000.0])


# ---------------------------------------------------------------- build_master_flats


def _cards(master):
    return {c[0]: c[1] for c in master.header_cards}


@pytest.mark.parametrize("stream", [False, True])
def test_build_master_flats_twilight_group(tmp_path, stream, caplog):
    frames, truth = _make_frames(levels=LEVELS, amp_range=(1.0, 3.0), seed=21)
    # T80S style: no OBJECT, recognised from the file name only.
    metas = _write_group(tmp_path, frames, prefix="skyflat")
    settings = _settings(max_memory_mb=2.0 if stream else 4096.0)
    outdir = tmp_path / "masters"
    caplog.set_level(logging.WARNING)
    masters = build_master_flats(metas, settings=settings, profile=_profile(), outdir=outdir)
    assert list(masters) == ["R:sky"]
    assert not [r for r in caplog.records if "compact positive peaks" in r.message]
    assert (outdir / "master_flat_R_sky.fits").exists()
    assert not (outdir / "master_flat_R_dome.fits").exists()
    assert not list(outdir.glob(".flat_stream_*"))  # scratch removed

    master = masters["R:sky"]
    cards = _cards(master)
    assert cards["FLATKIND"] == "twilight"
    assert cards["FLATSCAL"] == "center_median"
    assert cards["STARMASK"] is True
    assert cards["STARPASS"] == 2
    assert cards["STARNSIG"] == 4.0
    assert cards["STARMFRC"] > 0
    assert cards["NUSEDMIN"] >= 3
    assert cards["FLATQCN"] == 0
    assert cards["FLATQCT"] == 1.03 and cards["FLATQCW"] == 61
    # Normalised master follows the truth (stars gone), to the noise.
    resid = master.data / (truth / np.median(truth[48:144, 48:144])) - 1.0
    assert np.abs(resid).max() < 0.02
    assert cards["FLATNORM"] == pytest.approx(float(np.median(LEVELS)), rel=0.02)

    # The same data through the old method leave many peaks (sensitivity of the QC).
    old = combine_stack(frames, CalibSettings())
    assert flat_star_qc(old.data / np.median(old.data)).n_peaks >= 20


def test_build_master_flats_warns_on_star_residuals(tmp_path, caplog):
    """A dome group (no masking) of star-laden frames at different levels trips the QC warning."""
    frames, _truth_img = _make_frames(levels=LEVELS, amp_range=(2.0, 4.0), seed=3)
    metas = _write_group(tmp_path, frames, prefix="lamp", object="DomeFlat")
    caplog.set_level(logging.WARNING)
    params = TwilightParams(qc_warn_per_mpix=100.0)
    masters = build_master_flats(
        metas,
        settings=_settings(),
        profile=_profile(),
        outdir=tmp_path / "masters",
        twilight_params=params,
    )
    cards = _cards(masters["R:dome"])
    assert cards["FLATKIND"] == "dome"
    assert cards["STARMASK"] is False
    assert cards["FLATQCN"] > 5
    assert any("compact positive peaks" in r.message for r in caplog.records)


def test_dome_group_unchanged_by_default(tmp_path):
    frames, _truth_img = _make_frames(levels=LEVELS, n_stars=1, seed=4)
    metas = _write_group(tmp_path, frames, prefix="lamp", object="DomeFlat")
    masters = build_master_flats(
        metas, settings=_settings(), profile=_profile(), outdir=tmp_path / "masters"
    )
    expected = combine_stack(frames, CalibSettings(), read_noise_e=5.0)
    expected_norm, _ = normalise_flat(expected.data, CalibSettings())
    np.testing.assert_array_equal(masters["R:dome"].data, expected_norm)
    cards = _cards(masters["R:dome"])
    assert cards["FLATSCAL"] == "none"
    assert "FLATLREF" not in cards


def test_scale_all_flats_option_scales_dome_before_clipping(tmp_path):
    frames, _truth_img = _make_frames(levels=LEVELS, n_stars=1, seed=4)
    metas = _write_group(tmp_path, frames, prefix="lamp", object="DomeFlat")
    masters = build_master_flats(
        metas,
        settings=_settings(),
        profile=_profile(),
        outdir=tmp_path / "masters",
        twilight_params=TwilightParams(scale_all_flats=True),
    )
    levels = [frame_level(f.astype(np.float64), 0.5) for f in frames]
    _, scales = scales_from_levels(levels)
    expected = combine_stack(frames, CalibSettings(), scales=scales, read_noise_e=5.0)
    expected_norm, _ = normalise_flat(expected.data, CalibSettings())
    # frame_level here is computed on float64 copies, the build's on float32: last-bit differences
    np.testing.assert_allclose(masters["R:dome"].data, expected_norm, rtol=1e-5)
    cards = _cards(masters["R:dome"])
    assert cards["FLATSCAL"] == "center_median"
    assert cards["STARMASK"] is False


def test_single_twilight_frame_is_flagged_not_masked(tmp_path, caplog):
    frames, _truth_img = _make_frames(levels=[25000.0], seed=6)
    metas = _write_group(tmp_path, frames, prefix="skyflat")
    caplog.set_level(logging.WARNING)
    masters = build_master_flats(
        metas, settings=_settings(), profile=_profile(), outdir=tmp_path / "masters"
    )
    cards = _cards(masters["R:sky"])
    assert cards["FLATKIND"] == "twilight"
    assert cards["STARMASK"] is False
    assert any("single usable frame" in r.message for r in caplog.records)


# ---------------------------------------------------------------- review safeguards


def test_qc_default_threshold_and_warning_message(tmp_path, caplog):
    """Review F2: 2 peaks per Mpx by default; the warning gives count, rate and threshold."""
    assert TwilightParams().qc_warn_per_mpix == 2.0
    frames, _truth_img = _make_frames(levels=LEVELS, amp_range=(2.0, 4.0), seed=3)
    metas = _write_group(tmp_path, frames, prefix="lamp", object="DomeFlat")
    caplog.set_level(logging.WARNING)
    masters = build_master_flats(
        metas, settings=_settings(), profile=_profile(), outdir=tmp_path / "masters"
    )
    n_peaks = _cards(masters["R:dome"])["FLATQCN"]
    rate = 1.0e6 * n_peaks / (SHAPE[0] * SHAPE[1])
    assert rate > 2.0
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    qc = [m for m in messages if "compact positive peaks" in m]
    assert len(qc) == 1
    assert f"{n_peaks} compact positive peaks" in qc[0]
    assert f"{rate:.2f} per Mpx" in qc[0]
    assert "threshold 2.00 per Mpx" in qc[0]


def test_mixed_flat_group_warns_and_is_combined_as_dome(tmp_path, caplog):
    """Review F4a: dome and sky flats not separated -> a mixed group is flagged, not silent."""
    frames, _truth_img = _make_frames(levels=LEVELS[:6], n_stars=1, seed=14)
    metas = _write_group(tmp_path, frames[:3], prefix="sky", object="SkyFlat")
    metas += _write_group(tmp_path, frames[3:], prefix="lamp", object="DomeFlat")
    caplog.set_level(logging.WARNING)
    masters = build_master_flats(
        metas,
        settings=_settings(separate_dome_and_sky_flats=False),
        profile=_profile(),
        outdir=tmp_path / "masters",
    )
    assert list(masters) == ["R"]
    cards = _cards(masters["R"])
    assert cards["FLATKIND"] == "dome" and cards["STARMASK"] is False
    mixed = [r.getMessage() for r in caplog.records if "is mixed" in r.getMessage()]
    assert len(mixed) == 1 and "3 of 6" in mixed[0] and "group R" in mixed[0]


def test_pure_groups_do_not_warn_about_mixing(tmp_path, caplog):
    frames, _truth_img = _make_frames(levels=LEVELS[:4], n_stars=1, seed=15)
    metas = _write_group(tmp_path, frames, prefix="sky", object="SkyFlat")
    caplog.set_level(logging.WARNING)
    build_master_flats(
        metas,
        settings=_settings(separate_dome_and_sky_flats=False),
        profile=_profile(),
        outdir=tmp_path / "masters",
    )
    assert not [r for r in caplog.records if "is mixed" in r.getMessage()]


def test_frame_without_a_level_is_dropped_with_a_warning(tmp_path, caplog):
    """Review F6: a NaN in the central region must not abort the build."""
    frames, _truth_img = _make_frames(levels=LEVELS[:5], n_stars=2, seed=16)
    frames[1][100:105, 100:105] = np.nan  # a patch: the NaN-ignoring level is still fine
    frames[3][48:144, 48:144] = np.nan  # the whole central region: no level at all
    metas = _write_group(tmp_path, frames, prefix="skyflat")
    profile = dataclasses.replace(_profile(), saturation_adu=None)  # no level cut to catch it first
    caplog.set_level(logging.WARNING)
    masters = build_master_flats(
        metas, settings=_settings(), profile=profile, outdir=tmp_path / "masters"
    )
    cards = _cards(masters["R:sky"])
    assert cards["FLATNFRM"] == 4 and cards["NCOMBINE"] == 4
    assert cards["STARMASK"] is True
    assert np.isfinite(masters["R:sky"].data).all()
    dropped = [r.getMessage() for r in caplog.records if "dropping flat frame" in r.getMessage()]
    assert len(dropped) == 1 and "skyflat-03.fits" in dropped[0]


def test_group_without_any_level_raises(tmp_path):
    frames, _truth_img = _make_frames(levels=LEVELS[:3], n_stars=1, seed=17)
    for frame in frames:
        frame[48:144, 48:144] = np.nan
    metas = _write_group(tmp_path, frames, prefix="skyflat")
    profile = dataclasses.replace(_profile(), saturation_adu=None)
    with pytest.raises(CalibrationError, match="central-region level"):
        build_master_flats(metas, settings=_settings(), profile=profile, outdir=tmp_path / "m")


def test_reused_master_without_flatkind_is_flagged(tmp_path, caplog):
    """Review F4c: a master built before the fix (no FLATKIND card) warns, is not rebuilt."""
    old = tmp_path / "master_flat_R_dome.fits"
    fits.PrimaryHDU(data=np.ones((4, 4), dtype=np.float32)).writeto(old)
    new = tmp_path / "master_flat_G_sky.fits"
    hdu = fits.PrimaryHDU(data=np.ones((4, 4), dtype=np.float32))
    hdu.header["FLATKIND"] = "twilight"
    hdu.writeto(new)
    caplog.set_level(logging.WARNING)
    assert _warn_if_flat_master_predates_flatkind(old) is True
    assert _warn_if_flat_master_predates_flatkind(new) is False
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(messages) == 1
    assert "master_flat_R_dome.fits" in messages[0] and "FLATKIND" in messages[0]
    assert "rebuilt automatically" in messages[0]
    assert not np.isnan(fits.getdata(old)).any()  # untouched
