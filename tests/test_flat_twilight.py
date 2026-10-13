"""Tests for robo43.calib.twilight and its use in robo43.calib.flat.

Synthetic data only: twilight flats at very different levels with stars at
different positions (the telescope moves between twilight exposures), checking
that stars are removed from the master, that levels and gradients are handled,
that the in-memory and streaming paths agree, and that dome flats are unchanged.
"""

from __future__ import annotations

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
    scales_from_levels,
)
from robo43.config import CalibSettings, RuntimeSettings, Settings
from robo43.exceptions import CalibrationError
from robo43.instruments.profile import InstrumentLayout, InstrumentProfile, resolve_profile
from robo43.io.headers import read_meta
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
    assert info.n_fallback_px == 0
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


def test_fallback_keeps_structure_present_in_every_frame():
    frames, truth = _make_frames(levels=LEVELS, n_stars=3, fixed_bump=True, seed=13)
    levels = [frame_level(f, 0.5) for f in frames]
    result, info = combine_twilight(_loaders(frames), SHAPE, CalibSettings(), levels=levels)
    assert info.n_fallback_px > 0  # the bump is masked in (nearly) every frame
    ratio = result.data[90:93, 100:103] / (info.l_ref * truth[90:93, 100:103])
    assert ratio.min() > 1.07  # the +10 % structure survives
    assert int(result.n_used.min()) >= 1
    cards = {c[0]: c[1] for c in info.header_cards()}
    assert cards["STARNFB"] == info.n_fallback_px
    assert cards["STARMINF"] == 5  # max(3, half of the 10 frames)
    assert cards["STARPERS"] == 0.5


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
def test_build_master_flats_twilight_group(tmp_path, stream):
    frames, truth = _make_frames(levels=LEVELS, amp_range=(1.0, 3.0), seed=21)
    # T80S style: no OBJECT, recognised from the file name only.
    metas = _write_group(tmp_path, frames, prefix="skyflat")
    settings = _settings(max_memory_mb=2.0 if stream else 4096.0)
    outdir = tmp_path / "masters"
    masters = build_master_flats(metas, settings=settings, profile=_profile(), outdir=outdir)
    assert list(masters) == ["R:sky"]
    assert (outdir / "master_flat_R_sky.fits").exists()
    assert not (outdir / "master_flat_R_dome.fits").exists()
    assert not list(outdir.glob(".flat_stream_*"))  # scratch removed

    master = masters["R:sky"]
    cards = _cards(master)
    assert cards["FLATKIND"] == "twilight"
    assert cards["FLATSCAL"] == "center_median"
    assert cards["STARMASK"] is True
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
