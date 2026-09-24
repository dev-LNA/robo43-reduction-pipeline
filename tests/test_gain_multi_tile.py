"""Tests for per-amplifier-tile gain measurement and application.

Covers three things the single-tile tests in test_gain.py cannot:

1. Recovering distinct, known per-tile gains from a synthetic 2x2-amplifier
   mosaic by photon transfer (robo43.calib.gain.measure_gain).
2. Level-matched flat pairing across nights/levels
   (robo43.calib.gain._level_matched_pairs).
3. That converting ADU->electrons per tile *before* flat-fielding -- both
   when the master flat itself is built (robo43.calib.flat) and when a
   science frame is reduced (robo43.reduce.frame) -- removes the amplifier
   seam a single whole-frame gain would leave behind.

Builds its own synthetic FITS files and FrameMeta objects; does not use
tests/conftest.py.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest
from astropy.io import fits

from robo43.calib.flat import build_master_flats
from robo43.calib.gain import (
    GainMeasurement,
    GainMeasurementSet,
    _level_matched_pairs,
    measure_gain,
)
from robo43.calib.overscan import OverscanGeometry
from robo43.calib.set import CalibrationSet
from robo43.config import CalibSettings, RuntimeSettings, Settings
from robo43.instruments.profile import InstrumentLayout, InstrumentProfile
from robo43.io.meta import FrameMeta
from robo43.reduce.frame import reduce_frame

TRUE_TILE_K = [0.7, 0.8, 0.9, 1.0]  # row-major over a 2x2 amplifier mosaic
TRUE_RN_E = 5.0
BIAS_LEVEL_ADU = 300.0


def _profile(n_amplifiers: int = 4) -> InstrumentProfile:
    return InstrumentProfile(
        name="test-multiamp",
        matches={},
        pixel_scale_arcsec=0.5,
        gain_e_per_adu=None,
        read_noise_e=99.0,  # deliberately wrong: proves the measured RN wins
        saturation_adu=60000.0,
        nonlinearity_adu=None,
        binning_keyword="XBINNING",
        layout=InstrumentLayout(n_amplifiers=n_amplifiers),
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


def _write_fits(path, data) -> None:
    hdu = fits.PrimaryHDU(data=data.astype(np.float32))
    hdu.writeto(path, overwrite=True)


def _meta(
    path, shape, *, frame_type, filter=None, exptime=5.0, night=None,
    gain_setting=10.0, readout_mode="MULTIAMP", binning=1,
) -> FrameMeta:
    return FrameMeta(
        path=path,
        naxis1=shape[1],
        naxis2=shape[0],
        frame_type=frame_type,
        exptime=exptime,
        date_obs=None,
        filter=filter,
        ra_deg=None,
        dec_deg=None,
        pixel_scale=None,
        ccd_temp=None,
        gain_setting=gain_setting,
        read_noise=None,
        saturation=None,
        airmass=None,
        object="DomeFlat" if frame_type == "flat" else None,
        telescope=None,
        instrument=None,
        observer=None,
        binning=binning,
        readout_mode=readout_mode,
        focal_length_mm=None,
        site_lat_deg=None,
        site_lon_deg=None,
        site_elevation_m=None,
        mjd=None,
        night=night,
        is_valid=True,
        profile_name="test-multiamp",
        origins={},
    )


RAW_SIZE = 800
_MID = RAW_SIZE // 2  # overscan strip sits at columns/rows (_MID - 1, _MID)


def _mosaic_geometry(tile: int) -> tuple[OverscanGeometry, tuple[slice, slice]]:
    """A 2x2-amplifier geometry: raw 800x800, 2px interior overscan strips
    on both axes, giving 4 tiles of 399x399 each in the 798x798 stitched
    output -- large enough that a 25%-linear-dimension central crop
    (settings.calib.gain_region_frac's default) still samples ~10^4 pixels
    per tile, for a tight photon-transfer recovery. Returns the geometry and
    ``tile``'s own stitched-space slice.
    """
    geom = OverscanGeometry(
        shape=(RAW_SIZE, RAW_SIZE),
        columns=(_MID - 1, _MID),
        rows=(_MID - 1, _MID),
        bias_level=BIAS_LEVEL_ADU,
        flat_level=20000.0,
        detected=True,
        source="detected",
    )
    tiles = geom.science_tile_slices
    assert len(tiles) == 4
    return geom, tiles[tile]


def _raw_tile_ranges() -> list[tuple[slice, slice]]:
    """The 4 tiles' RAW (pre-trim) row/col ranges, row-major, matching
    _mosaic_geometry's interior overscan strip split."""
    row_ranges = [slice(0, _MID - 1), slice(_MID + 1, RAW_SIZE)]
    col_ranges = [slice(0, _MID - 1), slice(_MID + 1, RAW_SIZE)]
    return [(r, c) for r in row_ranges for c in col_ranges]


def _write_mosaic_frame(
    path, *, rng, signal_e_per_tile, read_noise_e=TRUE_RN_E, k_per_tile=TRUE_TILE_K,
) -> None:
    """A raw RAW_SIZE x RAW_SIZE frame: each of the 4 tiles reads out at
    BIAS_LEVEL_ADU + (poisson(signal_e) + read_noise) / k for its own tile,
    with the 2px interior strips (both axes) at a flat BIAS_LEVEL_ADU (no
    noise) standing in for overscan/prescan -- so apply_overscan's per-tile
    correction removes exactly the pedestal this function put in.
    ``signal_e_per_tile`` of 0 (bias frames) uses no Poisson term.
    """
    data = np.full((RAW_SIZE, RAW_SIZE), BIAS_LEVEL_ADU, dtype=np.float64)
    for (row_sl, col_sl), k in zip(_raw_tile_ranges(), k_per_tile, strict=True):
        shape = (row_sl.stop - row_sl.start, col_sl.stop - col_sl.start)
        rn = rng.normal(0.0, read_noise_e, size=shape)
        if signal_e_per_tile > 0:
            photons = rng.poisson(signal_e_per_tile, size=shape).astype(np.float64)
        else:
            photons = 0.0
        data[row_sl, col_sl] = BIAS_LEVEL_ADU + (photons + rn) / k
    _write_fits(path, data)


def test_measure_gain_recovers_per_tile_gains_from_synthetic_2x2_ptc(tmp_path):
    """Four amplifiers with distinct true gains (0.7/0.8/0.9/1.0 e-/ADU):
    per-tile PTC recovers each within ~2%, and the frame-level summary is
    their median."""
    rng = np.random.default_rng(20240921)
    geom, _ = _mosaic_geometry(0)

    metas = []
    for i in range(4):
        path = tmp_path / f"bias_{i}.fits"
        _write_mosaic_frame(path, rng=rng, signal_e_per_tile=0.0)
        metas.append(_meta(path, (RAW_SIZE, RAW_SIZE), frame_type="bias"))

    for filt_idx, signal_e in enumerate([10000.0, 13000.0, 16000.0, 19000.0, 22000.0]):
        for j in range(2):
            path = tmp_path / f"flat_{filt_idx}_{j}.fits"
            _write_mosaic_frame(path, rng=rng, signal_e_per_tile=signal_e)
            metas.append(_meta(path, (RAW_SIZE, RAW_SIZE), frame_type="flat", filter=f"FILT{filt_idx}"))

    profile = _profile()
    settings = _settings()
    result = measure_gain(metas, settings=settings, profile=profile, overscan=geom)

    tiles = result.tiles_for("MULTIAMP", 10.0, 1)
    assert tiles is not None
    assert len(tiles) == 4

    for tile_idx, true_k in enumerate(TRUE_TILE_K):
        tm = tiles[tile_idx]
        assert tm.tile_index == tile_idx
        assert tm.n_tiles == 4
        assert tm.gain_e_per_adu == pytest.approx(true_k, rel=0.02), (
            f"tile {tile_idx}: expected K~{true_k}, got {tm.gain_e_per_adu}"
        )
        assert tm.read_noise_e == pytest.approx(TRUE_RN_E, rel=0.15)

    summary = result.get("MULTIAMP", 10.0, 1)
    assert summary is not None
    assert summary.tile_index == -1
    assert summary.gain_e_per_adu == pytest.approx(float(np.median(TRUE_TILE_K)), rel=0.02)


def test_level_matched_pairs_pairs_within_tolerance_and_leaves_mismatched_unpaired():
    """Frames pair with their nearest-level match within tolerance; a level
    outside tolerance of every other frame is left unpaired, never forced."""
    a = FrameMeta(
        path=None, naxis1=1, naxis2=1, frame_type="flat", exptime=1.0, date_obs=None,
        filter="R", ra_deg=None, dec_deg=None, pixel_scale=None, ccd_temp=None,
        gain_setting=10.0, read_noise=None, saturation=None, airmass=None, object=None,
        telescope=None, instrument=None, observer=None, binning=1, readout_mode=None,
        focal_length_mm=None, site_lat_deg=None, site_lon_deg=None, site_elevation_m=None,
        mjd=None, night=None, is_valid=True, profile_name="x", origins={},
    )
    import dataclasses as dc

    b = dc.replace(a, path="b")
    c = dc.replace(a, path="c")
    d = dc.replace(a, path="d")
    a = dc.replace(a, path="a")

    # a=20000, b=20500 (2.5% apart -> pairs), c=15000 isolated, d=14900
    # (0.7% from c -> pairs). Tolerance 5%.
    items = [(a, 20000.0), (b, 20500.0), (c, 15000.0), (d, 14900.0)]
    pairs = _level_matched_pairs(items, tol_frac=0.05)

    pair_paths = sorted(tuple(sorted((p1.path, p2.path))) for p1, p2 in pairs)
    assert pair_paths == [("a", "b"), ("c", "d")]


def test_level_matched_pairs_respects_tight_tolerance():
    """Tightening the tolerance below a pair's actual level gap leaves both
    frames unpaired rather than forcing a mismatched pair."""
    import dataclasses as dc

    base = FrameMeta(
        path="x", naxis1=1, naxis2=1, frame_type="flat", exptime=1.0, date_obs=None,
        filter="R", ra_deg=None, dec_deg=None, pixel_scale=None, ccd_temp=None,
        gain_setting=10.0, read_noise=None, saturation=None, airmass=None, object=None,
        telescope=None, instrument=None, observer=None, binning=1, readout_mode=None,
        focal_length_mm=None, site_lat_deg=None, site_lon_deg=None, site_elevation_m=None,
        mjd=None, night=None, is_valid=True, profile_name="x", origins={},
    )
    m1 = dc.replace(base, path="m1")
    m2 = dc.replace(base, path="m2")
    items = [(m1, 20000.0), (m2, 21000.0)]  # 5% apart

    assert _level_matched_pairs(items, tol_frac=0.10) != []
    assert _level_matched_pairs(items, tol_frac=0.01) == []


def test_per_tile_gain_before_flat_removes_amplifier_seam(tmp_path):
    """A master flat built with per-tile gain conversion, and a science
    frame reduced the same way, both stay seam-free across the 2x2
    amplifier boundaries -- even though each amplifier's raw ADU level
    differs (uniform illumination divided by four different K's). Applying
    only a single frame-level gain (the pre-per-tile-storage fallback)
    instead leaves a real seam, which this test also confirms as contrast.
    """
    rng = np.random.default_rng(777)
    geom, _ = _mosaic_geometry(0)
    profile = _profile()
    settings = _settings()

    key = ("MULTIAMP", 10.0, 1)
    tile_measurements = tuple(
        GainMeasurement(
            readout_mode="MULTIAMP", gain_setting=10.0, binning=1,
            gain_e_per_adu=k, read_noise_e=TRUE_RN_E, n_pairs=3, scatter_frac=0.0,
            tile_index=i, n_tiles=4,
        )
        for i, k in enumerate(TRUE_TILE_K)
    )
    summary = GainMeasurement(
        readout_mode="MULTIAMP", gain_setting=10.0, binning=1,
        gain_e_per_adu=float(np.median(TRUE_TILE_K)), read_noise_e=TRUE_RN_E,
        n_pairs=3, scatter_frac=0.0, tile_index=-1, n_tiles=4,
    )
    per_tile_gain_measurements = GainMeasurementSet({key: summary}, {key: tile_measurements})
    frame_level_only_gain_measurements = GainMeasurementSet({key: summary})  # no tile_measurements

    # --- Build a master flat with genuine amplifier-to-amplifier ADU steps
    # (uniform illumination, 15000 e-, divided by 4 different K's) -------
    flat_signal_e = 15000.0
    flat_metas = []
    for i in range(2):
        path = tmp_path / f"flat_{i}.fits"
        _write_mosaic_frame(path, rng=rng, signal_e_per_tile=flat_signal_e, read_noise_e=20.0)
        flat_metas.append(_meta(path, (RAW_SIZE, RAW_SIZE), frame_type="flat", filter="R"))

    masters = build_master_flats(
        flat_metas, settings=settings, profile=profile, outdir=tmp_path / "masters",
        overscan=geom, gain_measurements=per_tile_gain_measurements,
    )
    master_flat = masters["R:dome"]
    tiles = geom.science_tile_slices
    flat_tile_medians = [float(np.median(master_flat.data[r, c])) for r, c in tiles]
    flat_spread = (max(flat_tile_medians) - min(flat_tile_medians)) / np.mean(flat_tile_medians)
    assert flat_spread < 0.01, f"master flat has a seam across tiles: {flat_tile_medians}"

    # --- Reduce a science frame through the same per-tile gain, then the
    # now-seam-free flat -----------------------------------------------
    sci_signal_e = 8000.0
    raw_path = tmp_path / "science.fits"
    _write_mosaic_frame(raw_path, rng=rng, signal_e_per_tile=sci_signal_e, read_noise_e=1.0)
    bias_master = np.full((RAW_SIZE - 2, RAW_SIZE - 2), BIAS_LEVEL_ADU, dtype=np.float32)
    bias_path = tmp_path / "master_bias.fits"
    _write_fits(bias_path, bias_master)

    def _reduce_with(gain_measurements):
        calib = CalibrationSet(
            bias=bias_master, bias_var=None, bias_path=bias_path,
            flats={"R": master_flat.data}, flat_var={}, flat_paths={"R": master_flat.path},
            darks=(), dark_current_map=None, dark_current_path=None, mask=None,
            gain_measurements=gain_measurements,
        )
        sci_meta = _meta(raw_path, (RAW_SIZE, RAW_SIZE), frame_type="science", filter="R")
        product, _status = reduce_frame(
            sci_meta, calib, settings=_settings(apply_gain=True), profile=profile,
            overscan=geom,
        )
        return [float(np.median(product.sci[r, c])) for r, c in tiles]

    per_tile_medians = _reduce_with(per_tile_gain_measurements)
    per_tile_spread = (max(per_tile_medians) - min(per_tile_medians)) / np.mean(per_tile_medians)
    assert per_tile_spread < 0.02, (
        f"per-tile-converted science frame has a seam across tiles: {per_tile_medians}"
    )
    for m in per_tile_medians:
        assert m == pytest.approx(sci_signal_e, rel=0.05)

    # Contrast: falling back to a single frame-level gain (as if
    # tile_measurements had never been stored) leaves a real seam, because
    # the flat was built per-tile but the science frame is not.
    frame_level_medians = _reduce_with(frame_level_only_gain_measurements)
    frame_level_spread = (
        (max(frame_level_medians) - min(frame_level_medians)) / np.mean(frame_level_medians)
    )
    assert frame_level_spread > 0.05, (
        f"expected a visible seam with only a frame-level gain, got {frame_level_medians}"
    )
