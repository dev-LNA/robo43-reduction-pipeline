"""Tests for the per-tile gain pattern (``[instrument.tile_gain_pattern]``).

The per-tile PTC gains of robo43.calib.gain are wrong by ~1 % rms per
measurement; the relative tile gain pattern of a readout mode is stable and is
tabulated in the instrument profile instead. Covers: parsing and validation of
the table, ``K_run * p_t`` gains (``apply_tile_gain_pattern`` and through
``measure_gain``), flat and science conversion on the same gains, the header
cards, the derivation of the pattern from master flats
(``robo43.calib.tilegain`` and ``robo43 tilegain``) on synthetic flats with
injected per-tile factors, and master-flat reuse refusing masters built without
the pattern.

Builds its own synthetic FITS files and inline TOML profiles; does not use
tests/conftest.py.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from robo43.calib.flat import build_master_flats
from robo43.calib.gain import (
    GainMeasurement,
    GainMeasurementSet,
    apply_tile_gain_pattern,
    measure_gain,
    per_tile_header_cards,
    tile_gain_pattern_for,
)
from robo43.calib.overscan import OverscanGeometry
from robo43.calib.pool import (
    NightScanner,
    apply_window_pools,
    find_nearest_master_flat,
    install_master_flat,
)
from robo43.calib.set import CalibrationSet
from robo43.calib.tilegain import (
    analyse_master,
    derive_pattern,
    equal_tile_slices,
    fit_tile_gains,
    format_pattern_toml,
    read_master_tile_gains,
    step_summary,
    tile_grid,
)
from robo43.cli.main import main
from robo43.config import CalibSettings, RuntimeSettings, Settings
from robo43.exceptions import ConfigError
from robo43.instruments.profile import (
    InstrumentCalibOptions,
    InstrumentLayout,
    InstrumentProfile,
    load_profile_file,
    resolve_profile,
)
from robo43.io.headers import read_meta
from robo43.io.meta import FrameMeta
from robo43.reduce.changelog import _STEP_PARAM_KEYS, _step_entry
from robo43.reduce.frame import reduce_frame

# --------------------------------------------------------------------------
# profile table: parsing and validation
# --------------------------------------------------------------------------

_BASE_TOML = """
[instrument]
name = "test-instrument"

[instrument.layout]
n_amplifiers = 4

[instrument.keywords]
gain_setting = ["PG5_2"]
binning = ["PG5_4"]

[instrument.readout_mode_rule]
default = "mode5"

[[instrument.readout_mode_rule.votes]]
keyword = "PG5_1"
map = { "17" = "mode0", "68" = "mode5" }
"""
_PATTERN = (0.98, 1.0, 1.0, 1.02)  # median 1


def _load(tmp_path: Path, table: str, base: str = _BASE_TOML):
    path = tmp_path / "profile.toml"
    path.write_text(base + "\n[instrument.tile_gain_pattern]\n" + table, encoding="utf-8")
    return load_profile_file(str(path))


def test_tile_gain_pattern_table_parses_per_mode(tmp_path):
    profile = _load(tmp_path, "mode5 = [0.98, 1.0, 1.0, 1.02]\nmode0 = [1.01, 0.99, 1.0, 1.0]\n")
    assert profile.tile_gain_pattern == {
        "mode5": (0.98, 1.0, 1.0, 1.02),
        "mode0": (1.01, 0.99, 1.0, 1.0),
    }
    pattern = tile_gain_pattern_for(profile, "mode5")
    assert pattern is not None and pattern.label == "table:mode5"
    assert pattern.values == _PATTERN
    assert tile_gain_pattern_for(profile, "mode9") is None
    assert tile_gain_pattern_for(profile, None) is None


def test_profile_without_the_table_has_no_pattern(tmp_path):
    path = tmp_path / "profile.toml"
    path.write_text(_BASE_TOML, encoding="utf-8")
    profile = load_profile_file(str(path))
    assert profile.tile_gain_pattern == {}
    assert tile_gain_pattern_for(profile, "mode5") is None


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ("mode5 = [0.98, 1.0, 1.02]\n", "3 value"),  # wrong tile count
        ("mode5 = [0.98, 1.0, 1.0, 1.02, 1.0]\n", "5 value"),
        ("mode5 = [0.0, 1.0, 1.0, 1.02]\n", "> 0"),
        ("mode5 = [-0.98, 1.0, 1.0, 1.02]\n", "> 0"),
        ('mode5 = [0.98, 1.0, 1.0, "x"]\n', "numbers"),
        ("mode5 = [0.98, 1.0, 1.0, true]\n", "numbers"),
        ("mode5 = [1.1, 1.1, 1.1, 1.1]\n", "median 1"),  # not normalised
        ("mode5 = 1.0\n", "list"),
        ("mode9 = [0.98, 1.0, 1.0, 1.02]\n", "unknown readout mode"),
    ],
)
def test_tile_gain_pattern_table_validation_raises_config_error(tmp_path, table, message):
    with pytest.raises(ConfigError, match=message):
        _load(tmp_path, table)


def test_tile_gain_pattern_must_be_a_table(tmp_path):
    path = tmp_path / "profile.toml"
    scalar = "tile_gain_pattern = 3\n\n[instrument.layout]"
    path.write_text(_BASE_TOML.replace("[instrument.layout]", scalar), encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a table"):
        load_profile_file(str(path))


def test_packaged_t80s_tables_have_16_normalised_values_for_both_modes():
    profile = resolve_profile(name="t80s")
    assert set(profile.tile_gain_pattern) == {"mode0", "mode5"}
    for mode, values in profile.tile_gain_pattern.items():
        assert len(values) == profile.layout.n_amplifiers == 16, mode
        assert float(np.median(values)) == pytest.approx(1.0, abs=1e-5)
        assert all(0.9 < v < 1.1 for v in values)
    # the readout modes are not interchangeable: 0.46 % rms between the two tables
    ratio = np.log(np.asarray(profile.tile_gain_pattern["mode5"]))
    ratio -= np.log(np.asarray(profile.tile_gain_pattern["mode0"]))
    assert 0.003 < float(np.sqrt(np.mean((ratio - ratio.mean()) ** 2))) < 0.007


def test_other_packaged_profiles_have_no_pattern():
    for name in ("robo43", "generic"):
        assert resolve_profile(name=name).tile_gain_pattern == {}


# --------------------------------------------------------------------------
# K_run * p_t gains
# --------------------------------------------------------------------------

_PTC = (0.93, 0.95, 0.91, 0.94)  # noisy per-tile PTC values of one group


def _profile(tile_gain_pattern=None, n_amplifiers=4, **calib) -> InstrumentProfile:
    return InstrumentProfile(
        name="test-instrument",
        matches={},
        pixel_scale_arcsec=0.5,
        gain_e_per_adu=None,
        read_noise_e=99.0,
        saturation_adu=60000.0,
        nonlinearity_adu=None,
        binning_keyword="XBINNING",
        layout=InstrumentLayout(n_amplifiers=n_amplifiers),
        valid_filters=("R", "G"),
        filter_map={},
        keywords={},
        imagetyp_map={},
        site_lat_deg=None,
        site_lon_deg=None,
        site_elevation_m=None,
        index_scales=None,
        search_radius_deg=0.5,
        calib=InstrumentCalibOptions(**calib),
        tile_gain_pattern=tile_gain_pattern or {},
    )


def _record(gain, *, tile_index=-1, n_tiles=4, mode="mode5", rn=4.0) -> GainMeasurement:
    return GainMeasurement(
        readout_mode=mode, gain_setting=1.0, binning=1, gain_e_per_adu=gain, read_noise_e=rn,
        n_pairs=3, scatter_frac=0.01, tile_index=tile_index, n_tiles=n_tiles,
    )


def _ptc_set(mode="mode5") -> GainMeasurementSet:
    summary = _record(float(np.median(_PTC)), mode=mode)
    tiles = tuple(_record(g, tile_index=i, mode=mode) for i, g in enumerate(_PTC))
    return GainMeasurementSet({(mode, 1.0, 1): summary}, {(mode, 1.0, 1): tiles})


def test_per_tile_gains_are_k_run_times_the_pattern():
    ptc = _ptc_set()
    result = apply_tile_gain_pattern(ptc, _profile({"mode5": _PATTERN}))
    summary = result.get("mode5", 1.0, 1)
    tiles = result.tiles_for("mode5", 1.0, 1)
    k_run = float(np.median(_PTC))
    assert summary.gain_e_per_adu == k_run  # K_run measured exactly as before
    assert summary.gain_pattern == "table:mode5"
    assert [t.gain_e_per_adu for t in tiles] == pytest.approx([k_run * p for p in _PATTERN])
    assert float(np.median([t.gain_e_per_adu for t in tiles])) == pytest.approx(k_run)
    for tile, ptc_gain in zip(tiles, _PTC, strict=True):
        assert tile.ptc_gain_e_per_adu == ptc_gain  # the noisy value is kept for diagnostics
        assert tile.gain_pattern == "table:mode5"
        # the read noise in ADU was measured: in electrons it follows the gain used
        assert tile.read_noise_e == pytest.approx(4.0 * tile.gain_e_per_adu / ptc_gain)
    assert [t.tile_index for t in tiles] == [0, 1, 2, 3]
    # the input set is not modified
    assert ptc.tiles_for("mode5", 1.0, 1)[0].gain_e_per_adu == _PTC[0]
    assert ptc.get("mode5", 1.0, 1).gain_pattern is None


def test_no_pattern_leaves_the_measurements_untouched():
    ptc = _ptc_set()
    assert apply_tile_gain_pattern(ptc, _profile()) is ptc
    assert apply_tile_gain_pattern(ptc, None) is ptc
    other_mode = _profile({"mode0": _PATTERN})
    assert apply_tile_gain_pattern(ptc, other_mode) == ptc


def test_unmeasurable_k_run_means_no_gain_and_no_pattern():
    empty = GainMeasurementSet({})
    assert apply_tile_gain_pattern(empty, _profile({"mode5": _PATTERN})) == empty


def test_pattern_with_the_wrong_tile_count_keeps_the_ptc_gains(caplog):
    ptc = _ptc_set()
    profile = _profile({"mode5": (0.9, 1.0, 1.0, 1.0, 1.1)}, n_amplifiers=5)
    with caplog.at_level(logging.WARNING, logger="robo43.calib.gain"):
        result = apply_tile_gain_pattern(ptc, profile)
    assert result == ptc
    assert any("5 value(s)" in r.getMessage() for r in caplog.records)


def test_a_group_without_tile_records_keeps_its_summary_only():
    ptc = _ptc_set()
    summary_only = GainMeasurementSet(ptc.measurements)
    assert apply_tile_gain_pattern(summary_only, _profile({"mode5": _PATTERN})) == summary_only


def test_apply_tile_gain_pattern_is_idempotent():
    profile = _profile({"mode5": _PATTERN})
    once = apply_tile_gain_pattern(_ptc_set(), profile)
    assert apply_tile_gain_pattern(once, profile) == once


def test_json_of_an_unpatterned_set_is_unchanged_and_a_patterned_set_round_trips(tmp_path):
    ptc = _ptc_set()
    old_keys = {f.name for f in dataclasses.fields(GainMeasurement)} - {
        "ptc_gain_e_per_adu", "gain_pattern",
    }
    assert all(set(record) == old_keys for record in ptc.to_json())
    patterned = apply_tile_gain_pattern(ptc, _profile({"mode5": _PATTERN}))
    assert all("gain_pattern" in record for record in patterned.to_json())
    path = tmp_path / "gain_measurements.json"
    patterned.write(path)
    assert GainMeasurementSet.read(path) == patterned
    ptc.write(path)
    assert GainMeasurementSet.read(path) == ptc


def test_header_cards_record_the_pattern_and_the_gains_actually_used():
    plain = _ptc_set()
    assert "GAINPAT" not in {c[0] for c in plain.get("mode5", 1.0, 1).header_cards()}
    patterned = apply_tile_gain_pattern(plain, _profile({"mode5": _PATTERN}))
    cards = {c[0]: c[1] for c in patterned.get("mode5", 1.0, 1).header_cards()}
    assert cards["GAINPAT"] == "table:mode5"
    assert cards["GAINVAL"] == pytest.approx(float(np.median(_PTC)))
    tile_cards = {c[0]: c[1] for c in per_tile_header_cards(patterned.tiles_for("mode5", 1.0, 1))}
    for i, p in enumerate(_PATTERN):
        assert tile_cards[f"GAIN{i:02d}"] == pytest.approx(float(np.median(_PTC)) * p)


def test_changelog_lists_gainpat_with_the_gain_parameters():
    assert "GAINPAT" in _STEP_PARAM_KEYS["gain"]
    header = fits.Header()
    header["GAINVAL"] = 0.93
    header["GAINPAT"] = "table:mode5"
    entry = _step_entry("gain", applied=True, product_header=header)
    assert entry["params"] == {"GAINVAL": 0.93, "GAINPAT": "table:mode5"}


# --------------------------------------------------------------------------
# measure_gain, master flat and science frame on the same gains
# --------------------------------------------------------------------------

TRUE_TILE_K = [0.7, 0.8, 0.9, 1.0]  # row-major over a 2x2 amplifier mosaic
TRUE_RN_E = 5.0
BIAS_LEVEL_ADU = 300.0
RAW_SIZE = 800
_MID = RAW_SIZE // 2
# the true relative tile gains, median 1: what a derived table would hold
TRUE_PATTERN = tuple(np.asarray(TRUE_TILE_K) / np.median(TRUE_TILE_K))


def _settings(**calib) -> Settings:
    return Settings(calib=CalibSettings(**calib), runtime=RuntimeSettings())


def _mosaic_geometry() -> OverscanGeometry:
    return OverscanGeometry(
        shape=(RAW_SIZE, RAW_SIZE), columns=(_MID - 1, _MID), rows=(_MID - 1, _MID),
        bias_level=BIAS_LEVEL_ADU, flat_level=20000.0, detected=True, source="detected",
    )


def _raw_tile_ranges():
    ranges = [slice(0, _MID - 1), slice(_MID + 1, RAW_SIZE)]
    return [(r, c) for r in ranges for c in ranges]


def _write_mosaic_frame(path, *, rng, signal_e, read_noise_e=TRUE_RN_E) -> None:
    data = np.full((RAW_SIZE, RAW_SIZE), BIAS_LEVEL_ADU, dtype=np.float64)
    for (row_sl, col_sl), k in zip(_raw_tile_ranges(), TRUE_TILE_K, strict=True):
        shape = (row_sl.stop - row_sl.start, col_sl.stop - col_sl.start)
        rn = rng.normal(0.0, read_noise_e, size=shape)
        photons = rng.poisson(signal_e, size=shape).astype(np.float64) if signal_e > 0 else 0.0
        data[row_sl, col_sl] = BIAS_LEVEL_ADU + (photons + rn) / k
    fits.PrimaryHDU(data=data.astype(np.float32)).writeto(path, overwrite=True)


def _meta(path, *, frame_type, filter=None, mode="MULTIAMP") -> FrameMeta:
    return FrameMeta(
        path=path, naxis1=RAW_SIZE, naxis2=RAW_SIZE, frame_type=frame_type, exptime=5.0,
        date_obs=None, filter=filter, ra_deg=None, dec_deg=None, pixel_scale=None,
        ccd_temp=None, gain_setting=10.0, read_noise=None, saturation=None, airmass=None,
        object="DomeFlat" if frame_type == "flat" else None, telescope=None, instrument=None,
        observer=None, binning=1, readout_mode=mode, focal_length_mm=None, site_lat_deg=None,
        site_lon_deg=None, site_elevation_m=None, mjd=None, night=None, is_valid=True,
        profile_name="test-instrument", origins={},
    )


def test_measure_gain_returns_k_run_times_pattern_and_keeps_the_ptc_values(tmp_path):
    rng = np.random.default_rng(20241005)
    geom = _mosaic_geometry()
    metas = []
    for i in range(4):
        path = tmp_path / f"bias_{i}.fits"
        _write_mosaic_frame(path, rng=rng, signal_e=0.0)
        metas.append(_meta(path, frame_type="bias"))
    for filt_idx, signal_e in enumerate([10000.0, 13000.0, 16000.0, 19000.0]):
        for j in range(2):
            path = tmp_path / f"flat_{filt_idx}_{j}.fits"
            _write_mosaic_frame(path, rng=rng, signal_e=signal_e)
            metas.append(_meta(path, frame_type="flat", filter=f"FILT{filt_idx}"))

    plain = measure_gain(metas, settings=_settings(), profile=_profile(), overscan=geom)
    profile = _profile({"MULTIAMP": TRUE_PATTERN})
    patterned = measure_gain(metas, settings=_settings(), profile=profile, overscan=geom)

    k_run = plain.get("MULTIAMP", 10.0, 1).gain_e_per_adu
    assert patterned.get("MULTIAMP", 10.0, 1).gain_e_per_adu == k_run  # measured exactly as today
    tiles = patterned.tiles_for("MULTIAMP", 10.0, 1)
    plain_tiles = plain.tiles_for("MULTIAMP", 10.0, 1)
    rows = zip(tiles, plain_tiles, TRUE_PATTERN, TRUE_TILE_K, strict=True)
    for tile, plain_tile, p, true_k in rows:
        assert tile.gain_e_per_adu == pytest.approx(k_run * p)
        assert tile.ptc_gain_e_per_adu == plain_tile.gain_e_per_adu
        # K_run is ~1 % off the truth for the median tile; the pattern removes the per-tile part
        expected_ratio = k_run / np.median(TRUE_TILE_K)
        assert tile.gain_e_per_adu / true_k == pytest.approx(expected_ratio, rel=1e-9)


def _tile_medians(data, geom):
    return [float(np.median(data[r, c])) for r, c in geom.science_tile_slices]


def _spread(values) -> float:
    return (max(values) - min(values)) / float(np.mean(values))


def test_flat_and_science_use_the_same_pattern_gains_and_stay_seam_free(tmp_path):
    rng = np.random.default_rng(4242)
    geom = _mosaic_geometry()
    profile = _profile({"MULTIAMP": TRUE_PATTERN})
    # noisy PTC values: 2 % high, 1.5 % low, 1 % high, 2 % low relative to the truth
    ptc = [k * f for k, f in zip(TRUE_TILE_K, (1.02, 0.985, 1.01, 0.98), strict=True)]
    summary = GainMeasurement(
        readout_mode="MULTIAMP", gain_setting=10.0, binning=1,
        gain_e_per_adu=float(np.median(ptc)), read_noise_e=TRUE_RN_E, n_pairs=3,
        scatter_frac=0.0, tile_index=-1, n_tiles=4,
    )
    tiles = tuple(
        dataclasses.replace(summary, gain_e_per_adu=g, tile_index=i) for i, g in enumerate(ptc)
    )
    key = ("MULTIAMP", 10.0, 1)
    ptc_set = GainMeasurementSet({key: summary}, {key: tiles})
    pattern_set = apply_tile_gain_pattern(ptc_set, profile)

    flat_metas = []
    for i in range(2):
        path = tmp_path / f"flat_{i}.fits"
        _write_mosaic_frame(path, rng=rng, signal_e=15000.0, read_noise_e=20.0)
        flat_metas.append(_meta(path, frame_type="flat", filter="R"))
    raw_path = tmp_path / "science.fits"
    _write_mosaic_frame(raw_path, rng=rng, signal_e=8000.0, read_noise_e=1.0)
    # apply_overscan already removes the pedestal, so the master bias is a zero-level frame
    bias = np.zeros((RAW_SIZE - 2, RAW_SIZE - 2), dtype=np.float32)
    bias_path = tmp_path / "master_bias.fits"
    fits.PrimaryHDU(data=bias).writeto(bias_path)

    def _run(gain_set, label):
        outdir = tmp_path / label
        master = build_master_flats(
            flat_metas, settings=_settings(), profile=profile, outdir=outdir,
            overscan=geom, gain_measurements=gain_set,
        )["R:dome"]
        calib = CalibrationSet(
            bias=bias, bias_var=None, bias_path=bias_path, flats={"R": master.data}, flat_var={},
            flat_paths={"R": master.path}, darks=(), dark_current_map=None,
            dark_current_path=None, mask=None, gain_measurements=gain_set,
        )
        product, _status = reduce_frame(
            _meta(raw_path, frame_type="science", filter="R"), calib,
            settings=_settings(apply_gain=True), profile=profile, overscan=geom,
        )
        return fits.getheader(outdir / "master_flat_R_dome.fits", 0), master, product

    flat_header, master, product = _run(pattern_set, "pattern")
    k_run = float(np.median(ptc))
    # (i) the same gains: flat and science carry the same GAINPAT and per-tile gains K_run * p_t
    assert flat_header["GAINPAT"] == product.header["GAINPAT"] == "table:MULTIAMP"
    for i, p in enumerate(TRUE_PATTERN):
        assert flat_header[f"GAIN{i:02d}"] == pytest.approx(k_run * p)
        assert product.header[f"GAIN{i:02d}"] == pytest.approx(k_run * p)
    assert flat_header["GAINVAL"] == product.header["GAINVAL"] == pytest.approx(k_run)
    # (ii) with the pattern, flat and flat-fielded science are seam-free to well under the PTC error
    assert _spread(_tile_medians(master.data, geom)) < 0.004
    assert _spread(_tile_medians(product.sci, geom)) < 0.004

    # (iii) without the pattern the PTC errors leave ~1-2 % steps in the e- master flat
    ptc_header, ptc_master, ptc_product = _run(ptc_set, "ptc")
    assert "GAINPAT" not in ptc_header and "GAINPAT" not in ptc_product.header
    assert _spread(_tile_medians(ptc_master.data, geom)) > 0.015
    # (iv) in-run, the flat-fielded science changes only by one global factor (the central level
    # that normalises the flat), never per tile
    ratios = np.asarray(_tile_medians(product.sci, geom)) / np.asarray(
        _tile_medians(ptc_product.sci, geom))
    assert (ratios.max() - ratios.min()) / ratios.mean() < 5e-4


def test_group_without_a_pattern_is_built_exactly_as_before(tmp_path):
    rng = np.random.default_rng(7)
    geom = _mosaic_geometry()
    key = ("MULTIAMP", 10.0, 1)
    summary = GainMeasurement(
        readout_mode="MULTIAMP", gain_setting=10.0, binning=1, gain_e_per_adu=0.85,
        read_noise_e=TRUE_RN_E, n_pairs=3, scatter_frac=0.0, tile_index=-1, n_tiles=4,
    )
    tiles = tuple(dataclasses.replace(summary, gain_e_per_adu=g, tile_index=i)
                  for i, g in enumerate(TRUE_TILE_K))
    gain_set = GainMeasurementSet({key: summary}, {key: tiles})
    metas = []
    for i in range(2):
        path = tmp_path / f"flat_{i}.fits"
        _write_mosaic_frame(path, rng=rng, signal_e=15000.0, read_noise_e=20.0)
        metas.append(_meta(path, frame_type="flat", filter="R"))
    for label, profile in (("a", _profile()), ("b", _profile({"OTHER": TRUE_PATTERN}))):
        build_master_flats(
            metas, settings=_settings(), profile=profile, outdir=tmp_path / label, overscan=geom,
            gain_measurements=apply_tile_gain_pattern(gain_set, profile),
        )
    with (
        fits.open(tmp_path / "a" / "master_flat_R_dome.fits") as a,
        fits.open(tmp_path / "b" / "master_flat_R_dome.fits") as b,
    ):
        assert np.array_equal(a[0].data, b[0].data)
        # (CHECKSUM's comment carries the write time)
        cards_a = [(c.keyword, c.value) for c in a[0].header.cards if c.keyword != "CHECKSUM"]
        cards_b = [(c.keyword, c.value) for c in b[0].header.cards if c.keyword != "CHECKSUM"]
        assert cards_a == cards_b
        assert "GAINPAT" not in a[0].header


# --------------------------------------------------------------------------
# derivation from master flats (synthetic masters, injected per-tile factors)
# --------------------------------------------------------------------------

GRID_COLS, GRID_ROWS, TILE = 4, 2, 200
SHAPE = (GRID_ROWS * TILE, GRID_COLS * TILE)
N_TILES = GRID_COLS * GRID_ROWS
SLICES = equal_tile_slices(SHAPE, GRID_COLS, GRID_ROWS)


def _illumination() -> np.ndarray:
    y, x = np.mgrid[0:SHAPE[0], 0:SHAPE[1]].astype(np.float64)
    u, v = x / SHAPE[1], y / SHAPE[0]
    return 1.0 + 0.10 * u + 0.05 * v + 0.02 * u**2


def _tile_array(per_tile) -> np.ndarray:
    out = np.empty(SHAPE)
    for value, (rows, cols) in zip(per_tile, SLICES, strict=True):
        out[rows, cols] = value
    return out


def _true_gains(seed=1) -> np.ndarray:
    return 1.0 + np.random.default_rng(seed).uniform(-0.03, 0.03, N_TILES)


def _write_master(path, true_gain, used_gain, *, seed=0, noise=1e-5, cards=True, adu=False):
    """A normalised master flat as robo43 writes it: ADU level = illumination / true gain
    (a higher gain reads lower), multiplied tile by tile by ``used_gain`` (the e- conversion),
    normalised by the central median; GAIN00.. cards record ``used_gain``."""
    rng = np.random.default_rng(seed)
    jitter = 1.0 + noise * rng.standard_normal(SHAPE)
    adu_image = _illumination() / _tile_array(true_gain) * jitter
    data = adu_image if adu else adu_image * _tile_array(used_gain)
    centre = data[SHAPE[0] // 4:3 * SHAPE[0] // 4, SHAPE[1] // 4:3 * SHAPE[1] // 4]
    hdu = fits.PrimaryHDU(data=(data / np.median(centre)).astype(np.float32))
    hdu.header["BUNIT"] = "1"
    hdu.header["GAINSET"] = 1.0
    hdu.header["READOUTM"] = "mode5"
    if adu:
        hdu.header["GAINVAL"] = "NONE"
    elif cards:
        hdu.header["GAINVAL"] = float(np.median(used_gain))
        for i, g in enumerate(used_gain):
            hdu.header[f"GAIN{i:02d}"] = float(g)
    hdu.writeto(path, overwrite=True)
    return path


def _expected_pattern(true_gain) -> np.ndarray:
    return true_gain / np.median(true_gain)


def test_tile_grid_and_equal_tile_slices():
    grid = tile_grid(SLICES)
    assert (grid.n_cols, grid.n_rows, grid.n_tiles) == (GRID_COLS, GRID_ROWS, N_TILES)
    assert SLICES[5] == (slice(TILE, 2 * TILE), slice(TILE, 2 * TILE))
    with pytest.raises(ValueError, match="equal tiles"):
        equal_tile_slices((401, 800), 4, 2)
    with pytest.raises(ValueError, match="row-major"):
        tile_grid([SLICES[1], SLICES[0], *SLICES[2:]])
    with pytest.raises(ValueError, match="grid"):
        tile_grid(SLICES[:-1])


def test_derivation_recovers_injected_tile_gains_to_1e_4(tmp_path):
    true_gain = _true_gains()
    used = true_gain * (1.0 + np.random.default_rng(5).normal(0.0, 0.01, N_TILES))  # PTC errors
    master = _write_master(tmp_path / "master_flat_R_dome.fits", true_gain, used)
    result = analyse_master(master, SLICES)
    assert result.gain_source == "header"
    np.testing.assert_allclose(result.pattern, _expected_pattern(true_gain), rtol=1e-4)
    assert float(np.median(result.pattern)) == pytest.approx(1.0)
    # the sign convention: a higher true gain reads a lower ADU level and gets a higher p_t
    hi = int(np.argmax(true_gain))
    assert result.pattern[hi] == result.pattern.max()
    # diagnostics: the boundary steps are pure per-tile factors (residual at the null level)
    assert result.fit_residual_rms_pct < 0.02
    assert result.adu_steps.null_rms_pct < 0.02
    assert result.adu_steps.n_steps == 10  # 6 vertical + 4 horizontal boundaries of 4 x 2 tiles
    expected_err = np.log(used / np.median(used)) - np.log(_expected_pattern(true_gain))
    expected_err -= expected_err.mean()
    assert result.used_gain_over_pattern_rms_pct == pytest.approx(
        100.0 * float(np.sqrt(np.mean(expected_err**2))), abs=0.02)
    # the stored e- master has steps of the size of the gain errors, the ADU one of the true gains
    assert result.e_steps.rms_pct > 0.4
    assert result.adu_steps.rms_pct > result.e_steps.rms_pct


def test_gains_come_from_header_cards_the_json_beside_the_master_or_an_adu_master(tmp_path):
    true_gain = _true_gains()
    used = true_gain * (1.0 + np.random.default_rng(6).normal(0.0, 0.01, N_TILES))
    expected = _expected_pattern(true_gain)

    header_dir = tmp_path / "header"
    header_dir.mkdir()
    from_header = analyse_master(_write_master(header_dir / "m.fits", true_gain, used), SLICES)

    json_dir = tmp_path / "json"
    json_dir.mkdir()
    master = _write_master(json_dir / "m.fits", true_gain, used, cards=False)
    key = ("UNKNOWN", 1.0, 1)  # the readout mode of such records is not the master's READOUTM
    tiles = tuple(_record(g, tile_index=i, n_tiles=N_TILES, mode="UNKNOWN")
                  for i, g in enumerate(used))
    GainMeasurementSet({key: _record(float(np.median(used)), n_tiles=N_TILES, mode="UNKNOWN")},
                       {key: tiles}).write(json_dir / "gain_measurements.json")
    assert read_master_tile_gains(master, N_TILES)[1] == "json"
    from_json = analyse_master(master, SLICES)

    adu_dir = tmp_path / "adu"
    adu_dir.mkdir()
    from_adu = analyse_master(_write_master(adu_dir / "m.fits", true_gain, used, adu=True), SLICES)
    assert from_adu.gain_source == "adu"
    assert from_adu.used_gain_over_pattern_rms_pct is None

    for result in (from_header, from_json, from_adu):
        np.testing.assert_allclose(result.pattern, expected, rtol=1e-4)
    assert from_adu.e_steps.rms_pct == pytest.approx(from_adu.adu_steps.rms_pct)

    bare = _write_master(tmp_path / "bare.fits", true_gain, used, cards=False)
    with pytest.raises(ValueError, match="no per-tile gains"):
        read_master_tile_gains(bare, N_TILES)


def test_derive_pattern_averages_masters_and_reports_the_scatter(tmp_path):
    true_gain = _true_gains()
    analyses = []
    for seed in range(3):
        used = true_gain * (1.0 + np.random.default_rng(10 + seed).normal(0.0, 0.01, N_TILES))
        master = _write_master(tmp_path / f"m{seed}.fits", true_gain, used, seed=seed)
        analyses.append(analyse_master(master, SLICES))
    derivation = derive_pattern(analyses, "mode5")
    np.testing.assert_allclose(derivation.values, _expected_pattern(true_gain), rtol=1e-4)
    assert float(np.median(derivation.values)) == pytest.approx(1.0)
    assert derivation.scatter_pct.shape == (N_TILES,)
    assert float(derivation.scatter_pct.max()) < 0.02  # white noise only: tiles' scatter ~0.005 %
    assert len(derivation.loo_rms_pct) == 3 and max(derivation.loo_rms_pct) < 0.03
    report = derivation.to_dict()
    assert report["n_masters"] == 3 and len(report["masters"]) == 3
    assert json.loads(json.dumps(report))["mode"] == "mode5"
    with pytest.raises(ValueError, match="no masters"):
        derive_pattern([], "mode5")


def test_a_master_built_with_the_pattern_has_no_amplifier_steps_and_matches_another_set(tmp_path):
    true_gain = _true_gains()
    expected = _expected_pattern(true_gain)
    rng = np.random.default_rng(3)
    ptc_a = true_gain * (1.0 + rng.normal(0.0, 0.01, N_TILES))
    ptc_b = true_gain * (1.0 + rng.normal(0.0, 0.01, N_TILES))
    a = _write_master(tmp_path / "a.fits", true_gain, ptc_a, seed=1)
    b = _write_master(tmp_path / "b.fits", true_gain, ptc_b, seed=2)
    # today: two gain sets leave ~1 % rms steps between the e- masters, and in each master
    before = step_summary(a, SLICES, versus=b)
    assert before.rms_pct > 0.5 and step_summary(a, SLICES).rms_pct > 0.5
    # with the pattern: gains K_run * p_t of each run (K_run differs, as nights' K differ)
    new_a = _write_master(tmp_path / "new_a.fits", true_gain, 0.93 * expected, seed=1)
    new_b = _write_master(tmp_path / "new_b.fits", true_gain, 0.95 * expected, seed=2)
    assert step_summary(new_a, SLICES).rms_pct < 0.02
    after = step_summary(new_a, SLICES, versus=new_b)
    assert after.rms_pct < 0.02 and after.max_abs_pct < 0.05
    assert after.null_rms_pct < 0.02


def test_boundary_estimator_refuses_tiles_too_small_for_the_strips_and_margin(tmp_path):
    true_gain = _true_gains()
    master = _write_master(tmp_path / "m.fits", true_gain, true_gain)
    with pytest.raises(ValueError, match=r"too (short|narrow)"):
        analyse_master(master, SLICES, margin=100)  # 200-px tiles cannot spare 2 x 100 px
    with pytest.raises(ValueError, match="no room"):
        analyse_master(master, SLICES, strip=110)   # two 110-px strips do not fit in a 200-px tile
    zeros = tmp_path / "zeros.fits"
    fits.PrimaryHDU(data=np.zeros(SHAPE, dtype=np.float32)).writeto(zeros)
    with pytest.raises(ValueError, match="non-positive"):
        step_summary(zeros, SLICES)


def test_fit_tile_gains_rejects_disconnected_tiles():
    from robo43.calib.tilegain import BoundaryStep

    steps = [BoundaryStep(0, 1, "col", 0.01), BoundaryStep(2, 3, "col", -0.02)]
    with pytest.raises(ValueError, match="connect"):
        fit_tile_gains(steps, 4)
    pattern, residuals = fit_tile_gains([BoundaryStep(0, 1, "col", 0.01),
                                         BoundaryStep(1, 2, "col", 0.01)], 3)
    assert float(np.median(pattern)) == pytest.approx(1.0)
    assert pattern[0] > pattern[1] > pattern[2]  # ADU rises 0 -> 2, so the gain falls
    assert float(np.abs(residuals).max()) < 1e-12


def test_formatted_table_loads_back_into_a_profile(tmp_path):
    true_gain = _true_gains()
    master = _write_master(tmp_path / "m.fits", true_gain, true_gain)
    derivation = derive_pattern([analyse_master(master, SLICES)], "mode5")
    toml = format_pattern_toml(derivation)
    base = _BASE_TOML.replace("n_amplifiers = 4", f"n_amplifiers = {N_TILES}")
    profile = _load(tmp_path, toml, base=base)  # 6 decimals still satisfy the median-1 check
    np.testing.assert_allclose(profile.tile_gain_pattern["mode5"], derivation.values, atol=1e-6)


def test_tilegain_cli_derives_and_checks_steps(tmp_path, monkeypatch, capsys):
    profile = _profile(n_amplifiers=N_TILES)
    monkeypatch.setattr("robo43.cli.tilegain.resolve_profile", lambda **_kw: profile)
    true_gain = _true_gains()
    used = true_gain * (1.0 + np.random.default_rng(8).normal(0.0, 0.01, N_TILES))
    master = _write_master(tmp_path / "m.fits", true_gain, used)
    toml_out, report = tmp_path / "pattern.toml", tmp_path / "report.json"
    code = main([
        "tilegain", "derive", str(master), "--instrument", "t80s", "--mode", "mode5",
        "--grid", f"{GRID_COLS}x{GRID_ROWS}", "--toml-out", str(toml_out),
        "--report-json", str(report),
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "[instrument.tile_gain_pattern]" in out and "fit residual rms" in out
    values = json.loads(report.read_text())["values"]
    np.testing.assert_allclose(values, _expected_pattern(true_gain), rtol=1e-4)
    assert toml_out.read_text().startswith("mode5 = [")

    other = _write_master(tmp_path / "o.fits", true_gain, true_gain, seed=4)
    code = main(["tilegain", "steps", str(master), "--versus", str(other), "--instrument",
                 "t80s", "--grid", f"{GRID_COLS}x{GRID_ROWS}"])
    assert code == 0 and "rms" in capsys.readouterr().out
    # a grid whose tile count differs from the profile's amplifier count is refused
    assert main(["tilegain", "steps", str(master), "--instrument", "t80s", "--grid", "8x2"]) == 2
    # no --instrument: refused
    assert main(["tilegain", "steps", str(master), "--grid", "4x2"]) == 2


# --------------------------------------------------------------------------
# master-flat reuse needs the same pattern
# --------------------------------------------------------------------------


def _reuse_master(root: Path, night: str, *, pattern: str | None, readout="mode5",
                  n_frames=25) -> Path:
    calib = root / night / "calib"
    calib.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(data=np.ones((64, 64), dtype=np.float32))
    hdu.header["BUNIT"] = "1"
    hdu.header["GAINSET"] = 1.0
    hdu.header["READOUTM"] = readout
    hdu.header["FLATNACC"] = n_frames
    if pattern is not None:
        hdu.header["GAINPAT"] = pattern
    path = calib / "master_flat_R_dome.fits"
    hdu.writeto(path)
    return path


def _nearest(root, pattern, night=date(2024, 1, 20)):
    return find_nearest_master_flat(
        [root], night, filter_name="R", gain_setting=1.0, readout_mode="mode5",
        min_frames=20, gain_pattern=pattern,
    )[0]


def test_reuse_refuses_masters_built_without_the_pattern_and_logs_at_debug(tmp_path, caplog):
    red = tmp_path / "reduced"
    _reuse_master(red, "20240119", pattern=None)             # nearest, but pattern-less
    _reuse_master(red, "20240115", pattern="table:mode0")    # another mode's pattern
    with caplog.at_level(logging.DEBUG, logger="robo43.calib.pool"):
        assert _nearest(red, "table:mode5") is None
    skipped = [r for r in caplog.records if "GAINPAT" in r.getMessage()]
    assert len(skipped) == 2 and all(r.levelno == logging.DEBUG for r in skipped)
    _reuse_master(red, "20240110", pattern="table:mode5")    # farther, but built with it
    choice = _nearest(red, "table:mode5")
    assert choice.night == date(2024, 1, 10) and choice.distance_days == -10


def test_reuse_without_a_profile_pattern_is_unchanged(tmp_path):
    red = tmp_path / "reduced"
    _reuse_master(red, "20240119", pattern=None)
    _reuse_master(red, "20240110", pattern="table:mode5")
    assert _nearest(red, None).night == date(2024, 1, 19)


def test_installed_reused_master_keeps_its_gainpat_card(tmp_path):
    red = tmp_path / "reduced"
    src = _reuse_master(red, "20240112", pattern="table:mode5")
    choice = _nearest(red, "table:mode5", night=date(2024, 1, 20))
    dest = install_master_flat(choice, tmp_path / "calib", current_night=date(2024, 1, 20),
                               window_nights=30, target_frames=20)
    assert src.exists() and fits.getheader(dest, 0)["GAINPAT"] == "table:mode5"


def _write_raw(path: Path, *, frame_type: str, level: float = 1000.0) -> Path:
    hdu = fits.PrimaryHDU(data=np.full((64, 64), level, dtype=np.float32))
    for key, value in (("IMAGETYP", frame_type.upper()), ("EXPTIME", 10.0), ("FILTER", "R"),
                       ("DATE-OBS", "2024-01-01T00:00:00"), ("GAIN", 1.0), ("XBINNING", 2),
                       ("READOUTM", "mode5"), ("RDNOISE", 5.0), ("SATURATE", 60000.0)):
        hdu.header[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(path, overwrite=True)
    return path


def test_window_pooling_passes_the_profile_pattern_to_the_master_search(tmp_path):
    raw, red = tmp_path / "raw", tmp_path / "reduced"
    night = raw / "20240120"
    sci = _write_raw(night / "sci.fits", frame_type="science")
    flats = [
        _write_raw(night / f"flat_{i}.fits", frame_type="flat", level=20000.0) for i in range(3)
    ]
    _reuse_master(red, "20240119", pattern=None)             # nearest master: no pattern
    _reuse_master(red, "20240101", pattern="table:mode5")
    settings = _settings()
    calib = {
        "flat_pool": True, "flat_pool_window_nights": 30, "flat_pool_target_frames": 20,
        "flat_master_roots": (str(red),),
    }
    metas = [read_meta(p, _profile(**calib), settings=settings) for p in (sci, *flats)]
    assert metas[0].readout_mode == "mode5"

    def _choice(profile):
        res = apply_window_pools(
            metas, profile=profile, settings=settings, night_dir=night,
            current_night=date(2024, 1, 20), scanner=NightScanner(profile, settings),
        )
        return res.flats["R"].reuse

    assert _choice(_profile(**calib)).night == date(2024, 1, 19)           # no table: unchanged
    with_table = _profile({"mode5": _PATTERN}, **calib)
    assert _choice(with_table).night == date(2024, 1, 1)                  # the pattern master
    other = _profile({"mode0": _PATTERN}, **calib)
    assert _choice(other).night == date(2024, 1, 19)                       # no mode5 pattern


# --------------------------------------------------------------------------
# robo43 reduce end to end
# --------------------------------------------------------------------------


def _reduce_inputs(raw: Path) -> list[Path]:
    """Science, 4 bias and 4 flat frames of readout mode ``mode5`` (Poisson flats, so that
    the photon-transfer gain is measurable)."""
    night = raw / "20240120"
    rng = np.random.default_rng(11)
    paths = []
    for name, frame_type, level, sigma in (
        ("sci", "science", 1000.0, 3.0), *[(f"bias_{i}", "bias", 1000.0, 3.0) for i in range(4)],
    ):
        data = (level + rng.normal(0.0, sigma, (64, 64))).astype(np.float32)
        paths.append(_write_array(night / f"{name}.fits", data, frame_type))
    for i in range(4):
        data = rng.poisson(20000.0 + 100.0 * i, (64, 64)).astype(np.float32)
        paths.append(_write_array(night / f"flat_{i}.fits", data, "flat"))
    return paths


def _write_array(path: Path, data: np.ndarray, frame_type: str) -> Path:
    hdu = fits.PrimaryHDU(data=data)
    for key, value in (("IMAGETYP", frame_type.upper()), ("EXPTIME", 10.0), ("FILTER", "R"),
                       ("DATE-OBS", "2024-01-01T00:00:00"), ("GAIN", 1.0), ("XBINNING", 1),
                       ("READOUTM", "mode5"), ("RDNOISE", 5.0), ("SATURATE", 60000.0)):
        hdu.header[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(path, overwrite=True)
    return path


def _reduce(tmp_path, monkeypatch, profile, label):
    monkeypatch.setattr("robo43.cli.reduce.resolve_profile", lambda **_kw: profile)
    calib, outdir = tmp_path / label / "calib", tmp_path / label / "output"
    inputs = _reduce_inputs(tmp_path / label / "raw")
    code = main(["reduce", *map(str, inputs), "--calibdir", str(calib), "--outdir", str(outdir),
                 "--no-overscan", "--loglevel", "WARNING"])
    return code, calib, outdir


def test_reduce_applies_the_pattern_and_records_gainpat_on_every_product(tmp_path, monkeypatch):
    code, calib, outdir = _reduce(tmp_path, monkeypatch, _profile(n_amplifiers=1), "plain")
    assert code == 0
    plain = json.loads((calib / "gain_measurements.json").read_text())
    assert all("gain_pattern" not in r and "ptc_gain_e_per_adu" not in r for r in plain)
    assert "GAINPAT" not in fits.getheader(calib / "master_flat_R_dome.fits", 0)
    assert "GAINPAT" not in fits.getheader(outdir / "sci_proc.fits", 0)

    # a one-tile profile with a one-value table: gains K_run * 1.0, same as the PTC value, but
    # recorded as table gains everywhere
    profile = _profile({"mode5": (1.0,)}, n_amplifiers=1)
    code, calib, outdir = _reduce(tmp_path, monkeypatch, profile, "pattern")
    assert code == 0
    records = json.loads((calib / "gain_measurements.json").read_text())
    assert records and all(r["gain_pattern"] == "table:mode5" for r in records)
    tile = next(r for r in records if r["tile_index"] == 0)
    assert tile["ptc_gain_e_per_adu"] == pytest.approx(tile["gain_e_per_adu"])
    assert fits.getheader(calib / "master_flat_R_dome.fits", 0)["GAINPAT"] == "table:mode5"
    assert fits.getheader(calib / "master_bias.fits", 0)["GAINPAT"] == "table:mode5"
    product = fits.getheader(outdir / "sci_proc.fits", 0)
    assert product["GAINPAT"] == "table:mode5" and product["GAINVAL"] > 0
    log = [json.loads(line) for line in (outdir / "change_log.jsonl").read_text().splitlines()]
    gain_step = next(s for s in log[0]["steps"] if s["step"] == "gain")
    assert gain_step["params"]["GAINPAT"] == "table:mode5"
