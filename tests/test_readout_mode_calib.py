"""Tests that calibration frames and masters of different camera readout modes
("mode0" / "mode5" on T80S) are never combined or reused for one another:
master-flat reuse, the master-bias inputs (pooled, explicit and borrowed
frames), the backward bias fallback, the gain-measurement groups and the
READOUTM card of the built master bias.

Builds its own synthetic FITS files under tmp_path in a root/YYYYMMDD/ layout;
does not use fixtures or conftest.py, matching tests/test_calib_pool.py.
"""

from __future__ import annotations

import contextlib
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from robo43.calib.bias import build_master_bias
from robo43.calib.fallback import camera_config_key, collect_fallback_frames
from robo43.calib.gain import measure_gain
from robo43.calib.pool import (
    NightScanner,
    apply_window_pools,
    enforce_single_bias_mode,
    find_nearest_master_flat,
    pool_bias_frames,
    restrict_bias_to_mode,
)
from robo43.cli.main import main
from robo43.config import CalibSettings, RuntimeSettings, Settings
from robo43.exceptions import CalibrationError, ConfigError
from robo43.instruments.profile import (
    InstrumentCalibOptions,
    InstrumentLayout,
    InstrumentProfile,
)
from robo43.io.headers import read_meta
from robo43.logging_setup import ROOT_LOGGER_NAME


@pytest.fixture(autouse=True)
def _reset_logging():
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    handlers = list(logger.handlers)
    level = logger.level
    propagate = logger.propagate
    yield
    for handler in logger.handlers:
        if handler not in handlers:
            handler.close()
    logger.handlers = handlers
    logger.setLevel(level)
    logger.propagate = propagate


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self, level: int) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno == level]


@contextlib.contextmanager
def _capture(name: str):
    """Capture the log records of logger ``name``, independent of logging setup."""
    logger = logging.getLogger(name)
    handler = _Records()
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def _profile(**calib) -> InstrumentProfile:
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
        valid_filters=("R", "G", "B"),
        filter_map={},
        keywords={},
        imagetyp_map={},
        site_lat_deg=None,
        site_lon_deg=None,
        site_elevation_m=None,
        index_scales=None,
        search_radius_deg=0.5,
        calib=InstrumentCalibOptions(**calib),
    )


def _settings(**calib) -> Settings:
    return Settings(
        paths=None, astrometry=None, photometry=None,
        calib=CalibSettings(**calib),
        coadd=None, diagnostics=None, db=None, service=None,
        runtime=RuntimeSettings(), targets=None,
    )


def _write(path: Path, *, frame_type: str, mode: str | None, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    level = 20000.0 if frame_type == "flat" else 1000.0
    data = (np.ones((64, 64)) * level + rng.normal(0.0, 3.0, (64, 64))).astype(np.float32)
    hdu = fits.PrimaryHDU(data=data)
    hdu.header["IMAGETYP"] = frame_type.upper()
    hdu.header["EXPTIME"] = 10.0
    hdu.header["FILTER"] = "R"
    hdu.header["DATE-OBS"] = "2024-01-01T00:00:00"
    hdu.header["GAIN"] = 1.0
    hdu.header["XBINNING"] = 1
    if mode is not None:
        hdu.header["READOUTM"] = mode
    hdu.header["RDNOISE"] = 5.0
    hdu.header["SATURATE"] = 60000.0
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(path, overwrite=True)
    return path


def _night(root: Path, night: str, *, mode: str | None, bias: int = 0, tag: str = "") -> Path:
    d = root / night
    d.mkdir(parents=True, exist_ok=True)
    for i in range(bias):
        _write(d / f"bias_{tag}{i:02d}.fits", frame_type="bias", mode=mode, seed=i)
    return d


def _metas(paths, profile=None, settings=None):
    profile = profile or _profile()
    settings = settings or _settings()
    return [read_meta(p, profile, settings=settings) for p in paths]


def _bias_modes(metas) -> list[str | None]:
    return sorted(m.readout_mode or "UNKNOWN" for m in metas if m.frame_type == "bias")


def _write_master(root: Path, night: str, *, readout: str | None, filter_key="R_dome") -> Path:
    calib = root / night / "calib"
    calib.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(data=np.ones((64, 64), dtype=np.float32))
    hdu.header["GAINSET"] = 1.0
    if readout is not None:
        hdu.header["READOUTM"] = readout
    hdu.header["FLATNACC"] = 25
    path = calib / f"master_flat_{filter_key}.fits"
    hdu.writeto(path)
    return path


# --- master flat reuse -----------------------------------------------------


def _nearest(root, mode, night=date(2024, 1, 20)):
    return find_nearest_master_flat(
        [root], night, filter_name="R", gain_setting=1.0, readout_mode=mode, min_frames=20,
    )[0]


def test_master_flat_reuse_refuses_a_mode_mismatched_master(tmp_path):
    red = tmp_path / "reduced"
    _write_master(red, "20240119", readout="mode0")
    assert _nearest(red, "mode5") is None
    assert _nearest(red, "mode0").night == date(2024, 1, 19)


def test_master_flat_reuse_refuses_a_mode_less_master_for_a_derived_mode(tmp_path):
    red = tmp_path / "reduced"
    _write_master(red, "20240119", readout="UNKNOWN")      # built before the mode existed
    _write_master(red, "20240118", readout=None)            # no READOUTM card at all
    assert _nearest(red, "mode5") is None
    assert _nearest(red, "mode0") is None
    # ... but it still serves a frame that has no readout mode (unchanged).
    assert _nearest(red, None).night == date(2024, 1, 19)


def test_master_flat_reuse_skips_to_a_farther_master_of_the_right_mode(tmp_path):
    red = tmp_path / "reduced"
    _write_master(red, "20240119", readout="mode0")
    _write_master(red, "20240118", readout="UNKNOWN")
    far = _write_master(red, "20231201", readout="mode5")
    choice = _nearest(red, "mode5")
    assert choice.path == far


def test_master_flat_reuse_logs_each_skipped_master_at_debug(tmp_path):
    red = tmp_path / "reduced"
    _write_master(red, "20240119", readout="mode0")
    _write_master(red, "20240118", readout=None)
    with _capture("robo43.calib.pool") as cap:
        assert _nearest(red, "mode5") is None
    debug = cap.messages(logging.DEBUG)
    assert sum("READOUTM" in m for m in debug) == 2
    assert any("'mode0'" in m and "'mode5'" in m for m in debug)
    assert any("None" in m for m in debug)
    assert cap.messages(logging.WARNING) == []


# --- bias pooling and the backward fallback never mix modes ---------------


def test_bias_pool_never_mixes_modes(tmp_path):
    raw = tmp_path / "raw"
    _night(raw, "20240120", mode="mode5", bias=3, tag="a")
    _night(raw, "20240120", mode="mode0", bias=4, tag="b")      # same night, other mode
    _night(raw, "20240118", mode="mode0", bias=5)
    _night(raw, "20240122", mode="mode5", bias=2)
    profile, settings = _profile(), _settings()
    target = ("mode5", 1.0, 1, 64, 64)
    sel = pool_bias_frames(
        raw / "20240120", date(2024, 1, 20),
        target_config=target, window_nights=10, target_frames=100,
        explicit_metas=[], scanner=NightScanner(profile, settings),
        profile=profile, settings=settings,
    )
    assert len(sel.metas) == 5
    assert {m.readout_mode for m in sel.metas} == {"mode5"}
    assert {camera_config_key(m) for m in sel.metas} == {target}


def test_backward_fallback_never_picks_another_mode(tmp_path):
    raw = tmp_path / "raw"
    (raw / "20240120").mkdir(parents=True)
    _night(raw, "20240115", mode="mode0", bias=30)       # nearer, wrong mode
    _night(raw, "20240110", mode="mode5", bias=25)
    profile, settings = _profile(), _settings()
    result = collect_fallback_frames(
        raw / "20240120", date(2024, 1, 20),
        frame_type="bias", target_config=("mode5", 1.0, 1, 64, 64),
        min_count=20, max_nights=30, profile=profile, settings=settings,
    )
    assert result.satisfied
    assert result.source_nights == ("20240110",)
    assert {m.readout_mode for m in result.metas} == {"mode5"}
    only_other = collect_fallback_frames(
        raw / "20240120", date(2024, 1, 20),
        frame_type="bias", target_config=("mode0", 1.0, 1, 64, 64),
        min_count=40, max_nights=30, profile=profile, settings=settings,
    )
    assert not only_other.satisfied
    assert {m.readout_mode for m in only_other.metas} == {"mode0"}


def test_pool_beyond_window_fallback_never_picks_another_mode(tmp_path):
    raw = tmp_path / "raw"
    (raw / "20240120").mkdir(parents=True)
    _night(raw, "20240105", mode="mode0", bias=30)       # beyond the +/-10 d window
    _night(raw, "20231225", mode="mode5", bias=25)
    profile, settings = _profile(), _settings(min_bias_frames=20, fallback_max_nights_bias=60)
    sel = pool_bias_frames(
        raw / "20240120", date(2024, 1, 20),
        target_config=("mode5", 1.0, 1, 64, 64), window_nights=10, target_frames=20,
        explicit_metas=[], scanner=NightScanner(profile, settings),
        profile=profile, settings=settings,
    )
    assert sel.beyond_window
    assert sel.nights == ("20231225",)
    assert {m.readout_mode for m in sel.metas} == {"mode5"}


# --- explicit bias inputs ---------------------------------------------------


def _science_and_mixed_bias(raw: Path):
    d = raw / "20240120"
    sci = _write(d / "sci.fits", frame_type="science", mode="mode5")
    paths = [sci]
    paths += [_write(d / f"bias_a{i}.fits", frame_type="bias", mode="mode5", seed=i)
              for i in range(4)]
    paths += [_write(d / f"bias_b{i}.fits", frame_type="bias", mode="mode0", seed=i)
              for i in range(3)]
    paths += [_write(d / f"bias_c{i}.fits", frame_type="bias", mode=None, seed=i)
              for i in range(2)]
    return paths


def test_mixed_mode_explicit_bias_keeps_the_science_mode_and_warns(tmp_path):
    metas = _metas(_science_and_mixed_bias(tmp_path / "raw"))
    with _capture("robo43.calib.pool") as cap:
        kept = enforce_single_bias_mode(metas)
    assert _bias_modes(kept) == ["mode5"] * 4
    assert [m for m in kept if m.frame_type == "science"] == [metas[0]]
    warnings_ = cap.messages(logging.WARNING)
    assert len(warnings_) == 1
    assert "mode0: 3" in warnings_[0] and "UNKNOWN: 2" in warnings_[0]
    assert "'mode5'" in warnings_[0]


def test_single_mode_bias_is_returned_unchanged_without_a_warning(tmp_path):
    d = tmp_path / "raw" / "20240120"
    paths = [_write(d / "sci.fits", frame_type="science", mode="mode5")]
    paths += [_write(d / f"bias_{i}.fits", frame_type="bias", mode="mode0", seed=i)
              for i in range(3)]
    metas = _metas(paths)
    with _capture("robo43.calib.pool") as cap:
        assert enforce_single_bias_mode(metas) == metas
    assert cap.messages(logging.WARNING) == []
    assert enforce_single_bias_mode([]) == []


def test_science_with_no_bias_of_its_mode_drops_all_the_bias_frames(tmp_path):
    d = tmp_path / "raw" / "20240120"
    paths = [_write(d / "sci.fits", frame_type="science", mode="mode5")]
    paths += [_write(d / f"bias_a{i}.fits", frame_type="bias", mode="mode0", seed=i)
              for i in range(2)]
    paths += [_write(d / f"bias_b{i}.fits", frame_type="bias", mode="mode1", seed=i)
              for i in range(2)]
    kept = enforce_single_bias_mode(_metas(paths))
    assert _bias_modes(kept) == []


def test_calibration_only_mixed_mode_bias_is_refused(tmp_path):
    d = tmp_path / "raw" / "20240120"
    paths = [_write(d / f"bias_a{i}.fits", frame_type="bias", mode="mode5", seed=i)
             for i in range(3)]
    paths += [_write(d / f"bias_b{i}.fits", frame_type="bias", mode="mode0", seed=i)
              for i in range(2)]
    with pytest.raises(ConfigError, match=r"mode0: 2, mode5: 3"):
        enforce_single_bias_mode(_metas(paths))


def test_calibration_only_single_mode_bias_is_accepted(tmp_path):
    d = tmp_path / "raw" / "20240120"
    paths = [_write(d / f"bias_{i}.fits", frame_type="bias", mode="mode0", seed=i)
             for i in range(3)]
    metas = _metas(paths)
    assert enforce_single_bias_mode(metas) == metas


def test_restrict_bias_to_mode_keeps_other_frame_types(tmp_path):
    metas = _metas(_science_and_mixed_bias(tmp_path / "raw"))
    kept = restrict_bias_to_mode(metas, "mode0", where="test")
    assert [m.frame_type for m in kept].count("science") == 1
    assert _bias_modes(kept) == ["mode0"] * 3
    assert restrict_bias_to_mode(metas, None, where="test") != kept


def test_apply_window_pools_drops_other_mode_explicit_bias_before_pooling(tmp_path):
    raw = tmp_path / "raw"
    paths = _science_and_mixed_bias(raw)
    _night(raw, "20240118", mode="mode5", bias=3)
    _night(raw, "20240119", mode="mode0", bias=6)
    profile = _profile(bias_pool=True, bias_pool_window_nights=10, bias_pool_target_frames=100)
    with _capture("robo43.calib.pool") as cap:
        res = apply_window_pools(
            _metas(paths, profile), profile=profile, settings=_settings(),
            night_dir=raw / "20240120", current_night=date(2024, 1, 20),
        )
    assert res.bias is not None
    assert _bias_modes(res.metas) == ["mode5"] * 7        # 4 explicit + 3 pooled
    assert {m.readout_mode for m in res.bias.metas} == {"mode5"}
    warnings_ = [m for m in cap.messages(logging.WARNING) if "dropped" in m]
    assert len(warnings_) == 1
    assert "mode0: 3" in warnings_[0] and "UNKNOWN: 2" in warnings_[0]


# --- master bias builder and its READOUTM card ----------------------------


def test_build_master_bias_refuses_mixed_modes(tmp_path):
    d = tmp_path / "raw" / "20240120"
    paths = [_write(d / f"bias_a{i}.fits", frame_type="bias", mode="mode5", seed=i)
             for i in range(3)]
    paths += [_write(d / f"bias_b{i}.fits", frame_type="bias", mode="mode0", seed=i)
              for i in range(3)]
    with pytest.raises(CalibrationError, match="different readout modes"):
        build_master_bias(
            _metas(paths), settings=_settings(), profile=_profile(), outdir=tmp_path / "calib"
        )
    assert not (tmp_path / "calib" / "master_bias.fits").exists()


@pytest.mark.parametrize("mode", ["mode0", "mode5"])
def test_master_bias_records_the_mode_it_was_built_from(tmp_path, mode):
    d = tmp_path / "raw" / "20240120"
    paths = [_write(d / f"bias_{i}.fits", frame_type="bias", mode=mode, seed=i)
             for i in range(4)]
    master = build_master_bias(
        _metas(paths), settings=_settings(), profile=_profile(), outdir=tmp_path / "calib"
    )
    cards = {c[0]: c[1] for c in master.header_cards}
    assert cards["READOUTM"] == mode
    with fits.open(tmp_path / "calib" / "master_bias.fits") as hdul:
        assert hdul[0].header["READOUTM"] == mode


# --- gain measurement groups ------------------------------------------------


def test_gain_measurement_groups_bias_frames_per_mode(tmp_path, caplog):
    d = tmp_path / "raw" / "20240120"
    paths = [_write(d / f"bias_a{i}.fits", frame_type="bias", mode="mode5", seed=i)
             for i in range(3)]
    paths += [_write(d / f"bias_b{i}.fits", frame_type="bias", mode="mode0", seed=i)
              for i in range(2)]
    with caplog.at_level(logging.INFO, logger="robo43.calib.gain"):
        measure_gain(_metas(paths), settings=_settings(), profile=_profile())
    groups = [r.getMessage() for r in caplog.records if "gain group" in r.getMessage()]
    assert any("readout=mode0" in m and "2 bias(es)" in m for m in groups)
    assert any("readout=mode5" in m and "3 bias(es)" in m for m in groups)
    assert len(groups) == 2


# --- robo43 reduce end to end ----------------------------------------------


def _run_reduce(tmp_path, monkeypatch, profile, inputs):
    monkeypatch.setattr("robo43.cli.reduce.resolve_profile", lambda **_kw: profile)
    calib_dir, outdir = tmp_path / "calib", tmp_path / "output"
    code = main([
        "reduce", *[str(i) for i in inputs],
        "--calibdir", str(calib_dir), "--outdir", str(outdir),
        "--no-overscan", "--loglevel", "INFO",
    ])
    return code, calib_dir, outdir


def test_reduce_with_mixed_mode_explicit_bias_builds_the_science_mode_master(
    tmp_path, monkeypatch
):
    paths = _science_and_mixed_bias(tmp_path / "raw")
    code, calib, outdir = _run_reduce(tmp_path, monkeypatch, _profile(), paths)
    assert code == 0
    with fits.open(calib / "master_bias.fits") as hdul:
        assert hdul[0].header["NCOMBINE"] == 4
        assert hdul[0].header["READOUTM"] == "mode5"
    assert (outdir / "sci_proc.fits").exists()


def test_reduce_calibration_only_mixed_mode_bias_fails_without_a_master(
    tmp_path, monkeypatch
):
    paths = [p for p in _science_and_mixed_bias(tmp_path / "raw") if "sci" not in p.name]
    code, calib, _outdir = _run_reduce(tmp_path, monkeypatch, _profile(), paths)
    assert code == 2
    assert not (calib / "master_bias.fits").exists()
