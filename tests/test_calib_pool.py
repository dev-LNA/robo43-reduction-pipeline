"""Tests for robo43.calib.pool: symmetric-window bias/flat pooling, nearest
existing master-flat reuse, and their wiring into ``robo43 reduce`` via the
``[instrument.calib]`` keys ``bias_pool``, ``flat_pool_window_nights`` and
``flat_master_roots``.

Builds its own synthetic FITS files under tmp_path in a root/YYYYMMDD/ layout;
does not use fixtures or conftest.py, matching tests/test_flat_pool.py.
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
from robo43.calib.pool import (
    MasterFlatChoice,
    NightScanner,
    apply_window_pools,
    find_nearest_master_flat,
    infer_science_night,
    install_master_flat,
    limit_bias_for_gain,
    pool_bias_frames,
    pool_flat_frames_window,
    window_night_dirs,
)
from robo43.cli.main import main
from robo43.config import CalibSettings, RuntimeSettings, Settings
from robo43.instruments.profile import (
    InstrumentCalibOptions,
    InstrumentLayout,
    InstrumentProfile,
)
from robo43.io.headers import read_meta
from robo43.logging_setup import ROOT_LOGGER_NAME

READOUT = "High Gain Mode 16BIT"
TARGET_CONFIG = (READOUT, 10.0, 2, 64, 64)


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
def _capture():
    """Capture robo43.calib.pool log records, independent of logging setup."""
    logger = logging.getLogger("robo43.calib.pool")
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


def _write(
    path: Path,
    *,
    frame_type: str,
    filter_name: str = "R",
    gain: float = 10.0,
    binning: int = 2,
    level: float = 1000.0,
    valid: bool = True,
    noise: float = 0.0,
    seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    data = (np.ones((64, 64)) * level + rng.normal(0.0, noise, (64, 64))).astype(np.float32)
    hdu = fits.PrimaryHDU(data=data)
    hdu.header["IMAGETYP"] = frame_type.upper()
    hdu.header["EXPTIME"] = 10.0
    hdu.header["FILTER"] = filter_name
    hdu.header["VALID"] = valid
    hdu.header["DATE-OBS"] = "2024-01-01T00:00:00"
    hdu.header["GAIN"] = gain
    hdu.header["XBINNING"] = binning
    hdu.header["READOUTM"] = READOUT
    hdu.header["RDNOISE"] = 5.0
    hdu.header["SATURATE"] = 60000.0
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(path, overwrite=True)
    return path


def _night(root: Path, night: str, *, bias: int = 0, flats: int = 0, **kw) -> Path:
    """Create root/night with ``bias`` bias and ``flats`` R-flat frames."""
    d = root / night
    d.mkdir(parents=True, exist_ok=True)
    for i in range(bias):
        _write(d / f"bias_{i:02d}.fits", frame_type="bias", **kw)
    for i in range(flats):
        _write(d / f"flat_r_{i:02d}.fits", frame_type="flat", level=20000.0, **kw)
    return d


def _bias_pool(root: Path, night: str, *, window=10, target=100, explicit=(), scanner=None,
               settings=None, profile=None):
    profile = profile or _profile()
    settings = settings or _settings()
    scanner = scanner or NightScanner(profile, settings)
    y, m, d = int(night[:4]), int(night[4:6]), int(night[6:])
    return pool_bias_frames(
        root / night, date(y, m, d),
        target_config=TARGET_CONFIG, window_nights=window, target_frames=target,
        explicit_metas=list(explicit), scanner=scanner, profile=profile, settings=settings,
    )


def _flat_pool(root: Path, night: str, *, window=30, target=20, explicit=(), scanner=None,
               roots=(), filter_name="R", fallback_max=365, settings=None, profile=None):
    profile = profile or _profile()
    settings = settings or _settings()
    scanner = scanner or NightScanner(profile, settings)
    y, m, d = int(night[:4]), int(night[4:6]), int(night[6:])
    return pool_flat_frames_window(
        root / night, date(y, m, d),
        target_config=TARGET_CONFIG, filter_name=filter_name,
        window_nights=window, target_frames=target, explicit_metas=list(explicit),
        scanner=scanner, fallback_max_nights=fallback_max, master_roots=roots,
    )


def _metas(paths, profile=None, settings=None):
    profile = profile or _profile()
    settings = settings or _settings()
    return [read_meta(p, profile, settings=settings) for p in paths]


def _write_master(root: Path, night: str, filter_key: str = "R_dome", *, n_frames=25,
                  gainset=10.0, readout=READOUT, coldpix=None) -> Path:
    calib = root / night / "calib"
    calib.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(data=np.ones((64, 64), dtype=np.float32))
    hdu.header["BUNIT"] = "1"
    if gainset is not None:
        hdu.header["GAINSET"] = gainset
    hdu.header["READOUTM"] = readout
    if n_frames is not None:
        hdu.header["FLATNACC"] = n_frames
    if coldpix:
        hdu.header["COLDPIX"] = coldpix
        fits.PrimaryHDU(data=np.zeros((64, 64), dtype=np.uint8)).writeto(calib / coldpix)
    path = calib / f"master_flat_{filter_key}.fits"
    hdu.writeto(path)
    return path


# --------------------------------------------------------------------------
# window_night_dirs / infer_science_night / limit_bias_for_gain
# --------------------------------------------------------------------------


def test_window_night_dirs_is_symmetric_nearest_first_earlier_on_tie(tmp_path):
    root = tmp_path / "raw"
    for n in ("20240120", "20240119", "20240121", "20240117", "20240124", "20240110",
              "20240131", "notanight"):
        (root / n).mkdir(parents=True)
    got = window_night_dirs(root / "20240120", date(2024, 1, 20), 10)
    assert [p.name for _, p in got] == [
        "20240120", "20240119", "20240121", "20240117", "20240124", "20240110",
    ]
    # 20240131 is 11 days away: outside a +/-10 window.


def test_infer_science_night(tmp_path):
    root = tmp_path / "raw"
    d = _night(root, "20240120")
    sci = _write(d / "sci.fits", frame_type="science")
    other = _write(root / "20240121" / "sci.fits", frame_type="science")
    metas = _metas([sci])
    assert infer_science_night(metas) == (d, date(2024, 1, 20))
    assert infer_science_night(_metas([sci, other])) is None
    loose = _write(tmp_path / "loose" / "sci.fits", frame_type="science")
    assert infer_science_night(_metas([loose])) is None


def test_limit_bias_for_gain_keeps_first_n_per_config_and_other_frames(tmp_path):
    root = tmp_path / "raw"
    d = _night(root, "20240120", bias=5, flats=2)
    sci = _write(d / "sci.fits", frame_type="science")
    metas = _metas(sorted(d.glob("*.fits")))
    kept = limit_bias_for_gain(metas, 3)
    assert sum(m.frame_type == "bias" for m in kept) == 3
    assert [m.path.name for m in kept if m.frame_type == "bias"] == [
        "bias_00.fits", "bias_01.fits", "bias_02.fits",
    ]
    assert sum(m.frame_type == "flat" for m in kept) == 2
    assert any(m.path == sci for m in kept)
    assert len(limit_bias_for_gain(metas, 0)) == len(metas)


# --------------------------------------------------------------------------
# bias pooling
# --------------------------------------------------------------------------


def test_bias_pool_uses_earlier_and_later_nights_within_window(tmp_path):
    root = tmp_path / "raw"
    _night(root, "20240120", bias=2)
    _night(root, "20240118", bias=2)   # 2 d earlier
    _night(root, "20240123", bias=2)   # 3 d later
    _night(root, "20240130", bias=2)   # exactly 10 d later: inside
    _night(root, "20240131", bias=2)   # 11 d later: outside
    _night(root, "20240109", bias=2)   # 11 d earlier: outside
    sel = _bias_pool(root, "20240120", target=1000)
    assert set(sel.nights) == {"20240120", "20240118", "20240123", "20240130"}
    assert len(sel.metas) == 8
    assert sel.borrowed is True
    assert sel.short is True


def test_bias_pool_is_nearest_first_whole_nights_and_stops_at_target(tmp_path):
    root = tmp_path / "raw"
    _night(root, "20240120", bias=3)
    _night(root, "20240119", bias=3)   # 1 d earlier
    _night(root, "20240121", bias=3)   # 1 d later
    _night(root, "20240117", bias=3)   # 3 d earlier
    _night(root, "20240125", bias=3)   # 5 d later
    profile, settings = _profile(), _settings()
    scanner = NightScanner(profile, settings)
    sel = _bias_pool(root, "20240120", target=8, scanner=scanner)
    # own 3 + 20240119 (3) = 6 < 8, + 20240121 (3) = 9 >= 8: whole night kept,
    # overshooting the target; nothing farther is added or even opened.
    assert sel.nights == ("20240120", "20240119", "20240121")
    assert len(sel.metas) == 9
    assert sel.short is False
    scanned = {p.name for p in scanner.scanned}
    assert scanned == {"20240120", "20240119", "20240121"}
    # frames are ordered nearest night first, own night first
    assert [m.path.parent.name for m in sel.metas[:3]] == ["20240120"] * 3


def test_bias_pool_always_includes_own_night_even_beyond_target(tmp_path):
    root = tmp_path / "raw"
    _night(root, "20240120", bias=5)
    _night(root, "20240119", bias=5)
    profile, settings = _profile(), _settings()
    scanner = NightScanner(profile, settings)
    sel = _bias_pool(root, "20240120", target=3, scanner=scanner)
    assert sel.nights == ("20240120",)
    assert len(sel.metas) == 5
    assert sel.borrowed is False
    assert [p.name for p in scanner.scanned] == ["20240120"]


def test_bias_pool_ties_take_the_earlier_night_first(tmp_path):
    root = tmp_path / "raw"
    _night(root, "20240120")
    _night(root, "20240119", bias=4)
    _night(root, "20240121", bias=4)
    sel = _bias_pool(root, "20240120", target=4)
    assert sel.nights == ("20240119",)
    assert len(sel.metas) == 4


def test_bias_pool_matches_readout_configuration_and_validity(tmp_path):
    root = tmp_path / "raw"
    d = _night(root, "20240119", bias=2)
    _write(d / "bias_gain30.fits", frame_type="bias", gain=30.0)
    _write(d / "bias_bin1.fits", frame_type="bias", binning=1)
    _write(d / "bias_invalid.fits", frame_type="bias", valid=False)
    _write(d / "flat_r.fits", frame_type="flat", level=20000.0)
    sel = _bias_pool(root, "20240120", target=100)
    assert sorted(m.path.name for m in sel.metas) == ["bias_00.fits", "bias_01.fits"]


def test_bias_pool_short_logs_warning_with_count_and_window(tmp_path):
    root = tmp_path / "raw"
    _night(root, "20240120", bias=2)
    _night(root, "20240118", bias=3)
    with _capture() as cap:
        sel = _bias_pool(root, "20240120", window=10, target=100)
    assert sel.short is True and len(sel.metas) == 5
    warnings = cap.messages(logging.WARNING)
    assert len(warnings) == 1
    assert "5 frame(s)" in warnings[0]
    assert "target 100" in warnings[0]
    assert "+/-10" in warnings[0]


def test_bias_pool_full_pool_logs_info_not_warning(tmp_path):
    root = tmp_path / "raw"
    _night(root, "20240120", bias=4)
    with _capture() as cap:
        _bias_pool(root, "20240120", target=3)
    assert cap.messages(logging.WARNING) == []
    assert any("4 frame(s)" in m for m in cap.messages(logging.INFO))


def test_bias_pool_zero_in_window_falls_back_beyond_the_window(tmp_path):
    root = tmp_path / "raw"
    (root / "20240120").mkdir(parents=True)
    _night(root, "20240105", bias=4)   # 15 d earlier: beyond +/-10, within 30
    _night(root, "20240210", bias=9)   # later and beyond the window: never used
    with _capture() as cap:
        sel = _bias_pool(root, "20240120", window=10, target=100)
    assert sel.beyond_window is True
    assert sel.nights == ("20240105",)
    assert len(sel.metas) == 4
    assert any("falling back" in m for m in cap.messages(logging.WARNING))


def test_bias_pool_zero_in_window_and_nothing_beyond_gives_empty(tmp_path):
    root = tmp_path / "raw"
    (root / "20240120").mkdir(parents=True)
    _night(root, "20231001", bias=9)   # more than 30 d back
    sel = _bias_pool(root, "20240120")
    assert sel.metas == ()
    assert sel.nights == ()


def test_bias_pool_dedupes_explicit_inputs_and_keeps_outside_window(tmp_path):
    root = tmp_path / "raw"
    own = _night(root, "20240120", bias=2)
    _night(root, "20240119", bias=2)
    far = _night(root, "20231101", bias=1)   # far outside the window
    explicit = _metas(
        [
            own / "bias_00.fits",
            # same file as a sibling scan will find, spelled differently
            root / "20240120" / ".." / "20240119" / "bias_00.fits",
            far / "bias_00.fits",
        ]
    )
    sel = _bias_pool(root, "20240120", target=100, explicit=explicit)
    names = sorted((m.path.parent.name, m.path.name) for m in sel.metas)
    assert len(names) == len(set(names)) == 5
    assert ("20231101", "bias_00.fits") in names      # explicit frame kept
    assert sel.n_explicit_only == 1                  # only the far one is not in the scan
    assert sel.nights[0] == "20240120"
    assert sel.nights[-1] == "20231101"              # farthest night last


# --------------------------------------------------------------------------
# flat pooling
# --------------------------------------------------------------------------


def test_flat_pool_window_is_symmetric_and_matches_filter_and_config(tmp_path):
    root = tmp_path / "raw"
    (root / "20240120").mkdir(parents=True)
    _night(root, "20240110", flats=2)      # 10 d earlier
    _night(root, "20240130", flats=2)      # 10 d later
    _night(root, "20240215", flats=5)      # 26 d later: inside +/-30
    _night(root, "20240301", flats=5)      # 41 d later: outside
    d = root / "20240110"
    _write(d / "flat_g.fits", frame_type="flat", filter_name="G", level=20000.0)
    _write(d / "flat_r_gain30.fits", frame_type="flat", gain=30.0, level=20000.0)
    _write(d / "flat_r_bin1.fits", frame_type="flat", binning=1, level=20000.0)
    sel = _flat_pool(root, "20240120", window=30, target=1000)
    assert sel.nights == ("20240110", "20240130", "20240215")
    assert len(sel.metas) == 9
    assert all(m.filter == "R" for m in sel.metas)
    assert sel.reuse is None


def test_flat_pool_stops_at_target_after_whole_nights(tmp_path):
    root = tmp_path / "raw"
    (root / "20240120").mkdir(parents=True)
    _night(root, "20240118", flats=7)
    _night(root, "20240122", flats=7)
    _night(root, "20240115", flats=7)
    _night(root, "20240201", flats=7)
    profile, settings = _profile(), _settings()
    scanner = NightScanner(profile, settings)
    sel = _flat_pool(root, "20240120", target=20, scanner=scanner)
    assert sel.nights == ("20240118", "20240122", "20240115")
    assert len(sel.metas) == 21
    assert "20240201" not in {p.name for p in scanner.scanned}


def test_flat_pool_short_without_master_roots_builds_what_is_available(tmp_path):
    root = tmp_path / "raw"
    _night(root, "20240120", flats=3)
    with _capture() as cap:
        sel = _flat_pool(root, "20240120", target=20)
    assert len(sel.metas) == 3 and sel.short and sel.reuse is None
    warnings = cap.messages(logging.WARNING)
    assert len(warnings) == 1 and "3 frame(s)" in warnings[0] and "+/-30" in warnings[0]


def test_flat_pool_zero_in_window_without_roots_uses_old_beyond_window_fallback(tmp_path):
    root = tmp_path / "raw"
    (root / "20240120").mkdir(parents=True)
    _night(root, "20231201", flats=4)     # 50 d earlier: beyond +/-30
    _night(root, "20231115", flats=4)
    _night(root, "20240301", flats=4)     # later: never used by the fallback
    sel = _flat_pool(root, "20240120", window=30, target=20, fallback_max=365)
    assert sel.beyond_window is True
    assert sel.nights == ("20231201",)
    assert len(sel.metas) == 4


# --------------------------------------------------------------------------
# nearest existing master flat
# --------------------------------------------------------------------------


def test_find_nearest_master_flat_either_direction_and_tie_goes_earlier(tmp_path):
    red = tmp_path / "reduced"
    _write_master(red, "20240110")           # 10 d earlier
    _write_master(red, "20240126")           # 6 d later
    choice, n_short = find_nearest_master_flat(
        [red], date(2024, 1, 20), filter_name="R", gain_setting=10.0,
        readout_mode=READOUT, min_frames=20,
    )
    assert choice.night == date(2024, 1, 26) and choice.distance_days == 6
    assert choice.n_frames == 25 and n_short == 0
    _write_master(red, "20240114")           # 6 d earlier: ties with 20240126
    choice, _ = find_nearest_master_flat(
        [red], date(2024, 1, 20), filter_name="R", gain_setting=10.0,
        readout_mode=READOUT, min_frames=20,
    )
    assert choice.night == date(2024, 1, 14) and choice.distance_days == -6


def test_find_nearest_master_flat_has_no_maximum_distance(tmp_path):
    red = tmp_path / "reduced"
    _write_master(red, "20230101")
    choice, _ = find_nearest_master_flat(
        [red], date(2024, 1, 20), filter_name="R", gain_setting=10.0,
        readout_mode=READOUT, min_frames=20,
    )
    assert choice is not None and choice.distance_days == -384


def test_find_nearest_master_flat_skips_other_filter_gain_readout_and_own_night(tmp_path):
    red = tmp_path / "reduced"
    _write_master(red, "20240119", filter_key="G_dome")            # wrong filter
    _write_master(red, "20240118", gainset=30.0)                    # wrong gain setting
    _write_master(red, "20240117", readout="Other Mode")            # wrong readout
    _write_master(red, "20240120")                                  # the night itself
    _write_master(red, "20240101", gainset=None, readout="UNKNOWN")  # missing GAINSET ok
    choice, _ = find_nearest_master_flat(
        [red], date(2024, 1, 20), filter_name="R", gain_setting=10.0,
        readout_mode=None, min_frames=20,
    )
    assert choice.night == date(2024, 1, 1)
    none, _ = find_nearest_master_flat(
        [red], date(2024, 1, 20), filter_name="B", gain_setting=10.0,
        readout_mode=None, min_frames=20,
    )
    assert none is None


def test_find_nearest_master_flat_frame_count_rule(tmp_path):
    red = tmp_path / "reduced"
    _write_master(red, "20240119", n_frames=13)       # built from too few frames
    _write_master(red, "20240101", n_frames=None)     # no count card: accepted
    choice, n_short = find_nearest_master_flat(
        [red], date(2024, 1, 20), filter_name="R", gain_setting=10.0,
        readout_mode=READOUT, min_frames=20,
    )
    assert choice.night == date(2024, 1, 1) and choice.n_frames is None
    assert n_short == 1
    _write_master(red, "20231201", n_frames=25)
    (red / "20240101" / "calib" / "master_flat_R_dome.fits").unlink()
    choice, n_short = find_nearest_master_flat(
        [red], date(2024, 1, 20), filter_name="R", gain_setting=10.0,
        readout_mode=READOUT, min_frames=20,
    )
    assert choice.night == date(2023, 12, 1) and n_short == 1


def test_find_nearest_master_flat_missing_root_is_empty(tmp_path):
    assert find_nearest_master_flat(
        [tmp_path / "nope"], date(2024, 1, 20), filter_name="R", gain_setting=10.0,
        readout_mode=READOUT, min_frames=20,
    ) == (None, 0)


def test_flat_pool_short_window_reuses_nearest_master(tmp_path):
    raw, red = tmp_path / "raw", tmp_path / "reduced"
    _night(raw, "20240120", flats=3)                 # short pool: 3 < 20
    master = _write_master(red, "20240112")
    with _capture() as cap:
        sel = _flat_pool(raw, "20240120", target=20, roots=(red,))
    assert sel.reuse == MasterFlatChoice(
        path=master, night=date(2024, 1, 12), distance_days=-8, n_frames=25
    )
    assert len(sel.metas) == 3      # kept for the gain measurement only
    assert cap.messages(logging.WARNING) == []
    assert any("reusing the existing master" in m for m in cap.messages(logging.INFO))


def test_flat_pool_zero_in_window_reuses_nearest_master_not_raw_fallback(tmp_path):
    raw, red = tmp_path / "raw", tmp_path / "reduced"
    (raw / "20240120").mkdir(parents=True)
    _night(raw, "20231201", flats=30)    # the old raw-frame fallback would take this
    _write_master(red, "20240301")
    sel = _flat_pool(raw, "20240120", target=20, roots=(red,))
    assert sel.reuse is not None and sel.reuse.night == date(2024, 3, 1)
    assert sel.metas == ()
    assert sel.beyond_window is False


def test_flat_pool_full_window_never_looks_for_a_master(tmp_path):
    raw, red = tmp_path / "raw", tmp_path / "reduced"
    _night(raw, "20240120", flats=25)
    _write_master(red, "20240119")
    sel = _flat_pool(raw, "20240120", target=20, roots=(red,))
    assert sel.reuse is None and len(sel.metas) == 25


def test_flat_pool_without_any_master_builds_short_pool_with_warning(tmp_path):
    raw, red = tmp_path / "raw", tmp_path / "reduced"
    _night(raw, "20240119", flats=4)
    _write_master(red, "20240118", filter_key="G_dome")     # other filter only
    _write_master(red, "20240117", n_frames=13)             # too short
    with _capture() as cap:
        sel = _flat_pool(raw, "20240120", target=20, roots=(red,))
    assert sel.reuse is None and len(sel.metas) == 4
    text = " ".join(cap.messages(logging.WARNING))
    assert "4 R flat frame(s)" in text and "no existing master" in text
    assert "1 older master(s) too short" in text


def test_flat_pool_no_pool_and_no_master_gives_empty_selection(tmp_path):
    raw, red = tmp_path / "raw", tmp_path / "reduced"
    (raw / "20240120").mkdir(parents=True)
    red.mkdir()
    with _capture() as cap:
        sel = _flat_pool(raw, "20240120", target=20, roots=(red,))
    assert sel.reuse is None and sel.metas == ()
    assert any("continuing without a master flat" in m for m in cap.messages(logging.WARNING))


def test_install_master_flat_copies_with_provenance_and_leaves_source_alone(tmp_path):
    red = tmp_path / "reduced"
    src = _write_master(red, "20240112", coldpix="coldpix_R_dome.fits")
    before = src.read_bytes()
    calib = tmp_path / "calib"
    choice = MasterFlatChoice(src, date(2024, 1, 12), -8, 25)
    dest = install_master_flat(
        choice, calib, current_night=date(2024, 1, 20), window_nights=30, target_frames=20
    )
    assert dest == calib / "master_flat_R_dome.fits"
    assert src.read_bytes() == before
    assert (calib / "coldpix_R_dome.fits").exists()
    assert not list(calib.glob(".*.tmp"))
    h = fits.getheader(dest, 0)
    assert h["CALFBACK"] is True
    assert h["FLATNGT"] == "20240112"
    assert h["FLATREUS"] is True
    assert h["FLATRFIL"] == "master_flat_R_dome.fits"
    assert h["FLATRNGT"] == "20240112"
    assert h["FLATRDST"] == -8
    assert h["FLATRNFR"] == 25
    assert h["FLATWIN"] == 30 and h["FLATTGT"] == 20
    assert str(src) in "".join(str(x) for x in h["HISTORY"])
    assert h["GAINSET"] == 10.0          # the source's own cards survive


def test_install_master_flat_refuses_to_copy_onto_itself(tmp_path):
    src = _write_master(tmp_path / "reduced", "20240112")
    choice = MasterFlatChoice(src, date(2024, 1, 12), -8, 25)
    with pytest.raises(ValueError):
        install_master_flat(
            choice, src.parent, current_night=date(2024, 1, 20),
            window_nights=30, target_frames=20,
        )


# --------------------------------------------------------------------------
# apply_window_pools
# --------------------------------------------------------------------------


def test_apply_window_pools_is_a_no_op_for_a_profile_without_the_keys(tmp_path):
    root = tmp_path / "raw"
    d = _night(root, "20240120", bias=2, flats=2)
    _night(root, "20240119", bias=5, flats=5)
    sci = _write(d / "sci.fits", frame_type="science")
    metas = _metas([sci, *sorted(d.glob("bias_*.fits")), *sorted(d.glob("flat_*.fits"))])

    class _NoScan:
        def calib_frames(self, _path):
            raise AssertionError("must not scan any night")

    for profile in (
        _profile(),
        _profile(flat_pool=True),                       # legacy backward pool only
        _profile(flat_pool_window_nights=30),           # window without flat_pool
    ):
        res = apply_window_pools(
            metas, profile=profile, settings=_settings(),
            night_dir=d, current_night=date(2024, 1, 20), scanner=_NoScan(),
        )
        assert res.metas == list(metas)
        assert res.bias is None and res.flats == {}


def test_apply_window_pools_merges_explicit_frames_and_replaces_pooled_types(tmp_path):
    root = tmp_path / "raw"
    d = _night(root, "20240120", bias=2)
    _night(root, "20240118", bias=3, flats=4)
    _night(root, "20240125", flats=2)
    sci = _write(d / "sci.fits", frame_type="science")
    explicit_flat = root / "20240118" / "flat_r_00.fits"
    metas = _metas([sci, d / "bias_00.fits", explicit_flat])
    profile = _profile(
        bias_pool=True, bias_pool_target_frames=4, flat_pool=True,
        flat_pool_window_nights=30, flat_pool_target_frames=20,
    )
    res = apply_window_pools(
        metas, profile=profile, settings=_settings(),
        night_dir=d, current_night=date(2024, 1, 20),
    )
    bias = [m for m in res.metas if m.frame_type == "bias"]
    flats = [m for m in res.metas if m.frame_type == "flat"]
    assert len(bias) == 5 and len({m.path for m in bias}) == 5   # own 2 + 3 earlier
    assert len(flats) == 6 and len({m.path for m in flats}) == 6
    assert [m for m in res.metas if m.frame_type == "science"] == [metas[0]]
    assert res.bias_active and res.flat_active
    assert res.flats["R"].nights == ("20240118", "20240125")


def test_apply_window_pools_limits_flats_to_science_filters_and_filter_option(tmp_path):
    root = tmp_path / "raw"
    d = _night(root, "20240120", flats=2)
    _write(d / "flat_g.fits", frame_type="flat", filter_name="G", level=20000.0)
    sci_r = _write(d / "sci_r.fits", frame_type="science")
    sci_g = _write(d / "sci_g.fits", frame_type="science", filter_name="G")
    metas = _metas([sci_r, sci_g])
    profile = _profile(flat_pool=True, flat_pool_window_nights=30)
    both = apply_window_pools(
        metas, profile=profile, settings=_settings(), night_dir=d,
        current_night=date(2024, 1, 20),
    )
    assert set(both.flats) == {"G", "R"}
    only_g = apply_window_pools(
        metas, profile=profile, settings=_settings(), night_dir=d,
        current_night=date(2024, 1, 20), filters=["G"],
    )
    assert set(only_g.flats) == {"G"}


# --------------------------------------------------------------------------
# build_master_bias: memory-bounded path and provenance cards
# --------------------------------------------------------------------------


def _bias_frames(root: Path, n: int) -> list:
    d = root / "20240120"
    paths = [
        _write(d / f"bias_{i:02d}.fits", frame_type="bias", level=1000.0, noise=3.0, seed=i)
        for i in range(n)
    ]
    return _metas(paths)


def test_build_master_bias_streaming_matches_in_memory(tmp_path, monkeypatch):
    import robo43.calib.bias as bias_module

    calls = {"lazy": 0, "stack": 0}
    real_lazy, real_stack = bias_module.combine_lazy, bias_module.combine_stack

    def _lazy(*a, **k):
        calls["lazy"] += 1
        return real_lazy(*a, **k)

    def _stack(*a, **k):
        calls["stack"] += 1
        return real_stack(*a, **k)

    monkeypatch.setattr(bias_module, "combine_lazy", _lazy)
    monkeypatch.setattr(bias_module, "combine_stack", _stack)
    metas = _bias_frames(tmp_path / "raw", 9)
    profile = _profile()
    in_mem = build_master_bias(
        metas, settings=_settings(), profile=profile, outdir=tmp_path / "mem"
    )
    assert calls == {"lazy": 0, "stack": 1}
    streamed = build_master_bias(
        metas, settings=_settings(max_memory_mb=1e-6), profile=profile,
        outdir=tmp_path / "stream",
    )
    assert calls == {"lazy": 1, "stack": 1}
    np.testing.assert_array_equal(in_mem.data, streamed.data)
    np.testing.assert_array_equal(in_mem.variance, streamed.variance)
    np.testing.assert_array_equal(in_mem.weight, streamed.weight)
    cards_a = {c[0]: c[1] for c in in_mem.header_cards}
    cards_b = {c[0]: c[1] for c in streamed.header_cards}
    assert cards_a == cards_b
    assert not list((tmp_path / "stream").glob(".bias_stream_*"))


def test_build_master_bias_streaming_cleans_up_on_shape_mismatch(tmp_path):
    metas = _bias_frames(tmp_path / "raw", 2)
    with fits.open(metas[1].path, mode="update") as hdul:
        hdul[0].data = np.ones((32, 64), dtype=np.float32)
    bad = _metas([metas[0].path, metas[1].path])
    from robo43.exceptions import CalibrationError

    with pytest.raises(CalibrationError):
        build_master_bias(
            bad, settings=_settings(max_memory_mb=1e-6), profile=_profile(),
            outdir=tmp_path / "stream",
        )
    assert not list((tmp_path / "stream").glob(".bias_stream_*"))


def test_build_master_bias_pool_cards_short_and_compact_forms(tmp_path):
    root = tmp_path / "raw"
    _night(root, "20240120", bias=2)
    _night(root, "20240118", bias=2)
    sel = _bias_pool(root, "20240120", target=3)
    master = build_master_bias(
        list(sel.metas), settings=_settings(), profile=_profile(),
        outdir=tmp_path / "calib", pool=sel,
    )
    cards = {c[0]: c[1] for c in master.header_cards}
    assert cards["CALFBACK"] is True
    assert cards["BIASNGT"] == "20240120,20240118"
    assert cards["BIASNFRM"] == 4 and cards["BIASWIN"] == 10 and cards["BIASTGT"] == 3
    assert any("Master bias pooled from 4 frames" in h for h in master.history)

    # A night list longer than a FITS string card uses the compact cards.
    many = type(sel)(
        kind="bias", filter_name=None, metas=sel.metas,
        nights=tuple(f"2024010{i}" for i in range(1, 10)),
        night_counts=tuple((f"2024010{i}", 1) for i in range(1, 10)),
        window_nights=10, target_frames=100, borrowed=True, short=True,
        beyond_window=False,
    )
    master = build_master_bias(
        list(sel.metas), settings=_settings(),
        profile=_profile(), outdir=tmp_path / "calib2", pool=many,
    )
    cards = {c[0]: c[1] for c in master.header_cards}
    assert "BIASNGT" not in cards
    assert cards["BIASN1"] == "20240101" and cards["BIASN2"] == "20240109"
    assert cards["BIASNNGT"] == 9


def test_build_master_flats_adds_pool_window_and_target_cards(tmp_path):
    from robo43.calib.flat import build_master_flats

    root = tmp_path / "raw"
    d = _night(root, "20240120", flats=3)
    metas = _metas(sorted(d.glob("flat_*.fits")))
    masters = build_master_flats(
        metas, settings=_settings(separate_dome_and_sky_flats=False), profile=_profile(),
        outdir=tmp_path / "calib", pool_window_nights=30, pool_target_frames=20,
    )
    cards = {c[0]: c[1] for c in masters["R"].header_cards}
    assert cards["FLATWIN"] == 30 and cards["FLATTGT"] == 20
    plain = build_master_flats(
        metas, settings=_settings(separate_dome_and_sky_flats=False), profile=_profile(),
        outdir=tmp_path / "calib_plain",
    )
    assert "FLATWIN" not in {c[0] for c in plain["R"].header_cards}


# --------------------------------------------------------------------------
# robo43 reduce end to end
# --------------------------------------------------------------------------


def _run_reduce(tmp_path, monkeypatch, profile, inputs, *extra):
    monkeypatch.setattr("robo43.cli.reduce.resolve_profile", lambda **_kw: profile)
    calib_dir, outdir = tmp_path / "calib", tmp_path / "output"
    code = main([
        "reduce", *[str(i) for i in inputs],
        "--calibdir", str(calib_dir), "--outdir", str(outdir),
        "--no-overscan", "--loglevel", "INFO", *extra,
    ])
    return code, calib_dir, outdir


def test_reduce_pools_bias_and_flats_both_sides_from_an_explicit_file_list(
    tmp_path, monkeypatch
):
    raw = tmp_path / "T80S"
    d = raw / "20240120"
    sci = _write(d / "science_r.fits", frame_type="science")
    _night(raw, "20240118", bias=3, flats=3)       # earlier
    _night(raw, "20240123", bias=3, flats=3)       # later
    _night(raw, "20240101", bias=3, flats=3)       # 19 d earlier: flats window only
    profile = _profile(
        bias_pool=True, bias_pool_window_nights=10, bias_pool_target_frames=100,
        flat_pool=True, flat_pool_window_nights=30, flat_pool_target_frames=20,
    )
    # The inputs are one science file plus one explicit bias from the earlier night.
    code, calib, outdir = _run_reduce(
        tmp_path, monkeypatch, profile, [sci, raw / "20240118" / "bias_00.fits"]
    )
    assert code == 0
    with fits.open(calib / "master_bias.fits") as hdul:
        h = hdul[0].header
        assert h["NCOMBINE"] == 6 and h["BIASNFRM"] == 6      # 3 earlier + 3 later, deduped
        assert h["CALFBACK"] is True
        assert h["BIASNGT"] == "20240118,20240123"
        assert h["BIASWIN"] == 10 and h["BIASTGT"] == 100
    flat_files = list(calib.glob("master_flat_R*.fits"))
    assert len(flat_files) == 1
    with fits.open(flat_files[0]) as hdul:
        h = hdul[0].header
        assert h["FLATNFRM"] == 9                              # three nights x 3 flats
        assert set(h["FLATNGT"].split(",")) == {"20240118", "20240123", "20240101"}
        assert h["FLATWIN"] == 30 and h["FLATTGT"] == 20
    assert (outdir / "science_r_proc.fits").exists()


def test_reduce_pool_is_off_for_a_profile_without_the_keys(tmp_path, monkeypatch):
    raw = tmp_path / "T80S"
    d = raw / "20240120"
    sci = _write(d / "science_r.fits", frame_type="science")
    own_bias = [_write(d / f"bias_{i}.fits", frame_type="bias") for i in range(2)]
    _night(raw, "20240118", bias=5)
    code, calib, _ = _run_reduce(tmp_path, monkeypatch, _profile(), [sci, *own_bias])
    assert code == 0
    with fits.open(calib / "master_bias.fits") as hdul:
        h = hdul[0].header
        assert h["NCOMBINE"] == 2
        assert h["CALFBACK"] is False
        assert "BIASNFRM" not in h and "BIASWIN" not in h


def test_reduce_no_calib_fallback_disables_the_pool(tmp_path, monkeypatch):
    raw = tmp_path / "T80S"
    d = raw / "20240120"
    sci = _write(d / "science_r.fits", frame_type="science")
    own_bias = [_write(d / f"bias_{i}.fits", frame_type="bias") for i in range(2)]
    _night(raw, "20240118", bias=5)
    profile = _profile(bias_pool=True)
    code, calib, _ = _run_reduce(
        tmp_path, monkeypatch, profile, [sci, *own_bias], "--no-calib-fallback"
    )
    assert code == 0
    with fits.open(calib / "master_bias.fits") as hdul:
        assert hdul[0].header["NCOMBINE"] == 2


def test_reduce_short_flat_pool_copies_the_nearest_master_instead_of_building(
    tmp_path, monkeypatch
):
    raw, red = tmp_path / "T80S", tmp_path / "reduced"
    d = raw / "20240120"
    sci = _write(d / "science_r.fits", frame_type="science")
    _night(raw, "20240118", bias=3, flats=3)       # only 3 flats in the window
    _write_master(red, "20240101", n_frames=13)    # too short: not eligible
    master = _write_master(red, "20240210", coldpix="coldpix_R_dome.fits")  # 21 d later
    profile = _profile(
        bias_pool=True, flat_pool=True, flat_pool_window_nights=30,
        flat_pool_target_frames=20, flat_master_roots=(str(red),),
    )
    code, calib, outdir = _run_reduce(tmp_path, monkeypatch, profile, [sci])
    assert code == 0
    out = calib / "master_flat_R_dome.fits"
    assert out.exists()
    assert not list(calib.glob("master_flat_R.fits"))       # nothing was built
    h = fits.getheader(out, 0)
    assert h["FLATREUS"] is True and h["FLATRNGT"] == "20240210" and h["FLATRDST"] == 21
    assert h["FLATRNFR"] == 25 and h["CALFBACK"] is True
    assert (calib / "coldpix_R_dome.fits").exists()
    assert fits.getheader(master, 0).get("FLATREUS") is None   # source untouched
    with fits.open(outdir / "science_r_proc.fits") as hdul:
        sci_header = hdul["SCI"].header
        assert sci_header["FLATFIL"] == "master_flat_R_dome.fits"
        assert sci_header["FLATNGT"] == "20240210"
        assert sci_header["CALFBACK"] is True


def test_reduce_short_flat_pool_without_any_master_builds_from_the_short_pool(
    tmp_path, monkeypatch
):
    raw, red = tmp_path / "T80S", tmp_path / "reduced"
    red.mkdir()
    d = raw / "20240120"
    sci = _write(d / "science_r.fits", frame_type="science")
    _night(raw, "20240118", bias=3, flats=3)
    profile = _profile(
        flat_pool=True, flat_pool_window_nights=30, flat_pool_target_frames=20,
        flat_master_roots=(str(red),),
    )
    explicit_bias = sorted((raw / "20240118").glob("bias_*"))
    code, calib, _ = _run_reduce(tmp_path, monkeypatch, profile, [sci, *explicit_bias])
    assert code == 0
    files = list(calib.glob("master_flat_R*.fits"))
    assert len(files) == 1
    h = fits.getheader(files[0], 0)
    assert h["FLATNFRM"] == 3 and "FLATREUS" not in h
    assert h["FLATWIN"] == 30


def test_reduce_dry_run_does_not_copy_a_master(tmp_path, monkeypatch):
    raw, red = tmp_path / "T80S", tmp_path / "reduced"
    d = raw / "20240120"
    sci = _write(d / "science_r.fits", frame_type="science")
    _night(raw, "20240118", bias=3, flats=3)
    _write_master(red, "20240210")
    profile = _profile(
        flat_pool=True, flat_pool_window_nights=30, flat_master_roots=(str(red),),
    )
    code, calib, _ = _run_reduce(tmp_path, monkeypatch, profile, [sci], "--dry-run")
    assert code == 0
    assert not calib.exists() or not list(calib.glob("master_flat_*"))
