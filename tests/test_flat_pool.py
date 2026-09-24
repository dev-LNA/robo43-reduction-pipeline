"""Tests for robo43.calib.fallback.pool_flat_frames (the always-on, per-filter
multi-night flat pooling policy) and its wiring into robo43.cli.reduce via
InstrumentProfile.calib.flat_pool.

Builds its own synthetic FITS files under tmp_path in a root/YYYYMMDD/ layout;
does not use fixtures or conftest.py, matching the style of
tests/test_calib_fallback.py.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from robo43.calib.fallback import pool_flat_frames
from robo43.cli.main import main
from robo43.config import CalibSettings, RuntimeSettings, Settings
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


def _profile(*, flat_pool: bool = False, **calib_overrides) -> InstrumentProfile:
    calib = InstrumentCalibOptions(flat_pool=flat_pool, **calib_overrides)
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
        calib=calib,
    )


def _settings(**calib_overrides) -> Settings:
    return Settings(
        paths=None,
        astrometry=None,
        photometry=None,
        calib=CalibSettings(**calib_overrides),
        coadd=None,
        diagnostics=None,
        db=None,
        service=None,
        runtime=RuntimeSettings(),
        targets=None,
    )


def _write_frame(
    path: Path,
    *,
    frame_type: str,
    filter_name: str = "R",
    gain: float = 10.0,
    binning: int = 2,
    naxis: tuple[int, int] = (64, 64),
    level: float = 1000.0,
) -> None:
    data = np.ones(naxis, dtype=np.float32) * level
    hdu = fits.PrimaryHDU(data=data)
    hdu.header["IMAGETYP"] = frame_type.upper()
    hdu.header["EXPTIME"] = 10.0
    hdu.header["FILTER"] = filter_name
    hdu.header["VALID"] = True
    hdu.header["DATE-OBS"] = "2024-01-01T00:00:00"
    hdu.header["NAXIS1"] = naxis[1]
    hdu.header["NAXIS2"] = naxis[0]
    hdu.header["GAIN"] = gain
    hdu.header["XBINNING"] = binning
    hdu.header["READOUTM"] = "High Gain Mode 16BIT"
    hdu.header["READNOISE"] = 5.0
    hdu.header["SATURATE"] = 60000.0
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(path, overwrite=True)


def _target_config():
    return ("High Gain Mode 16BIT", 10.0, 2, 64, 64)


def _local_metas(paths, profile, settings):
    return [read_meta(p, profile, settings=settings) for p in paths]


# --------------------------------------------------------------------------
# pool_flat_frames: unit tests
# --------------------------------------------------------------------------


def test_pool_includes_current_night_and_floor_nights(tmp_path):
    """j1/j2: current-night flats plus every night within the floor window
    are all pooled, current night listed first."""
    root = tmp_path / "T80S"
    night_dir = root / "20240120"
    night_dir.mkdir(parents=True)

    local_paths = []
    for i in range(2):
        p = night_dir / f"flat_r_{i}.fits"
        _write_frame(p, frame_type="flat", filter_name="R")
        local_paths.append(p)
    for i in range(3):
        _write_frame(root / "20240115" / f"flat_r_{i}.fits", frame_type="flat", filter_name="R")
    for i in range(2):
        _write_frame(root / "20240112" / f"flat_r_{i}.fits", frame_type="flat", filter_name="R")

    profile = _profile()
    settings = _settings()
    local_metas = _local_metas(local_paths, profile, settings)

    result = pool_flat_frames(
        night_dir,
        date(2024, 1, 20),
        local_metas=local_metas,
        target_config=_target_config(),
        filter_name="R",
        min_nights=10,
        max_nights=20,
        target_frames=20,
        fallback_max_nights=365,
        profile=profile,
        settings=settings,
    )

    # 2 local + 3 (20240115, 5 days back) + 2 (20240112, 8 days back).
    assert result.contributing_nights[0] == "20240120"
    assert set(result.contributing_nights) == {"20240120", "20240115", "20240112"}
    assert len(result.metas) == 2 + 3 + 2
    assert result.borrowed is True


def test_pool_extension_stops_once_target_reached(tmp_path):
    """j3: once the floor window is short of the target, nights are added
    nearest-first only until the target is reached (the whole night that
    reaches it is included, no further nights are scanned)."""
    root = tmp_path / "T80S"
    night_dir = root / "20240120"
    night_dir.mkdir(parents=True)

    # Floor window (<=5 days): 3 frames only, short of target=5.
    for i in range(3):
        _write_frame(root / "20240116" / f"flat_r_{i}.fits", frame_type="flat", filter_name="R")
    # Extension window: nearest night (20240110, 10 days back) has enough
    # alone to reach the target.
    for i in range(4):
        _write_frame(root / "20240110" / f"flat_r_{i}.fits", frame_type="flat", filter_name="R")
    # Further night in the extension window: must NOT be scanned once the
    # target is already met by 20240110.
    for i in range(10):
        _write_frame(root / "20240105" / f"flat_r_{i}.fits", frame_type="flat", filter_name="R")

    profile = _profile()
    settings = _settings()

    result = pool_flat_frames(
        night_dir,
        date(2024, 1, 20),
        local_metas=[],
        target_config=_target_config(),
        filter_name="R",
        min_nights=5,
        max_nights=15,
        target_frames=5,
        fallback_max_nights=365,
        profile=profile,
        settings=settings,
    )

    assert result.contributing_nights == ("20240116", "20240110")
    assert len(result.metas) == 3 + 4
    assert result.borrowed is True


def test_pool_accepts_fewer_than_target_within_window(tmp_path):
    """j4: a non-empty pool still short of the target after max_nights is
    used as-is, no search past max_nights."""
    root = tmp_path / "T80S"
    night_dir = root / "20240120"
    night_dir.mkdir(parents=True)

    for i in range(2):
        _write_frame(root / "20240110" / f"flat_r_{i}.fits", frame_type="flat", filter_name="R")
    # Well beyond max_nights=20; must not be reached since the pool is
    # non-empty (partial pools never search further).
    for i in range(10):
        _write_frame(root / "20231201" / f"flat_r_{i}.fits", frame_type="flat", filter_name="R")

    profile = _profile()
    settings = _settings()

    result = pool_flat_frames(
        night_dir,
        date(2024, 1, 20),
        local_metas=[],
        target_config=_target_config(),
        filter_name="R",
        min_nights=10,
        max_nights=20,
        target_frames=20,
        fallback_max_nights=365,
        profile=profile,
        settings=settings,
    )

    assert result.contributing_nights == ("20240110",)
    assert len(result.metas) == 2
    assert result.borrowed is True


def test_pool_falls_back_to_nearest_older_night_when_empty_within_window(tmp_path):
    """j5: nothing at all within max_nights days -- fall back to the single
    nearest older night (bounded by fallback_max_nights), not accumulated
    with any other night."""
    root = tmp_path / "T80S"
    night_dir = root / "20240120"
    night_dir.mkdir(parents=True)

    # Nothing within 20 days. Two older nights; only the nearest is used.
    for i in range(4):
        _write_frame(root / "20231201" / f"flat_r_{i}.fits", frame_type="flat", filter_name="R")
    for i in range(4):
        _write_frame(root / "20231115" / f"flat_r_{i}.fits", frame_type="flat", filter_name="R")

    profile = _profile()
    settings = _settings()

    result = pool_flat_frames(
        night_dir,
        date(2024, 1, 20),
        local_metas=[],
        target_config=_target_config(),
        filter_name="R",
        min_nights=10,
        max_nights=20,
        target_frames=20,
        fallback_max_nights=365,
        profile=profile,
        settings=settings,
    )

    assert result.contributing_nights == ("20231201",)
    assert len(result.metas) == 4
    assert result.borrowed is True


def test_pool_empty_when_nothing_found_at_all(tmp_path):
    root = tmp_path / "T80S"
    night_dir = root / "20240120"
    night_dir.mkdir(parents=True)

    profile = _profile()
    settings = _settings()

    result = pool_flat_frames(
        night_dir,
        date(2024, 1, 20),
        local_metas=[],
        target_config=_target_config(),
        filter_name="R",
        min_nights=10,
        max_nights=20,
        target_frames=20,
        fallback_max_nights=30,
        profile=profile,
        settings=settings,
    )

    assert result.metas == ()
    assert result.contributing_nights == ()
    assert result.borrowed is False


def test_pool_ignores_later_nights(tmp_path):
    """j6: a later sibling night must never contribute."""
    root = tmp_path / "T80S"
    night_dir = root / "20240120"
    night_dir.mkdir(parents=True)

    for i in range(10):
        _write_frame(root / "20240125" / f"flat_r_{i}.fits", frame_type="flat", filter_name="R")

    profile = _profile()
    settings = _settings()

    result = pool_flat_frames(
        night_dir,
        date(2024, 1, 20),
        local_metas=[],
        target_config=_target_config(),
        filter_name="R",
        min_nights=10,
        max_nights=20,
        target_frames=20,
        fallback_max_nights=365,
        profile=profile,
        settings=settings,
    )

    assert result.metas == ()
    assert result.borrowed is False


def test_pool_matches_filter_and_config_only(tmp_path):
    root = tmp_path / "T80S"
    night_dir = root / "20240120"
    night_dir.mkdir(parents=True)

    for i in range(3):
        _write_frame(root / "20240115" / f"flat_g_{i}.fits", frame_type="flat", filter_name="G")
    for i in range(3):
        _write_frame(
            root / "20240115" / f"flat_r_wrongbin_{i}.fits",
            frame_type="flat", filter_name="R", binning=1,
        )
    for i in range(2):
        _write_frame(root / "20240115" / f"flat_r_{i}.fits", frame_type="flat", filter_name="R")

    profile = _profile()
    settings = _settings()

    result = pool_flat_frames(
        night_dir,
        date(2024, 1, 20),
        local_metas=[],
        target_config=_target_config(),
        filter_name="R",
        min_nights=10,
        max_nights=20,
        target_frames=20,
        fallback_max_nights=365,
        profile=profile,
        settings=settings,
    )

    assert len(result.metas) == 2
    assert all(m.filter == "R" for m in result.metas)


# --------------------------------------------------------------------------
# InstrumentProfile.calib.flat_pool wiring in cli.reduce
# --------------------------------------------------------------------------


def test_reduce_pools_flats_with_flat_pool_profile(tmp_path, monkeypatch):
    """End-to-end: with a profile that enables flat_pool, a filter's master
    flat pools the current night's own flats with a previous night's, and
    records CALFBACK/FLATNGT/FLATNFRM provenance."""
    root = tmp_path / "T80S"
    frame_dir = root / "20240120"
    frame_dir.mkdir(parents=True)

    for i in range(3):
        _write_frame(
            root / "20240115" / f"flat_r_{i}.fits",
            frame_type="flat", filter_name="R", level=20000.0,
        )
    for i in range(3):
        _write_frame(frame_dir / f"bias_{i}.fits", frame_type="bias")
    for i in range(2):
        _write_frame(frame_dir / f"flat_r_{i}.fits", frame_type="flat", filter_name="R", level=20000.0)
    _write_frame(frame_dir / "science_r.fits", frame_type="science", filter_name="R")

    pool_profile = _profile(
        flat_pool=True, flat_pool_min_nights=10, flat_pool_max_nights=20,
        flat_pool_target_frames=20,
    )
    monkeypatch.setattr("robo43.cli.reduce.resolve_profile", lambda **kw: pool_profile)

    calib_dir = tmp_path / "calib"
    outdir = tmp_path / "output"

    result = main([
        "reduce",
        str(frame_dir),
        "--calibdir", str(calib_dir),
        "--outdir", str(outdir),
        "--no-overscan",
        "--loglevel", "INFO",
    ])
    assert result == 0

    r_flat_files = list(calib_dir.glob("master_flat_R*.fits"))
    assert len(r_flat_files) == 1
    with fits.open(r_flat_files[0]) as hdul:
        header = hdul[0].header
        assert header["CALFBACK"] is True
        assert set(header["FLATNGT"].split(",")) == {"20240120", "20240115"}
        assert header["FLATNFRM"] == 5


def test_reduce_flat_pool_false_keeps_zero_trigger_behaviour(tmp_path, monkeypatch):
    """j7: a profile with flat_pool=False (the default) must leave the
    existing zero-trigger fallback untouched -- a filter WITH local flats
    never triggers a search of previous nights at all, unlike pooling."""
    root = tmp_path / "T80S"
    frame_dir = root / "20240120"
    frame_dir.mkdir(parents=True)

    # A previous night has more/better flats, but must be ignored entirely:
    # the local night already has flats for R.
    for i in range(10):
        _write_frame(
            root / "20240115" / f"flat_r_{i}.fits",
            frame_type="flat", filter_name="R", level=20000.0,
        )
    for i in range(3):
        _write_frame(frame_dir / f"bias_{i}.fits", frame_type="bias")
    _write_frame(frame_dir / "flat_r.fits", frame_type="flat", filter_name="R", level=20000.0)
    _write_frame(frame_dir / "science_r.fits", frame_type="science", filter_name="R")

    no_pool_profile = _profile(flat_pool=False)
    monkeypatch.setattr("robo43.cli.reduce.resolve_profile", lambda **kw: no_pool_profile)

    calib_dir = tmp_path / "calib"
    outdir = tmp_path / "output"

    result = main([
        "reduce",
        str(frame_dir),
        "--calibdir", str(calib_dir),
        "--outdir", str(outdir),
        "--no-overscan",
        "--loglevel", "INFO",
    ])
    assert result == 0

    r_flat_files = list(calib_dir.glob("master_flat_R*.fits"))
    assert len(r_flat_files) == 1
    with fits.open(r_flat_files[0]) as hdul:
        header = hdul[0].header
        assert header["CALFBACK"] is False
        assert "FLATNGT" not in header
        # Only the local frame was combined -- the previous night's 10
        # frames were never pulled in.
        assert header["FLATNFRM"] == 1
