"""Tests for robo43.calib.fallback (multi-night calibration fallback) and its
wiring into robo43.cli.reduce.

Builds its own synthetic FITS files under tmp_path in a root/YYYYMMDD/ layout;
does not use fixtures or conftest.py, matching the style of the other calib
and cli test modules in this suite.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from robo43.calib.fallback import (
    collect_fallback_frames,
    dominant_camera_config,
    parse_night_dirname,
)
from robo43.cli.main import main
from robo43.config import CalibSettings, RuntimeSettings, Settings
from robo43.instruments.profile import InstrumentLayout, InstrumentProfile
from robo43.io.headers import read_meta
from robo43.logging_setup import ROOT_LOGGER_NAME


@pytest.fixture(autouse=True)
def _reset_logging():
    """Clean up logging state after each test to avoid pollution."""
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
        valid_filters=("R", "G", "B"),
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
    """Write a synthetic raw FITS frame with a controllable camera config."""
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


# --------------------------------------------------------------------------
# parse_night_dirname
# --------------------------------------------------------------------------


def test_parse_night_dirname_valid():
    from datetime import date

    assert parse_night_dirname("20240110") == date(2024, 1, 10)


@pytest.mark.parametrize("name", ["2024011", "202401100", "abcdefgh", "2024-01-10", "frames"])
def test_parse_night_dirname_invalid(name):
    assert parse_night_dirname(name) is None


def test_parse_night_dirname_rejects_invalid_calendar_date():
    assert parse_night_dirname("20250230") is None  # Feb 30 doesn't exist


# --------------------------------------------------------------------------
# dominant_camera_config
# --------------------------------------------------------------------------


def test_dominant_camera_config_empty():
    assert dominant_camera_config([]) is None


def test_dominant_camera_config_majority(tmp_path):
    profile = _profile()
    settings = _settings()
    _write_frame(tmp_path / "a.fits", frame_type="bias", gain=10.0, binning=2)
    _write_frame(tmp_path / "b.fits", frame_type="bias", gain=10.0, binning=2)
    _write_frame(tmp_path / "c.fits", frame_type="bias", gain=30.0, binning=2)
    metas = [
        read_meta(tmp_path / n, profile, settings=settings) for n in ("a.fits", "b.fits", "c.fits")
    ]
    config = dominant_camera_config(metas)
    assert config == ("High Gain Mode 16BIT", 10.0, 2, 64, 64)


# --------------------------------------------------------------------------
# collect_fallback_frames
# --------------------------------------------------------------------------


def _target_config():
    return ("High Gain Mode 16BIT", 10.0, 2, 64, 64)


def test_collect_fallback_frames_single_nearest_night_sufficient(tmp_path):
    root = tmp_path / "T80S"
    night_dir = root / "20240110"
    night_dir.mkdir(parents=True)

    # Nearest earlier night has enough bias frames alone.
    for i in range(3):
        _write_frame(root / "20240109" / f"bias_{i}.fits", frame_type="bias")
    # A night further back also has bias frames; must not be used or mixed in.
    for i in range(3):
        _write_frame(root / "20240105" / f"bias_{i}.fits", frame_type="bias")

    from datetime import date

    result = collect_fallback_frames(
        night_dir,
        date(2024, 1, 10),
        frame_type="bias",
        target_config=_target_config(),
        min_count=3,
        max_nights=30,
        profile=_profile(),
        settings=_settings(),
    )
    assert result.satisfied is True
    assert result.source_nights == ("20240109",)
    assert len(result.metas) == 3


def test_collect_fallback_frames_accumulates_when_no_single_night_suffices(tmp_path):
    root = tmp_path / "T80S"
    night_dir = root / "20240110"
    night_dir.mkdir(parents=True)

    _write_frame(root / "20240109" / "bias_0.fits", frame_type="bias")
    _write_frame(root / "20240109" / "bias_1.fits", frame_type="bias")
    _write_frame(root / "20240108" / "bias_0.fits", frame_type="bias")
    _write_frame(root / "20240108" / "bias_1.fits", frame_type="bias")

    from datetime import date

    result = collect_fallback_frames(
        night_dir,
        date(2024, 1, 10),
        frame_type="bias",
        target_config=_target_config(),
        min_count=3,
        max_nights=30,
        profile=_profile(),
        settings=_settings(),
    )
    assert result.satisfied is True
    assert result.source_nights == ("20240109", "20240108")
    assert len(result.metas) == 4


def test_collect_fallback_frames_ignores_later_nights(tmp_path):
    root = tmp_path / "T80S"
    night_dir = root / "20240110"
    night_dir.mkdir(parents=True)

    # A LATER night has plenty of bias frames; must never be used.
    for i in range(5):
        _write_frame(root / "20240111" / f"bias_{i}.fits", frame_type="bias")

    from datetime import date

    result = collect_fallback_frames(
        night_dir,
        date(2024, 1, 10),
        frame_type="bias",
        target_config=_target_config(),
        min_count=3,
        max_nights=30,
        profile=_profile(),
        settings=_settings(),
    )
    assert result.satisfied is False
    assert result.metas == ()
    assert result.source_nights == ()


def test_collect_fallback_frames_honours_window_limit(tmp_path):
    root = tmp_path / "T80S"
    night_dir = root / "20240110"
    night_dir.mkdir(parents=True)

    # 9 days back: outside a 5-day window.
    for i in range(3):
        _write_frame(root / "20240101" / f"bias_{i}.fits", frame_type="bias")

    from datetime import date

    result = collect_fallback_frames(
        night_dir,
        date(2024, 1, 10),
        frame_type="bias",
        target_config=_target_config(),
        min_count=3,
        max_nights=5,
        profile=_profile(),
        settings=_settings(),
    )
    assert result.satisfied is False
    assert result.metas == ()


def test_collect_fallback_frames_skips_config_mismatch(tmp_path):
    root = tmp_path / "T80S"
    night_dir = root / "20240110"
    night_dir.mkdir(parents=True)

    # Different binning: must not match the target config.
    for i in range(3):
        _write_frame(root / "20240109" / f"bias_{i}.fits", frame_type="bias", binning=1)

    from datetime import date

    result = collect_fallback_frames(
        night_dir,
        date(2024, 1, 10),
        frame_type="bias",
        target_config=_target_config(),  # binning=2
        min_count=3,
        max_nights=30,
        profile=_profile(),
        settings=_settings(),
    )
    assert result.satisfied is False
    assert result.metas == ()


def test_collect_fallback_frames_flat_matches_filter_only(tmp_path):
    root = tmp_path / "T80S"
    night_dir = root / "20240110"
    night_dir.mkdir(parents=True)

    for i in range(3):
        _write_frame(root / "20240109" / f"flat_g_{i}.fits", frame_type="flat", filter_name="G")
    for i in range(3):
        _write_frame(root / "20240109" / f"flat_r_{i}.fits", frame_type="flat", filter_name="R")

    from datetime import date

    result = collect_fallback_frames(
        night_dir,
        date(2024, 1, 10),
        frame_type="flat",
        target_config=_target_config(),
        min_count=3,
        max_nights=30,
        profile=_profile(),
        settings=_settings(),
        filter_name="R",
    )
    assert result.satisfied is True
    assert len(result.metas) == 3
    assert all(m.filter == "R" for m in result.metas)


# --------------------------------------------------------------------------
# CLI integration: robo43 reduce with --calibdir pointed at a night directory
# --------------------------------------------------------------------------


def test_reduce_borrows_bias_from_previous_night(tmp_path):
    root = tmp_path / "T80S"
    frame_dir = root / "20240110"
    frame_dir.mkdir(parents=True)

    # Previous night: bias frames only.
    for i in range(3):
        _write_frame(root / "20240109" / f"bias_{i}.fits", frame_type="bias")

    # Target night: science only, no bias.
    _write_frame(frame_dir / "science_1.fits", frame_type="science", filter_name="R")

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

    master_bias_path = calib_dir / "master_bias.fits"
    assert master_bias_path.exists()
    with fits.open(master_bias_path) as hdul:
        header = hdul[0].header
        assert header["CALFBACK"] is True
        assert header["BIASNGT"] == "20240109"

    proc_path = outdir / "science_1_proc.fits"
    assert proc_path.exists()
    with fits.open(proc_path) as hdul:
        sci_header = hdul["SCI"].header
        assert sci_header["CALFBACK"] is True
        assert sci_header["BIASNGT"] == "20240109"


def test_reduce_no_calib_fallback_flag_disables_borrowing(tmp_path):
    root = tmp_path / "T80S"
    frame_dir = root / "20240110"
    frame_dir.mkdir(parents=True)

    for i in range(3):
        _write_frame(root / "20240109" / f"bias_{i}.fits", frame_type="bias")

    _write_frame(frame_dir / "science_1.fits", frame_type="science", filter_name="R")

    calib_dir = tmp_path / "calib"
    outdir = tmp_path / "output"

    result = main([
        "reduce",
        str(frame_dir),
        "--calibdir", str(calib_dir),
        "--outdir", str(outdir),
        "--no-overscan",
        "--no-calib-fallback",
        "--loglevel", "INFO",
    ])
    assert result == 0

    # No fallback: master bias is never built (no local bias frames either).
    assert not (calib_dir / "master_bias.fits").exists()


def test_reduce_borrows_flat_for_missing_filter_only(tmp_path):
    root = tmp_path / "T80S"
    frame_dir = root / "20240110"
    frame_dir.mkdir(parents=True)

    # Previous night: flats for filter G only. Level well above the bias
    # level (1000.0) so bias subtraction does not zero the flat out.
    for i in range(5):
        _write_frame(
            root / "20240109" / f"flat_g_{i}.fits",
            frame_type="flat", filter_name="G", level=5000.0,
        )

    # Target night: local bias, local flat for R, science in both R and G.
    for i in range(3):
        _write_frame(frame_dir / f"bias_{i}.fits", frame_type="bias")
    _write_frame(frame_dir / "flat_r.fits", frame_type="flat", filter_name="R", level=5000.0)
    _write_frame(frame_dir / "science_r.fits", frame_type="science", filter_name="R")
    _write_frame(frame_dir / "science_g.fits", frame_type="science", filter_name="G")

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

    g_flat_files = list(calib_dir.glob("master_flat_G*.fits"))
    assert len(g_flat_files) == 1
    with fits.open(g_flat_files[0]) as hdul:
        header = hdul[0].header
        assert header["CALFBACK"] is True
        assert header["FLATNGT"] == "20240109"

    r_flat_files = list(calib_dir.glob("master_flat_R*.fits"))
    assert len(r_flat_files) == 1
    with fits.open(r_flat_files[0]) as hdul:
        assert hdul[0].header["CALFBACK"] is False
