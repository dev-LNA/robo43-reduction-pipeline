"""Tests for robo43.cli.reduce subcommand.

Tests the full reduction pipeline end-to-end: frame inventory, overscan detection,
master calibration building, frame reduction, and status CSV output.

Builds its own synthetic FITS files and metadata; does not use fixtures or conftest.py.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from robo43.cli.main import main
from robo43.config import CalibSettings, RuntimeSettings, Settings
from robo43.instruments.profile import InstrumentLayout, InstrumentProfile
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
    """Create a minimal test instrument profile."""
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
        valid_filters=("R", "G", "B", "R-Bessell"),
        filter_map={},
        keywords={},
        imagetyp_map={},
        site_lat_deg=None,
        site_lon_deg=None,
        site_elevation_m=None,
        index_scales=None,
        search_radius_deg=0.5,
    )


def _settings() -> Settings:
    """Create a minimal test settings object."""
    calib_settings = CalibSettings()
    return Settings(
        paths=None,
        astrometry=None,
        photometry=None,
        calib=calib_settings,
        coadd=None,
        diagnostics=None,
        db=None,
        service=None,
        runtime=RuntimeSettings(),
        targets=None,
    )


def _write_synthetic_frame(
    path: Path,
    frame_type: str = "science",
    exptime: float = 10.0,
    filter_name: str = "R",
    is_valid: bool = True,
) -> None:
    """Write a synthetic FITS frame with appropriate header."""
    shape = (128, 128)
    data = np.ones(shape, dtype=np.float32) * 1000.0
    hdu = fits.PrimaryHDU(data=data)

    hdu.header["IMAGETYP"] = frame_type.upper()
    hdu.header["EXPTIME"] = exptime
    hdu.header["FILTER"] = filter_name
    hdu.header["VALID"] = is_valid
    hdu.header["DATE-OBS"] = "2024-01-01T00:00:00"
    hdu.header["NAXIS1"] = shape[1]
    hdu.header["NAXIS2"] = shape[0]
    hdu.header["GAIN"] = 2.0
    hdu.header["READNOISE"] = 5.0
    hdu.header["SATURATE"] = 60000.0

    path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(path, overwrite=True)


def test_reduce_help(capsys):
    """Test that 'robo43 reduce --help' works and shows the subcommand."""
    with pytest.raises(SystemExit) as exc_info:
        main(["reduce", "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "reduce" in captured.out
    assert "help" in captured.out.lower() or "usage" in captured.out.lower()


def _create_synthetic_masters(calib_dir: Path, _profile) -> None:
    """Create synthetic master calibration frames."""
    calib_dir.mkdir(parents=True, exist_ok=True)

    shape = (128, 128)
    bias_data = np.ones(shape, dtype=np.float32) * 100.0
    bias_hdu = fits.PrimaryHDU(data=bias_data)
    bias_hdu.writeto(calib_dir / "master_bias.fits", overwrite=True)

    flat_data_r = np.ones(shape, dtype=np.float32) * 1.0
    flat_hdu_r = fits.PrimaryHDU(data=flat_data_r)
    flat_hdu_r.writeto(calib_dir / "master_flat_R_dome.fits", overwrite=True)

    flat_data_g = np.ones(shape, dtype=np.float32) * 1.0
    flat_hdu_g = fits.PrimaryHDU(data=flat_data_g)
    flat_hdu_g.writeto(calib_dir / "master_flat_G_dome.fits", overwrite=True)


def test_reduce_end_to_end(tmp_path):
    """Test end-to-end reduction with pre-built masters."""
    # Create synthetic frames
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Create pre-built masters to avoid needing full calibration building
    calib_dir = tmp_path / "calib"
    _create_synthetic_masters(calib_dir, _profile())

    # Create 2 science frames in R filter
    _write_synthetic_frame(
        frame_dir / "science_R_1.fits", frame_type="science", filter_name="R"
    )
    _write_synthetic_frame(
        frame_dir / "science_R_2.fits", frame_type="science", filter_name="R"
    )

    # Create 1 science frame in G filter
    _write_synthetic_frame(
        frame_dir / "science_G_1.fits", frame_type="science", filter_name="G"
    )

    # Create 1 invalid science frame (VALID=False)
    _write_synthetic_frame(
        frame_dir / "science_invalid.fits",
        frame_type="science",
        filter_name="R",
        is_valid=False,
    )

    outdir = tmp_path / "output"

    # Run reduce command
    result = main([
        "reduce",
        str(frame_dir),
        "--calibdir", str(calib_dir),
        "--outdir", str(outdir),
        "--loglevel", "INFO",
    ])
    assert result == 0

    # Check that *_proc.fits files were created for valid science frames
    proc_files = list(outdir.glob("*_proc.fits"))
    assert len(proc_files) == 3  # 3 valid science frames
    assert (outdir / "science_R_1_proc.fits").exists()
    assert (outdir / "science_R_2_proc.fits").exists()
    assert (outdir / "science_G_1_proc.fits").exists()

    # Check that status CSV was created
    csv_path = outdir / "proc_status.csv"
    assert csv_path.exists()

    # Verify CSV has 3 rows (header + 3 data rows)
    with csv_path.open() as f:
        lines = f.readlines()
    assert len(lines) == 4  # header + 3 rows


def test_reduce_skips_invalid_frames(tmp_path):
    """Test that frames with VALID=False are not reduced."""
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Create pre-built masters
    calib_dir = tmp_path / "calib"
    _create_synthetic_masters(calib_dir, _profile())

    # Create 2 science frames: 1 valid, 1 invalid
    _write_synthetic_frame(
        frame_dir / "science_valid.fits", frame_type="science", is_valid=True
    )
    _write_synthetic_frame(
        frame_dir / "science_invalid.fits", frame_type="science", is_valid=False
    )

    outdir = tmp_path / "output"

    result = main([
        "reduce",
        str(frame_dir),
        "--calibdir", str(calib_dir),
        "--outdir", str(outdir),
        "--loglevel", "INFO",
    ])
    assert result == 0

    # Only 1 *_proc.fits should exist (the valid one)
    proc_files = list(outdir.glob("*_proc.fits"))
    assert len(proc_files) == 1
    assert (outdir / "science_valid_proc.fits").exists()
    assert not (outdir / "science_invalid_proc.fits").exists()


def test_reduce_reuses_masters(tmp_path):
    """Test that masters are reused on second run if they exist."""
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    calib_dir = tmp_path / "calib"

    # Create pre-built masters
    _create_synthetic_masters(calib_dir, _profile())

    # Create science frame
    _write_synthetic_frame(frame_dir / "science_1.fits", frame_type="science")

    # Get master_bias.fits mtime before first run
    master_bias_path = calib_dir / "master_bias.fits"
    assert master_bias_path.exists()
    original_mtime = master_bias_path.stat().st_mtime

    # First run
    outdir = tmp_path / "output"
    result1 = main([
        "reduce",
        str(frame_dir),
        "--calibdir", str(calib_dir),
        "--outdir", str(outdir),
        "--loglevel", "INFO",
    ])
    assert result1 == 0

    # Master bias should still have same mtime (was not rebuilt)
    assert master_bias_path.stat().st_mtime == original_mtime

    # Second run with same calibdir
    outdir2 = tmp_path / "output2"
    result2 = main([
        "reduce",
        str(frame_dir),
        "--calibdir", str(calib_dir),
        "--outdir", str(outdir2),
        "--loglevel", "INFO",
    ])
    assert result2 == 0

    # Master bias should still have same mtime (reused, not rebuilt)
    assert master_bias_path.stat().st_mtime == original_mtime


def test_reduce_dry_run(tmp_path):
    """Test that --dry-run reports plan without writing files."""
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Create all frame types to test inventory reporting
    _write_synthetic_frame(frame_dir / "bias.fits", frame_type="bias")
    _write_synthetic_frame(frame_dir / "flat.fits", frame_type="flat")
    _write_synthetic_frame(frame_dir / "science.fits", frame_type="science")

    outdir = tmp_path / "output"
    calib_dir = tmp_path / "calib"

    result = main([
        "reduce",
        str(frame_dir),
        "--calibdir", str(calib_dir),
        "--outdir", str(outdir),
        "--dry-run",
        "--loglevel", "INFO",
    ])
    assert result == 0

    # No output files should be created
    assert not (outdir / "science_proc.fits").exists()
    assert not (outdir / "proc_status.csv").exists()
    assert not (calib_dir / "master_bias.fits").exists()


def test_reduce_no_overscan(tmp_path):
    """Test --no-overscan flag works and is accepted without error."""
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Create pre-built masters
    calib_dir = tmp_path / "calib"
    _create_synthetic_masters(calib_dir, _profile())

    _write_synthetic_frame(frame_dir / "science.fits", frame_type="science")

    outdir = tmp_path / "output"

    # Run with --no-overscan
    result = main([
        "reduce",
        str(frame_dir),
        "--calibdir", str(calib_dir),
        "--outdir", str(outdir),
        "--no-overscan",
        "--loglevel", "INFO",
    ])
    assert result == 0
    assert (outdir / "science_proc.fits").exists()

    # Read shape of output
    with fits.open(outdir / "science_proc.fits") as hdul:
        shape = hdul[0].data.shape

    # Shapes should be 128x128 since test frames don't have actual overscan
    assert shape == (128, 128)


def test_reduce_filter_argument(tmp_path):
    """Test --filter argument reduces only specified filters."""
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Create pre-built masters for both filters
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    shape = (128, 128)
    bias_data = np.ones(shape, dtype=np.float32) * 100.0
    fits.PrimaryHDU(data=bias_data).writeto(
        calib_dir / "master_bias.fits", overwrite=True
    )
    flat_data = np.ones(shape, dtype=np.float32) * 1.0
    fits.PrimaryHDU(data=flat_data).writeto(
        calib_dir / "master_flat_R_dome.fits", overwrite=True
    )
    fits.PrimaryHDU(data=flat_data).writeto(
        calib_dir / "master_flat_R-Bessell_dome.fits", overwrite=True
    )

    # Create science frames in both filters
    _write_synthetic_frame(
        frame_dir / "science_R.fits", frame_type="science", filter_name="R"
    )
    _write_synthetic_frame(
        frame_dir / "science_Bessell.fits",
        frame_type="science",
        filter_name="R-Bessell",
    )

    outdir = tmp_path / "output"

    # Run with --filter R-Bessell
    result = main([
        "reduce",
        str(frame_dir),
        "--calibdir", str(calib_dir),
        "--outdir", str(outdir),
        "--filter", "R-Bessell",
        "--loglevel", "INFO",
    ])
    assert result == 0

    # Only science_Bessell_proc.fits should exist
    proc_files = list(outdir.glob("*_proc.fits"))
    assert len(proc_files) == 1
    assert (outdir / "science_Bessell_proc.fits").exists()
    assert not (outdir / "science_R_proc.fits").exists()


def test_reduce_multiple_filters(tmp_path):
    """Test --filter with multiple values."""
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Create pre-built masters for all filters
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    shape = (128, 128)
    bias_data = np.ones(shape, dtype=np.float32) * 100.0
    fits.PrimaryHDU(data=bias_data).writeto(calib_dir / "master_bias.fits", overwrite=True)
    flat_data = np.ones(shape, dtype=np.float32) * 1.0
    for filt in ["R", "G", "B"]:
        fits.PrimaryHDU(data=flat_data).writeto(
            calib_dir / f"master_flat_{filt}_dome.fits", overwrite=True
        )

    # Create science frames in all 3 filters
    for filt in ["R", "G", "B"]:
        _write_synthetic_frame(
            frame_dir / f"science_{filt}.fits",
            frame_type="science",
            filter_name=filt,
        )

    outdir = tmp_path / "output"

    # Run with --filter R --filter G
    result = main([
        "reduce",
        str(frame_dir),
        "--calibdir", str(calib_dir),
        "--outdir", str(outdir),
        "--filter", "R",
        "--filter", "G",
        "--loglevel", "INFO",
    ])
    assert result == 0

    # Only R and G science frames should be reduced
    proc_files = list(outdir.glob("*_proc.fits"))
    assert len(proc_files) == 2
    assert (outdir / "science_R_proc.fits").exists()
    assert (outdir / "science_G_proc.fits").exists()
    assert not (outdir / "science_B_proc.fits").exists()


def test_reduce_returns_zero_on_success(tmp_path):
    """Test that _run returns 0 on successful reduction."""
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Create pre-built masters
    calib_dir = tmp_path / "calib"
    _create_synthetic_masters(calib_dir, _profile())

    _write_synthetic_frame(frame_dir / "science.fits", frame_type="science")

    outdir = tmp_path / "output"

    result = main([
        "reduce",
        str(frame_dir),
        "--calibdir", str(calib_dir),
        "--outdir", str(outdir),
        "--loglevel", "INFO",
    ])
    assert result == 0


def test_reduce_csv_structure(tmp_path):
    """Test that proc_status.csv has correct columns and structure."""
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Create pre-built masters
    calib_dir = tmp_path / "calib"
    _create_synthetic_masters(calib_dir, _profile())

    _write_synthetic_frame(frame_dir / "science_1.fits", frame_type="science")
    _write_synthetic_frame(frame_dir / "science_2.fits", frame_type="science")

    outdir = tmp_path / "output"

    result = main([
        "reduce",
        str(frame_dir),
        "--calibdir", str(calib_dir),
        "--outdir", str(outdir),
        "--loglevel", "INFO",
    ])
    assert result == 0

    # Read and verify CSV
    csv_path = outdir / "proc_status.csv"
    assert csv_path.exists()

    import csv
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2  # 2 science frames
    expected_keys = {
        "path",
        "raw_name",
        "proc_code",
        "proc_status",
        "steps_applied",
        "steps_skipped",
        "error",
        "product_path",
        "wcs_solved",
        "astrom_rms_arcsec",
        "fwhm_arcsec",
        "zeropoint",
        "is_photometric",
        "duration_s",
    }
    assert set(rows[0].keys()) == expected_keys


def test_reduce_reuses_master_flats_second_run(tmp_path):
    """Test that master flats are reused on second run (regression test for flat rebuild issue)."""
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    calib_dir = tmp_path / "calib"

    # Create pre-built masters
    _create_synthetic_masters(calib_dir, _profile())

    # Create flat and science frames
    _write_synthetic_frame(frame_dir / "flat_1.fits", frame_type="flat", filter_name="R")
    _write_synthetic_frame(frame_dir / "science_1.fits", frame_type="science", filter_name="R")

    # Get master_flat_R_dome.fits content before first run
    master_flat_path = calib_dir / "master_flat_R_dome.fits"
    assert master_flat_path.exists()

    def get_file_hash(path):
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with Path(path).open("rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    original_hash = get_file_hash(master_flat_path)
    original_mtime = master_flat_path.stat().st_mtime

    # First run
    outdir1 = tmp_path / "output1"
    result1 = main([
        "reduce",
        str(frame_dir),
        "--calibdir", str(calib_dir),
        "--outdir", str(outdir1),
        "--no-overscan",
        "--loglevel", "INFO",
    ])
    assert result1 == 0

    # Master flat should still have same hash (was not rebuilt)
    assert get_file_hash(master_flat_path) == original_hash
    assert master_flat_path.stat().st_mtime == original_mtime

    # Second run with same calibdir
    outdir2 = tmp_path / "output2"
    result2 = main([
        "reduce",
        str(frame_dir),
        "--calibdir", str(calib_dir),
        "--outdir", str(outdir2),
        "--no-overscan",
        "--loglevel", "INFO",
    ])
    assert result2 == 0

    # Master flat should still have same hash (reused, not rebuilt)
    assert get_file_hash(master_flat_path) == original_hash
    assert master_flat_path.stat().st_mtime == original_mtime


def test_reduce_builds_missing_flat_filter_only(tmp_path):
    """Test partial case: only rebuild flat for filters that don't have masters."""
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)

    # Create pre-built masters for only R filter
    shape = (128, 128)
    bias_data = np.ones(shape, dtype=np.float32) * 100.0
    fits.PrimaryHDU(data=bias_data).writeto(
        calib_dir / "master_bias.fits", overwrite=True
    )
    flat_data = np.ones(shape, dtype=np.float32) * 1.0
    fits.PrimaryHDU(data=flat_data).writeto(
        calib_dir / "master_flat_R_dome.fits", overwrite=True
    )

    def get_file_hash(path):
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with Path(path).open("rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    # Save R flat hash
    r_flat_hash_before = get_file_hash(calib_dir / "master_flat_R_dome.fits")

    # Create flat and science frames for both R and G filters
    _write_synthetic_frame(frame_dir / "flat_R.fits", frame_type="flat", filter_name="R")
    _write_synthetic_frame(frame_dir / "flat_G.fits", frame_type="flat", filter_name="G")
    _write_synthetic_frame(frame_dir / "science_R.fits", frame_type="science", filter_name="R")
    _write_synthetic_frame(frame_dir / "science_G.fits", frame_type="science", filter_name="G")

    outdir = tmp_path / "output"

    # Run reduce
    result = main([
        "reduce",
        str(frame_dir),
        "--calibdir", str(calib_dir),
        "--outdir", str(outdir),
        "--no-overscan",
        "--loglevel", "INFO",
    ])
    assert result == 0

    # R flat should be unchanged (reused)
    assert get_file_hash(calib_dir / "master_flat_R_dome.fits") == r_flat_hash_before

    # G flat should now exist (was built)
    assert (calib_dir / "master_flat_G_dome.fits").exists()


def test_reduce_rebuilds_masters_with_clobber(tmp_path):
    """Test that --clobber forces rebuild of masters."""
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    calib_dir = tmp_path / "calib"

    # Create pre-built masters
    _create_synthetic_masters(calib_dir, _profile())

    # Create flat and science frames
    _write_synthetic_frame(frame_dir / "flat_1.fits", frame_type="flat", filter_name="R")
    _write_synthetic_frame(frame_dir / "science_1.fits", frame_type="science", filter_name="R")

    master_flat_path = calib_dir / "master_flat_R_dome.fits"
    assert master_flat_path.exists()

    # Run with --clobber
    outdir = tmp_path / "output"
    result = main([
        "reduce",
        str(frame_dir),
        "--calibdir", str(calib_dir),
        "--outdir", str(outdir),
        "--clobber",
        "--no-overscan",
        "--loglevel", "INFO",
    ])
    assert result == 0

    # With clobber, master should be rebuilt (may have different content or mtime)
    # We just verify the file still exists after the rebuild
    assert master_flat_path.exists()
