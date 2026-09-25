"""Tests for cli.fixheader subcommand."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest
from astropy.io import fits

from robo43.cli.main import main
from robo43.logging_setup import ROOT_LOGGER_NAME


@pytest.fixture(autouse=True)
def _reset_logging():
    """Reset logging state between tests."""
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


def test_fixheader_help_exits_zero():
    """fixheader --help should exit with code 0."""
    with pytest.raises(SystemExit) as exc_info:
        main(["fixheader", "--help"])
    assert exc_info.value.code == 0


def test_fixheader_repairs_wrong_imagetyp_and_backs_up():
    """fixheader repairs IMAGETYP from OBJECT and creates .bak."""
    mini_sample = Path("/home/herpich/Documents/data/ROBO43/mini_sample")
    if not mini_sample.exists():
        pytest.skip("mini_sample not found")

    # Find a flat frame
    flat_file = next(iter(mini_sample.glob("*DomeFlat*.fits")))
    # Create a temp copy
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_file = tmp_path / flat_file.name
        shutil.copy2(flat_file, test_file)

        # Corrupt the IMAGETYP to something generic
        with fits.open(test_file, mode="update") as hdul:
            hdul[0].header["IMAGETYP"] = "Light Frame"

        # Run fixheader
        result = main(["fixheader", str(test_file)])
        assert result == 0

        # Check that IMAGETYP was repaired to FLAT
        with fits.open(test_file, mode="readonly") as hdul:
            assert hdul[0].header["IMAGETYP"] == "FLAT"

        # Check that .bak was created
        backup_path = test_file.with_name(test_file.name + ".bak")
        assert backup_path.exists()


def test_fixheader_skips_filter_rewrite_on_bias():
    """fixheader does not rewrite FILTER on ZERO frames."""
    mini_sample = Path("/home/herpich/Documents/data/ROBO43/mini_sample")
    if not mini_sample.exists():
        pytest.skip("mini_sample not found")

    # Find a bias frame
    bias_file = next(iter(mini_sample.glob("*bias*.fits")))
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_file = tmp_path / bias_file.name
        shutil.copy2(bias_file, test_file)

        # Get original FILTER
        with fits.open(test_file, mode="readonly") as hdul:
            original_filter = hdul[0].header.get("FILTER", "")

        # Run fixheader
        result = main(["fixheader", str(test_file)])
        assert result == 0

        # Check that FILTER is unchanged
        with fits.open(test_file, mode="readonly") as hdul:
            assert hdul[0].header.get("FILTER", "") == original_filter


def test_fixheader_dry_run_is_true_no_op(tmp_path):
    """--dry-run does not modify the file."""
    mini_sample = Path("/home/herpich/Documents/data/ROBO43/mini_sample")
    if not mini_sample.exists():
        pytest.skip("mini_sample not found")

    # Find a flat frame
    flat_file = next(iter(mini_sample.glob("*DomeFlat*.fits")))

    # Create a temp copy
    test_file = tmp_path / flat_file.name
    shutil.copy2(flat_file, test_file)

    # Corrupt the IMAGETYP
    with fits.open(test_file, mode="update") as hdul:
        hdul[0].header["IMAGETYP"] = "Light Frame"

    # Run fixheader with --dry-run
    result = main(["fixheader", str(test_file), "--dry-run"])
    assert result == 0

    # Check that no .bak was created (file was not modified)
    backup_path = test_file.with_name(test_file.name + ".bak")
    assert not backup_path.exists()


def test_fixheader_no_changes_needed_skips_backup(tmp_path):
    """File with canonical IMAGETYP/FILTER is skipped without backup."""
    mini_sample = Path("/home/herpich/Documents/data/ROBO43/mini_sample")
    if not mini_sample.exists():
        pytest.skip("mini_sample not found")

    # Find a flat frame (which should already have correct IMAGETYP)
    flat_file = next(iter(mini_sample.glob("*DomeFlat*.fits")))

    # Create a temp copy
    test_file = tmp_path / flat_file.name
    shutil.copy2(flat_file, test_file)

    # Run fixheader
    result = main(["fixheader", str(test_file)])
    assert result == 0

    # Check that no .bak was created (because no changes were made)
    backup_path = test_file.with_name(test_file.name + ".bak")
    assert not backup_path.exists()


def test_fixheader_missing_object_skipped(tmp_path):
    """File with missing/empty OBJECT is skipped."""
    mini_sample = Path("/home/herpich/Documents/data/ROBO43/mini_sample")
    if not mini_sample.exists():
        pytest.skip("mini_sample not found")

    flat_file = next(iter(mini_sample.glob("*DomeFlat*.fits")))

    # Create a temp copy
    test_file = tmp_path / flat_file.name
    shutil.copy2(flat_file, test_file)

    # Remove OBJECT
    with fits.open(test_file, mode="update") as hdul:
        if "OBJECT" in hdul[0].header:
            del hdul[0].header["OBJECT"]

    # Run fixheader
    result = main(["fixheader", str(test_file)])
    # Should return 0 because skipping is not a failure
    assert result == 0


def test_fixheader_dry_run_prints_would_change_and_no_sha_change(synthetic_night, tmp_path, capsys):
    """fixheader --dry-run prints 'would change' and leaves file unchanged."""
    import hashlib

    # Get the first FITS file from synthetic_night
    fits_files = sorted(synthetic_night.glob("*.fits"))
    assert len(fits_files) > 0, "synthetic_night has no FITS files"

    # Create a subdirectory to avoid copying to the same location
    test_dir = tmp_path / "test_frames"
    test_dir.mkdir()

    source_file = fits_files[0]
    test_file = test_dir / source_file.name
    shutil.copy2(source_file, test_file)

    # Set IMAGETYP to "Light Frame" and OBJECT to "DomeFlat" (which maps to FLAT)
    with fits.open(test_file, mode="update") as hdul:
        hdul[0].header["IMAGETYP"] = "Light Frame"
        hdul[0].header["OBJECT"] = "DomeFlat"

    # Calculate SHA256 of original file
    original_sha256 = hashlib.sha256(test_file.read_bytes()).hexdigest()

    # Run fixheader with --dry-run
    result = main(["fixheader", str(test_file), "--dry-run"])
    assert result == 0

    # Check that capsys contains "would change (dry-run)"
    captured = capsys.readouterr()
    assert "would change (dry-run)" in captured.out or "would change (dry-run)" in captured.err

    # Calculate SHA256 of file after --dry-run
    after_sha256 = hashlib.sha256(test_file.read_bytes()).hexdigest()

    # File should be unchanged
    assert original_sha256 == after_sha256
