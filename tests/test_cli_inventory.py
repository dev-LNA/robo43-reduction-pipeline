"""Tests for cli.inventory subcommand."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

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


def test_inventory_help_exits_zero():
    """inventory --help should exit with code 0."""
    with pytest.raises(SystemExit) as exc_info:
        main(["inventory", "--help"])
    assert exc_info.value.code == 0


def test_inventory_writes_csv(synthetic_night, tmp_path):
    """inventory writes CSV output."""
    output_path = tmp_path / "inventory.csv"
    result = main(
        [
            "inventory",
            str(synthetic_night),
            "-o",
            str(output_path),
        ]
    )
    assert result == 0
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert len(df) > 0


def test_inventory_frame_type_filter_cli(synthetic_night, tmp_path):
    """inventory --frame-type filters results."""
    output_path = tmp_path / "inventory.csv"
    result = main(
        [
            "inventory",
            str(synthetic_night),
            "-o",
            str(output_path),
            "--frame-type",
            "FLAT",
        ]
    )
    assert result == 0
    df = pd.read_csv(output_path)
    # All rows should be FLAT
    assert all(df["imagetyp"] == "FLAT")


def test_inventory_excludes_valid_false_by_default(synthetic_night, tmp_path, caplog):
    """inventory excludes frames with VALID=False by default, includes with --include-invalid."""
    import shutil
    from pathlib import Path

    from astropy.io import fits

    # Get the first FITS file from synthetic_night
    fits_files = sorted(synthetic_night.glob("*.fits"))
    assert len(fits_files) > 0, "synthetic_night has no FITS files"

    # Create a subdirectory for the modified frame
    test_dir = tmp_path / "test_frames"
    test_dir.mkdir()

    # Copy the first frame to test_dir and set VALID=False
    source_file = fits_files[0]
    test_file = test_dir / Path(source_file).name
    shutil.copy2(source_file, test_file)

    # Set VALID=False in the primary header
    with fits.open(test_file, mode="update") as hdul:
        hdul[0].header["VALID"] = False

    # Run inventory without --include-invalid
    output_path_1 = tmp_path / "inventory_exclude.csv"
    caplog.set_level(logging.INFO, logger=ROOT_LOGGER_NAME)
    result1 = main(
        [
            "inventory",
            str(synthetic_night),
            str(test_file),
            "-o",
            str(output_path_1),
        ]
    )
    assert result1 == 0

    df1 = pd.read_csv(output_path_1)
    # The frame with VALID=False should NOT be in the CSV
    assert str(test_file) not in df1["path"].values
    # The first run should have 33 rows (from synthetic_night): 5 bias, 4 dark, 5 flat, 3 science
    # (the invalid frame is excluded by default)
    assert len(df1) == 33

    # Run inventory WITH --include-invalid
    output_path_2 = tmp_path / "inventory_include.csv"
    caplog.clear()
    result2 = main(
        [
            "inventory",
            str(synthetic_night),
            str(test_file),
            "-o",
            str(output_path_2),
            "--include-invalid",
        ]
    )
    assert result2 == 0

    df2 = pd.read_csv(output_path_2)
    # The frame with VALID=False SHOULD be in the CSV with --include-invalid
    assert str(test_file) in df2["path"].values
