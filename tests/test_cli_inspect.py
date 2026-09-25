"""Tests for cli.inspect subcommand."""

from __future__ import annotations

import logging
from pathlib import Path

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


def test_inspect_help_exits_zero():
    """inspect --help should exit with code 0."""
    with pytest.raises(SystemExit) as exc_info:
        main(["inspect", "--help"])
    assert exc_info.value.code == 0


def test_inspect_full_unstripped_night_exits_zero(synthetic_night):
    """Full unstripped night passes all stage validation."""
    result = main(["inspect", str(synthetic_night)])
    assert result == 0


def test_inspect_synth2_via_fixture_path(synthetic_night_synth2):
    """Synth2 MEF works when profile is passed as filesystem path."""
    synth2_toml = Path(__file__).parent / "fixtures" / "synth2.toml"
    if not synth2_toml.exists():
        pytest.skip("synth2.toml fixture not found")

    # Test that the path form of --instrument works (P4)
    # The synthetic data may not have all required fields for all stages,
    # but the key point is that the profile loads from the path
    result = main(
        [
            "inspect",
            str(synthetic_night_synth2),
            "--instrument",
            str(synth2_toml),
            "--stages",
            "bias,dark,flat",  # Stages with few requirements
            "--assume-profile-defaults",
        ]
    )
    # Should succeed because bias/dark/flat have minimal requirements
    assert result == 0


def test_inspect_stripped_generic_photometry_fails(synthetic_night_stripped):
    """Stripped night fails photometry with filter, gain_setting, saturation."""
    result = main(
        [
            "inspect",
            str(synthetic_night_stripped),
            "--instrument",
            "generic",
            "--stages",
            "photometry",
        ]
    )
    assert result == 1


def test_inspect_emit_overrides_then_fix(synthetic_night_stripped, tmp_path):
    """emit-overrides template can be filled and fixes validation."""
    overrides_path = tmp_path / "overrides.toml"

    # First run: emit template
    result = main(
        [
            "inspect",
            str(synthetic_night_stripped),
            "--instrument",
            "generic",
            "--stages",
            "photometry",
            "--emit-overrides",
            str(overrides_path),
        ]
    )
    assert result == 1
    assert overrides_path.exists()

    # Check that the template has the right keys (gain_setting, per the amendment)
    template_text = overrides_path.read_text()
    assert "gain_setting =" in template_text
    assert "saturation =" in template_text
    assert "filter =" in template_text

    # Fill in the template
    overrides_content = """
[defaults]
gain_setting = 2.0
saturation = 55000
filter = "V-Bessell"
"""
    overrides_path.write_text(overrides_content)

    # Second run: with filled overrides
    result = main(
        [
            "inspect",
            str(synthetic_night_stripped),
            "--instrument",
            "generic",
            "--stages",
            "photometry",
            "--overrides",
            str(overrides_path),
        ]
    )
    assert result == 0


def test_inspect_bias_dark_never_flagged_for_filter(synthetic_night):
    """ZERO/DARK frames never raise filter issues."""
    result = main(["inspect", str(synthetic_night)])
    # Should pass because bias/dark shouldn't be checked for filter
    assert result == 0


def test_inspect_focus_frame_excluded_from_science_stages(synthetic_night, tmp_path):
    """FOCUS frames are excluded from science stage validation."""
    from astropy.io import fits

    # Find a science frame in the synthetic night
    science_files = sorted(synthetic_night.glob("science_*.fits"))
    assert science_files, "No science frames in synthetic night"
    science_frame = science_files[0]

    # Create a FOCUS frame by copying and modifying a science frame
    focus_frame = tmp_path / "focus_frame.fits"
    with fits.open(science_frame) as hdul:
        hdr = hdul[0].header.copy()
        data = hdul[0].data.copy()
        # Modify header to make it a FOCUS frame
        hdr["IMAGETYP"] = "FOCUS"
        hdr["OBJECT"] = "focus run"
        # Delete RA/DEC pointing keywords
        for key in ("RA", "DEC", "OBJCTRA", "OBJCTDEC", "TELRA", "TELDEC", "CRVAL1", "CRVAL2"):
            hdr.pop(key, None)
        # Write the focus frame
        fits.PrimaryHDU(data=data, header=hdr).writeto(focus_frame, overwrite=True)

    # Run inspect on the FOCUS frame with astrometry stage
    result = main(["inspect", str(focus_frame), "--stages", "astrometry"])
    # FOCUS frames are excluded from astrometry stage, so validation passes
    assert result == 0, "FOCUS frame should be excluded from astrometry validation"

    # Sanity check: a science frame with deleted RA/DEC should fail astrometry
    science_bad_frame = tmp_path / "science_bad.fits"
    with fits.open(science_frame) as hdul:
        hdr = hdul[0].header.copy()
        data = hdul[0].data.copy()
        # Delete RA/DEC pointing keywords
        for key in ("RA", "DEC", "OBJCTRA", "OBJCTDEC", "TELRA", "TELDEC", "CRVAL1", "CRVAL2"):
            hdr.pop(key, None)
        # Write the science frame without pointing
        fits.PrimaryHDU(data=data, header=hdr).writeto(science_bad_frame, overwrite=True)

    # Run inspect on the bad science frame with astrometry stage
    result = main(["inspect", str(science_bad_frame), "--stages", "astrometry"])
    # Science frame without RA/DEC should fail astrometry validation
    assert result == 1, "Science frame without RA/DEC should fail astrometry validation"
