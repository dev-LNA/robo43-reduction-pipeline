"""Tests for robo43.io.fits_io.read_header's empty-primary fallback.

T80S raw ``.fits.fz`` files have an empty primary HDU (NAXIS=0); the
instrument-match keywords (TELESCOP/INSTRUME) live only on the first image
extension. Before this fix, ``read_header(path, profile=None)`` -- used by
``robo43 reduce``'s own profile auto-detection, before any profile is known
-- only ever looked at HDU 0, so those keywords were invisible and
``resolve_profile`` fell through to the 'generic' profile unless
``--instrument t80s`` was given explicitly.

Builds its own tiny in-memory FITS files; does not use fixtures or
conftest.py.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from robo43.instruments.profile import resolve_profile
from robo43.io.fits_io import read_header


def _write_fpacked_style_file(path) -> None:
    """A minimal stand-in for a real fpacked file: empty primary HDU plus a
    compressed image extension carrying the instrument-match keywords."""
    primary = fits.PrimaryHDU()
    assert primary.header.get("NAXIS", 0) == 0

    data = np.zeros((4, 4), dtype=np.int16)
    comp = fits.CompImageHDU(data=data, name="SCI")
    comp.header["TELESCOP"] = "T80"
    comp.header["INSTRUME"] = "T80Cam"
    comp.header["NAXIS1"] = 4
    comp.header["NAXIS2"] = 4

    fits.HDUList([primary, comp]).writeto(path, overwrite=True)


def test_read_header_merges_first_image_extension_when_primary_is_empty(tmp_path):
    path = tmp_path / "fake_t80s.fits.fz"
    _write_fpacked_style_file(path)

    header = read_header(path, profile=None)

    assert header["TELESCOP"] == "T80"
    assert header["INSTRUME"] == "T80Cam"


def test_profile_auto_detection_resolves_t80s_from_fpacked_style_file(tmp_path):
    """The real bug: robo43 reduce's own auto-detection path
    (``resolve_profile(header=read_header(path, profile=None))``) must
    resolve 't80s', not fall through to 'generic'."""
    path = tmp_path / "fake_t80s.fits.fz"
    _write_fpacked_style_file(path)

    header = read_header(path, profile=None)
    profile = resolve_profile(header=header)

    assert profile.name == "t80s"


def test_read_header_untouched_when_primary_already_has_data(tmp_path):
    """Regression guard: a normal single-HDU file (the common case) is
    unaffected by the empty-primary fallback."""
    data = np.ones((4, 4), dtype=np.float32)
    hdu = fits.PrimaryHDU(data=data)
    hdu.header["TELESCOP"] = "SOMETHING-ELSE"
    path = tmp_path / "normal.fits"
    hdu.writeto(path, overwrite=True)

    header = read_header(path, profile=None)

    assert header["TELESCOP"] == "SOMETHING-ELSE"
    assert header["NAXIS"] == 2


def test_read_header_merges_first_image_extension_when_primary_naxis1_is_zero(tmp_path):
    """Some T80S raw files declare NAXIS=1 with NAXIS1=0 (empty primary) rather
    than NAXIS=0. Both forms mean no pixel data, and both should trigger the
    fallback to merge keywords from the first image extension."""
    path = tmp_path / "t80s_naxis1_zero.fits"

    # Create standard FITS file and write it.
    primary = fits.PrimaryHDU()
    data = np.zeros((4, 4), dtype=np.int16)
    comp = fits.CompImageHDU(data=data, name="SCI")
    comp.header["TELESCOP"] = "T80"
    comp.header["INSTRUME"] = "T80Cam"

    fits.HDUList([primary, comp]).writeto(path, overwrite=True)

    # Patch the primary header block to insert NAXIS1=0 and change NAXIS to 1.
    # The primary header block is 2880 bytes = 36 cards of 80 bytes each.
    # Original structure (cards 0-5, rest is blank):
    #   0: SIMPLE, 1: BITPIX, 2: NAXIS=0, 3: EXTEND, 4: END
    # New structure:
    #   0: SIMPLE, 1: BITPIX, 2: NAXIS=1, 3: NAXIS1=0, 4: EXTEND, 5: END
    with open(path, "r+b") as f:
        content = bytearray(f.read())

        # Extract the current cards (80 bytes each, all within first 2880 bytes).
        simple_card = content[0:80]
        bitpix_card = content[80:160]
        naxis_card = content[160:240]
        extend_card = content[240:320]
        end_card = content[320:400]
        blank_padding = content[400:2880]

        # Modify NAXIS card: change NAXIS=0 to NAXIS=1.
        naxis_card = bytearray(naxis_card)
        naxis_card[29:30] = b"1"  # Position 29 is where the 0 is in "NAXIS   =                    0"
        naxis_card = bytes(naxis_card)

        # Create NAXIS1=0 card.
        naxis1_card = b"NAXIS1  =                    0 "
        naxis1_card = naxis1_card.ljust(80)

        # Reconstruct the primary header block with new structure (must be exactly 2880 bytes).
        # Original: 4 cards + 1 end card + 31 blank cards = 36 cards = 2880 bytes
        # New: 5 cards + 1 end card + 30 blank cards = 36 cards = 2880 bytes
        # So we need to reduce blank padding by 80 bytes (one card).
        reduced_blank_padding = blank_padding[80:]  # Remove first 80 bytes from blank padding
        new_header = (
            simple_card
            + bitpix_card
            + naxis_card
            + naxis1_card
            + extend_card
            + end_card
            + reduced_blank_padding
        )

        content[0:2880] = new_header
        f.seek(0)
        f.write(content)
        f.truncate()

    # Verify the patch.
    with fits.open(path) as hdul:
        assert hdul[0].header["NAXIS"] == 1
        assert hdul[0].header["NAXIS1"] == 0

    header = read_header(path, profile=None)
    profile = resolve_profile(header=header)

    assert header["TELESCOP"] == "T80"
    assert header["INSTRUME"] == "T80Cam"
    assert profile.name == "t80s"
