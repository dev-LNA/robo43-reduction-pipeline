"""Regression tests for FOCUS frame classification (P1/P2)."""

from __future__ import annotations

from astropy.io import fits

from robo43.instruments.profile import resolve_profile
from robo43.io.headers import classify_frame_type, classify_from_object


class TestClassifyFromObjectFocus:
    """Unit tests for classify_from_object FOCUS case."""

    def test_classify_from_object_focus(self):
        assert classify_from_object("focus run 3") == "focus"

    def test_classify_from_object_focus_uppercase(self):
        assert classify_from_object("FOCUS TEST") == "focus"

    def test_classify_from_object_bias(self):
        assert classify_from_object("bias frame") == "bias"

    def test_classify_from_object_science(self):
        assert classify_from_object("M31") == "science"

    def test_classify_from_object_none(self):
        assert classify_from_object(None) is None


class TestClassifyFrameTypeFocus:
    """Test classify_frame_type with FOCUS classification."""

    def test_classify_frame_type_imagetyp_focus_frame_wins(self):
        """IMAGETYP='Focus Frame' maps to 'focus' via DEFAULT_IMAGETYP_MAP."""
        header = fits.Header()
        header["IMAGETYP"] = "Focus Frame"
        header["OBJECT"] = "Science Target"

        profile = resolve_profile(name="generic")
        frame_type, _source = classify_frame_type(header, profile)
        assert frame_type == "focus"

    def test_classify_frame_type_focus_from_object(self):
        """OBJECT='focus test' is classified as 'focus' without IMAGETYP."""
        header = fits.Header()
        header["OBJECT"] = "focus test"

        profile = resolve_profile(name="generic")
        frame_type, _source = classify_frame_type(header, profile)
        assert frame_type == "focus"

    def test_classify_frame_type_imagetyp_takes_precedence(self):
        """IMAGETYP='Focus Frame' wins over OBJECT='M31'."""
        header = fits.Header()
        header["IMAGETYP"] = "Focus Frame"
        header["OBJECT"] = "M31"

        profile = resolve_profile(name="generic")
        frame_type, _source = classify_frame_type(header, profile)
        assert frame_type == "focus"
