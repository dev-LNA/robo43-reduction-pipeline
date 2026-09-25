"""Regression tests for profile filesystem path support (P4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from robo43.exceptions import ConfigError
from robo43.instruments.profile import InstrumentProfile, resolve_profile


class TestResolveProfilePath:
    """Test resolve_profile accepting filesystem paths."""

    def test_resolve_profile_by_registered_name(self):
        """Registered name still works."""
        profile = resolve_profile(name="generic")
        assert isinstance(profile, InstrumentProfile)
        assert profile.name == "generic"

    def test_resolve_profile_by_toml_path(self):
        """TOML file path is accepted."""
        synth2_toml = Path(__file__).parent / "fixtures" / "synth2.toml"
        if not synth2_toml.exists():
            pytest.skip("synth2.toml fixture not found")

        profile = resolve_profile(name=str(synth2_toml))
        assert isinstance(profile, InstrumentProfile)

    def test_resolve_profile_unresolvable_name_raises(self):
        """Unresolvable name raises ConfigError with helpful message."""
        with pytest.raises(ConfigError) as exc_info:
            resolve_profile(name="nonexistent_instrument")

        error_msg = str(exc_info.value)
        assert "not found as a registered name" in error_msg
        assert "existing .toml file path" in error_msg

    def test_resolve_profile_pathlib_path(self):
        """pathlib.Path object is accepted."""
        synth2_toml = Path(__file__).parent / "fixtures" / "synth2.toml"
        if not synth2_toml.exists():
            pytest.skip("synth2.toml fixture not found")

        # Pass a Path object, not a string
        profile = resolve_profile(name=synth2_toml)
        assert isinstance(profile, InstrumentProfile)
