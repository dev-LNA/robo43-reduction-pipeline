"""Tests for T80S filter name canonicalisation via [instrument.filters.map]."""

from __future__ import annotations

from robo43.instruments.profile import resolve_profile


class TestT80sFilterMap:
    """Test T80S profile's filter name canonicalisation."""

    def test_t80s_canonical_filter_rsdss_to_r(self) -> None:
        """S-PLUS rSDSS band name maps to T80S canonical 'R'."""
        profile = resolve_profile(name="t80s")
        assert profile.canonical_filter("rSDSS") == "R"

    def test_t80s_canonical_filter_r_to_r(self) -> None:
        """T80S canonical 'R' band name maps to itself."""
        profile = resolve_profile(name="t80s")
        assert profile.canonical_filter("R") == "R"

    def test_t80s_canonical_filter_gsdss_to_g(self) -> None:
        """S-PLUS gSDSS band name maps to T80S canonical 'G'."""
        profile = resolve_profile(name="t80s")
        assert profile.canonical_filter("gSDSS") == "G"
