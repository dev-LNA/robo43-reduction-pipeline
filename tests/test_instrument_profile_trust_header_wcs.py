"""Tests for the ``trust_header_wcs`` instrument-profile flag.

Builds its own inline TOML strings; does not use fixtures or conftest.py.
"""

from __future__ import annotations

from robo43.instruments.profile import load_profile_file, resolve_profile

_BASE_TOML = """
[instrument]
name = "test-instrument"
"""


def test_trust_header_wcs_defaults_true_when_absent() -> None:
    profile = resolve_profile(profile_toml=_BASE_TOML)
    assert profile.trust_header_wcs is True


def test_trust_header_wcs_parses_explicit_false() -> None:
    profile = resolve_profile(profile_toml=_BASE_TOML + "trust_header_wcs = false\n")
    assert profile.trust_header_wcs is False


def test_trust_header_wcs_parses_explicit_true() -> None:
    profile = resolve_profile(profile_toml=_BASE_TOML + "trust_header_wcs = true\n")
    assert profile.trust_header_wcs is True


def test_t80s_profile_sets_trust_header_wcs_false() -> None:
    from importlib.resources import files

    path = files("robo43.data.instruments") / "t80s.toml"
    profile = load_profile_file(str(path))
    assert profile.trust_header_wcs is False
    assert profile.pixel_scale_arcsec == 0.55


def test_generic_profile_defaults_trust_header_wcs_true() -> None:
    from importlib.resources import files

    path = files("robo43.data.instruments") / "generic.toml"
    profile = load_profile_file(str(path))
    assert profile.trust_header_wcs is True
