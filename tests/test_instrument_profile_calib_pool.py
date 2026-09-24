"""Tests for the optional [instrument.calib] flat-pooling policy table.

Builds its own inline TOML strings; does not use fixtures or conftest.py.
"""

from __future__ import annotations

from robo43.instruments.profile import load_profile_file, resolve_profile

_BASE_TOML = """
[instrument]
name = "test-instrument"
"""


def test_calib_defaults_mean_pooling_off_when_table_absent() -> None:
    profile = resolve_profile(profile_toml=_BASE_TOML)
    assert profile.calib.flat_pool is False
    assert profile.calib.flat_pool_min_nights == 10
    assert profile.calib.flat_pool_max_nights == 20
    assert profile.calib.flat_pool_target_frames == 20


def test_calib_table_parses_explicit_values() -> None:
    toml = _BASE_TOML + """
[instrument.calib]
flat_pool = true
flat_pool_min_nights = 7
flat_pool_max_nights = 14
flat_pool_target_frames = 15
"""
    profile = resolve_profile(profile_toml=toml)
    assert profile.calib.flat_pool is True
    assert profile.calib.flat_pool_min_nights == 7
    assert profile.calib.flat_pool_max_nights == 14
    assert profile.calib.flat_pool_target_frames == 15


def test_calib_table_partial_override_keeps_other_defaults() -> None:
    toml = _BASE_TOML + """
[instrument.calib]
flat_pool = true
"""
    profile = resolve_profile(profile_toml=toml)
    assert profile.calib.flat_pool is True
    assert profile.calib.flat_pool_min_nights == 10
    assert profile.calib.flat_pool_max_nights == 20
    assert profile.calib.flat_pool_target_frames == 20


def test_t80s_profile_enables_pooling_with_documented_values() -> None:
    from importlib.resources import files

    path = files("robo43.data.instruments") / "t80s.toml"
    profile = load_profile_file(str(path))
    assert profile.calib.flat_pool is True
    assert profile.calib.flat_pool_min_nights == 10
    assert profile.calib.flat_pool_max_nights == 20
    assert profile.calib.flat_pool_target_frames == 20
    # Untouched existing tables must survive the targeted edit.
    assert profile.imagetyp_map["sky-flat"] == "flat"
    assert profile.layout.science_hdu == 1
    assert profile.layout.n_amplifiers == 16
    assert profile.site_lat_deg == -30.1678638889


def test_generic_profile_pooling_defaults_off() -> None:
    from importlib.resources import files

    path = files("robo43.data.instruments") / "generic.toml"
    profile = load_profile_file(str(path))
    assert profile.calib.flat_pool is False
