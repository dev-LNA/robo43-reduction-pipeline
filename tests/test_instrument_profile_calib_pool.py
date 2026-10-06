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
    # Symmetric-window pooling and master reuse default to off.
    assert profile.calib.flat_pool_window_nights == 0
    assert profile.calib.flat_master_roots == ()
    assert profile.calib.bias_pool is False


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
    assert profile.calib.flat_pool_window_nights == 30
    assert profile.calib.flat_pool_target_frames == 20
    assert profile.calib.flat_master_roots == ("/mnt/sto01/T80S/reduced",)
    assert profile.calib.bias_pool is True
    assert profile.calib.bias_pool_window_nights == 10
    assert profile.calib.bias_pool_target_frames == 20
    assert profile.calib.bias_pool_gain_max_frames == 12
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


def test_window_pool_keys_parse_and_validate() -> None:
    import pytest

    from robo43.exceptions import ConfigError

    toml = _BASE_TOML + """
[instrument.calib]
bias_pool = true
bias_pool_window_nights = 5
bias_pool_target_frames = 40
bias_pool_gain_max_frames = 0
flat_pool = true
flat_pool_window_nights = 15
flat_master_roots = ["/a/reduced", "/b/reduced"]
"""
    calib = resolve_profile(profile_toml=toml).calib
    assert calib.bias_pool is True
    assert (calib.bias_pool_window_nights, calib.bias_pool_target_frames) == (5, 40)
    assert calib.bias_pool_gain_max_frames == 0
    assert calib.flat_pool_window_nights == 15
    assert calib.flat_master_roots == ("/a/reduced", "/b/reduced")
    for bad in ("bias_pool_target_frames = 0", "flat_pool_window_nights = -1",
                "bias_pool_window_nights = -3"):
        with pytest.raises(ConfigError):
            resolve_profile(profile_toml=_BASE_TOML + "\n[instrument.calib]\n" + bad + "\n")
