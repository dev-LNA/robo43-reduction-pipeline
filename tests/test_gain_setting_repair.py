"""Tests for the profile-declared gain_setting repair
(``[instrument.gain_setting_repair]``): a frame whose gain-setting card was
lost to a failed header fill-up (T80S ``PG5_2 = 0``) gets ``value_above`` (1)
when its fallback card (T80S ``PG0_4``, "CCD Read Time") exceeds the threshold
and ``value_at_or_below`` (2) otherwise; with no usable fallback card it stays 0.

Builds its own synthetic headers and inline TOML profiles; does not use
fixtures or conftest.py.
"""

from __future__ import annotations

import logging
from importlib.resources import files

import pytest
from astropy.io import fits

from robo43.calib.fallback import camera_config_key
from robo43.exceptions import ConfigError
from robo43.instruments.profile import GainSettingRepair, load_profile_file, resolve_profile
from robo43.io.headers import parse_duration_seconds, read_meta, repair_gain_setting
from robo43.io.meta import Source

_BASE_TOML = """
[instrument]
name = "test-instrument"

[instrument.keywords]
gain_setting = ["PG5_2"]
binning = ["PG5_4"]
"""

_REPAIR_TOML = _BASE_TOML + """
[instrument.gain_setting_repair]
when_value = 0
fallback_keyword = "PG0_4"
threshold_seconds = 12.0
value_above = 1
value_at_or_below = 2
"""


def _repair_profile():
    return resolve_profile(profile_toml=_REPAIR_TOML)


def _header(pg5_2, pg0_4=None) -> fits.Header:
    header = fits.Header()
    header["NAXIS1"] = 64
    header["NAXIS2"] = 64
    header["PG5_2"] = pg5_2
    header["PG5_4"] = 1
    header["READOUTM"] = "High Gain Mode 16BIT"
    if pg0_4 is not None:
        header["PG0_4"] = pg0_4
    return header


def _gain(header: fits.Header, profile=None) -> float | None:
    profile = profile if profile is not None else _repair_profile()
    return read_meta("frame.fits", profile, header=header).gain_setting


# --- profile parsing -------------------------------------------------------


def test_profile_without_table_has_no_repair() -> None:
    profile = resolve_profile(profile_toml=_BASE_TOML)
    assert profile.gain_setting_repair is None


def test_repair_table_parses_values() -> None:
    assert _repair_profile().gain_setting_repair == GainSettingRepair(
        when_value=0.0,
        fallback_keyword="PG0_4",
        threshold_seconds=12.0,
        value_above=1.0,
        value_at_or_below=2.0,
    )


def test_t80s_profile_declares_the_repair() -> None:
    path = files("robo43.data.instruments") / "t80s.toml"
    profile = load_profile_file(str(path))
    assert profile.gain_setting_repair == GainSettingRepair(
        when_value=0.0,
        fallback_keyword="PG0_4",
        threshold_seconds=12.0,
        value_above=1.0,
        value_at_or_below=2.0,
    )
    assert profile.aliases_for("gain_setting")[0] == "PG5_2"


def test_other_packaged_profiles_have_no_repair() -> None:
    for name in ("generic.toml",):
        path = files("robo43.data.instruments") / name
        assert load_profile_file(str(path)).gain_setting_repair is None


def test_repair_table_missing_key_is_a_config_error() -> None:
    toml = _BASE_TOML + """
[instrument.gain_setting_repair]
when_value = 0
fallback_keyword = "PG0_4"
threshold_seconds = 12.0
value_above = 1
"""
    with pytest.raises(ConfigError, match="value_at_or_below"):
        resolve_profile(profile_toml=toml)


def test_repair_table_unknown_key_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="bogus"):
        resolve_profile(profile_toml=_REPAIR_TOML + "bogus = 1\n")


def test_repair_table_non_numeric_threshold_is_a_config_error() -> None:
    toml = _REPAIR_TOML.replace("threshold_seconds = 12.0", 'threshold_seconds = "x"')
    with pytest.raises(ConfigError, match="threshold_seconds"):
        resolve_profile(profile_toml=toml)


# --- duration parsing ------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (16.227, 16.227),
        (10, 10.0),
        ("16.227", 16.227),
        (" 19.235 ", 19.235),
        ("00:00:16.227", 16.227),
        ("00:01:02.5", 62.5),
        ("01:00:00", 3600.0),
        ("abc", None),
        ("", None),
        ("1:2", None),
        ("00:00:xx", None),
        ("nan", None),
        (True, None),
        (None, None),
        ([1], None),
    ],
)
def test_parse_duration_seconds(raw, expected) -> None:
    result = parse_duration_seconds(raw)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


# --- repair rule through read_meta -----------------------------------------


def test_zero_with_long_read_time_becomes_one() -> None:
    assert _gain(_header(0, 16.227)) == 1.0


def test_zero_with_short_read_time_becomes_two() -> None:
    assert _gain(_header(0, 10.0)) == 2.0


def test_zero_without_fallback_card_stays_zero() -> None:
    assert _gain(_header(0)) == 0.0


def test_zero_with_unparseable_fallback_card_stays_zero() -> None:
    assert _gain(_header(0, "not a time")) == 0.0


def test_threshold_is_strictly_greater_than() -> None:
    assert _gain(_header(0, 12.0)) == 2.0
    assert _gain(_header(0, 12.001)) == 1.0


@pytest.mark.parametrize("pg0_4", [None, 5.0, 16.227, 19.235])
def test_nonzero_setting_is_never_touched(pg0_4) -> None:
    assert _gain(_header(1, pg0_4)) == 1.0
    assert _gain(_header(2, pg0_4)) == 2.0


def test_string_read_time_is_parsed() -> None:
    assert _gain(_header(0, "00:00:16.227")) == 1.0
    assert _gain(_header(0, "00:00:10.000")) == 2.0
    assert _gain(_header(0, "16.227")) == 1.0


def test_profile_without_the_option_is_unchanged() -> None:
    profile = resolve_profile(profile_toml=_BASE_TOML)
    assert _gain(_header(0, 16.227), profile) == 0.0


def test_repaired_value_is_marked_derived_with_reason() -> None:
    meta = read_meta("frame.fits", _repair_profile(), header=_header(0, 16.227))
    origin = meta.origin("gain_setting")
    assert origin is not None
    assert origin.source is Source.DERIVED
    assert "PG5_2=0 -> 1" in origin.detail
    assert "PG0_4=16.227" in origin.detail
    assert "gain_setting" in meta.provenance_json()


def test_unrepaired_value_keeps_its_header_origin() -> None:
    profile = _repair_profile()
    for header in (_header(1, 19.235), _header(0)):
        origin = read_meta("frame.fits", profile, header=header).origin("gain_setting")
        assert origin is not None
        assert origin.source is Source.HEADER
        assert origin.detail == "PG5_2"


def test_repaired_frame_has_the_same_camera_config_as_a_native_one() -> None:
    profile = _repair_profile()
    repaired = read_meta("a.fits", profile, header=_header(0, 16.227))
    native = read_meta("b.fits", profile, header=_header(1, 19.235))
    assert camera_config_key(repaired) == camera_config_key(native)
    short = read_meta("c.fits", profile, header=_header(0, 10.0))
    assert camera_config_key(short) != camera_config_key(native)
    unresolved = read_meta("d.fits", profile, header=_header(0))
    assert camera_config_key(unresolved) != camera_config_key(native)


def test_repair_function_directly() -> None:
    profile = _repair_profile()
    value, detail, status = repair_gain_setting(_header(0, 16.227), profile, 0.0, "PG5_2")
    assert (value, status) == (1.0, "repaired")
    assert detail is not None
    assert "PG0_4=16.227 s > 12 s" in detail
    value, detail, status = repair_gain_setting(_header(0, 5.0), profile, 0.0, "PG5_2")
    assert (value, status) == (2.0, "repaired")
    assert detail is not None
    assert "<= 12 s" in detail
    value, detail, status = repair_gain_setting(_header(0), profile, 0.0, "PG5_2")
    assert (value, status) == (0.0, "unresolved")
    assert detail is not None
    assert "PG0_4" in detail
    assert repair_gain_setting(_header(1, 16.227), profile, 1.0, "PG5_2") == (1.0, None, "none")
    no_repair = resolve_profile(profile_toml=_BASE_TOML)
    assert repair_gain_setting(_header(0, 16.227), no_repair, 0.0, "PG5_2") == (
        0.0,
        None,
        "none",
    )


def _repair_messages(caplog) -> list[tuple[int, str]]:
    return [
        (r.levelno, r.getMessage())
        for r in caplog.records
        if "gain_setting repair" in r.getMessage()
    ]


def test_first_repair_warns_once_then_debug(caplog) -> None:
    from robo43.io import headers

    headers._GAIN_REPAIR_WARNED.clear()
    profile = _repair_profile()
    with caplog.at_level(logging.DEBUG, logger="robo43.io.headers"):
        for _ in range(3):
            read_meta("frame.fits", profile, header=_header(0, 16.227))
    assert [level for level, _ in _repair_messages(caplog)] == [
        logging.WARNING,
        logging.DEBUG,
        logging.DEBUG,
    ]


def test_missing_fallback_card_logs_a_warning_and_keeps_zero(caplog) -> None:
    from robo43.io import headers

    headers._GAIN_REPAIR_WARNED.clear()
    profile = _repair_profile()
    with caplog.at_level(logging.DEBUG, logger="robo43.io.headers"):
        metas = [read_meta("frame.fits", profile, header=_header(0)) for _ in range(3)]
    assert [m.gain_setting for m in metas] == [0.0, 0.0, 0.0]
    messages = _repair_messages(caplog)
    assert [level for level, _ in messages] == [logging.WARNING, logging.DEBUG, logging.DEBUG]
    assert "PG0_4 is missing or unparseable" in messages[0][1]


def test_unchanged_frames_log_nothing(caplog) -> None:
    profile = _repair_profile()
    with caplog.at_level(logging.DEBUG, logger="robo43.io.headers"):
        read_meta("frame.fits", profile, header=_header(1, 19.235))
    assert _repair_messages(caplog) == []
