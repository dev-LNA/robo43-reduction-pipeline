"""Tests for the profile-declared readout-mode derivation
(``[instrument.readout_mode_rule]``): T80S has two camera modes that must never
share calibrations, told apart by the header cards PG0_50 ("Gain S, Volts"),
PG5_1 ("DSI Sample Time") and PG5_2 ("Gain, (Normal)/(High)").  The rule sets
``FrameMeta.readout_mode`` ("mode0" / "mode5") in ``read_meta``, so the camera
configuration key, the gain group key and master-flat reuse all separate the
modes.

Builds its own synthetic headers and inline TOML profiles; does not use
fixtures or conftest.py.
"""

from __future__ import annotations

import logging
from importlib.resources import files
from pathlib import Path

import pytest
from astropy.io import fits

from robo43.calib.fallback import camera_config_key
from robo43.calib.gain import _group_key
from robo43.exceptions import ConfigError
from robo43.instruments.profile import (
    ReadoutModeRule,
    ReadoutModeVote,
    load_profile_file,
    resolve_profile,
)
from robo43.io import headers
from robo43.io.headers import derive_readout_mode, read_meta
from robo43.io.meta import Source
from robo43.io.validate import OverrideSet

_BASE_TOML = """
[instrument]
name = "test-instrument"

[instrument.keywords]
gain_setting = ["PG5_2"]
binning = ["PG5_4"]
"""

_RULE_TOML = _BASE_TOML + """
[instrument.readout_mode_rule]
default = "mode5"

[[instrument.readout_mode_rule.votes]]
keyword = "PG0_50"
threshold = -6.0
below = "mode0"
at_or_above = "mode5"

[[instrument.readout_mode_rule.votes]]
keyword = "PG5_1"
map = { "17" = "mode0", "68" = "mode5" }

[[instrument.readout_mode_rule.votes]]
keyword = "PG5_2"
map = { "0" = "mode0", "1" = "mode5" }
"""


def _rule_profile():
    return resolve_profile(profile_toml=_RULE_TOML)


def _header(pg0_50=None, pg5_1=None, pg5_2=None, **extra) -> fits.Header:
    header = fits.Header()
    header["NAXIS1"] = 64
    header["NAXIS2"] = 64
    header["PG5_4"] = 1
    if pg0_50 is not None:
        header["PG0_50"] = pg0_50
    if pg5_1 is not None:
        header["PG5_1"] = pg5_1
    if pg5_2 is not None:
        header["PG5_2"] = pg5_2
    for key, value in extra.items():
        header[key] = value
    return header


def _mode(header: fits.Header, profile=None) -> str | None:
    profile = profile if profile is not None else _rule_profile()
    return read_meta("frame.fits", profile, header=header).readout_mode


def _rule_messages(caplog) -> list[tuple[int, str]]:
    return [
        (r.levelno, r.getMessage())
        for r in caplog.records
        if "readout mode" in r.getMessage()
    ]


@pytest.fixture(autouse=True)
def _reset_warned():
    headers._READOUT_MODE_WARNED.clear()
    yield
    headers._READOUT_MODE_WARNED.clear()


# --- profile parsing -------------------------------------------------------


def test_profile_without_table_has_no_rule() -> None:
    assert resolve_profile(profile_toml=_BASE_TOML).readout_mode_rule is None


def test_rule_table_parses_values() -> None:
    assert _rule_profile().readout_mode_rule == ReadoutModeRule(
        default="mode5",
        votes=(
            ReadoutModeVote(
                keyword="PG0_50", threshold=-6.0, below="mode0", at_or_above="mode5"
            ),
            ReadoutModeVote(keyword="PG5_1", values=((17.0, "mode0"), (68.0, "mode5"))),
            ReadoutModeVote(keyword="PG5_2", values=((0.0, "mode0"), (1.0, "mode5"))),
        ),
    )


def test_t80s_profile_loads_the_rule() -> None:
    path = files("robo43.data.instruments") / "t80s.toml"
    profile = load_profile_file(str(path))
    assert profile.readout_mode_rule == _rule_profile().readout_mode_rule
    assert profile.aliases_for("gain_setting")[0] == "PG5_2"


def test_other_packaged_profiles_have_no_rule() -> None:
    path = files("robo43.data.instruments") / "generic.toml"
    assert load_profile_file(str(path)).readout_mode_rule is None


@pytest.mark.parametrize(
    ("toml", "match"),
    [
        (_RULE_TOML + "bogus = 1\n", "bogus"),
        (_RULE_TOML.replace('default = "mode5"\n', ""), "default"),
        (_RULE_TOML.replace('default = "mode5"', 'default = ""'), "default"),
        (_RULE_TOML.replace('default = "mode5"', "default = 5"), "default"),
        (
            _BASE_TOML + '[instrument.readout_mode_rule]\ndefault = "mode5"\nvotes = []\n',
            "votes",
        ),
        (
            _BASE_TOML + '[instrument.readout_mode_rule]\ndefault = "mode5"\n',
            "votes",
        ),
        (_RULE_TOML.replace("threshold = -6.0", 'threshold = "x"'), "threshold"),
        (_RULE_TOML.replace("threshold = -6.0", "threshold = true"), "threshold"),
        (_RULE_TOML.replace("threshold = -6.0", "threshold = nan"), "finite"),
        (_RULE_TOML.replace('below = "mode0"\n', ""), "below"),
        (_RULE_TOML.replace('keyword = "PG5_1"', 'keyword = ""'), "keyword"),
        (_RULE_TOML.replace('keyword = "PG5_1"\n', ""), "keyword"),
        (_RULE_TOML.replace('{ "17" = "mode0", "68" = "mode5" }', '{ "x" = "mode0" }'), "number"),
        (_RULE_TOML.replace('{ "17" = "mode0", "68" = "mode5" }', "{}"), "map"),
        (_RULE_TOML.replace('{ "17" = "mode0", "68" = "mode5" }', '{ "17" = 3 }'), "map value"),
        (
            _RULE_TOML.replace(
                'keyword = "PG5_1"\n', 'keyword = "PG5_1"\nthreshold = 1.0\n'
            ),
            "cannot combine",
        ),
        (
            _RULE_TOML.replace(
                'keyword = "PG5_1"\n', 'keyword = "PG5_1"\nbogus = 1\n'
            ),
            "bogus",
        ),
        (_BASE_TOML.replace("[instrument]\n", '[instrument]\nreadout_mode_rule = 3\n'), "table"),
    ],
)
def test_rule_table_errors_are_config_errors(toml, match) -> None:
    with pytest.raises(ConfigError, match=match):
        resolve_profile(profile_toml=toml)


def test_a_vote_with_neither_map_nor_threshold_is_a_config_error() -> None:
    toml = _BASE_TOML + """
[instrument.readout_mode_rule]
default = "mode5"

[[instrument.readout_mode_rule.votes]]
keyword = "PG5_1"
"""
    with pytest.raises(ConfigError, match="'map' or 'threshold'"):
        resolve_profile(profile_toml=toml)


# --- each vote alone -------------------------------------------------------


@pytest.mark.parametrize(
    ("pg0_50", "expected"),
    [
        (-13.0, "mode0"),
        (-12.96, "mode0"),
        (-12.87, "mode0"),
        (-6.0001, "mode0"),
        (-6.0, "mode5"),
        (0.0, "mode5"),
        (0.024, "mode5"),
        (0.121, "mode5"),
        ("-12.96", "mode0"),
        ("0.05", "mode5"),
    ],
)
def test_pg0_50_vote_alone(pg0_50, expected, caplog) -> None:
    with caplog.at_level(logging.DEBUG, logger="robo43.io.headers"):
        assert _mode(_header(pg0_50=pg0_50)) == expected
    assert _rule_messages(caplog) == []


@pytest.mark.parametrize(("pg5_1", "expected"), [(17, "mode0"), (68, "mode5"), (17.0, "mode0")])
def test_pg5_1_vote_alone(pg5_1, expected, caplog) -> None:
    with caplog.at_level(logging.DEBUG, logger="robo43.io.headers"):
        assert _mode(_header(pg5_1=pg5_1)) == expected
    assert _rule_messages(caplog) == []


@pytest.mark.parametrize(("pg5_2", "expected"), [(0, "mode0"), (1, "mode5")])
def test_pg5_2_vote_alone(pg5_2, expected, caplog) -> None:
    with caplog.at_level(logging.DEBUG, logger="robo43.io.headers"):
        assert _mode(_header(pg5_2=pg5_2)) == expected
    assert _rule_messages(caplog) == []


@pytest.mark.parametrize(
    "header",
    [
        _header(pg5_1=34),
        _header(pg5_1=0),
        _header(pg5_2=2),
        _header(pg5_2=-1),
        _header(pg0_50="not a voltage"),
        _header(pg0_50="nan"),
        _header(pg0_50=True),
        _header(pg5_1=""),
        _header(),
    ],
)
def test_an_unknown_missing_or_unparseable_card_casts_no_vote(header) -> None:
    result = derive_readout_mode(header, _rule_profile())
    assert result is not None
    mode, _detail, status = result
    assert (mode, status) == ("mode5", "default")


def test_a_lower_priority_vote_decides_when_higher_ones_cast_none() -> None:
    assert _mode(_header(pg5_1=34, pg5_2=0)) == "mode0"
    assert _mode(_header(pg0_50="junk", pg5_1=17)) == "mode0"
    assert _mode(_header(pg5_1=34, pg5_2=1)) == "mode5"


def test_the_derived_mode_names_its_deciding_card() -> None:
    meta = read_meta("frame.fits", _rule_profile(), header=_header(-12.96, 17, 0))
    origin = meta.origin("readout_mode")
    assert origin is not None
    assert origin.source is Source.DERIVED
    assert origin.detail == "mode0 from PG0_50=-12.96"
    assert "readout_mode" in meta.provenance_json()
    meta = read_meta("frame.fits", _rule_profile(), header=_header(pg5_1=68))
    assert meta.origin("readout_mode").detail == "mode5 from PG5_1=68"


def test_gain_setting_stays_the_raw_pg5_2_value() -> None:
    mode0 = read_meta("a.fits", _rule_profile(), header=_header(-12.96, 17, 0))
    mode5 = read_meta("b.fits", _rule_profile(), header=_header(0.05, 68, 1))
    assert (mode0.readout_mode, mode0.gain_setting) == ("mode0", 0.0)
    assert (mode5.readout_mode, mode5.gain_setting) == ("mode5", 1.0)
    for meta in (mode0, mode5):
        origin = meta.origin("gain_setting")
        assert origin is not None
        assert origin.source is Source.HEADER
        assert origin.detail == "PG5_2"


# --- priority, disagreement, default --------------------------------------


def test_the_highest_priority_vote_wins_a_disagreement() -> None:
    mode, detail, status = derive_readout_mode(_header(-12.96, 68, 1), _rule_profile())
    assert (mode, status) == ("mode0", "disagree")
    assert detail.startswith("mode0 from PG0_50=-12.96")
    assert "PG5_1=68 -> mode5" in detail
    assert "PG5_2=1 -> mode5" in detail
    assert _mode(_header(0.05, 17, 0)) == "mode5"
    assert _mode(_header(pg5_1=17, pg5_2=1)) == "mode0"
    assert _mode(_header(pg5_1=68, pg5_2=0)) == "mode5"


def test_agreeing_votes_are_status_ok() -> None:
    assert derive_readout_mode(_header(-12.96, 17, 0), _rule_profile())[2] == "ok"
    assert derive_readout_mode(_header(0.05, 68, 1), _rule_profile())[2] == "ok"


def test_disagreement_warns_once_then_debug(caplog) -> None:
    profile = _rule_profile()
    with caplog.at_level(logging.DEBUG, logger="robo43.io.headers"):
        for _ in range(3):
            read_meta("frame.fits", profile, header=_header(-12.96, 68, 1))
    assert [level for level, _ in _rule_messages(caplog)] == [
        logging.WARNING,
        logging.DEBUG,
        logging.DEBUG,
    ]
    assert "disagreeing" in _rule_messages(caplog)[0][1]


def test_no_vote_uses_the_default_with_a_warning_once_then_debug(caplog) -> None:
    profile = _rule_profile()
    with caplog.at_level(logging.DEBUG, logger="robo43.io.headers"):
        metas = [read_meta("frame.fits", profile, header=_header()) for _ in range(3)]
    assert [m.readout_mode for m in metas] == ["mode5", "mode5", "mode5"]
    messages = _rule_messages(caplog)
    assert [level for level, _ in messages] == [logging.WARNING, logging.DEBUG, logging.DEBUG]
    assert "by default" in messages[0][1]
    origin = metas[0].origin("readout_mode")
    assert origin is not None and origin.source is Source.DERIVED


def test_disagreement_and_default_are_warned_separately_per_profile(caplog) -> None:
    other = resolve_profile(
        profile_toml=_RULE_TOML.replace('name = "test-instrument"', 'name = "other"')
    )
    with caplog.at_level(logging.DEBUG, logger="robo43.io.headers"):
        read_meta("a.fits", _rule_profile(), header=_header(-12.96, 68, 1))
        read_meta("b.fits", _rule_profile(), header=_header())
        read_meta("c.fits", other, header=_header(-12.96, 68, 1))
        read_meta("d.fits", _rule_profile(), header=_header(-12.96, 68, 1))
    assert [level for level, _ in _rule_messages(caplog)] == [
        logging.WARNING,
        logging.WARNING,
        logging.WARNING,
        logging.DEBUG,
    ]


def test_agreeing_frames_log_nothing(caplog) -> None:
    with caplog.at_level(logging.DEBUG, logger="robo43.io.headers"):
        read_meta("frame.fits", _rule_profile(), header=_header(-12.96, 17, 0))
    assert _rule_messages(caplog) == []


# --- precedence ------------------------------------------------------------


def test_the_rule_replaces_a_header_alias_readout_mode() -> None:
    header = _header(-12.96, 17, 0, READOUTM="High Gain Mode 16BIT")
    assert _mode(header) == "mode0"


def test_an_explicit_override_wins_over_the_rule() -> None:
    overrides = OverrideSet.from_cli(["readout_mode=custom"])
    meta = read_meta(
        "frame.fits", _rule_profile(), header=_header(-12.96, 17, 0), overrides=overrides
    )
    assert meta.readout_mode == "custom"
    origin = meta.origin("readout_mode")
    assert origin is not None
    assert origin.source is Source.OVERRIDE


def test_an_override_of_another_field_does_not_stop_the_rule() -> None:
    overrides = OverrideSet.from_cli(["gain_setting=7"])
    meta = read_meta(
        "frame.fits", _rule_profile(), header=_header(-12.96, 17, 0), overrides=overrides
    )
    assert (meta.readout_mode, meta.gain_setting) == ("mode0", 7.0)


def test_a_profile_without_the_table_is_unchanged(caplog) -> None:
    profile = resolve_profile(profile_toml=_BASE_TOML)
    with caplog.at_level(logging.DEBUG, logger="robo43.io.headers"):
        assert _mode(_header(-12.96, 17, 0), profile) is None
        assert _mode(_header(-12.96, 17, 0, READOUTM="High Gain Mode 16BIT"), profile) == (
            "High Gain Mode 16BIT"
        )
    assert _rule_messages(caplog) == []
    meta = read_meta("frame.fits", profile, header=_header(-12.96, 17, 0))
    assert meta.origin("readout_mode") is None
    assert derive_readout_mode(_header(-12.96, 17, 0), profile) is None


# --- calibration grouping --------------------------------------------------


def test_camera_config_key_separates_the_modes_even_with_the_same_gain_setting() -> None:
    profile = _rule_profile()
    mode0 = read_meta("a.fits", profile, header=_header(pg0_50=-12.96, pg5_2=1))
    mode5 = read_meta("b.fits", profile, header=_header(pg0_50=0.05, pg5_2=1))
    assert mode0.gain_setting == mode5.gain_setting == 1.0
    assert (mode0.readout_mode, mode5.readout_mode) == ("mode0", "mode5")
    assert camera_config_key(mode0) != camera_config_key(mode5)
    again = read_meta("c.fits", profile, header=_header(pg0_50=-12.9, pg5_2=1))
    assert camera_config_key(again) == camera_config_key(mode0)


def test_gain_group_keys_separate_the_modes() -> None:
    profile = _rule_profile()
    mode0 = read_meta("a.fits", profile, header=_header(-12.96, 17, 0))
    mode5 = read_meta("b.fits", profile, header=_header(0.05, 68, 1))
    mode0_same_gain = read_meta("c.fits", profile, header=_header(pg0_50=-12.96, pg5_2=1))
    assert _group_key(mode0) == ("mode0", 0.0, 1)
    assert _group_key(mode5) == ("mode5", 1.0, 1)
    assert _group_key(mode0_same_gain) == ("mode0", 1.0, 1)
    assert len({_group_key(mode0), _group_key(mode5), _group_key(mode0_same_gain)}) == 3


def test_recorded_readoutm_card_carries_the_derived_mode() -> None:
    """The READOUTM card of every reduced product and master comes from
    FrameMeta.readout_mode (calib.gain header cards), i.e. the derived mode."""
    from robo43.calib.gain import no_measurement_cards

    meta = read_meta("a.fits", _rule_profile(), header=_header(-12.96, 17, 0))
    cards = {key: value for key, value, _ in no_measurement_cards(
        meta.gain_setting, meta.readout_mode
    )}
    assert cards["READOUTM"] == "mode0"
    assert cards["GAINSET"] == 0.0


def test_real_t80s_headers_if_available() -> None:
    raw = Path("/mnt/sto01/T80S/raw")
    june = sorted((raw / "20250609").glob("bias-*.fits.fz")) if raw.is_dir() else []
    dec = sorted((raw / "20251218").glob("bias-*.fits.fz")) if raw.is_dir() else []
    if not june or not dec:
        pytest.skip("T80S raw frames not mounted")
    profile = load_profile_file(str(files("robo43.data.instruments") / "t80s.toml"))
    a = read_meta(june[0], profile)
    b = read_meta(dec[0], profile)
    assert (a.readout_mode, a.gain_setting) == ("mode0", 0.0)
    assert (b.readout_mode, b.gain_setting) == ("mode5", 1.0)
    assert camera_config_key(a) != camera_config_key(b)
