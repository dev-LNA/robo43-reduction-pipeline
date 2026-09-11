from __future__ import annotations

import pytest

from robo43.config import (
    CATALOG_FORMATS,
    COADD_SCALINGS,
    COMBINE_METHODS,
    DIAGNOSTIC_LEVELS,
    SCIENCE_SECTIONS,
    load_settings,
)
from robo43.exceptions import ConfigError


def test_precedence_file_env_cli(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[astrometry]\nsearch_radius_deg = 1.0\n")

    file_only = load_settings(config_file=config_path, use_user_config=False, env={})
    assert file_only.astrometry.search_radius_deg == 1.0

    env_over_file = load_settings(
        config_file=config_path,
        use_user_config=False,
        env={"ROBO43_ASTROMETRY_SEARCH_RADIUS_DEG": "2.0"},
    )
    assert env_over_file.astrometry.search_radius_deg == 2.0

    cli_over_all = load_settings(
        config_file=config_path,
        use_user_config=False,
        env={"ROBO43_ASTROMETRY_SEARCH_RADIUS_DEG": "2.0"},
        cli={"astrometry.search_radius_deg": 3.0},
    )
    assert cli_over_all.astrometry.search_radius_deg == 3.0


def test_unknown_section_raises(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[nonsense_section]\nfoo = 1\n")
    with pytest.raises(ConfigError):
        load_settings(config_file=config_path, use_user_config=False, env={})


def test_unknown_key_raises(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[astrometry]\nnot_a_real_field = 1\n")
    with pytest.raises(ConfigError):
        load_settings(config_file=config_path, use_user_config=False, env={})


def test_invalid_diagnostics_level_raises():
    with pytest.raises(ConfigError):
        load_settings(use_user_config=False, env={}, cli={"diagnostics.level": "bogus"})


def test_invalid_combine_method_raises():
    with pytest.raises(ConfigError):
        load_settings(use_user_config=False, env={}, cli={"calib.combine_method": "bogus"})


def test_invalid_catalog_format_raises():
    with pytest.raises(ConfigError):
        load_settings(
            use_user_config=False, env={}, cli={"photometry.catalog_format": "bogus"}
        )


def test_invalid_coadd_scale_by_raises():
    with pytest.raises(ConfigError):
        load_settings(use_user_config=False, env={}, cli={"coadd.scale_by": "bogus"})


def test_valid_diagnostics_level_loads():
    assert "full" in DIAGNOSTIC_LEVELS
    settings = load_settings(
        use_user_config=False, env={}, cli={"diagnostics.level": "full"}
    )
    assert settings.diagnostics.level == "full"


def test_valid_enum_values_are_module_constants():
    assert COMBINE_METHODS
    assert CATALOG_FORMATS
    assert COADD_SCALINGS


def test_config_hash_stable_under_runtime_change():
    baseline = load_settings(use_user_config=False, env={})
    variant = load_settings(
        use_user_config=False, env={}, cli={"runtime.nprocs": 4}
    )
    assert variant.config_hash() == baseline.config_hash()


def test_config_hash_stable_under_db_change():
    baseline = load_settings(use_user_config=False, env={})
    variant = load_settings(use_user_config=False, env={}, cli={"db.enabled": True})
    assert variant.config_hash() == baseline.config_hash()


def test_config_hash_stable_under_paths_change(tmp_path):
    baseline = load_settings(use_user_config=False, env={})
    variant = load_settings(
        use_user_config=False, env={}, cli={"paths.workdir": str(tmp_path)}
    )
    assert variant.config_hash() == baseline.config_hash()


def test_config_hash_stable_under_diagnostics_change():
    baseline = load_settings(use_user_config=False, env={})
    variant = load_settings(
        use_user_config=False, env={}, cli={"diagnostics.level": "full"}
    )
    assert variant.config_hash() == baseline.config_hash()


def test_config_hash_changes_under_science_section_change():
    baseline = load_settings(use_user_config=False, env={})

    zp_variant = load_settings(
        use_user_config=False, env={}, cli={"photometry.instrumental_zp": 21.0}
    )
    assert zp_variant.config_hash() != baseline.config_hash()

    combine_variant = load_settings(
        use_user_config=False, env={}, cli={"calib.combine_method": "median"}
    )
    assert combine_variant.config_hash() != baseline.config_hash()


def test_science_sections_constant_used_for_hash():
    assert set(SCIENCE_SECTIONS) == {"astrometry", "calib", "coadd", "photometry", "targets"}


def test_strict_and_assume_profile_defaults_rejected():
    with pytest.raises(ConfigError):
        load_settings(
            use_user_config=False,
            env={},
            cli={"runtime.strict": True, "runtime.assume_profile_defaults": True},
        )
