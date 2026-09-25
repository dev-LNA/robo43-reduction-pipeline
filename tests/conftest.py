"""Shared pytest fixtures for the test suite."""

import pytest

from robo43.config import load_settings
from tests.fixtures.make_synthetic import make_synthetic_night


@pytest.fixture
def synthetic_night(tmp_path):
    make_synthetic_night(tmp_path, instrument="robo43", seed=0)
    return tmp_path


@pytest.fixture
def synthetic_night_synth2(tmp_path):
    make_synthetic_night(tmp_path, instrument="synth2", seed=0)
    return tmp_path


@pytest.fixture
def synthetic_night_stripped(tmp_path):
    make_synthetic_night(
        tmp_path, instrument="robo43", seed=0, strip=("gain", "pixel_scale", "filter")
    )
    return tmp_path


@pytest.fixture
def default_settings():
    return load_settings(use_user_config=False, env={})
