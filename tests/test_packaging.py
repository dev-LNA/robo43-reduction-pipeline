from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

import robo43
import robo43.provenance
from robo43.config import load_settings
from robo43.exceptions import ExternalToolError
from robo43.external import find_sextractor

_SEXTRACTOR_DATA_FILES = (
    "default.conv",
    "default.nnw",
    "default.sex",
    "default.param",
    "sextractor.xsl",
)


def test_sextractor_data_files_reachable_via_importlib_resources():
    data_dir = robo43.provenance.package_data_dir()
    for name in _SEXTRACTOR_DATA_FILES:
        assert data_dir.joinpath(name).is_file()


def test_package_data_path_returns_existing_paths():
    for name in ("default.sex", "default.param", "default.conv", "default.nnw"):
        path = robo43.provenance.package_data_path(name)
        assert isinstance(path, Path)
        assert path.exists()


def test_package_data_dir_independent_of_cwd(tmp_path):
    """Not a perfect proof that no ``__file__`` arithmetic is involved -- a
    ``__file__``-relative approach would typically still resolve correctly
    from any working directory unless the package were relocated post-install,
    which cannot easily be simulated in-place. This is a reasonable proxy: it
    confirms ``package_data_dir`` does not depend on the caller's cwd.
    """
    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        data_dir = robo43.provenance.package_data_dir()
        assert data_dir.joinpath("config.default.toml").is_file()
    finally:
        os.chdir(original_cwd)


def test_provenance_cards_omits_pipegit_when_git_sha_is_none(monkeypatch):
    monkeypatch.setattr(robo43, "git_sha", lambda: None)
    settings = load_settings(use_user_config=False, env={})
    cards = robo43.provenance.provenance_cards(settings)
    assert not any(card[0] == "PIPEGIT" for card in cards)
    assert any(card[0] == "PIPEVER" for card in cards)
    assert any(card[0] == "PIPECFG" for card in cards)
    assert any(card[0] == "PIPEDATE" for card in cards)


def test_provenance_cards_includes_pipegit_in_real_checkout():
    settings = load_settings(use_user_config=False, env={})
    cards = robo43.provenance.provenance_cards(settings)
    assert any(card[0] == "PIPEGIT" for card in cards)


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"0123456789abcdef" * 4)
    assert robo43.provenance.sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_sha256_file_chunking_does_not_change_digest(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"0123456789abcdef" * 4)
    default_digest = robo43.provenance.sha256_file(path)
    chunked_digest = robo43.provenance.sha256_file(path, chunk=16)
    assert chunked_digest == default_digest


def test_find_sextractor_succeeds():
    resolved = find_sextractor()
    assert isinstance(resolved, str)
    assert Path(resolved).is_file()


def test_find_sextractor_raises_with_helpful_message_when_path_empty(monkeypatch):
    monkeypatch.setenv("PATH", "")
    with pytest.raises(ExternalToolError) as exc_info:
        find_sextractor()
    msg = str(exc_info.value)
    assert "source-extractor" in msg
    assert "apt install source-extractor" in msg
