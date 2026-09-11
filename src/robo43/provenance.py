"""Provenance stamping: the single place that answers "what made this product".

Every later phase calls into this module to determine which packaged data
file to hand an external tool, and to build the ``PIPE*`` FITS header cards
that trace a product back to the exact code and configuration that made it.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from datetime import UTC, datetime
from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astropy.io import fits

import robo43

if TYPE_CHECKING:
    from robo43.config import Settings

__all__ = [
    "apply_cards",
    "package_data_dir",
    "package_data_path",
    "provenance_cards",
    "sha256_file",
]

#: Kept alive for the life of this process so materialised paths from
#: :func:`package_data_path` remain valid for the external tools that use
#: them (e.g. source-extractor, invoked as a subprocess after we return).
_MATERIALISED_STACK = contextlib.ExitStack()
_MATERIALISED_PATHS: dict[str, Path] = {}


def package_data_dir() -> Traversable:
    """The ``robo43.data`` package as a :class:`~importlib.resources.abc.Traversable`."""
    return files("robo43.data")


def package_data_path(name: str) -> Path:
    """Materialise one packaged data file to a real filesystem path.

    Source-extractor needs an actual path on disk, not a ``Traversable`` or a
    zip member. The materialised path is cached by ``name`` and kept alive
    for the life of this process.
    """
    if name in _MATERIALISED_PATHS:
        return _MATERIALISED_PATHS[name]
    resource = package_data_dir() / name
    path = _MATERIALISED_STACK.enter_context(as_file(resource))
    if not path.exists():
        msg = f"packaged data file not found: {name}"
        raise FileNotFoundError(msg)
    _MATERIALISED_PATHS[name] = path
    return path


def sha256_file(path: str | os.PathLike[str], *, chunk: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file, read in ``chunk``-byte increments."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def provenance_cards(settings: Settings) -> list[tuple[str, Any, str]]:
    """Build the ``PIPE*`` header cards that trace a product to its origin.

    A value that cannot be determined -- currently only ``PIPEGIT`` when the
    checkout is not a git work tree -- is omitted entirely, never written as
    a sentinel.
    """
    cards: list[tuple[str, Any, str]] = [
        ("PIPEVER", robo43.__version__, "pipeline version"),
    ]
    git_sha = robo43.git_sha()
    if git_sha is not None:
        cards.append(("PIPEGIT", git_sha, "git SHA"))
    cards.append(("PIPECFG", settings.config_hash(), "config hash"))
    cards.append(
        ("PIPEDATE", datetime.now(UTC).isoformat(timespec="seconds"), "UTC of reduction")
    )
    return cards


def apply_cards(header: fits.Header, cards: list[tuple[str, Any, str]]) -> None:
    """Write ``cards`` into ``header``, skipping any whose value is ``None``."""
    for key, value, comment in cards:
        if value is None:
            continue
        header[key] = (value, comment)
