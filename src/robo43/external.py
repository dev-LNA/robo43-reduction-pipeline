"""Discovery of required external binaries.

Declares the external tool dependencies explicitly and fails with a clear,
actionable message when one is absent, instead of the opaque traceback the
legacy ``sewpy`` wrapper produced when ``source-extractor`` was missing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from robo43.exceptions import ExternalToolError

__all__ = [
    "SEXTRACTOR_CANDIDATES",
    "find_sextractor",
    "require",
    "sextractor_version",
]

SEXTRACTOR_CANDIDATES: tuple[str, ...] = ("source-extractor", "sex", "sextractor")


def _resolve(candidate: str) -> str | None:
    """Return an executable path for ``candidate``, or ``None`` if unusable."""
    if os.sep in candidate:
        path = Path(candidate)
        return candidate if path.is_file() and os.access(path, os.X_OK) else None
    return shutil.which(candidate)


def require(name: str, candidates: Sequence[str]) -> str:
    """Find the first usable executable among ``candidates``.

    Raises :class:`~robo43.exceptions.ExternalToolError` naming ``name`` and
    listing ``candidates`` if none is found on ``PATH``.
    """
    for candidate in candidates:
        resolved = _resolve(candidate)
        if resolved is not None:
            return resolved
    tried = ", ".join(candidates)
    msg = f"{name} not found (tried: {tried}). Install it and ensure it is on PATH."
    raise ExternalToolError(msg)


def find_sextractor(explicit: str | None = None) -> str:
    """Locate the source-extractor executable.

    If ``explicit`` is given, only that name/path is checked. Otherwise each
    of :data:`SEXTRACTOR_CANDIDATES` is tried in order via ``PATH`` lookup.
    """
    if explicit is not None:
        resolved = _resolve(explicit)
        if resolved is not None:
            return resolved
        msg = (
            f"source-extractor not found: {explicit!r} is not an executable. "
            "Install it with `apt install source-extractor` (Debian) or point "
            "to a valid binary."
        )
        raise ExternalToolError(msg)
    try:
        return require("source-extractor", SEXTRACTOR_CANDIDATES)
    except ExternalToolError as exc:
        tried = ", ".join(SEXTRACTOR_CANDIDATES)
        msg = (
            f"source-extractor not found (tried: {tried}). "
            "Install it with `apt install source-extractor` (Debian)."
        )
        raise ExternalToolError(msg) from exc


def sextractor_version(exe: str | None = None) -> str | None:
    """Best-effort ``source-extractor`` version string, or ``None`` if unavailable."""
    try:
        resolved = find_sextractor(explicit=exe)
    except ExternalToolError:
        return None
    try:
        result = subprocess.run(
            [resolved, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    first_line = result.stdout.splitlines()[0].strip()
    return first_line or None
