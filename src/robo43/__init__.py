"""ROBO43 reduction pipeline.

A telescope-agnostic photometric reduction pipeline. See ``REFACTORING_PLAN.md``
for the architecture this package implements.
"""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path

__version__ = "0.2.0.dev0"

__all__ = ["__version__", "git_sha"]


@functools.lru_cache(maxsize=1)
def git_sha() -> str | None:
    """Short git SHA of the checkout this package was imported from.

    Returns ``None`` for an installed copy that is not inside a git work tree,
    which is the normal case for a released wheel.  The value is recorded in
    every product header as ``PIPEGIT`` so a product can be traced back to the
    exact code that made it.
    """
    repo = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    sha = out.stdout.strip()
    return sha or None
