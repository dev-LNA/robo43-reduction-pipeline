"""A registry subcommand modules publish themselves into on import.

Each ``robo43/cli/<name>.py`` module builds a :class:`Subcommand` and calls
:func:`register` at module import time.  ``robo43.cli.main`` imports those
modules lazily and then reads back whatever registered itself successfully,
so an optional subcommand whose dependency is missing simply never appears.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

__all__ = ["Subcommand", "all_subcommands", "register"]


@dataclass
class Subcommand:
    name: str
    help: str
    add_arguments: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace], int]


_REGISTRY: dict[str, Subcommand] = {}


def register(cmd: Subcommand) -> None:
    if cmd.name in _REGISTRY:
        msg = f"subcommand {cmd.name!r} is already registered"
        raise ValueError(msg)
    _REGISTRY[cmd.name] = cmd


def all_subcommands() -> list[Subcommand]:
    return [_REGISTRY[name] for name in sorted(_REGISTRY)]
