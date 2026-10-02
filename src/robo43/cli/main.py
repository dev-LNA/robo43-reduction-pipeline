"""The ``robo43`` command-line entry point.

Subcommand modules register themselves with :mod:`robo43.cli.registry` as a
side effect of being imported.  This module imports each candidate module
lazily and tolerates the ones that do not exist yet, or whose optional
dependency is missing, so ``--help`` works no matter which subset of
subcommands is present.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import traceback
from collections.abc import Sequence
from typing import TYPE_CHECKING

import robo43
from robo43.cli.common import add_common_arguments, configure_from_args
from robo43.cli.registry import all_subcommands
from robo43.exceptions import Robo43Error

if TYPE_CHECKING:
    from robo43.cli.registry import Subcommand

__all__ = ["build_parser", "main"]

logger = logging.getLogger(__name__)

SUBCOMMAND_MODULES: tuple[str, ...] = (
    "inventory",
    "inspect",
    "fixheader",
    "calib",
    "reduce",
    "astrometry",
    "coadd",
    "photometry",
    "forced",
    "catalog",
    "colour",
    "thumb",
    "serve",
    "status",
    "requeue",
    "db",
    "config",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robo43")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {robo43.__version__}",
    )
    add_common_arguments(parser)

    for name in SUBCOMMAND_MODULES:
        try:
            importlib.import_module(f"robo43.cli.{name}")
        except ImportError:
            logger.debug("subcommand module %r not available, skipping", name)
            continue

    subparsers = parser.add_subparsers(dest="command", required=True)
    for subcommand in all_subcommands():
        subparser = subparsers.add_parser(subcommand.name, help=subcommand.help)
        add_common_arguments(subparser)
        subcommand.add_arguments(subparser)
        subparser.set_defaults(_subcommand=subcommand)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    subcommand: Subcommand | None = getattr(args, "_subcommand", None)
    if subcommand is None:
        print("error: no subcommand matched", file=sys.stderr)
        return 2

    def _dispatch() -> int:
        configure_from_args(args)
        return subcommand.run(args)

    try:
        if args.pdb:
            import pdb

            try:
                return _dispatch()
            except Exception:
                pdb.post_mortem(sys.exc_info()[2])
                raise
        return _dispatch()
    except KeyboardInterrupt:
        return 130
    except Robo43Error as exc:
        debug = bool(args.verbose or args.loglevel == "DEBUG")
        print(f"error: {exc}", file=sys.stderr)
        if debug:
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
