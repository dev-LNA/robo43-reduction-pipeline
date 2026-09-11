"""Flags shared by every ``robo43`` subcommand.

``--verbose`` and ``--loglevel`` used to interact badly (``--loglevel DEBUG``
alone did nothing unless ``--verbose`` was also given); this module resolves
that aliasing in one place so every subcommand behaves the same way.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

from robo43.config import Settings, load_settings
from robo43.logging_setup import setup_logging

if TYPE_CHECKING:
    pass

__all__ = [
    "add_common_arguments",
    "cli_overrides",
    "configure_from_args",
    "settings_from_args",
]


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="path to a config.toml file")
    parser.add_argument(
        "--loglevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="explicit log level; overrides config/env when given",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="alias for --loglevel DEBUG",
    )
    parser.add_argument("--logfile", default=None, help="path to write log records to")
    parser.add_argument(
        "--instrument",
        default=None,
        help="instrument profile name (consumed by a later phase)",
    )
    parser.add_argument(
        "--overrides",
        default=None,
        help="path to an overrides TOML file (consumed by a later phase)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="ad-hoc override, repeatable (consumed by a later phase)",
    )
    parser.add_argument(
        "--assume-profile-defaults",
        action="store_true",
        default=False,
    )
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument("--strict", action="store_true", default=False)
    strict_group.add_argument("--permissive", action="store_true", default=False)
    parser.add_argument(
        "--diagnostics",
        choices=["none", "summary", "full"],
        default=None,
    )
    parser.add_argument("--np", type=int, default=None, help="number of worker processes")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--clobber", action="store_true", default=False)
    parser.add_argument("--pdb", action="store_true", default=False)


def cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    cli: dict[str, Any] = {}

    effective_loglevel = "DEBUG" if args.verbose else args.loglevel
    if effective_loglevel is not None:
        cli["runtime.loglevel"] = effective_loglevel

    if args.np is not None:
        cli["runtime.nprocs"] = args.np
    if args.strict:
        cli["runtime.strict"] = True
    elif args.permissive:
        cli["runtime.strict"] = False
    if args.assume_profile_defaults:
        cli["runtime.assume_profile_defaults"] = True
    if args.dry_run:
        cli["runtime.dry_run"] = True
    if args.clobber:
        cli["runtime.clobber"] = True
    if args.pdb:
        cli["runtime.pdb"] = True
    if args.diagnostics is not None:
        cli["diagnostics.level"] = args.diagnostics
    if args.logfile is not None:
        cli["runtime.logfile"] = args.logfile

    return cli


def settings_from_args(args: argparse.Namespace) -> Settings:
    cli = cli_overrides(args)
    return load_settings(config_file=args.config, cli=cli)


def configure_from_args(args: argparse.Namespace) -> Settings:
    settings = settings_from_args(args)
    setup_logging(
        level=settings.runtime.loglevel,
        logfile=settings.runtime.logfile,
        json_format=False,
    )
    return settings
