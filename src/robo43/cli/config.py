"""``robo43 config`` -- inspect the resolved configuration.

This is the first subcommand module in the framework and doubles as its
integration proof: ``robo43.cli.main`` imports it lazily just like every
other subcommand module, and it exercises the full common-flags -> Settings
-> logging pipeline.
"""

from __future__ import annotations

import argparse

from robo43 import config as robo43_config
from robo43.cli.common import cli_overrides, settings_from_args
from robo43.cli.registry import Subcommand, register

__all__ = ["SUBCOMMAND"]


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("action", choices=["show", "path"])


def _show(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    origins = robo43_config.origin_report(
        config_file=args.config,
        cli=cli_overrides(args),
        env=None,
    )
    payload = settings.to_dict()
    for section in sorted(payload):
        print(f"[{section}]")
        values = payload[section]
        for key in sorted(values):
            value = values[key]
            dotted = f"{section}.{key}"
            origin = origins.get(dotted)
            if origin is not None:
                print(f"{key} = {value!r}  # {origin}")
            else:
                print(f"{key} = {value!r}")
        print()
    return 0


def _path(_args: argparse.Namespace) -> int:
    path = robo43_config.USER_CONFIG_PATH
    status = "exists" if path.is_file() else "not found"
    print(f"{path} ({status})")
    return 0


def _run(args: argparse.Namespace) -> int:
    if args.action == "show":
        return _show(args)
    return _path(args)


SUBCOMMAND = Subcommand(
    name="config",
    help="Inspect the resolved configuration.",
    add_arguments=_add_arguments,
    run=_run,
)
register(SUBCOMMAND)
