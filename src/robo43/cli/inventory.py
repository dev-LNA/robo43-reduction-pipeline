"""``robo43 inventory`` -- build a CSV/Parquet inventory of raw FITS files."""

from __future__ import annotations

import argparse
import logging

from robo43.cli.common import settings_from_args
from robo43.cli.registry import Subcommand, register
from robo43.instruments.profile import resolve_profile
from robo43.io.inventory import build_inventory, filter_inventory, write_inventory
from robo43.io.validate import OverrideSet

__all__ = ["SUBCOMMAND"]

logger = logging.getLogger(__name__)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("inputs", nargs="+", help="raw frame files, directories, or list files")
    parser.add_argument("-o", "--output", default=None, help="write to this .csv/.parquet path")
    parser.add_argument("--frame-type", default=None, help="ZERO/DARK/OBJECT/FLAT/FOCUS")
    parser.add_argument("--filter", dest="filter_", default=None)
    parser.add_argument("--object", dest="object_", default=None)
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="keep frames whose header marks them VALID = False",
    )


def _run(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)

    # Build overrides: file first, then CLI (CLI wins on merge)
    overrides = None
    if args.overrides:
        overrides = OverrideSet.from_toml(args.overrides)
    if args.set:
        cli_ov = OverrideSet.from_cli(args.set)
        overrides = cli_ov if overrides is None else overrides.merge(cli_ov)

    # Resolve profile: None for auto-resolution, or explicit name
    profile = resolve_profile(name=args.instrument) if args.instrument else None

    # Build inventory with auto-resolution when profile is None
    df = build_inventory(
        args.inputs,
        profile=profile,
        overrides=overrides,
        settings=settings,
        nprocs=settings.runtime.nprocs,
    )

    # Determine valid_only from --include-invalid flag
    valid_only = not args.include_invalid

    # Count invalid frames if excluding them
    if valid_only:
        df_all = filter_inventory(
            df,
            frame_type=args.frame_type,
            filter=args.filter_,
            object=args.object_,
            valid_only=False,
        )
        n_invalid = len(df_all) - len(
            filter_inventory(
                df,
                frame_type=args.frame_type,
                filter=args.filter_,
                object=args.object_,
                valid_only=True,
            )
        )
        if n_invalid > 0:
            logger.info(
                "excluded %d frame(s) marked VALID = False (use --include-invalid to keep them)",
                n_invalid,
            )

    # Filter by frame_type, filter, object
    df = filter_inventory(
        df,
        frame_type=args.frame_type,
        filter=args.filter_,
        object=args.object_,
        valid_only=valid_only,
    )

    # Write or print
    if args.output:
        write_inventory(df, args.output)
        print(f"wrote {len(df)} rows to {args.output}")
    else:
        print(df.to_string(index=False))

    return 0


SUBCOMMAND = Subcommand(
    name="inventory",
    help="Build a CSV/Parquet inventory of raw FITS files.",
    add_arguments=_add_arguments,
    run=_run,
)
register(SUBCOMMAND)
