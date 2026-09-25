"""``robo43 inspect`` -- preflight validation and override-file generation."""

from __future__ import annotations

import argparse
import logging

from robo43.cli.common import settings_from_args
from robo43.cli.registry import Subcommand, register
from robo43.instruments.keywords import CANONICAL_IMAGETYP
from robo43.instruments.profile import resolve_profile
from robo43.io.fits_io import iter_fits_files, read_header
from robo43.io.headers import read_meta
from robo43.io.validate import STAGE_REQUIREMENTS, OverrideSet, ValidationReport, validate

__all__ = ["SUBCOMMAND"]

logger = logging.getLogger(__name__)

_STAGE_APPLIES_TO: dict[str, tuple[str, ...]] = {
    "bias": ("ZERO",),
    "dark": ("DARK",),
    "flat": ("FLAT",),
    "reduce": ("FLAT", "OBJECT"),
    "astrometry": ("OBJECT",),
    "photometry": ("OBJECT",),
    "zeropoint": ("OBJECT",),
}


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("inputs", nargs="+")
    parser.add_argument(
        "--stages",
        default=None,
        help="comma-separated subset of: " + ", ".join(sorted(STAGE_REQUIREMENTS)),
    )
    parser.add_argument("--emit-overrides", default=None, metavar="PATH")


def _run(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)

    # Build overrides: file first, then CLI (CLI wins on merge)
    overrides = None
    if args.overrides:
        overrides = OverrideSet.from_toml(args.overrides)
    if args.set:
        cli_ov = OverrideSet.from_cli(args.set)
        overrides = cli_ov if overrides is None else overrides.merge(cli_ov)

    # Determine the set of stages to validate
    stages = tuple(args.stages.split(",")) if args.stages else tuple(STAGE_REQUIREMENTS)

    # Collect all metas, resilient to per-file errors
    metas = []
    for path in iter_fits_files(args.inputs):
        try:
            # Auto-resolve profile per file (two-read mode)
            file_profile = None
            if args.instrument:
                file_profile = resolve_profile(name=args.instrument)
            else:
                file_profile = resolve_profile(header=read_header(path, profile=None))

            meta = read_meta(path, file_profile, overrides=overrides, settings=settings)
            metas.append(meta)
        except Exception as exc:
            logger.warning("%s: could not read metadata: %s", path, exc)
            continue

    # Validate, scoped by stage applicability
    all_issues = []
    for stage in stages:
        imagetyps = _STAGE_APPLIES_TO[stage]
        subset = tuple(
            m for m in metas if CANONICAL_IMAGETYP.get(m.frame_type) in imagetyps
        )
        report = validate(subset, stages=(stage,), strict=settings.runtime.strict)
        all_issues.extend(report.issues)

    # Build final report
    final_report = ValidationReport(metas=tuple(metas), issues=tuple(all_issues))

    # Print report
    print(final_report.as_table())

    # Emit overrides template if requested
    if args.emit_overrides:
        final_report.emit_overrides_template(args.emit_overrides)
        print(f"wrote override template to {args.emit_overrides}")

    return 0 if final_report.ok else 1


SUBCOMMAND = Subcommand(
    name="inspect",
    help="Preflight validation and override-file generation.",
    add_arguments=_add_arguments,
    run=_run,
)
register(SUBCOMMAND)
