"""``robo43 tilegain`` -- derive and check per-tile gain patterns from master flats.

``robo43 tilegain derive --instrument t80s --mode mode5 --grid 8x2 MASTER...``
derives the relative tile gain pattern of a readout mode from the amplifier-
boundary continuity of one or more master flats (each next to its
``gain_measurements.json``, or carrying ``GAIN00..`` cards) and prints the
``[instrument.tile_gain_pattern]`` entry with its diagnostics;
``robo43 tilegain steps ... MASTER [--versus OTHER]`` prints a master's amplifier
steps (or those of ``MASTER / OTHER``), which is how a pattern is validated.
See :mod:`robo43.calib.tilegain` for the method.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from astropy.io import fits

from robo43.calib.tilegain import (
    MARGIN_PX,
    STRIP_PX,
    analyse_master,
    derive_pattern,
    equal_tile_slices,
    format_pattern_toml,
    step_summary,
    write_report_json,
)
from robo43.cli.common import settings_from_args
from robo43.cli.registry import Subcommand, register
from robo43.exceptions import Robo43Error
from robo43.instruments.profile import resolve_profile

__all__ = ["SUBCOMMAND"]

logger = logging.getLogger(__name__)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "action",
        choices=["derive", "steps"],
        help="derive: tile gain pattern of --mode from the masters; "
        "steps: amplifier-step statistics of each master (or of master / --versus)",
    )
    parser.add_argument("masters", nargs="+", help="master flat FITS file(s)")
    parser.add_argument(
        "--mode", default=None, help="readout mode the masters were built in (derive)"
    )
    parser.add_argument("--versus", default=None, help="second master: steps of master / versus")
    geometry = parser.add_mutually_exclusive_group(required=True)
    geometry.add_argument(
        "--grid", default=None, metavar="CxR",
        help="regular tile grid of the trimmed frame, columns x rows (T80S: 8x2)",
    )
    geometry.add_argument(
        "--raw-geometry", action="append", default=None, metavar="PATH",
        help="raw FITS file or directory (repeatable) holding at least one bias and one flat: "
        "the tile slices are then detected exactly as 'robo43 reduce' does",
    )
    parser.add_argument("--strip", type=int, default=STRIP_PX, help="strip width (px)")
    parser.add_argument("--margin", type=int, default=MARGIN_PX, help="boundary end margin (px)")
    parser.add_argument("--report-json", default=None, help="write the derivation as JSON (derive)")
    parser.add_argument(
        "--toml-out", default=None, help="write the TOML entry to this file (derive)"
    )


def _tile_slices(args, profile, shape):
    if args.grid is not None:
        try:
            n_cols, n_rows = (int(v) for v in args.grid.lower().split("x"))
        except ValueError as exc:
            msg = f"--grid must look like 8x2, got {args.grid!r}"
            raise Robo43Error(msg) from exc
        if n_cols * n_rows != profile.layout.n_amplifiers:
            msg = (
                f"--grid {args.grid} has {n_cols * n_rows} tiles but profile "
                f"{profile.name!r} declares {profile.layout.n_amplifiers} amplifiers"
            )
            raise Robo43Error(msg)
        try:
            return equal_tile_slices(shape, n_cols, n_rows)
        except ValueError as exc:
            raise Robo43Error(str(exc)) from exc
    from robo43.calib.overscan import detect_overscan
    from robo43.io.fits_io import iter_fits_files
    from robo43.io.headers import read_meta

    settings = settings_from_args(args)
    metas = [
        read_meta(path, profile, overrides=None, settings=settings)
        for raw in args.raw_geometry
        for path in iter_fits_files([raw])
    ]
    slices = detect_overscan(metas, settings=settings, profile=profile).science_tile_slices
    if len(slices) != profile.layout.n_amplifiers:
        msg = (
            f"detected {len(slices)} tile(s) from --raw-geometry, profile "
            f"{profile.name!r} declares {profile.layout.n_amplifiers} amplifiers"
        )
        raise Robo43Error(msg)
    return slices


def _master_shape(path: str) -> tuple[int, int]:
    header = fits.getheader(path, 0)
    return int(header["NAXIS2"]), int(header["NAXIS1"])


def _run(args: argparse.Namespace) -> int:
    if not args.instrument:
        msg = "--instrument is required (the tile count comes from the profile)"
        raise Robo43Error(msg)
    profile = resolve_profile(name=args.instrument)
    slices = _tile_slices(args, profile, _master_shape(args.masters[0]))

    if args.action == "steps":
        for master in args.masters:
            result = step_summary(master, slices, versus=args.versus,
                                  strip=args.strip, margin=args.margin)
            label = f"{Path(master).name} / {Path(args.versus).name}" if args.versus else master
            print(
                f"{label}: {result.n_steps} steps, rms {result.rms_pct:.4f} %, "
                f"max |step| {result.max_abs_pct:.4f} %, null rms {result.null_rms_pct:.4f} %"
            )
        return 0

    if not args.mode:
        msg = "derive needs --mode (the readout mode the masters were built in)"
        raise Robo43Error(msg)
    analyses = []
    for master in args.masters:
        header_mode = str(fits.getheader(master, 0).get("READOUTM", "UNKNOWN")).strip()
        if header_mode not in (args.mode, "UNKNOWN"):
            logger.warning("%s: READOUTM %r differs from --mode %r", master, header_mode, args.mode)
        analysis = analyse_master(master, slices, strip=args.strip, margin=args.margin)
        analyses.append(analysis)
        used = analysis.used_gain_over_pattern_rms_pct
        print(
            f"{master}: gains from {analysis.gain_source}; step rms ADU "
            f"{analysis.adu_steps.rms_pct:.3f} % / stored {analysis.e_steps.rms_pct:.3f} %; "
            f"fit residual rms {analysis.fit_residual_rms_pct:.3f} % (max "
            f"{analysis.fit_residual_max_pct:.3f} %), null rms "
            f"{analysis.adu_steps.null_rms_pct:.3f} %; "
            + ("ADU master (no gains used)" if used is None
               else f"used gains vs pattern rms {used:.2f} %")
        )
    derivation = derive_pattern(analyses, args.mode)
    print(f"\n{len(analyses)} master(s); per-tile scatter between masters (std of 100 ln p):")
    print("  " + " ".join(f"{v:.3f}" for v in derivation.scatter_pct))
    print(f"  rms {np.sqrt(np.mean(derivation.scatter_pct ** 2)):.3f} %, "
          f"max {derivation.scatter_pct.max():.3f} %")
    if derivation.loo_rms_pct:
        print("leave-one-out rms (%): " + " ".join(f"{v:.3f}" for v in derivation.loo_rms_pct))
    toml = format_pattern_toml(derivation)
    print("\n[instrument.tile_gain_pattern]\n" + toml)
    if args.toml_out:
        Path(args.toml_out).write_text(toml + "\n", encoding="utf-8")
    if args.report_json:
        write_report_json(derivation, args.report_json)
    return 0


SUBCOMMAND = Subcommand(
    name="tilegain",
    help="Derive/check the per-tile gain pattern of a readout mode from master flats.",
    add_arguments=_add_arguments,
    run=_run,
)
register(SUBCOMMAND)
