"""``robo43 reduce`` -- reduce raw science frames to processed products.

Orchestrates the full reduction pipeline: inventories input frames, detects or
applies overscan geometry, builds or reuses calibration masters (bias, darks,
flats), reduces all valid science frames to *_proc.fits, writes a status CSV,
and appends one entry per frame to change_log.jsonl recording every change
made to its data and header (see robo43.reduce.changelog).
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from robo43.calib.bias import build_master_bias, select_frames
from robo43.calib.dark import build_master_darks
from robo43.calib.flat import build_master_flats, group_flats_by_filter
from robo43.calib.gain import DEFAULT_GAIN_MEASUREMENTS_NAME, GainMeasurementSet, measure_gain
from robo43.calib.masks import MaskSet
from robo43.calib.overscan import detect_overscan
from robo43.calib.set import CalibrationSet
from robo43.cli.common import configure_from_args
from robo43.cli.registry import Subcommand, register
from robo43.instruments.profile import resolve_profile
from robo43.io.fits_io import iter_fits_files, read_header
from robo43.io.headers import read_meta
from robo43.io.validate import OverrideSet
from robo43.reduce.changelog import append_change_log
from robo43.reduce.frame import reduce_frame

if TYPE_CHECKING:
    from robo43.io.meta import FrameMeta


__all__ = ["SUBCOMMAND"]

logger = logging.getLogger(__name__)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("inputs", nargs="+", help="raw frame files or directories")
    parser.add_argument(
        "--calibdir",
        default=None,
        help="directory for masters (default: settings.resolved_calibdir or first input dir)",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="directory for products and CSV (default: settings or first input dir)",
    )
    parser.add_argument(
        "--no-overscan",
        action="store_true",
        dest="no_overscan",
        help="skip overscan detection, correction, and trimming",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=None,
        help="reduce only these filters (repeatable; default: all filters)",
    )


def _run(args: argparse.Namespace) -> int:
    settings = configure_from_args(args)

    # Resolve directories
    first_input = Path(args.inputs[0])
    calibdir_path = Path(args.calibdir) if args.calibdir else (
        Path(settings.paths.resolved_calibdir()) if settings.paths else first_input.parent
    )
    outdir_path = Path(args.outdir) if args.outdir else (
        Path(settings.paths.resolved_outdir()) if settings.paths else first_input.parent
    )

    # Collect input files to discover profile from first file if needed
    logger.info("inventorying input frames from %d locations", len(args.inputs))
    all_paths = list(iter_fits_files(args.inputs, recursive=False))
    logger.info("found %d FITS files", len(all_paths))
    if not all_paths:
        logger.error("no FITS files found")
        return 2

    # Resolve instrument profile
    try:
        if args.instrument:
            profile = resolve_profile(name=args.instrument)
            logger.info(
                "instrument profile '%s' resolved from --instrument argument",
                args.instrument,
            )
        else:
            first_file_header = read_header(all_paths[0], profile=None)
            profile = resolve_profile(header=first_file_header)
            logger.info(
                "instrument profile '%s' resolved from data header", profile.name
            )
    except Exception as exc:
        logger.error("failed to resolve instrument profile: %s", exc)
        return 2

    # Parse metadata overrides
    overrides = None
    if args.overrides:
        try:
            overrides = OverrideSet.from_toml(args.overrides)
        except Exception as exc:
            logger.error("failed to parse overrides: %s", exc)
            return 2

    metas: list[FrameMeta] = []
    for path in all_paths:
        try:
            meta = read_meta(path, profile, overrides=overrides, settings=settings)
            metas.append(meta)
        except Exception as exc:
            logger.warning("failed to read metadata for %s: %s", path, exc)

    logger.info("successfully read %d frame metadata", len(metas))

    # Summarize frame types
    frame_type_counts: dict[str, int] = {}
    for meta in metas:
        frame_type = meta.frame_type or "unknown"
        frame_type_counts[frame_type] = frame_type_counts.get(frame_type, 0) + 1
    logger.info("frame inventory: %s", frame_type_counts)

    # Overscan detection
    overscan_geometry = None
    if not args.no_overscan:
        try:
            overscan_geometry = detect_overscan(
                metas,
                settings=settings,
                profile=profile,
                max_flats=3,
                max_biases=2,
            )
            logger.info(
                "overscan geometry: %d columns, %d rows (source: %s)",
                len(overscan_geometry.columns),
                len(overscan_geometry.rows),
                overscan_geometry.source,
            )
        except Exception as exc:
            logger.error("failed to detect overscan: %s", exc)
            return 2
    else:
        logger.info("overscan detection disabled via --no-overscan")

    # Dry-run mode: report plan and exit
    if args.dry_run:
        logger.info("=== DRY RUN MODE ===")

        # Report what masters would be built/reused
        bias_metas = select_frames(metas, "bias")
        dark_metas = select_frames(metas, "dark")
        flat_metas = select_frames(metas, "flat")
        science_metas = select_frames(metas, "science")

        bias_master_path = calibdir_path / "master_bias.fits"
        if bias_metas:
            if bias_master_path.exists():
                logger.info("master_bias.fits: would REUSE")
            else:
                logger.info("master_bias.fits: would BUILD from %d frames", len(bias_metas))
        else:
            logger.info("master_bias.fits: no bias frames, would SKIP")

        if dark_metas:
            dark_files = list(calibdir_path.glob("master_dark_*.fits"))
            if dark_files:
                logger.info("master darks: would REUSE %d files", len(dark_files))
            else:
                logger.info("master darks: would BUILD from %d frames", len(dark_metas))
        else:
            logger.info("master darks: no dark frames, would SKIP")

        flat_files = list(calibdir_path.glob("master_flat_*.fits"))
        if flat_metas:
            if flat_files:
                logger.info("master flats: would REUSE %d files", len(flat_files))
            else:
                logger.info("master flats: would BUILD")
        else:
            logger.info("master flats: no flat frames, would SKIP")

        # Filter science frames by --filter if given
        if args.filter:
            filtered_science = [m for m in science_metas if m.filter in args.filter]
            logger.info(
                "science frames: would reduce %d/%d (filter: %s)",
                len(filtered_science),
                len(science_metas),
                ",".join(args.filter),
            )
        else:
            logger.info("science frames: would reduce %d", len(science_metas))

        # Count invalid frames
        invalid_count = sum(1 for m in science_metas if not m.is_valid)
        if invalid_count:
            logger.info("invalid frames: would SKIP %d", invalid_count)

        logger.info("=== END DRY RUN ===")
        return 0

    # Create output directories
    try:
        calibdir_path.mkdir(parents=True, exist_ok=True)
        outdir_path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.error("failed to create directories: %s", exc)
        return 2

    # Gain measurement (runtime photon-transfer from this night's own flats
    # and biases; see robo43.calib.gain). Cached alongside the other masters
    # so a rerun does not re-measure. A group with no measurement (e.g. no
    # flats/biases at that GAIN setting) is absent from the set entirely --
    # reduce_frame() then reduces frames in that group in ADU.
    gain_file = calibdir_path / DEFAULT_GAIN_MEASUREMENTS_NAME
    if gain_file.exists():
        logger.info("%s exists, loading cached gain measurements", gain_file.name)
        gain_measurements = GainMeasurementSet.read(gain_file)
    else:
        try:
            gain_measurements = measure_gain(
                metas, settings=settings, profile=profile, overscan=overscan_geometry
            )
            gain_measurements.write(gain_file)
            logger.info(
                "measured gain for %d group(s): %s",
                len(gain_measurements.measurements),
                sorted(gain_measurements.measurements),
            )
        except Exception as exc:
            logger.error("failed to measure gain: %s", exc)
            gain_measurements = GainMeasurementSet({})

    # Build master bias
    bias_master = None
    bias_metas = select_frames(metas, "bias")
    if bias_metas:
        bias_master_path = calibdir_path / "master_bias.fits"
        if bias_master_path.exists():
            logger.info("master_bias.fits exists, loading cached version")
            try:
                from robo43.calib.master import MasterFrame
                bias_master = MasterFrame.load(bias_master_path)
            except Exception as exc:
                logger.warning("failed to load cached master_bias: %s", exc)
                logger.info("rebuilding master_bias")
                try:
                    bias_master = build_master_bias(
                        bias_metas,
                        settings=settings,
                        profile=profile,
                        outdir=calibdir_path,
                        overscan=overscan_geometry,
                        gain_measurements=gain_measurements,
                    )
                except Exception as exc:
                    logger.error("failed to build master_bias: %s", exc)
                    return 2
        else:
            logger.info("building master_bias from %d frames", len(bias_metas))
            try:
                bias_master = build_master_bias(
                    bias_metas,
                    settings=settings,
                    profile=profile,
                    outdir=calibdir_path,
                    overscan=overscan_geometry,
                    gain_measurements=gain_measurements,
                )
            except Exception as exc:
                logger.error("failed to build master_bias: %s", exc)
                return 2
    else:
        logger.warning("no bias frames found; continuing without master bias")

    # Build master darks
    dark_metas = select_frames(metas, "dark")
    if dark_metas:
        # Check if dark masters already exist
        dark_files = list(calibdir_path.glob("master_dark_*.fits"))
        dark_current_file = calibdir_path / "dark_current_map.fits"

        if dark_files and dark_current_file.exists():
            logger.info(
                "dark masters exist, loading cached versions (files: %s, dark_current_map.fits)",
                ", ".join(f.name for f in dark_files),
            )
        else:
            logger.info("building master darks from %d frames", len(dark_metas))
            try:
                _dark_masters, _dark_current_map = build_master_darks(
                    dark_metas,
                    settings=settings,
                    profile=profile,
                    outdir=calibdir_path,
                    bias=bias_master,
                    gain_measurements=gain_measurements,
                )
            except Exception as exc:
                logger.error("failed to build master_darks: %s", exc)
                return 2
    else:
        logger.info("no dark frames found; continuing without master darks")

    # Build master flats
    flat_metas = select_frames(metas, "flat")
    if flat_metas:
        # Determine which filters are represented and which masters already exist
        separate_dome_and_sky = settings.calib.separate_dome_and_sky_flats
        flat_groups = group_flats_by_filter(flat_metas, separate_dome_and_sky=separate_dome_and_sky)

        if flat_groups:
            # Check which filters need to be built
            filters_to_build = []
            filters_reused = []

            for group_key in flat_groups:
                # A filter's master already exists if ANY of these files is present:
                # master_flat_<group_key>.fits, master_flat_<group_key>_dome.fits,
                # master_flat_<group_key>_sky.fits
                # Note: group_key already has ':' replaced with '_' in the actual filename
                filename_key = group_key.replace(":", "_")
                flat_master_files = list(calibdir_path.glob(f"master_flat_{filename_key}*.fits"))

                if flat_master_files:
                    filters_reused.append(group_key)
                else:
                    filters_to_build.append(group_key)

            # Filter flat_metas to only include frames for filters that need building
            flat_metas_to_build = [
                m for m in flat_metas
                if (m.filter is not None and (
                    m.filter in filters_to_build or
                    f"{m.filter}:dome" in filters_to_build or
                    f"{m.filter}:sky" in filters_to_build
                ))
            ]

            if filters_reused:
                logger.info("master flats exist and will be reused: %s", ", ".join(filters_reused))

            if flat_metas_to_build:
                logger.info("building master flats for filters: %s (from %d frames)",
                           ", ".join(filters_to_build), len(flat_metas_to_build))

                # Load mask if present
                mask_obj = None
                mask_path = calibdir_path / "mask.fits"
                if mask_path.exists():
                    try:
                        mask_obj = MaskSet.read(mask_path)
                        logger.info("loaded bad-pixel mask from %s", mask_path)
                    except Exception as exc:
                        logger.warning("failed to load mask from %s: %s", mask_path, exc)

                try:
                    _flat_masters = build_master_flats(
                        flat_metas_to_build,
                        settings=settings,
                        profile=profile,
                        outdir=calibdir_path,
                        bias=bias_master,
                        mask=mask_obj,
                        overscan=overscan_geometry,
                        gain_measurements=gain_measurements,
                    )
                except Exception as exc:
                    logger.error("failed to build master_flats: %s", exc)
                    return 2
            else:
                logger.info("all master flats already exist, skipping build")
        else:
            logger.info("no flat frames found; continuing without master flats")
    else:
        logger.info("no flat frames found; continuing without master flats")

    # Load calibration set
    try:
        filters_to_load = args.filter if args.filter else None
        calib_set = CalibrationSet.load(
            calibdir_path,
            settings=settings,
            profile=profile,
            filters=filters_to_load,
        )
    except Exception as exc:
        logger.error("failed to load calibration set: %s", exc)
        return 2

    # Reduce science frames
    science_metas = select_frames(metas, "science")
    science_metas_valid = [m for m in science_metas if m.is_valid]

    if science_metas and len(science_metas) != len(science_metas_valid):
        invalid_count = len(science_metas) - len(science_metas_valid)
        logger.warning("skipping %d invalid frames (VALID=False)", invalid_count)

    # Filter by --filter if given
    if args.filter:
        science_metas_valid = [m for m in science_metas_valid if m.filter in args.filter]

    logger.info("reducing %d science frames", len(science_metas_valid))

    statuses: list = []
    n_success = 0
    n_degraded = 0
    n_failed = 0

    for sci_meta in science_metas_valid:
        try:
            product, status = reduce_frame(
                sci_meta,
                calib_set,
                settings=settings,
                profile=profile,
                overscan=overscan_geometry,
            )

            # Determine output filename
            raw_stem = sci_meta.path.stem.rsplit(".", 1)[0]  # Remove .fits if present
            if raw_stem.endswith(".fit"):
                raw_stem = raw_stem[:-4]
            proc_filename = f"{raw_stem}_proc.fits"
            proc_path = outdir_path / proc_filename

            # Write product
            try:
                product.write(proc_path, overwrite=settings.runtime.clobber)
                status.product_path = proc_path
            except Exception as exc:
                logger.error("failed to write %s: %s", proc_path, exc)
                status.proc_code = 4
                status.proc_status = "FAILED"
                status.error = str(exc)

            # Append this frame's entry to the per-frame change log. Never
            # allowed to fail the reduction: append_change_log catches its
            # own errors, but re-reading the raw header here can also raise
            # (e.g. the raw file vanished between reduction and now), so
            # that is guarded too.
            try:
                raw_header_for_log = read_header(sci_meta.path, profile)
                append_change_log(
                    outdir_path, sci_meta, raw_header_for_log, product.header, status
                )
            except Exception as exc:
                logger.warning(
                    "failed to write change-log entry for %s: %s", sci_meta.path, exc
                )

            # Tally status
            if status.proc_code == 0:
                n_success += 1
            elif status.proc_code == 3:
                n_degraded += 1
            elif status.proc_code == 4:
                n_failed += 1

            statuses.append(status.as_row())

        except Exception as exc:
            logger.error("failed to reduce %s: %s", sci_meta.path, exc)
            n_failed += 1
            # Create a minimal status record
            statuses.append({
                "path": str(sci_meta.path),
                "raw_name": sci_meta.path.name,
                "proc_code": 4,
                "proc_status": "FAILED",
                "steps_applied": "",
                "steps_skipped": "",
                "error": str(exc),
                "product_path": None,
                "wcs_solved": False,
                "astrom_rms_arcsec": None,
                "fwhm_arcsec": None,
                "zeropoint": None,
                "is_photometric": None,
                "duration_s": None,
            })

    # Write status CSV
    if statuses:
        csv_path = outdir_path / "proc_status.csv"
        try:
            fieldnames = statuses[0].keys()
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(statuses)
            logger.info("wrote status CSV: %s", csv_path)
        except Exception as exc:
            logger.error("failed to write status CSV: %s", exc)
            return 2

    # Print summary
    logger.info(
        "reduction complete: %d success, %d degraded, %d failed out of %d frames",
        n_success,
        n_degraded,
        n_failed,
        len(science_metas_valid),
    )

    return 0


SUBCOMMAND = Subcommand(
    name="reduce",
    help="Reduce raw science frames to processed products.",
    add_arguments=_add_arguments,
    run=_run,
)
register(SUBCOMMAND)
