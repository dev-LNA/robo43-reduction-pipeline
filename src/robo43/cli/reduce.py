"""``robo43 reduce`` -- reduce raw science frames to processed products.

Orchestrates the full reduction pipeline: inventories input frames, detects or
applies overscan geometry, builds or reuses calibration masters (bias, darks,
flats), reduces all valid science frames to *_proc.fits, writes a status CSV,
and appends one entry per frame to change_log.jsonl recording every change
made to its data and header (see robo43.reduce.changelog).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robo43.calib.bias import build_master_bias, select_frames
from robo43.calib.dark import build_master_darks
from robo43.calib.fallback import (
    collect_fallback_frames,
    dominant_camera_config,
    parse_night_dirname,
    pool_flat_frames,
)
from robo43.calib.flat import build_master_flats, group_flats_by_filter
from robo43.calib.gain import DEFAULT_GAIN_MEASUREMENTS_NAME, GainMeasurementSet, measure_gain
from robo43.calib.masks import MaskSet, build_mask, read_cold_pixel_masks
from robo43.calib.master import MasterFrame
from robo43.calib.overscan import detect_overscan
from robo43.calib.pool import (
    WindowPoolResult,
    apply_window_pools,
    enforce_single_bias_mode,
    infer_science_night,
    install_master_flat,
    limit_bias_for_gain,
)
from robo43.calib.set import CalibrationSet
from robo43.cli.common import configure_from_args
from robo43.cli.registry import Subcommand, register
from robo43.exceptions import ConfigError
from robo43.instruments.profile import resolve_profile
from robo43.io.fits_io import iter_fits_files, read_header
from robo43.io.headers import read_meta
from robo43.io.validate import OverrideSet
from robo43.logging_setup import log_listener, worker_log_config
from robo43.memory import STAGE_BYTES_PER_PIXEL, log_peak_rss_summary, release_memory, safe_worker_count
from robo43.reduce.changelog import append_change_log
from robo43.reduce.frame import reduce_frame

if TYPE_CHECKING:
    from robo43.calib.overscan import OverscanGeometry
    from robo43.config import Settings
    from robo43.instruments.profile import InstrumentProfile
    from robo43.io.meta import FrameMeta


__all__ = ["SUBCOMMAND"]

logger = logging.getLogger(__name__)


def _compose_static_mask(calibdir_path: Path, settings: Settings) -> None:
    """Write ``calibdir/mask.fits`` from hot (darks) and cold (flats) pixels.

    Reuses an existing ``mask.fits`` unless a dark-current map or cold-pixel
    file is newer.  Hot pixels come from the dark-current map, cold pixels
    from every ``coldpix_*.fits``; with neither, no mask is written.  Never
    aborts the reduction: any failure is logged and the run continues without
    a static mask.
    """
    mask_path = calibdir_path / "mask.fits"
    if mask_path.exists():
        sources = [
            p
            for p in (
                calibdir_path / "dark_current_map.fits",
                calibdir_path / "master_dark_current.fits",
                *calibdir_path.glob("coldpix_*.fits"),
            )
            if p.exists()
        ]
        mask_mtime = mask_path.stat().st_mtime_ns
        if all(p.stat().st_mtime_ns <= mask_mtime for p in sources):
            logger.info("mask.fits exists, reusing")
            return
        logger.info("mask.fits is older than its dark/cold-pixel inputs, rebuilding")
    try:
        hot = None
        dc_file = calibdir_path / "dark_current_map.fits"
        if not dc_file.exists():
            dc_file = calibdir_path / "master_dark_current.fits"
        if dc_file.exists():
            hot = MasterFrame.load(dc_file).data
        cold, cold_files = read_cold_pixel_masks(calibdir_path)
        if hot is not None and cold is not None and hot.shape != cold.shape:
            logger.error(
                "dark-current map shape %s does not match cold-pixel mask shape %s; "
                "ignoring the dark-current map for mask.fits",
                hot.shape,
                cold.shape,
            )
            hot = None
        if hot is None and cold is None:
            logger.info("no dark-current map or cold-pixel mask; not writing mask.fits")
            return
        source = "hot+cold" if hot is not None and cold is not None else (
            "cold" if cold is not None else "hot"
        )
        mask = build_mask(dark_current_map=hot, cold_pixel_mask=cold, settings=settings.calib)
        cards: list[tuple[str, Any, str]] = [
            ("MASKSRC", source, "Static mask sources (hot=darks, cold=flats)"),
            ("COLDNFIL", len(cold_files), "Number of cold-pixel mask files"),
        ]
        for i, name in enumerate(cold_files[:99], start=1):
            cards.append((f"COLDF{i:02d}", name, "Cold-pixel mask file"))
        mask.write(mask_path, cards=cards, overwrite=True)
        logger.info("wrote %s (sources: %s)", mask_path, source)
        logger.info("%s", mask.text_summary())
    except Exception as exc:
        logger.error("failed to build mask.fits: %s; continuing without a static mask", exc)


#: Populated once per worker -- by ``_init_science_worker`` in a pooled
#: (``nprocs > 1``) run, or directly by ``_run`` for the ``nprocs == 1``
#: in-process fallback -- so ``reduce_frame_worker`` never has to receive
#: them as per-task arguments. ``calib_set`` in particular (master bias,
#: flats, gain maps: several hundred MB) must never be pickled once per
#: frame; re-loading it from ``calibdir_path`` once per worker process,
#: inside the pool's initializer, avoids that without relying on any
#: fork-specific memory sharing (see ``_init_science_worker``'s docstring for
#: why "fork" is not used here).
_worker_calib: CalibrationSet | None = None
_worker_settings: Settings | None = None
_worker_profile: InstrumentProfile | None = None
_worker_overscan: OverscanGeometry | None = None
_worker_outdir: Path | None = None
_worker_changelog_lock: Any = None


def _init_science_worker(
    calibdir_path: Path,
    settings: Settings,
    profile: InstrumentProfile,
    overscan_geometry: OverscanGeometry | None,
    outdir_path: Path,
    filters: list[str] | None,
    changelog_lock: Any,
    log_queue: Any,
    loglevel: str,
) -> None:
    """Pool initializer: build this worker's own ``CalibrationSet`` once.

    Runs exactly once per worker process (as long as the pool's
    ``max_tasks_per_child`` stays at its default of "never recycle"), not
    once per frame. ``calib_set`` is loaded fresh from ``calibdir_path``
    here -- the same call ``_run`` itself made to build the parent's
    ``calib_set`` -- rather than being pickled from the parent and sent
    through the per-task call queue: the "fork" start method would let a
    worker inherit the parent's already-loaded ``calib_set`` copy-on-write
    for free, but ``ProcessPoolExecutor`` itself already runs its own
    ``_ExecutorManagerThread`` (and a queue-feeder thread) in the parent by
    the time any worker process is spawned -- a longstanding, acknowledged
    CPython hazard (https://github.com/python/cpython/issues/90622) where a
    'fork' worker can be created from an already multi-threaded parent, with
    a real (if low-probability) deadlock risk. Re-loading per worker instead
    avoids that risk entirely, matching this stage's ``settings``/``profile``
    to whatever the parent resolved, at the cost of ``CalibrationSet.load()``
    running ``nprocs`` times (from files the parent has almost certainly just
    read or written, so mostly page-cache-warm) instead of once.
    """
    global _worker_calib, _worker_settings, _worker_profile
    global _worker_overscan, _worker_outdir, _worker_changelog_lock

    worker_log_config(log_queue, loglevel)
    _worker_calib = CalibrationSet.load(
        calibdir_path, settings=settings, profile=profile, filters=filters
    )
    _worker_settings = settings
    _worker_profile = profile
    _worker_overscan = overscan_geometry
    _worker_outdir = outdir_path
    _worker_changelog_lock = changelog_lock


def reduce_frame_worker(sci_meta: FrameMeta) -> tuple[dict[str, Any], int]:
    """Reduce one science frame, using the module-level globals set by
    ``_init_science_worker`` (pooled run) or directly by ``_run`` (serial
    ``nprocs == 1`` run).

    This is the exact per-frame body the serial science loop used to run
    inline, now called identically either way, so output is identical
    either way. ``changelog_lock`` only serializes concurrent appends onto
    the shared ``change_log.jsonl`` between sibling worker processes; it
    changes nothing about what gets written.

    Returns ``(row, proc_code)``: ``row`` is exactly what
    ``ReductionStatus.as_row()`` produces (or the minimal failure fallback
    below), and ``proc_code`` is returned alongside it so the caller can
    tally n_success/n_degraded/n_failed without re-parsing the row.
    """
    calib_set = _worker_calib
    settings = _worker_settings
    profile = _worker_profile
    overscan_geometry = _worker_overscan
    outdir_path = _worker_outdir
    changelog_lock = _worker_changelog_lock

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
        # that is guarded too. The lock only serializes this append
        # against sibling worker processes writing the same
        # change_log.jsonl; it does not change what gets written.
        try:
            raw_header_for_log = read_header(sci_meta.path, profile)
            with changelog_lock:
                append_change_log(
                    outdir_path, sci_meta, raw_header_for_log, product.header, status
                )
        except Exception as exc:
            logger.warning(
                "failed to write change-log entry for %s: %s", sci_meta.path, exc
            )

        return status.as_row(), status.proc_code

    except Exception as exc:
        logger.error("failed to reduce %s: %s", sci_meta.path, exc)
        row = {
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
        }
        return row, 4


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
    parser.add_argument(
        "--no-calib-fallback",
        action="store_true",
        dest="no_calib_fallback",
        help="disable multi-night calibration fallback for missing bias/dark/flat frames",
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

    # Multi-night calibration fallback (T80S): only active when the input
    # is a single night directory named YYYYMMDD.
    night_dir: Path | None = None
    current_night: date | None = None
    if first_input.is_dir():
        parsed_night = parse_night_dirname(first_input.name)
        if parsed_night is not None:
            night_dir = first_input
            current_night = parsed_night

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

    # Multi-night calibration fallback (T80S): when this run's night is
    # missing bias, dark, or flat (per filter) frames entirely, search
    # strictly earlier sibling YYYYMMDD night directories for matching raw
    # frames and merge them into `metas` so overscan detection, gain
    # measurement, and master-building all see them exactly as if they had
    # been part of this night's own inventory (see robo43.calib.fallback).
    bias_fallback_nights: tuple[str, ...] = ()
    dark_fallback_nights: tuple[str, ...] = ()
    #: group_key -> (contributing nights incl. current night, any borrowed?)
    flat_fallback_source_nights: dict[str, tuple[tuple[str, ...], bool]] = {}
    calib_fallback_enabled = (
        settings.calib.calib_fallback
        and not args.no_calib_fallback
        and night_dir is not None
    )
    if not settings.calib.calib_fallback or args.no_calib_fallback:
        logger.info("multi-night calibration fallback disabled")
    elif night_dir is None:
        logger.info(
            "multi-night calibration fallback inactive: input is not a single "
            "YYYYMMDD night directory"
        )

    # Symmetric-window pooling (robo43.calib.pool), enabled per instrument
    # profile ([instrument.calib] bias_pool / flat_pool_window_nights). It
    # works from the science frames' own directory, so it also applies when
    # the inputs are an explicit file list rather than one night directory;
    # explicit bias/flat inputs are merged into the pools without duplicates.
    # A profile without those keys never enters this block.
    pool_result: WindowPoolResult | None = None
    pool_current_night: date | None = None
    window_pool_requested = profile.calib.bias_pool or (
        profile.calib.flat_pool and profile.calib.flat_pool_window_nights > 0
    )
    if window_pool_requested:
        pool_night = (
            (night_dir, current_night)
            if night_dir is not None and current_night is not None
            else infer_science_night(
                [m for m in select_frames(metas, "science") if m.is_valid]
            )
        )
        if not settings.calib.calib_fallback or args.no_calib_fallback:
            logger.info("calibration pooling disabled (multi-night calibration fallback off)")
        elif pool_night is None:
            logger.warning(
                "calibration pooling inactive: the science frames are not all in "
                "one YYYYMMDD night directory"
            )
        else:
            pool_night_dir, pool_current_night = pool_night
            pool_result = apply_window_pools(
                metas,
                profile=profile,
                settings=settings,
                night_dir=pool_night_dir,
                current_night=pool_current_night,
                filters=args.filter,
            )
            metas = pool_result.metas
            for pooled_flats in pool_result.flats.values():
                if pooled_flats.reuse is not None:
                    continue  # an existing master is copied instead of built
                for group_key in group_flats_by_filter(
                    list(pooled_flats.metas),
                    separate_dome_and_sky=settings.calib.separate_dome_and_sky_flats,
                ):
                    flat_fallback_source_nights[group_key] = (
                        pooled_flats.nights,
                        pooled_flats.borrowed,
                    )
    bias_pool_active = pool_result is not None and pool_result.bias_active
    flat_window_active = pool_result is not None and pool_result.flat_active
    flat_window_requested = pool_result is not None and (
        profile.calib.flat_pool and profile.calib.flat_pool_window_nights > 0
    )
    #: filter -> existing master flat to copy into calibdir instead of building
    #: one (the short in-window pool then only feeds the gain measurement).
    planned_flat_reuse = {
        filter_name: sel.reuse
        for filter_name, sel in (pool_result.flats.items() if pool_result else ())
        if sel.reuse is not None
    }

    if calib_fallback_enabled:
        if not bias_pool_active and not select_frames(metas, "bias"):
            non_bias_metas = [m for m in metas if m.frame_type != "bias"]
            target_config = dominant_camera_config(non_bias_metas)
            if target_config is None:
                logger.warning(
                    "no bias frames for night %s; cannot determine a camera "
                    "configuration to search previous nights for a fallback",
                    night_dir.name,
                )
            else:
                result = collect_fallback_frames(
                    night_dir,
                    current_night,
                    frame_type="bias",
                    target_config=target_config,
                    min_count=settings.calib.min_bias_frames,
                    max_nights=settings.calib.fallback_max_nights_bias,
                    profile=profile,
                    settings=settings,
                )
                if result.satisfied:
                    logger.warning(
                        "no bias frames for night %s; borrowing %d bias frame(s) "
                        "from night(s) %s",
                        night_dir.name, len(result.metas), ",".join(result.source_nights),
                    )
                    metas.extend(result.metas)
                    bias_fallback_nights = result.source_nights
                else:
                    logger.warning(
                        "no bias frames for night %s and fewer than %d found in "
                        "the previous %d night(s); continuing without master bias",
                        night_dir.name,
                        settings.calib.min_bias_frames,
                        settings.calib.fallback_max_nights_bias,
                    )

        if not select_frames(metas, "dark"):
            non_dark_metas = [m for m in metas if m.frame_type != "dark"]
            target_config = dominant_camera_config(non_dark_metas)
            if target_config is not None:
                result = collect_fallback_frames(
                    night_dir,
                    current_night,
                    frame_type="dark",
                    target_config=target_config,
                    min_count=settings.calib.min_dark_frames,
                    max_nights=settings.calib.fallback_max_nights_bias,
                    profile=profile,
                    settings=settings,
                )
                if result.satisfied:
                    logger.warning(
                        "no dark frames for night %s; borrowing %d dark frame(s) "
                        "from night(s) %s",
                        night_dir.name, len(result.metas), ",".join(result.source_nights),
                    )
                    metas.extend(result.metas)
                    dark_fallback_nights = result.source_nights

        science_metas_for_fallback = select_frames(metas, "science")
        expected_filters = {
            m.filter for m in science_metas_for_fallback if m.filter is not None
        }
        separate_dome_and_sky = settings.calib.separate_dome_and_sky_flats
        local_flat_metas_all = select_frames(metas, "flat")
        local_flat_groups = group_flats_by_filter(
            local_flat_metas_all, separate_dome_and_sky=separate_dome_and_sky
        )
        covered_filters = {key.split(":", 1)[0] for key in local_flat_groups}

        if flat_window_requested:
            # Symmetric-window flat pools were already built above
            # (apply_window_pools); neither legacy branch below applies.
            pass
        elif profile.calib.flat_pool:
            # New pooling policy (per instrument profile): always pool, per
            # filter, whether or not the night has any local flats of its
            # own for that filter -- see robo43.calib.fallback.pool_flat_frames.
            local_filters_present = {
                m.filter for m in local_flat_metas_all if m.filter is not None
            }
            for pool_filter in sorted(local_filters_present | expected_filters):
                filter_local_metas = [
                    m for m in local_flat_metas_all if m.filter == pool_filter
                ]
                if filter_local_metas:
                    target_config = dominant_camera_config(filter_local_metas)
                else:
                    filter_science_metas = [
                        m for m in science_metas_for_fallback if m.filter == pool_filter
                    ]
                    target_config = dominant_camera_config(filter_science_metas)
                if target_config is None:
                    logger.warning(
                        "no flat or science frames for filter %s on night %s; "
                        "cannot determine a camera configuration for flat "
                        "pooling",
                        pool_filter, night_dir.name,
                    )
                    continue

                pool_result = pool_flat_frames(
                    night_dir,
                    current_night,
                    local_metas=filter_local_metas,
                    target_config=target_config,
                    filter_name=pool_filter,
                    min_nights=profile.calib.flat_pool_min_nights,
                    max_nights=profile.calib.flat_pool_max_nights,
                    target_frames=profile.calib.flat_pool_target_frames,
                    fallback_max_nights=settings.calib.fallback_max_nights_flat,
                    profile=profile,
                    settings=settings,
                )

                local_paths = {m.path for m in filter_local_metas}
                borrowed_metas = [
                    m for m in pool_result.metas if m.path not in local_paths
                ]
                if borrowed_metas:
                    metas.extend(borrowed_metas)

                n_pool = len(pool_result.metas)
                if n_pool == 0:
                    logger.warning(
                        "no flat frames for filter %s on night %s within the "
                        "previous %d night(s); continuing without a master "
                        "flat for that filter",
                        pool_filter,
                        night_dir.name,
                        settings.calib.fallback_max_nights_flat,
                    )
                    continue

                if n_pool < profile.calib.flat_pool_target_frames:
                    logger.info(
                        "flat pool for filter %s on night %s has only %d "
                        "frame(s) (target %d); using what is available from "
                        "night(s) %s",
                        pool_filter,
                        night_dir.name,
                        n_pool,
                        profile.calib.flat_pool_target_frames,
                        ",".join(pool_result.contributing_nights),
                    )
                else:
                    logger.info(
                        "flat pool for filter %s on night %s has %d frame(s) "
                        "from night(s) %s",
                        pool_filter,
                        night_dir.name,
                        n_pool,
                        ",".join(pool_result.contributing_nights),
                    )

                for group_key in group_flats_by_filter(
                    list(pool_result.metas), separate_dome_and_sky=separate_dome_and_sky
                ):
                    flat_fallback_source_nights[group_key] = (
                        pool_result.contributing_nights,
                        pool_result.borrowed,
                    )
        else:
            # Existing zero-trigger behaviour (unchanged): only search
            # previous nights when a filter has no local flats at all.
            for missing_filter in sorted(expected_filters - covered_filters):
                filter_science_metas = [
                    m for m in science_metas_for_fallback if m.filter == missing_filter
                ]
                target_config = dominant_camera_config(filter_science_metas)
                if target_config is None:
                    logger.warning(
                        "no flat frames for filter %s on night %s; cannot determine "
                        "a camera configuration to search previous nights for a "
                        "fallback",
                        missing_filter, night_dir.name,
                    )
                    continue
                result = collect_fallback_frames(
                    night_dir,
                    current_night,
                    frame_type="flat",
                    target_config=target_config,
                    min_count=settings.calib.min_flat_frames,
                    max_nights=settings.calib.fallback_max_nights_flat,
                    profile=profile,
                    settings=settings,
                    filter_name=missing_filter,
                )
                if result.satisfied:
                    logger.warning(
                        "no flat frames for filter %s on night %s; borrowing %d "
                        "flat frame(s) from night(s) %s",
                        missing_filter,
                        night_dir.name,
                        len(result.metas),
                        ",".join(result.source_nights),
                    )
                    metas.extend(result.metas)
                    for group_key in group_flats_by_filter(
                        list(result.metas), separate_dome_and_sky=separate_dome_and_sky
                    ):
                        flat_fallback_source_nights[group_key] = (
                            result.source_nights,
                            True,
                        )
                else:
                    logger.warning(
                        "no flat frames for filter %s on night %s and fewer than "
                        "%d found in the previous %d night(s); continuing without "
                        "a master flat for that filter",
                        missing_filter,
                        night_dir.name,
                        settings.calib.min_flat_frames,
                        settings.calib.fallback_max_nights_flat,
                    )

    # Only bias frames of one readout mode may be combined into the master bias
    # (and, per group, measured for gain): explicit inputs and borrowed frames
    # of another mode than the science frames' are dropped with a WARNING; a
    # calibration-only run spanning several modes is refused.
    try:
        metas = enforce_single_bias_mode(metas)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2

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

    if settings.runtime.release_memory:
        release_memory("reduce.overscan_detection", level=logging.INFO)

    # Dry-run mode: report plan and exit
    if args.dry_run:
        logger.info("=== DRY RUN MODE ===")

        # Report what masters would be built/reused
        bias_metas = select_frames(metas, "bias")
        dark_metas = select_frames(metas, "dark")
        flat_metas = [
            m for m in select_frames(metas, "flat") if m.filter not in planned_flat_reuse
        ]
        science_metas = select_frames(metas, "science")
        for reuse_filter, reuse_choice in planned_flat_reuse.items():
            logger.info(
                "master_flat %s: would COPY %s (night %s, %+d day(s))",
                reuse_filter,
                reuse_choice.path,
                reuse_choice.night.strftime("%Y%m%d"),
                reuse_choice.distance_days,
            )

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

        if (calibdir_path / "mask.fits").exists():
            logger.info("mask.fits: would REUSE")
        else:
            min_flats = settings.calib.cold_pixel_min_flats
            flat_group_sizes = [
                len(group)
                for group in group_flats_by_filter(
                    flat_metas,
                    separate_dome_and_sky=settings.calib.separate_dome_and_sky_flats,
                ).values()
            ]
            hot_source = bool(dark_metas) or (calibdir_path / "dark_current_map.fits").exists()
            cold_source = any(n >= min_flats for n in flat_group_sizes) or any(
                calibdir_path.glob("coldpix_*.fits")
            )
            if hot_source or cold_source:
                logger.info(
                    "mask.fits: would BUILD from %s",
                    " + ".join(
                        label
                        for label, used in (
                            ("hot pixels", hot_source),
                            ("cold pixels", cold_source),
                        )
                        if used
                    ),
                )
            else:
                logger.info(
                    "mask.fits: no dark frames and no flat group with >= %d frames, would SKIP",
                    min_flats,
                )

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
                # A pooled bias set is capped for the measurement (it keeps one
                # full-frame float64 array alive per bias frame): nearest nights.
                limit_bias_for_gain(metas, profile.calib.bias_pool_gain_max_frames)
                if bias_pool_active
                else metas,
                settings=settings,
                profile=profile,
                overscan=overscan_geometry,
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

    if settings.runtime.release_memory:
        release_memory("reduce.gain_measurement", level=logging.INFO)

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
                        fallback_source_nights=bias_fallback_nights,
                        pool=pool_result.bias if bias_pool_active else None,
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
                    fallback_source_nights=bias_fallback_nights,
                    pool=pool_result.bias if bias_pool_active else None,
                )
            except Exception as exc:
                logger.error("failed to build master_bias: %s", exc)
                return 2
    else:
        logger.warning("no bias frames found; continuing without master bias")

    if settings.runtime.release_memory:
        release_memory("reduce.master_bias", level=logging.INFO)

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
                    overscan=overscan_geometry,
                    gain_measurements=gain_measurements,
                    fallback_source_nights=dark_fallback_nights,
                )
            except Exception as exc:
                logger.error("failed to build master_darks: %s", exc)
                return 2
    else:
        logger.info("no dark frames found; continuing without master darks")

    if settings.runtime.release_memory:
        release_memory("reduce.master_dark", level=logging.INFO)

    # Reuse existing master flats where the pool policy chose them over a short
    # pool: copy each into calibdir (unless one is already there) with
    # provenance cards. A failed copy leaves that filter to be built from its
    # (short) pool.
    reused_flat_filters: set[str] = set()
    for reuse_filter, reuse_choice in planned_flat_reuse.items():
        existing = list(calibdir_path.glob(f"master_flat_{reuse_filter}*.fits"))
        if existing:
            logger.info(
                "master flat for filter %s already in calibdir (%s); not copying %s",
                reuse_filter, existing[0].name, reuse_choice.path,
            )
            reused_flat_filters.add(reuse_filter)
            continue
        try:
            installed = install_master_flat(
                reuse_choice,
                calibdir_path,
                current_night=pool_current_night,
                window_nights=profile.calib.flat_pool_window_nights,
                target_frames=profile.calib.flat_pool_target_frames,
            )
        except Exception as exc:
            logger.error(
                "failed to reuse master flat %s for filter %s: %s; building from the "
                "short pool instead",
                reuse_choice.path, reuse_filter, exc,
            )
            continue
        reused_flat_filters.add(reuse_filter)
        logger.warning(
            "master flat for filter %s reused from night %s (%+d day(s), %s frame(s)): "
            "%s -> %s",
            reuse_filter,
            reuse_choice.night.strftime("%Y%m%d"),
            reuse_choice.distance_days,
            reuse_choice.n_frames if reuse_choice.n_frames is not None else "unknown",
            reuse_choice.path,
            installed,
        )

    # Build master flats
    flat_metas = [
        m for m in select_frames(metas, "flat") if m.filter not in reused_flat_filters
    ]
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
                # master_flat_<filter>.fits, master_flat_<filter>_dome.fits,
                # master_flat_<filter>_sky.fits. The check is by filter, whatever
                # the group's kind: a master written under another kind suffix
                # (e.g. twilight flats that older versions labelled _dome) still
                # counts as present.
                group_filter = group_key.split(":", 1)[0]
                flat_master_files = [
                    candidate
                    for candidate in (
                        calibdir_path / f"master_flat_{group_filter}.fits",
                        calibdir_path / f"master_flat_{group_filter}_dome.fits",
                        calibdir_path / f"master_flat_{group_filter}_sky.fits",
                    )
                    if candidate.exists()
                ]

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
                        fallback_source_nights=flat_fallback_source_nights,
                        pool_window_nights=(
                            profile.calib.flat_pool_window_nights if flat_window_active else None
                        ),
                        pool_target_frames=(
                            profile.calib.flat_pool_target_frames if flat_window_active else None
                        ),
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

    if settings.runtime.release_memory:
        release_memory("reduce.master_flat", level=logging.INFO)

    # Compose the static bad-pixel mask (hot from darks, cold from flats)
    _compose_static_mask(calibdir_path, settings)

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

    if settings.runtime.release_memory:
        release_memory("reduce.before_science_loop", level=logging.INFO)

    statuses: list = []
    n_success = 0
    n_degraded = 0
    n_failed = 0

    global _worker_calib, _worker_settings, _worker_profile
    global _worker_overscan, _worker_outdir, _worker_changelog_lock

    if science_metas_valid:
        nprocs_requested = max(1, settings.runtime.nprocs)
        nprocs = nprocs_requested
        if nprocs_requested > 1:
            sample_header = read_header(science_metas_valid[0].path, profile)
            naxis1 = int(sample_header.get("NAXIS1", 0) or 0)
            naxis2 = int(sample_header.get("NAXIS2", 0) or 0)
            per_worker_mb = (
                naxis1 * naxis2 * STAGE_BYTES_PER_PIXEL["reduce.science"] / (1024.0 * 1024.0)
                if naxis1 and naxis2
                else 0.0
            )
            nprocs = safe_worker_count(
                nprocs_requested,
                per_worker_mb,
                settings.runtime.memory_reserve_mb,
                stage="reduce.science",
            )
        logger.info(
            "reducing %d science frame(s) using %d worker process(es)",
            len(science_metas_valid),
            nprocs,
        )

        if nprocs > 1:
            # "spawn", not "fork": ProcessPoolExecutor's own manager and
            # queue-feeder threads are already running in this process by
            # the time any worker is spawned (see _init_science_worker's
            # docstring), so "fork" here would carry the same
            # fork-after-thread hazard CPython itself has an open issue
            # about. calib_set is therefore rebuilt once per worker (inside
            # the initializer) rather than pickled from the parent, and
            # nothing this loop passes to reduce_frame_worker per task
            # carries calib_set, settings, profile, or overscan_geometry --
            # only the (small) FrameMeta.
            mp_ctx = multiprocessing.get_context("spawn")
            mp_queue: multiprocessing.Queue = mp_ctx.Queue()
            changelog_lock = mp_ctx.Lock()
            listener = log_listener(mp_queue)
            try:
                # max_tasks_per_child is deliberately left at its default (a
                # worker handles every task assigned to it for its whole
                # lifetime): setting it to 1 would rerun
                # _init_science_worker's CalibrationSet.load() once per
                # frame instead of once per worker, which is exactly the
                # per-task cost this design avoids.
                with ProcessPoolExecutor(
                    max_workers=nprocs,
                    mp_context=mp_ctx,
                    initializer=_init_science_worker,
                    initargs=(
                        calibdir_path,
                        settings,
                        profile,
                        overscan_geometry,
                        outdir_path,
                        filters_to_load,
                        changelog_lock,
                        mp_queue,
                        settings.runtime.loglevel,
                    ),
                ) as pool:
                    results_iter = pool.map(reduce_frame_worker, science_metas_valid)
                    for frame_index, (row, proc_code) in enumerate(results_iter):
                        statuses.append(row)
                        if proc_code == 0:
                            n_success += 1
                        elif proc_code == 3:
                            n_degraded += 1
                        elif proc_code == 4:
                            n_failed += 1
                        if settings.runtime.release_memory and (frame_index + 1) % 10 == 0:
                            release_memory(f"reduce.science_loop:frame_{frame_index + 1}")
            finally:
                listener.stop()
        else:
            _worker_calib = calib_set
            _worker_settings = settings
            _worker_profile = profile
            _worker_overscan = overscan_geometry
            _worker_outdir = outdir_path
            _worker_changelog_lock = contextlib.nullcontext()
            for frame_index, sci_meta in enumerate(science_metas_valid):
                row, proc_code = reduce_frame_worker(sci_meta)
                statuses.append(row)
                if proc_code == 0:
                    n_success += 1
                elif proc_code == 3:
                    n_degraded += 1
                elif proc_code == 4:
                    n_failed += 1
                if settings.runtime.release_memory and (frame_index + 1) % 10 == 0:
                    release_memory(f"reduce.science_loop:frame_{frame_index + 1}")

    if settings.runtime.release_memory:
        release_memory("reduce.after_science_loop", level=logging.INFO)

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

    log_peak_rss_summary("reduce")
    return 0


SUBCOMMAND = Subcommand(
    name="reduce",
    help="Reduce raw science frames to processed products.",
    add_arguments=_add_arguments,
    run=_run,
)
register(SUBCOMMAND)
