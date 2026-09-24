"""``robo43 astrometry`` -- solve a WCS for existing ``_proc.fits`` products.

Reads each input's ``SCI``/``DQ`` planes and header, runs
:func:`robo43.astrometry.solve.solve_frame`, and on success writes the
resulting WCS/SIP and provenance cards into every plane (see
:mod:`robo43.astrometry.wcs_header`). By default nothing is modified in
place: a solved copy is written under ``--outdir`` (or a sibling
``astrometry`` directory next to the input), and only ``--in-place``
overwrites the input itself. A per-input status CSV is written alongside the
outputs, following the convention ``robo43 reduce`` uses for
``proc_status.csv``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import functools
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from astropy.io import fits

from robo43.astrometry.solve import solve_frame
from robo43.astrometry.wcs_header import apply_solution
from robo43.cli.common import configure_from_args
from robo43.cli.registry import Subcommand, register
from robo43.instruments.profile import resolve_profile
from robo43.io.fits_io import iter_fits_files, read_header, write_fits
from robo43.io.headers import normalise_radec
from robo43.logging_setup import log_listener, worker_log_config
from robo43.memory import STAGE_BYTES_PER_PIXEL, log_peak_rss_summary, release_memory, safe_worker_count
from robo43.products.dq import DQ_DTYPE

if TYPE_CHECKING:
    from robo43.config import Settings
    from robo43.instruments.profile import InstrumentProfile

__all__ = ["SUBCOMMAND"]

logger = logging.getLogger(__name__)

#: Basename of the per-run status CSV, alongside robo43.cli.reduce's
#: proc_status.csv convention.
STATUS_CSV_NAME = "astrometry_status.csv"

STATUS_FIELDNAMES: tuple[str, ...] = (
    "path",
    "raw_name",
    "output_path",
    "solved",
    "rms_arcsec",
    "n_detected",
    "n_matched",
    "logodds",
    "scale_arcsec_per_pixel",
    "rung",
    "blind",
    "index_series",
    "index_scales",
    "duration_s",
    "error",
)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("inputs", nargs="+", help="_proc.fits files or directories")
    parser.add_argument(
        "--index-dir", default=None, help="override astrometry.index_dir for this run"
    )
    parser.add_argument(
        "--scales",
        default=None,
        help="comma-separated astrometry.net index scale numbers, overriding derivation",
    )
    parser.add_argument(
        "--max-sources", type=int, default=None, help="override astrometry.max_sources"
    )
    parser.add_argument(
        "--rms-max",
        type=float,
        default=None,
        dest="rms_max",
        help="override astrometry.rms_max_arcsec",
    )
    parser.add_argument("--sip-order", type=int, default=None, help="override astrometry.sip_order")
    parser.add_argument(
        "--no-position-hint",
        action="store_true",
        default=False,
        help="solve blind (ignore the header RA/DEC) for every input",
    )
    placement = parser.add_mutually_exclusive_group()
    placement.add_argument(
        "--in-place", action="store_true", default=False, help="overwrite each input file"
    )
    placement.add_argument(
        "--outdir",
        default=None,
        help="directory for solved copies (default: <input dir>/astrometry)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="allow overwriting an existing output file",
    )


def _settings_with_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    overrides: dict[str, Any] = {}
    if args.index_dir is not None:
        overrides["index_dir"] = Path(args.index_dir)
    if args.scales is not None:
        overrides["index_scales"] = tuple(int(s) for s in args.scales.split(","))
    if args.max_sources is not None:
        overrides["max_sources"] = args.max_sources
    if args.rms_max is not None:
        overrides["rms_max_arcsec"] = args.rms_max
    if args.sip_order is not None:
        overrides["sip_order"] = args.sip_order
    if not overrides:
        return settings
    return settings.replace(astrometry=dataclasses.replace(settings.astrometry, **overrides))


def _radec_from_header(
    header: fits.Header, profile: InstrumentProfile
) -> tuple[float | None, float | None]:
    """Header RA/Dec via the profile's own alias order (mirrors
    ``robo43.io.headers._find_alias_value`` without importing that private
    helper): the first alias present wins."""
    ra_raw = next(
        (
            header.get(alias)
            for alias in profile.aliases_for("ra")
            if header.get(alias) not in (None, "")
        ),
        None,
    )
    dec_raw = next(
        (
            header.get(alias)
            for alias in profile.aliases_for("dec")
            if header.get(alias) not in (None, "")
        ),
        None,
    )
    return normalise_radec(ra_raw, dec_raw)


def _pixel_scale_arcsec(header: fits.Header, profile: InstrumentProfile) -> float | None:
    """Plate scale in arcsec/pixel: 206.265 * XPIXSZ / FOCALLEN, using the
    profile's own aliases for those two keywords.

    Deliberately NOT multiplied by the binning keyword, unlike
    robo43.io.headers.derive_pixel_scale -- for this instrument (QHY600M)
    XPIXSZ is already the *binned* pixel size (verified against the
    ROBO43 mini-sample: this formula reproduces the documented 0.5277"/px at
    XBINNING=2, while multiplying by binning again would double it to
    ~1.0555"/px). derive_pixel_scale was not changed to match, since its
    binning multiplication may be correct for some other instrument profile;
    see this change's report for the discrepancy. Falls back to the
    profile's static pixel_scale_arcsec, then None.
    """
    pixsz = next(
        (
            header.get(alias)
            for alias in profile.aliases_for("pixel_size")
            if header.get(alias) is not None
        ),
        None,
    )
    focal_length = next(
        (
            header.get(alias)
            for alias in profile.aliases_for("focal_length")
            if header.get(alias) is not None
        ),
        None,
    )
    if pixsz is not None and focal_length not in (None, 0):
        try:
            return 206.265 * float(pixsz) / float(focal_length)
        except (TypeError, ValueError):
            pass
    return profile.pixel_scale_arcsec


def _output_path(path: Path, *, in_place: bool, outdir: Path | None) -> Path:
    if in_place:
        return path
    target_dir = outdir if outdir is not None else path.parent / "astrometry"
    return target_dir / path.name


def _status_row(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "raw_name": path.name,
        "output_path": None,
        "solved": False,
        "rms_arcsec": None,
        "n_detected": None,
        "n_matched": None,
        "logodds": None,
        "scale_arcsec_per_pixel": None,
        "rung": None,
        "blind": None,
        "index_series": None,
        "index_scales": None,
        "duration_s": None,
        "error": None,
    }


def _process_one(
    path: Path,
    *,
    profile: InstrumentProfile,
    settings: Settings,
    in_place: bool,
    outdir: Path | None,
    overwrite: bool,
    force_blind: bool,
    dry_run: bool,
) -> dict[str, Any]:
    row = _status_row(path)
    start = time.monotonic()
    out_path = _output_path(path, in_place=in_place, outdir=outdir)
    row["output_path"] = str(out_path)

    try:
        with fits.open(path) as hdul:
            sci = np.array(hdul["SCI"].data, dtype=np.float32)
            dq = np.array(hdul["DQ"].data, dtype=DQ_DTYPE)
            header = hdul["SCI"].header

            pixel_scale_arcsec = _pixel_scale_arcsec(header, profile)
            ra_deg, dec_deg = _radec_from_header(header, profile)

            if pixel_scale_arcsec is None:
                row["error"] = "could not derive a pixel scale from the header or profile"
                row["duration_s"] = time.monotonic() - start
                return row

            solution = solve_frame(
                sci,
                dq,
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                pixel_scale_arcsec=pixel_scale_arcsec,
                seeing_fwhm_arcsec=settings.photometry.seeing_fwhm_arcsec,
                settings=settings.astrometry,
                profile_index_scales=profile.index_scales,
                force_blind=force_blind,
            )

            row.update(
                solved=solution.solved,
                rms_arcsec=solution.rms_arcsec,
                n_detected=solution.n_detected,
                n_matched=solution.n_matched,
                logodds=solution.logodds,
                scale_arcsec_per_pixel=solution.scale_arcsec_per_pixel,
                rung=solution.rung,
                blind=solution.blind,
                index_series=solution.index_series,
                index_scales=",".join(str(s) for s in solution.index_scales),
            )

            if not solution.solved:
                row["error"] = solution.failure_reason
                row["duration_s"] = time.monotonic() - start
                return row

            if dry_run:
                logger.info(
                    'dry-run: %s would solve (rms=%.3f") -> %s', path, solution.rms_arcsec, out_path
                )
                row["duration_s"] = time.monotonic() - start
                return row

            apply_solution(hdul, solution)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            write_fits(out_path, hdul, overwrite=overwrite or in_place)

    except Exception as exc:  # never let one frame's error abort the whole batch
        logger.error("astrometry failed for %s: %s", path, exc)
        row["error"] = str(exc)

    row["duration_s"] = time.monotonic() - start
    return row


def _run(args: argparse.Namespace) -> int:
    settings = configure_from_args(args)
    settings = _settings_with_overrides(settings, args)

    paths = list(iter_fits_files(args.inputs, patterns=["*_proc.fits"], recursive=False))
    if not paths:
        logger.error("no _proc.fits files found among %s", args.inputs)
        return 2

    try:
        if args.instrument:
            profile = resolve_profile(name=args.instrument)
        else:
            profile = resolve_profile(header=read_header(paths[0], profile=None))
    except Exception as exc:
        logger.error("failed to resolve instrument profile: %s", exc)
        return 2

    outdir = Path(args.outdir) if args.outdir else None
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)

    logger.info("solving astrometry for %d frame(s)", len(paths))

    sample_header = read_header(paths[0], profile=None)
    naxis1 = int(sample_header.get("NAXIS1", 0) or 0)
    naxis2 = int(sample_header.get("NAXIS2", 0) or 0)
    per_worker_mb = (
        naxis1 * naxis2 * STAGE_BYTES_PER_PIXEL["astrometry"] / (1024.0 * 1024.0)
        if naxis1 and naxis2
        else 0.0
    )
    nprocs = safe_worker_count(
        max(1, settings.runtime.nprocs),
        per_worker_mb,
        settings.runtime.memory_reserve_mb,
        stage="astrometry",
    )

    fn = functools.partial(
        _process_one,
        profile=profile,
        settings=settings,
        in_place=args.in_place,
        outdir=outdir,
        overwrite=args.overwrite,
        force_blind=args.no_position_hint,
        dry_run=settings.runtime.dry_run,
    )

    if settings.runtime.release_memory:
        release_memory("astrometry.before_pool", level=logging.INFO)

    rows: list[dict[str, Any]] = []
    if nprocs > 1:
        mp_ctx = multiprocessing.get_context("spawn")
        mp_queue: multiprocessing.Queue = mp_ctx.Queue()
        listener = log_listener(mp_queue)
        try:
            with ProcessPoolExecutor(
                max_workers=nprocs,
                mp_context=mp_ctx,
                max_tasks_per_child=1,
                initializer=worker_log_config,
                initargs=(mp_queue, settings.runtime.loglevel),
            ) as pool:
                for row in pool.map(fn, paths):
                    rows.append(row)
                    if settings.runtime.release_memory:
                        release_memory("astrometry.frame")
        finally:
            listener.stop()
    else:
        for path in paths:
            rows.append(fn(path))
            if settings.runtime.release_memory:
                release_memory("astrometry.frame")

    status_dir = outdir if outdir is not None else paths[0].parent
    csv_path = status_dir / STATUS_CSV_NAME
    try:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("wrote status CSV: %s", csv_path)
    except OSError as exc:
        logger.error("failed to write status CSV: %s", exc)
        return 2

    n_solved = sum(1 for row in rows if row["solved"])
    logger.info("astrometry complete: %d/%d frame(s) solved", n_solved, len(rows))
    log_peak_rss_summary("astrometry")
    return 0


SUBCOMMAND = Subcommand(
    name="astrometry",
    help="Solve an astrometric WCS for _proc.fits products.",
    add_arguments=_add_arguments,
    run=_run,
)
register(SUBCOMMAND)
