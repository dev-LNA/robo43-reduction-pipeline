"""``robo43 forced`` -- fixed-centroid aperture photometry for a whole night.

Run after ``robo43 photometry``: it reads each frame's existing ``CATALOG`` (seeing,
useful-source count, transparency), picks the centroid frames, builds the master
sky positions (:mod:`robo43.photometry.forced` documents the method) and measures every
frame at those positions.  Outputs, next to the inputs unless ``--outdir`` is given:

* ``<stem>_forced_catalog.csv`` per frame -- the columns of ``<stem>_catalog.csv``
  (``FLUX_APER_1..N`` ...), so ``relphot`` reads it with ``--photometry forced``;
* with ``--fits-ext``, a ``CATALOG_FORCED`` extension appended to the product
  (``--in-place`` or into ``--outdir``), the analogue of ``photometry``'s ``CATALOG``;
* ``forced_reference.csv`` (master sources), ``forced_status.csv`` (per frame: FWHM,
  ``n_useful``, transparency, score, whether it supplied centroids, counts, timing, peak RSS)
  and ``forced_params.json`` (settings, chosen frames, config hash).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import functools
import json
import logging
import multiprocessing
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

import robo43
from robo43.cli.common import configure_from_args
from robo43.cli.photometry import _flatten_aperture_columns_for_csv
from robo43.cli.registry import Subcommand, register
from robo43.exceptions import PhotometryError
from robo43.instruments.profile import resolve_profile
from robo43.io.fits_io import iter_fits_files, read_header, write_fits
from robo43.logging_setup import log_listener, worker_log_config
from robo43.memory import (
    STAGE_BYTES_PER_PIXEL,
    log_peak_rss_summary,
    read_peak_rss_mb,
    release_memory,
    safe_worker_count,
)
from robo43.photometry.forced import (
    FrameGeometry,
    FrameInfo,
    MasterCatalog,
    build_master,
    measure_forced,
    measure_transparency,
    read_master,
    score_frames,
    select_centroid_frames,
    summarize_frame,
    write_master,
)
from robo43.products.catalog import build_catalog_hdu, read_catalog
from robo43.products.dq import DQ_DTYPE

if TYPE_CHECKING:
    from robo43.config import Settings
    from robo43.instruments.profile import InstrumentProfile

__all__ = ["SUBCOMMAND"]

logger = logging.getLogger(__name__)

STATUS_CSV_NAME = "forced_status.csv"
REFERENCE_CSV_NAME = "forced_reference.csv"
PARAMS_JSON_NAME = "forced_params.json"
FORCED_HDU_NAME = "CATALOG_FORCED"

STATUS_FIELDNAMES: tuple[str, ...] = (
    "path",
    "raw_name",
    "astrometry_ok",
    "astrometry_rms",
    "fwhm_arcsec",
    "n_catalog",
    "n_useful",
    "transparency",
    "n_transp_stars",
    "eligible",
    "score",
    "centroid_frame",
    "n_master",
    "n_measured",
    "n_offframe",
    "n_flagged",
    "sky_level",
    "sky_rms",
    "catalog_csv",
    "duration_s",
    "peak_rss_mb",
    "error",
)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("inputs", nargs="+", help="_proc.fits files or directories")
    parser.add_argument(
        "--outdir",
        default=None,
        help="directory for the CSV catalogues and summaries (default: the input directory)",
    )
    parser.add_argument(
        "--fits-ext",
        action="store_true",
        default=False,
        dest="fits_ext",
        help=(
            f"also append a {FORCED_HDU_NAME} extension to each product: rewrites the input "
            "with --in-place, otherwise a copy in --outdir"
        ),
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        default=False,
        dest="in_place",
        help="with --fits-ext, modify each input file in place",
    )
    parser.add_argument(
        "--master",
        default=None,
        help="measure at the positions of this forced_reference.csv instead of the built master",
    )
    parser.add_argument(
        "--select-only",
        action="store_true",
        default=False,
        dest="select_only",
        help="choose the centroid frames and write the master/status files, measure nothing",
    )
    parser.add_argument("--n-frames", type=int, default=None, dest="n_frames",
                        help="override forced.n_centroid_frames")
    parser.add_argument("--transparency-min", type=float, default=None, dest="transparency_min",
                        help="override forced.transparency_min")
    parser.add_argument("--seeing-power", type=float, default=None, dest="seeing_power",
                        help="override forced.seeing_power")
    parser.add_argument("--snr-useful", type=float, default=None, dest="snr_useful",
                        help="override forced.snr_useful")


def _settings_with_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    mapping = {
        "n_centroid_frames": args.n_frames,
        "transparency_min": args.transparency_min,
        "seeing_power": args.seeing_power,
        "snr_useful": args.snr_useful,
    }
    overrides = {k: v for k, v in mapping.items() if v is not None}
    if not overrides:
        return settings
    return settings.replace(forced=dataclasses.replace(settings.forced, **overrides))


def _csv_path(path: Path, outdir: Path) -> Path:
    return outdir / f"{path.stem}_forced_catalog.csv"


def _astrometry_state(header: fits.Header, rms_max: float) -> tuple[bool, float]:
    solved = header.get("ASTRSOLV") is True
    try:
        rms = float(header.get("ASTRRMS", np.nan))
    except (TypeError, ValueError):
        rms = float("nan")
    return bool(solved and np.isfinite(rms) and rms <= rms_max), rms


def _summarize_path(path: Path, settings: Settings) -> FrameInfo:
    """Read one product's ``CATALOG`` and primary header (no pixels) and summarise it."""
    table, metrics = read_catalog(path)
    header = read_header(path, profile=None)
    ok, rms = _astrometry_state(header, settings.forced.astrometry_rms_max_arcsec)
    try:
        exptime = float(header.get("EXPTIME", 0.0) or 0.0)
    except (TypeError, ValueError):
        exptime = 0.0
    radii = metrics.aper_rad or settings.photometry.aperture_radii_arcsec
    return summarize_frame(
        table,
        path=str(path),
        fwhm_arcsec=metrics.fwhm_arc if metrics.fwhm_arc is not None else float("nan"),
        fwhm_px=metrics.fwhm_pix if metrics.fwhm_pix is not None else float("nan"),
        exptime=exptime,
        astrometry_ok=ok,
        astrometry_rms=rms,
        aperture_radii_arcsec=radii,
        settings=settings,
    )


def _geometry(path: Path) -> FrameGeometry:
    header = read_header(path, profile=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs = WCS(header)
    return FrameGeometry(wcs=wcs, naxis1=int(header["NAXIS1"]), naxis2=int(header["NAXIS2"]))


def _choose_and_build(
    paths: list[Path], infos: list[FrameInfo], settings: Settings
) -> tuple[MasterCatalog, dict[str, Any], list[dict[str, Any]]]:
    """Frame quality -> centroid frames -> master; also the params dict and quality rows."""
    fs = settings.forced
    transparency, n_used, ref = measure_transparency(infos, settings)
    fwhm = np.array([i.fwhm_arcsec for i in infos])
    n_useful = np.array([i.n_useful for i in infos])
    ok = np.array([i.astrometry_ok for i in infos])
    eligible, score, relaxed = score_frames(fwhm, n_useful, transparency, ok, fs)
    if relaxed:
        logger.warning(
            "no frame reaches transparency >= %.2f; centroid frames chosen without that cut",
            fs.transparency_min,
        )
    chosen = select_centroid_frames(score, fwhm, fs.n_centroid_frames)
    if not chosen:
        msg = "no frame is usable for centroids (astrometry and catalogue are required)"
        raise PhotometryError(msg)

    seeing_only = select_centroid_frames(
        np.where(ok & np.isfinite(fwhm), -fwhm, -np.inf), fwhm, fs.n_centroid_frames
    )
    logger.info("transparency reference frame: %s", paths[ref].name)
    logger.info("centroid frames (score = n_useful/max * (fwhm_min/fwhm)^%.1f):", fs.seeing_power)
    for rank, i in enumerate(chosen, start=1):
        logger.info(
            "  %d. %s  FWHM %.2f\"  t %.3f  n_useful %d  score %.3f",
            rank, paths[i].name, fwhm[i], transparency[i], n_useful[i], score[i],
        )
    logger.info(
        "best-seeing-only would have chosen: %s", ", ".join(paths[i].name for i in seeing_only)
    )

    tables = [read_catalog(paths[i])[0] for i in chosen]
    geoms = [_geometry(paths[i]) for i in chosen]
    master = build_master(tables, geoms, fs)
    logger.info(
        "master: %d sources (catalogues of the centroid frames: %s)",
        len(master), ", ".join(str(len(t)) for t in tables),
    )

    rows = []
    for i, info in enumerate(infos):
        rows.append(
            {
                "path": info.path,
                "raw_name": paths[i].name,
                "astrometry_ok": info.astrometry_ok,
                "astrometry_rms": info.astrometry_rms,
                "fwhm_arcsec": info.fwhm_arcsec,
                "n_catalog": info.n_sources,
                "n_useful": info.n_useful,
                "transparency": transparency[i],
                "n_transp_stars": int(n_used[i]),
                "eligible": bool(eligible[i]),
                "score": score[i] if np.isfinite(score[i]) else "",
                "centroid_frame": (chosen.index(i) + 1) if i in chosen else "",
            }
        )
    params = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "robo43_version": robo43.__version__,
        "config_hash": settings.config_hash(),
        "forced_settings": dataclasses.asdict(fs),
        "aperture_radii_arcsec": list(settings.photometry.aperture_radii_arcsec),
        "transparency_reference": paths[ref].name,
        "centroid_frames": [paths[i].name for i in chosen],
        "best_seeing_only_frames": [paths[i].name for i in seeing_only],
        "n_master": len(master),
        "transparency_relaxed": bool(relaxed),
    }
    return master, params, rows


def _status_row(path: Path, quality: dict[str, Any] | None) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(STATUS_FIELDNAMES)
    row["path"] = str(path)
    row["raw_name"] = path.name
    row["catalog_csv"] = ""
    if quality:
        row.update(quality)
    return row


def _process_one(
    path: Path,
    *,
    master: MasterCatalog,
    quality: dict[str, Any] | None,
    profile: InstrumentProfile,
    settings: Settings,
    outdir: Path,
    fits_ext: bool,
    in_place: bool,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    row = _status_row(path, quality)
    start = time.monotonic()
    if settings.forced.skip_unsolved and quality is not None and not quality["astrometry_ok"]:
        msg = "astrometry not solved: frame skipped (forced.skip_unsolved)"
        logger.warning("%s: %s", path, msg)
        row["error"] = msg
        row["duration_s"] = time.monotonic() - start
        return row
    try:
        frame_table, metrics = read_catalog(path)
        with fits.open(path) as hdul:
            sci = np.array(hdul["SCI"].data, dtype=np.float32)
            err = np.array(hdul["ERR"].data, dtype=np.float32)
            dq = np.array(hdul["DQ"].data, dtype=DQ_DTYPE)
            header = hdul["SCI"].header
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                wcs = WCS(header, hdul)
            if dry_run:
                logger.info("dry-run: %s would be measured at %d positions", path, len(master))
                row["duration_s"] = time.monotonic() - start
                return row
            result = measure_forced(
                sci, err, dq, master=master, header=header, settings=settings,
                profile=profile, wcs=wcs, frame_table=frame_table, fwhm_px=metrics.fwhm_pix,
            )
            del sci, err, dq
            row.update(
                n_master=result.n_master,
                n_measured=result.n_measured,
                n_offframe=result.n_offframe,
                n_flagged=int(np.count_nonzero(np.asarray(result.table["FLAGS"]))),
                sky_level=result.sky_level,
                sky_rms=result.sky_rms,
            )
            outdir.mkdir(parents=True, exist_ok=True)
            if fits_ext:
                forced_metrics = dataclasses.replace(metrics, n_sources=len(result.table))
                hdu = build_catalog_hdu(
                    result.table, forced_metrics,
                    aperture_radii=settings.photometry.aperture_radii_arcsec,
                )
                hdu.name = FORCED_HDU_NAME
                while FORCED_HDU_NAME in hdul:
                    del hdul[FORCED_HDU_NAME]
                hdul.append(hdu)
                write_fits(
                    path if in_place else outdir / path.name, hdul,
                    overwrite=overwrite or in_place,
                )
    except Exception as exc:  # one frame's failure must not abort the night
        logger.error("forced photometry failed for %s: %s", path, exc)
        row["error"] = str(exc)
        row["duration_s"] = time.monotonic() - start
        return row

    # the product write above succeeded; the CSV is the relphot-facing output and is
    # reported as a failure of the frame if it cannot be written
    csv_path = _csv_path(path, outdir)
    try:
        flat = _flatten_aperture_columns_for_csv(result.table)
        flat.write(csv_path, format="csv", overwrite=True)
        row["catalog_csv"] = str(csv_path)
    except Exception as exc:
        logger.error("failed to write %s: %s", csv_path, exc)
        row["error"] = f"csv: {exc}"
    row["duration_s"] = time.monotonic() - start
    row["peak_rss_mb"] = read_peak_rss_mb()
    return row


def _write_status(outdir: Path, rows: list[dict[str, Any]]) -> None:
    with (outdir / STATUS_CSV_NAME).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _run(args: argparse.Namespace) -> int:
    settings = configure_from_args(args)
    settings = _settings_with_overrides(settings, args)

    paths = list(iter_fits_files(args.inputs, patterns=["*_proc.fits"], recursive=False))
    if not paths:
        logger.error("no _proc.fits files found among %s", args.inputs)
        return 2
    outdir = Path(args.outdir) if args.outdir else paths[0].parent
    if args.fits_ext and not args.in_place and outdir.resolve() == paths[0].parent.resolve():
        logger.error("--fits-ext would overwrite the inputs: use --in-place or another --outdir")
        return 2

    try:
        if args.instrument:
            profile = resolve_profile(name=args.instrument)
        else:
            profile = resolve_profile(header=read_header(paths[0], profile=None))
    except Exception as exc:
        logger.error("failed to resolve instrument profile: %s", exc)
        return 2

    nprocs_req = max(1, settings.runtime.nprocs)
    logger.info("forced photometry: %d frame(s)", len(paths))

    outdir.mkdir(parents=True, exist_ok=True)
    quality_by_path: dict[str, dict[str, Any]] = {}
    params: dict[str, Any] = {}
    try:
        with ThreadPoolExecutor(max_workers=min(nprocs_req, 8)) as pool:
            infos = list(pool.map(functools.partial(_summarize_path, settings=settings), paths))
        catalog_radii = infos[0].aperture_radii_arcsec
        if tuple(catalog_radii) != tuple(settings.photometry.aperture_radii_arcsec):
            logger.warning(
                "aperture radii follow the frames' own catalogues %s, not the configured %s",
                catalog_radii, tuple(settings.photometry.aperture_radii_arcsec),
            )
            settings = settings.replace(
                photometry=dataclasses.replace(
                    settings.photometry, aperture_radii_arcsec=tuple(catalog_radii)
                )
            )
        built_master, params, quality_rows = _choose_and_build(paths, infos, settings)
        quality_by_path = {r["path"]: r for r in quality_rows}
        if args.master:
            master = read_master(args.master)
            params["master_from"] = str(args.master)
            logger.info("master read from %s: %d sources", args.master, len(master))
        else:
            master = built_master
    except Exception as exc:
        logger.error("frame selection failed: %s", exc)
        return 2

    params["n_master"] = len(master)
    write_master(master, outdir / REFERENCE_CSV_NAME)
    (outdir / PARAMS_JSON_NAME).write_text(json.dumps(params, indent=2, default=str))
    rows = [_status_row(p, quality_by_path.get(str(p))) for p in paths]
    _write_status(outdir, rows)
    if args.select_only:
        logger.info("--select-only: wrote the master, status and params files to %s", outdir)
        return 0

    sample = read_header(paths[0], profile=None)
    naxis1 = int(sample.get("NAXIS1", 0) or 0)
    naxis2 = int(sample.get("NAXIS2", 0) or 0)
    per_worker_mb = (
        naxis1 * naxis2 * STAGE_BYTES_PER_PIXEL["forced"] / (1024.0 * 1024.0)
        if naxis1 and naxis2
        else 0.0
    )
    nprocs = safe_worker_count(
        nprocs_req, per_worker_mb, settings.runtime.memory_reserve_mb, stage="forced"
    )

    fn = functools.partial(
        _process_one,
        master=master,
        profile=profile,
        settings=settings,
        outdir=outdir,
        fits_ext=args.fits_ext,
        in_place=args.in_place,
        overwrite=settings.runtime.clobber,
        dry_run=settings.runtime.dry_run,
    )
    jobs = [(p, quality_by_path.get(str(p))) for p in paths]

    if settings.runtime.release_memory:
        release_memory("forced.before_pool", level=logging.INFO)

    results: list[dict[str, Any]] = []
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
                for row in pool.map(_call, [(fn, p, q) for p, q in jobs]):
                    results.append(row)
        finally:
            listener.stop()
    else:
        for p, q in jobs:
            results.append(fn(p, quality=q))
            if settings.runtime.release_memory:
                release_memory("forced.frame")

    try:
        _write_status(outdir, results)
    except OSError as exc:
        logger.error("failed to write status CSV: %s", exc)
        return 2

    n_ok = sum(1 for row in results if row["error"] is None)
    logger.info("forced photometry complete: %d/%d frame(s) measured", n_ok, len(results))
    log_peak_rss_summary("forced")
    return 0 if n_ok else 1


def _call(job: tuple[Any, Path, dict[str, Any] | None]) -> dict[str, Any]:
    fn, path, quality = job
    return fn(path, quality=quality)


SUBCOMMAND = Subcommand(
    name="forced",
    help="Fixed-centroid (forced) aperture photometry for a night of _proc.fits products.",
    add_arguments=_add_arguments,
    run=_run,
)
register(SUBCOMMAND)
