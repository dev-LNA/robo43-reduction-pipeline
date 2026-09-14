"""``robo43 photometry`` -- measure photometry (source detection, aperture/auto fluxes, FWHM, zero point) on existing _proc.fits products.

Reads each input's ``SCI``/``ERR``/``DQ`` planes and header, runs :func:`robo43.photometry.catalog.measure_sources`, and on success writes the resulting CATALOG and CATFLAGS HDUs into every plane. By default nothing is modified in place: a measured copy is written under ``--outdir`` (or a sibling ``photometry`` directory next to the input), and only ``--in-place`` overwrites the input itself. A per-input status CSV is written alongside the outputs, following the convention ``robo43 reduce`` uses for ``proc_status.csv``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from robo43.cli.common import configure_from_args
from robo43.cli.registry import Subcommand, register
from robo43.instruments.profile import resolve_profile
from robo43.io.fits_io import iter_fits_files, read_header, write_fits
from robo43.photometry.catalog import measure_sources
from robo43.products.catalog import build_catalog_hdu
from robo43.products.dq import DQ_DTYPE, dq_bintable

if TYPE_CHECKING:
    from robo43.config import Settings
    from robo43.instruments.profile import InstrumentProfile

__all__ = ["SUBCOMMAND"]

logger = logging.getLogger(__name__)

STATUS_CSV_NAME = "photometry_status.csv"

STATUS_FIELDNAMES: tuple[str, ...] = (
    "path",
    "raw_name",
    "output_path",
    "n_sources",
    "fwhm_px",
    "fwhm_arcsec",
    "sky_level",
    "sky_rms",
    "zp",
    "n_satur",
    "n_badpix",
    "catalog_csv",
    "duration_s",
    "error",
)

#: The CATALOG columns that are true 2-D (N_sources, N_apertures) vectors --
#: see robo43.products.catalog.CATALOG_COLUMNS.
_APERTURE_VECTOR_COLUMNS: tuple[str, ...] = (
    "FLUX_APER",
    "FLUXERR_APER",
    "MAG_APER",
    "MAGERR_APER",
)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("inputs", nargs="+", help="_proc.fits files or directories")
    placement = parser.add_mutually_exclusive_group()
    placement.add_argument(
        "--in-place", action="store_true", default=False, help="modify each input file in place"
    )
    placement.add_argument(
        "--outdir",
        default=None,
        help="directory for updated products (default: <input dir>/photometry)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="allow overwriting an existing output file",
    )
    parser.add_argument(
        "--zp",
        type=float,
        default=None,
        help="override settings.photometry.instrumental_zp",
    )
    parser.add_argument(
        "--detect-thresh",
        type=float,
        default=None,
        dest="detect_thresh",
        help="override settings.photometry.detect_thresh",
    )
    parser.add_argument(
        "--aper",
        default=None,
        help="comma-separated aperture radii in arcsec, overriding settings.photometry.aperture_radii_arcsec",
    )
    parser.add_argument(
        "--catalog-csv",
        action="store_true",
        default=False,
        dest="catalog_csv",
        help="also write a sibling <stem>_catalog.csv next to each output",
    )


def _settings_with_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    overrides: dict[str, Any] = {}
    if args.zp is not None:
        overrides["instrumental_zp"] = args.zp
    if args.detect_thresh is not None:
        overrides["detect_thresh"] = args.detect_thresh
    if args.aper is not None:
        overrides["aperture_radii_arcsec"] = tuple(float(a) for a in args.aper.split(","))
    if not overrides:
        return settings
    return settings.replace(photometry=dataclasses.replace(settings.photometry, **overrides))


def _resolved_gain(header: fits.Header) -> float | None:
    """Read header.get("GAINVAL"). Return None if the card is absent, or if its value is the literal string "NONE" (case-insensitive, after stripping whitespace). GAIN must never be used, citing GAINVAL as the only correct source."""
    gainval = header.get("GAINVAL")
    if gainval is None:
        return None
    if isinstance(gainval, str) and gainval.strip().upper() == "NONE":
        return None
    try:
        return float(gainval)
    except (TypeError, ValueError):
        return None


def _output_path(path: Path, *, in_place: bool, outdir: Path | None) -> Path:
    if in_place:
        return path
    target_dir = outdir if outdir is not None else path.parent / "photometry"
    return target_dir / path.name


def _status_row(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "raw_name": path.name,
        "output_path": None,
        "n_sources": None,
        "fwhm_px": None,
        "fwhm_arcsec": None,
        "sky_level": None,
        "sky_rms": None,
        "zp": None,
        "n_satur": None,
        "n_badpix": None,
        "catalog_csv": "",
        "duration_s": None,
        "error": None,
    }


def _flatten_aperture_columns_for_csv(table: Table) -> Table:
    """Expand the CATALOG table's 2-D per-aperture columns into scalar columns.

    ``FLUX_APER``, ``FLUXERR_APER``, ``MAG_APER`` and ``MAGERR_APER`` are
    true ``(N_sources, N_apertures)`` vector columns in the CATALOG HDU, but
    astropy's plain CSV writer refuses any column with dimension > 1
    ("column(s) with dimension > 1 cannot be written with this format, try
    using 'ecsv'"). Rather than switching the sibling export to ECSV, each
    vector column is unpacked here into one scalar column per aperture
    radius, 1-based (``FLUX_APER_1`` ... ``FLUX_APER_N``), on a *copy* of
    the table -- the caller's table (the one build_catalog_hdu also
    receives, and that ends up in the FITS CATALOG HDU) is never mutated,
    so the FITS product keeps the original true vector columns unchanged.
    """
    flat = table.copy()
    for name in _APERTURE_VECTOR_COLUMNS:
        if name not in flat.colnames:
            continue
        column = flat[name]
        if column.ndim < 2:
            continue
        n_aper = column.shape[-1]
        for i in range(n_aper):
            flat[f"{name}_{i + 1}"] = column[:, i]
        del flat[name]
    return flat


def _process_one(
    path: Path,
    *,
    profile: InstrumentProfile,
    settings: Settings,
    in_place: bool,
    outdir: Path | None,
    overwrite: bool,
    catalog_csv: bool,
    dry_run: bool,
) -> dict[str, Any]:
    row = _status_row(path)
    start = time.monotonic()
    out_path = _output_path(path, in_place=in_place, outdir=outdir)
    row["output_path"] = str(out_path)
    result = None

    try:
        with fits.open(path) as hdul:
            sci = np.array(hdul["SCI"].data, dtype=np.float32)
            err = np.array(hdul["ERR"].data, dtype=np.float32)
            dq = np.array(hdul["DQ"].data, dtype=DQ_DTYPE)
            header = hdul["SCI"].header

            wcs = WCS(header, hdul)
            gain = _resolved_gain(header)

            result = measure_sources(
                sci, err, dq, header=header, settings=settings, profile=profile, wcs=wcs, gain_e_per_adu=gain
            )

            row.update(
                n_sources=result.n_sources,
                fwhm_px=result.fwhm_px,
                fwhm_arcsec=result.fwhm_arcsec,
                sky_level=result.sky_level,
                sky_rms=result.sky_rms,
                # The effective instrumental zero point actually used for
                # this frame (after any --zp override), not
                # result.metrics.zp -- that field is the *calibrated* zero
                # point from a reference-catalog fit and is None whenever
                # no such calibration ran, which left this status column
                # blank on every real run.
                zp=settings.photometry.instrumental_zp,
                n_satur=result.metrics.n_satur,
                n_badpix=result.metrics.n_badpix,
            )

            if dry_run:
                logger.info('dry-run: %s would measure -> %s', path, out_path)
                row["duration_s"] = time.monotonic() - start
                return row

            catalog_hdu = build_catalog_hdu(
                result.table, result.metrics, aperture_radii=settings.photometry.aperture_radii_arcsec
            )
            catflags_hdu = dq_bintable()

            for name in ("CATALOG", "CATFLAGS"):
                while name in hdul:
                    del hdul[name]
            hdul.append(catalog_hdu)
            hdul.append(catflags_hdu)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            write_fits(out_path, hdul, overwrite=overwrite or in_place)

    except Exception as exc:  # never let one frame's error abort the whole batch
        logger.error("photometry failed for %s: %s", path, exc)
        row["error"] = str(exc)
        row["duration_s"] = time.monotonic() - start
        return row

    # The product write above already succeeded (this point is only reached
    # after a successful, non-dry-run write). The sibling CSV export is an
    # optional convenience layered on top of that already-written product,
    # so a failure here must never turn an already-successful frame into a
    # reported failure -- it gets its own try/except and only ever affects
    # the catalog_csv status field, never row["error"].
    if catalog_csv:
        csv_path = out_path.parent / f"{out_path.stem}_catalog.csv"
        try:
            _flatten_aperture_columns_for_csv(result.table).write(
                csv_path, format="csv", overwrite=True
            )
            row["catalog_csv"] = str(csv_path)
        except Exception as exc:
            logger.warning("failed to write catalog CSV %s for %s: %s", csv_path, path, exc)

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

    logger.info("measuring photometry for %d frame(s)", len(paths))

    def _one(path: Path) -> dict[str, Any]:
        return _process_one(
            path,
            profile=profile,
            settings=settings,
            in_place=args.in_place,
            outdir=outdir,
            overwrite=args.overwrite,
            catalog_csv=args.catalog_csv,
            dry_run=settings.runtime.dry_run,
        )

    nprocs = max(1, settings.runtime.nprocs)
    if nprocs > 1:
        with ThreadPoolExecutor(max_workers=nprocs) as pool:
            rows = list(pool.map(_one, paths))
    else:
        rows = [_one(path) for path in paths]

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

    n_ok = sum(1 for row in rows if row["error"] is None)
    logger.info("photometry complete: %d/%d frame(s) measured", n_ok, len(rows))
    return 0


SUBCOMMAND = Subcommand(
    name="photometry",
    help="Measure photometry for _proc.fits products.",
    add_arguments=_add_arguments,
    run=_run,
)
register(SUBCOMMAND)
