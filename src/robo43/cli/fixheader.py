"""``robo43 fixheader`` -- repair raw FITS headers."""

from __future__ import annotations

import argparse
import datetime
import logging
import shutil

from robo43 import __version__
from robo43.cli.common import settings_from_args
from robo43.cli.registry import Subcommand, register
from robo43.exceptions import Robo43Error
from robo43.instruments.keywords import CANONICAL_IMAGETYP
from robo43.instruments.profile import resolve_profile
from robo43.io.fits_io import iter_fits_files, open_fits, read_header, science_hdus
from robo43.io.headers import _find_alias_value, classify_from_object

__all__ = ["SUBCOMMAND"]

logger = logging.getLogger(__name__)


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("inputs", nargs="+")


def _run(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)

    results = []
    any_failed = False

    for path in iter_fits_files(args.inputs):
        result = {
            "path": str(path),
            "imagetyp_changed": False,
            "filter_changed": False,
            "skipped_reason": None,
            "is_dry_run": settings.runtime.dry_run,
        }
        try:
            # Resolve profile
            if args.instrument:
                profile = resolve_profile(name=args.instrument)
            else:
                profile = resolve_profile(header=read_header(path, profile=None))

            header = read_header(path, profile)

            # Get OBJECT value
            raw_object, _ = _find_alias_value(header, profile, "object")
            object_text = str(raw_object) if raw_object is not None else None

            # Classify from OBJECT
            category = classify_from_object(object_text)
            if category is None:
                logger.info("%s: OBJECT missing/empty, skipping", path)
                result["skipped_reason"] = "OBJECT missing/empty"
                results.append(result)
                continue

            target_imagetyp = CANONICAL_IMAGETYP[category]

            # Get current IMAGETYP
            raw_imagetyp, _ = _find_alias_value(header, profile, "imagetyp")
            current_imagetyp = str(raw_imagetyp) if raw_imagetyp is not None else None

            imagetyp_changed = current_imagetyp != target_imagetyp

            # Filter remap (skip for bias/dark)
            filter_changed = False
            if category not in ("bias", "dark"):
                raw_filter, _ = _find_alias_value(header, profile, "filter")
                if raw_filter is not None:
                    target_filter = profile.canonical_filter(str(raw_filter))
                    filter_changed = target_filter != raw_filter

            # Check if any changes needed
            if not imagetyp_changed and not filter_changed:
                logger.info("%s: no changes needed", path)
                result["skipped_reason"] = "no changes needed"
                results.append(result)
                continue

            # Handle dry-run
            if settings.runtime.dry_run:
                if imagetyp_changed:
                    logger.info(
                        "%s: would change IMAGETYP %r -> %r",
                        path,
                        current_imagetyp,
                        target_imagetyp,
                    )
                if filter_changed:
                    logger.info("%s: would change FILTER", path)
                result["imagetyp_changed"] = imagetyp_changed
                result["filter_changed"] = filter_changed
                results.append(result)
                continue

            # Create backup
            backup_path = path.with_name(path.name + ".bak")
            try:
                shutil.copy2(path, backup_path)
            except OSError as exc:
                raise Robo43Error(
                    f"{path}: could not create backup, refusing to modify: {exc}"
                ) from exc

            # Update FITS file
            utc_iso_now = datetime.datetime.now(datetime.UTC).isoformat()
            with open_fits(path, mode="update") as hdul:
                # Find which HDU holds each keyword
                imagetyp_hdu = None
                filter_hdu = None

                # Check primary HDU first
                if imagetyp_changed and (
                    "IMAGETYP" in hdul[0].header
                    or any(
                        alias in hdul[0].header
                        for alias in profile.aliases_for("imagetyp")
                    )
                ):
                    imagetyp_hdu = hdul[0]
                if filter_changed and (
                    "FILTER" in hdul[0].header
                    or any(alias in hdul[0].header for alias in profile.aliases_for("filter"))
                ):
                    filter_hdu = hdul[0]

                # Check science HDUs
                sci_hdus = science_hdus(hdul, profile) if len(hdul) > 1 else []
                if (
                    imagetyp_changed
                    and imagetyp_hdu is None
                    and sci_hdus
                    and (
                        "IMAGETYP" in sci_hdus[0].header
                        or any(
                            alias in sci_hdus[0].header
                            for alias in profile.aliases_for("imagetyp")
                        )
                    )
                ):
                    imagetyp_hdu = sci_hdus[0]
                if (
                    filter_changed
                    and filter_hdu is None
                    and sci_hdus
                    and (
                        "FILTER" in sci_hdus[0].header
                        or any(
                            alias in sci_hdus[0].header
                            for alias in profile.aliases_for("filter")
                        )
                    )
                ):
                    filter_hdu = sci_hdus[0]

                # Default to primary if not found
                if imagetyp_changed and imagetyp_hdu is None:
                    imagetyp_hdu = hdul[0]
                if filter_changed and filter_hdu is None:
                    filter_hdu = hdul[0]

                # Update headers
                if imagetyp_changed:
                    imagetyp_hdu.header["IMAGETYP"] = target_imagetyp
                    imagetyp_hdu.header.add_history(
                        f"robo43 fixheader: IMAGETYP {current_imagetyp!r} -> {target_imagetyp!r} "
                        f"(robo43 {__version__}, {utc_iso_now})"
                    )

                if filter_changed:
                    target_filter = profile.canonical_filter(str(raw_filter))
                    filter_hdu.header["FILTER"] = target_filter
                    filter_hdu.header.add_history(
                        f"robo43 fixheader: FILTER {raw_filter!r} -> {target_filter!r} "
                        f"(robo43 {__version__}, {utc_iso_now})"
                    )

            result["imagetyp_changed"] = imagetyp_changed
            result["filter_changed"] = filter_changed
            results.append(result)

        except Robo43Error as exc:
            logger.error("%s: %s", path, exc)
            result["skipped_reason"] = str(exc)
            any_failed = True
            results.append(result)
            continue
        except Exception as exc:
            logger.error("%s: unexpected error: %s", path, exc)
            result["skipped_reason"] = f"error: {exc}"
            results.append(result)
            continue

    # Print summary
    if results:
        print("\nFixheader Results:")
        print(f"{'Path':<60} {'Changed':<20} {'Status':<30}")
        print("-" * 110)
        for r in results:
            changed = "IMAGETYP, FILTER" if (r["imagetyp_changed"] and r["filter_changed"]) else (
                "IMAGETYP" if r["imagetyp_changed"] else (
                    "FILTER" if r["filter_changed"] else "none"
                )
            )
            if r["skipped_reason"]:
                status = r["skipped_reason"]
            elif r["is_dry_run"] and (r["imagetyp_changed"] or r["filter_changed"]):
                status = "would change (dry-run)"
            else:
                status = "success"
            print(f"{r['path']:<60} {changed:<20} {status:<30}")

    return 2 if any_failed else 0


SUBCOMMAND = Subcommand(
    name="fixheader",
    help="Repair raw FITS headers.",
    add_arguments=_add_arguments,
    run=_run,
)
register(SUBCOMMAND)
