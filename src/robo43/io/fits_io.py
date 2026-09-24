"""Low-level FITS file access: discovery, opening, reading, and writing.

This is the ONLY module in the package allowed to index ``hdul[0]`` (or any
other HDU) directly; every other module goes through :func:`science_hdus` or
:func:`read_image`.  No function here returns an open
:class:`astropy.io.fits.HDUList` -- every path that opens a file closes it,
via the :func:`open_fits` context manager.
"""

from __future__ import annotations

import glob as glob_module
import logging
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from astropy.io import fits

from robo43.instruments.profile import InstrumentProfile

logger = logging.getLogger(__name__)

FITS_SUFFIXES: tuple[str, ...] = (
    ".fits", ".fit", ".fts",
    ".fits.fz", ".fits.gz", ".fit.fz", ".fit.gz",
    ".fts.fz", ".fts.gz",
)


def _is_fits_suffix(path: Path) -> bool:
    lowered = str(path).lower()
    return any(lowered.endswith(suf) for suf in FITS_SUFFIXES)


def _iter_one(source: str | os.PathLike[str]) -> list[Path]:
    path = Path(source)
    text = str(source)

    if any(ch in text for ch in ("*", "?", "[")):
        return [Path(p) for p in glob_module.glob(text)]  # noqa: PTH207

    if path.is_dir():
        patterns = tuple(f"*{suf}" for suf in FITS_SUFFIXES)
        found: list[Path] = []
        for pattern in patterns:
            found.extend(path.glob(pattern))
        return found

    if path.is_file():
        if _is_fits_suffix(path):
            return [path]
        # Text file listing paths, one per line.
        results: list[Path] = []
        base_dir = path.parent
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            entry = Path(stripped)
            if not entry.is_absolute():
                entry = base_dir / entry
            results.append(entry)
        return results

    return []


def iter_fits_files(
    paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    patterns: Sequence[str] | None = None,
    *,
    recursive: bool = False,
) -> Iterator[Path]:
    """Yield FITS file paths from a directory, glob, file, or list-file.

    ``paths`` may be a single directory, glob pattern, FITS file, or a text
    file listing paths (one per line); or a sequence of any of those.  Output
    is always sorted and deduplicated.
    """
    if isinstance(paths, (str, os.PathLike)):
        sources: Sequence[str | os.PathLike[str]] = [paths]
    else:
        sources = paths

    collected: list[Path] = []
    for source in sources:
        path = Path(source)
        text = str(source)
        is_glob = any(ch in text for ch in ("*", "?", "["))
        if not is_glob and path.is_dir():
            use_patterns = patterns if patterns is not None else tuple(
                f"*{suf}" for suf in FITS_SUFFIXES
            )
            for pattern in use_patterns:
                if recursive:
                    collected.extend(path.rglob(pattern))
                else:
                    collected.extend(path.glob(pattern))
        else:
            collected.extend(_iter_one(source))

    unique = sorted(set(collected))
    yield from unique


def _is_dataless(header: fits.Header) -> bool:
    """True when a header describes no pixel data.

    Real T80S primaries declare ``NAXIS = 1`` with ``NAXIS1 = 0`` rather
    than ``NAXIS = 0``, so both forms must count as empty.
    """
    naxis = int(header.get("NAXIS", 0) or 0)
    return naxis == 0 or any(int(header.get(f"NAXIS{i}", 0) or 0) == 0 for i in range(1, naxis + 1))


@contextmanager
def open_fits(path: str | os.PathLike[str], mode: str = "readonly") -> Iterator[fits.HDUList]:
    """Open a FITS file as a context manager, guaranteeing it is closed."""
    with fits.open(path, mode=mode) as hdul:
        yield hdul


def science_hdus(
    hdul: fits.HDUList, profile: InstrumentProfile
) -> list[fits.ImageHDU | fits.PrimaryHDU | fits.CompImageHDU]:
    """Return the science-data HDU(s) of ``hdul`` per ``profile.layout``."""
    science_hdu = profile.layout.science_hdu

    if isinstance(science_hdu, int):
        return [hdul[science_hdu]]

    if science_hdu == "all":
        return [
            hdu
            for hdu in hdul
            if hdu.data is not None
            and isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU))
        ]

    try:
        return [hdul[science_hdu]]
    except KeyError as exc:
        msg = (
            f"instrument profile {profile.name!r}: science extension "
            f"{science_hdu!r} not found in FITS file"
        )
        raise KeyError(msg) from exc


def read_image(
    path: str | os.PathLike[str], profile: InstrumentProfile, *, hdu: int | str | None = None
) -> tuple[np.ndarray, fits.Header]:
    """Read one image HDU as float32 data with BSCALE/BZERO applied."""
    with open_fits(path) as hdul:
        chosen = hdul[hdu] if hdu is not None else science_hdus(hdul, profile)[0]
        data = np.array(chosen.data, dtype=np.float32)
        header = chosen.header.copy()
    return data, header


def read_header(
    path: str | os.PathLike[str], profile: InstrumentProfile | None = None
) -> fits.Header:
    """Read a file's header once, merging in science-extension cards if needed.

    When ``profile`` is given, cards are merged in from its declared science
    extension (unless that is HDU 0 or ``"all"``). When ``profile`` is
    ``None`` -- profile auto-detection's own header read, before any profile
    is known -- and HDU 0 has no data (``NAXIS == 0``, an empty primary such
    as a fpacked file's), cards are instead merged in from the first image
    extension (``CompImageHDU`` or ``ImageHDU``) found, so instrument-match
    keywords living only on that extension (e.g. T80S's ``TELESCOP``/
    ``INSTRUME``) are visible to
    :func:`robo43.instruments.profile.resolve_profile`.
    """
    with open_fits(path) as hdul:
        header = hdul[0].header.copy()
        if profile is not None:
            science_hdu = profile.layout.science_hdu
            if science_hdu != 0 and science_hdu != "all" and len(hdul) > 1:
                try:
                    ext = hdul[science_hdu]
                except (IndexError, KeyError):
                    ext = None
                if ext is not None and ext is not hdul[0]:
                    header.update(ext.header)
        elif _is_dataless(header) and len(hdul) > 1:
            for ext in hdul[1:]:
                if isinstance(ext, (fits.ImageHDU, fits.CompImageHDU)):
                    header.update(ext.header)
                    break
        return header


def parse_region(spec: str) -> tuple[slice, slice]:
    """Parse an IRAF-style ``"[x1:x2,y1:y2]"`` region into numpy ``(row, col)`` slices.

    IRAF regions are 1-based, inclusive, and given in x,y (column,row) order.
    The returned slices are 0-based, half-open, and in numpy (row, col) order
    so that ``array[row_slice, col_slice]`` selects the region.
    """
    text = spec.strip()
    if not (text.startswith("[") and text.endswith("]")):
        msg = f"invalid region spec {spec!r}: must be of the form '[x1:x2,y1:y2]'"
        raise ValueError(msg)

    inner = text[1:-1]
    parts = inner.split(",")
    if len(parts) != 2:
        msg = f"invalid region spec {spec!r}: expected exactly one comma-separated x,y pair"
        raise ValueError(msg)

    def _parse_axis(axis_spec: str) -> tuple[int, int]:
        bounds = axis_spec.split(":")
        if len(bounds) != 2:
            msg = f"invalid region spec {spec!r}: expected 'lo:hi' bounds, got {axis_spec!r}"
            raise ValueError(msg)
        try:
            lo, hi = int(bounds[0]), int(bounds[1])
        except ValueError as exc:
            msg = f"invalid region spec {spec!r}: non-numeric bound in {axis_spec!r}"
            raise ValueError(msg) from exc
        return lo, hi

    x1, x2 = _parse_axis(parts[0])
    y1, y2 = _parse_axis(parts[1])

    col_slice = slice(x1 - 1, x2)
    row_slice = slice(y1 - 1, y2)
    return row_slice, col_slice


def write_fits(
    path: str | os.PathLike[str],
    hdus: fits.HDUList | Sequence[fits.PrimaryHDU | fits.ImageHDU],
    *,
    overwrite: bool = False,
    checksum: bool = True,
    compress: bool = False,
) -> Path:
    """Write a FITS file atomically: write to a temp path, fsync, then replace."""
    final_path = Path(path)

    if not overwrite and final_path.exists():
        msg = f"{final_path} already exists (pass overwrite=True to replace it)"
        raise FileExistsError(msg)

    if isinstance(hdus, fits.HDUList):
        hdul = hdus
    else:
        hdu_list = list(hdus)
        if compress:
            converted = []
            for i, item in enumerate(hdu_list):
                if (
                    i > 0
                    and item.data is not None
                    and not isinstance(item, fits.CompImageHDU)
                    and isinstance(item, fits.ImageHDU)
                ):
                    converted.append(fits.CompImageHDU(data=item.data, header=item.header))
                else:
                    converted.append(item)
            hdu_list = converted
        hdul = fits.HDUList(hdu_list)

    tmp_path = final_path.with_name(final_path.name + ".tmp")
    try:
        hdul.writeto(tmp_path, overwrite=True, checksum=checksum)
        with tmp_path.open("rb+") as fh:
            fh.flush()
            os.fsync(fh.fileno())
        tmp_path.replace(final_path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return final_path
