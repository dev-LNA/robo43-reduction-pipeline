"""The one-file-per-frame reduced product: a multi-extension FITS container.

Refactoring plan section 4.4 replaces the old scattered sidecar files (a
science image here, an error map there, a catalogue in yet another place)
with a single multi-extension FITS (MEF) file per reduced frame, so the
image and its catalogue can never drift apart on disk. :class:`ReducedProduct`
is that container: ``SCI``/``ERR``/``DQ`` planes plus an optional catalogue,
always written and read together as one atomic unit.

``ERR`` carries the 1-sigma uncertainty -- never variance -- and ``DQ`` is a
bit-encoded data quality plane (see :mod:`robo43.products.dq`); together they
let every downstream consumer of a reduced frame know both how much to trust
each pixel and why. :func:`calibration_cards` builds the header-card block
that traces a product back to exactly which calibration masters and pipeline
version made it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from astropy.io import fits
from astropy.table import Table

from robo43.io.fits_io import write_fits
from robo43.products.dq import DQ_DTYPE, dq_bintable
from robo43.products.weight import weight_header_cards
from robo43.provenance import provenance_cards, sha256_file

if TYPE_CHECKING:
    from robo43.config import Settings
    from robo43.instruments.profile import InstrumentProfile
    from robo43.io.meta import FrameMeta

__all__ = ["ReducedProduct", "calibration_cards"]


@dataclass(slots=True)
class ReducedProduct:
    """The full contents of one reduced frame's multi-extension FITS file.

    HDU layout (exact order, on :meth:`to_hdulist`/write):
    ``SCI`` (primary), ``ERR``, ``DQ``, then the optional ``WEIGHT``
    extension, then the optional ``CATALOG`` and ``CATFLAGS`` extensions.
    """

    sci: np.ndarray
    err: np.ndarray
    dq: np.ndarray
    header: fits.Header
    catalog: Table | None = None
    catalog_header: fits.Header | None = None
    bunit: str = "electron"
    weight: np.ndarray | None = None

    def to_hdulist(self, *, include_catflags: bool = True) -> fits.HDUList:
        """Assemble the full MEF ``HDUList`` for this product."""
        sci_header = self.header.copy()
        sci_header["EXTNAME"] = "SCI"
        sci_header["BUNIT"] = self.bunit
        primary = fits.PrimaryHDU(data=np.asarray(self.sci, dtype=np.float32), header=sci_header)

        err_header = fits.Header()
        err_header["BUNIT"] = self.bunit
        err_header["COMMENT"] = "1-sigma uncertainty, same units as SCI"
        err_hdu = fits.ImageHDU(
            data=np.asarray(self.err, dtype=np.float32), header=err_header, name="ERR"
        )

        dq_hdu = fits.ImageHDU(data=np.asarray(self.dq, dtype=DQ_DTYPE), name="DQ")

        hdus: list[fits.PrimaryHDU | fits.ImageHDU | fits.BinTableHDU] = [
            primary,
            err_hdu,
            dq_hdu,
        ]

        if self.weight is not None:
            weight_header = fits.Header()
            for key, value, comment in weight_header_cards(self.weight, bunit=self.bunit):
                weight_header[key] = (value, comment)
            weight_header["COMMENT"] = "Inverse-variance weight; zero at flagged pixels"
            weight_hdu = fits.ImageHDU(
                data=np.asarray(self.weight, dtype=np.float32),
                header=weight_header,
                name="WEIGHT",
            )
            hdus.append(weight_hdu)

        if self.catalog is not None:
            cat_header = (
                self.catalog_header.copy() if self.catalog_header is not None else fits.Header()
            )
            cat_hdu = fits.BinTableHDU(
                data=fits.table_to_hdu(self.catalog).data, header=cat_header
            )
            if self.catalog_header is None:
                cat_hdu.header["EXTNAME"] = "CATALOG"
            hdus.append(cat_hdu)
            if include_catflags:
                hdus.append(dq_bintable())

        return fits.HDUList(hdus)

    @classmethod
    def from_hdulist(cls, hdul: fits.HDUList) -> ReducedProduct:
        """Reconstruct a :class:`ReducedProduct` from an open ``HDUList``."""
        sci = np.array(hdul["SCI"].data, dtype=np.float32)
        header = hdul["SCI"].header.copy()
        err = np.array(hdul["ERR"].data, dtype=np.float32)
        dq = np.array(hdul["DQ"].data, dtype=DQ_DTYPE)
        bunit = str(header.get("BUNIT", "electron"))

        catalog: Table | None = None
        catalog_header: fits.Header | None = None
        if "CATALOG" in hdul:
            catalog = Table(hdul["CATALOG"].data)
            catalog_header = hdul["CATALOG"].header.copy()

        weight: np.ndarray | None = None
        if "WEIGHT" in hdul:
            weight = np.array(hdul["WEIGHT"].data, dtype=np.float32)

        return cls(
            sci=sci,
            err=err,
            dq=dq,
            header=header,
            catalog=catalog,
            catalog_header=catalog_header,
            bunit=bunit,
            weight=weight,
        )

    def write(
        self, path: str | os.PathLike[str], *, overwrite: bool = False, compress: bool = False
    ) -> Path:
        """Write this product to ``path`` atomically, via :func:`write_fits`."""
        return write_fits(path, self.to_hdulist(), overwrite=overwrite, compress=compress)

    @classmethod
    def read(cls, path: str | os.PathLike[str]) -> ReducedProduct:
        """Read a :class:`ReducedProduct` back from a MEF file on disk."""
        with fits.open(path) as hdul:
            return cls.from_hdulist(hdul)


def calibration_cards(
    *,
    bias: str | os.PathLike[str] | None = None,
    dark: str | os.PathLike[str] | None = None,
    flat: str | os.PathLike[str] | None = None,
    mask: str | os.PathLike[str] | None = None,
    dark_scale: float | None = None,
    crrej: bool = False,
    overscan: bool = False,
    profile: InstrumentProfile | None = None,
    meta: FrameMeta | None = None,
    settings: Settings | None = None,
) -> list[tuple[str, Any, str]]:
    """Build the header-card block tracing a product to the masters that made it.

    NOTE (contract gap, flagged upward, not silently worked around): Appendix
    B.1 also lists ``ASTSOLVR ASTRMS1 ASTRMS2 ASTNREF ASTEPOCH WCSSOLVD
    NCOMBINE TEXPTIME`` as belonging to this same card block, but this
    function has no parameter carrying a ``SolveResult`` or coadd-membership
    data, so it cannot compute them. Whichever later phase holds that data
    (astrometry solving, coadd assembly) is responsible for appending those
    cards itself via :func:`robo43.provenance.apply_cards` on top of what this
    function returns.
    """
    cards: list[tuple[str, Any, str]] = []

    if settings is not None:
        cards.extend(provenance_cards(settings))

    if profile is not None:
        cards.append(("INSTPROF", profile.name, "instrument profile name"))
        cards.append(("INSTPVER", profile.version, "instrument profile version"))

    # A *skipped* master is recorded as the literal string "NONE" -- it was
    # deliberately not applied, which is different from a metric that simply
    # could not be computed (those are omitted entirely, per DARKSCL below).
    masters = (
        (bias, "BIASFIL", "master bias"),
        (dark, "DARKFIL", "master dark"),
        (flat, "FLATFIL", "master flat"),
        (mask, "MASKFIL", "bad-pixel mask"),
    )
    for path, keyword, comment in masters:
        if path is not None:
            cards.append((keyword, Path(path).name, f"{comment} used"))
            cards.append(("HISTORY", f"{keyword} sha256={sha256_file(path)}", ""))
        else:
            cards.append((keyword, "NONE", f"no {comment} applied"))

    # DARKSCL is an ordinary derived metric, not a skipped/applied master, so
    # it follows the normal omit-if-unknown rule.
    if dark_scale is not None:
        cards.append(("DARKSCL", dark_scale, "dark scaling factor applied"))

    cards.append(("CRREJ", bool(crrej), "cosmic-ray rejection applied"))
    cards.append(("OSCANCOR", bool(overscan), "overscan correction applied"))

    if meta is not None:
        for i, record in enumerate(meta.provenance_records(), start=1):
            cards.append((f"KWSRC{i}", record, ""))

    return cards
