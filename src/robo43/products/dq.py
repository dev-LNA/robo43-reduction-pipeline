"""The data-quality bit assignment.

These bit numbers end up in archived FITS files, so they are fixed for good.
Bits 0-5 are the ones the refactoring plan specifies in section 5.1; the rest
extend it and must not be renumbered.

A ``DQ`` plane is an ``int32`` image in an HDU named ``DQ``.  Nothing in the
pipeline is allowed to invent its own bad-pixel encoding.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["DQ", "DQ_DESCRIPTIONS", "DQ_DTYPE", "dq_bintable", "dq_summary"]

#: Data quality planes are written as signed 32-bit integers: wide enough for
#: the bits below with room to grow, and universally readable.
DQ_DTYPE = np.int32


class DQ(enum.IntFlag):
    """Bit-encoded per-pixel data quality."""

    GOOD = 0
    HOT = 1 << 0
    DEAD = 1 << 1
    UNSTABLE = 1 << 2
    SATURATED = 1 << 3
    COSMIC_RAY = 1 << 4
    NO_DATA = 1 << 5
    NONLINEAR = 1 << 6
    BAD_COLUMN = 1 << 7
    NO_FLAT = 1 << 8
    CLIPPED = 1 << 9


DQ_DESCRIPTIONS: Mapping[DQ, str] = {
    DQ.HOT: "hot pixel (from the normalised dark-current map)",
    DQ.DEAD: "dead/cold or low-QE pixel (from flat frames)",
    DQ.UNSTABLE: "unstable pixel (high variance across the bias stack)",
    DQ.SATURATED: "saturated in this frame",
    DQ.COSMIC_RAY: "cosmic ray identified in this frame",
    DQ.NO_DATA: "no data (outside the detector after reprojection)",
    DQ.NONLINEAR: "above the detector non-linearity threshold in this frame",
    DQ.BAD_COLUMN: "member of a persistently deviant column",
    DQ.NO_FLAT: "no master flat was available for this filter",
    DQ.CLIPPED: "rejected by coadd outlier rejection",
}

#: The static classes: properties of the detector rather than of one exposure.
#: ``calib/masks.py`` builds these once per night; the rest are per-frame.
STATIC_FLAGS = DQ.HOT | DQ.DEAD | DQ.UNSTABLE | DQ.BAD_COLUMN

#: The classes that disqualify a pixel from photometry of any kind.
FATAL_FLAGS = (
    DQ.HOT | DQ.DEAD | DQ.UNSTABLE | DQ.SATURATED | DQ.COSMIC_RAY
    | DQ.NO_DATA | DQ.BAD_COLUMN
)


def dq_bintable(extname: str = "CATFLAGS") -> fits.BinTableHDU:
    """The optional ``CATFLAGS`` HDU: a bit-to-meaning legend.

    Shipping the legend inside the product means a reader never has to consult
    this source file to interpret a ``DQ`` plane or a ``QUALITY_MASK`` column.
    """
    flags = [flag for flag in DQ if flag is not DQ.GOOD]
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="BIT", format="J", array=np.array([flag.bit_length() - 1 for flag in flags])
            ),
            fits.Column(name="VALUE", format="J", array=np.array([int(flag) for flag in flags])),
            fits.Column(name="NAME", format="20A", array=np.array([flag.name for flag in flags])),
            fits.Column(
                name="MEANING",
                format="72A",
                array=np.array([DQ_DESCRIPTIONS[flag] for flag in flags]),
            ),
        ],
        name=extname,
    )
    hdu.header["COMMENT"] = "Bit-to-meaning legend for the DQ plane and QUALITY_MASK column."
    return hdu


def dq_summary(mask: np.ndarray) -> dict[str, float]:
    """Affected pixel fraction per quality class, plus the overall figure.

    The plan asks for "a human-readable summary of the affected pixel
    fraction"; this is the machine-readable half of it, and the values go
    straight into the mask file's header and the nightly report.
    """
    if mask.size == 0:
        msg = "cannot summarise an empty DQ plane"
        raise ValueError(msg)
    total = float(mask.size)
    summary = {
        flag.name.lower(): float(np.count_nonzero(mask & int(flag))) / total
        for flag in DQ
        if flag is not DQ.GOOD
    }
    summary["any"] = float(np.count_nonzero(mask != 0)) / total
    summary["fatal"] = float(np.count_nonzero(mask & int(FATAL_FLAGS))) / total
    return summary


def describe(value: int) -> str:
    """``"HOT|SATURATED"`` for one pixel's DQ value, for log messages."""
    names = [flag.name for flag in DQ if flag is not DQ.GOOD and value & int(flag)]
    return "|".join(names) if names else "GOOD"
