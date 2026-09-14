"""The per-frame inverse-variance ``WEIGHT`` plane.

A weight plane is the reciprocal of the pixel variance: ``w = 1 / err**2``.
It exists so that downstream combination (coaddition, PSF fitting, anything
that averages pixels across frames) can weigh each pixel by how much it is
worth trusting, in one array, without every consumer having to re-derive
inverse variance from ``ERR`` and re-apply the ``DQ`` plane itself. A pixel is
driven to exactly zero weight -- never a small positive number -- wherever it
is unusable: any ``FATAL_FLAGS`` bit set in ``DQ``, or a non-finite or
non-positive ``ERR`` value. Zero weight is the convention a combiner checks
for to exclude a pixel outright, so this module guarantees no NaN or inf ever
leaks into a ``WEIGHT`` plane.

Calibration masters (bias, dark, flat, the bad-pixel mask itself) never carry
a ``WEIGHT`` plane; only reduced science frames do.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robo43.products.dq import FATAL_FLAGS

__all__ = ["WEIGHT_DTYPE", "build_weight", "weight_header_cards", "weight_summary"]

#: Weight planes are written as 32-bit floats: enough precision for an
#: inverse-variance weight, and it matches SCI/ERR.
WEIGHT_DTYPE = np.float32


def build_weight(err: np.ndarray, dq: np.ndarray, *, bad_flags: int = FATAL_FLAGS) -> np.ndarray:
    """Build the inverse-variance weight plane from an ``ERR`` plane and a ``DQ`` plane.

    ``w = 1 / err**2`` everywhere the pixel is usable; exactly ``0.0`` wherever
    any bit of ``bad_flags`` is set in ``dq``, or ``err`` is non-finite or
    ``<= 0``. Neither input is mutated, and the result never contains a NaN
    or an inf.

    Args:
        err: 1-sigma uncertainty plane, any shape.
        dq: Data-quality plane, same shape as ``err``.
        bad_flags: DQ bits that zero a pixel's weight. Defaults to
            :data:`robo43.products.dq.FATAL_FLAGS`. Accepts a ``DQ`` flag
            combination or a plain int.

    Returns:
        A float32 array of the same shape as ``err``.

    Raises:
        ValueError: if ``err.shape != dq.shape``.
    """
    if err.shape != dq.shape:
        msg = f"err shape {err.shape} does not match dq shape {dq.shape}"
        raise ValueError(msg)

    err = np.asarray(err)
    dq = np.asarray(dq)

    bad = (dq.astype(np.int64) & int(bad_flags)) != 0
    bad |= ~np.isfinite(err)
    bad |= err <= 0

    safe_err = np.where(bad, 1.0, err).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = 1.0 / (safe_err ** 2)

    weight = np.where(bad | ~np.isfinite(weight), 0.0, weight)
    # Guard against overflow to inf when narrowing an extreme float64 weight
    # (from a tiny but positive err) down to float32.
    with np.errstate(over="ignore"):
        weight32 = weight.astype(WEIGHT_DTYPE)
    weight32 = np.where(np.isfinite(weight32), weight32, np.float32(0.0)).astype(WEIGHT_DTYPE)
    return weight32


def weight_summary(weight: np.ndarray) -> dict[str, float]:
    """Fraction of zero-weight pixels, plus median/mean over the positive ones.

    ``median`` and ``mean`` are computed over the strictly-positive weights
    only, so a handful of very good pixels are not diluted by the flagged
    majority. All three are ``0.0`` -- never NaN -- when no pixel has a
    positive weight.
    """
    total = float(weight.size)
    positive = weight[weight > 0]
    zero_fraction = 1.0 - (float(positive.size) / total) if total > 0 else 0.0
    if positive.size == 0:
        return {"zero_fraction": zero_fraction, "median": 0.0, "mean": 0.0}
    return {
        "zero_fraction": zero_fraction,
        "median": float(np.median(positive)),
        "mean": float(np.mean(positive)),
    }


def weight_header_cards(
    weight: np.ndarray, *, bunit: str, bad_flags: int = FATAL_FLAGS
) -> list[tuple[str, Any, str]]:
    """FITS header cards for the ``WEIGHT`` HDU: type, units, and a summary."""
    summary = weight_summary(weight)
    return [
        ("WHTTYPE", "INVVAR", "Weight plane type: inverse variance"),
        ("BUNIT", f"1/({bunit})**2", "Unit of the WEIGHT plane"),
        ("WHTFLAGS", int(bad_flags), "DQ bitmask zeroed out of the weight"),
        ("WHTZEROF", summary["zero_fraction"], "Fraction of pixels with zero weight"),
        ("WHTMED", summary["median"], "Median of the positive weights"),
    ]
