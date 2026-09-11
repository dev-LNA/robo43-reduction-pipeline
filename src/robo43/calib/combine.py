"""Sigma-clipped stack combination for master calibration frames.

Legacy ``src/make_calibration_frames.py`` (bias building at lines 88-101,
flat building at lines 161-183) collapses the stack of input frames with
``np.median`` *first* and only then runs ``sigma_clip`` on the resulting
2-D image. That clips pixels relative to their spatial *neighbours* in the
already-combined image, rather than relative to the other frames at the
same pixel -- it destroys real structure such as a gradient or a hot
column, instead of rejecting genuine per-frame outliers (cosmic rays,
transient defects).

The fix implemented here is to sigma-clip along the stack axis (axis=0,
across frames, per pixel) *before* combining. This module also fixes two
related bugs in the legacy code: a ``weight_map = np.where(mask, 0,
1.0/variance)`` that divides by zero (and warns) wherever the per-pixel
variance across the stack is zero, and a master bias that never had its
masked pixels filled (unlike the flat), risking a ``MaskedArray`` reaching
``fits.PrimaryHDU``.

Scope limit: this module only combines in-memory ``numpy`` stacks
(``Sequence[np.ndarray]``) or row-block loaders supplied by the caller; it
does not know how to read frames from disk (that is ``robo43.io``'s job,
kept out of this module's dependency graph on purpose).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy.stats import sigma_clip

from robo43.config import CalibSettings
from robo43.exceptions import CalibrationError

logger = logging.getLogger(__name__)

__all__ = [
    "CombineResult",
    "combine_lazy",
    "combine_stack",
    "row_block_size",
    "sigma_clip_stack",
]

# Asymptotic relative efficiency of the sample median vs. the sample mean for
# normally distributed data is 2/pi; the median's variance is therefore
# inflated by pi/2 relative to var(mean) = var_sample/n. Applied to
# var_sample/n_used below when method == "median".
_MEDIAN_VARIANCE_FACTOR = math.pi / 2.0

# When no read noise is supplied there is no physical variance floor available,
# so a combined variance below this fraction of the frame's median variance is
# treated as unmeasurable rather than real.  Without this guard a perfectly
# stable pixel receives an effectively infinite inverse-variance weight.
_MIN_VARIANCE_FRACTION = 1e-6


@dataclass(frozen=True, slots=True)
class CombineResult:
    """The outcome of combining a stack of frames into one master frame."""

    data: np.ndarray  # float32, 2-D, never masked, no NaN/inf
    variance: np.ndarray  # float32, variance OF THE COMBINED PIXEL
    weight: np.ndarray  # float32, inverse-variance weight, 0 where unusable
    mask: np.ndarray  # bool, True = pixel unusable in the master
    n_used: np.ndarray  # int16, surviving frames per pixel after clipping
    n_input: int
    method: str
    rejected_fraction: float
    sigma_low: float  # the sigma_clip_low actually used
    sigma_high: float  # the sigma_clip_high actually used
    maxiters: int  # the sigma_clip_maxiters actually used

    def header_cards(self) -> list[tuple[str, Any, str]]:
        """FITS header cards summarising how this master frame was built."""
        return [
            ("NCOMBINE", self.n_input, "Number of input frames"),
            ("COMBMETH", self.method, "Stack combination method"),
            ("CLIPLO", self.sigma_low, "Sigma-clip lower threshold"),
            ("CLIPHI", self.sigma_high, "Sigma-clip upper threshold"),
            ("CLIPITER", self.maxiters, "Sigma-clip max iterations"),
            (
                "REJFRAC",
                self.rejected_fraction,
                "Fraction of frame*pixel entries rejected",
            ),
            ("NUSEDMIN", int(self.n_used.min()), "Minimum surviving frames at any pixel"),
            ("NUSEDMAX", int(self.n_used.max()), "Maximum surviving frames at any pixel"),
        ]


def sigma_clip_stack(stack: np.ndarray, settings: CalibSettings) -> np.ma.MaskedArray:
    """Sigma-clip a 3-D stack (n_frames, H, W) along axis 0.

    This clips across frames, per pixel -- the corrected statistic. Legacy
    code collapsed the stack first and clipped the resulting 2-D image
    spatially, which rejects pixels relative to their *neighbours* instead
    of relative to the other frames at the same pixel, and destroys real
    structure such as a gradient or a hot column.

    If ``stack.shape[0] <= 1`` clipping is meaningless; the stack is
    returned unmasked (documented degrade-gracefully behaviour). Callers
    that require at least 2 frames (:func:`combine_stack`,
    :func:`combine_lazy`) validate that themselves and raise
    :class:`CalibrationError` before reaching here.
    """
    if stack.shape[0] <= 1:
        return np.ma.masked_array(stack, mask=np.zeros(stack.shape, dtype=bool))
    return sigma_clip(
        stack,
        sigma_lower=settings.sigma_clip_low,
        sigma_upper=settings.sigma_clip_high,
        maxiters=settings.sigma_clip_maxiters,
        axis=0,
        masked=True,
        cenfunc="median",
        stdfunc="std",
    )


def _combine_core(
    stack: np.ndarray,  # float64, shape (n, h, w), already scaled
    settings: CalibSettings,
    *,
    method: str,
    read_noise_e: float | None,
    gain_e_per_adu: float | None,
    weights: Sequence[float] | None,
    fill: float,
) -> CombineResult:
    """Sigma-clip and combine an already-assembled stack (shared core)."""
    n = stack.shape[0]
    clipped = sigma_clip_stack(stack, settings)
    mask_cube = np.ma.getmaskarray(clipped)  # always a real bool array, never nomask
    n_used = (~mask_cube).sum(axis=0).astype(np.int16)

    if method == "mean":
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape != (n,):
                raise CalibrationError(
                    f"weights length {w.shape} does not match n_frames {n}"
                )
            w_cube = np.ma.array(np.broadcast_to(w[:, None, None], stack.shape), mask=mask_cube)
            wsum = w_cube.sum(axis=0)
            data_masked = (clipped * w_cube).sum(axis=0) / wsum
            var_sample = clipped.var(axis=0, ddof=1)
            w2sum = (w_cube**2).sum(axis=0)
            var_combined = var_sample * w2sum / (wsum**2)
        else:
            data_masked = clipped.mean(axis=0)
            var_sample = clipped.var(axis=0, ddof=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                var_combined = var_sample / n_used
    elif method == "median":
        if weights is not None:
            logger.warning("weights are ignored for method='median'")
        data_masked = np.ma.median(clipped, axis=0)
        var_sample = clipped.var(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            var_combined = (var_sample / n_used) * _MEDIAN_VARIANCE_FACTOR
    else:
        raise CalibrationError(
            f"unknown combine method {method!r} (expected 'mean' or 'median')"
        )

    var_mask = np.ma.getmaskarray(var_combined)
    var_arr = np.ma.filled(var_combined.astype(np.float64), np.nan)

    undefined = var_mask | ~np.isfinite(var_arr) | (var_arr <= 0)

    if read_noise_e is not None:
        rn_adu = read_noise_e / gain_e_per_adu if gain_e_per_adu else read_noise_e
        floor = rn_adu**2
        # A genuine lower bound, not merely a substitute where the variance is
        # undefined.  A dead or perfectly stable pixel has a sample variance of
        # zero -- or, after floating-point rounding, something like 1e-28, which
        # survives a `<= 0` test and yields a weight of 1e28.  One such pixel
        # would swamp every inverse-variance-weighted combination downstream,
        # so the read-noise variance is imposed as a floor everywhere.
        var_arr = np.where(undefined, floor, np.maximum(var_arr, floor))
        unmeasurable = np.zeros_like(undefined)
    else:
        # With no read noise supplied there is no physical floor to impose, so
        # instead refuse to believe a variance far below the frame's own noise
        # level: it cannot be measured from a handful of frames, and treating
        # it as real produces the same runaway weight.  Such pixels are marked
        # unusable rather than given an invented variance.
        finite_positive = var_arr[np.isfinite(var_arr) & (var_arr > 0)]
        if finite_positive.size:
            scale = float(np.median(finite_positive))
            unmeasurable = np.isfinite(var_arr) & (var_arr > 0) & (
                var_arr < _MIN_VARIANCE_FRACTION * scale
            )
        else:
            unmeasurable = np.zeros_like(undefined)

    with np.errstate(divide="ignore", invalid="ignore"):
        weight_arr = 1.0 / var_arr

    bad = undefined | unmeasurable | ~np.isfinite(weight_arr)
    if read_noise_e is not None:
        # The floor made every pixel's variance usable except where no frame
        # survived at all.
        bad = unmeasurable | ~np.isfinite(weight_arr) | ~np.isfinite(var_arr)
    bad = bad | (n_used == 0)

    weight_arr = np.where(bad, 0.0, weight_arr)
    var_arr = np.where(bad, 0.0, var_arr)

    data_arr = np.ma.filled(data_masked.astype(np.float64), fill)
    data_arr = np.where(n_used == 0, fill, data_arr)

    rejected_fraction = float(mask_cube.sum()) / float(mask_cube.size)

    return CombineResult(
        data=data_arr.astype(np.float32),
        variance=var_arr.astype(np.float32),
        weight=weight_arr.astype(np.float32),
        mask=bad,
        n_used=n_used,
        n_input=n,
        method=method,
        rejected_fraction=rejected_fraction,
        sigma_low=settings.sigma_clip_low,
        sigma_high=settings.sigma_clip_high,
        maxiters=settings.sigma_clip_maxiters,
    )


def combine_stack(
    frames: Sequence[np.ndarray],
    settings: CalibSettings,
    *,
    method: str | None = None,
    scales: Sequence[float] | None = None,
    read_noise_e: float | None = None,
    gain_e_per_adu: float | None = None,
    weights: Sequence[float] | None = None,
    fill: float = 0.0,
) -> CombineResult:
    """Sigma-clip (across frames, per pixel) and combine a stack of frames.

    ``frames`` must be a sequence of 2-D ``numpy`` arrays, all the same
    shape -- this function does not know how to load frames from disk (that
    is deliberately left to ``robo43.io``, kept out of this module).

    Requires at least 2 frames: a single frame cannot be sigma-clipped, so
    ``n_frames == 1`` raises :class:`CalibrationError` rather than silently
    degrading -- callers with a single frame should use it directly instead
    of calling this function.
    """
    frame_list = list(frames)
    if not frame_list:
        raise CalibrationError("combine_stack received an empty list of frames")

    arrays: list[np.ndarray] = []
    first_shape: tuple[int, ...] | None = None
    for i, frame in enumerate(frame_list):
        arr = np.asarray(frame)
        if arr.ndim != 2:
            raise CalibrationError(
                f"frame {i} has ndim={arr.ndim}, expected a 2-D array"
            )
        if first_shape is None:
            first_shape = arr.shape
        elif arr.shape != first_shape:
            raise CalibrationError(
                f"frame {i} has shape {arr.shape}, expected {first_shape} "
                "(all frames must share the same shape)"
            )
        arrays.append(arr)

    method_ = method or settings.combine_method
    if method_ not in ("mean", "median"):
        raise CalibrationError(
            f"unknown combine method {method_!r} (expected 'mean' or 'median')"
        )

    n = len(arrays)
    if n == 1:
        raise CalibrationError(
            "combine_stack requires at least 2 frames for sigma-clipped "
            "combination (got 1); a single frame cannot be sigma-clipped -- "
            "use it directly instead of calling combine_stack"
        )

    stack = np.stack(arrays).astype(np.float64)

    if scales is not None:
        scale_arr = np.asarray(scales, dtype=np.float64)
        if scale_arr.shape != (n,):
            raise CalibrationError(
                f"scales length {scale_arr.shape} does not match n_frames {n}"
            )
        stack = stack * scale_arr[:, None, None]

    return _combine_core(
        stack,
        settings,
        method=method_,
        read_noise_e=read_noise_e,
        gain_e_per_adu=gain_e_per_adu,
        weights=weights,
        fill=fill,
    )


def row_block_size(shape: tuple[int, int], n_frames: int, max_memory_mb: float) -> int:
    """Rows per chunk for :func:`combine_lazy`'s memory-bounded loop.

    The legacy code (``make_calibration_frames.py:88-101``) builds a Python
    list of ``n_frames`` full H*W float64 arrays and calls ``np.median``,
    which internally allocates a further full-size copy: roughly
    ``2 * n_frames * H * W * 8`` bytes resident at once, independent of how
    much memory the caller allows.

    Here we bound memory to one row-block of the stack instead: a block of
    ``rows`` rows costs roughly ``2 * n_frames * rows * W * 8`` bytes (the
    raw stack slice plus one working copy from sigma_clip/combine). Solve
    ``2 * n_frames * rows * W * 8 <= max_memory_mb * 1024**2`` for ``rows``.
    """
    h, w = shape
    max_bytes = max_memory_mb * 1024.0 * 1024.0
    per_row_bytes = 2.0 * n_frames * w * 8.0
    if per_row_bytes <= 0:
        return h
    block = int(max_bytes // per_row_bytes)
    return max(1, min(block, h))


def combine_lazy(
    loaders: Sequence[Callable[[tuple[slice, slice]], np.ndarray]],
    shape: tuple[int, int],
    settings: CalibSettings,
    *,
    method: str | None = None,
    scales: Sequence[float] | None = None,
    read_noise_e: float | None = None,
    gain_e_per_adu: float | None = None,
    weights: Sequence[float] | None = None,
    fill: float = 0.0,
) -> CombineResult:
    """Memory-bounded equivalent of :func:`combine_stack`.

    Each element of ``loaders`` is a callable taking a ``(row_slice,
    col_slice)`` tuple and returning the corresponding 2-D chunk for one
    frame (e.g. ``loader((slice(0, 64), slice(0, W)))``). Frames are
    combined row-block by row-block, so at most one row-block's stack
    (``n_frames * block_rows * W`` floats) is held in memory at once,
    unlike :func:`combine_stack` which holds the whole stack.

    Per-pixel sigma-clipping and combination have no cross-row or
    cross-column interaction, so this produces bit-identical float32
    results to :func:`combine_stack` given the same input data.
    """
    loader_list = list(loaders)
    if not loader_list:
        raise CalibrationError("combine_lazy received an empty list of loaders")

    method_ = method or settings.combine_method
    if method_ not in ("mean", "median"):
        raise CalibrationError(
            f"unknown combine method {method_!r} (expected 'mean' or 'median')"
        )

    n = len(loader_list)
    if n == 1:
        raise CalibrationError(
            "combine_lazy requires at least 2 frames for sigma-clipped "
            "combination (got 1); a single frame cannot be sigma-clipped -- "
            "use it directly instead of calling combine_lazy"
        )

    if scales is not None:
        scale_arr = np.asarray(scales, dtype=np.float64)
        if scale_arr.shape != (n,):
            raise CalibrationError(
                f"scales length {scale_arr.shape} does not match n_frames {n}"
            )
    else:
        scale_arr = None

    h, w = shape
    block = row_block_size(shape, n, settings.max_memory_mb)

    data = np.empty(shape, dtype=np.float32)
    variance = np.empty(shape, dtype=np.float32)
    weight = np.empty(shape, dtype=np.float32)
    mask = np.empty(shape, dtype=bool)
    n_used = np.empty(shape, dtype=np.int16)

    weighted_rejected = 0.0
    for r0 in range(0, h, block):
        r1 = min(r0 + block, h)
        rows = slice(r0, r1)
        cols = slice(0, w)
        block_stack = np.stack(
            [np.asarray(loader((rows, cols))) for loader in loader_list]
        ).astype(np.float64)

        if scale_arr is not None:
            block_stack = block_stack * scale_arr[:, None, None]

        block_result = _combine_core(
            block_stack,
            settings,
            method=method_,
            read_noise_e=read_noise_e,
            gain_e_per_adu=gain_e_per_adu,
            weights=weights,
            fill=fill,
        )

        data[rows, cols] = block_result.data
        variance[rows, cols] = block_result.variance
        weight[rows, cols] = block_result.weight
        mask[rows, cols] = block_result.mask
        n_used[rows, cols] = block_result.n_used

        block_entries = n * (r1 - r0) * w
        weighted_rejected += block_result.rejected_fraction * block_entries

    rejected_fraction = weighted_rejected / float(n * h * w)

    return CombineResult(
        data=data,
        variance=variance,
        weight=weight,
        mask=mask,
        n_used=n_used,
        n_input=n,
        method=method_,
        rejected_fraction=rejected_fraction,
        sigma_low=settings.sigma_clip_low,
        sigma_high=settings.sigma_clip_high,
        maxiters=settings.sigma_clip_maxiters,
    )
