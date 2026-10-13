"""Twilight (sky) flat handling: classification, level scaling, star masking, QC.

Twilight flats are taken with the telescope moved between exposures, so no star
repeats at the same pixel -- but the frame levels differ strongly (the sky
fades or brightens through the sequence).  Combining them with a per-pixel
sigma clip on the *unscaled* frames fails: the across-frame scatter is
dominated by the level differences, 3-sigma clipping never reaches a star, and
every star survives in the master as a compact positive residual.

This module supplies the pieces :func:`robo43.calib.flat.build_master_flats`
uses for a twilight group:

* :func:`is_twilight_flat` -- reliable twilight/sky classification of one flat
  (OBJECT, raw IMAGETYP, or file name);
* :func:`scales_from_levels` / :func:`frame_level` -- per-frame multiplicative
  scales that bring every frame to a common level (the median of the frame
  levels), applied *before* clipping;
* :func:`combine_twilight` -- the memory-bounded (row-block) combine: each
  scaled frame is compared with a heavily smoothed copy of itself, positive
  excursions above ``star_nsigma`` robust sigmas are masked (dilated by
  ``star_dilate_px``), the masked pixels are excluded from the sigma-clipped
  combine, and pixels masked in too many frames (stars never repeat at one
  pixel: that is real structure, e.g. a hot pixel) are combined from all
  their frames instead;
* :func:`flat_star_qc` -- counts compact positive peaks left in a normalised
  master (3x3 mean of master / local median above a threshold) so a master
  with star residuals is detected and recorded in the header.

The same code path serves the in-memory and the on-disk (memmap) storage of
the corrected frames: both are presented as row-block ``loaders``.
"""

from __future__ import annotations

import logging
import math
import re
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from robo43.calib.combine import CombineResult, _combine_core, row_block_size
from robo43.calib.masks import _block_grid_medians, _block_interp_weights
from robo43.config import CalibSettings
from robo43.exceptions import CalibrationError
from robo43.io.meta import FrameMeta

logger = logging.getLogger(__name__)

__all__ = [
    "StarQC",
    "TwilightInfo",
    "TwilightParams",
    "central_region",
    "combine_twilight",
    "flat_star_qc",
    "frame_level",
    "is_twilight_flat",
    "method_cards",
    "scales_from_levels",
]

#: Fraction of the block-memory budget :func:`combine_twilight` uses for its
#: row blocks: the star detection holds extra float32 working arrays per frame
#: on top of the stack that :func:`robo43.calib.combine.row_block_size` sizes.
_BLOCK_MEMORY_FRACTION = 0.75

#: File names such as ``skyflat-20251224-001234.fits.fz``, ``sky_flat_01.fits``
#: or ``twilight-R-03.fits``.
_TWILIGHT_NAME_RE = re.compile(r"sky[\s_-]?flat|twilight")

#: Block rows read per step while building a frame's smooth reference grid.
_REF_CHUNK_BLOCKS = 32

#: Rows per chunk when the QC ratio image is built.
_QC_CHUNK_ROWS = 256


@dataclass(frozen=True, slots=True)
class TwilightParams:
    """Tunables of the twilight-flat combine and of the star-residual QC.

    ``star_nsigma``: detection threshold, in robust sigmas of the frame's
    residual from its smooth reference, for the pixel itself and for its 3x3
    mean.  ``star_block_px`` / ``star_smooth_blocks``: the reference is the
    ``star_block_px`` block median of the scaled frame, median-filtered over
    ``star_smooth_blocks`` x ``star_smooth_blocks`` blocks (80 px for the
    defaults) and bilinearly interpolated back.  ``star_dilate_px``: square
    dilation of the detections to cover the wings.  ``star_min_frames``: a
    pixel left with fewer unmasked frames than this (capped at ``n - 1``) is
    combined from all its frames instead; so is a pixel masked in more than
    ``star_persist_frac`` of the frames (see :func:`combine_twilight`).
    ``scale_all_flats``: also scale
    non-twilight (dome) flats to a common level before clipping.  ``qc_*``:
    the star-residual QC (window of the local median, threshold on the 3x3
    mean of master / local median, blob area range counted as a compact peak,
    warning density in peaks per Mpixel).  The 3x3 mean is used because a
    sensor's pixel-response scatter (0.9 % rms on T80S) puts ~25 000 single
    pixels per 85 Mpx above 1.03 in a perfectly clean master.
    """

    star_nsigma: float = 4.0
    star_block_px: int = 16
    star_smooth_blocks: int = 5
    star_dilate_px: int = 3
    star_min_frames: int = 3
    star_persist_frac: float = 0.5
    scale_all_flats: bool = False
    qc_window_px: int = 61
    qc_threshold: float = 1.03
    qc_warn_per_mpix: float = 30.0
    qc_min_area_px: int = 4
    qc_max_area_px: int = 400

    def __post_init__(self) -> None:
        problems = []
        if not self.star_nsigma > 0:
            problems.append("star_nsigma must be > 0")
        if self.star_block_px < 2:
            problems.append("star_block_px must be >= 2")
        if self.star_smooth_blocks < 1 or self.star_smooth_blocks % 2 == 0:
            problems.append("star_smooth_blocks must be a positive odd integer")
        if self.star_dilate_px < 0:
            problems.append("star_dilate_px must be >= 0")
        if self.star_min_frames < 1:
            problems.append("star_min_frames must be >= 1")
        if not 0.0 < self.star_persist_frac <= 1.0:
            problems.append("star_persist_frac must be in (0, 1]")
        if self.qc_window_px < 3:
            problems.append("qc_window_px must be >= 3")
        if not self.qc_threshold > 1.0:
            problems.append("qc_threshold must be > 1")
        if self.qc_warn_per_mpix < 0:
            problems.append("qc_warn_per_mpix must be >= 0")
        if self.qc_min_area_px < 1 or self.qc_max_area_px < self.qc_min_area_px:
            problems.append("qc_min_area_px must be >= 1 and <= qc_max_area_px")
        if problems:
            raise CalibrationError("invalid TwilightParams: " + "; ".join(problems))


def is_twilight_flat(meta: FrameMeta) -> bool:
    """True when this flat frame is a twilight (sky) flat.

    Any one of three signals suffices: the OBJECT value (lower-cased) contains
    "sky" or "twilight"; the raw IMAGETYP value recorded in
    ``meta.origins["imagetyp"]`` (``robo43.io.headers``) contains "sky" or
    "twilight" (T80S writes ``IMAGETYP = 'sky-flat'`` and no OBJECT); or the
    file name looks like ``skyflat-*`` / ``sky_flat*`` / ``twilight*``.
    """
    obj = (meta.object or "").lower()
    if "sky" in obj or "twilight" in obj:
        return True
    origin = meta.origins.get("imagetyp") if meta.origins else None
    if origin is not None:
        raw = str(origin.detail).lower()
        if "sky" in raw or "twilight" in raw:
            return True
    return bool(_TWILIGHT_NAME_RE.search(Path(meta.path).name.lower()))


def central_region(shape: tuple[int, int], frac: float) -> tuple[slice, slice]:
    """Centred region of ``frac`` of each axis; same arithmetic as ``normalise_flat``."""
    h, w = shape
    row_half = max(1, int(h * frac / 2))
    col_half = max(1, int(w * frac / 2))
    return (
        slice(h // 2 - row_half, h // 2 + row_half),
        slice(w // 2 - col_half, w // 2 + col_half),
    )


def frame_level(frame: np.ndarray, frac: float) -> float:
    """Median of the central ``frac`` region of a corrected flat frame."""
    return float(np.median(frame[central_region(frame.shape, frac)]))


def scales_from_levels(levels: Sequence[float]) -> tuple[float, np.ndarray]:
    """``(l_ref, scales)`` bringing every frame to the median frame level.

    ``scales[i] = l_ref / levels[i]`` with ``l_ref`` the median of ``levels``,
    so the scaled data stay in the frames' own units (ADU or electrons).
    Raises :class:`CalibrationError` when a level is not finite and positive.
    """
    arr = np.asarray(levels, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise CalibrationError(f"flat frame levels must be finite and > 0, got {list(levels)}")
    l_ref = float(np.median(arr))
    return l_ref, l_ref / arr


def method_cards(
    kind: str, *, scaled: bool, l_ref: float | None
) -> list[tuple[str, Any, str]]:
    """Header cards for a flat group combined *without* star masking."""
    cards: list[tuple[str, Any, str]] = [
        ("FLATKIND", kind, "Flat illumination type of this group"),
        (
            "FLATSCAL",
            "center_median" if scaled else "none",
            "Frames scaled to a common level before clipping",
        ),
    ]
    if scaled and l_ref is not None:
        cards.append(("FLATLREF", l_ref, "Common level the frames were scaled to"))
    cards.append(("STARMASK", False, "Per-frame star masking applied"))
    return cards


@dataclass(frozen=True, slots=True)
class TwilightInfo:
    """What :func:`combine_twilight` did, for the master's header."""

    l_ref: float
    scales: tuple[float, ...]
    params: TwilightParams
    block_px: int
    min_frames: int  # a pixel with fewer unmasked frames than this is combined unmasked
    masked_fraction: float  # of frame*pixel entries excluded as stars
    n_fallback_px: int  # pixels combined from all frames (too few unmasked)

    def header_cards(self) -> list[tuple[str, Any, str]]:
        p = self.params
        return [
            ("FLATKIND", "twilight", "Flat illumination type of this group"),
            ("FLATSCAL", "center_median", "Frames scaled to a common level before clipping"),
            ("FLATLREF", self.l_ref, "Common level the frames were scaled to"),
            ("STARMASK", True, "Per-frame star masking applied"),
            ("STARNSIG", p.star_nsigma, "Star detection threshold (robust sigma)"),
            ("STARBLK", self.block_px, "Smooth-reference block size (px)"),
            ("STARSMTH", p.star_smooth_blocks, "Smooth-reference median filter (blocks)"),
            ("STARDIL", p.star_dilate_px, "Star mask dilation (px)"),
            ("STARPERS", p.star_persist_frac, "Masked in more frames than this: not stars"),
            ("STARMINF", self.min_frames, "Min unmasked frames per pixel (else fallback)"),
            ("STARMFRC", self.masked_fraction, "Fraction of frame*pixel entries masked"),
            ("STARNFB", self.n_fallback_px, "Pixels combined unmasked (too few frames)"),
        ]


def _interp_axis(n: int, box: int, n_blocks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bilinear (clamped) interpolation indices/weights of block centres along one axis.

    Blocks are the ``n_blocks`` full ``box``-px blocks from pixel 0; pixels
    beyond the first/last centre take the edge block's value.
    """
    if n_blocks == 1:
        zeros = np.zeros(n, dtype=np.intp)
        return zeros, zeros, np.zeros(n, dtype=np.float32)
    centres = np.arange(n_blocks) * box + (box - 1) / 2.0
    pos = np.arange(n, dtype=np.float64)
    lo = np.clip(np.searchsorted(centres, pos, side="right") - 1, 0, n_blocks - 2)
    weight = np.clip((pos - centres[lo]) / box, 0.0, 1.0)
    return lo, lo + 1, weight.astype(np.float32)


def _reference_grid(
    loader: Callable[[tuple[slice, slice]], np.ndarray],
    shape: tuple[int, int],
    scale: float,
    *,
    box: int,
    smooth: int,
    fill_level: float,
) -> np.ndarray:
    """Heavily smoothed block-median grid ``(h // box, w // box)`` of one scaled frame.

    Read in strips of block rows so no full frame is held; non-finite pixels
    take ``fill_level``.  The block medians are median-filtered over
    ``smooth`` x ``smooth`` blocks (clipped to the grid) so stars, which fill
    only part of a block, do not enter the reference.
    """
    h, w = shape
    n_by, n_bx = h // box, w // box
    grid = np.empty((n_by, n_bx), dtype=np.float32)
    for b0 in range(0, n_by, _REF_CHUNK_BLOCKS):
        b1 = min(b0 + _REF_CHUNK_BLOCKS, n_by)
        raw = np.asarray(loader((slice(b0 * box, b1 * box), slice(0, n_bx * box))))
        strip = raw.astype(np.float32) * np.float32(scale)
        strip[~np.isfinite(strip)] = np.float32(fill_level)
        blocks = strip.reshape(b1 - b0, box, n_bx, box).transpose(0, 2, 1, 3)
        grid[b0:b1] = np.median(blocks.reshape(b1 - b0, n_bx, box * box), axis=-1)
    odd_limit = min(n_by - (1 - n_by % 2), n_bx - (1 - n_bx % 2))
    size = min(smooth, odd_limit)
    if size >= 3:
        grid = ndimage.median_filter(grid, size=size, mode="nearest")
    return grid


def _upsample(
    grid: np.ndarray,
    y: tuple[np.ndarray, np.ndarray, np.ndarray],
    x: tuple[np.ndarray, np.ndarray, np.ndarray],
    ra: int,
    rb: int,
) -> np.ndarray:
    """Bilinear upsample of ``grid`` to rows ``ra:rb`` and every column (float32)."""
    y_lo, y_hi, y_w = y
    x_lo, x_hi, x_w = x
    yw = y_w[ra:rb, None]
    band = grid[y_lo[ra:rb]] * (np.float32(1.0) - yw) + grid[y_hi[ra:rb]] * yw
    return band[:, x_lo] * (np.float32(1.0) - x_w) + band[:, x_hi] * x_w


def _robust_sigma(values: np.ndarray) -> tuple[float, float]:
    """``(median, 1.4826 * MAD)`` of a float array."""
    med = float(np.median(values))
    return med, 1.4826 * float(np.median(np.abs(values - med)))


def _detect_stars(
    frame: np.ndarray, reference: np.ndarray, core: slice, params: TwilightParams
) -> np.ndarray:
    """Boolean star mask (same shape as ``frame``) of one scaled frame block.

    The residual ``frame / reference - 1`` is thresholded at ``star_nsigma``
    robust sigmas (sigma and median taken over the ``core`` rows only, the
    haloed rows being context for the filters) -- once for the pixel and once
    for its 3x3 mean, which reaches faint wings -- positive side only, then
    dilated.  Non-finite pixels are always masked.
    """
    finite = np.isfinite(frame)
    good = finite & (reference > 0)
    resid = np.divide(frame, reference, out=np.ones_like(frame), where=good) - np.float32(1.0)
    smoothed = ndimage.uniform_filter(resid, size=3, mode="nearest")
    detected = ~finite
    for image in (resid, smoothed):
        med, sigma = _robust_sigma(image[core])
        if sigma > 0.0 and np.isfinite(sigma):
            detected |= image > med + params.star_nsigma * sigma
    dilate = params.star_dilate_px
    if dilate > 0:
        size = 2 * dilate + 1
        grown = ndimage.maximum_filter1d(detected.view(np.uint8), size, axis=0, mode="constant")
        grown = ndimage.maximum_filter1d(grown, size, axis=1, mode="constant")
        detected = grown.astype(bool)
    return detected


def combine_twilight(
    loaders: Sequence[Callable[[tuple[slice, slice]], np.ndarray]],
    shape: tuple[int, int],
    settings: CalibSettings,
    *,
    levels: Sequence[float],
    params: TwilightParams | None = None,
    method: str | None = None,
    read_noise_e: float | None = None,
    gain_e_per_adu: float | None = None,
    fill: float = 0.0,
) -> tuple[CombineResult, TwilightInfo]:
    """Scale, star-mask and sigma-clip-combine twilight flats, row block by row block.

    ``loaders`` and ``shape`` are as for :func:`robo43.calib.combine.combine_lazy`
    (any storage: in-memory arrays or a memmap); ``levels`` holds each frame's
    level (see :func:`frame_level`).  Every frame is multiplied by
    ``l_ref / level`` (``l_ref`` = median level) before anything else, so the
    sigma clip compares like with like and the master (and its variance) stay
    in the frames' units at level ``l_ref``.

    Per row block (with a halo of ``star_dilate_px + 2`` rows for the filters)
    each frame's stars are masked by :func:`_detect_stars` against that
    frame's smooth reference grid, and masked entries become NaN, which the
    sigma clip excludes.  A pixel left with fewer unmasked frames than
    ``min_frames`` (``star_min_frames`` capped at ``n - 1``, but at least the
    frames outside ``star_persist_frac`` of them) is combined from all its
    frames: stars do not repeat at a pixel, so a pixel masked in most frames
    is real flat structure (hot pixels, edges), and masking only the frames
    where noise lifts it over the threshold would bias it low.
    ``CombineResult.rejected_fraction`` counts clip rejections
    only; the star-masked fraction is on the returned :class:`TwilightInfo`.
    """
    params = params if params is not None else TwilightParams()
    loader_list = list(loaders)
    n = len(loader_list)
    if n < 2:
        raise CalibrationError(
            f"combine_twilight requires at least 2 frames (got {n}); a single "
            "frame cannot be sigma-clipped or star-masked"
        )
    method_ = method or settings.combine_method
    if method_ not in ("mean", "median"):
        raise CalibrationError(f"unknown combine method {method_!r} (expected 'mean' or 'median')")
    if len(levels) != n:
        raise CalibrationError(f"levels length {len(levels)} does not match n_frames {n}")

    l_ref, scales = scales_from_levels(levels)
    h, w = shape
    box = min(params.star_block_px, max(2, min(h, w) // 4))
    n_by, n_bx = h // box, w // box
    grids = [
        _reference_grid(
            loader, shape, float(scale), box=box, smooth=params.star_smooth_blocks, fill_level=l_ref
        )
        for loader, scale in zip(loader_list, scales, strict=True)
    ]
    y_interp = _interp_axis(h, box, n_by)
    x_interp = _interp_axis(w, box, n_bx)

    halo = params.star_dilate_px + 2
    min_frames = max(
        min(params.star_min_frames, max(1, n - 1)),
        math.ceil(n * (1.0 - params.star_persist_frac)),
    )
    rows_per_block = row_block_size(shape, n, settings.max_memory_mb * _BLOCK_MEMORY_FRACTION)

    data = np.empty(shape, dtype=np.float32)
    variance = np.empty(shape, dtype=np.float32)
    weight = np.empty(shape, dtype=np.float32)
    mask = np.empty(shape, dtype=bool)
    n_used = np.empty(shape, dtype=np.int16)

    clip_rejected = 0.0
    n_masked_total = 0
    n_fallback = 0
    for r0 in range(0, h, rows_per_block):
        r1 = min(r0 + rows_per_block, h)
        ra, rb = max(0, r0 - halo), min(h, r1 + halo)
        core = slice(r0 - ra, r0 - ra + (r1 - r0))
        stack = np.empty((n, r1 - r0, w), dtype=np.float64)
        star_cube = np.zeros((n, r1 - r0, w), dtype=bool)
        for i, loader in enumerate(loader_list):
            raw = np.asarray(loader((slice(ra, rb), slice(0, w))))
            scaled = raw.astype(np.float64) * float(scales[i])
            reference = _upsample(grids[i], y_interp, x_interp, ra, rb)
            star_cube[i] = _detect_stars(scaled.astype(np.float32), reference, core, params)[core]
            stack[i] = scaled[core]
        too_few = (n - star_cube.sum(axis=0)) < min_frames
        n_fallback += int(too_few.sum())
        if too_few.any():
            star_cube[:, too_few] = False
        n_masked = int(star_cube.sum())
        n_masked_total += n_masked
        stack[star_cube] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # astropy: NaNs "automatically clipped"
            block = _combine_core(
                stack,
                settings,
                method=method_,
                read_noise_e=read_noise_e,
                gain_e_per_adu=gain_e_per_adu,
                weights=None,
                fill=fill,
            )
        rows = slice(r0, r1)
        data[rows] = block.data
        variance[rows] = block.variance
        weight[rows] = block.weight
        mask[rows] = block.mask
        n_used[rows] = block.n_used
        clip_rejected += max(0.0, block.rejected_fraction * n * (r1 - r0) * w - n_masked)

    total = float(n * h * w)
    result = CombineResult(
        data=data,
        variance=variance,
        weight=weight,
        mask=mask,
        n_used=n_used,
        n_input=n,
        method=method_,
        rejected_fraction=clip_rejected / total,
        sigma_low=settings.sigma_clip_low,
        sigma_high=settings.sigma_clip_high,
        maxiters=settings.sigma_clip_maxiters,
    )
    info = TwilightInfo(
        l_ref=l_ref,
        scales=tuple(float(s) for s in scales),
        params=params,
        block_px=box,
        min_frames=min_frames,
        masked_fraction=n_masked_total / total,
        n_fallback_px=n_fallback,
    )
    logger.info(
        "twilight combine: %d frames scaled to %.1f, %.4f%% of entries star-masked, "
        "%d px fell back to unmasked frames",
        n,
        l_ref,
        100.0 * info.masked_fraction,
        n_fallback,
    )
    return result, info


@dataclass(frozen=True, slots=True)
class StarQC:
    """Compact positive peaks left in a normalised master flat."""

    n_peaks: int  # blobs with min_area <= area <= max_area
    n_extended: int  # larger blobs (not counted as peaks)
    max_ratio: float  # largest 3x3 mean of master / local median over the frame
    window_px: int
    threshold: float
    n_pixels: int  # frame size, for the density

    @property
    def per_mpix(self) -> float:
        """Compact peaks per million pixels."""
        return 1.0e6 * self.n_peaks / max(1, self.n_pixels)

    def header_cards(self) -> list[tuple[str, Any, str]]:
        return [
            ("FLATQCN", self.n_peaks, "Compact positive peaks above FLATQCT"),
            ("FLATQCL", self.n_extended, "Extended blobs above FLATQCT (not counted)"),
            ("FLATQCX", self.max_ratio, "Max 3x3 mean of master / local median"),
            ("FLATQCT", self.threshold, "QC threshold (3x3 mean / local median)"),
            ("FLATQCW", self.window_px, "QC local-median window (px)"),
        ]


def flat_star_qc(master: np.ndarray, params: TwilightParams | None = None) -> StarQC:
    """Count compact positive peaks of a normalised master flat.

    The local level is the block median of ``qc_window_px`` x ``qc_window_px``
    blocks, bilinearly interpolated between block centres.  The 3x3 mean of
    ``master / local`` is thresholded at ``qc_threshold`` and the pixels are
    grouped into 8-connected blobs; blobs of ``qc_min_area_px`` to
    ``qc_max_area_px`` pixels are the compact peaks (star residuals), larger
    ones are counted separately as extended and smaller ones are ignored
    (single-pixel response scatter).
    """
    params = params if params is not None else TwilightParams()
    if master.ndim != 2:
        raise CalibrationError(f"flat_star_qc: master must be 2-D, got shape {master.shape}")
    h, w = master.shape
    box = int(min(params.qc_window_px, max(h, w)))
    grid = _block_grid_medians(master, box)
    finite_blocks = np.isfinite(grid)
    if not finite_blocks.any():
        return StarQC(0, 0, float("nan"), box, params.qc_threshold, h * w)
    grid = np.where(finite_blocks, grid, np.median(grid[finite_blocks]))
    y_lo, y_hi, y_w = _block_interp_weights(h, box, grid.shape[0])
    x_lo, x_hi, x_w = _block_interp_weights(w, box, grid.shape[1])

    candidates = np.zeros((h, w), dtype=bool)
    max_ratio = float("-inf")
    for r0 in range(0, h, _QC_CHUNK_ROWS):
        r1 = min(r0 + _QC_CHUNK_ROWS, h)
        ra, rb = max(0, r0 - 1), min(h, r1 + 1)  # one row of context for the 3x3 mean
        yw = y_w[ra:rb, None]
        band = (1.0 - yw) * grid[y_lo[ra:rb]] + yw * grid[y_hi[ra:rb]]
        local = band[:, x_lo] * (1.0 - x_w) + band[:, x_hi] * x_w
        chunk = np.asarray(master[ra:rb], dtype=np.float64)
        valid = np.isfinite(chunk) & (local > 0)
        ratio = np.ones(chunk.shape)
        np.divide(chunk, local, out=ratio, where=valid)
        smooth = ndimage.uniform_filter(ratio, size=3, mode="nearest")
        smooth = smooth[r0 - ra : r0 - ra + (r1 - r0)]
        max_ratio = max(max_ratio, float(smooth.max()))
        candidates[r0:r1] = smooth > params.qc_threshold
    labels, _n_labels = ndimage.label(candidates, structure=np.ones((3, 3), dtype=bool))
    n_peaks = n_extended = 0
    for label_idx, region in enumerate(ndimage.find_objects(labels), start=1):
        if region is None:
            continue
        area = int(np.count_nonzero(labels[region] == label_idx))
        if area > params.qc_max_area_px:
            n_extended += 1
        elif area >= params.qc_min_area_px:
            n_peaks += 1
    return StarQC(n_peaks, n_extended, max_ratio, box, params.qc_threshold, h * w)
