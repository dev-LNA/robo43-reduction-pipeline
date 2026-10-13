"""Twilight (sky) flat handling: classification, level scaling, star masking, QC.

Twilight flats are taken with the telescope moved between exposures, so stars
do not repeat at one pixel (or only within a small dither) -- but the frame
levels differ strongly (the sky fades or brightens through the sequence).
Combining them with a per-pixel sigma clip on the *unscaled* frames fails: the
across-frame scatter is dominated by the level differences, 3-sigma clipping
never reaches a star, and every star survives in the master as a compact
positive residual.

This module supplies the pieces :func:`robo43.calib.flat.build_master_flats`
uses for a twilight group:

* :func:`is_twilight_flat` -- twilight/sky classification of one flat from the
  raw IMAGETYP, the OBJECT value or the file name (word-token matching, in
  that order of priority);
* :func:`scales_from_levels` / :func:`frame_level` -- per-frame multiplicative
  scales that bring every frame to a common level (the median of the frame
  levels), applied *before* clipping;
* :func:`combine_twilight` -- the memory-bounded (row-block), two-pass combine
  (see below);
* :func:`flat_star_qc` -- counts compact positive peaks left in a normalised
  master (3x3 mean of master / local median above a threshold) so a master
  with star residuals is detected and recorded in the header.

Two-pass star masking (:func:`combine_twilight`).  Pass 1 compares every scaled
frame with a heavily smoothed copy of itself, masks the positive excursions
(dilated) and builds a preliminary master from the unmasked entries.  A pixel
masked in (nearly) all frames is either static structure (hot pixel, edge: its
residual is the same in every frame) or a star that sits there in every frame
because the dither is small (its residual changes from frame to frame); the
across-frame scatter of the residual tells them apart.  Pass 2 divides each
frame by the preliminary master -- static structure divides out, stars do not
-- detects the positive residual against the smooth reference of *that ratio*,
masks it (dilated) and sigma-clip-combines the unmasked entries.  A pixel with
a star in every frame (nothing left to combine) takes the preliminary master's
smooth value, flagged with zero weight, ``n_used`` 0 and a set mask.

The same code path serves the in-memory and the on-disk (memmap) storage of
the corrected frames: both are presented as row-block ``loaders``.
"""

from __future__ import annotations

import logging
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
from robo43.instruments.profile import InstrumentProfile
from robo43.io.headers import read_raw_imagetyp
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
    "raw_imagetyp",
    "scales_from_levels",
]

#: Fraction of the block-memory budget :func:`combine_twilight` uses for its
#: row blocks: the star detection holds extra float32 working arrays per frame
#: on top of the stack that :func:`robo43.calib.combine.row_block_size` sizes.
_BLOCK_MEMORY_FRACTION = 0.75

#: Whole words (case-folded) that mark a twilight flat in IMAGETYP, OBJECT or a
#: file name; a word ending in "flat"/"flats" whose stem is one of
#: ``_TWILIGHT_STEMS`` (``twilightflat``) counts too.
_TWILIGHT_WORDS = frozenset({"sky", "twilight", "twi", "dusk", "dawn", "skyflat", "twiflat"})
_TWILIGHT_STEMS = frozenset({"sky", "twilight", "twi", "dusk", "dawn"})

_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
_LETTER_DIGIT_RE = re.compile(r"(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])")
_NON_ALNUM_RE = re.compile(r"[^0-9A-Za-z]+")

#: ``(path, mtime_ns, size, profile name)`` -> raw IMAGETYP read from the header.
_IMAGETYP_CACHE: dict[tuple[str, int, int, str], str | None] = {}
_IMAGETYP_CACHE_MAX = 8192

#: A pixel left with too few unmasked frames is "star-like" (not static
#: structure) when at least this many pixels within ``2 * star_dilate_px + 2``
#: of it (a square window) have an across-frame residual variance above the
#: static tolerance: a star puts dozens of varying pixels in its neighbourhood,
#: noise puts almost none (see :func:`_mask_block`).
_STATIC_MIN_VARYING = 8

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
    ``star_block_px`` block median of the scaled frame (pass 1) or of the
    frame / preliminary master ratio (pass 2), median-filtered over
    ``star_smooth_blocks`` x ``star_smooth_blocks`` blocks (80 px for the
    defaults) and bilinearly interpolated back.  ``star_dilate_px``: square
    dilation of the detections to cover the wings.  ``star_min_frames``: a
    pixel left with fewer unmasked frames than this (capped at ``n - 1``) is
    tested for static structure.  ``star_static_tol``: a pixel whose
    across-frame residual variance exceeds ``star_static_tol`` ** 2 times the
    block's typical (median) pixel variance is "varying" (a star of changing
    strength sits on it); such a pixel with fewer unmasked frames than
    ``star_min_frames`` that has at least 8 varying pixels within
    ``2 * star_dilate_px + 2`` px is star-like, any other one is static
    structure (hot pixel, edge) and is combined from all its frames.
    ``scale_all_flats``: also scale non-twilight
    (dome) flats to a common level before clipping.  ``qc_*``: the
    star-residual QC (window of the local median, threshold on the 3x3 mean of
    master / local median, blob area range counted as a compact peak, warning
    density in peaks per Mpixel).  The 3x3 mean is used because a sensor's
    pixel-response scatter (0.9 % rms on T80S) puts ~25 000 single pixels per
    85 Mpx above 1.03 in a perfectly clean master.
    """

    star_nsigma: float = 4.0
    star_block_px: int = 16
    star_smooth_blocks: int = 5
    star_dilate_px: int = 3
    star_min_frames: int = 3
    star_static_tol: float = 3.0
    scale_all_flats: bool = False
    qc_window_px: int = 61
    qc_threshold: float = 1.03
    qc_warn_per_mpix: float = 2.0
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
        if not self.star_static_tol > 0:
            problems.append("star_static_tol must be > 0")
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


def _word_tokens(text: str) -> list[str]:
    """Lower-case words of ``text``: split on non-alphanumerics, camelCase and letter/digit."""
    spaced = _LETTER_DIGIT_RE.sub(" ", _CAMEL_RE.sub(" ", text))
    return [token.lower() for token in _NON_ALNUM_RE.split(spaced) if token]


def _is_twilight_word(token: str) -> bool:
    candidates = [token]
    if len(token) > 3 and token.endswith("s"):
        candidates.append(token[:-1])  # plural: skyflats, twilights
    for cand in candidates:
        if cand in _TWILIGHT_WORDS:
            return True
        if cand.endswith("flat") and cand[: -len("flat")] in _TWILIGHT_STEMS:
            return True
    return False


def _text_verdict(text: str | None) -> bool | None:
    """``False`` for a "dome" word in ``text``, ``True`` for a twilight word, else ``None``.

    A dome word wins over a twilight word in the same text ("dome_sky_flat",
    "Dome flat (no sky)").
    """
    if not text:
        return None
    tokens = _word_tokens(str(text))
    if any("dome" in token for token in tokens):
        return False
    if any(_is_twilight_word(token) for token in tokens):
        return True
    return None


def raw_imagetyp(meta: FrameMeta, profile: InstrumentProfile | None = None) -> str | None:
    """Raw IMAGETYP string of a frame.

    Taken from ``meta.origins["imagetyp"]`` (recorded by
    :func:`robo43.io.headers.read_meta`); when an inventory predating that
    origin has none, the FITS header is read again
    (:func:`robo43.io.headers.read_raw_imagetyp`, cached per file).  ``None``
    when neither yields a value (including an unreadable or missing file).
    """
    origin = meta.origins.get("imagetyp") if meta.origins else None
    if origin is not None and str(origin.detail).strip():
        return str(origin.detail)
    path = Path(meta.path)
    try:
        stat = path.stat()
    except OSError:
        return None
    profile_name = profile.name if profile is not None else ""
    key = (str(path), stat.st_mtime_ns, stat.st_size, profile_name)
    if key not in _IMAGETYP_CACHE:
        if len(_IMAGETYP_CACHE) >= _IMAGETYP_CACHE_MAX:
            _IMAGETYP_CACHE.clear()
        _IMAGETYP_CACHE[key] = read_raw_imagetyp(path, profile)
    return _IMAGETYP_CACHE[key]


def is_twilight_flat(meta: FrameMeta, profile: InstrumentProfile | None = None) -> bool:
    """True when this flat frame is a twilight (sky) flat.

    The raw IMAGETYP (T80S writes ``IMAGETYP = 'sky-flat'`` and no OBJECT), the
    OBJECT value and the file name are examined in that order of priority; the
    first one that carries a signal decides.  Each is split into words (on
    non-alphanumerics, camelCase and letter/digit boundaries); a word in
    {sky, twilight, twi, dusk, dawn, skyflat, twiflat} is a twilight signal,
    any word containing "dome" is a dome signal, and a dome signal wins over a
    twilight one in the same source ("dome_sky_flat_01.fits", "Husky Dome" and
    "Dome flat (no sky)" are not twilight flats; "TwiFlat", "FLAT-DUSK" and
    "dawn flat" are).  A frame with no signal anywhere is not twilight.
    """
    for text in (raw_imagetyp(meta, profile), meta.object, Path(meta.path).name):
        verdict = _text_verdict(text)
        if verdict is not None:
            return verdict
    return False


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
    """NaN-ignoring median of the central ``frac`` region of a corrected flat frame.

    ``nan`` when the region has no finite value (the caller drops such a frame).
    """
    region = frame[central_region(frame.shape, frac)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # "All-NaN slice encountered"
        return float(np.nanmedian(region))


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
    min_frames: int  # a pixel with fewer unmasked frames than this is tested for static structure
    masked_fraction: float  # of frame*pixel entries excluded as stars (final pass)
    n_static_px: int  # pass 1: masked in most frames but static structure -> combined unmasked
    n_filled_px: int  # pass 2: a star in every frame -> preliminary master's smooth value

    def header_cards(self) -> list[tuple[str, Any, str]]:
        p = self.params
        return [
            ("FLATKIND", "twilight", "Flat illumination type of this group"),
            ("FLATSCAL", "center_median", "Frames scaled to a common level before clipping"),
            ("FLATLREF", self.l_ref, "Common level the frames were scaled to"),
            ("STARMASK", True, "Per-frame star masking applied"),
            ("STARPASS", 2, "Star-mask passes (2nd on frame/prelim master)"),
            ("STARNSIG", p.star_nsigma, "Star detection threshold (robust sigma)"),
            ("STARBLK", self.block_px, "Smooth-reference block size (px)"),
            ("STARSMTH", p.star_smooth_blocks, "Smooth-reference median filter (blocks)"),
            ("STARDIL", p.star_dilate_px, "Star mask dilation (px)"),
            ("STARSTAT", p.star_static_tol, "Static-structure scatter tolerance (x noise)"),
            ("STARMINF", self.min_frames, "Min unmasked frames before static test"),
            ("STARMFRC", self.masked_fraction, "Fraction of frame*pixel entries masked"),
            ("STARNFB", self.n_static_px, "Static px masked in most frames, kept"),
            ("STARNFL", self.n_filled_px, "Px with a star in every frame, filled"),
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
    divisor: np.ndarray | None = None,
) -> np.ndarray:
    """Heavily smoothed block-median grid ``(h // box, w // box)`` of one scaled frame.

    Read in strips of block rows so no full frame is held; non-finite pixels
    take ``fill_level``.  With ``divisor`` (the preliminary master) the grid is
    that of ``scaled frame / divisor`` instead, a pixel where either is not
    finite or the divisor is not positive taking ``fill_level``.  The block
    medians are median-filtered over ``smooth`` x ``smooth`` blocks (clipped
    to the grid) so stars, which fill only part of a block, do not enter the
    reference.
    """
    h, w = shape
    n_by, n_bx = h // box, w // box
    grid = np.empty((n_by, n_bx), dtype=np.float32)
    for b0 in range(0, n_by, _REF_CHUNK_BLOCKS):
        b1 = min(b0 + _REF_CHUNK_BLOCKS, n_by)
        raw = np.asarray(loader((slice(b0 * box, b1 * box), slice(0, n_bx * box))))
        strip = raw.astype(np.float32) * np.float32(scale)
        if divisor is None:
            strip[~np.isfinite(strip)] = np.float32(fill_level)
        else:
            denom = np.asarray(divisor[b0 * box : b1 * box, : n_bx * box], dtype=np.float32)
            usable = np.isfinite(strip) & np.isfinite(denom) & (denom > 0)
            ratio = np.full(strip.shape, np.float32(fill_level), dtype=np.float32)
            np.divide(strip, denom, out=ratio, where=usable)
            strip = ratio
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
) -> tuple[np.ndarray, np.ndarray]:
    """``(star mask, residual)`` of one scaled frame block, both of the frame's shape.

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
    return detected, resid


def _median_variance_factor(dof: int) -> float:
    """Median of a chi-square variable with ``dof`` degrees of freedom, divided by ``dof``
    (Wilson-Hilferty approximation)."""
    return (1.0 - 2.0 / (9.0 * dof)) ** 3


def _nan_median_axis0(stack: np.ndarray) -> np.ndarray:
    """Median over axis 0 ignoring non-finite entries (float64); NaN where none is finite.

    Sorts a float32 copy of the stack (half the memory of the float64 stack).
    """
    n = stack.shape[0]
    srt = stack.astype(np.float32)
    usable = np.isfinite(srt)
    srt[~usable] = np.inf
    srt.sort(axis=0)
    count = usable.sum(axis=0)
    lo = np.clip((count - 1) // 2, 0, n - 1)[None]
    hi = np.clip(count // 2, 0, n - 1)[None]
    low = np.take_along_axis(srt, lo, axis=0)[0].astype(np.float64)
    high = np.take_along_axis(srt, hi, axis=0)[0].astype(np.float64)
    med = 0.5 * (low + high)
    med[count == 0] = np.nan
    return med


@dataclass(slots=True)
class _BlockMasks:
    """Result of :func:`_mask_block` for one row block (core rows only)."""

    stack: np.ndarray  # (n, rows, w) float64, scaled frames
    star: np.ndarray  # (n, rows, w) bool, entries to exclude
    static: np.ndarray  # (rows, w) bool, masked in most frames but static: star mask lifted
    empty: np.ndarray  # (rows, w) bool, no frame left to combine
    ref_mean: np.ndarray  # (rows, w) float64, mean over frames of the reference


def _mask_block(
    loaders: Sequence[Callable[[tuple[slice, slice]], np.ndarray]],
    scales: np.ndarray,
    grids: Sequence[np.ndarray],
    y_interp: tuple[np.ndarray, np.ndarray, np.ndarray],
    x_interp: tuple[np.ndarray, np.ndarray, np.ndarray],
    rows: tuple[int, int, int, int],
    width: int,
    prelim: np.ndarray | None,
    params: TwilightParams,
    min_frames: int,
) -> _BlockMasks:
    """Scale, read and star-mask one row block ``r0:r1`` (haloed rows ``ra:rb``).

    ``rows = (ra, rb, r0, r1)``; the halo carries the filters' context and the
    neighbourhood of the static test.  Each frame is compared with its
    reference: the smooth grid, upsampled (pass 1, ``prelim`` None), or the
    smooth grid of the frame / preliminary-master ratio times the preliminary
    master (pass 2).

    A pixel left with fewer than ``min_frames`` unmasked frames is either a
    star sitting on it in every frame (small dither) or static structure.  The
    residual of static structure is the same in every frame, so its
    across-frame variance is only noise; that of a star changes with the
    star's position.  The variance is compared with the block's median pixel
    variance (``star_static_tol`` ** 2, "varying"): the pixel is star-like
    when at least ``_STATIC_MIN_VARYING`` varying pixels lie within
    ``2 * star_dilate_px + 2`` px of it (a star's faint wings are not
    individually distinguishable from noise, its core is), otherwise static,
    and its star mask is lifted.  A star-like pixel with no unmasked frame is
    ``empty``, as is one with no finite value at all.
    """
    ra, rb, r0, r1 = rows
    n = len(loaders)
    nr = r1 - r0
    nh = rb - ra
    core = slice(r0 - ra, r1 - ra)
    stack = np.empty((n, nr, width), dtype=np.float64)
    star = np.zeros((n, nh, width), dtype=bool)
    sum_r = np.zeros((nh, width), dtype=np.float64)
    sum_r2 = np.zeros((nh, width), dtype=np.float64)
    ref_sum = np.zeros((nh, width), dtype=np.float64)
    for i, loader in enumerate(loaders):
        raw = np.asarray(loader((slice(ra, rb), slice(0, width))))
        scaled = raw.astype(np.float64) * float(scales[i])
        reference = _upsample(grids[i], y_interp, x_interp, ra, rb)
        if prelim is not None:
            reference = reference * prelim[ra:rb]
        detected, resid = _detect_stars(scaled.astype(np.float32), reference, core, params)
        star[i] = detected
        stack[i] = scaled[core]
        r = resid.astype(np.float64)
        sum_r += r
        sum_r2 += r * r
        ref_sum += reference

    n_clean = n - star.sum(axis=0)
    few = n_clean < min_frames
    static = np.zeros((nr, width), dtype=bool)
    empty = np.zeros((nr, width), dtype=bool)
    if few[core].any():
        var = np.maximum((sum_r2 - sum_r * sum_r / n) / (n - 1), 0.0)
        noise_var = float(np.median(var[core])) / _median_variance_factor(n - 1)
        varying = few & (var > params.star_static_tol**2 * noise_var)
        window = 2 * (2 * params.star_dilate_px + 2) + 1
        n_varying = ndimage.uniform_filter(varying.astype(np.float32), size=window, mode="constant")
        starlike = few & (n_varying * np.float32(window * window) >= _STATIC_MIN_VARYING - 0.5)
        static_halo = few & ~starlike
        star[:, static_halo] = False
        static = static_halo[core]
        no_data = ~np.isfinite(stack).any(axis=0)
        empty = (starlike[core] & (n_clean[core] == 0)) | (few[core] & no_data)
    return _BlockMasks(stack, star[:, core], static, empty, ref_sum[core] / n)


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
    """Scale, star-mask (two passes) and sigma-clip-combine twilight flats, row block by row block.

    ``loaders`` and ``shape`` are as for :func:`robo43.calib.combine.combine_lazy`
    (any storage: in-memory arrays or a memmap); ``levels`` holds each frame's
    level (see :func:`frame_level`).  Every frame is multiplied by
    ``l_ref / level`` (``l_ref`` = median level) before anything else, so the
    sigma clip compares like with like and the master (and its variance) stay
    in the frames' units at level ``l_ref``.

    Both passes run per row block (with a halo of ``3 * star_dilate_px + 4``
    rows for the filters and the static test), see :func:`_mask_block`.  Pass 1 masks each frame's
    stars against its smooth reference grid and stores a preliminary master
    (float32, one frame): the median of the unmasked entries, the median of
    all frames at a static pixel, the mean smooth reference at a pixel with a
    star in every frame.  Pass 2 builds each frame's reference grid on
    ``frame / preliminary master`` -- static structure (hot pixels, edges)
    divides out, so it is no longer mistaken for a star, and a star that
    repeats within a few pixels from frame to frame (small dither) stays
    visible -- masks the stars again, sets masked entries to NaN and
    sigma-clips them out.  A pixel with a star in every frame has nothing
    left to combine: it takes the preliminary master's value with zero
    weight, ``n_used`` 0 and the mask set.  Where a pixel ends with a single
    contributing frame in a group of three or more, its weight is zeroed and
    its mask set (the variance of one sample is unknown).
    ``CombineResult.rejected_fraction`` counts clip rejections only; the
    star-masked fraction is on the returned :class:`TwilightInfo`.

    A group must hold at least 2 frames; ``min_frames`` is
    ``star_min_frames`` capped at ``n - 1``.  Cost: about twice the
    single-pass time (every frame is read and filtered twice more in pass 2)
    and one extra float32 frame of memory; the row-block stack is unchanged.
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
    y_interp = _interp_axis(h, box, n_by)
    x_interp = _interp_axis(w, box, n_bx)

    halo = (params.star_dilate_px + 2) + (2 * params.star_dilate_px + 2)  # filters + static test
    min_frames = min(params.star_min_frames, max(1, n - 1))
    rows_per_block = row_block_size(shape, n, settings.max_memory_mb * _BLOCK_MEMORY_FRACTION)
    blocks = [
        (max(0, r0 - halo), min(h, r0 + rows_per_block + halo), r0, min(r0 + rows_per_block, h))
        for r0 in range(0, h, rows_per_block)
    ]

    # Pass 1: each frame against its own smooth reference -> preliminary master.
    grids = [
        _reference_grid(
            loader, shape, float(scale), box=box, smooth=params.star_smooth_blocks, fill_level=l_ref
        )
        for loader, scale in zip(loader_list, scales, strict=True)
    ]
    prelim = np.empty(shape, dtype=np.float32)
    n_static = 0
    for rows in blocks:
        bm = _mask_block(
            loader_list, scales, grids, y_interp, x_interp, rows, w, None, params, min_frames
        )
        n_static += int(bm.static.sum())
        bm.stack[bm.star] = np.nan
        med = _nan_median_axis0(bm.stack)
        nothing = ~np.isfinite(med)
        med[nothing] = bm.ref_mean[nothing]
        prelim[rows[2] : rows[3]] = med
        del bm

    # Pass 2: each frame against its smooth reference of frame / preliminary master.
    grids = [
        _reference_grid(
            loader,
            shape,
            float(scale),
            box=box,
            smooth=params.star_smooth_blocks,
            fill_level=1.0,
            divisor=prelim,
        )
        for loader, scale in zip(loader_list, scales, strict=True)
    ]
    data = np.empty(shape, dtype=np.float32)
    variance = np.empty(shape, dtype=np.float32)
    weight = np.empty(shape, dtype=np.float32)
    mask = np.empty(shape, dtype=bool)
    n_used = np.empty(shape, dtype=np.int16)

    clip_rejected = 0.0
    n_masked_total = 0
    n_filled = 0
    n_static_final = 0
    for rows in blocks:
        r0, r1 = rows[2], rows[3]
        bm = _mask_block(
            loader_list, scales, grids, y_interp, x_interp, rows, w, prelim, params, min_frames
        )
        n_static_final += int(bm.static.sum())
        n_filled += int(bm.empty.sum())
        n_masked = int(bm.star.sum())
        n_masked_total += n_masked
        bm.stack[bm.star] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # astropy: NaNs "automatically clipped"
            block = _combine_core(
                bm.stack,
                settings,
                method=method_,
                read_noise_e=read_noise_e,
                gain_e_per_adu=gain_e_per_adu,
                weights=None,
                fill=fill,
            )
        block_data = block.data
        block_weight = block.weight
        block_mask = block.mask
        if bm.empty.any():
            block_data[bm.empty] = prelim[r0:r1][bm.empty]
            block_weight[bm.empty] = 0.0
            block_mask[bm.empty] = True
        if n >= 3:
            single = block.n_used == 1
            if single.any():
                block_weight[single] = 0.0
                block_mask[single] = True
        sel = slice(r0, r1)
        data[sel] = block_data
        variance[sel] = block.variance
        weight[sel] = block_weight
        mask[sel] = block_mask
        n_used[sel] = block.n_used
        clip_rejected += max(0.0, block.rejected_fraction * n * (r1 - r0) * w - n_masked)
        del bm

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
        n_static_px=n_static,
        n_filled_px=n_filled,
    )
    logger.info(
        "twilight combine: %d frames scaled to %.1f, %.4f%% of entries star-masked, "
        "%d static px kept (pass 1; %d in pass 2), %d px with a star in every frame filled",
        n,
        l_ref,
        100.0 * info.masked_fraction,
        n_static,
        n_static_final,
        n_filled,
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
