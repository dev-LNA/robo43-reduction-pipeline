"""Overscan detection and correction from FLAT frames.

Overscan geometry is detected at runtime from FLAT frames plus BIAS frames,
never hardcoded, because the overscan region varies with observation mode
and instrument+telescope combination.

Detection rule: a whole row or column in a FLAT frame that reads at bias
level cannot be illuminated sensor area, hence must be overscan. This module
detects such lines, checks them across multiple flats (intersection), and
trims them from every image after subtracting their median level.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from robo43.calib.bias import select_frames
from robo43.config import Settings
from robo43.exceptions import CalibrationError
from robo43.io.fits_io import parse_region, read_image

if TYPE_CHECKING:
    from robo43.instruments.profile import InstrumentProfile
    from robo43.io.meta import FrameMeta

logger = logging.getLogger(__name__)

__all__ = ["OverscanGeometry", "apply_overscan", "detect_overscan"]


@dataclass(frozen=True, slots=True)
class OverscanGeometry:
    """Detected or profiled overscan region: columns and rows to trim."""

    shape: tuple[int, int]              # (ny, nx) of the raw frame
    columns: tuple[int, ...]            # 0-based column indices, sorted
    rows: tuple[int, ...]               # 0-based row indices, sorted
    bias_level: float
    flat_level: float
    detected: bool                      # True when columns or rows is non-empty
    source: str                         # "detected", "profile", or "none"

    @property
    def science_slice(self) -> tuple[slice, slice]:
        """Return (row_slice, col_slice) for the science region.

        The science region is the raw frame with all overscan rows and columns
        removed. Because overscan is always whole lines, compute the retained
        index ranges as the complement of the detected indices, and return
        slices covering the largest contiguous retained block on each axis.
        Logs a WARNING if the retained region on either axis is not contiguous.
        """
        ny, nx = self.shape
        all_cols = set(range(nx))
        all_rows = set(range(ny))

        retained_cols = sorted(all_cols - set(self.columns))
        retained_rows = sorted(all_rows - set(self.rows))

        def _largest_contiguous_block(indices: list[int]) -> tuple[int, int]:
            """Return (start, end+1) of the largest contiguous block."""
            if not indices:
                return 0, 0

            blocks = []
            block_start = indices[0]
            block_end = indices[0] + 1

            for idx in indices[1:]:
                if idx == block_end:
                    block_end += 1
                else:
                    blocks.append((block_start, block_end))
                    block_start = idx
                    block_end = idx + 1

            blocks.append((block_start, block_end))

            if len(blocks) > 1:
                logger.warning(
                    f"Retained region on an axis is not contiguous; "
                    f"using largest block. Blocks: {blocks}"
                )

            largest = max(blocks, key=lambda b: b[1] - b[0]) if blocks else (0, 0)
            return largest

        col_start, col_end = _largest_contiguous_block(retained_cols)
        row_start, row_end = _largest_contiguous_block(retained_rows)

        return slice(row_start, row_end), slice(col_start, col_end)

    def header_cards(self) -> list[tuple[str, Any, str]]:
        """Return FITS header cards describing the geometry."""
        cards: list[tuple[str, Any, str]] = []
        cards.append(("OSCANNCO", len(self.columns), "Overscan columns detected"))
        cards.append(("OSCANNRO", len(self.rows), "Overscan rows detected"))
        cards.append(("OSCANSRC", self.source, "How overscan geometry was obtained"))
        return cards


def detect_overscan(
    metas: Sequence[FrameMeta],
    *,
    settings: Settings,
    profile: InstrumentProfile,
    max_flats: int = 3,
    max_biases: int = 2,
) -> OverscanGeometry:
    """Detect overscan geometry from FLAT and BIAS frames.

    Returns:
        An OverscanGeometry with detected=True and non-empty columns/rows if
        overscan was found, or detected=False if no overscan was detected.
        The source field indicates "detected", "profile", or "none".

    Never raises; always returns a valid geometry (possibly empty).
    """
    bias_metas = select_frames(metas, "bias")[:max_biases]
    flat_metas = select_frames(metas, "flat")[:max_flats]

    if not settings.calib.overscan_detect:
        if profile.layout.overscan is not None:
            logger.info(
                "Runtime overscan detection disabled; using profile region."
            )
            row_slice, col_slice = parse_region(profile.layout.overscan)
            cols = tuple(range(col_slice.start, col_slice.stop))
            rows = tuple(range(row_slice.start, row_slice.stop))
            return OverscanGeometry(
                shape=(0, 0),
                columns=cols,
                rows=rows,
                bias_level=np.nan,
                flat_level=np.nan,
                detected=True,
                source="profile",
            )
        else:
            logger.info(
                "Runtime overscan detection disabled; no profile region set."
            )
            return OverscanGeometry(
                shape=(0, 0),
                columns=(),
                rows=(),
                bias_level=np.nan,
                flat_level=np.nan,
                detected=False,
                source="none",
            )

    if not bias_metas or not flat_metas:
        logger.warning(
            "Overscan detection needs both bias and flat frames; "
            f"found {len(bias_metas)} bias(es) and {len(flat_metas)} flat(s)."
        )
        return OverscanGeometry(
            shape=(0, 0),
            columns=(),
            rows=(),
            bias_level=np.nan,
            flat_level=np.nan,
            detected=False,
            source="none",
        )

    bias_data_list = []
    for meta in bias_metas:
        data, _header = read_image(meta.path, profile)
        bias_data_list.append(data)

    bias_medians = [np.median(data) for data in bias_data_list]
    bias_level = float(np.median(bias_medians))

    flat_levels = []
    detected_cols_per_flat = []
    detected_rows_per_flat = []

    for meta in flat_metas:
        data, _header = read_image(meta.path, profile)
        ny, nx = data.shape
        flat_level_frame = float(np.median(data))
        flat_levels.append(flat_level_frame)

        span = flat_level_frame - bias_level
        if span <= 0:
            logger.warning(
                f"Flat frame {meta.path}: level {flat_level_frame:.1f} "
                f"<= bias level {bias_level:.1f}; skipping for detection."
            )
            detected_cols_per_flat.append(set())
            detected_rows_per_flat.append(set())
            continue

        threshold = bias_level + settings.calib.overscan_bias_tol_frac * span

        bias_cols = set()
        for col_idx in range(nx):
            col_median = float(np.median(data[:, col_idx]))
            if col_median <= threshold:
                bias_cols.add(col_idx)

        bias_rows = set()
        for row_idx in range(ny):
            row_median = float(np.median(data[row_idx, :]))
            if row_median <= threshold:
                bias_rows.add(row_idx)

        detected_cols_per_flat.append(bias_cols)
        detected_rows_per_flat.append(bias_rows)

    if not detected_cols_per_flat or not detected_rows_per_flat:
        return OverscanGeometry(
            shape=(ny, nx),
            columns=(),
            rows=(),
            bias_level=bias_level,
            flat_level=float(np.median(flat_levels)) if flat_levels else np.nan,
            detected=False,
            source="none",
        )

    intersect_cols = set.intersection(*detected_cols_per_flat)
    intersect_rows = set.intersection(*detected_rows_per_flat)

    def _drop_short_runs(indices: set[int], axis_len: int) -> set[int]:
        """Drop runs of consecutive indices shorter than overscan_min_line_frac."""
        if not indices:
            return set()

        sorted_idx = sorted(indices)
        min_run_len = int(settings.calib.overscan_min_line_frac * axis_len)
        if min_run_len < 1:
            min_run_len = 1

        runs = []
        run_start = sorted_idx[0]
        run_end = sorted_idx[0] + 1

        for idx in sorted_idx[1:]:
            if idx == run_end:
                run_end += 1
            else:
                runs.append((run_start, run_end))
                run_start = idx
                run_end = idx + 1

        runs.append((run_start, run_end))

        retained = set()
        for run_start, run_end in runs:
            if run_end - run_start >= min_run_len:
                retained.update(range(run_start, run_end))

        return retained

    final_cols = _drop_short_runs(intersect_cols, nx)
    final_rows = _drop_short_runs(intersect_rows, ny)

    logger.info(
        f"Detected {len(final_cols)} overscan columns and {len(final_rows)} "
        f"overscan rows from {len(flat_metas)} flat frame(s)."
    )

    geom = OverscanGeometry(
        shape=(ny, nx),
        columns=tuple(sorted(final_cols)),
        rows=tuple(sorted(final_rows)),
        bias_level=bias_level,
        flat_level=float(np.median(flat_levels)) if flat_levels else np.nan,
        detected=len(final_cols) > 0 or len(final_rows) > 0,
        source="detected",
    )

    if geom.detected:
        row_slice, col_slice = geom.science_slice
        logger.info(
            f"Science region after trim: rows [{row_slice.start}:{row_slice.stop}], "
            f"cols [{col_slice.start}:{col_slice.stop}]"
        )

    return geom


def apply_overscan(
    data: np.ndarray,
    geometry: OverscanGeometry,
    *,
    settings: Settings,
) -> tuple[np.ndarray, list[tuple[str, Any, str]]]:
    """Subtract overscan level, trim, and return the result with header cards.

    If overscan correction is disabled or no overscan was detected, returns
    the data unchanged and records OSCANCOR=False, TRIMMED=False.

    Returns:
        (corrected_data, header_cards): the data (still copy-safe, never mutated
        in-place) and the list of (keyword, value, comment) tuples for FITS.

    Raises:
        CalibrationError: if data.shape does not match geometry.shape.
    """
    cards: list[tuple[str, Any, str]] = []
    result = np.array(data, copy=True)

    if result.shape != geometry.shape:
        msg = (
            f"Data shape {result.shape} does not match geometry shape "
            f"{geometry.shape}"
        )
        raise CalibrationError(msg)

    if not geometry.detected or not settings.calib.overscan_correct:
        cards.append(("OSCANCOR", False, "Overscan correction applied"))
        cards.append(("TRIMMED", False, "Frame trimmed to science region"))
        return result, cards

    level = 0.0
    if geometry.columns and geometry.rows:
        overscan_pixels = result[
            np.ix_(list(geometry.rows), list(geometry.columns))
        ]
        level = float(np.median(overscan_pixels))
    elif geometry.columns:
        overscan_pixels = result[:, list(geometry.columns)]
        level = float(np.median(overscan_pixels))
    elif geometry.rows:
        overscan_pixels = result[list(geometry.rows), :]
        level = float(np.median(overscan_pixels))

    if not np.isfinite(level):
        logger.warning(
            f"Computed overscan level is not finite ({level}); "
            f"skipping subtraction."
        )
        cards.append(("OSCANCOR", False, "Overscan correction applied"))
        cards.append(("TRIMMED", False, "Frame trimmed to science region"))
        return result, cards

    result = result - level
    cards.append(("OSCANCOR", True, "Overscan correction applied"))
    cards.append(("OSCANLVL", level, "Overscan median level (ADU)"))

    n_columns = len(geometry.columns)
    n_rows = len(geometry.rows)
    cards.append(("OSCANNCO", n_columns, "Overscan columns detected"))
    cards.append(("OSCANNRO", n_rows, "Overscan rows detected"))
    cards.append(("OSCANSRC", geometry.source, "How overscan geometry was obtained"))

    row_slice, col_slice = geometry.science_slice
    result = result[row_slice, col_slice]

    iraf_region = (
        f"[{col_slice.start + 1}:{col_slice.stop},"
        f"{row_slice.start + 1}:{row_slice.stop}]"
    )
    cards.append(("TRIMMED", True, "Frame trimmed to science region"))
    cards.append(("TRIMREG", iraf_region, "Trim region (IRAF)"))

    return result, cards
