"""Overscan detection and correction from FLAT frames.

Overscan geometry is detected at runtime from FLAT frames plus BIAS frames,
never hardcoded, because the overscan region varies with observation mode
and instrument+telescope combination.

Detection rule: a whole row or column in a FLAT frame that reads at bias
level cannot be illuminated sensor area, hence must be overscan. This module
detects such lines, checks them across multiple flats (intersection), and
trims them from every image after subtracting their median level.

Some instruments (e.g. a multi-amplifier mosaic) carry overscan strips in
the *interior* of the array rather than only along its edges, which splits
the retained science region on one or both axes into more than one
contiguous block. :meth:`OverscanGeometry.science_blocks` reports those
blocks; :func:`apply_overscan` subtracts a per-block ("per-tile") level and
stitches the tiles back together for that case, while the ordinary single
overscan-region layout (exactly one retained block per axis) keeps the
original single-level, single-slice behaviour unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate
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

__all__ = ["OverscanGeometry", "apply_overscan", "detect_overscan", "trim_to_science"]

#: A block is dropped as a spurious sliver when it is shorter than this
#: fraction of the longest retained block on the same axis.
_SLIVER_FRAC = 0.05


def _contiguous_runs(indices: list[int]) -> list[tuple[int, int]]:
    """Return the contiguous (start, stop) runs in a sorted list of indices."""
    if not indices:
        return []

    runs: list[tuple[int, int]] = []
    run_start = indices[0]
    run_end = indices[0] + 1

    for idx in indices[1:]:
        if idx == run_end:
            run_end += 1
        else:
            runs.append((run_start, run_end))
            run_start = idx
            run_end = idx + 1

    runs.append((run_start, run_end))
    return runs


def _retained_blocks(
    axis_len: int, removed: tuple[int, ...], axis_name: str
) -> list[tuple[int, int]]:
    """Contiguous retained blocks on one axis: the complement of ``removed``.

    Blocks shorter than ``_SLIVER_FRAC`` of the longest retained block on
    this axis are spurious slivers, dropped and logged as a WARNING rather
    than kept as a science block.
    """
    retained = sorted(set(range(axis_len)) - set(removed))
    runs = _contiguous_runs(retained)
    if not runs:
        return []

    longest = max(stop - start for start, stop in runs)
    threshold = _SLIVER_FRAC * longest

    kept = [run for run in runs if (run[1] - run[0]) >= threshold]
    dropped = [run for run in runs if (run[1] - run[0]) < threshold]
    if dropped:
        logger.warning(
            f"Dropping {len(dropped)} spurious retained block(s) on the "
            f"{axis_name} axis, shorter than {_SLIVER_FRAC:.0%} of the "
            f"longest retained block ({longest} px): {dropped}"
        )

    return kept


def _assign_overscan_halves(
    blocks: list[tuple[int, int]], runs: list[tuple[int, int]]
) -> dict[tuple[int, int], list[int]]:
    """Map each retained block to the overscan indices of its neighbour run(s).

    A run immediately adjacent to exactly one retained block (an edge run,
    or a run next to only one surviving block after sliver removal) is
    assigned to that block whole. A run sitting strictly between two
    neighbouring blocks is split at its midpoint, so each half is assigned
    to the nearer block. A run adjacent to no retained block contributes to
    no tile's level.
    """
    assignment: dict[tuple[int, int], list[int]] = {block: [] for block in blocks}
    block_by_stop = {block[1]: block for block in blocks}
    block_by_start = {block[0]: block for block in blocks}

    for run_start, run_stop in runs:
        left_block = block_by_stop.get(run_start)
        right_block = block_by_start.get(run_stop)
        if left_block is not None and right_block is not None:
            mid = run_start + (run_stop - run_start) // 2
            assignment[left_block].extend(range(run_start, mid))
            assignment[right_block].extend(range(mid, run_stop))
        elif left_block is not None:
            assignment[left_block].extend(range(run_start, run_stop))
        elif right_block is not None:
            assignment[right_block].extend(range(run_start, run_stop))

    return assignment


def _apply_overscan_tiled(
    result: np.ndarray,
    geometry: OverscanGeometry,
    row_blocks: list[tuple[int, int]],
    col_blocks: list[tuple[int, int]],
) -> tuple[np.ndarray, list[tuple[str, Any, str]]] | None:
    """Subtract a per-tile overscan level and stitch the tiles back together.

    Per-tile level: when overscan columns exist, the median of the overscan
    columns immediately adjacent to the tile's column block (see
    :func:`_assign_overscan_halves`), restricted to the tile's row range.
    When there are no overscan columns, the analogous adjacent row runs
    restricted to the tile's column range.

    Returns ``None`` when any tile's level is not finite, so the caller can
    fall back to the same whole-frame skip used by the single-block path.
    """
    use_columns = bool(geometry.columns)
    if use_columns:
        col_runs = _contiguous_runs(sorted(geometry.columns))
        col_assignment = _assign_overscan_halves(col_blocks, col_runs)
        row_assignment: dict[tuple[int, int], list[int]] = {}
    else:
        row_runs = _contiguous_runs(sorted(geometry.rows))
        row_assignment = _assign_overscan_halves(row_blocks, row_runs)
        col_assignment = {}

    levels: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
    for row_block in row_blocks:
        row_slice = slice(*row_block)
        for col_block in col_blocks:
            col_slice = slice(*col_block)
            if use_columns:
                cols = col_assignment.get(col_block, [])
                level = (
                    float(np.median(result[row_slice, cols])) if cols else float("nan")
                )
            else:
                rows = row_assignment.get(row_block, [])
                level = (
                    float(np.median(result[rows, col_slice])) if rows else float("nan")
                )
            levels[(row_block, col_block)] = level

    if not all(np.isfinite(level) for level in levels.values()):
        non_finite = {key: val for key, val in levels.items() if not np.isfinite(val)}
        logger.warning(
            f"Computed overscan level is not finite for {len(non_finite)} "
            f"tile(s); skipping subtraction. Tiles: {non_finite}"
        )
        return None

    tile_rows = []
    for row_block in row_blocks:
        row_slice = slice(*row_block)
        tiles = [
            result[row_slice, slice(*col_block)] - levels[(row_block, col_block)]
            for col_block in col_blocks
        ]
        tile_rows.append(np.concatenate(tiles, axis=1))
    stitched = np.concatenate(tile_rows, axis=0)

    level_values = list(levels.values())
    cards: list[tuple[str, Any, str]] = []
    cards.append(("OSCANCOR", True, "Overscan correction applied"))
    cards.append(
        ("OSCANLVL", float(np.median(level_values)), "Median overscan level across tiles (ADU)")
    )
    cards.append(
        ("OSCANLMN", float(np.min(level_values)), "Minimum per-tile overscan level (ADU)")
    )
    cards.append(
        ("OSCANLMX", float(np.max(level_values)), "Maximum per-tile overscan level (ADU)")
    )
    cards.append(("OSCANNCO", len(geometry.columns), "Overscan columns detected"))
    cards.append(("OSCANNRO", len(geometry.rows), "Overscan rows detected"))
    cards.append(("OSCANSRC", geometry.source, "How overscan geometry was obtained"))

    row_start, row_stop = row_blocks[0][0], row_blocks[-1][1]
    col_start, col_stop = col_blocks[0][0], col_blocks[-1][1]
    iraf_region = f"[{col_start + 1}:{col_stop},{row_start + 1}:{row_stop}]"

    cards.append(("TRIMMED", True, "Frame trimmed to science region"))
    cards.append(
        ("TRIMNSEG", len(row_blocks) * len(col_blocks), "Number of stitched science tiles")
    )
    cards.append(
        (
            "TRIMREG",
            iraf_region,
            "Bbox of stitched tiles (IRAF, raw)",
        )
    )

    return stitched, cards


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
    def science_blocks(self) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Return ``(row_blocks, col_blocks)``: the retained contiguous index
        ranges on each axis, each a list of ``(start, stop)`` tuples, after
        dropping spurious slivers (see :func:`_retained_blocks`).

        The ordinary single-overscan-region layout yields exactly one block
        per axis. An instrument with overscan strips in the interior of the
        array (e.g. one run per amplifier boundary) yields more than one;
        :func:`apply_overscan` stitches the ``row_block`` x ``col_block``
        tiles back together, each corrected by its own local overscan level.
        """
        ny, nx = self.shape
        row_blocks = _retained_blocks(ny, self.rows, "row")
        col_blocks = _retained_blocks(nx, self.columns, "column")
        return row_blocks, col_blocks

    @property
    def science_slice(self) -> tuple[slice, slice]:
        """Return (row_slice, col_slice) for the science region.

        Defined only for the single-block case: exactly one retained block
        per axis after sliver removal (see :attr:`science_blocks`), which is
        the ordinary single-overscan-region layout. When either axis retains
        more than one block -- overscan strips in the interior of the array
        split it -- there is no single contiguous science region; use
        :attr:`science_blocks` for the per-axis block list, and
        :func:`apply_overscan` to stitch the tiles instead of calling this
        property.

        Raises:
            CalibrationError: if either axis retains more than one block.
        """
        row_blocks, col_blocks = self.science_blocks

        if not row_blocks or not col_blocks:
            return slice(0, 0), slice(0, 0)

        if len(row_blocks) > 1 or len(col_blocks) > 1:
            msg = (
                f"science_slice is undefined for a multi-block geometry "
                f"(row_blocks={row_blocks}, col_blocks={col_blocks}); use "
                f"science_blocks and apply_overscan's tiled stitching instead."
            )
            raise CalibrationError(msg)

        return slice(*row_blocks[0]), slice(*col_blocks[0])

    @property
    def science_tile_slices(self) -> list[tuple[slice, slice]]:
        """Row-major list of ``(row_slice, col_slice)`` tuples, one per
        science tile, in the coordinate space of the array
        :func:`apply_overscan` returns (i.e. *after* overscan subtraction
        and trimming/stitching) -- not the raw coordinates
        :attr:`science_blocks`, :attr:`columns`, and :attr:`rows` use.

        Because :func:`apply_overscan`'s tiled branch concatenates the
        retained ``row_block`` x ``col_block`` tiles in row-major order with
        no gaps, each tile's stitched-space extent is simply the cumulative
        sum of the raw block sizes on each axis, in that same order -- true
        for both the single-block layout (exactly one tile, spanning the
        whole trimmed array) and a multi-block, multi-amplifier layout,
        which is why this one property serves both. Tile ``i`` is
        amplifier ``i`` under this pipeline's one-tile-per-amplifier
        convention (see :mod:`robo43.calib.gain`).

        Returns ``[]`` if either axis retains no science block at all.
        """
        row_blocks, col_blocks = self.science_blocks
        if not row_blocks or not col_blocks:
            return []

        row_sizes = [stop - start for start, stop in row_blocks]
        col_sizes = [stop - start for start, stop in col_blocks]
        row_offsets = list(accumulate([0, *row_sizes]))
        col_offsets = list(accumulate([0, *col_sizes]))

        tiles: list[tuple[slice, slice]] = []
        for i in range(len(row_blocks)):
            row_slice = slice(row_offsets[i], row_offsets[i + 1])
            for j in range(len(col_blocks)):
                col_slice = slice(col_offsets[j], col_offsets[j + 1])
                tiles.append((row_slice, col_slice))
        return tiles

    def header_cards(self) -> list[tuple[str, Any, str]]:
        """Return FITS header cards describing the geometry."""
        cards: list[tuple[str, Any, str]] = []
        cards.append(("OSCANNCO", len(self.columns), "Overscan columns detected"))
        cards.append(("OSCANNRO", len(self.rows), "Overscan rows detected"))
        cards.append(("OSCANSRC", self.source, "How overscan geometry was obtained"))
        return cards

    def tile_header_cards(self) -> list[tuple[str, Any, str]]:
        """Per-tile pixel-boundary cards in the trimmed/stitched frame.

        Returns ``TILE00``..``TILEnn`` (zero-padded 2-digit tile index, an
        IRAF-style 1-indexed inclusive ``[x1:x2,y1:y2]`` region string per
        entry, mirroring TRIMREG's own format), one per
        :attr:`science_tile_slices` entry, in that same row-major order.
        Returns ``[]`` for a single-tile geometry (no boundaries worth
        recording) -- mirrors
        :func:`robo43.calib.gain.per_tile_header_cards`'s own >1-tile
        threshold. This is the only place a science tile's pixel bounds are
        recorded on a reduced product; nothing else reconstructs them from
        OSCANNCO/OSCANNRO/TRIMREG alone (see
        :func:`robo43.photometry.gaia_zp.parse_tile_slices`, the reader).
        """
        tiles = self.science_tile_slices
        if len(tiles) <= 1:
            return []
        cards: list[tuple[str, Any, str]] = []
        for idx, (row_slice, col_slice) in enumerate(tiles):
            region = (
                f"[{col_slice.start + 1}:{col_slice.stop},"
                f"{row_slice.start + 1}:{row_slice.stop}]"
            )
            cards.append((f"TILE{idx:02d}", region, f"Tile {idx} region (IRAF, trimmed frame)"))
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
        row_blocks, col_blocks = geom.science_blocks
        if len(row_blocks) == 1 and len(col_blocks) == 1:
            row_slice, col_slice = geom.science_slice
            logger.info(
                f"Science region after trim: rows [{row_slice.start}:{row_slice.stop}], "
                f"cols [{col_slice.start}:{col_slice.stop}]"
            )
        else:
            logger.info(
                f"Science region after trim: {len(row_blocks)} row block(s) x "
                f"{len(col_blocks)} col block(s); rows {row_blocks}, cols {col_blocks}"
            )

    return geom


def trim_to_science(data: np.ndarray, geometry: OverscanGeometry) -> np.ndarray:
    """Select (and, for a multi-block layout, stitch) ``data``'s science region.

    Reproduces exactly the trim/stitch geometry :func:`apply_overscan`
    applies to the science array itself -- via the same
    :attr:`OverscanGeometry.science_blocks` -- with no level subtracted, so
    a boolean or integer array computed in the *raw* frame's coordinate
    system (e.g. a saturation or non-linearity mask evaluated on the raw ADU
    read, before overscan correction) ends up pixel-aligned with the
    (possibly trimmed/stitched) science data ``apply_overscan`` returns.

    Callers must apply this only when ``apply_overscan`` actually trimmed
    the corresponding science data -- e.g. by checking that its returned
    array's shape differs from ``data.shape`` here -- since one of
    ``apply_overscan``'s own no-op fallbacks (``overscan_correct=False``, no
    detected overscan, or a non-finite computed level) leaves the science
    data untrimmed too, in which case this function must not be called.

    Returns ``data`` unchanged if either axis retains no science block at
    all (mirrors :attr:`OverscanGeometry.science_blocks`'s own empty case).
    """
    row_blocks, col_blocks = geometry.science_blocks
    if not row_blocks or not col_blocks:
        return data

    if len(row_blocks) == 1 and len(col_blocks) == 1:
        row_slice, col_slice = slice(*row_blocks[0]), slice(*col_blocks[0])
        return data[row_slice, col_slice]

    tile_rows = []
    for row_block in row_blocks:
        row_slice = slice(*row_block)
        tiles = [data[row_slice, slice(*col_block)] for col_block in col_blocks]
        tile_rows.append(np.concatenate(tiles, axis=1))
    return np.concatenate(tile_rows, axis=0)


def apply_overscan(
    data: np.ndarray,
    geometry: OverscanGeometry,
    *,
    settings: Settings,
) -> tuple[np.ndarray, list[tuple[str, Any, str]]]:
    """Subtract overscan level, trim, and return the result with header cards.

    If overscan correction is disabled or no overscan was detected, returns
    the data unchanged and records OSCANCOR=False, TRIMMED=False.

    When the retained science region is a single contiguous block on both
    axes (the ordinary single-overscan-region layout), this subtracts one
    global median level and trims to that one block, exactly as before. When
    overscan strips in the interior of the array split either axis into more
    than one retained block, this instead subtracts a per-tile level (see
    :func:`_apply_overscan_tiled`) and stitches the ``row_block`` x
    ``col_block`` tiles back together; a non-finite tile level falls back to
    the same whole-frame skip as the single-block path.

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

    row_blocks, col_blocks = geometry.science_blocks

    if not row_blocks or not col_blocks:
        logger.warning(
            "No retained science region after overscan removal on at least "
            "one axis; skipping correction."
        )
        cards.append(("OSCANCOR", False, "Overscan correction applied"))
        cards.append(("TRIMMED", False, "Frame trimmed to science region"))
        return result, cards

    if len(row_blocks) > 1 or len(col_blocks) > 1:
        tiled = _apply_overscan_tiled(result, geometry, row_blocks, col_blocks)
        if tiled is not None:
            return tiled
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
