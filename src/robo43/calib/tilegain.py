"""Derive and check the relative per-tile (per-amplifier) gain pattern of a readout mode.

A photon-transfer gain is wrong by 0.5-1.1 % rms per tile and per measurement
(:mod:`robo43.calib.gain`), which leaves ~1 % rms amplifier steps in every
electron master flat. The true *relative* tile gains are a stable property of
the readout mode, and a flat field shows them directly: across each boundary
between two tiles the illumination is continuous, so the ratio of the ADU levels
just on either side of the boundary is the ratio of the two tiles' gains.

Method (per master flat)
------------------------
1. **ADU levels.** A master is normalised and in electrons (each tile was
   multiplied by the gain recorded in its header, ``GAIN00..GAINnn``, or in the
   ``gain_measurements.json`` beside it); dividing each tile by that gain gives
   the gain-independent ADU flat, exactly (the master build is per-tile
   multiplicative and scale-free). A master with no gain at all
   (``GAINVAL = 'NONE'``) already is an ADU flat.
2. **Boundary steps.** For every boundary between two adjacent tiles (8 x 2
   tiles of T80S: 14 vertical + 8 horizontal boundaries), ``strip`` (16) pixel
   wide strips on both sides are medianed over the boundary's length minus a
   ``margin`` (64 px) at each end. The step is the log ratio of the two
   adjacent strips, corrected for the local illumination gradient with the next
   strips on both sides: ``s = lnR0 - lnL0 - (lnR1 - lnR0)/2 - (lnL0 - lnL1)/2``.
3. **Fit.** ``s(a -> b) = x_a - x_b`` for ``x = ln(gain)`` (an ADU level is
   inversely proportional to its tile's gain): a least-squares fit of one
   ``x_t`` per tile (22 equations for 16 unknowns, one overall gauge), reported
   as ``p_t = exp(x_t) / median(exp(x))``, the median-1 normalisation that
   makes ``K_run * p_t`` have median ``K_run`` like the summary K.
4. **Diagnostics.** The fit residual rms over the boundary equations and the
   *null* rms of the same estimator at pseudo-boundaries inside every tile (a
   column at the tile centre and a row at the tile centre, 32 for 16 tiles)
   where there is no step: a residual at the null level means the steps are
   pure per-tile factors.
5. **Average.** Several masters give the mean of ``ln p_t`` (re-normalised) and
   the per-tile scatter between masters; masters that share input flats are not
   independent, so the scatter is a lower bound of the true one.

:func:`step_summary` applies the same estimator to a master as it is (the e-
master's amplifier steps) or to the ratio of two masters (cross-gain-set steps),
which is how a pattern is validated.

Tile geometry is never hard-coded here: callers pass the row-major tile slices
of :attr:`robo43.calib.overscan.OverscanGeometry.science_tile_slices`, or
:func:`equal_tile_slices` for a regular grid of the trimmed frame.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from robo43.calib.gain import DEFAULT_GAIN_MEASUREMENTS_NAME, GainMeasurementSet

logger = logging.getLogger(__name__)

__all__ = [
    "MasterAnalysis",
    "PatternDerivation",
    "StepSummary",
    "TileGrid",
    "analyse_master",
    "boundary_steps",
    "derive_pattern",
    "equal_tile_slices",
    "fit_tile_gains",
    "format_pattern_toml",
    "null_steps",
    "read_master_tile_gains",
    "step_summary",
    "tile_grid",
    "write_report_json",
]

#: Strip width (px) on each side of a boundary.
STRIP_PX = 16
#: Margin (px) left out at both ends of a boundary (corners, edge effects).
MARGIN_PX = 64

TileSlices = Sequence[tuple[slice, slice]]
#: ``level(rows, cols, tile)`` -> the ADU-equivalent median of a strip of ``tile``.
LevelFn = Callable[[slice, slice, int], float]


@dataclass(frozen=True, slots=True)
class TileGrid:
    """A regular row-major grid of tiles: ``rows[i]``, ``cols[j]`` bound tile ``i*n_cols + j``."""

    rows: tuple[tuple[int, int], ...]
    cols: tuple[tuple[int, int], ...]

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return len(self.cols)

    @property
    def n_tiles(self) -> int:
        return self.n_rows * self.n_cols


@dataclass(frozen=True, slots=True)
class BoundaryStep:
    """One gradient-corrected log step across a boundary: ``ln(level_b / level_a)``.

    ``tile_a`` is the left (or lower-row) tile and ``tile_b`` its right (or
    upper-row-index+1) neighbour; ``axis`` is ``"col"`` for a boundary between
    horizontally adjacent tiles and ``"row"`` between vertically adjacent ones.
    """

    tile_a: int
    tile_b: int
    axis: str
    log_step: float


def tile_grid(tile_slices: TileSlices) -> TileGrid:
    """The regular grid behind row-major ``tile_slices`` (raises ``ValueError`` if irregular)."""
    if not tile_slices:
        msg = "no tile slices"
        raise ValueError(msg)
    rows: list[tuple[int, int]] = []
    cols: list[tuple[int, int]] = []
    for row_sl, col_sl in tile_slices:
        r = (int(row_sl.start), int(row_sl.stop))
        c = (int(col_sl.start), int(col_sl.stop))
        if r not in rows:
            rows.append(r)
        if c not in cols:
            cols.append(c)
    grid = TileGrid(tuple(rows), tuple(cols))
    if grid.n_tiles != len(tile_slices):
        msg = f"{len(tile_slices)} tile slices do not form a {grid.n_rows} x {grid.n_cols} grid"
        raise ValueError(msg)
    for idx, (row_sl, col_sl) in enumerate(tile_slices):
        i, j = divmod(idx, grid.n_cols)
        if (int(row_sl.start), int(row_sl.stop)) != rows[i] or (
            int(col_sl.start), int(col_sl.stop)
        ) != cols[j]:
            msg = f"tile slices are not in row-major order of a regular grid (tile {idx})"
            raise ValueError(msg)
    return grid


def equal_tile_slices(
    shape: tuple[int, int], n_cols: int, n_rows: int
) -> list[tuple[slice, slice]]:
    """Row-major slices of an ``n_cols`` x ``n_rows`` grid of equal tiles of an ``(ny, nx)``
    frame."""
    ny, nx = shape
    if n_cols < 1 or n_rows < 1 or nx % n_cols or ny % n_rows:
        msg = f"a {nx} x {ny} frame does not divide into {n_cols} x {n_rows} equal tiles"
        raise ValueError(msg)
    tile_h, tile_w = ny // n_rows, nx // n_cols
    return [
        (slice(i * tile_h, (i + 1) * tile_h), slice(j * tile_w, (j + 1) * tile_w))
        for i in range(n_rows)
        for j in range(n_cols)
    ]


def _log_step(l1: float, l0: float, r0: float, r1: float) -> float:
    """Gradient-corrected log step from the strip levels ``l1 | l0 | r0 | r1`` (outermost
    left, inner left, inner right, outermost right)."""
    if not all(math.isfinite(v) and v > 0 for v in (l1, l0, r0, r1)):
        msg = f"non-positive or non-finite strip level(s) {(l1, l0, r0, r1)}"
        raise ValueError(msg)
    ln = math.log
    return ln(r0) - ln(l0) - 0.5 * (ln(r1) - ln(r0)) - 0.5 * (ln(l0) - ln(l1))


def _check_room(lo: int, hi: int, at: int, strip: int, what: str) -> None:
    if at - 2 * strip < lo or at + 2 * strip > hi:
        msg = f"{what}: boundary at {at} leaves no room for two {strip}-px strips on each side"
        raise ValueError(msg)


def _col_step(level: LevelFn, grid: TileGrid, i: int, c: int, t_a: int, t_b: int,
              strip: int, margin: int) -> float:
    r0, r1 = grid.rows[i]
    if r1 - r0 <= 2 * margin:
        msg = f"tile rows {r0}:{r1} are too short for a {margin}-px margin at both ends"
        raise ValueError(msg)
    rows = slice(r0 + margin, r1 - margin)
    return _log_step(
        level(rows, slice(c - 2 * strip, c - strip), t_a),
        level(rows, slice(c - strip, c), t_a),
        level(rows, slice(c, c + strip), t_b),
        level(rows, slice(c + strip, c + 2 * strip), t_b),
    )


def _row_step(level: LevelFn, grid: TileGrid, j: int, r: int, t_a: int, t_b: int,
              strip: int, margin: int) -> float:
    c0, c1 = grid.cols[j]
    if c1 - c0 <= 2 * margin:
        msg = f"tile columns {c0}:{c1} are too narrow for a {margin}-px margin at both ends"
        raise ValueError(msg)
    cols = slice(c0 + margin, c1 - margin)
    return _log_step(
        level(slice(r - 2 * strip, r - strip), cols, t_a),
        level(slice(r - strip, r), cols, t_a),
        level(slice(r, r + strip), cols, t_b),
        level(slice(r + strip, r + 2 * strip), cols, t_b),
    )


def boundary_steps(
    level: LevelFn, tile_slices: TileSlices, *, strip: int = STRIP_PX, margin: int = MARGIN_PX
) -> list[BoundaryStep]:
    """Gradient-corrected log steps at every tile boundary (14 + 8 for 8 x 2 tiles)."""
    grid = tile_grid(tile_slices)
    steps: list[BoundaryStep] = []
    for i in range(grid.n_rows):
        for j in range(grid.n_cols):
            t = i * grid.n_cols + j
            if j + 1 < grid.n_cols:
                c = grid.cols[j][1]
                _check_room(grid.cols[j][0], grid.cols[j + 1][1], c, strip, f"tile {t}|{t + 1}")
                steps.append(BoundaryStep(
                    t, t + 1, "col", _col_step(level, grid, i, c, t, t + 1, strip, margin)))
            if i + 1 < grid.n_rows:
                r = grid.rows[i][1]
                below = t + grid.n_cols
                _check_room(grid.rows[i][0], grid.rows[i + 1][1], r, strip, f"tile {t}/{below}")
                steps.append(BoundaryStep(
                    t, below, "row", _row_step(level, grid, j, r, t, below, strip, margin)))
    return steps


def null_steps(
    level: LevelFn, tile_slices: TileSlices, *, strip: int = STRIP_PX, margin: int = MARGIN_PX
) -> np.ndarray:
    """The same estimator at pseudo-boundaries inside each tile, where there is no step: a
    column at the tile centre and a row at the tile centre (32 values for 16 tiles)."""
    grid = tile_grid(tile_slices)
    values: list[float] = []
    for i in range(grid.n_rows):
        for j in range(grid.n_cols):
            t = i * grid.n_cols + j
            c = (grid.cols[j][0] + grid.cols[j][1]) // 2
            r = (grid.rows[i][0] + grid.rows[i][1]) // 2
            values.append(_col_step(level, grid, i, c, t, t, strip, margin))
            values.append(_row_step(level, grid, j, r, t, t, strip, margin))
    return np.asarray(values, dtype=np.float64)


def fit_tile_gains(steps: Sequence[BoundaryStep], n_tiles: int) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares log tile gains from boundary steps: ``(p, residuals)``.

    ``p`` has median 1 (``K_run * p`` then has median ``K_run``); ``residuals`` are
    ``step - (x_a - x_b)`` per boundary equation, in the order of ``steps``.
    """
    n_eq = len(steps)
    design = np.zeros((n_eq + 1, n_tiles))
    rhs = np.zeros(n_eq + 1)
    for k, step in enumerate(steps):
        design[k, step.tile_a] = 1.0
        design[k, step.tile_b] = -1.0
        rhs[k] = step.log_step
    design[n_eq, :] = 1.0  # gauge: sum of the log gains is zero
    if np.linalg.matrix_rank(design) < n_tiles:
        msg = "the boundary steps do not connect all tiles"
        raise ValueError(msg)
    x, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    residuals = rhs[:n_eq] - design[:n_eq] @ x
    gain = np.exp(x)
    return gain / np.median(gain), residuals


def read_master_tile_gains(path: str | Path, n_tiles: int) -> tuple[np.ndarray, str]:
    """The per-tile gains (e-/ADU) a master flat was built with, and where they came from.

    Order of preference: the master's own ``GAIN00..GAINnn`` cards (``"header"``);
    the tile records of ``gain_measurements.json`` in the same directory whose gain
    setting (and readout mode, when the master's ``READOUTM`` is known and the
    record's is not ``UNKNOWN``) match (``"json"``); ones for a master recorded as an
    ADU flat, ``GAINVAL = 'NONE'`` (``"adu"``). Raises ``ValueError`` otherwise.
    """
    path = Path(path)
    header = fits.getheader(path, 0)
    cards = [f"GAIN{i:02d}" for i in range(n_tiles)]
    if all(c in header for c in cards):
        return np.asarray([float(header[c]) for c in cards], dtype=np.float64), "header"
    json_path = path.with_name(DEFAULT_GAIN_MEASUREMENTS_NAME)
    if json_path.exists():
        gain_set = GainMeasurementSet.read(json_path)
        setting = header.get("GAINSET")
        mode = str(header.get("READOUTM", "UNKNOWN")).strip()
        candidates = []
        for key, tiles in gain_set.tile_measurements.items():
            if len(tiles) != n_tiles:
                continue
            if setting is not None and not math.isclose(key[1], float(setting), rel_tol=1e-6):
                continue
            if mode != "UNKNOWN" and key[0] not in (mode, "UNKNOWN"):
                continue
            candidates.append(tiles)
        if len(candidates) == 1:
            return np.asarray([t.gain_e_per_adu for t in candidates[0]]), "json"
        if candidates:
            msg = f"{json_path}: {len(candidates)} tile groups match {path.name}; ambiguous"
            raise ValueError(msg)
    if str(header.get("GAINVAL", "")).strip().upper() == "NONE":
        return np.ones(n_tiles), "adu"
    msg = f"{path}: no per-tile gains (GAIN00.. cards or {DEFAULT_GAIN_MEASUREMENTS_NAME}) found"
    raise ValueError(msg)


def _median_level(image: np.ndarray, gains: np.ndarray | None) -> LevelFn:
    def level(rows: slice, cols: slice, tile: int) -> float:
        value = float(np.median(image[rows, cols]))
        return value / float(gains[tile]) if gains is not None else value

    return level


def _ratio_level(num: np.ndarray, den: np.ndarray) -> LevelFn:
    def level(rows: slice, cols: slice, tile: int) -> float:  # noqa: ARG001
        return float(np.median(np.asarray(num[rows, cols], dtype=np.float64) / den[rows, cols]))

    return level


@dataclass(frozen=True, slots=True)
class StepSummary:
    """Boundary-step statistics in percent (``100 * log step``)."""

    n_steps: int
    rms_pct: float
    max_abs_pct: float
    null_rms_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {"n_steps": self.n_steps, "rms_pct": self.rms_pct,
                "max_abs_pct": self.max_abs_pct, "null_rms_pct": self.null_rms_pct}


def _summarise(level: LevelFn, tile_slices: TileSlices, strip: int, margin: int) -> StepSummary:
    steps = np.asarray([s.log_step for s in boundary_steps(level, tile_slices, strip=strip,
                                                           margin=margin)])
    null = null_steps(level, tile_slices, strip=strip, margin=margin)
    return StepSummary(
        n_steps=len(steps),
        rms_pct=float(100.0 * np.sqrt(np.mean(steps**2))),
        max_abs_pct=float(100.0 * np.max(np.abs(steps))),
        null_rms_pct=float(100.0 * np.sqrt(np.mean(null**2))),
    )


def step_summary(
    path: str | Path,
    tile_slices: TileSlices,
    *,
    versus: str | Path | None = None,
    strip: int = STRIP_PX,
    margin: int = MARGIN_PX,
) -> StepSummary:
    """Amplifier-step statistics of a master flat as it is (its own units), or of the ratio
    ``master / versus`` of two masters (the steps left between two gain sets)."""
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
        if versus is None:
            return _summarise(_median_level(data, None), tile_slices, strip, margin)
        with fits.open(versus, memmap=True) as other:
            return _summarise(_ratio_level(data, other[0].data), tile_slices, strip, margin)


@dataclass(frozen=True, slots=True)
class MasterAnalysis:
    """One master flat's tile gains from boundary continuity, with its diagnostics."""

    path: str
    gain_source: str
    pattern: np.ndarray  # p_t, median 1
    e_steps: StepSummary  # steps of the master as stored (electrons for a gained master)
    adu_steps: StepSummary  # steps of the gain-independent ADU flat
    fit_residual_rms_pct: float
    fit_residual_max_pct: float
    # rms over tiles of ln(used gain / pattern), mean removed: the error of the gains the master was
    # built with; ``None`` for an ADU master (no gains were used)
    used_gain_over_pattern_rms_pct: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "gain_source": self.gain_source,
            "pattern": [float(v) for v in self.pattern],
            "e_steps": self.e_steps.to_dict(),
            "adu_steps": self.adu_steps.to_dict(),
            "fit_residual_rms_pct": self.fit_residual_rms_pct,
            "fit_residual_max_pct": self.fit_residual_max_pct,
            "used_gain_over_pattern_rms_pct": self.used_gain_over_pattern_rms_pct,
        }


def analyse_master(
    path: str | Path,
    tile_slices: TileSlices,
    *,
    strip: int = STRIP_PX,
    margin: int = MARGIN_PX,
) -> MasterAnalysis:
    """Derive one master flat's relative tile gains (see the module docstring)."""
    grid = tile_grid(tile_slices)
    gains, source = read_master_tile_gains(path, grid.n_tiles)
    with fits.open(path, memmap=True) as hdul:
        data = hdul[0].data
        last_row = max(sl[0].stop for sl in tile_slices)
        last_col = max(sl[1].stop for sl in tile_slices)
        if data.shape != (last_row, last_col):
            msg = (
                f"{path}: master shape {data.shape} does not match the tile grid "
                f"{(last_row, last_col)}"
            )
            raise ValueError(msg)
        adu_level = _median_level(data, gains)
        steps = boundary_steps(adu_level, tile_slices, strip=strip, margin=margin)
        pattern, residuals = fit_tile_gains(steps, grid.n_tiles)
        adu_summary = _summarise(adu_level, tile_slices, strip, margin)
        e_summary = _summarise(_median_level(data, None), tile_slices, strip, margin)
    used = np.log(gains / np.median(gains)) - np.log(pattern)
    used -= used.mean()
    used_rms = None if source == "adu" else float(100.0 * np.sqrt(np.mean(used**2)))
    return MasterAnalysis(
        path=str(path),
        gain_source=source,
        pattern=pattern,
        e_steps=e_summary,
        adu_steps=adu_summary,
        fit_residual_rms_pct=float(100.0 * np.sqrt(np.mean(residuals**2))),
        fit_residual_max_pct=float(100.0 * np.max(np.abs(residuals))),
        used_gain_over_pattern_rms_pct=used_rms,
    )


@dataclass(frozen=True, slots=True)
class PatternDerivation:
    """The mean tile gain pattern of one readout mode over several masters."""

    mode: str
    values: np.ndarray  # p_t, median 1
    scatter_pct: np.ndarray  # std over masters of 100*ln p_t per tile (ddof=1; zeros for 1 master)
    loo_rms_pct: tuple[float, ...]  # per master: rms over tiles of its ln p minus the others' mean
    masters: tuple[MasterAnalysis, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "n_masters": len(self.masters),
            "values": [float(v) for v in self.values],
            "scatter_pct": [float(v) for v in self.scatter_pct],
            "scatter_rms_pct": float(np.sqrt(np.mean(self.scatter_pct**2))),
            "scatter_max_pct": float(np.max(self.scatter_pct)),
            "loo_rms_pct": list(self.loo_rms_pct),
            "masters": [m.to_dict() for m in self.masters],
        }


def derive_pattern(masters: Sequence[MasterAnalysis], mode: str) -> PatternDerivation:
    """Average the masters' ``ln p_t`` and re-normalise to median 1; report the scatter."""
    if not masters:
        msg = "no masters to derive a pattern from"
        raise ValueError(msg)
    logs = np.log(np.vstack([m.pattern for m in masters]))
    mean = logs.mean(axis=0)
    values = np.exp(mean) / np.median(np.exp(mean))
    n = len(masters)
    scatter = 100.0 * logs.std(axis=0, ddof=1) if n > 1 else np.zeros(logs.shape[1])
    loo: list[float] = []
    if n > 1:
        for k in range(n):
            others = np.delete(logs, k, axis=0).mean(axis=0)
            diff = logs[k] - others
            diff -= diff.mean()
            loo.append(float(100.0 * np.sqrt(np.mean(diff**2))))
    return PatternDerivation(mode=mode, values=values, scatter_pct=scatter,
                             loo_rms_pct=tuple(loo), masters=tuple(masters))


def format_pattern_toml(derivation: PatternDerivation, *, per_line: int = 8) -> str:
    """The ``mode = [...]`` entry for ``[instrument.tile_gain_pattern]`` (6 decimals)."""
    values = [f"{v:.6f}" for v in derivation.values]
    lines = [f"{derivation.mode} = ["]
    for start in range(0, len(values), per_line):
        lines.append("    " + ", ".join(values[start:start + per_line]) + ",")
    lines.append("]")
    return "\n".join(lines)


def write_report_json(derivation: PatternDerivation, path: str | Path) -> None:
    """Write the derivation and its diagnostics as JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(derivation.to_dict(), indent=2), encoding="utf-8")
