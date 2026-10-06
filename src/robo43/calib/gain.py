"""Gain (e-/ADU) measurement by photon transfer, from FLAT and ZERO frames.

The QHY600M's ``GAIN`` header keyword is a dimensionless vendor setting
index (this camera's real frames take only the values 10 or 30), not an
e-/ADU conversion factor -- so, exactly like the overscan region in
:mod:`robo43.calib.overscan`, the physical gain cannot be hardcoded in the
instrument profile and cannot be read off a header keyword. It is instead
measured at runtime from that night's own flat and bias frames, per
``(readout_mode, gain_setting, binning)`` group, by the standard
photon-transfer-curve (PTC) relation::

    K [e-/ADU] = ((mu_f1 + mu_f2) - (mu_b1 + mu_b2)) / (var(F1 - F2) - var(B1 - B2))

using flat pairs matched in filter and exposure time and bias pairs shared
across the group, with read noise falling out of the same bias pairs as
``RN [e-] = K * sqrt(var(B1 - B2) / 2)``.

The two variance terms that feed the PTC ratio above (the flat term and the
bias term) are each computed by :func:`_mad_var`, an MAD-based, Gaussian-
normalised scale estimator (``mad_std(diff) ** 2``), rather than by sigma-
clipping. Using the *same* estimator for both terms is what makes their
residual robustness bias cancel between numerator and denominator: real
ROBO43 bias differences have heavy non-Gaussian tails (hot pixels, random-
telegraph noise) that a 3-sigma clip strips out of the bias term far more
aggressively than out of the near-Gaussian, shot-noise-dominated flat term,
so clipping's bias does *not* cancel and leaves K biased high. Read noise, by
contrast, is derived by :func:`_raw_var` from the *unclipped* sample
variance of the bias difference: those same tails are genuine read noise
that must be counted, not suppressed by either clipping or a robust
estimator, or the reported read noise comes out biased low. Do not
consolidate these two paths -- they are deliberately different estimators
for deliberately different quantities.

A group with fewer than ``settings.calib.gain_min_frames_per_group`` flats
or biases -- or whose flat pairs all fail the saturation/non-linearity or
positive-variance-excess checks -- yields no measurement at all, rather
than a guessed value; the caller (:mod:`robo43.reduce.frame`) reduces such a
frame in ADU instead of converting it with an invented factor.

Per-amplifier (per-tile) measurement
-------------------------------------
An instrument whose overscan geometry splits the trimmed frame into more
than one contiguous science block per axis (T80S/T80Cam: 16 amplifiers, 2
row blocks x 8 col blocks) has one photon-transfer curve *per tile*, not
one for the whole mosaic -- amplifiers differ in gain and read noise, and a
single whole-frame crop straddling amplifier boundaries mixes their levels
and noise into one biased estimate. Tile geometry is never hardcoded: it
comes from :attr:`robo43.calib.overscan.OverscanGeometry.science_tile_slices`
(row-major, 0-based, in the coordinate space :func:`~robo43.calib.overscan.
apply_overscan` returns), which degenerates to a single tile spanning the
whole frame for an instrument with no interior overscan strips (ROBO43) --
so the single-amplifier path below is numerically the n_tiles==1 case of
the same code, not a separate implementation.

Flat pairing is level-matched rather than assumed consecutive: pooled T80S
flats span several nights at different lamp/sky levels, so frames are
grouped by (filter, night) and paired within a group by nearest median
level, accepting a pair only when the two levels differ by no more than
``settings.calib.gain_flat_pair_level_tol_frac`` (relative to the higher of
the two). A matched pair's fainter frame is scaled, in its debiased signal,
to the brighter frame's level before differencing (``r = signal_bright /
signal_faint``), which is what lets pairs of slightly different level be
combined without residual flat-pattern leakage; the PTC relation is
generalised accordingly (see :func:`_pair_gain_estimate`), and reduces
exactly to the classical two-equal-level-frame formula when ``r == 1``.

Tile gain pattern (``[instrument.tile_gain_pattern]``)
------------------------------------------------------
A per-tile PTC value is wrong by 0.5-1.1 % rms per measurement (one or a few
flat pairs; the bias-pair variance is the limit), which leaves ~1 % rms
amplifier steps in every electron master flat and in any science frame that
is flat-fielded with a master built from another gain set. The *relative*
tile gain pattern is a stable property of the readout mode (flat continuity
across the amplifier boundaries gives it to ~0.03 %; see
:mod:`robo43.calib.tilegain` and ``tests/reports/t80s_flat_crossmode_REPORT.md``
in the relphot repository). When the instrument profile tabulates it for a
frame group's readout mode, :func:`apply_tile_gain_pattern` (called at the end
of :func:`measure_gain`) replaces every tile's gain by ``K_run * p_t``, where
``K_run`` is the group's frame-level summary K (the median of the per-tile PTC
values, measured exactly as before) and ``p_t`` is the tabulated pattern,
normalised to median 1 so that the median of the tile gains stays ``K_run``.
The noisy per-tile PTC value is kept in ``ptc_gain_e_per_adu``; the master flat
build and the science conversion read ``gain_e_per_adu`` and so share the same
gains by construction. A group with no tabulated pattern, or without a
measurable ``K_run``, is untouched.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from astropy.stats import mad_std

from robo43.calib.bias import apply_overscan_trim, select_frames
from robo43.config import Settings
from robo43.io.fits_io import read_image

if TYPE_CHECKING:
    from robo43.calib.overscan import OverscanGeometry
    from robo43.instruments.profile import InstrumentProfile
    from robo43.io.meta import FrameMeta

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_GAIN_MEASUREMENTS_NAME",
    "GainMeasurement",
    "GainMeasurementSet",
    "TileGainPattern",
    "apply_tile_gain_pattern",
    "measure_gain",
    "no_measurement_cards",
    "per_tile_header_cards",
    "tile_gain_pattern_for",
]

#: Default basename of the cached measurement file written into the calib
#: directory alongside the master bias/dark/flat frames.
DEFAULT_GAIN_MEASUREMENTS_NAME = "gain_measurements.json"

GainKey = tuple[str, float, int]  # (readout_mode, gain_setting, binning)


@dataclass(frozen=True, slots=True)
class GainMeasurement:
    """One measured (readout_mode, gain_setting, binning) group, or one tile
    within it.

    ``tile_index`` is ``-1`` for the group's frame-level summary record
    (median K and read noise across tiles; this is what
    :meth:`GainMeasurementSet.get` returns, unchanged from before per-tile
    measurement existed) and ``0..n_tiles-1`` -- row-major over
    :attr:`~robo43.calib.overscan.OverscanGeometry.science_tile_slices` --
    for a per-amplifier record (see :meth:`GainMeasurementSet.tiles_for`).
    An instrument with a single science tile (ROBO43) has ``n_tiles == 1``
    and its one tile record's values are numerically identical to the
    frame-level summary's. Both defaults (``tile_index=-1``, ``n_tiles=1``)
    make a pre-existing ``gain_measurements.json`` written before per-tile
    measurement existed parse unchanged as an all-frame-level, single-tile
    set: see :meth:`GainMeasurementSet.read`.
    """

    readout_mode: str
    gain_setting: float
    binning: int
    gain_e_per_adu: float
    read_noise_e: float
    n_pairs: int
    scatter_frac: float
    source: str = "measured"
    # Ratio of the raw (unclipped) bias-pair variance to the MAD-based one.
    # Near 1 for ordinary data; a real bias difference's heavy tails alone
    # produce a modest, expected excess (~1.1-1.5x on real ROBO43 data). A
    # ratio well outside that -- a pathological frame such as a cosmic-ray
    # storm or a light leak -- is flagged via a warning log when it exceeds
    # settings.calib.gain_var_disagreement_factor, but is always reported
    # here rather than silently folded into either variance estimate.
    bias_var_ratio: float = 1.0
    # -1 = frame-level summary; >=0 = this tile's own (amplifier) index.
    tile_index: int = -1
    # Total number of science tiles (amplifiers) in this group; 1 for a
    # single-amplifier instrument or an unmeasurable multi-amp geometry.
    n_tiles: int = 1
    # The tile's own photon-transfer value, kept for diagnostics when a tile
    # gain pattern replaced ``gain_e_per_adu`` (see apply_tile_gain_pattern);
    # ``None`` when ``gain_e_per_adu`` is the PTC value itself.
    ptc_gain_e_per_adu: float | None = None
    # Label of the tile gain pattern behind this group's per-tile gains
    # (``"table:<readout_mode>"``), or ``None`` when the per-tile PTC values
    # are used as measured.
    gain_pattern: str | None = None

    def header_cards(self) -> list[tuple[str, Any, str]]:
        """FITS header cards recording this measurement on a reduced product.

        Always the frame-level (median-across-tiles) summary's cards,
        whichever ``GainMeasurement`` this is called on; per-amplifier
        cards come from :func:`per_tile_header_cards` instead.
        """
        cards: list[tuple[str, Any, str]] = [
            ("GAINVAL", self.gain_e_per_adu, "measured gain (e-/ADU)"),
            ("GAINSRC", self.source, "origin of GAINVAL: measured/none"),
            ("GAINSET", self.gain_setting, "camera GAIN header setting index"),
            ("READOUTM", self.readout_mode, "camera readout mode"),
            ("GAINRN", self.read_noise_e, "measured read noise, e- (bias pairs)"),
            ("GAINNPR", self.n_pairs, "flat pairs used for gain measurement"),
            ("GAINSCT", self.scatter_frac, "relative scatter across gain pairs"),
            ("GAINNTL", self.n_tiles, "number of amplifier tiles measured"),
        ]
        if self.gain_pattern is not None:
            cards.append(
                ("GAINPAT", self.gain_pattern, "tile gains = GAINVAL x tabulated pattern")
            )
        return cards


def per_tile_header_cards(
    tile_measurements: Sequence[GainMeasurement] | None,
) -> list[tuple[str, Any, str]]:
    """FITS header cards for a per-amplifier gain/read-noise breakdown.

    Returns ``GAIN00``..``GAIN15``/``RDNS00``..``RDNS15`` (zero-padded
    2-digit tile index, both <=8-character FITS keywords), one pair per
    tile, in tile-index order. Returns ``[]`` when ``tile_measurements`` is
    ``None`` or has one or zero entries -- the frame-level GAINVAL/GAINRN
    from :meth:`GainMeasurement.header_cards` already cover that case, and
    a per-amplifier breakdown of a single amplifier would be redundant.
    """
    if not tile_measurements or len(tile_measurements) <= 1:
        return []
    cards: list[tuple[str, Any, str]] = []
    for tm in tile_measurements:
        idx = tm.tile_index if tm.tile_index >= 0 else 0
        what = "gain used" if tm.gain_pattern is not None else "measured gain"
        cards.append((f"GAIN{idx:02d}", tm.gain_e_per_adu, f"{what}, tile {idx} (e-/ADU)"))
        cards.append(
            (f"RDNS{idx:02d}", tm.read_noise_e, f"measured read noise, tile {idx} (e-)")
        )
    return cards


def no_measurement_cards(
    gain_setting: float | None, readout_mode: str | None
) -> list[tuple[str, Any, str]]:
    """FITS header cards for a frame whose gain setting has no measurement."""
    return [
        ("GAINVAL", "NONE", "no measured gain for this setting"),
        ("GAINSRC", "none", "origin of GAINVAL: measured/none"),
        (
            "GAINSET",
            gain_setting if gain_setting is not None else "NONE",
            "camera GAIN header setting index",
        ),
        ("READOUTM", readout_mode or "UNKNOWN", "camera readout mode"),
    ]


#: Record fields written to ``gain_measurements.json`` only while they are set,
#: so a group without a tile gain pattern is serialised byte-for-byte as before.
_OPTIONAL_RECORD_FIELDS = ("ptc_gain_e_per_adu", "gain_pattern")


def _record_dict(measurement: GainMeasurement) -> dict[str, Any]:
    record = asdict(measurement)
    for name in _OPTIONAL_RECORD_FIELDS:
        if record[name] is None:
            del record[name]
    return record


@dataclass(frozen=True, slots=True)
class TileGainPattern:
    """A profile's relative per-tile gain pattern for one readout mode.

    ``values[t]`` is tile ``t``'s gain (e-/ADU) relative to the frame-level
    summary K: normalised to median 1, so ``K_run * values`` has median
    ``K_run`` -- the same definition as the summary K, the median over tiles.
    Tile order is the row-major order of
    :attr:`~robo43.calib.overscan.OverscanGeometry.science_tile_slices`.
    """

    readout_mode: str
    values: tuple[float, ...]

    @property
    def label(self) -> str:
        """The ``GAINPAT`` header value recognising this pattern on a product."""
        return f"table:{self.readout_mode}"


def tile_gain_pattern_for(
    profile: InstrumentProfile | None, readout_mode: str | None
) -> TileGainPattern | None:
    """The profile's tile gain pattern for ``readout_mode``, or ``None`` when
    the profile has no ``[instrument.tile_gain_pattern]`` table or no entry for
    that mode (a frame without a readout mode never matches)."""
    table = getattr(profile, "tile_gain_pattern", None)
    if not table or not readout_mode:
        return None
    mode = str(readout_mode).strip()
    values = table.get(mode)
    if values is None:
        return None
    return TileGainPattern(readout_mode=mode, values=tuple(values))


def apply_tile_gain_pattern(
    gain_set: GainMeasurementSet, profile: InstrumentProfile | None
) -> GainMeasurementSet:
    """Replace each group's per-tile PTC gains by ``K_run * p_t`` where the
    profile tabulates a pattern for the group's readout mode.

    ``K_run`` is the group's frame-level summary ``gain_e_per_adu`` (left
    untouched). Each tile record keeps its own PTC value in
    ``ptc_gain_e_per_adu`` and gets ``gain_pattern = "table:<mode>"``; its read
    noise in electrons is rescaled by the same gain ratio (the noise in ADU is
    what was measured). A group with no pattern for its mode, no per-tile
    records, or a tile count different from the pattern's is returned
    unchanged (the last case with a WARNING). Idempotent, and a no-op for a
    profile without a table.
    """
    if not getattr(profile, "tile_gain_pattern", None):
        return gain_set
    measurements = dict(gain_set.measurements)
    tile_measurements = dict(gain_set.tile_measurements)
    for key, summary in gain_set.measurements.items():
        pattern = tile_gain_pattern_for(profile, key[0])
        if pattern is None:
            continue
        tiles = gain_set.tile_measurements.get(key)
        if not tiles or len(tiles) != len(pattern.values):
            logger.warning(
                "gain group %s: tile gain pattern %s has %d value(s) but the group has "
                "%d tile record(s); keeping the per-tile PTC gains.",
                key, pattern.label, len(pattern.values), len(tiles) if tiles else 0,
            )
            continue
        k_run = summary.gain_e_per_adu
        new_tiles = []
        for tm, p in zip(tiles, pattern.values, strict=True):
            gain = k_run * p
            ptc = tm.ptc_gain_e_per_adu if tm.ptc_gain_e_per_adu is not None else tm.gain_e_per_adu
            new_tiles.append(
                dataclasses.replace(
                    tm,
                    gain_e_per_adu=gain,
                    read_noise_e=tm.read_noise_e * gain / tm.gain_e_per_adu,
                    ptc_gain_e_per_adu=ptc,
                    gain_pattern=pattern.label,
                )
            )
        tile_measurements[key] = tuple(new_tiles)
        measurements[key] = dataclasses.replace(summary, gain_pattern=pattern.label)
        logger.info(
            "gain group %s: per-tile gains = K_run (%.4f e-/ADU) x tile gain pattern %s "
            "(PTC per-tile values kept as ptc_gain_e_per_adu).",
            key, k_run, pattern.label,
        )
    return GainMeasurementSet(measurements, tile_measurements)


def _normalise_key(
    readout_mode: str | None, gain_setting: float | None, binning: int | None
) -> GainKey | None:
    if gain_setting is None:
        return None
    return (
        (readout_mode or "UNKNOWN").strip(),
        float(gain_setting),
        int(binning) if binning is not None else 1,
    )


@dataclass(frozen=True, slots=True)
class GainMeasurementSet:
    """Every gain measurement obtained for one night, keyed by group.

    ``measurements`` holds each group's frame-level summary (``tile_index
    == -1``), exactly as before per-tile measurement existed.
    ``tile_measurements`` additionally holds, for the same key, the tuple of
    per-tile records in tile-index order (length ``n_tiles``, including for
    the single-tile ROBO43 case, where it is a 1-tuple numerically identical
    to the frame-level summary) -- see :meth:`tiles_for`. A
    ``gain_measurements.json`` written before this field existed has no
    tile-index-bearing records at all, so it round-trips into an empty
    ``tile_measurements`` -- callers must treat a missing/empty result from
    :meth:`tiles_for` as "fall back to the frame-level summary applied
    uniformly", never as "no measurement".
    """

    measurements: dict[GainKey, GainMeasurement]
    tile_measurements: dict[GainKey, tuple[GainMeasurement, ...]] = field(default_factory=dict)

    def get(
        self,
        readout_mode: str | None,
        gain_setting: float | None,
        binning: int | None,
    ) -> GainMeasurement | None:
        """Look up the measurement for a frame's own (readout_mode, gain_setting,
        binning), or ``None`` when that group was never measured."""
        key = _normalise_key(readout_mode, gain_setting, binning)
        if key is None:
            return None
        return self.measurements.get(key)

    def tiles_for(
        self,
        readout_mode: str | None,
        gain_setting: float | None,
        binning: int | None,
    ) -> tuple[GainMeasurement, ...] | None:
        """Per-tile measurements for a frame's own group, in tile-index order.

        Returns ``None`` when this group has no per-tile records at all --
        either because it was never measured, or because it was measured by
        a version of this module that predates per-tile storage. The caller
        must fall back to applying the frame-level summary from :meth:`get`
        uniformly across the whole frame in that case (numerically identical
        to today's single-amplifier behaviour), never treat it as "no gain".
        """
        key = _normalise_key(readout_mode, gain_setting, binning)
        if key is None:
            return None
        return self.tile_measurements.get(key)

    def to_json(self) -> list[dict[str, Any]]:
        records = [_record_dict(m) for m in self.measurements.values()]
        for tiles in self.tile_measurements.values():
            records.extend(_record_dict(m) for m in tiles)
        return records

    def write(self, path: str | Path) -> None:
        """Write this set to disk as JSON, creating parent directories as needed."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_json(), handle, indent=2)

    @classmethod
    def read(cls, path: str | Path) -> GainMeasurementSet:
        """Read a previously written set, or return an empty set.

        Never raises: a missing or corrupt file is logged and treated as no
        measurements available, matching the graceful-degradation behaviour
        of every other cached calibration product. Records with
        ``tile_index == -1`` (including every record from a file written
        before this field existed, via its dataclass default) populate
        ``measurements``; records with ``tile_index >= 0`` populate
        ``tile_measurements``, sorted back into tile-index order.
        """
        in_path = Path(path)
        if not in_path.exists():
            return cls({})
        try:
            with in_path.open("r", encoding="utf-8") as handle:
                records = json.load(handle)
        except Exception as exc:
            logger.warning("failed to read gain measurements from %s: %s", in_path, exc)
            return cls({})

        measurements: dict[GainKey, GainMeasurement] = {}
        tile_records: dict[GainKey, list[GainMeasurement]] = {}
        for record in records:
            try:
                measurement = GainMeasurement(**record)
            except Exception as exc:
                logger.warning("skipping malformed gain measurement record %r: %s", record, exc)
                continue
            key = _normalise_key(
                measurement.readout_mode, measurement.gain_setting, measurement.binning
            )
            if key is None:
                continue
            if measurement.tile_index >= 0:
                tile_records.setdefault(key, []).append(measurement)
            else:
                measurements[key] = measurement

        tile_measurements = {
            key: tuple(sorted(tiles, key=lambda m: m.tile_index))
            for key, tiles in tile_records.items()
        }
        return cls(measurements, tile_measurements)


def _central_crop(data: np.ndarray, frac: float) -> np.ndarray:
    """Return the central ``frac``-fraction (per axis) region of ``data``."""
    h, w = data.shape
    row_half = max(1, int(h * frac / 2))
    col_half = max(1, int(w * frac / 2))
    r0 = h // 2 - row_half
    r1 = h // 2 + row_half
    c0 = w // 2 - col_half
    c1 = w // 2 + col_half
    return data[r0:r1, c0:c1]


def _tile_slices(overscan: OverscanGeometry | None) -> list[tuple[slice, slice]]:
    """Row-major tile slices for this run's amplifier geometry.

    Delegates to :attr:`~robo43.calib.overscan.OverscanGeometry.
    science_tile_slices` -- never hardcoded -- falling back to a single
    tile spanning the whole (already overscan-trimmed) frame when no
    geometry is available or it detected no science blocks at all.
    """
    if overscan is not None:
        tiles = overscan.science_tile_slices
        if tiles:
            return tiles
    return [(slice(None), slice(None))]


def _load_tiles(
    meta: FrameMeta,
    *,
    profile: InstrumentProfile,
    settings: Settings,
    overscan: OverscanGeometry | None,
    tile_slices: Sequence[tuple[slice, slice]],
) -> list[np.ndarray]:
    """Read one frame, apply overscan/trim exactly as the master builders do,
    then return one central-cropped sampling region per tile, in the same
    (row-major) order as ``tile_slices``."""
    data, _header = read_image(meta.path, profile)
    if overscan is not None:
        from robo43.calib.overscan import apply_overscan

        trimmed, _cards = apply_overscan(data, overscan, settings=settings)
    else:
        trimmed, _cards = apply_overscan_trim(data, profile=profile, settings=settings)
    trimmed = trimmed.astype(np.float64)
    frac = settings.calib.gain_region_frac
    return [_central_crop(trimmed[row_sl, col_sl], frac) for row_sl, col_sl in tile_slices]


def _frame_level(crops: Sequence[np.ndarray]) -> float:
    """A representative whole-frame level from a frame's per-tile crops.

    Used only to decide which flats pair together (see
    :func:`_level_matched_pairs`); every tile contributes so this tracks
    the frame's overall illumination even though each tile is only
    centrally sampled.
    """
    return float(np.median(np.concatenate([c.ravel() for c in crops])))


def _consecutive_pairs(items: Sequence[FrameMeta]) -> list[tuple[FrameMeta, FrameMeta]]:
    """Pair up (0,1), (2,3), ... ; an odd one out is left unpaired."""
    return [(items[i], items[i + 1]) for i in range(0, len(items) - 1, 2)]


def _level_matched_pairs(
    items: Sequence[tuple[FrameMeta, float]], tol_frac: float
) -> list[tuple[FrameMeta, FrameMeta]]:
    """Pair frames of nearest median level, within ``tol_frac`` relative
    tolerance, greedily consuming each frame at most once.

    ``items`` is sorted here by descending level; for each not-yet-used
    frame (from the brightest down), the nearest not-yet-used lower-or-equal
    level within tolerance is paired with it. Because the working list stays
    sorted by descending level throughout, the relative gap to the current
    candidate is non-decreasing as the search moves to fainter frames, so the
    search for frame ``i`` can stop at the first candidate outside tolerance.
    A level of zero or a frame that never finds a match within tolerance is
    left unpaired -- never forced into a mismatched pair.
    """
    remaining = sorted(items, key=lambda item: item[1], reverse=True)
    used: set[FrameMeta] = set()
    pairs: list[tuple[FrameMeta, FrameMeta]] = []
    n = len(remaining)
    for i in range(n):
        m1, lvl1 = remaining[i]
        if m1 in used or lvl1 <= 0:
            continue
        best_j: int | None = None
        best_rel: float | None = None
        for j in range(i + 1, n):
            m2, lvl2 = remaining[j]
            if m2 in used or lvl2 <= 0:
                continue
            rel = (lvl1 - lvl2) / lvl1
            if rel > tol_frac:
                break
            if best_rel is None or rel < best_rel:
                best_rel = rel
                best_j = j
        if best_j is not None:
            m2, _lvl2 = remaining[best_j]
            pairs.append((m1, m2))
            used.add(m1)
            used.add(m2)
    return pairs


def _mad_var(diff: np.ndarray) -> float:
    """MAD-based variance of a difference image: ``mad_std(diff) ** 2``.

    Used for *both* PTC variance terms that produce K (the bias term and the
    flat term). ``mad_std`` carries the 1.4826 normalisation that makes it
    unbiased for the Gaussian core of the distribution while still being
    robust to cosmic rays and dust; applying the identical estimator to both
    terms is what lets its residual bias cancel between them. Must not be
    used for the read-noise estimate -- see :func:`_raw_var`.
    """
    return float(mad_std(diff) ** 2)


def _raw_var(diff: np.ndarray) -> float:
    """Unclipped sample variance of a difference image: ``var(diff, ddof=1)``.

    Used only for the read-noise estimate. Real CMOS bias differences carry
    heavy non-Gaussian tails (hot pixels, random-telegraph noise) that are
    genuine read noise, not outliers -- neither sigma-clipping nor a robust
    scale estimator may suppress them here, or the reported read noise comes
    out biased low.
    """
    return float(np.var(diff, ddof=1))


def _rejected_by_level(
    level_adu: float, bias_level_adu: float, profile: InstrumentProfile
) -> bool:
    """True when a flat's signal (above bias) is saturated or non-linear."""
    signal = level_adu - bias_level_adu
    if profile.nonlinearity_adu is not None and signal > profile.nonlinearity_adu:
        return True
    return bool(profile.saturation_adu is not None and signal > profile.saturation_adu)


def _pair_gain_estimate(
    d1: np.ndarray,
    d2: np.ndarray,
    bias_mu: float,
    bias_var_mad: float,
) -> float | None:
    """One tile's K [e-/ADU] estimate from one level-matched flat pair.

    ``d1``/``d2`` are the two frames' central-crop arrays for this tile (raw
    ADU, not yet bias-subtracted); ``d1`` is taken to be the brighter (or
    equally bright) of the two by convention of the caller, but the formula
    below does not require that -- it is derived generally.

    Generalises the classical two-equal-level PTC formula to a pair whose
    debiased signals differ by a ratio ``r = signal1 / signal2``: the
    fainter frame's *debiased* signal is scaled by ``r`` before
    differencing (so the difference has zero mean regardless of the level
    mismatch), and the noise terms are propagated through that same scaling
    -- read noise (from the bias pairs, assumed equal for both frames of
    this pair) enters as ``(1 + r**2)`` while shot noise enters as
    ``(1 + r)`` (because scaling the fainter frame's shot noise variance by
    ``r**2`` and its own shot-noise level by ``r`` before combining with the
    brighter frame's collapses to ``r * signal1``, not ``r**2 * signal1``).
    At ``r == 1`` (equal levels) this reduces exactly to
    ``K = 2 * signal / (var(d1 - d2) - bias_var_mad)``, the original formula.

    Returns ``None`` when either signal is non-positive, the resulting
    denominator is non-finite or non-positive, or the resulting K is not a
    finite positive number.
    """
    mu1 = float(np.median(d1))
    mu2 = float(np.median(d2))
    signal1 = mu1 - bias_mu
    signal2 = mu2 - bias_mu
    if signal1 <= 0 or signal2 <= 0:
        return None

    r = signal1 / signal2
    diff = (d1 - bias_mu) - r * (d2 - bias_mu)
    var_f = _mad_var(diff)
    denom = var_f - 0.5 * bias_var_mad * (1.0 + r**2)
    if not np.isfinite(denom) or denom <= 0:
        return None

    k = signal1 * (1.0 + r) / denom
    if np.isfinite(k) and k > 0:
        return float(k)
    return None


def _group_key(meta: FrameMeta) -> GainKey | None:
    return _normalise_key(meta.readout_mode, meta.gain_setting, meta.binning)


def measure_gain(
    metas: Sequence[FrameMeta],
    *,
    settings: Settings,
    profile: InstrumentProfile,
    overscan: OverscanGeometry | None = None,
) -> GainMeasurementSet:
    """Measure e-/ADU gain and read noise for every measurable group, per
    amplifier tile.

    Groups FLAT and ZERO (bias) frames by their own ``(readout_mode,
    gain_setting, binning)`` -- never by filter, since bias/dark FILTER
    values are meaningless (see the IMAGETYP/FILTER convention this
    pipeline follows). A group needs at least
    ``settings.calib.gain_min_frames_per_group`` of each kind; groups with
    fewer, or whose flat pairs all fail the saturation/non-linearity or
    positive-variance-excess checks, are silently absent from the returned
    set -- never guessed at. Never raises.

    ``overscan`` supplies the amplifier tile geometry via
    :attr:`~robo43.calib.overscan.OverscanGeometry.science_tile_slices`
    (row-major, 0-based); ``None``, or a geometry with no science blocks,
    yields a single tile spanning the whole frame -- the ROBO43
    single-amplifier case, numerically the ``n_tiles == 1`` instance of the
    same per-tile code path. When ``profile.layout.n_amplifiers`` disagrees
    with the number of tiles actually detected, a warning is logged and the
    detected tile count is used (amplifier geometry is derived at runtime,
    never hardcoded).

    Flats are grouped by ``(filter, night)`` and paired within each group by
    nearest median level (see :func:`_level_matched_pairs`); every pair
    found is used, not just one per group. Each tile's K is the median over
    its own pairs' per-pair estimates (see :func:`_pair_gain_estimate}`); a
    group's frame-level summary (``tile_index == -1``) is the median of its
    tiles' K/read-noise, for header cards and callers that predate per-tile
    measurement. When ``profile`` tabulates a tile gain pattern for a group's
    readout mode, the returned per-tile gains are ``K_run * p_t`` instead (see
    :func:`apply_tile_gain_pattern`); ``K_run`` and every other quantity above
    are unchanged.
    """
    if not settings.calib.gain_measure:
        logger.info("Runtime gain measurement disabled (calib.gain_measure=False).")
        return GainMeasurementSet({})

    tile_slices = _tile_slices(overscan)
    n_tiles = len(tile_slices)
    if profile.layout.n_amplifiers and n_tiles != profile.layout.n_amplifiers:
        logger.warning(
            "detected %d science tile(s) from the overscan geometry, but "
            "the instrument profile declares layout.n_amplifiers=%d; using "
            "the detected tile count (amplifier geometry is derived at "
            "runtime, never hardcoded).",
            n_tiles, profile.layout.n_amplifiers,
        )

    bias_metas_all = select_frames(metas, "bias")
    flat_metas_all = select_frames(metas, "flat")

    groups: dict[GainKey, dict[str, list[FrameMeta]]] = {}
    for meta in bias_metas_all:
        key = _group_key(meta)
        if key is not None:
            groups.setdefault(key, {"bias": [], "flat": []})["bias"].append(meta)
    for meta in flat_metas_all:
        key = _group_key(meta)
        if key is not None:
            groups.setdefault(key, {"bias": [], "flat": []})["flat"].append(meta)

    min_frames = settings.calib.gain_min_frames_per_group
    tol_frac = settings.calib.gain_flat_pair_level_tol_frac
    measurements: dict[GainKey, GainMeasurement] = {}
    tile_measurements: dict[GainKey, tuple[GainMeasurement, ...]] = {}

    for key in sorted(groups):
        readout_mode, gain_setting, binning = key
        bias_group = groups[key]["bias"]
        flat_group = groups[key]["flat"]

        if len(bias_group) < min_frames or len(flat_group) < min_frames:
            logger.info(
                "gain group readout=%s setting=%g binning=%d: %d bias(es), "
                "%d flat(s); need >= %d of each -- no measurement.",
                readout_mode, gain_setting, binning,
                len(bias_group), len(flat_group), min_frames,
            )
            continue

        # --- Per-tile bias statistics, from consecutive bias pairs -------
        bias_tiles = {
            meta: _load_tiles(
                meta, profile=profile, settings=settings, overscan=overscan,
                tile_slices=tile_slices,
            )
            for meta in bias_group
        }

        per_tile_bias_mu: list[list[float]] = [[] for _ in range(n_tiles)]
        per_tile_bias_var_mad: list[list[float]] = [[] for _ in range(n_tiles)]
        per_tile_bias_var_raw: list[list[float]] = [[] for _ in range(n_tiles)]

        for b1, b2 in _consecutive_pairs(bias_group):
            crops1 = bias_tiles[b1]
            crops2 = bias_tiles[b2]
            for t in range(n_tiles):
                d1, d2 = crops1[t], crops2[t]
                if d1.shape != d2.shape:
                    logger.warning(
                        "gain group %s tile %d: bias pair %s/%s shape "
                        "mismatch; skipping pair.", key, t, b1.path, b2.path,
                    )
                    continue
                diff = d1 - d2
                per_tile_bias_mu[t].append(0.5 * (float(np.median(d1)) + float(np.median(d2))))
                per_tile_bias_var_mad[t].append(_mad_var(diff))
                per_tile_bias_var_raw[t].append(_raw_var(diff))

        disagreement_factor = settings.calib.gain_var_disagreement_factor
        tile_bias_mu: list[float | None] = [None] * n_tiles
        tile_bias_var_mad: list[float | None] = [None] * n_tiles
        tile_bias_var_raw: list[float | None] = [None] * n_tiles
        tile_bias_var_ratio: list[float] = [1.0] * n_tiles

        for t in range(n_tiles):
            if not per_tile_bias_mu[t]:
                continue
            mu = float(np.median(per_tile_bias_mu[t]))
            var_mad = float(np.median(per_tile_bias_var_mad[t]))
            var_raw = float(np.median(per_tile_bias_var_raw[t]))
            ratio = var_raw / var_mad if var_mad > 0 else float("inf")
            if not np.isfinite(ratio) or ratio > disagreement_factor or ratio < 1.0 / disagreement_factor:
                logger.warning(
                    "gain group %s tile %d: raw/MAD-based bias variance "
                    "ratio %.3f is outside the %.2fx guard factor -- "
                    "possible cosmic-ray storm, light leak, or (for a "
                    "multi-amplifier instrument) an amplifier boundary "
                    "inside this tile's sampling region; K and read noise "
                    "for this tile may be unreliable.",
                    key, t, ratio, disagreement_factor,
                )
            tile_bias_mu[t] = mu
            tile_bias_var_mad[t] = var_mad
            tile_bias_var_raw[t] = var_raw
            tile_bias_var_ratio[t] = ratio

        if all(mu is None for mu in tile_bias_mu):
            logger.warning("gain group %s: no usable bias pairs on any tile; no measurement.", key)
            continue

        # --- Per-tile K, from level-matched flat pairs within (filter, night) ---
        flat_by_filter_night: dict[tuple[Any, Any], list[FrameMeta]] = {}
        for flat_meta in flat_group:
            flat_by_filter_night.setdefault((flat_meta.filter, flat_meta.night), []).append(
                flat_meta
            )

        per_tile_gains: list[list[float]] = [[] for _ in range(n_tiles)]

        for (_filt, _night), fmetas in flat_by_filter_night.items():
            if len(fmetas) < 2:
                continue
            flat_tiles = {
                meta: _load_tiles(
                    meta, profile=profile, settings=settings, overscan=overscan,
                    tile_slices=tile_slices,
                )
                for meta in fmetas
            }
            levels = [(meta, _frame_level(flat_tiles[meta])) for meta in fmetas]
            for f1, f2 in _level_matched_pairs(levels, tol_frac):
                crops1 = flat_tiles[f1]
                crops2 = flat_tiles[f2]
                for t in range(n_tiles):
                    bias_mu_t = tile_bias_mu[t]
                    if bias_mu_t is None:
                        continue
                    d1, d2 = crops1[t], crops2[t]
                    if d1.shape != d2.shape:
                        logger.warning(
                            "gain group %s tile %d: flat pair %s/%s shape "
                            "mismatch; skipping pair.", key, t, f1.path, f2.path,
                        )
                        continue
                    mu1 = float(np.median(d1))
                    mu2 = float(np.median(d2))
                    if _rejected_by_level(mu1, bias_mu_t, profile) or _rejected_by_level(
                        mu2, bias_mu_t, profile
                    ):
                        logger.warning(
                            "gain group %s tile %d: rejecting saturated/"
                            "non-linear flat pair %s/%s.", key, t, f1.path, f2.path,
                        )
                        continue
                    k = _pair_gain_estimate(d1, d2, bias_mu_t, tile_bias_var_mad[t])
                    if k is not None:
                        per_tile_gains[t].append(k)

        tile_results: list[GainMeasurement | None] = []
        for t in range(n_tiles):
            gains = per_tile_gains[t]
            if not gains or tile_bias_mu[t] is None:
                tile_results.append(None)
                continue
            k_median = float(np.median(gains))
            scatter_frac = (
                float(np.std(gains, ddof=1) / k_median) if len(gains) > 1 and k_median else 0.0
            )
            if scatter_frac > settings.calib.gain_scatter_warn_frac:
                logger.warning(
                    "gain group %s tile %d: relative scatter %.4f across %d "
                    "flat pair(s) exceeds %.4f; K=%.4f e-/ADU may be unreliable.",
                    key, t, scatter_frac, len(gains),
                    settings.calib.gain_scatter_warn_frac, k_median,
                )
            read_noise_e = k_median * math.sqrt(max(tile_bias_var_raw[t] or 0.0, 0.0) / 2.0)
            tile_results.append(
                GainMeasurement(
                    readout_mode=readout_mode,
                    gain_setting=gain_setting,
                    binning=binning,
                    gain_e_per_adu=k_median,
                    read_noise_e=read_noise_e,
                    n_pairs=len(gains),
                    scatter_frac=scatter_frac,
                    source="measured",
                    bias_var_ratio=tile_bias_var_ratio[t],
                    tile_index=t,
                    n_tiles=n_tiles,
                )
            )

        usable_tiles = [tr for tr in tile_results if tr is not None]
        if not usable_tiles:
            logger.warning("gain group %s: no tile produced a usable flat-pair measurement; no measurement.", key)
            continue

        summary = GainMeasurement(
            readout_mode=readout_mode,
            gain_setting=gain_setting,
            binning=binning,
            gain_e_per_adu=float(np.median([tr.gain_e_per_adu for tr in usable_tiles])),
            read_noise_e=float(np.median([tr.read_noise_e for tr in usable_tiles])),
            n_pairs=int(round(float(np.median([tr.n_pairs for tr in usable_tiles])))),
            scatter_frac=float(np.median([tr.scatter_frac for tr in usable_tiles])),
            source="measured",
            bias_var_ratio=float(np.median([tr.bias_var_ratio for tr in usable_tiles])),
            tile_index=-1,
            n_tiles=n_tiles,
        )
        measurements[key] = summary

        # A tile with no usable pairs of its own falls back to the group's
        # frame-level summary rather than leaving a gap: downstream per-tile
        # application always has exactly n_tiles entries to index into.
        tile_measurements[key] = tuple(
            tr if tr is not None else dataclasses.replace(summary, tile_index=t)
            for t, tr in enumerate(tile_results)
        )

        logger.info(
            "measured gain for readout=%s setting=%g binning=%d over %d "
            "tile(s): summary K=%.4f e-/ADU, RN=%.3f e-; per-tile K=%s",
            readout_mode, gain_setting, binning, n_tiles,
            summary.gain_e_per_adu, summary.read_noise_e,
            [round(tr.gain_e_per_adu, 4) if tr is not None else None for tr in tile_results],
        )

    return apply_tile_gain_pattern(GainMeasurementSet(measurements, tile_measurements), profile)
