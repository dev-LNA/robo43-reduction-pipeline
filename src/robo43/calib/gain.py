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
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
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
    "measure_gain",
    "no_measurement_cards",
]

#: Default basename of the cached measurement file written into the calib
#: directory alongside the master bias/dark/flat frames.
DEFAULT_GAIN_MEASUREMENTS_NAME = "gain_measurements.json"

GainKey = tuple[str, float, int]  # (readout_mode, gain_setting, binning)


@dataclass(frozen=True, slots=True)
class GainMeasurement:
    """One measured (readout_mode, gain_setting, binning) group."""

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

    def header_cards(self) -> list[tuple[str, Any, str]]:
        """FITS header cards recording this measurement on a reduced product."""
        return [
            ("GAINVAL", self.gain_e_per_adu, "measured gain (e-/ADU)"),
            ("GAINSRC", self.source, "origin of GAINVAL: measured/none"),
            ("GAINSET", self.gain_setting, "camera GAIN header setting index"),
            ("READOUTM", self.readout_mode, "camera readout mode"),
            ("GAINRN", self.read_noise_e, "measured read noise, e- (from the same bias pairs)"),
            ("GAINNPR", self.n_pairs, "flat pairs used for gain measurement"),
            ("GAINSCT", self.scatter_frac, "relative scatter across gain pairs"),
        ]


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
    """Every gain measurement obtained for one night, keyed by group."""

    measurements: dict[GainKey, GainMeasurement]

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

    def to_json(self) -> list[dict[str, Any]]:
        return [asdict(m) for m in self.measurements.values()]

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
        of every other cached calibration product.
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
        for record in records:
            try:
                measurement = GainMeasurement(**record)
            except Exception as exc:
                logger.warning("skipping malformed gain measurement record %r: %s", record, exc)
                continue
            key = _normalise_key(
                measurement.readout_mode, measurement.gain_setting, measurement.binning
            )
            if key is not None:
                measurements[key] = measurement
        return cls(measurements)


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


def _load_region(
    meta: FrameMeta,
    *,
    profile: InstrumentProfile,
    settings: Settings,
    overscan: OverscanGeometry | None,
) -> np.ndarray:
    """Read one frame, apply overscan/trim exactly as the master builders do,
    then crop to the central sampling region."""
    data, _header = read_image(meta.path, profile)
    if overscan is not None:
        from robo43.calib.overscan import apply_overscan

        trimmed, _cards = apply_overscan(data, overscan, settings=settings)
    else:
        trimmed, _cards = apply_overscan_trim(data, profile=profile, settings=settings)
    return _central_crop(trimmed.astype(np.float64), settings.calib.gain_region_frac)


def _consecutive_pairs(items: Sequence[FrameMeta]) -> list[tuple[FrameMeta, FrameMeta]]:
    """Pair up (0,1), (2,3), ... ; an odd one out is left unpaired."""
    return [(items[i], items[i + 1]) for i in range(0, len(items) - 1, 2)]


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


def _group_key(meta: FrameMeta) -> GainKey | None:
    return _normalise_key(meta.readout_mode, meta.gain_setting, meta.binning)


def measure_gain(
    metas: Sequence[FrameMeta],
    *,
    settings: Settings,
    profile: InstrumentProfile,
    overscan: OverscanGeometry | None = None,
) -> GainMeasurementSet:
    """Measure e-/ADU gain and read noise for every measurable group.

    Groups FLAT and ZERO (bias) frames by their own ``(readout_mode,
    gain_setting, binning)`` -- never by filter, since bias/dark FILTER
    values are meaningless (see the IMAGETYP/FILTER convention this
    pipeline follows). A group needs at least
    ``settings.calib.gain_min_frames_per_group`` of each kind; groups with
    fewer, or whose flat pairs all fail the saturation/non-linearity or
    positive-variance-excess checks, are silently absent from the returned
    set -- never guessed at. Never raises.
    """
    if not settings.calib.gain_measure:
        logger.info("Runtime gain measurement disabled (calib.gain_measure=False).")
        return GainMeasurementSet({})

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
    measurements: dict[GainKey, GainMeasurement] = {}

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

        bias_mus: list[float] = []
        bias_vars_mad: list[float] = []
        bias_vars_raw: list[float] = []
        for b1, b2 in _consecutive_pairs(bias_group):
            d1 = _load_region(b1, profile=profile, settings=settings, overscan=overscan)
            d2 = _load_region(b2, profile=profile, settings=settings, overscan=overscan)
            if d1.shape != d2.shape:
                logger.warning(
                    "gain group %s: bias pair %s/%s shape mismatch; skipping pair.",
                    key, b1.path, b2.path,
                )
                continue
            diff = d1 - d2
            bias_mus.append(0.5 * (float(np.median(d1)) + float(np.median(d2))))
            bias_vars_mad.append(_mad_var(diff))
            bias_vars_raw.append(_raw_var(diff))

        if not bias_mus:
            logger.warning("gain group %s: no usable bias pairs; no measurement.", key)
            continue

        bias_mu = float(np.median(bias_mus))
        # MAD-based: consistent with the flat-pair estimator below, so the
        # robust-estimator bias cancels out of K's denominator (var_f - bias_var_mad).
        bias_var_mad = float(np.median(bias_vars_mad))
        # Unclipped: read noise must count the bias difference's real,
        # heavy non-Gaussian tails, not just its Gaussian core.
        bias_var_raw = float(np.median(bias_vars_raw))

        bias_var_ratio = (
            bias_var_raw / bias_var_mad if bias_var_mad > 0 else float("inf")
        )
        disagreement_factor = settings.calib.gain_var_disagreement_factor
        if (
            not np.isfinite(bias_var_ratio)
            or bias_var_ratio > disagreement_factor
            or bias_var_ratio < 1.0 / disagreement_factor
        ):
            logger.warning(
                "gain group %s: raw/MAD-based bias variance ratio %.3f is "
                "outside the %.2fx guard factor -- possible cosmic-ray storm "
                "or light leak; K and read noise for this group may be "
                "unreliable.", key, bias_var_ratio, disagreement_factor,
            )

        flat_by_filter_exp: dict[tuple[Any, Any], list[FrameMeta]] = {}
        for flat_meta in flat_group:
            flat_by_filter_exp.setdefault(
                (flat_meta.filter, flat_meta.exptime), []
            ).append(flat_meta)

        gains: list[float] = []
        for fmetas in flat_by_filter_exp.values():
            for f1, f2 in _consecutive_pairs(fmetas):
                d1 = _load_region(f1, profile=profile, settings=settings, overscan=overscan)
                d2 = _load_region(f2, profile=profile, settings=settings, overscan=overscan)
                if d1.shape != d2.shape:
                    logger.warning(
                        "gain group %s: flat pair %s/%s shape mismatch; skipping pair.",
                        key, f1.path, f2.path,
                    )
                    continue

                level1 = float(np.median(d1))
                level2 = float(np.median(d2))
                if _rejected_by_level(level1, bias_mu, profile) or _rejected_by_level(
                    level2, bias_mu, profile
                ):
                    logger.warning(
                        "gain group %s: rejecting saturated/non-linear flat pair "
                        "%s/%s.", key, f1.path, f2.path,
                    )
                    continue

                mu_f = 0.5 * (level1 + level2)
                var_f = _mad_var(d1 - d2)
                denom = var_f - bias_var_mad
                if not np.isfinite(denom) or denom <= 0:
                    logger.warning(
                        "gain group %s: flat pair %s/%s has non-positive variance "
                        "excess over the bias pairs; skipping.", key, f1.path, f2.path,
                    )
                    continue

                k = 2.0 * (mu_f - bias_mu) / denom
                if np.isfinite(k) and k > 0:
                    gains.append(k)

        if not gains:
            logger.warning("gain group %s: no usable flat pairs; no measurement.", key)
            continue

        k_median = float(np.median(gains))
        scatter_frac = (
            float(np.std(gains, ddof=1) / k_median) if len(gains) > 1 and k_median else 0.0
        )
        if scatter_frac > settings.calib.gain_scatter_warn_frac:
            logger.warning(
                "gain group %s: relative scatter %.4f across %d flat pair(s) "
                "exceeds %.4f; K=%.4f e-/ADU may be unreliable.",
                key, scatter_frac, len(gains),
                settings.calib.gain_scatter_warn_frac, k_median,
            )

        read_noise_e = k_median * math.sqrt(max(bias_var_raw, 0.0) / 2.0)

        measurements[key] = GainMeasurement(
            readout_mode=readout_mode,
            gain_setting=gain_setting,
            binning=binning,
            gain_e_per_adu=k_median,
            read_noise_e=read_noise_e,
            n_pairs=len(gains),
            scatter_frac=scatter_frac,
            source="measured",
            bias_var_ratio=bias_var_ratio,
        )
        logger.info(
            "measured gain for readout=%s setting=%g binning=%d: K=%.4f e-/ADU, "
            "RN=%.3f e-, %d flat pair(s), scatter=%.4f, bias_var_ratio=%.3f",
            readout_mode, gain_setting, binning,
            k_median, read_noise_e, len(gains), scatter_frac, bias_var_ratio,
        )

    return GainMeasurementSet(measurements)
