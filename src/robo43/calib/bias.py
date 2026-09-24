"""Master bias construction.

Legacy ``src/make_calibration_frames.py`` bias building (lines 88-101)
collapses the stack with ``np.median`` before sigma-clipping spatially; this
module instead delegates the actual combination to
:func:`robo43.calib.combine.combine_stack`, which clips across frames per
pixel.  It also fixes the missing overscan/trim handling entirely absent from
the legacy bias path.

Interpretation note: ``settings`` here is the *full* ``Settings`` tree (not
just ``CalibSettings``), because ``settings.runtime.clobber`` must be honoured
by these builders and ``RuntimeSettings`` lives outside ``CalibSettings``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robo43.calib.combine import combine_stack
from robo43.calib.master import MasterFrame
from robo43.config import Settings
from robo43.exceptions import CalibrationError
from robo43.instruments.profile import InstrumentProfile
from robo43.io.fits_io import parse_region, read_image
from robo43.io.meta import FrameMeta

if TYPE_CHECKING:
    from robo43.calib.gain import GainMeasurementSet
    from robo43.calib.overscan import OverscanGeometry

logger = logging.getLogger(__name__)

__all__ = [
    "apply_overscan_trim",
    "build_master_bias",
    "resolve_common_gain",
    "select_frames",
]


def select_frames(metas: Sequence[FrameMeta], frame_type: str) -> list[FrameMeta]:
    """Return the metas whose ``frame_type`` matches, input order preserved."""
    return [meta for meta in metas if meta.frame_type == frame_type]


def resolve_common_gain(
    metas: Sequence[FrameMeta], *, profile: InstrumentProfile  # noqa: ARG001
) -> tuple[float, str]:
    """Resolve one gain *setting* shared by every frame in ``metas``.

    ``gain_setting`` is the raw, dimensionless ``GAIN`` header index (10 or
    30 on the real QHY600M data this pipeline targets) -- never an e-/ADU
    conversion factor, which this function has nothing to do with. It is
    resolved per-frame by :func:`robo43.io.headers.read_meta` straight from
    the header. Combining frames shot at different gain *settings* into one
    master must not silently paper over the difference, so this only checks
    consistency; the physical e-/ADU value used to build the master (and
    recorded on it) comes from :mod:`robo43.calib.gain`'s runtime
    measurement, looked up separately by the caller.

    ``profile`` is accepted for signature symmetry with the rest of this
    module's builders but is not consulted: there is no profile fallback for
    a per-frame setting index.

    Returns ``(gain_setting, origin)`` where ``origin`` is ``"header"`` or
    ``"alias"`` -- the :class:`~robo43.io.meta.Source` of the meta that
    supplied it (the first one, when several metas agree). Raises
    :class:`CalibrationError` when two metas disagree on the setting
    (relative tolerance ``1e-6``), naming the conflicting frames, rather
    than silently combining across settings; also raises when no meta
    carries a gain setting at all.
    """
    known = [
        (m.path, m.gain_setting, m.origin("gain_setting"))
        for m in metas
        if m.gain_setting is not None
    ]
    if not known:
        msg = "no gain-setting value available from any input frame header"
        raise CalibrationError(msg)

    reference = known[0][1]
    for _path, setting, _origin in known[1:]:
        if not math.isclose(setting, reference, rel_tol=1e-6):
            detail = ", ".join(f"{p.name}={g:g}" for p, g, _ in known)
            msg = f"inconsistent gain setting among combined frames: {detail}"
            raise CalibrationError(msg)

    first_origin = known[0][2]
    origin_label = first_origin.source.value if first_origin is not None else "header"
    return reference, origin_label


def apply_overscan_trim(
    data: np.ndarray,
    *,
    profile: InstrumentProfile,
    settings: Settings,
) -> tuple[np.ndarray, list[tuple[str, Any, str]]]:
    """Subtract the overscan median, then trim, per ``profile.layout``.

    Each step is independently optional (a ``None`` region skips it) and its
    outcome is always recorded as a header card.
    """
    cards: list[tuple[str, Any, str]] = []
    result = data

    overscan_region = profile.layout.overscan
    if settings.calib.overscan_correct and overscan_region is not None:
        row_slice, col_slice = parse_region(overscan_region)
        level = float(np.median(result[row_slice, col_slice]))
        result = result - level
        cards.append(("OSCANCOR", True, "Overscan correction applied"))
        cards.append(("OSCANLVL", level, "Overscan median level (ADU)"))
        cards.append(("OSCANREG", overscan_region, "Overscan region (IRAF)"))
    else:
        cards.append(("OSCANCOR", False, "Overscan correction applied"))

    trim_region = profile.layout.trim
    if trim_region is not None:
        row_slice, col_slice = parse_region(trim_region)
        result = result[row_slice, col_slice]
        cards.append(("TRIMMED", True, "Frame trimmed"))
        cards.append(("TRIMREG", trim_region, "Trim region (IRAF)"))
    else:
        cards.append(("TRIMMED", False, "Frame trimmed"))

    return result, cards


def build_master_bias(
    metas: Sequence[FrameMeta],
    *,
    settings: Settings,
    profile: InstrumentProfile,
    outdir: str | Path,
    overscan: OverscanGeometry | None = None,
    gain_measurements: GainMeasurementSet | None = None,
    fallback_source_nights: tuple[str, ...] = (),
) -> MasterFrame:
    """Build and write a master bias frame from ``metas``.

    ``gain_measurements`` is the runtime PTC measurement for the whole night
    (see :mod:`robo43.calib.gain`); the e-/ADU value and read noise recorded
    on this master, and used for its read-noise-floor variance, are looked
    up from it for this master's own (readout_mode, gain_setting, binning)
    and are ``None`` when that group has no measurement -- never a profile
    fallback.
    """
    bias_metas = select_frames(metas, "bias")
    if not bias_metas:
        msg = "no bias frames found to build a master bias"
        raise CalibrationError(msg)

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    trimmed_frames: list[np.ndarray] = []
    oscan_cards: list[tuple[str, Any, str]] = []
    for i, meta in enumerate(bias_metas):
        data, _header = read_image(meta.path, profile)
        if overscan is not None:
            from robo43.calib.overscan import apply_overscan
            trimmed, cards = apply_overscan(data, overscan, settings=settings)
        else:
            trimmed, cards = apply_overscan_trim(data, profile=profile, settings=settings)
        trimmed_frames.append(trimmed)
        if i == 0:
            oscan_cards = cards

    gain_setting, _gain_setting_origin = resolve_common_gain(bias_metas, profile=profile)
    readout_mode = next(
        (m.readout_mode for m in bias_metas if m.readout_mode is not None), None
    )
    binning = next((m.binning for m in bias_metas if m.binning is not None), 1)
    measurement = (
        gain_measurements.get(readout_mode, gain_setting, binning)
        if gain_measurements is not None
        else None
    )
    gain_e_per_adu = measurement.gain_e_per_adu if measurement is not None else None
    read_noise_e = measurement.read_noise_e if measurement is not None else profile.read_noise_e

    if len(trimmed_frames) >= 2:
        result = combine_stack(
            trimmed_frames,
            settings.calib,
            read_noise_e=read_noise_e,
            gain_e_per_adu=gain_e_per_adu,
        )
        combine_cards = result.header_cards()
        master_data = result.data
        master_variance = result.variance
        master_weight = result.weight
    else:
        master_data = trimmed_frames[0].astype(np.float32)
        master_variance = None
        master_weight = None
        combine_cards = [("NCOMBINE", 1, "Number of input frames")]

    if measurement is not None:
        gain_cards = measurement.header_cards()
    else:
        from robo43.calib.gain import no_measurement_cards

        gain_cards = no_measurement_cards(gain_setting, readout_mode)

    fallback_cards: list[tuple[str, Any, str]] = [
        ("CALFBACK", bool(fallback_source_nights), "Calibration borrowed from a previous night"),
    ]
    if fallback_source_nights:
        fallback_cards.append(
            ("BIASNGT", ",".join(fallback_source_nights), "Source night(s) for borrowed bias")
        )

    header_cards = oscan_cards + combine_cards + gain_cards + fallback_cards

    master = MasterFrame(
        data=master_data,
        variance=master_variance,
        weight=master_weight,
        header_cards=header_cards,
        bunit="adu",
    )
    master.write(outdir_path / "master_bias.fits", overwrite=settings.runtime.clobber)
    return master
