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
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robo43.calib.combine import combine_lazy, combine_stack
from robo43.calib.master import MasterFrame
from robo43.config import Settings
from robo43.exceptions import CalibrationError
from robo43.instruments.profile import InstrumentProfile
from robo43.io.fits_io import parse_region, read_image
from robo43.io.meta import FrameMeta

if TYPE_CHECKING:
    from robo43.calib.gain import GainMeasurementSet
    from robo43.calib.overscan import OverscanGeometry
    from robo43.calib.pool import PoolSelection

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


def _bias_pool_cards(pool: PoolSelection, n_frames: int) -> list[tuple[str, Any, str]]:
    """Provenance cards for a window-pooled master bias (see
    :mod:`robo43.calib.pool`).

    ``CALFBACK`` is True when any night other than the science night
    contributed.  The night list goes into ``BIASNGT`` (the card
    :class:`robo43.calib.set.CalibrationSet` reads back) while it fits a FITS
    string card (68 characters); a longer list is instead recorded as
    ``BIASN1`` / ``BIASN2`` (earliest / latest night) and ``BIASNNGT`` (count).
    """
    cards: list[tuple[str, Any, str]] = [
        ("CALFBACK", pool.borrowed, "Calibration borrowed from a previous night"),
    ]
    joined = ",".join(pool.nights)
    if len(joined) <= 68:
        cards.append(("BIASNGT", joined, "Source night(s) of this master bias"))
    else:
        ordered = sorted(pool.nights)
        cards.append(("BIASN1", ordered[0], "Earliest contributing night"))
        cards.append(("BIASN2", ordered[-1], "Latest contributing night"))
        cards.append(("BIASNNGT", len(pool.nights), "Number of contributing nights"))
    cards.append(("BIASNFRM", n_frames, "Bias frames actually combined"))
    cards.append(("BIASWIN", pool.window_nights, "Bias pool window (+/- nights)"))
    cards.append(("BIASTGT", pool.target_frames, "Bias pool target frames"))
    return cards


def build_master_bias(
    metas: Sequence[FrameMeta],
    *,
    settings: Settings,
    profile: InstrumentProfile,
    outdir: str | Path,
    overscan: OverscanGeometry | None = None,
    gain_measurements: GainMeasurementSet | None = None,
    fallback_source_nights: tuple[str, ...] = (),
    pool: PoolSelection | None = None,
) -> MasterFrame:
    """Build and write a master bias frame from ``metas``.

    When the stack would occupy more than ``settings.calib.max_memory_mb``
    (estimated as two float32 copies of every raw frame, as for the flats),
    the trimmed frames are written to a disk-backed ``np.memmap`` scratch file
    in ``outdir`` and combined from there row block by row block with
    :func:`robo43.calib.combine.combine_lazy`; smaller stacks are combined in
    memory exactly as before.  The combined data are identical either way.

    ``pool`` (a :class:`robo43.calib.pool.PoolSelection`) adds the pooling
    provenance cards ``BIASNGT`` / ``BIASNFRM`` / ``BIASWIN`` / ``BIASTGT``
    and replaces the ``fallback_source_nights`` cards.

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

    modes = sorted({(m.readout_mode or "").strip() or "UNKNOWN" for m in bias_metas})
    if len(modes) > 1:
        msg = (
            "refusing to combine bias frames of different readout modes into one "
            f"master bias: {', '.join(modes)}"
        )
        raise CalibrationError(msg)

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    n_bias = len(bias_metas)
    max_bytes = settings.calib.max_memory_mb * 1024.0 * 1024.0
    estimated_bytes = 2.0 * n_bias * bias_metas[0].naxis1 * bias_metas[0].naxis2 * 4.0
    stream = n_bias >= 2 and estimated_bytes > max_bytes

    trimmed_frames: list[np.ndarray] = []  # used only when not streaming
    oscan_cards: list[tuple[str, Any, str]] = []
    scratch_path: Path | None = None
    memmap_arr: np.memmap | None = None
    try:
        for i, meta in enumerate(bias_metas):
            data, _header = read_image(meta.path, profile)
            if overscan is not None:
                from robo43.calib.overscan import apply_overscan
                trimmed, cards = apply_overscan(data, overscan, settings=settings)
            else:
                trimmed, cards = apply_overscan_trim(data, profile=profile, settings=settings)
            del data
            if i == 0:
                oscan_cards = cards
                if stream:
                    scratch_path = outdir_path / f".bias_stream_{os.getpid()}.dat"
                    memmap_arr = np.memmap(
                        scratch_path,
                        mode="w+",
                        dtype=trimmed.dtype,
                        shape=(n_bias, *trimmed.shape),
                    )
            if stream:
                if trimmed.shape != memmap_arr.shape[1:]:
                    msg = (
                        f"frame {i} has shape {trimmed.shape}, expected "
                        f"{memmap_arr.shape[1:]} (all frames must share the same shape)"
                    )
                    raise CalibrationError(msg)
                memmap_arr[i] = trimmed
            else:
                trimmed_frames.append(trimmed)
            del trimmed

        master_data, master_variance, master_weight, combine_cards, gain_cards = _combine_bias(
            bias_metas,
            trimmed_frames=trimmed_frames,
            memmap_arr=memmap_arr if stream else None,
            settings=settings,
            profile=profile,
            gain_measurements=gain_measurements,
        )
    finally:
        if memmap_arr is not None:
            del memmap_arr
        if scratch_path is not None and scratch_path.exists():
            scratch_path.unlink()

    fallback_cards: list[tuple[str, Any, str]]
    if pool is not None:
        fallback_cards = _bias_pool_cards(pool, n_bias)
    else:
        fallback_cards = [
            (
                "CALFBACK",
                bool(fallback_source_nights),
                "Calibration borrowed from a previous night",
            ),
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
        history=(
            [
                f"Master bias pooled from {n_bias} frames, nights "
                + ", ".join(f"{night}({count})" for night, count in pool.night_counts)
            ]
            if pool is not None
            else []
        ),
        bunit="adu",
    )
    master.write(outdir_path / "master_bias.fits", overwrite=settings.runtime.clobber)
    return master


def _combine_bias(
    bias_metas: Sequence[FrameMeta],
    *,
    trimmed_frames: list[np.ndarray],
    memmap_arr: np.memmap | None,
    settings: Settings,
    profile: InstrumentProfile,
    gain_measurements: GainMeasurementSet | None,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    list[tuple[str, Any, str]],
    list[tuple[str, Any, str]],
]:
    """Combine the trimmed bias frames (in memory or from the scratch memmap).

    Returns ``(data, variance, weight, combine_cards, gain_cards)``.
    """
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

    n_frames = len(bias_metas)
    if n_frames >= 2:
        if memmap_arr is not None:
            memmap_arr.flush()

            def _make_loader(idx: int):
                def loader(slices):
                    rows, cols = slices
                    return np.asarray(memmap_arr[idx, rows, cols])

                return loader

            result = combine_lazy(
                [_make_loader(i) for i in range(n_frames)],
                tuple(memmap_arr.shape[1:]),
                settings.calib,
                read_noise_e=read_noise_e,
                gain_e_per_adu=gain_e_per_adu,
            )
        else:
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

    return master_data, master_variance, master_weight, combine_cards, gain_cards
