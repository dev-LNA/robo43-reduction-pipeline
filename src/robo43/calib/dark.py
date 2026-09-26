"""Master dark construction, dark-current mapping, and dark selection/scaling.

Legacy ``src/make_calibration_frames.py`` has only a stub for dark support.
This module groups dark frames by (exptime, ccd_temp), builds one master per
group via :func:`robo43.calib.combine.combine_stack`, derives a composite
dark-current map (e-/s/pixel), and provides the scaling logic used by
per-frame reduction to pick and scale the best available dark for a science
frame's exposure time and temperature.

Interpretation note: ``settings`` here is the *full* ``Settings`` tree (not
just ``CalibSettings``), because ``settings.runtime.clobber`` must be
honoured by these builders and ``RuntimeSettings`` lives outside
``CalibSettings``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robo43.calib.bias import apply_overscan_trim, resolve_common_gain, select_frames
from robo43.calib.combine import combine_stack
from robo43.calib.master import MasterFrame
from robo43.config import Settings
from robo43.exceptions import CalibrationError
from robo43.instruments.profile import InstrumentProfile
from robo43.io.fits_io import read_image
from robo43.io.meta import FrameMeta

if TYPE_CHECKING:
    from robo43.calib.gain import GainMeasurementSet
    from robo43.calib.overscan import OverscanGeometry

logger = logging.getLogger(__name__)

__all__ = [
    "DarkEntry",
    "DarkSelection",
    "build_master_darks",
    "scale_dark",
    "select_dark",
]


@dataclass(frozen=True, slots=True)
class DarkEntry:
    path: Path
    exptime: float
    ccd_temp: float | None
    data: np.ndarray | None
    variance: np.ndarray | None
    n_combine: int


@dataclass(frozen=True, slots=True)
class DarkSelection:
    entry: DarkEntry | None
    scale: float
    method: str
    reason: str


def _group_darks(
    dark_metas: Sequence[FrameMeta], settings: Settings
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for meta in dark_metas:
        placed = False
        for g in groups:
            exp_ok = (
                abs(meta.exptime - g["anchor_exp"])
                <= settings.calib.dark_exptime_tolerance_s
            )
            temp_ok = (meta.ccd_temp is None and g["anchor_temp"] is None) or (
                meta.ccd_temp is not None
                and g["anchor_temp"] is not None
                and abs(meta.ccd_temp - g["anchor_temp"])
                <= settings.calib.dark_temp_tolerance_c
            )
            if exp_ok and temp_ok:
                g["metas"].append(meta)
                placed = True
                break
        if not placed:
            groups.append(
                {
                    "anchor_exp": meta.exptime,
                    "anchor_temp": meta.ccd_temp,
                    "metas": [meta],
                }
            )
    return groups


def build_master_darks(
    metas: Sequence[FrameMeta],
    *,
    settings: Settings,
    profile: InstrumentProfile,
    outdir: str | Path,
    bias: MasterFrame | None = None,
    overscan: OverscanGeometry | None = None,
    gain_measurements: GainMeasurementSet | None = None,
    fallback_source_nights: tuple[str, ...] = (),
) -> tuple[dict[tuple[float, float | None], MasterFrame], MasterFrame]:
    """Build and write one master dark per (exptime, ccd_temp) group, plus a
    composite dark-current map.

    ``gain_measurements`` is the night's runtime PTC measurement (see
    :mod:`robo43.calib.gain`); each group's e-/ADU value is looked up from it
    by that group's own (readout_mode, gain_setting, binning) and is
    ``None`` when unmeasured -- never a profile fallback. The composite
    dark-current map (e-/s) needs a numeric gain to convert from ADU/s; a
    group with no measurement falls back to 1.0 (i.e. contributes an
    ADU/s-equivalent, not a true e-/s current) and is logged as such, rather
    than raising or silently invoking the profile.

    ``overscan`` is the runtime-detected geometry used for the master bias;
    when given, darks are trimmed with it so they match the bias shape.
    """
    dark_metas_all = select_frames(metas, "dark")
    if not dark_metas_all:
        msg = "no dark frames found to build master darks"
        raise CalibrationError(msg)

    dark_metas: list[FrameMeta] = []
    for meta in dark_metas_all:
        if meta.exptime is None:
            logger.warning("skipping dark frame with no exptime: %s", meta.path)
            continue
        dark_metas.append(meta)

    if not dark_metas:
        msg = "no dark frames with a valid exptime found"
        raise CalibrationError(msg)

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    groups = _group_darks(dark_metas, settings)

    masters: dict[tuple[float, float | None], MasterFrame] = {}
    group_currents: list[np.ndarray] = []
    group_n_combines: list[int] = []
    group_temps: list[float] = []
    group_temp_n_combines: list[int] = []

    for g in groups:
        group_metas: list[FrameMeta] = g["metas"]
        gain_setting, _gain_setting_origin = resolve_common_gain(group_metas, profile=profile)
        readout_mode = next(
            (m.readout_mode for m in group_metas if m.readout_mode is not None), None
        )
        binning = next((m.binning for m in group_metas if m.binning is not None), 1)
        measurement = (
            gain_measurements.get(readout_mode, gain_setting, binning)
            if gain_measurements is not None
            else None
        )
        gain_e_per_adu = measurement.gain_e_per_adu if measurement is not None else None
        read_noise_e = (
            measurement.read_noise_e if measurement is not None else profile.read_noise_e
        )

        frames: list[np.ndarray] = []
        for meta in group_metas:
            data, _header = read_image(meta.path, profile)
            if overscan is not None:
                from robo43.calib.overscan import apply_overscan

                trimmed, _cards = apply_overscan(data, overscan, settings=settings)
            else:
                trimmed, _cards = apply_overscan_trim(data, profile=profile, settings=settings)
            if bias is not None:
                if trimmed.shape != bias.data.shape:
                    msg = (
                        f"dark frame {meta.path} shape {trimmed.shape} does not "
                        f"match master bias shape {bias.data.shape}"
                    )
                    raise CalibrationError(msg)
                trimmed = trimmed - bias.data
            frames.append(trimmed)

        if len(frames) >= 2:
            result = combine_stack(
                frames,
                settings.calib,
                read_noise_e=read_noise_e,
                gain_e_per_adu=gain_e_per_adu,
            )
            group_data = result.data
            group_variance = result.variance
            n_combine = result.n_input
            combine_cards = result.header_cards()
        else:
            group_data = frames[0].astype(np.float32)
            group_variance = None
            n_combine = 1
            combine_cards = [("NCOMBINE", 1, "Number of input frames")]

        mean_exp = float(np.mean([m.exptime for m in group_metas]))
        temps = [m.ccd_temp for m in group_metas if m.ccd_temp is not None]
        mean_temp: float | None = float(np.mean(temps)) if temps else None

        header_cards: list[tuple[str, Any, str]] = [
            ("EXPTIME", mean_exp, "Mean exposure time (s)"),
        ]
        if mean_temp is not None:
            header_cards.append(("CCDTEMP", mean_temp, "Mean CCD temperature (C)"))
        bias_filename = (
            bias.path.name if bias is not None and bias.path is not None else "NONE"
        )
        header_cards.append(("BIASFIL", bias_filename, "Master bias applied"))
        if measurement is not None:
            header_cards.extend(measurement.header_cards())
        else:
            from robo43.calib.gain import no_measurement_cards

            header_cards.extend(no_measurement_cards(gain_setting, readout_mode))
        header_cards.extend(combine_cards)
        header_cards.append(
            ("CALFBACK", bool(fallback_source_nights), "Calibration borrowed from a previous night")
        )
        if fallback_source_nights:
            header_cards.append(
                ("DARKNGT", ",".join(fallback_source_nights), "Source night(s) for borrowed dark")
            )

        group_master = MasterFrame(
            data=group_data,
            variance=group_variance,
            header_cards=header_cards,
            bunit="adu",
        )

        temp_str = f"{mean_temp:.1f}" if mean_temp is not None else "unknown"
        group_path = outdir_path / f"master_dark_{mean_exp:g}s_{temp_str}C.fits"
        group_master.write(group_path, overwrite=settings.runtime.clobber)

        masters[(mean_exp, mean_temp)] = group_master

        if gain_e_per_adu is None:
            logger.warning(
                "dark group readout=%s setting=%g: no measured gain; "
                "dark-current map contribution for this group is an "
                "ADU/s-equivalent, not true e-/s.",
                readout_mode, gain_setting,
            )
        current_e_per_s = (group_data * (gain_e_per_adu if gain_e_per_adu is not None else 1.0)) / mean_exp
        group_currents.append(current_e_per_s)
        group_n_combines.append(n_combine)
        if mean_temp is not None:
            group_temps.append(mean_temp)
            group_temp_n_combines.append(n_combine)

        del frames
        if settings.runtime.release_memory:
            from robo43.memory import release_memory

            release_memory(f"dark:{mean_exp:g}s_{temp_str}")

    composite = np.average(
        np.stack(group_currents), axis=0, weights=group_n_combines
    ).astype(np.float32)

    dctref: float | None = None
    if group_temps:
        dctref = float(np.average(group_temps, weights=group_temp_n_combines))

    dc_cards: list[tuple[str, Any, str]] = []
    if dctref is not None:
        dc_cards.append(
            ("DCTREF", dctref, "Reference temp of dark-current map (C)")
        )
    dc_cards.append(("NGROUPS", len(groups), "Number of exptime/temp groups combined"))

    dark_current_master = MasterFrame(
        data=composite,
        header_cards=dc_cards,
        bunit="electron/s",
    )
    dark_current_master.write(
        outdir_path / "dark_current_map.fits", overwrite=settings.runtime.clobber
    )

    return masters, dark_current_master


def select_dark(
    darks: Sequence[DarkEntry],
    dark_current_map: np.ndarray | None,
    dark_current_ref_temp: float | None,
    exptime: float,
    ccd_temp: float | None,
    *,
    settings: Settings,
    profile: InstrumentProfile,
    gain: float | None = None,
) -> DarkSelection:
    """Choose the best available master dark for a target exptime/temperature.

    ``gain`` is the *target* frame's own measured e-/ADU value (from
    :mod:`robo43.calib.gain`, looked up by the caller for this frame's own
    (readout_mode, gain_setting, binning)) -- used only in the
    ``current_map`` branch, to convert the physical electron/s dark-current
    map back into the ADU units of the frame it will be subtracted from.
    There is no profile fallback: when the frame's group has no
    measurement, ``gain`` is ``None`` and the conversion instead treats the
    map as ADU/s-equivalent (see the warning logged where this is applied).
    """
    for entry in darks:
        exp_ok = abs(entry.exptime - exptime) <= settings.calib.dark_exptime_tolerance_s
        temp_ok = (ccd_temp is None and entry.ccd_temp is None) or (
            ccd_temp is not None
            and entry.ccd_temp is not None
            and abs(entry.ccd_temp - ccd_temp) <= settings.calib.dark_temp_tolerance_c
        )
        if exp_ok and temp_ok:
            reason = (
                f"exact match: master dark at exptime={entry.exptime}s, "
                f"ccd_temp={entry.ccd_temp} matches target exptime={exptime}s, "
                f"ccd_temp={ccd_temp} within tolerance"
            )
            return DarkSelection(entry=entry, scale=1.0, method="exact", reason=reason)

    for entry in darks:
        temp_ok = (ccd_temp is None and entry.ccd_temp is None) or (
            ccd_temp is not None
            and entry.ccd_temp is not None
            and abs(entry.ccd_temp - ccd_temp) <= settings.calib.dark_temp_tolerance_c
        )
        if temp_ok:
            scale = exptime / entry.exptime if entry.exptime else 1.0
            reason = (
                f"scaled match: master dark at exptime={entry.exptime}s, "
                f"ccd_temp={entry.ccd_temp} matches temperature but not exposure "
                f"time; scaling by {scale:.4f} to reach target exptime={exptime}s"
            )
            return DarkSelection(entry=entry, scale=scale, method="scaled", reason=reason)

    if dark_current_map is not None:
        if (
            dark_current_ref_temp is not None
            and ccd_temp is not None
            and abs(ccd_temp - dark_current_ref_temp)
            > settings.calib.dark_temp_extrap_max_c
        ):
            reason = (
                f"refused: target ccd_temp={ccd_temp} is more than "
                f"{settings.calib.dark_temp_extrap_max_c}C from the dark-current "
                f"map reference temperature {dark_current_ref_temp}; extrapolation "
                "too large, applying bias-only correction"
            )
            return DarkSelection(entry=None, scale=0.0, method="none", reason=reason)

        if gain is None:
            logger.warning(
                "no measured gain for this frame's group; converting the "
                "dark-current map assuming it is already ADU/s-equivalent."
            )
        effective_gain = gain if gain is not None else 1.0
        map_adu_per_s = dark_current_map / effective_gain
        wrapper = DarkEntry(
            path=Path("dark_current_map.fits"),
            exptime=1.0,
            ccd_temp=dark_current_ref_temp,
            data=map_adu_per_s.astype(np.float32),
            variance=None,
            n_combine=0,
        )
        reason = (
            f"current_map: no master dark matches target exptime={exptime}s, "
            f"ccd_temp={ccd_temp}; falling back to the dark-current map "
            f"(reference temp={dark_current_ref_temp})"
        )
        return DarkSelection(entry=wrapper, scale=1.0, method="current_map", reason=reason)

    return DarkSelection(
        entry=None,
        scale=0.0,
        method="none",
        reason="no master dark and no dark-current map available; bias-only",
    )


def scale_dark(
    selection: DarkSelection,
    exptime: float,
    ccd_temp: float | None,
    *,
    settings: Settings,
) -> tuple[np.ndarray, list[tuple[str, Any, str]]]:
    """Scale ``selection.entry`` to the target exptime/temperature."""
    if selection.method == "none" or selection.entry is None:
        cards: list[tuple[str, Any, str]] = [
            ("DARKFIL", "NONE", "No usable master dark applied"),
            ("DARKMETH", "none", "Dark selection method"),
        ]
        return np.zeros((), dtype=np.float32), cards

    entry = selection.entry
    if entry.data is None:
        msg = f"dark selection {selection.method!r} has no data to scale"
        raise CalibrationError(msg)

    if selection.method == "exact":
        scale = 1.0
        data = entry.data
    elif selection.method == "scaled":
        scale = exptime / entry.exptime if entry.exptime else 1.0
        data = entry.data * scale
    elif selection.method == "current_map":
        exp_scale = exptime / entry.exptime if entry.exptime else exptime
        if ccd_temp is not None and entry.ccd_temp is not None:
            temp_factor = 2.0 ** (
                (ccd_temp - entry.ccd_temp) / settings.calib.dark_doubling_temp_c
            )
        else:
            temp_factor = 1.0
        scale = exp_scale * temp_factor
        data = entry.data * scale
    else:
        msg = f"unknown dark selection method {selection.method!r}"
        raise CalibrationError(msg)

    cards = [
        (
            "DARKFIL",
            entry.path.name if entry.path else "NONE",
            "Master dark applied",
        ),
        ("DARKMETH", selection.method, "Dark selection method"),
        ("DARKSCL", float(scale), "Dark scaling factor applied"),
    ]
    return data.astype(np.float32), cards
