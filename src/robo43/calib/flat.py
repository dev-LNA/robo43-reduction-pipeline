"""Master flat-field construction.

Per-filter master flats are built by combining and normalizing calibration flat
frames. The ``separate_dome_and_sky_flats`` setting was previously unspecified;
this module adopts the semantics that when enabled, flats whose OBJECT field
(lower-cased) contains "sky" or "twilight" are grouped separately from "dome"
flats under the same filter, allowing independent masters per illumination source.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robo43.calib.bias import apply_overscan_trim, resolve_common_gain, select_frames
from robo43.calib.combine import combine_lazy, combine_stack
from robo43.calib.master import MasterFrame
from robo43.config import Settings
from robo43.exceptions import CalibrationError
from robo43.instruments.profile import InstrumentProfile
from robo43.io.fits_io import read_image
from robo43.io.meta import FrameMeta
from robo43.products.dq import DQ, DQ_DTYPE, FATAL_FLAGS

if TYPE_CHECKING:
    from robo43.calib.gain import GainMeasurementSet
    from robo43.calib.masks import MaskSet
    from robo43.calib.overscan import OverscanGeometry

logger = logging.getLogger(__name__)

__all__ = ["build_master_flats", "group_flats_by_filter", "normalise_flat"]


def group_flats_by_filter(
    metas: Sequence[FrameMeta], *, separate_dome_and_sky: bool = True
) -> dict[str, list[FrameMeta]]:
    """Group flat frames by filter, optionally separating dome and sky flats.

    Returns a dict mapping group keys to lists of FrameMeta objects in input
    order. When ``separate_dome_and_sky`` is True, flats whose ``object``
    (lower-cased) contains "sky" or "twilight" are grouped to key
    ``"{filter}:sky"``; all others to ``"{filter}:dome"``. When False, all
    flats of a filter go to the same key regardless of object. Flats with
    ``filter is None`` are skipped with a warning.
    """
    flat_metas = select_frames(metas, "flat")

    groups: dict[str, list[FrameMeta]] = {}
    for meta in flat_metas:
        if meta.filter is None:
            logger.warning("Skipping flat frame %s with filter=None", meta.path)
            continue

        if separate_dome_and_sky:
            obj_lower = (meta.object or "").lower()
            if "sky" in obj_lower or "twilight" in obj_lower:
                key = f"{meta.filter}:sky"
            else:
                key = f"{meta.filter}:dome"
        else:
            key = meta.filter

        if key not in groups:
            groups[key] = []
        groups[key].append(meta)

    return groups


def normalise_flat(
    data: np.ndarray, settings
) -> tuple[np.ndarray, list[tuple[str, Any, str]]]:
    """Normalize a flat frame to a median of 1.0 in a central region.

    Computes the normalization level as the median of data in a centred
    region covering ``settings.flat_norm_region_frac`` of each axis.
    Raises CalibrationError if the level is not finite or <= 0.

    Returns the normalized data and header cards recording the level and region.
    """
    h, w = data.shape
    frac = settings.flat_norm_region_frac

    # Compute the central region, at least 1 pixel per axis
    row_half = max(1, int(h * frac / 2))
    col_half = max(1, int(w * frac / 2))

    r0 = h // 2 - row_half
    r1 = h // 2 + row_half
    c0 = w // 2 - col_half
    c1 = w // 2 + col_half

    level = float(np.median(data[r0:r1, c0:c1]))

    if not np.isfinite(level) or level <= 0:
        msg = f"flat normalization level not finite or <= 0: {level}"
        raise CalibrationError(msg)

    normalized = data / level

    cards = [
        ("FLATNORM", level, "Flat normalisation level (ADU)"),
        ("FLATNREG", frac, "Central fraction used for normalisation"),
    ]

    return normalized, cards


def build_master_flats(
    metas: Sequence[FrameMeta],
    *,
    settings: Settings,
    profile: InstrumentProfile,
    outdir: str | Path,
    bias: MasterFrame | None = None,
    dark_current_map: np.ndarray | None = None,
    mask: MaskSet | None = None,
    overscan: OverscanGeometry | None = None,
    gain_measurements: GainMeasurementSet | None = None,
) -> dict[str, MasterFrame]:
    """Build and write master flat frames, one per filter (and illumination source).

    Processes each group in sorted key order:
    1. Reads and trims each frame (recording overscan cards from the first)
    2. Subtracts bias if provided
    3. Subtracts dark current if provided and exptime is known
    4. Applies level cut: accepts only frames with median in
       [flat_min_level_frac, flat_max_level_frac] * saturation_adu
    5. Combines accepted frames (or uses the single frame if n==1)
    6. Normalizes the combined data
    7. Applies mask and sets fatal pixels to 1.0
    8. Writes master frame to outdir/master_flat_<key>.fits (: -> _)

    Returns a dict mapping group keys to the written MasterFrame objects.
    Raises CalibrationError if no flat frames are found or if a group has
    no frames surviving the level cut.
    """
    separate_dome_and_sky = settings.calib.separate_dome_and_sky_flats
    groups = group_flats_by_filter(metas, separate_dome_and_sky=separate_dome_and_sky)
    if not groups:
        msg = "no flat frames found to build master flats"
        raise CalibrationError(msg)

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    masters: dict[str, MasterFrame] = {}

    for group_key in sorted(groups.keys()):
        group_metas = groups[group_key]
        logger.info("Processing flat group %s with %d frames", group_key, len(group_metas))

        # Read and trim all frames
        trimmed_frames: list[np.ndarray] = []
        oscan_cards: list[tuple[str, Any, str]] = []
        for i, meta in enumerate(group_metas):
            data, _header = read_image(meta.path, profile)
            if overscan is not None:
                from robo43.calib.overscan import apply_overscan
                trimmed, cards = apply_overscan(data, overscan, settings=settings)
            else:
                trimmed, cards = apply_overscan_trim(data, profile=profile, settings=settings)
            trimmed_frames.append(trimmed)
            if i == 0:
                oscan_cards = cards

        # Subtract bias and dark current, compute medians for level cut
        processed_frames: list[np.ndarray] = []
        frame_medians: list[float] = []
        bias_cards: list[tuple[str, Any, str]] = []
        dark_cards: list[tuple[str, Any, str]] = []

        for i, (trimmed, meta) in enumerate(zip(trimmed_frames, group_metas, strict=True)):
            working = trimmed.astype(np.float64)

            if bias is not None:
                if working.shape != bias.data.shape:
                    msg = (
                        f"bias shape {bias.data.shape} does not match trimmed frame "
                        f"shape {working.shape}"
                    )
                    raise CalibrationError(msg)
                working = working - bias.data
                if i == 0:
                    bias_basename = Path(bias.path).name if bias.path else "NONE"
                    bias_cards.append(("BIASFIL", bias_basename, "Master bias applied"))

            if dark_current_map is not None and meta.exptime is not None:
                working = working - (dark_current_map * meta.exptime)
                if i == 0:
                    dark_cards.append(
                        ("DARKFIL", "applied", "Dark current scaling applied")
                    )

            if i == 0 and bias is None:
                bias_cards.append(("BIASFIL", "NONE", "No bias subtraction"))
            if i == 0 and dark_current_map is None:
                dark_cards.append(("DARKFIL", "NONE", "No dark current subtraction"))

            med = float(np.median(working))
            frame_medians.append(med)
            processed_frames.append(working.astype(np.float32))

        # Apply level cut
        level_cut_applied = True
        if profile.saturation_adu is not None:
            min_level = settings.calib.flat_min_level_frac * profile.saturation_adu
            max_level = settings.calib.flat_max_level_frac * profile.saturation_adu

            accepted_frames: list[np.ndarray] = []
            accepted_indices: list[int] = []
            for i, (frame, med) in enumerate(zip(processed_frames, frame_medians, strict=True)):
                if min_level <= med <= max_level:
                    accepted_frames.append(frame)
                    accepted_indices.append(i)
                else:
                    logger.warning(
                        "Rejecting flat frame %s (median=%.2f, range=[%.2f, %.2f])",
                        group_metas[i].path,
                        med,
                        min_level,
                        max_level,
                    )
        else:
            level_cut_applied = False
            logger.warning(
                "Skipping level cut for group %s: profile declares no saturation level",
                group_key,
            )
            accepted_frames = processed_frames
            accepted_indices = list(range(len(processed_frames)))

        n_accepted = len(accepted_frames)
        n_rejected = len(processed_frames) - n_accepted
        level_cut_cards = [
            ("FLATNACC", n_accepted, "Flat frames accepted after level cut"),
            ("FLATNREJ", n_rejected, "Flat frames rejected by level cut"),
            ("FLATLCUT", level_cut_applied, "Flat level cut applied"),
        ]

        if n_accepted == 0:
            logger.warning("No frames in group %s survive level cut; skipping", group_key)
            continue

        accepted_metas = [group_metas[i] for i in accepted_indices]
        gain_setting, _gain_setting_origin = resolve_common_gain(accepted_metas, profile=profile)
        readout_mode = next(
            (m.readout_mode for m in accepted_metas if m.readout_mode is not None), None
        )
        binning = next((m.binning for m in accepted_metas if m.binning is not None), 1)
        measurement = (
            gain_measurements.get(readout_mode, gain_setting, binning)
            if gain_measurements is not None
            else None
        )
        gain_e_per_adu = measurement.gain_e_per_adu if measurement is not None else None
        read_noise_e = (
            measurement.read_noise_e if measurement is not None else profile.read_noise_e
        )
        if measurement is not None:
            gain_cards = measurement.header_cards()
        else:
            from robo43.calib.gain import no_measurement_cards

            gain_cards = no_measurement_cards(gain_setting, readout_mode)

        # Combine frames
        if n_accepted >= 2:
            # Check if we need lazy loading
            frame_size_bytes = n_accepted * accepted_frames[0].size * 4  # float32
            max_bytes = settings.calib.max_memory_mb * 1024.0 * 1024.0

            if frame_size_bytes > max_bytes:
                # Use lazy loading
                # Create loaders for each accepted frame
                def make_loader(frame_data):
                    def loader(slices):
                        rows, cols = slices
                        return frame_data[rows, cols]
                    return loader

                loaders = [make_loader(frame) for frame in accepted_frames]
                result = combine_lazy(
                    loaders,
                    accepted_frames[0].shape,
                    settings.calib,
                    read_noise_e=read_noise_e,
                    gain_e_per_adu=gain_e_per_adu,
                )
            else:
                # Use stack combination
                result = combine_stack(
                    accepted_frames,
                    settings.calib,
                    read_noise_e=read_noise_e,
                    gain_e_per_adu=gain_e_per_adu,
                )

            combine_cards = result.header_cards()
            master_data = result.data
            master_variance = result.variance
            master_weight = result.weight
        else:
            # Single frame
            master_data = accepted_frames[0].astype(np.float32)
            master_variance = None
            master_weight = None
            combine_cards = [("NCOMBINE", 1, "Number of input frames")]

        # Normalize the flat
        normalized_data, norm_cards = normalise_flat(master_data, settings.calib)

        # Scale variance by level squared
        if master_variance is not None:
            norm_level = norm_cards[0][1]
            normalized_variance = master_variance / (norm_level**2)
        else:
            normalized_variance = None

        # Apply mask
        master_mask = None
        mask_cards: list[tuple[str, Any, str]] = []

        if mask is not None:
            # Start with the DQ mask
            master_mask = np.asarray(mask.dq, dtype=DQ_DTYPE).copy()

            # Set fatal pixels to 1.0 in normalized data
            fatal_mask = (master_mask & int(FATAL_FLAGS)) != 0
            normalized_data[fatal_mask] = 1.0

            # Flag dead pixels (below threshold)
            dead_mask = normalized_data < settings.calib.dead_pixel_frac
            master_mask[dead_mask] |= int(DQ.DEAD)

            mask_cards.append(("MASKFIL", "applied", "DQ mask applied"))
        else:
            mask_cards.append(("MASKFIL", "NONE", "No mask applied"))

        # Assemble header cards
        header_cards = (
            oscan_cards
            + bias_cards
            + dark_cards
            + level_cut_cards
            + gain_cards
            + combine_cards
            + norm_cards
            + mask_cards
        )

        # Create history line
        history = f"Master flat built from {n_accepted} frames in group {group_key}"

        # Create MasterFrame
        master = MasterFrame(
            data=normalized_data,
            variance=normalized_variance,
            weight=master_weight,
            mask=master_mask,
            header_cards=header_cards,
            history=[history],
            bunit="1",  # Normalized flat is dimensionless
        )

        # Write to disk
        output_filename = f"master_flat_{group_key.replace(':', '_')}.fits"
        output_path = outdir_path / output_filename
        master.write(output_path, overwrite=settings.runtime.clobber)

        masters[group_key] = master
        logger.info("Wrote master flat to %s", output_path)

    return masters
