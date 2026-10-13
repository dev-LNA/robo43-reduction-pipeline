"""Master flat-field construction.

Per-filter master flats are built by combining and normalizing calibration flat
frames. The ``separate_dome_and_sky_flats`` setting was previously unspecified;
this module adopts the semantics that when enabled, twilight (sky) flats --
recognised by :func:`robo43.calib.twilight.is_twilight_flat` from the OBJECT
value, the raw IMAGETYP value or the file name -- are grouped separately from
"dome" flats under the same filter, allowing independent masters per
illumination source. A twilight group is scaled to a common level and
star-masked before it is combined (:mod:`robo43.calib.twilight`).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robo43.calib.bias import apply_overscan_trim, resolve_common_gain, select_frames
from robo43.calib.combine import combine_lazy, combine_stack
from robo43.calib.masks import cold_pixel_mask, write_cold_pixel_mask
from robo43.calib.master import MasterFrame
from robo43.calib.twilight import (
    TwilightParams,
    combine_twilight,
    flat_star_qc,
    frame_level,
    is_twilight_flat,
    method_cards,
    scales_from_levels,
)
from robo43.config import Settings
from robo43.exceptions import CalibrationError
from robo43.instruments.profile import InstrumentProfile
from robo43.io.fits_io import read_image
from robo43.io.meta import FrameMeta
from robo43.products.dq import DQ, DQ_DTYPE, FATAL_FLAGS

if TYPE_CHECKING:
    from robo43.calib.gain import GainMeasurement, GainMeasurementSet
    from robo43.calib.masks import MaskSet
    from robo43.calib.overscan import OverscanGeometry

logger = logging.getLogger(__name__)

__all__ = ["build_master_flats", "group_flats_by_filter", "normalise_flat"]


def group_flats_by_filter(
    metas: Sequence[FrameMeta], *, separate_dome_and_sky: bool = True
) -> dict[str, list[FrameMeta]]:
    """Group flat frames by filter, optionally separating dome and sky flats.

    Returns a dict mapping group keys to lists of FrameMeta objects in input
    order. When ``separate_dome_and_sky`` is True, twilight flats
    (:func:`robo43.calib.twilight.is_twilight_flat`: OBJECT, raw IMAGETYP or
    file name says sky/twilight) are grouped to key ``"{filter}:sky"``; all
    others to ``"{filter}:dome"``. When False, all
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
            key = f"{meta.filter}:sky" if is_twilight_flat(meta) else f"{meta.filter}:dome"
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


def _read_and_correct(
    meta: FrameMeta,
    *,
    profile: InstrumentProfile,
    settings: Settings,
    overscan: OverscanGeometry | None,
    bias: MasterFrame | None,
    dark_current_map: np.ndarray | None,
    tile_slices: Sequence[tuple[slice, slice]] | None = None,
    tile_gains: tuple[GainMeasurement, ...] | None = None,
) -> tuple[np.ndarray, float, list[tuple[str, Any, str]]]:
    """Read one raw flat frame and apply overscan trim, bias, and dark subtraction.

    Returns ``(corrected_float32, median, oscan_cards)``. The median is
    computed on the float64 working array *before* it is cast down to
    float32 (and before any per-tile gain conversion below), matching the
    precision the pre-streaming code used -- this keeps the level cut
    numerically identical regardless of which storage path (in-memory list
    or on-disk memmap) the caller uses, and keeps it comparing an ADU median
    against ``profile.saturation_adu`` (an ADU quantity) regardless of
    whether the returned array ends up in ADU or electrons. ``oscan_cards``
    is only meaningful for a group's first frame.

    When ``tile_gains`` is given (one entry per ``tile_slices`` region, same
    row-major order, both from the same overscan geometry as
    :mod:`robo43.calib.gain`'s measurement), each tile of the bias/dark-
    corrected array is multiplied by its own measured e-/ADU gain here --
    before the flat level cut and before combining -- so the master flat
    this frame contributes to is built in electrons, per amplifier tile.
    The caller only passes ``tile_gains`` for a genuine multi-tile
    (multi-amplifier) group; for a single-tile group this conversion is a
    uniform scalar multiply that :func:`normalise_flat` would cancel
    exactly anyway, so the caller leaves it ``None`` there and this
    function's output is then byte-for-bit what it always was.
    """
    data, _header = read_image(meta.path, profile)
    if overscan is not None:
        from robo43.calib.overscan import apply_overscan

        trimmed, oscan_cards = apply_overscan(data, overscan, settings=settings)
    else:
        trimmed, oscan_cards = apply_overscan_trim(data, profile=profile, settings=settings)

    working = trimmed.astype(np.float64)

    if bias is not None:
        if working.shape != bias.data.shape:
            msg = (
                f"bias shape {bias.data.shape} does not match trimmed frame "
                f"shape {working.shape}"
            )
            raise CalibrationError(msg)
        working = working - bias.data

    if dark_current_map is not None and meta.exptime is not None:
        working = working - (dark_current_map * meta.exptime)

    median = float(np.median(working))

    if tile_gains is not None and tile_slices is not None:
        for tile_idx, (row_sl, col_sl) in enumerate(tile_slices):
            if tile_idx >= len(tile_gains):
                continue
            working[row_sl, col_sl] = (
                working[row_sl, col_sl] * tile_gains[tile_idx].gain_e_per_adu
            )

    return working.astype(np.float32), median, oscan_cards


def _flat_level_cut(
    frame_medians: Sequence[float],
    group_metas: Sequence[FrameMeta],
    group_key: str,
    profile: InstrumentProfile,
    settings: Settings,
) -> tuple[list[int], list[tuple[str, Any, str]], bool]:
    """Indices of ``group_metas`` surviving the flat level cut, plus its cards.

    Accepts frames whose median falls in
    ``[flat_min_level_frac, flat_max_level_frac] * profile.saturation_adu``.
    When the profile declares no saturation level, the cut is skipped and
    every frame is accepted.
    """
    if profile.saturation_adu is not None:
        min_level = settings.calib.flat_min_level_frac * profile.saturation_adu
        max_level = settings.calib.flat_max_level_frac * profile.saturation_adu

        accepted_indices: list[int] = []
        for i, med in enumerate(frame_medians):
            if min_level <= med <= max_level:
                accepted_indices.append(i)
            else:
                logger.warning(
                    "Rejecting flat frame %s (median=%.2f, range=[%.2f, %.2f])",
                    group_metas[i].path,
                    med,
                    min_level,
                    max_level,
                )
        level_cut_applied = True
    else:
        logger.warning(
            "Skipping level cut for group %s: profile declares no saturation level",
            group_key,
        )
        accepted_indices = list(range(len(frame_medians)))
        level_cut_applied = False

    n_accepted = len(accepted_indices)
    n_rejected = len(frame_medians) - n_accepted
    cards = [
        ("FLATNACC", n_accepted, "Flat frames accepted after level cut"),
        ("FLATNREJ", n_rejected, "Flat frames rejected by level cut"),
        ("FLATLCUT", level_cut_applied, "Flat level cut applied"),
    ]
    return accepted_indices, cards, level_cut_applied


def _flat_fallback_cards(
    contributing_nights: tuple[str, ...],
    borrowed: bool,
    n_accepted: int,
) -> list[tuple[str, Any, str]]:
    """Provenance cards: CALFBACK, source night(s), and frames combined.

    ``contributing_nights`` lists every night (including the current one)
    that contributed at least one frame to the pool feeding this group;
    ``borrowed`` is True when any night other than the current one did (set
    by the caller from the multi-night fallback/pooling result -- this
    function does not know which night is "current"). A comma-joined night
    list that would not fit a single FITS string card (68-character limit)
    is instead recorded as the earliest/latest contributing night plus a
    count, using <=8-character keywords (FLATN1, FLATN2, FLATNNGT).
    """
    cards: list[tuple[str, Any, str]] = [
        ("CALFBACK", borrowed, "Calibration borrowed from a previous night"),
    ]
    if contributing_nights:
        joined = ",".join(contributing_nights)
        if len(joined) <= 68:
            cards.append(
                ("FLATNGT", joined, "Source night(s) of this master flat")
            )
        else:
            ordered = sorted(contributing_nights)
            cards.append(("FLATN1", ordered[0], "Earliest contributing night"))
            cards.append(("FLATN2", ordered[-1], "Latest contributing night"))
            cards.append(
                ("FLATNNGT", len(contributing_nights), "Number of contributing nights")
            )
    cards.append(("FLATNFRM", n_accepted, "Flat frames actually combined"))
    return cards


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
    fallback_source_nights: Mapping[str, tuple[tuple[str, ...], bool]] | None = None,
    pool_window_nights: int | None = None,
    pool_target_frames: int | None = None,
    twilight_params: TwilightParams | None = None,
) -> dict[str, MasterFrame]:
    """Build and write master flat frames, one per filter (and illumination source).

    Processes each group in sorted key order:
    1. Reads and overscan/bias/dark-corrects each frame, computing its median
       for the level cut. When a group's frames would occupy more than
       ``settings.calib.max_memory_mb`` (estimated conservatively from each
       frame's raw ``NAXIS1``/``NAXIS2``, counting both the corrected copy and
       a working copy), the corrected frames are written to a disk-backed
       ``np.memmap`` scratch file in ``outdir`` instead of being held in a
       Python list, and combined from there row-block by row-block; small
       groups are combined fully in memory exactly as before, so results are
       numerically identical either way (:func:`robo43.calib.combine.combine_lazy`
       is bit-identical to :func:`robo43.calib.combine.combine_stack`).
    2. Applies the level cut: accepts only frames with median in
       [flat_min_level_frac, flat_max_level_frac] * saturation_adu
    3. Combines accepted frames (or uses the single frame if n==1). A twilight
       group (every frame :func:`robo43.calib.twilight.is_twilight_flat`) is
       scaled to a common level (central-region median) before clipping and
       its stars are masked per frame (:func:`robo43.calib.twilight.combine_twilight`),
       in memory and streaming alike; a dome group is combined as before unless
       ``twilight_params.scale_all_flats`` also scales it
    4. Normalizes the combined data and counts the compact positive peaks left
       in it (:func:`robo43.calib.twilight.flat_star_qc`; cards ``FLATQC*``,
       WARNING above ``twilight_params.qc_warn_per_mpix``)
    5. Builds the cold-pixel mask from the normalized master (skipped when
       fewer than ``settings.calib.cold_pixel_min_flats`` flats were combined)
       and writes it to outdir/coldpix_<key>.fits (: -> _)
    6. Applies mask and sets fatal pixels to 1.0; cold pixels are flagged DEAD
    7. Writes master frame to outdir/master_flat_<key>.fits (: -> _)

    ``fallback_source_nights`` maps a group key to ``(contributing_nights,
    borrowed)``: every night (including the current one) that contributed a
    frame to that group's pool, and whether any night other than the current
    one did. A group absent from the mapping gets ``CALFBACK=False`` and no
    ``FLATNGT`` card, exactly as before this parameter's pooling-era meaning
    was added.

    ``pool_window_nights`` / ``pool_target_frames`` (both ``None`` unless the
    flats came from :mod:`robo43.calib.pool`) add the pooling-policy cards
    ``FLATWIN`` (window, +/- nights) and ``FLATTGT`` (target frames) to every
    master built in this call; the contributing nights (``FLATNGT`` or
    ``FLATN1``/``FLATN2``/``FLATNNGT``) and the frames combined
    (``FLATNFRM``) come from ``fallback_source_nights`` as before.

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
    max_bytes = settings.calib.max_memory_mb * 1024.0 * 1024.0
    tw_params = twilight_params if twilight_params is not None else TwilightParams()

    for group_key in sorted(groups.keys()):
        group_metas = groups[group_key]
        n_group = len(group_metas)
        logger.info("Processing flat group %s with %d frames", group_key, n_group)

        # A group is twilight only when every frame is; such a group is scaled
        # to a common level and star-masked. Dome groups are scaled only when
        # tw_params.scale_all_flats asks for it.
        group_twilight = all(is_twilight_flat(m) for m in group_metas)
        scale_group = group_twilight or tw_params.scale_all_flats

        # Conservative (raw, untrimmed) estimate of holding both a corrected
        # copy and a transient working copy of every frame in this group at
        # once, decided before reading any pixel data.
        estimated_bytes = 2.0 * n_group * group_metas[0].naxis1 * group_metas[0].naxis2 * 4.0
        stream = estimated_bytes > max_bytes

        # Resolve this group's (readout_mode, gain_setting, binning) key and
        # its runtime PTC measurement *before* reading any frame, so each
        # frame's data can be gain-converted (per amplifier tile) as it is
        # read below. Raises CalibrationError, exactly as before, when the
        # group's own frames disagree on gain_setting -- never silently
        # combined. The level cut afterwards still compares each frame's own
        # ADU median (computed before that conversion, in _read_and_correct)
        # against profile.saturation_adu, an ADU quantity, regardless of
        # whether this group ends up gain-converted or not.
        group_gain_setting, _group_gain_origin = resolve_common_gain(group_metas, profile=profile)
        group_readout_mode = next(
            (m.readout_mode for m in group_metas if m.readout_mode is not None), None
        )
        group_binning = next((m.binning for m in group_metas if m.binning is not None), 1)
        group_measurement = (
            gain_measurements.get(group_readout_mode, group_gain_setting, group_binning)
            if gain_measurements is not None
            else None
        )
        group_tile_gains = (
            gain_measurements.tiles_for(group_readout_mode, group_gain_setting, group_binning)
            if gain_measurements is not None
            else None
        )
        gain_e_per_adu = group_measurement.gain_e_per_adu if group_measurement is not None else None
        read_noise_e = (
            group_measurement.read_noise_e if group_measurement is not None else profile.read_noise_e
        )
        from robo43.calib.gain import no_measurement_cards, per_tile_header_cards

        gain_cards = (
            group_measurement.header_cards()
            if group_measurement is not None
            else no_measurement_cards(group_gain_setting, group_readout_mode)
        )
        # Applying the per-tile conversion during the read loop below is
        # worthwhile only for a genuine multi-tile group: for a single tile
        # it is a uniform scalar multiply that normalise_flat() cancels out
        # exactly anyway, so leaving it unapplied there keeps a
        # single-amplifier instrument's (ROBO43's) master flat byte-for-bit
        # unchanged.
        apply_tile_gains = (
            group_tile_gains if group_tile_gains and len(group_tile_gains) > 1 else None
        )
        tile_slices = (
            (overscan.science_tile_slices if overscan is not None else [(slice(None), slice(None))])
            if apply_tile_gains is not None
            else None
        )
        gain_cards = gain_cards + per_tile_header_cards(group_tile_gains)
        # When the data fed to combine_* below is already in electrons
        # (per-tile gain applied during the read loop), gain_e_per_adu=1.0
        # tells its own variance floor not to convert again; otherwise (no
        # per-tile measurement) the data stays ADU and this is the
        # frame-level e-/ADU value exactly as before.
        combine_gain_e_per_adu = 1.0 if apply_tile_gains is not None else gain_e_per_adu

        oscan_cards: list[tuple[str, Any, str]] = []
        bias_cards: list[tuple[str, Any, str]] = []
        dark_cards: list[tuple[str, Any, str]] = []
        frame_medians: list[float] = []
        frame_levels: list[float] = []  # central-region medians, only when scale_group
        frame_shape: tuple[int, int] | None = None
        processed_frames: list[np.ndarray] = []  # used only when not streaming

        scratch_path: Path | None = None
        memmap_arr: np.memmap | None = None

        try:
            for i, meta in enumerate(group_metas):
                processed, median, cards = _read_and_correct(
                    meta,
                    profile=profile,
                    settings=settings,
                    overscan=overscan,
                    bias=bias,
                    dark_current_map=dark_current_map,
                    tile_slices=tile_slices,
                    tile_gains=apply_tile_gains,
                )

                if i == 0:
                    oscan_cards = cards
                    frame_shape = processed.shape
                    if stream:
                        scratch_path = (
                            outdir_path
                            / f".flat_stream_{group_key.replace(':', '_')}_{os.getpid()}.dat"
                        )
                        memmap_arr = np.memmap(
                            scratch_path,
                            mode="w+",
                            dtype=np.float32,
                            shape=(n_group, *frame_shape),
                        )
                    if bias is not None:
                        bias_basename = Path(bias.path).name if bias.path else "NONE"
                        bias_cards.append(("BIASFIL", bias_basename, "Master bias applied"))
                    else:
                        bias_cards.append(("BIASFIL", "NONE", "No bias subtraction"))
                    if dark_current_map is not None:
                        dark_cards.append(
                            ("DARKFIL", "applied", "Dark current scaling applied")
                        )
                    else:
                        dark_cards.append(("DARKFIL", "NONE", "No dark current subtraction"))
                elif processed.shape != frame_shape:
                    msg = (
                        f"flat frame {meta.path} has shape {processed.shape}, expected "
                        f"{frame_shape} (all frames in a group must share the same shape)"
                    )
                    raise CalibrationError(msg)

                frame_medians.append(median)
                if scale_group:
                    frame_levels.append(
                        frame_level(processed, settings.calib.flat_norm_region_frac)
                    )

                if stream:
                    memmap_arr[i] = processed
                    del processed
                else:
                    processed_frames.append(processed)

            if stream:
                memmap_arr.flush()

            accepted_indices, level_cut_cards, _level_cut_applied = _flat_level_cut(
                frame_medians, group_metas, group_key, profile, settings
            )
            n_accepted = len(accepted_indices)

            if n_accepted == 0:
                logger.warning("No frames in group %s survive level cut; skipping", group_key)
                continue

            accepted_metas = [group_metas[i] for i in accepted_indices]

            group_fallback = (fallback_source_nights or {}).get(group_key)
            if group_fallback is not None:
                contributing_nights, borrowed = group_fallback
            else:
                contributing_nights, borrowed = (), False
            fallback_cards = _flat_fallback_cards(contributing_nights, borrowed, n_accepted)
            if pool_window_nights is not None:
                fallback_cards.append(
                    ("FLATWIN", pool_window_nights, "Flat pool window (+/- nights)")
                )
            if pool_target_frames is not None:
                fallback_cards.append(
                    ("FLATTGT", pool_target_frames, "Flat pool target frames")
                )

            if n_accepted >= 2:
                group_scales: list[float] | None = None
                group_l_ref: float | None = None
                tw_info = None
                if scale_group:
                    acc_levels = [frame_levels[i] for i in accepted_indices]
                    group_l_ref, scale_array = scales_from_levels(acc_levels)
                    group_scales = [float(s) for s in scale_array]
                if group_twilight:
                    if stream:
                        def _tw_stream_loader(idx: int, memmap=memmap_arr):
                            def loader(slices):
                                rows, cols = slices
                                return np.asarray(memmap[idx, rows, cols])

                            return loader

                        tw_loaders = [_tw_stream_loader(i) for i in accepted_indices]
                    else:
                        def _tw_memory_loader(frame_data):
                            def loader(slices):
                                rows, cols = slices
                                return frame_data[rows, cols]

                            return loader

                        tw_loaders = [
                            _tw_memory_loader(processed_frames[i]) for i in accepted_indices
                        ]
                    result, tw_info = combine_twilight(
                        tw_loaders,
                        frame_shape,
                        settings.calib,
                        levels=acc_levels,
                        params=tw_params,
                        read_noise_e=read_noise_e,
                        gain_e_per_adu=combine_gain_e_per_adu,
                    )
                    del tw_loaders  # drop the memmap reference before the scratch file goes
                elif stream:
                    def _make_loader(idx: int):
                        def loader(slices):
                            rows, cols = slices
                            return np.asarray(memmap_arr[idx, rows, cols])

                        return loader

                    loaders = [_make_loader(i) for i in accepted_indices]
                    result = combine_lazy(
                        loaders,
                        frame_shape,
                        settings.calib,
                        scales=group_scales,
                        read_noise_e=read_noise_e,
                        gain_e_per_adu=combine_gain_e_per_adu,
                    )
                else:
                    accepted_frames = [processed_frames[i] for i in accepted_indices]
                    frame_size_bytes = n_accepted * accepted_frames[0].size * 4
                    if frame_size_bytes > max_bytes:
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
                            scales=group_scales,
                            read_noise_e=read_noise_e,
                            gain_e_per_adu=combine_gain_e_per_adu,
                        )
                    else:
                        result = combine_stack(
                            accepted_frames,
                            settings.calib,
                            scales=group_scales,
                            read_noise_e=read_noise_e,
                            gain_e_per_adu=combine_gain_e_per_adu,
                        )

                combine_cards = result.header_cards()
                if tw_info is not None:
                    combine_cards = combine_cards + tw_info.header_cards()
                else:
                    combine_cards = combine_cards + method_cards(
                        "dome", scaled=group_scales is not None, l_ref=group_l_ref
                    )
                master_data = result.data
                master_variance = result.variance
                master_weight = result.weight
            else:
                single_idx = accepted_indices[0]
                single = memmap_arr[single_idx] if stream else processed_frames[single_idx]
                master_data = np.array(single, dtype=np.float32)
                master_variance = None
                master_weight = None
                combine_cards = [
                    ("NCOMBINE", 1, "Number of input frames"),
                    *method_cards(
                        "twilight" if group_twilight else "dome", scaled=False, l_ref=None
                    ),
                ]
                if group_twilight:
                    logger.warning(
                        "twilight flat group %s has a single usable frame: its stars "
                        "cannot be removed",
                        group_key,
                    )
        finally:
            if memmap_arr is not None:
                del memmap_arr
            if scratch_path is not None and scratch_path.exists():
                scratch_path.unlink()

        # Normalize the flat
        normalized_data, norm_cards = normalise_flat(master_data, settings.calib)

        # Star-residual QC on the normalised master, before any masking below.
        star_qc = flat_star_qc(normalized_data, tw_params)
        if star_qc.per_mpix > tw_params.qc_warn_per_mpix:
            logger.warning(
                "master flat %s has %d compact positive peaks (%.1f per Mpx > %.1f) above "
                "%.2f x its local median: star residuals are likely (max ratio %.3f)",
                group_key,
                star_qc.n_peaks,
                star_qc.per_mpix,
                tw_params.qc_warn_per_mpix,
                star_qc.threshold,
                star_qc.max_ratio,
            )
        else:
            logger.info(
                "master flat %s: %d compact positive peaks above %.2f x local median",
                group_key,
                star_qc.n_peaks,
                star_qc.threshold,
            )

        # Cold pixels are judged on the normalized master BEFORE any fatal-pixel
        # replacement below, so an already-masked pixel cannot hide its defect.
        # n_accepted counts the flats that survived the level cut and were
        # combined (not the inputs).
        cold = cold_pixel_mask(normalized_data, n_accepted, settings.calib)
        coldpix_name = "NONE"
        if cold is not None:
            coldpix_name = f"coldpix_{group_key.replace(':', '_')}.fits"
            write_cold_pixel_mask(
                outdir_path / coldpix_name,
                cold,
                n_flats=n_accepted,
                settings=settings.calib,
                overwrite=True,  # regenerated together with its master
            )
            logger.info("Wrote cold-pixel mask to %s", outdir_path / coldpix_name)

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
            if cold is not None:
                master_mask[cold] |= int(DQ.DEAD)

            mask_cards.append(("MASKFIL", "applied", "DQ mask applied"))
        else:
            mask_cards.append(("MASKFIL", "NONE", "No mask applied"))
        mask_cards.append(("COLDPIX", coldpix_name, "Cold-pixel mask file"))

        # Assemble header cards
        header_cards = (
            oscan_cards
            + bias_cards
            + dark_cards
            + level_cut_cards
            + gain_cards
            + combine_cards
            + norm_cards
            + star_qc.header_cards()
            + mask_cards
            + fallback_cards
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

        del processed_frames
        if settings.runtime.release_memory:
            from robo43.memory import release_memory

            release_memory(f"flat:{group_key}")

    return masters
