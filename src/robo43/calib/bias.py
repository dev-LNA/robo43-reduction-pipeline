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
    from robo43.calib.overscan import OverscanGeometry

logger = logging.getLogger(__name__)

__all__ = ["apply_overscan_trim", "build_master_bias", "select_frames"]


def select_frames(metas: Sequence[FrameMeta], frame_type: str) -> list[FrameMeta]:
    """Return the metas whose ``frame_type`` matches, input order preserved."""
    return [meta for meta in metas if meta.frame_type == frame_type]


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
) -> MasterFrame:
    """Build and write a master bias frame from ``metas``."""
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

    if len(trimmed_frames) >= 2:
        result = combine_stack(
            trimmed_frames,
            settings.calib,
            read_noise_e=profile.read_noise_e,
            gain_e_per_adu=profile.gain_e_per_adu,
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

    header_cards = oscan_cards + combine_cards

    master = MasterFrame(
        data=master_data,
        variance=master_variance,
        weight=master_weight,
        header_cards=header_cards,
        bunit="adu",
    )
    master.write(outdir_path / "master_bias.fits", overwrite=settings.runtime.clobber)
    return master
