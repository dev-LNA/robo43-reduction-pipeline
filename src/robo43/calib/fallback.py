"""Multi-night calibration fallback for a night missing bias, dark, or flat frames.

T80S nights sometimes arrive without bias frames (rarely) or without flat
frames for one or more filters (often). When the night being reduced has
*zero* raw frames of a calibration type, this module searches strictly
earlier sibling ``YYYYMMDD`` night directories (under the same parent) for
matching raw frames of that type and returns them, so the existing master
builders (:mod:`robo43.calib.bias`, :mod:`robo43.calib.dark`,
:mod:`robo43.calib.flat`) can build a master from borrowed frames exactly as
they would from the current night's own.

Only ever reads headers (via :func:`robo43.io.fits_io.iter_fits_files` and
:func:`robo43.io.headers.read_meta`), never pixel data, and stops scanning as
soon as enough matching frames have been found (nearest night first).

Matching never considers ``FILTER`` for bias/dark frames (the project's own
rule: the filter wheel position on a bias/dark frame is meaningless), and for
flats matches on the caller-supplied canonical filter name in addition to the
camera configuration below.

Camera-configuration matching uses ``(readout_mode, gain_setting, binning,
naxis1, naxis2)`` -- the same fields the rest of the pipeline already groups
calibration frames by, plus the raw (untrimmed) frame shape.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from robo43.io.fits_io import iter_fits_files
from robo43.io.headers import read_meta

if TYPE_CHECKING:
    from robo43.config import Settings
    from robo43.instruments.profile import InstrumentProfile
    from robo43.io.meta import FrameMeta

logger = logging.getLogger(__name__)

__all__ = [
    "CameraConfig",
    "FallbackResult",
    "FlatPoolResult",
    "camera_config_key",
    "collect_fallback_frames",
    "dominant_camera_config",
    "parse_night_dirname",
    "pool_flat_frames",
]

#: (readout_mode, gain_setting, binning, naxis1, naxis2)
CameraConfig = tuple[str | None, float | None, int | None, int, int]

_NIGHT_DIRNAME_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")


def parse_night_dirname(name: str) -> date | None:
    """Parse a ``YYYYMMDD`` directory name into a :class:`~datetime.date`.

    Returns ``None`` when ``name`` is not exactly 8 digits or is not a valid
    calendar date (e.g. ``"20250230"``).
    """
    match = _NIGHT_DIRNAME_RE.match(name)
    if match is None:
        return None
    year, month, day = (int(group) for group in match.groups())
    try:
        return date(year, month, day)
    except ValueError:
        return None


def camera_config_key(meta: FrameMeta) -> CameraConfig:
    """The tuple a borrowed frame must match against the target night's own.

    Filter is deliberately excluded: callers handle it separately (only
    relevant for flats; bias/dark are never matched or grouped by filter).
    """
    return (meta.readout_mode, meta.gain_setting, meta.binning, meta.naxis1, meta.naxis2)


def dominant_camera_config(metas: Sequence[FrameMeta]) -> CameraConfig | None:
    """The most common :func:`camera_config_key` among ``metas``, or ``None``."""
    counts = Counter(camera_config_key(m) for m in metas)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


@dataclass(frozen=True, slots=True)
class FallbackResult:
    """Outcome of a previous-night search for one calibration type (and,
    for flats, one filter)."""

    #: Borrowed frames found, in the order they were collected. Empty when
    #: nothing matched.
    metas: tuple[FrameMeta, ...]
    #: Source night(s), as ``YYYYMMDD`` strings, in the order they
    #: contributed frames (nearest first).
    source_nights: tuple[str, ...]
    #: True once ``metas`` reached the caller's ``min_count``. When False,
    #: ``metas`` may still be non-empty (a partial pool below the minimum);
    #: callers must treat that as "nothing suitable found" and must not use
    #: a partial result.
    satisfied: bool


def _previous_night_dirs(
    night_dir: Path, current_night: date, max_nights: int, min_days: int = 0
) -> list[tuple[date, Path]]:
    """Sibling ``YYYYMMDD`` directories under ``night_dir.parent`` strictly
    earlier than ``current_night`` and within ``max_nights`` calendar days,
    nearest first.  ``min_days`` (default 0: no effect, since every night
    returned is already at least one day back) additionally drops nights
    ``min_days`` days back or closer."""
    parent = night_dir.parent
    if not parent.is_dir():
        return []
    resolved_night_dir = night_dir.resolve()
    found: list[tuple[date, Path]] = []
    for entry in parent.iterdir():
        if not entry.is_dir() or entry.resolve() == resolved_night_dir:
            continue
        night = parse_night_dirname(entry.name)
        if night is None or night >= current_night:
            continue
        if (current_night - night).days > max_nights:
            continue
        if (current_night - night).days <= min_days:
            continue
        found.append((night, entry))
    found.sort(key=lambda pair: pair[0], reverse=True)
    return found


def collect_fallback_frames(
    night_dir: Path,
    current_night: date,
    *,
    frame_type: str,
    target_config: CameraConfig,
    min_count: int,
    max_nights: int,
    profile: InstrumentProfile,
    settings: Settings,
    filter_name: str | None = None,
    skip_within_days: int = 0,
) -> FallbackResult:
    """Search strictly earlier sibling nights for ``min_count`` frames of
    ``frame_type`` matching ``target_config``.

    ``skip_within_days`` (default 0: search every earlier night) leaves out
    the nights that many days back or closer -- used by
    :mod:`robo43.calib.pool` to search only beyond its window.

    ``filter_name`` must be given (and is matched exactly) for
    ``frame_type == "flat"``; it is ignored for ``"bias"`` and ``"dark"``,
    which are never matched or grouped by filter.

    Reads headers only, nearest night first, and stops at the first night
    that satisfies the requirement:

    1. If a single night's own matching frames already reach ``min_count``,
       returns immediately using only that night (never mixed with any
       nearer night's partial matches).
    2. Otherwise its matches (even zero) are added to a running pool drawn
       from the nearer nights already visited; once that pool reaches
       ``min_count``, returns it (a master mixing those nights).
    3. If the window is exhausted first, returns whatever was pooled
       (possibly empty) with ``satisfied=False``.
    """
    pooled_metas: list[FrameMeta] = []
    pooled_nights: list[str] = []

    for night, night_path in _previous_night_dirs(
        night_dir, current_night, max_nights, min_days=skip_within_days
    ):
        night_str = night.strftime("%Y%m%d")
        candidates = list(iter_fits_files([night_path], recursive=False))
        night_matches: list[FrameMeta] = []
        for path in candidates:
            try:
                meta = read_meta(path, profile, settings=settings)
            except Exception as exc:
                logger.debug(
                    "calib fallback scan: failed to read header for %s: %s", path, exc
                )
                continue
            if meta.frame_type != frame_type:
                continue
            if frame_type == "flat" and meta.filter != filter_name:
                continue
            if camera_config_key(meta) != target_config:
                continue
            night_matches.append(meta)

        if len(night_matches) >= min_count:
            return FallbackResult(
                metas=tuple(night_matches), source_nights=(night_str,), satisfied=True
            )

        if night_matches:
            pooled_metas.extend(night_matches)
            pooled_nights.append(night_str)
            if len(pooled_metas) >= min_count:
                return FallbackResult(
                    metas=tuple(pooled_metas),
                    source_nights=tuple(pooled_nights),
                    satisfied=True,
                )

    return FallbackResult(
        metas=tuple(pooled_metas), source_nights=tuple(pooled_nights), satisfied=False
    )


@dataclass(frozen=True, slots=True)
class FlatPoolResult:
    """Outcome of pooling one filter's flat frames across nights.

    Unlike :class:`FallbackResult` (used for bias/dark, and for flats when
    pooling is disabled -- see :func:`collect_fallback_frames`), pooling is
    always active for a filter once enabled: it runs whether or not the
    current night has any local flats of its own, and folds the current
    night's own matching frames into the pool alongside any borrowed ones.
    """

    #: Every matching frame in the pool: the current night's own frames (if
    #: any) first, then borrowed nights' frames in the order their nights
    #: were folded in (nearest first). Empty when nothing matched at all.
    metas: tuple[FrameMeta, ...]
    #: Every night that contributed >=1 frame, current night first (if it
    #: contributed), then earlier nights nearest-first. Empty iff ``metas``
    #: is empty.
    contributing_nights: tuple[str, ...]
    #: True when any night other than the current one contributed.
    borrowed: bool


def pool_flat_frames(
    night_dir: Path,
    current_night: date,
    *,
    local_metas: Sequence[FrameMeta],
    target_config: CameraConfig,
    filter_name: str,
    min_nights: int,
    max_nights: int,
    target_frames: int,
    fallback_max_nights: int,
    profile: InstrumentProfile,
    settings: Settings,
) -> FlatPoolResult:
    """Pool one filter's flat frames: this night's own plus earlier nights.

    Always active for a filter once the instrument profile enables pooling
    (``InstrumentProfile.calib.flat_pool``), regardless of whether the
    current night has any local flats of its own for that filter. Reads
    headers only, nearest night first:

    1. ``local_metas`` (the current night's own matching flats, already read
       by the caller) seed the pool.
    2. Every earlier sibling night within ``min_nights`` calendar days is
       folded in -- the whole night, whether or not the pool already reached
       ``target_frames``.
    3. If the pool is still short of ``target_frames``, earlier nights
       beyond ``min_nights`` and up to ``max_nights`` calendar days are
       folded in, nearest first, stopping as soon as a whole night's worth
       brings the pool to ``target_frames``.
    4. A non-empty pool still short of ``target_frames`` after ``max_nights``
       is returned as-is (the caller logs this; pooling never reaches further
       for a partial pool).
    5. An empty pool after ``max_nights`` falls through to a single-night
       search, nearest first, of nights strictly beyond ``max_nights`` and
       within ``fallback_max_nights`` calendar days; the nearest such night
       with any matching frames is returned alone (never accumulated with
       others). If none is found, an empty result is returned.
    """
    current_night_str = current_night.strftime("%Y%m%d")
    pooled_metas: list[FrameMeta] = list(local_metas)
    contributing_nights: list[str] = [current_night_str] if local_metas else []

    def _night_matches(night_path: Path) -> list[FrameMeta]:
        matches: list[FrameMeta] = []
        for path in iter_fits_files([night_path], recursive=False):
            try:
                meta = read_meta(path, profile, settings=settings)
            except Exception as exc:
                logger.debug(
                    "flat pool scan: failed to read header for %s: %s", path, exc
                )
                continue
            if meta.frame_type != "flat":
                continue
            if meta.filter != filter_name:
                continue
            if camera_config_key(meta) != target_config:
                continue
            matches.append(meta)
        return matches

    floor_nights = _previous_night_dirs(night_dir, current_night, min_nights)
    for night, night_path in floor_nights:
        matches = _night_matches(night_path)
        if matches:
            pooled_metas.extend(matches)
            contributing_nights.append(night.strftime("%Y%m%d"))

    if len(pooled_metas) < target_frames:
        extension_nights = _previous_night_dirs(night_dir, current_night, max_nights)[
            len(floor_nights) :
        ]
        for night, night_path in extension_nights:
            if len(pooled_metas) >= target_frames:
                break
            matches = _night_matches(night_path)
            if matches:
                pooled_metas.extend(matches)
                contributing_nights.append(night.strftime("%Y%m%d"))

    if pooled_metas:
        return FlatPoolResult(
            metas=tuple(pooled_metas),
            contributing_nights=tuple(contributing_nights),
            borrowed=any(n != current_night_str for n in contributing_nights),
        )

    for night, night_path in _previous_night_dirs(
        night_dir, current_night, fallback_max_nights
    ):
        if (current_night - night).days <= max_nights:
            continue
        matches = _night_matches(night_path)
        if matches:
            return FlatPoolResult(
                metas=tuple(matches),
                contributing_nights=(night.strftime("%Y%m%d"),),
                borrowed=True,
            )

    return FlatPoolResult(metas=(), contributing_nights=(), borrowed=False)
