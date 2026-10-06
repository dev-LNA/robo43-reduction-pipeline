"""Symmetric-window calibration pooling (bias and per-filter flats).

Unlike :mod:`robo43.calib.fallback` (which only ever looks at *earlier* nights
and, for bias, only when the night has none), this module pools calibration
frames from sibling ``YYYYMMDD`` night directories on **both** sides of the
night being reduced:

* the night's own directory is always scanned first (distance 0);
* sibling nights within ``window_nights`` calendar days (earlier and later)
  are then folded in nearest first -- on a tie the earlier night goes first --
  one *whole* night at a time, until the pool holds at least ``target_frames``
  frames.  The night that crosses the target is kept whole, so the pool can
  overshoot the target by up to one night's worth of frames;
* only frames matching the science frames' camera configuration
  (:func:`robo43.calib.fallback.camera_config_key`: readout mode, gain
  setting, binning and raw frame shape) -- and, for flats, the filter -- are
  counted, and frames flagged ``VALID = False`` are ignored;
* frames passed explicitly by the caller are merged with the pool, de-duplicated
  on their resolved path.  They are never filtered out (the caller chose
  them) and they count toward the target;
* a bias pool short of the target is used as-is and a WARNING with the count
  and the window is logged;
* when the window holds *no* usable bias frame at all, the pre-existing
  beyond-window fallback of :mod:`robo43.calib.fallback` applies unchanged
  (earlier nights only, ``fallback_max_nights_bias``);
* a flat pool short of the target is NOT turned into a master when the
  profile lists reduced trees to search (``flat_master_roots``): the existing
  master flat of that filter closest in time to the night (either direction,
  no maximum distance, earlier night on a tie) is reused instead -- see
  :func:`find_nearest_master_flat` and :func:`install_master_flat`.  Only when
  no suitable master exists is the short pool built, with a WARNING.  Without
  ``flat_master_roots`` a short flat pool is built as-is and an empty one falls
  back to the older single-night raw-frame search, as before.

Only headers are read here, never pixel data, and nights farther away than
the one that reaches the target are never opened.
"""

from __future__ import annotations

import logging
import math
import shutil
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from robo43.calib.fallback import (
    CameraConfig,
    _previous_night_dirs,
    camera_config_key,
    collect_fallback_frames,
    dominant_camera_config,
    parse_night_dirname,
)
from robo43.exceptions import ConfigError
from robo43.io.fits_io import iter_fits_files
from robo43.io.headers import read_meta

if TYPE_CHECKING:
    from robo43.config import Settings
    from robo43.instruments.profile import InstrumentProfile
    from robo43.io.meta import FrameMeta

logger = logging.getLogger(__name__)

__all__ = [
    "MasterFlatChoice",
    "NightScanner",
    "PoolSelection",
    "WindowPoolResult",
    "apply_window_pools",
    "enforce_single_bias_mode",
    "find_nearest_master_flat",
    "infer_science_night",
    "install_master_flat",
    "limit_bias_for_gain",
    "pool_bias_frames",
    "pool_flat_frames_window",
    "restrict_bias_to_mode",
    "window_night_dirs",
]

#: Sort key used for frames whose night cannot be determined (explicit frames
#: living outside any ``YYYYMMDD`` directory): they go last.
_UNKNOWN_DISTANCE = 10**9


#: Header keywords that may carry the number of frames a master flat was
#: built from, in order of preference.
_MASTER_FRAME_COUNT_KEYS = ("FLATNACC", "FLATNFRM", "NCOMBINE")


@dataclass(frozen=True, slots=True)
class MasterFlatChoice:
    """An existing master flat picked for reuse instead of a short pool."""

    path: Path
    #: The reduced night the master was built for (its ``<root>/YYYYMMDD``).
    night: date
    #: ``night`` minus the science night, in days (negative = earlier).
    distance_days: int
    #: Frames the master was built from (``None`` if its header says nothing).
    n_frames: int | None


@dataclass(frozen=True, slots=True)
class PoolSelection:
    """One pooled set of calibration frames (bias, or one filter's flats)."""

    #: ``"bias"`` or ``"flat"``.
    kind: str
    #: Canonical filter name for flats, ``None`` for bias.
    filter_name: str | None
    #: Every frame of the pool, nearest night first (own night first), each
    #: night's frames in file-name order.
    metas: tuple[FrameMeta, ...]
    #: Contributing nights as ``YYYYMMDD`` strings, same order as ``metas``.
    nights: tuple[str, ...]
    #: ``(night, n_frames)`` per contributing night, same order as ``nights``.
    night_counts: tuple[tuple[str, int], ...]
    window_nights: int
    target_frames: int
    #: True when any night other than the science night contributed.
    borrowed: bool
    #: True when ``len(metas) < target_frames``.
    short: bool
    #: True when the window held nothing and the beyond-window fallback ran.
    beyond_window: bool
    #: Number of frames that came from the caller's explicit list only (not
    #: found again by the directory scan).
    n_explicit_only: int = 0
    #: Flats only: the existing master chosen instead of building one from
    #: ``metas`` (which are then kept for the gain measurement only).
    reuse: MasterFlatChoice | None = None


@dataclass(frozen=True, slots=True)
class WindowPoolResult:
    """What :func:`apply_window_pools` did to a run's frame list."""

    #: The run's metas with bias / pooled-filter flats replaced by the pools.
    metas: list[FrameMeta]
    bias: PoolSelection | None = None
    #: Pooled flats keyed by canonical filter name.
    flats: dict[str, PoolSelection] = field(default_factory=dict)

    @property
    def bias_active(self) -> bool:
        return self.bias is not None

    @property
    def flat_active(self) -> bool:
        return bool(self.flats)


def window_night_dirs(
    night_dir: Path, current_night: date, window_nights: int
) -> list[tuple[date, Path]]:
    """``night_dir`` itself first, then the sibling ``YYYYMMDD`` directories
    within ``window_nights`` calendar days on either side, nearest first
    (the earlier night first on a tie)."""
    found: list[tuple[date, Path]] = []
    parent = night_dir.parent
    if parent.is_dir() and window_nights > 0:
        resolved_night_dir = night_dir.resolve()
        for entry in parent.iterdir():
            if not entry.is_dir() or entry.resolve() == resolved_night_dir:
                continue
            night = parse_night_dirname(entry.name)
            if night is None or abs((night - current_night).days) > window_nights:
                continue
            found.append((night, entry))
    found.sort(key=lambda pair: (abs((pair[0] - current_night).days), pair[0]))
    return [(current_night, night_dir), *found]


class NightScanner:
    """Reads the bias and flat headers of night directories, once per night.

    ``read_meta`` is applied to every FITS file of a night directory (the
    science frames are skipped after the header read); the result is cached
    so the bias pool and every filter's flat pool share one pass.
    """

    def __init__(self, profile: InstrumentProfile, settings: Settings) -> None:
        self._profile = profile
        self._settings = settings
        self._cache: dict[Path, tuple[FrameMeta, ...]] = {}
        #: Night directories actually opened (for tests and the log).
        self.scanned: list[Path] = []

    def calib_frames(self, night_path: Path) -> tuple[FrameMeta, ...]:
        key = night_path.resolve()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        self.scanned.append(night_path)
        frames: list[FrameMeta] = []
        for path in iter_fits_files([night_path], recursive=False):
            try:
                meta = read_meta(path, self._profile, settings=self._settings)
            except Exception as exc:
                logger.debug("calib pool scan: failed to read header for %s: %s", path, exc)
                continue
            if meta.frame_type in ("bias", "flat") and meta.is_valid:
                frames.append(meta)
        result = tuple(sorted(frames, key=lambda m: m.path.name))
        self._cache[key] = result
        return result


def _path_key(path: Path) -> str:
    return str(Path(path).resolve())


def _meta_night(meta: FrameMeta) -> date | None:
    """The night a frame belongs to: its directory name when that is
    ``YYYYMMDD``, otherwise ``meta.night``."""
    from_dir = parse_night_dirname(Path(meta.path).parent.name)
    return from_dir if from_dir is not None else meta.night


def _order_key(meta: FrameMeta, current_night: date) -> tuple[int, date, str]:
    night = _meta_night(meta)
    if night is None:
        return (_UNKNOWN_DISTANCE, date.max, Path(meta.path).name)
    return (abs((night - current_night).days), night, Path(meta.path).name)


def _night_label(meta: FrameMeta) -> str:
    night = _meta_night(meta)
    return night.strftime("%Y%m%d") if night is not None else "unknown"


def _matches(
    meta: FrameMeta,
    frame_type: str,
    target_config: CameraConfig,
    filter_name: str | None,
) -> bool:
    if meta.frame_type != frame_type or not meta.is_valid:
        return False
    if frame_type == "flat" and meta.filter != filter_name:
        return False
    return camera_config_key(meta) == target_config


def _describe(night_counts: Sequence[tuple[str, int]]) -> str:
    return ", ".join(f"{night}({count})" for night, count in night_counts)


def _finish(
    pooled: Sequence[FrameMeta],
    *,
    kind: str,
    filter_name: str | None,
    current_night: date,
    window_nights: int,
    target_frames: int,
    beyond_window: bool,
    n_explicit_only: int,
    night_name: str,
    reuse: MasterFlatChoice | None = None,
) -> PoolSelection:
    ordered = sorted(pooled, key=lambda m: _order_key(m, current_night))
    counts: list[tuple[str, int]] = []
    for meta in ordered:
        label = _night_label(meta)
        if counts and counts[-1][0] == label:
            counts[-1] = (label, counts[-1][1] + 1)
        else:
            counts.append((label, 1))
    current_label = current_night.strftime("%Y%m%d")
    what = kind if filter_name is None else f"{kind} (filter {filter_name})"
    selection = PoolSelection(
        kind=kind,
        filter_name=filter_name,
        metas=tuple(ordered),
        nights=tuple(label for label, _ in counts),
        night_counts=tuple(counts),
        window_nights=window_nights,
        target_frames=target_frames,
        borrowed=any(label != current_label for label, _ in counts),
        short=len(ordered) < target_frames,
        beyond_window=beyond_window,
        n_explicit_only=n_explicit_only,
        reuse=reuse,
    )
    if reuse is not None:
        logger.info(
            "%s pool for night %s has only %d frame(s) (target %d) within +/-%d "
            "day(s); reusing the existing master %s (night %s, %+d day(s), %s "
            "frame(s)) instead of building one",
            what, night_name, len(ordered), target_frames, window_nights,
            reuse.path, reuse.night.strftime("%Y%m%d"), reuse.distance_days,
            reuse.n_frames if reuse.n_frames is not None else "unknown",
        )
        return selection
    if not ordered:
        return selection
    if selection.short:
        logger.warning(
            "%s pool for night %s has only %d frame(s) (target %d) within +/-%d "
            "day(s)%s; building the master from what is available (night(s) %s)",
            what, night_name, len(ordered), target_frames, window_nights,
            " (frames found beyond the window)" if beyond_window else "",
            _describe(selection.night_counts),
        )
    else:
        logger.info(
            "%s pool for night %s: %d frame(s) from %d night(s) within +/-%d "
            "day(s): %s",
            what, night_name, len(ordered), len(selection.nights), window_nights,
            _describe(selection.night_counts),
        )
    return selection


def _pool_window(
    night_dir: Path,
    current_night: date,
    *,
    frame_type: str,
    target_config: CameraConfig,
    filter_name: str | None,
    window_nights: int,
    target_frames: int,
    explicit_metas: Sequence[FrameMeta],
    scanner: NightScanner,
) -> tuple[list[FrameMeta], int]:
    """Whole-night, nearest-first pooling inside the window.

    Returns ``(frames, n_explicit_only)``; ``frames`` is unordered.
    """
    pooled: list[FrameMeta] = []
    seen: set[str] = set()
    for meta in explicit_metas:
        key = _path_key(meta.path)
        if key not in seen:
            seen.add(key)
            pooled.append(meta)
    explicit_keys = set(seen)

    scanned_keys: set[str] = set()
    window = window_night_dirs(night_dir, current_night, window_nights)
    for index, (_night, path) in enumerate(window):
        # The own night (index 0) is always scanned; every other night only
        # while the pool is still short of the target.
        if index > 0 and len(pooled) >= target_frames:
            break
        for meta in scanner.calib_frames(path):
            if not _matches(meta, frame_type, target_config, filter_name):
                continue
            key = _path_key(meta.path)
            scanned_keys.add(key)
            if key in seen:
                continue
            seen.add(key)
            pooled.append(meta)
    return pooled, len(explicit_keys - scanned_keys)


def pool_bias_frames(
    night_dir: Path,
    current_night: date,
    *,
    target_config: CameraConfig,
    window_nights: int,
    target_frames: int,
    explicit_metas: Sequence[FrameMeta],
    scanner: NightScanner,
    profile: InstrumentProfile,
    settings: Settings,
) -> PoolSelection:
    """Pool bias frames within ``+/- window_nights`` days of ``current_night``."""
    pooled, n_explicit_only = _pool_window(
        night_dir, current_night,
        frame_type="bias", target_config=target_config, filter_name=None,
        window_nights=window_nights, target_frames=target_frames,
        explicit_metas=explicit_metas, scanner=scanner,
    )
    beyond = False
    if not pooled:
        logger.warning(
            "no bias frame within +/-%d day(s) of night %s; falling back to "
            "earlier nights up to %d day(s) back",
            window_nights, night_dir.name, settings.calib.fallback_max_nights_bias,
        )
        result = collect_fallback_frames(
            night_dir, current_night,
            frame_type="bias", target_config=target_config,
            min_count=settings.calib.min_bias_frames,
            max_nights=settings.calib.fallback_max_nights_bias,
            profile=profile, settings=settings,
            skip_within_days=window_nights,
        )
        if result.satisfied:
            pooled = [m for m in result.metas if m.is_valid]
            beyond = True
        else:
            logger.warning(
                "fewer than %d bias frame(s) found for night %s in the previous "
                "%d night(s); continuing without master bias",
                settings.calib.min_bias_frames, night_dir.name,
                settings.calib.fallback_max_nights_bias,
            )
    return _finish(
        pooled, kind="bias", filter_name=None, current_night=current_night,
        window_nights=window_nights, target_frames=target_frames,
        beyond_window=beyond, n_explicit_only=n_explicit_only,
        night_name=night_dir.name,
    )


def _same_readout(a: str | None, b: str | None) -> bool:
    """Whether a master's ``READOUTM`` card ``a`` and a frame's readout mode ``b``
    are the same mode; ``None``, empty and ``UNKNOWN`` all stand for no mode,
    which matches only another no-mode value."""
    return (str(a).strip() if a else "UNKNOWN") == (str(b).strip() if b else "UNKNOWN")


def find_nearest_master_flat(
    roots: Sequence[str | Path],
    current_night: date,
    *,
    filter_name: str,
    gain_setting: float | None,
    readout_mode: str | None,
    min_frames: int,
    exclude: Sequence[Path] = (),
) -> tuple[MasterFlatChoice | None, int]:
    """The existing master flat of ``filter_name`` closest in time to
    ``current_night``, searched in ``<root>/YYYYMMDD/calib/master_flat_*.fits``.

    Returns ``(choice, n_disqualified)``.  "Closest" is the smallest absolute
    difference in calendar days, in either direction and with no maximum; on a
    tie the earlier night wins.  The night itself (distance 0) is never a
    candidate.  A master qualifies when

    * its file name carries ``filter_name`` (the master header has no filter
      card; ``master_flat_<filter>[_dome|_sky].fits``);
    * its ``GAINSET`` card equals the science frames' gain setting (a missing
      card is accepted) and its ``READOUTM`` card equals their readout mode
      (e.g. the profile-derived ``mode0`` / ``mode5``): a master with no
      ``READOUTM`` card or ``READOUTM = UNKNOWN`` (built before the readout mode
      was known) matches only frames without a readout mode and is skipped,
      logged at DEBUG, for a frame with one;
    * its header frame count (``FLATNACC``, else ``FLATNFRM``, else
      ``NCOMBINE``) is at least ``min_frames`` -- a master without any count
      card is accepted.  Masters failing only this test are counted in
      ``n_disqualified``.

    Only primary headers are read.
    """
    from astropy.io import fits

    from robo43.calib.set import _parse_flat_filter_name

    excluded = {Path(p).resolve() for p in exclude}
    best: tuple[tuple[int, date], MasterFlatChoice] | None = None
    n_disqualified = 0
    for root in roots:
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        for entry in sorted(root_path.iterdir()):
            night = parse_night_dirname(entry.name)
            if night is None or night == current_night:
                continue
            for path in sorted((entry / "calib").glob("master_flat_*.fits")):
                if path.resolve() in excluded:
                    continue
                if _parse_flat_filter_name(path.name) != filter_name:
                    continue
                try:
                    header = fits.getheader(path, 0)
                except Exception as exc:
                    logger.debug("master flat scan: cannot read %s: %s", path, exc)
                    continue
                master_gain = header.get("GAINSET")
                if (
                    gain_setting is not None
                    and master_gain is not None
                    and not math.isclose(float(master_gain), float(gain_setting), rel_tol=1e-6)
                ):
                    continue
                master_readout = header.get("READOUTM")
                if not _same_readout(master_readout, readout_mode):
                    logger.debug(
                        "master flat scan: skipping %s: READOUTM %r does not match "
                        "the frames' readout mode %r",
                        path, master_readout, readout_mode,
                    )
                    continue
                n_frames = next(
                    (int(header[k]) for k in _MASTER_FRAME_COUNT_KEYS if k in header), None
                )
                if n_frames is not None and n_frames < min_frames:
                    n_disqualified += 1
                    continue
                distance = (night - current_night).days
                key = (abs(distance), night)
                if best is None or key < best[0]:
                    best = (
                        key,
                        MasterFlatChoice(
                            path=path, night=night, distance_days=distance, n_frames=n_frames
                        ),
                    )
    return (best[1] if best is not None else None), n_disqualified


def install_master_flat(
    choice: MasterFlatChoice,
    calibdir: str | Path,
    *,
    current_night: date,
    window_nights: int,
    target_frames: int,
) -> Path:
    """Copy ``choice.path`` into ``calibdir`` under its own (normal) name and
    add provenance cards to the copy; return the copy's path.

    Cards: ``CALFBACK`` = True; ``FLATNGT`` = the master's own night list if it
    has one, else its night (what :class:`robo43.calib.set.CalibrationSet`
    copies onto every reduced frame); ``FLATREUS`` = True, ``FLATRFIL`` (source
    file name), ``FLATRNGT`` (source night), ``FLATRDST`` (source night minus
    this night, days), ``FLATRNFR`` (frames the source was built from, when
    known), ``FLATWIN`` / ``FLATTGT`` (the pool policy that was not met); a
    HISTORY line carries the full source path.  The cold-pixel mask file named
    by the master's ``COLDPIX`` card, if it sits beside it, is copied too.
    """
    from astropy.io import fits

    calib_path = Path(calibdir)
    calib_path.mkdir(parents=True, exist_ok=True)
    dest = calib_path / choice.path.name
    if dest.resolve() == choice.path.resolve():
        raise ValueError(f"refusing to reuse {choice.path} onto itself")
    tmp = calib_path / f".{choice.path.name}.{current_night:%Y%m%d}.tmp"
    shutil.copyfile(choice.path, tmp)
    try:
        with fits.open(tmp, mode="update", checksum=True) as hdul:
            header = hdul[0].header
            night = choice.night.strftime("%Y%m%d")
            header["CALFBACK"] = (True, "Calibration borrowed from a previous night")
            if "FLATNGT" not in header and "FLATN1" not in header:
                header["FLATNGT"] = (night, "Source night(s) of this master flat")
            header["FLATREUS"] = (True, "Master flat reused from another night")
            header["FLATRFIL"] = (choice.path.name, "Reused master flat file")
            header["FLATRNGT"] = (night, "Night the reused master was built for")
            header["FLATRDST"] = (choice.distance_days, "Reused master night minus this night (d)")
            if choice.n_frames is not None:
                header["FLATRNFR"] = (choice.n_frames, "Frames in the reused master")
            header["FLATWIN"] = (window_nights, "Flat pool window (+/- nights)")
            header["FLATTGT"] = (target_frames, "Flat pool target frames")
            header.add_history(f"Master flat reused from {choice.path}")
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    try:
        coldpix = str(fits.getheader(dest, 0).get("COLDPIX", "NONE"))
    except Exception:
        coldpix = "NONE"
    if coldpix not in ("", "NONE"):
        source_coldpix = choice.path.parent / coldpix
        if source_coldpix.is_file():
            shutil.copyfile(source_coldpix, calib_path / coldpix)
    return dest


def pool_flat_frames_window(
    night_dir: Path,
    current_night: date,
    *,
    target_config: CameraConfig,
    filter_name: str,
    window_nights: int,
    target_frames: int,
    explicit_metas: Sequence[FrameMeta],
    scanner: NightScanner,
    fallback_max_nights: int,
    master_roots: Sequence[str | Path] = (),
) -> PoolSelection:
    """Pool one filter's flats within ``+/- window_nights`` days of
    ``current_night``.

    Flats must match the science frames' camera configuration (readout mode,
    gain setting, binning, frame shape) because the master of one filter is
    built in a single gain group and a mixed group is refused by the builder.

    With ``master_roots`` (the profile's ``flat_master_roots``), a pool short of
    ``target_frames`` -- empty included -- is replaced by the nearest existing
    master flat (see :func:`find_nearest_master_flat`); the short pool's frames
    stay in the selection (the gain measurement uses raw flat pairs) but no
    master is built from them.  When no master qualifies the short pool is
    built with a WARNING, and an empty one means no master flat for the filter.
    Without ``master_roots`` an empty pool falls back to the nearest earlier
    night's frames within ``fallback_max_nights``.
    """
    pooled, n_explicit_only = _pool_window(
        night_dir, current_night,
        frame_type="flat", target_config=target_config, filter_name=filter_name,
        window_nights=window_nights, target_frames=target_frames,
        explicit_metas=explicit_metas, scanner=scanner,
    )
    beyond = False
    reuse: MasterFlatChoice | None = None
    if master_roots:
        if len(pooled) < target_frames:
            reuse, n_disqualified = find_nearest_master_flat(
                master_roots, current_night,
                filter_name=filter_name,
                gain_setting=target_config[1],
                readout_mode=target_config[0],
                min_frames=target_frames,
            )
            if reuse is None:
                logger.warning(
                    "%d %s flat frame(s) within +/-%d day(s) of night %s (target %d) "
                    "and no existing master flat of that filter with >= %d frames "
                    "(%d older master(s) too short) under %s%s",
                    len(pooled), filter_name, window_nights, night_dir.name,
                    target_frames, target_frames, n_disqualified,
                    ", ".join(str(r) for r in master_roots),
                    "; building from the short pool" if pooled
                    else "; continuing without a master flat for that filter",
                )
    elif not pooled:
        logger.warning(
            "no %s flat within +/-%d day(s) of night %s; falling back to the "
            "nearest earlier night within %d day(s)",
            filter_name, window_nights, night_dir.name, fallback_max_nights,
        )
        for _night, path in _previous_night_dirs(
            night_dir, current_night, fallback_max_nights, min_days=window_nights
        ):
            matches = [
                m for m in scanner.calib_frames(path)
                if _matches(m, "flat", target_config, filter_name)
            ]
            if matches:
                pooled = matches
                beyond = True
                break
        else:
            logger.warning(
                "no %s flat found for night %s within %d day(s); continuing "
                "without a master flat for that filter",
                filter_name, night_dir.name, fallback_max_nights,
            )
    return _finish(
        pooled, kind="flat", filter_name=filter_name, current_night=current_night,
        window_nights=window_nights, target_frames=target_frames,
        beyond_window=beyond, n_explicit_only=n_explicit_only,
        night_name=night_dir.name, reuse=reuse,
    )


def infer_science_night(science_metas: Sequence[FrameMeta]) -> tuple[Path, date] | None:
    """The ``(night directory, night)`` shared by every science frame, or
    ``None`` when the frames do not all live in one ``YYYYMMDD`` directory."""
    parents = {Path(m.path).resolve().parent for m in science_metas}
    if len(parents) != 1:
        return None
    parent_as_given = Path(science_metas[0].path).parent
    night = parse_night_dirname(parent_as_given.name)
    if night is None:
        night = parse_night_dirname(next(iter(parents)).name)
        parent_as_given = next(iter(parents))
    if night is None:
        return None
    return parent_as_given, night


def limit_bias_for_gain(metas: Sequence[FrameMeta], max_frames: int) -> list[FrameMeta]:
    """``metas`` with at most ``max_frames`` bias frames per camera
    configuration (the first ones, i.e. the nearest nights); every other frame
    is kept.  ``max_frames <= 0`` keeps everything.

    The gain measurement keeps one full-frame float64 view per bias frame
    alive, so it must not see a pool of 100+ frames.
    """
    if max_frames <= 0:
        return list(metas)
    kept: list[FrameMeta] = []
    counts: dict[tuple[str | None, float | None, int | None], int] = {}
    for meta in metas:
        if meta.frame_type == "bias":
            key = (meta.readout_mode, meta.gain_setting, meta.binning)
            counts[key] = counts.get(key, 0) + 1
            if counts[key] > max_frames:
                continue
        kept.append(meta)
    return kept


def _mode_label(meta: FrameMeta) -> str:
    """A frame's readout mode as compared between frames: ``UNKNOWN`` for none."""
    return (meta.readout_mode or "").strip() or "UNKNOWN"


def restrict_bias_to_mode(
    metas: Sequence[FrameMeta], mode: str | None, *, where: str
) -> list[FrameMeta]:
    """``metas`` without the bias frames whose readout mode is not ``mode``.

    Frames of every other type are kept.  When bias frames are dropped, one
    WARNING names ``where`` and lists the dropped count per readout mode.
    """
    keep_label = (mode or "").strip() or "UNKNOWN"
    kept: list[FrameMeta] = []
    dropped: Counter[str] = Counter()
    for meta in metas:
        if meta.frame_type == "bias" and _mode_label(meta) != keep_label:
            dropped[_mode_label(meta)] += 1
        else:
            kept.append(meta)
    if dropped:
        logger.warning(
            "%s: dropped %d bias frame(s) of a readout mode other than the science "
            "frames' %r (never combined into one master bias); dropped per mode: %s",
            where, sum(dropped.values()), keep_label,
            ", ".join(f"{label}: {count}" for label, count in sorted(dropped.items())),
        )
    return kept


def enforce_single_bias_mode(metas: Sequence[FrameMeta]) -> list[FrameMeta]:
    """Make sure the bias frames of a run share one readout mode.

    A run whose bias frames are all of one mode is returned unchanged.  With
    several modes among them: when the run has valid science frames, only the
    bias frames of the science frames' (most common) mode are kept and the
    others are dropped with a WARNING listing the dropped count per mode
    (:func:`restrict_bias_to_mode`); a calibration-only run (no valid science
    frame to choose a mode by) raises :class:`~robo43.exceptions.ConfigError`
    instead of combining them.
    """
    bias = [m for m in metas if m.frame_type == "bias"]
    per_mode = Counter(_mode_label(m) for m in bias)
    if len(per_mode) <= 1:
        return list(metas)
    science = [m for m in metas if m.frame_type == "science" and m.is_valid]
    if not science:
        msg = (
            "the bias frames span several readout modes ("
            + ", ".join(f"{label}: {count}" for label, count in sorted(per_mode.items()))
            + ") and there are no science frames to choose between them; refusing to "
            "combine different readout modes into one master bias"
        )
        raise ConfigError(msg)
    target = Counter(_mode_label(m) for m in science).most_common(1)[0][0]
    return restrict_bias_to_mode(metas, target, where="master bias inputs")


def apply_window_pools(
    metas: Sequence[FrameMeta],
    *,
    profile: InstrumentProfile,
    settings: Settings,
    night_dir: Path,
    current_night: date,
    filters: Sequence[str] | None = None,
    scanner: NightScanner | None = None,
) -> WindowPoolResult:
    """Replace a run's bias (and per-filter flat) frames by their pools.

    ``metas`` are the run's own inputs; their bias frames and the flats of each
    pooled filter are treated as explicit frames (merged, de-duplicated, never
    dropped).  Which pools run is decided by the instrument profile
    (``bias_pool`` and ``flat_pool`` with ``flat_pool_window_nights > 0``);
    when neither is enabled the input is returned unchanged.
    """
    opts = profile.calib
    bias_active = bool(opts.bias_pool)
    flat_active = bool(opts.flat_pool) and opts.flat_pool_window_nights > 0
    out = list(metas)
    if not bias_active and not flat_active:
        return WindowPoolResult(metas=out)

    scanner = scanner or NightScanner(profile, settings)
    science = [m for m in metas if m.frame_type == "science" and m.is_valid]
    if len({camera_config_key(m) for m in science}) > 1:
        logger.warning(
            "science frames of night %s span several camera configurations; "
            "pooling calibration frames for the dominant one only",
            night_dir.name,
        )

    bias_sel: PoolSelection | None = None
    flat_sels: dict[str, PoolSelection] = {}

    if bias_active:
        config = dominant_camera_config(science)
        if config is None:
            logger.warning(
                "no science frames for night %s; cannot determine a camera "
                "configuration for bias pooling",
                night_dir.name,
            )
        else:
            bias_sel = pool_bias_frames(
                night_dir, current_night,
                target_config=config,
                window_nights=opts.bias_pool_window_nights,
                target_frames=opts.bias_pool_target_frames,
                explicit_metas=[
                    m for m in restrict_bias_to_mode(
                        metas, config[0], where="explicit bias inputs"
                    )
                    if m.frame_type == "bias"
                ],
                scanner=scanner, profile=profile, settings=settings,
            )
            out = [m for m in out if m.frame_type != "bias"] + list(bias_sel.metas)

    if flat_active:
        science_filters = sorted({m.filter for m in science if m.filter is not None})
        if filters:
            science_filters = [f for f in science_filters if f in filters]
        for filter_name in science_filters:
            config = dominant_camera_config([m for m in science if m.filter == filter_name])
            if config is None:
                continue
            flat_sel = pool_flat_frames_window(
                night_dir, current_night,
                target_config=config,
                filter_name=filter_name,
                window_nights=opts.flat_pool_window_nights,
                target_frames=opts.flat_pool_target_frames,
                explicit_metas=[
                    m for m in metas if m.frame_type == "flat" and m.filter == filter_name
                ],
                scanner=scanner,
                fallback_max_nights=settings.calib.fallback_max_nights_flat,
                master_roots=opts.flat_master_roots,
            )
            out = [
                m for m in out
                if not (m.frame_type == "flat" and m.filter == filter_name)
            ] + list(flat_sel.metas)
            flat_sels[filter_name] = flat_sel

    return WindowPoolResult(metas=out, bias=bias_sel, flats=flat_sels)
