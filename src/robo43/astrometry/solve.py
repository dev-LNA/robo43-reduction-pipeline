"""Astrometric solving: index selection, the solver cache, and the fallback ladder.

Wraps the ``astrometry`` package (the local astrometry.net-index solver, not
a network service -- see ``settings.astrometry.use_web_fallback``, which this
module does not implement) around :mod:`robo43.astrometry.centroids`'
detections. The algorithm here -- size-hint tolerance, the hinted-then-blind
ladder, the 2.0" match radius and residual-RMS quality gate -- was validated
empirically against the four ROBO43 mini-sample frames: every frame solves
on the first or second rung in 1.2-2.0 s with RMS 0.15-0.18", well inside the
default 0.3" gate.

Index files are never downloaded by this module. ``astrometry.Series.index_files``
will fetch a missing scale over the network the first time it is asked for
one, which is wrong for an offline reduction machine; :func:`_scales_present_on_disk`
instead checks, with plain filesystem stats, that every file a scale needs is
already cached before that scale is ever passed to ``index_files`` (at which
point every path already exists and the call is a pure no-op lookup, e.g.
0.1-0.2 s to build a :class:`astrometry.Solver`). Built solvers are cached
module-level, keyed by ``(index_dir, series name, frozenset(scales))``, so
that one-time cost is paid at most once per process.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import astrometry
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

from robo43.astrometry.centroids import (
    FALLBACK_LADDER,
    DetectedStar,
    DetectionRung,
    detect_stars_for_rung,
    estimate_background,
    measure_fwhm,
)

if TYPE_CHECKING:
    from robo43.config import AstrometrySettings

logger = logging.getLogger(__name__)

__all__ = [
    "AstrometrySolution",
    "derive_index_scales",
    "solve_frame",
]

#: astrometry.net's own quad-size-to-scale-index table (arcmin), per the
#: interface contract; scales >= 10 double every 2 steps.
_SCALE_TABLE_BASE: dict[int, tuple[float, float]] = {
    0: (2.0, 2.8),
    1: (2.8, 4.0),
    2: (4.0, 5.6),
    3: (5.6, 8.0),
    4: (8.0, 11.0),
    5: (11.0, 16.0),
    6: (16.0, 22.0),
    7: (22.0, 30.0),
    8: (30.0, 42.0),
    9: (42.0, 60.0),
}

#: Series tried in order for each candidate scale set: 5200 (Tycho-2+Gaia-DR2)
#: is recommended for fields under 1 degree -- this instrument's field is
#: ~0.7 deg -- 4100 (Tycho-2 only) for wider fields, 4200 (2MASS) last since
#: its own documentation says most users prefer the other two.
_SERIES_PRIORITY: tuple[Any, ...] = (
    astrometry.series_5200,
    astrometry.series_4100,
    astrometry.series_4200,
)

#: settings.astrometry.scale_tolerance (packaged default 0.03) is too tight to
#: comfortably bracket the ~0.19% discrepancy between this instrument's
#: header-derived plate scale (0.5277"/px) and the solver's own measurement
#: (0.5287"/px); verified against the mini-sample that 0.05 is a safe floor.
_MIN_SCALE_TOLERANCE = 0.05

#: Matching radius for the post-solve residual-RMS quality gate.
_MATCH_RADIUS_ARCSEC = 2.0

#: Not part of settings.photometry.edge_margin_px (that field is for the
#: separate SExtractor photometry catalogue's own cuts): this is the
#: validated spike's own DAOStarFinder border_width/edge margin.
_DEFAULT_EDGE_MARGIN_PX = 10

_SOLVER_CACHE: dict[tuple[str, str, frozenset[int]], astrometry.Solver] = {}
_SOLVER_CACHE_LOCK = threading.Lock()


@dataclasses.dataclass(frozen=True, slots=True)
class AstrometrySolution:
    """The outcome of one :func:`solve_frame` call.

    ``wcs_cards`` is empty and every numeric field is ``None``/``0``/``()``
    when ``solved`` is ``False``; ``failure_reason`` is always set in that
    case. A solve attempt that found an index match but failed the residual
    RMS or minimum-matched-sources gate still reports its RMS/matched count
    here (via ``failure_reason``) rather than collapsing to "no match".
    """

    solved: bool
    wcs_cards: tuple[tuple[str, Any, str], ...]
    rms_arcsec: float | None
    n_detected: int
    n_matched: int
    logodds: float | None
    scale_arcsec_per_pixel: float | None
    center_ra_deg: float | None
    center_dec_deg: float | None
    sip_order: int
    rung: int
    blind: bool
    index_series: str | None
    index_scales: tuple[int, ...]
    failure_reason: str | None


def _scale_range_arcmin(n: int) -> tuple[float, float]:
    """The astrometry.net quad-size range (arcmin) for scale index ``n``."""
    if n < 0:
        msg = f"scale index must be >= 0, got {n}"
        raise ValueError(msg)
    if n in _SCALE_TABLE_BASE:
        return _SCALE_TABLE_BASE[n]
    lo, hi = _scale_range_arcmin(n - 2)
    return lo * 2.0, hi * 2.0


def derive_index_scales(
    field_width_arcmin: float,
    field_height_arcmin: float,
    *,
    min_frac: float = 0.10,
    max_frac: float = 1.0,
    max_scale: int = 19,
) -> tuple[int, ...]:
    """Candidate index scales whose quad-size range overlaps
    ``[min_frac, max_frac]`` of the smaller field dimension.

    Returns every overlapping scale from 0 to ``max_scale``; the caller
    intersects this with whichever scales are actually cached on disk (see
    :func:`_scales_present_on_disk`) so an untuned, generous candidate set
    here never causes a network fetch.
    """
    min_dim = min(field_width_arcmin, field_height_arcmin)
    target_lo = min_frac * min_dim
    target_hi = max_frac * min_dim
    scales = []
    for n in range(0, max_scale + 1):
        lo, hi = _scale_range_arcmin(n)
        if hi > target_lo and lo < target_hi:
            scales.append(n)
    return tuple(scales)


def _scales_present_on_disk(
    series: Any, index_dir: Path, candidate_scales: tuple[int, ...]
) -> set[int]:
    """Which of ``candidate_scales`` have every one of their index files
    already cached under ``index_dir/<series.name>/``. Pure filesystem
    stats -- never triggers a download."""
    root = Path(index_dir).resolve() / series.name
    present: set[int] = set()
    for scale in candidate_scales:
        sizes = series.scale_to_sizes.get(scale)
        if not sizes:
            continue
        if len(sizes) == 1:
            url = series.url_pattern.format(scale=f"{scale:02d}", index="")
            paths = [root / url.rsplit("/", 1)[1]]
        else:
            paths = []
            for index in range(len(sizes)):
                url = series.url_pattern.format(scale=f"{scale:02d}", index=f"-{index:02d}")
                paths.append(root / url.rsplit("/", 1)[1])
        if paths and all(p.is_file() for p in paths):
            present.add(scale)
    return present


def _get_cached_solver(index_dir: Path, series: Any, scales: frozenset[int]) -> astrometry.Solver:
    key = (str(Path(index_dir).resolve()), series.name, scales)
    with _SOLVER_CACHE_LOCK:
        solver = _SOLVER_CACHE.get(key)
        if solver is None:
            index_files = series.index_files(cache_directory=index_dir, scales=set(scales))
            solver = astrometry.Solver(index_files)
            _SOLVER_CACHE[key] = solver
        return solver


def _select_index_files(
    index_dir: Path, candidate_scales: tuple[int, ...]
) -> tuple[astrometry.Solver, str, tuple[int, ...]] | None:
    """First series (in :data:`_SERIES_PRIORITY` order) with at least one of
    ``candidate_scales`` fully cached, and a solver built from exactly the
    cached subset. ``None`` when no series has any of them."""
    for series in _SERIES_PRIORITY:
        present = _scales_present_on_disk(series, index_dir, candidate_scales)
        if present:
            solver = _get_cached_solver(index_dir, series, frozenset(present))
            return solver, series.name, tuple(sorted(present))
    return None


def _capped_rung(rung: DetectionRung, cap: int) -> DetectionRung:
    """``rung`` with ``max_stars`` reduced to ``cap`` when ``cap`` is smaller.
    Never raises ``max_stars``: the ladder's own validated values are an
    upper bound, ``settings.astrometry.max_sources`` a further cap on top."""
    if cap and cap < rung.max_stars:
        return dataclasses.replace(rung, max_stars=cap)
    return rung


def _attempt_solve(
    solver: astrometry.Solver,
    stars: tuple[DetectedStar, ...],
    size_hint: astrometry.SizeHint,
    *,
    position_hint: astrometry.PositionHint | None,
    sip_order: int,
) -> astrometry.Match | None:
    stars_xy = [(s.x, s.y) for s in stars]
    solution = solver.solve(
        stars=stars_xy,
        size_hint=size_hint,
        position_hint=position_hint,
        solution_parameters=astrometry.SolutionParameters(
            sip_order=sip_order,
            # STOP on the first candidate that clears astrometry.net's own
            # log-odds threshold: this is what makes a solve 0.2-2 s instead
            # of exhausting every quad in the index.
            logodds_callback=lambda _logodds: astrometry.Action.STOP,
        ),
    )
    if not solution.has_match():
        return None
    return solution.best_match()


def _unsolved(*, reason: str, n_detected: int = 0) -> AstrometrySolution:
    return AstrometrySolution(
        solved=False,
        wcs_cards=(),
        rms_arcsec=None,
        n_detected=n_detected,
        n_matched=0,
        logodds=None,
        scale_arcsec_per_pixel=None,
        center_ra_deg=None,
        center_dec_deg=None,
        sip_order=0,
        rung=-1,
        blind=False,
        index_series=None,
        index_scales=(),
        failure_reason=reason,
    )


def _build_solution(
    match: astrometry.Match,
    stars: tuple[DetectedStar, ...],
    *,
    rung: int,
    blind: bool,
    series_name: str,
    scales_used: tuple[int, ...],
    sip_order: int,
    settings: AstrometrySettings,
) -> AstrometrySolution:
    header = fits.Header()
    for key, value in match.wcs_fields.items():
        # value is already the (value, comment) tuple astropy.io.fits.Header
        # wants; fits.Header(dict_of_tuples) itself raises, hence the loop.
        header[key] = value
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs = WCS(header)

    xs = np.array([s.x - 1.0 for s in stars])
    ys = np.array([s.y - 1.0 for s in stars])
    detected_sky = wcs.pixel_to_world(xs, ys)

    ref_ra = np.array([s.ra_deg for s in match.stars])
    ref_dec = np.array([s.dec_deg for s in match.stars])
    n_matched = 0
    rms_arcsec: float | None = None
    if ref_ra.size > 0:
        ref_sky = SkyCoord(ra=ref_ra * u.deg, dec=ref_dec * u.deg)
        _, sep2d, _ = detected_sky.match_to_catalog_sky(ref_sky)
        good = sep2d.arcsec < _MATCH_RADIUS_ARCSEC
        n_matched = int(np.count_nonzero(good))
        if n_matched > 0:
            rms_arcsec = float(np.sqrt(np.mean(sep2d.arcsec[good] ** 2)))

    solved = (
        rms_arcsec is not None
        and rms_arcsec <= settings.rms_max_arcsec
        and n_matched >= settings.min_sources
    )
    failure_reason: str | None = None
    if not solved:
        if rms_arcsec is None:
            failure_reason = "no detected star matched an index star within 2.0 arcsec"
        elif n_matched < settings.min_sources:
            failure_reason = f"matched sources {n_matched} below min_sources={settings.min_sources}"
        else:
            failure_reason = (
                f'residual rms {rms_arcsec:.3f}" exceeds rms_max_arcsec={settings.rms_max_arcsec}'
            )

    wcs_cards = tuple((card.keyword, card.value, card.comment) for card in header.cards)

    return AstrometrySolution(
        solved=solved,
        wcs_cards=wcs_cards,
        rms_arcsec=rms_arcsec,
        n_detected=len(stars),
        n_matched=n_matched,
        logodds=float(match.logodds),
        scale_arcsec_per_pixel=float(match.scale_arcsec_per_pixel),
        center_ra_deg=float(match.center_ra_deg),
        center_dec_deg=float(match.center_dec_deg),
        sip_order=sip_order,
        rung=rung,
        blind=blind,
        index_series=series_name,
        index_scales=scales_used,
        failure_reason=failure_reason,
    )


def solve_frame(
    sci: np.ndarray,
    dq: np.ndarray,
    *,
    ra_deg: float | None,
    dec_deg: float | None,
    pixel_scale_arcsec: float,
    seeing_fwhm_arcsec: float,
    settings: AstrometrySettings,
    profile_index_scales: tuple[int, ...] | None = None,
    force_blind: bool = False,
) -> AstrometrySolution:
    """Solve one frame's WCS: detect stars, then the hinted/blind fallback ladder.

    ``settings`` is the ``AstrometrySettings`` section (``index_dir``,
    ``index_scales``, ``search_radius_deg``, ``scale_tolerance``,
    ``frame_budget_s``, ``max_sources``, ``min_sources``, ``rms_max_arcsec``,
    ``sip_order`` are used; see this package's ``solve.py`` module docstring
    and the caller's report for the fields that are not). ``profile_index_scales``
    is the instrument profile's own ``[instrument.astrometry] index_scales``
    (``None`` when the profile says ``"derive"``), used only when ``settings``
    itself does not specify an explicit scale set.

    Never raises: every failure path returns an ``AstrometrySolution`` with
    ``solved=False`` and ``failure_reason`` set.
    """
    start = time.monotonic()

    if settings.index_dir is None:
        return _unsolved(reason="astrometry.index_dir is not configured")
    index_dir = Path(settings.index_dir)

    if not np.isfinite(pixel_scale_arcsec) or pixel_scale_arcsec <= 0:
        return _unsolved(reason=f"invalid pixel scale for astrometry: {pixel_scale_arcsec!r}")

    height, width = sci.shape
    field_w_arcmin = width * pixel_scale_arcsec / 60.0
    field_h_arcmin = height * pixel_scale_arcsec / 60.0

    explicit_scales = settings.index_scales or profile_index_scales
    candidate_scales = (
        explicit_scales if explicit_scales else derive_index_scales(field_w_arcmin, field_h_arcmin)
    )

    selection = _select_index_files(index_dir, candidate_scales)
    if selection is None:
        return _unsolved(
            reason=(
                f"no astrometry index files cached under {index_dir} for candidate "
                f"scales {candidate_scales}"
            )
        )
    solver, series_name, scales_used = selection

    tolerance = max(settings.scale_tolerance, _MIN_SCALE_TOLERANCE)
    size_hint = astrometry.SizeHint(
        lower_arcsec_per_pixel=pixel_scale_arcsec * (1.0 - tolerance),
        upper_arcsec_per_pixel=pixel_scale_arcsec * (1.0 + tolerance),
    )

    have_pointing = ra_deg is not None and dec_deg is not None
    position_hint = (
        astrometry.PositionHint(
            ra_deg=ra_deg, dec_deg=dec_deg, radius_deg=settings.search_radius_deg
        )
        if have_pointing and not force_blind
        else None
    )

    med, sd = estimate_background(sci)
    fallback_px = seeing_fwhm_arcsec / pixel_scale_arcsec
    fwhm_px = measure_fwhm(sci, med, sd, fallback_px=fallback_px)

    last_stars: tuple[DetectedStar, ...] = ()
    last_failed: AstrometrySolution | None = None

    for rung_index, rung in enumerate(FALLBACK_LADDER):
        if settings.frame_budget_s and (time.monotonic() - start) > settings.frame_budget_s:
            return _unsolved(
                reason="astrometry.frame_budget_s exceeded before every rung was tried"
            )

        stars = detect_stars_for_rung(
            sci,
            dq,
            fwhm_px=fwhm_px,
            background_med=med,
            background_sd=sd,
            rung=_capped_rung(rung, settings.max_sources),
            edge_margin_px=_DEFAULT_EDGE_MARGIN_PX,
        )
        if not stars:
            continue
        last_stars = stars

        blind = position_hint is None
        match = _attempt_solve(
            solver, stars, size_hint, position_hint=position_hint, sip_order=settings.sip_order
        )
        if match is not None:
            solution = _build_solution(
                match,
                stars,
                rung=rung_index,
                blind=blind,
                series_name=series_name,
                scales_used=scales_used,
                sip_order=settings.sip_order,
                settings=settings,
            )
            if solution.solved:
                return solution
            last_failed = solution

    if position_hint is not None and last_stars:
        logger.info(
            "astrometry: hinted solve failed on every rung; retrying blind (no position hint)"
        )
        match = _attempt_solve(
            solver, last_stars, size_hint, position_hint=None, sip_order=settings.sip_order
        )
        if match is not None:
            solution = _build_solution(
                match,
                last_stars,
                rung=len(FALLBACK_LADDER),
                blind=True,
                series_name=series_name,
                scales_used=scales_used,
                sip_order=settings.sip_order,
                settings=settings,
            )
            if solution.solved:
                return solution
            last_failed = solution

    if last_failed is not None:
        return last_failed
    return _unsolved(
        reason="no astrometric match found (all hinted rungs and the blind retry failed)",
        n_detected=len(last_stars),
    )
