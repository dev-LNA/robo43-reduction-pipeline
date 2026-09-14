"""Star centroiding for astrometric solving: background, FWHM, detection.

This module turns a reduced ``SCI`` plane into the short list of bright,
well-behaved stars an astrometry.net-style solver needs. The background
estimate, the flux-weighted-moment FWHM measurement, and the two-rung
DAOStarFinder fallback ladder below were validated empirically against the
four ROBO43 mini-sample ``_proc.fits`` frames (see the astrometry spike this
module implements): every frame solves on the first or second rung in
1.2-2.0 s, with residual RMS 0.15-0.18". None of these numbers -- the rung
thresholds, the 30-sigma peak-finding threshold, the 8-pixel stamp half-width
-- are free parameters; they are the validated configuration and must not be
tuned further here.

A fixed FWHM fails on one of the four mini-sample frames (Halpha_0002), which
is why :func:`measure_fwhm` is not optional: every frame gets its own
flux-weighted-moment measurement from its own brightest peaks, falling back
to the instrument's nominal seeing only when too few peaks yield a usable
moment.

Saturated pixels are identified from the ``DQ`` plane's ``SATURATED`` bit --
already computed correctly against the frame's own ADU-scale saturation
level earlier in the reduction, before the gain-to-electrons step -- rather
than re-deriving a saturation threshold against ``SCI``, which by the time a
``_proc.fits`` is written is already in electrons. ``DQ.COSMIC_RAY`` is
deliberately never checked here: a pre-existing bug elsewhere in the pipeline
flags roughly a third of all pixels as cosmic rays, but zero of the brightest
300 peaks in the mini-sample land on a CR-flagged pixel, so star cores are
intact and masking on that bit would only discard good detections.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder, find_peaks

from robo43.products.dq import DQ

__all__ = [
    "FALLBACK_LADDER",
    "DetectedStar",
    "DetectionResult",
    "DetectionRung",
    "detect_stars",
    "detect_stars_for_rung",
    "estimate_background",
    "measure_fwhm",
]

#: A raw DAOStarFinder result below this size is treated as "this rung found
#: nothing usable" and the caller moves on to the next rung.
_MIN_RAW_DETECTIONS = 10

#: How many of the brightest peaks feed the flux-weighted-moment FWHM fit.
_FWHM_STAMP_HALF_WIDTH = 8
_FWHM_MAX_PEAKS = 120
_FWHM_PEAK_SIGMA = 30.0
_FWHM_MIN_VALID_STAMPS = 5
_FWHM_SIGMA_BOUNDS = (0.5, 10.0)
_FWHM_SIGMA_TO_FWHM = 2.3548


@dataclass(frozen=True, slots=True)
class DetectionRung:
    """One rung of the DAOStarFinder fallback ladder."""

    threshold_sigma: float
    sharplo: float
    sharphi: float
    round_limit: float
    max_stars: int


#: Rung 0 is deliberately looser than photutils' own defaults
#: (sharplo=0.2, sharphi=1.0, round_limit=0.5), which were verified to fail
#: on the Halpha_0002 mini-sample frame. Tried in order; the first rung
#: yielding >= 10 raw DAOStarFinder detections is used.
FALLBACK_LADDER: tuple[DetectionRung, ...] = (
    DetectionRung(threshold_sigma=5.0, sharplo=0.10, sharphi=1.5, round_limit=0.75, max_stars=250),
    DetectionRung(threshold_sigma=3.0, sharplo=0.05, sharphi=2.0, round_limit=0.90, max_stars=400),
)


@dataclass(frozen=True, slots=True)
class DetectedStar:
    """One detected source, in 1-based FITS pixel coordinates."""

    x: float
    y: float
    flux: float
    peak: float
    sharpness: float
    roundness: float


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """The star list :func:`detect_stars` produced, plus the diagnostics
    (FWHM, background, which rung) that produced it."""

    stars: tuple[DetectedStar, ...]
    fwhm_px: float
    background_med: float
    background_sd: float
    rung: int


def estimate_background(
    sci: np.ndarray, *, stride: int = 4, sigma: float = 3.0, maxiters: int = 5
) -> tuple[float, float]:
    """Sigma-clipped ``(median, stddev)`` from a strided subsample of ``sci``.

    The strided subsample (every 4th pixel by default) is what keeps this
    under 0.1 s on a full 4788x3194 frame; :class:`~photutils.background.Background2D`
    is deliberately not used here (validated against the mini-sample: this
    estimator is sufficient and an order of magnitude faster).
    """
    _, med, sd = sigma_clipped_stats(sci[::stride, ::stride], sigma=sigma, maxiters=maxiters)
    return float(med), float(sd)


def measure_fwhm(sci: np.ndarray, med: float, sd: float, *, fallback_px: float) -> float:
    """Flux-weighted second-moment FWHM (pixels) from the brightest peaks.

    Finds up to 4000 peaks at ``med + 30*sd``, keeps the brightest
    :data:`_FWHM_MAX_PEAKS`, and on each of their (2*8+1)-pixel stamps
    (background-subtracted, negatives clipped to zero) computes a
    flux-weighted centroid and second moments ``sxx``/``syy``; a stamp's
    sigma is ``sqrt(0.5*(sxx+syy))``, kept only if it falls in
    ``(0.5, 10.0)`` pixels. Returns ``2.3548 * median(sigma)`` when at least 5
    stamps yield a usable sigma, otherwise ``fallback_px`` unchanged.
    """
    table = find_peaks(
        sci,
        threshold=med + _FWHM_PEAK_SIGMA * sd,
        box_size=15,
        border_width=_FWHM_STAMP_HALF_WIDTH + 2,
        npeaks=4000,
    )
    if table is None or len(table) < _FWHM_MIN_VALID_STAMPS:
        return fallback_px

    table.sort("peak_value", reverse=True)
    table = table[:_FWHM_MAX_PEAKS]

    half = _FWHM_STAMP_HALF_WIDTH
    axis = np.arange(-half, half + 1)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    stamp_size = 2 * half + 1
    sigma_lo, sigma_hi = _FWHM_SIGMA_BOUNDS

    sigmas: list[float] = []
    for xp, yp in zip(table["x_peak"], table["y_peak"], strict=True):
        cut = sci[yp - half : yp + half + 1, xp - half : xp + half + 1] - med
        if cut.shape != (stamp_size, stamp_size):
            continue
        weights = np.clip(cut, 0, None)
        total = float(weights.sum())
        if total <= 0:
            continue
        cx = float((weights * xx).sum() / total)
        cy = float((weights * yy).sum() / total)
        sxx = float((weights * (xx - cx) ** 2).sum() / total)
        syy = float((weights * (yy - cy) ** 2).sum() / total)
        if not np.isfinite(sxx + syy) or sxx <= 0 or syy <= 0:
            continue
        sigma = float(np.sqrt(0.5 * (sxx + syy)))
        if sigma_lo < sigma < sigma_hi:
            sigmas.append(sigma)

    if len(sigmas) < _FWHM_MIN_VALID_STAMPS:
        return fallback_px
    return float(_FWHM_SIGMA_TO_FWHM * np.median(sigmas))


def detect_stars_for_rung(
    sci: np.ndarray,
    dq: np.ndarray,
    *,
    fwhm_px: float,
    background_med: float,
    background_sd: float,
    rung: DetectionRung,
    edge_margin_px: int = 10,
) -> tuple[DetectedStar, ...]:
    """Run one :class:`DetectionRung` of DAOStarFinder, applying the saturation,
    edge, and non-finite filters. Returns an empty tuple when this rung finds
    fewer than 10 raw sources (the rung is considered to have failed); never
    raises.

    ``brightest`` is set to twice ``rung.max_stars`` so the flux-sort/cut
    below still has enough candidates after the filters remove some.
    Returned coordinates are 1-based FITS pixel coordinates (photutils
    centroids are 0-based).
    """
    finder = DAOStarFinder(
        fwhm=fwhm_px,
        threshold=rung.threshold_sigma * background_sd,
        sharplo=rung.sharplo,
        sharphi=rung.sharphi,
        roundlo=-rung.round_limit,
        roundhi=rung.round_limit,
        brightest=rung.max_stars * 2,
        exclude_border=True,
    )
    sources = finder(sci - background_med)
    if sources is None or len(sources) < _MIN_RAW_DETECTIONS:
        return ()

    sources.sort("flux", reverse=True)
    height, width = sci.shape
    finite = np.isfinite(sci)

    stars: list[DetectedStar] = []
    for row in sources:
        xc = float(row["xcentroid"])
        yc = float(row["ycentroid"])
        xi = round(xc)
        yi = round(yc)
        if xi < edge_margin_px or xi >= width - edge_margin_px:
            continue
        if yi < edge_margin_px or yi >= height - edge_margin_px:
            continue
        if not finite[yi, xi]:
            continue
        if int(dq[yi, xi]) & int(DQ.SATURATED):
            continue
        stars.append(
            DetectedStar(
                x=xc + 1.0,
                y=yc + 1.0,
                flux=float(row["flux"]),
                peak=float(row["peak"]),
                sharpness=float(row["sharpness"]),
                roundness=float(row["roundness1"]),
            )
        )
        if len(stars) >= rung.max_stars:
            break

    return tuple(stars)


def detect_stars(
    sci: np.ndarray,
    dq: np.ndarray,
    *,
    pixel_scale_arcsec: float,
    seeing_fwhm_arcsec: float,
    edge_margin_px: int = 10,
    ladder: Sequence[DetectionRung] = FALLBACK_LADDER,
) -> DetectionResult:
    """Background estimate, FWHM measurement, then the fallback ladder.

    Tries each rung of ``ladder`` in order and returns the first one that
    yields at least one usable star (see :func:`detect_stars_for_rung`);
    ``result.rung`` records which. When every rung fails, ``result.stars``
    is empty and ``result.rung`` is ``-1``. Never raises.
    """
    med, sd = estimate_background(sci)
    fallback_px = seeing_fwhm_arcsec / pixel_scale_arcsec if pixel_scale_arcsec > 0 else 4.0
    fwhm_px = measure_fwhm(sci, med, sd, fallback_px=fallback_px)

    for rung_index, rung in enumerate(ladder):
        stars = detect_stars_for_rung(
            sci,
            dq,
            fwhm_px=fwhm_px,
            background_med=med,
            background_sd=sd,
            rung=rung,
            edge_margin_px=edge_margin_px,
        )
        if stars:
            return DetectionResult(
                stars=stars, fwhm_px=fwhm_px, background_med=med, background_sd=sd, rung=rung_index
            )

    return DetectionResult(stars=(), fwhm_px=fwhm_px, background_med=med, background_sd=sd, rung=-1)
