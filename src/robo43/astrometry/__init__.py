"""Astrometric WCS solving for ``_proc.fits`` products.

:mod:`robo43.astrometry.centroids` turns a reduced ``SCI`` plane into a star
list; :mod:`robo43.astrometry.solve` turns that star list into a WCS via a
local astrometry.net-index solve; :mod:`robo43.astrometry.wcs_header` writes
the result into a MEF. ``robo43 astrometry`` (see ``robo43.cli.astrometry``)
is the command-line entry point that ties the three together.
"""

from __future__ import annotations

from robo43.astrometry.centroids import (
    FALLBACK_LADDER,
    DetectedStar,
    DetectionResult,
    DetectionRung,
    detect_stars,
    detect_stars_for_rung,
    estimate_background,
    measure_fwhm,
)
from robo43.astrometry.solve import AstrometrySolution, derive_index_scales, solve_frame
from robo43.astrometry.wcs_header import MIRRORED_HDU_NAMES, apply_solution, astrometry_cards

__all__ = [
    "FALLBACK_LADDER",
    "MIRRORED_HDU_NAMES",
    "AstrometrySolution",
    "DetectedStar",
    "DetectionResult",
    "DetectionRung",
    "apply_solution",
    "astrometry_cards",
    "derive_index_scales",
    "detect_stars",
    "detect_stars_for_rung",
    "estimate_background",
    "measure_fwhm",
    "solve_frame",
]
