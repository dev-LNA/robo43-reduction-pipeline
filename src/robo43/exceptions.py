"""The exception hierarchy for the pipeline.

Helpers in this package either return a valid result or raise one of these.
Returning an ``Exception`` instance as data -- which the legacy scripts did in
``run_sewpy``, ``run_daofinder`` and ``run_astrometry_solver1`` -- makes the
control flow depend on the ``TypeError`` the caller hits later, and is banned.
"""

from __future__ import annotations

__all__ = [
    "AstrometryError",
    "CalibrationError",
    "ConfigError",
    "DatabaseError",
    "DetectionError",
    "ExternalToolError",
    "MetadataError",
    "PhotometryError",
    "ProductError",
    "ReductionError",
    "Robo43Error",
    "ValidationError",
]


class Robo43Error(Exception):
    """Base class for every error the pipeline raises deliberately."""


class ConfigError(Robo43Error):
    """Malformed configuration file, unknown key, or unusable value."""


class MetadataError(Robo43Error):
    """A frame's metadata violates the keyword contract."""


class ValidationError(MetadataError):
    """Preflight validation failed; the run must not start."""


class CalibrationError(Robo43Error):
    """A master calibration frame could not be built or is unusable."""


class ReductionError(Robo43Error):
    """Per-frame reduction failed."""


class DetectionError(Robo43Error):
    """Source detection produced no usable source list."""


class AstrometryError(Robo43Error):
    """No acceptable astrometric solution was found."""


class PhotometryError(Robo43Error):
    """Photometric measurement or calibration failed."""


class ProductError(Robo43Error):
    """A product file could not be assembled or written."""


class DatabaseError(Robo43Error):
    """Database connection, query, or ingest failure."""


class ExternalToolError(Robo43Error):
    """A required external binary is missing or exited non-zero."""
