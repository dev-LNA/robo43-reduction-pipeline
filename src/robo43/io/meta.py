"""Per-frame metadata and where each value came from.

:class:`FrameMeta` is the pipeline's normalized view of a FITS frame's
header: tier 1 fields are always required, tier 2 fields are gated by which
stage is running, and tier 3 fields merely enrich the products.  Every value
also carries a :class:`FieldOrigin` recording which layer supplied it --
raw header, alias lookup, an override file, the instrument profile, a
derived computation, or a hard-coded default -- so a surprising number can
be traced back to its source (``robo43 config show``'s per-frame cousin).

``gain_setting`` and ``readout_mode`` are the camera's own dimensionless
GAIN index and READOUTM string, resolved per frame exactly like any other
tier-2 field.  Neither is a physical e-/ADU conversion factor: the QHY600M
camera's ``GAIN`` keyword is a vendor setting index, not a gain, so no field
here holds an e-/ADU value.  The physical conversion is measured at runtime
by :mod:`robo43.calib.gain` from that night's flats and biases, keyed by
``(readout_mode, gain_setting, binning)``, and is looked up from the
:class:`~robo43.calib.set.CalibrationSet` at reduction time -- never carried
on ``FrameMeta`` itself.
"""

from __future__ import annotations

import datetime
import enum
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astropy.time import Time


class Source(enum.StrEnum):
    HEADER = "header"
    ALIAS = "alias"
    OVERRIDE = "override"
    PROFILE = "profile"
    DERIVED = "derived"
    DEFAULT = "default"


@dataclass(frozen=True, slots=True)
class FieldOrigin:
    source: Source
    detail: str = ""


@dataclass(frozen=True, slots=True)
class FrameMeta:
    """Normalized metadata for one frame, plus per-field provenance."""

    path: Path
    # tier 1 -- always required
    naxis1: int
    naxis2: int
    frame_type: str | None
    exptime: float | None
    date_obs: Time | None
    # tier 2 -- stage-gated
    filter: str | None
    ra_deg: float | None
    dec_deg: float | None
    pixel_scale: float | None
    ccd_temp: float | None
    gain_setting: float | None
    read_noise: float | None
    saturation: float | None
    airmass: float | None
    # tier 3 -- enriching
    object: str | None
    telescope: str | None
    instrument: str | None
    observer: str | None
    binning: int | None
    readout_mode: str | None
    focal_length_mm: float | None
    site_lat_deg: float | None
    site_lon_deg: float | None
    site_elevation_m: float | None
    mjd: float | None
    night: datetime.date | None
    is_valid: bool
    profile_name: str
    origins: Mapping[str, FieldOrigin]

    def __hash__(self) -> int:
        return hash(self.path)

    def get(self, field: str) -> Any:
        return getattr(self, field)

    def origin(self, field: str) -> FieldOrigin | None:
        return self.origins.get(field)

    def missing(self, fields: Iterable[str]) -> tuple[str, ...]:
        return tuple(name for name in fields if self.get(name) is None)

    def provenance_records(self) -> list[str]:
        return [
            f"{field}={origin.source.value}:{origin.detail}"
            for field, origin in self.origins.items()
            if origin.source not in (Source.HEADER, Source.ALIAS)
        ]

    def provenance_json(self) -> dict[str, dict[str, str]]:
        return {
            field: {"source": origin.source.value, "detail": origin.detail}
            for field, origin in self.origins.items()
            if origin.source not in (Source.HEADER, Source.ALIAS)
        }
