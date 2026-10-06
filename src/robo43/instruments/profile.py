"""Instrument profiles: per-telescope keywords, layout, and site geometry.

A profile describes everything that varies from telescope to telescope --
plate scale, gain, saturation, the filter map, header keyword aliases, and
site location -- so the rest of the pipeline can stay instrument-agnostic.
Profiles are loaded from TOML files, either packaged with the distribution
(``robo43.data.instruments``) or dropped by a user into
``~/.config/robo43/instruments``, where a user file shadows a packaged one of
the same name.
"""

from __future__ import annotations

import logging
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import astropy.units as u
from astropy.coordinates import EarthLocation

from robo43.exceptions import ConfigError
from robo43.instruments.keywords import DEFAULT_IMAGETYP_MAP, DEFAULT_KEYWORD_ALIASES

if TYPE_CHECKING:
    from astropy.io import fits

logger = logging.getLogger(__name__)

USER_PROFILE_DIR = Path.home() / ".config" / "robo43" / "instruments"

#: Known keys inside the top-level ``[instrument]`` table.
_KNOWN_INSTRUMENT_KEYS = {
    "name",
    "matches",
    "pixel_scale_arcsec",
    "gain_e_per_adu",
    "read_noise_e",
    "saturation_adu",
    "nonlinearity_adu",
    "binning_keyword",
    "trust_header_wcs",
    "calib",
    "layout",
    "filters",
    "keywords",
    "imagetyp_map",
    "site",
    "astrometry",
    "version",
}


@dataclass(frozen=True, slots=True)
class InstrumentLayout:
    """CCD readout geometry: HDU, amplifiers, overscan/trim regions."""

    science_hdu: int | str = 0
    n_amplifiers: int = 1
    overscan: str | None = None
    trim: str | None = None
    flip_y: bool = False


@dataclass(frozen=True, slots=True)
class InstrumentCalibOptions:
    """Optional per-instrument calibration-fallback policy (``[instrument.calib]``).

    Defaults mean "pooling off": an instrument profile that omits this table
    keeps today's zero-trigger flat fallback
    (:func:`robo43.calib.fallback.collect_fallback_frames`) unchanged.
    Setting ``flat_pool = true`` switches every filter's flat build to the
    always-on multi-night pooling policy
    (:func:`robo43.calib.fallback.pool_flat_frames`): pool the current
    night's own flats with every matching flat from the previous
    ``flat_pool_min_nights`` calendar days, extend night by night up to
    ``flat_pool_max_nights`` days if the pool is still short of
    ``flat_pool_target_frames``, and accept fewer than the target rather
    than searching further within that window.

    Two further, independent policies pool on **both** sides of the night
    (:mod:`robo43.calib.pool`); both default to off, and a profile that does
    not set their keys behaves exactly as before:

    * ``bias_pool = true`` pools the master bias from the night's own bias
      plus sibling nights within ``bias_pool_window_nights`` calendar days
      (earlier and later), nearest first, whole nights, until at least
      ``bias_pool_target_frames`` frames.  ``bias_pool_gain_max_frames`` caps
      how many of those frames the runtime gain measurement reads (it keeps
      one full-frame array alive per bias frame); 0 = no cap.
    * ``flat_pool = true`` together with ``flat_pool_window_nights > 0``
      replaces the backward-only flat pool above by the same symmetric,
      nearest-first, whole-night pool within ``flat_pool_window_nights``
      days, until at least ``flat_pool_target_frames`` frames;
      ``flat_pool_min_nights`` / ``flat_pool_max_nights`` are then unused.
      With ``flat_master_roots`` (a list of directories ``<root>/YYYYMMDD/
      calib/`` holding earlier reductions' masters) a flat pool short of the
      target is replaced by the existing master of that filter closest in
      time (see :func:`robo43.calib.pool.find_nearest_master_flat`); without
      it a short pool is built as-is.

    In both, only frames matching the science frames' camera configuration
    (readout mode, gain setting, binning, frame shape) and, for flats, the
    filter are counted.
    """

    flat_pool: bool = False
    flat_pool_min_nights: int = 10
    flat_pool_max_nights: int = 20
    flat_pool_target_frames: int = 20
    flat_pool_window_nights: int = 0
    flat_master_roots: tuple[str, ...] = ()
    bias_pool: bool = False
    bias_pool_window_nights: int = 10
    bias_pool_target_frames: int = 20
    bias_pool_gain_max_frames: int = 12


@dataclass(frozen=True, slots=True)
class InstrumentProfile:
    """Everything the pipeline needs to know about one instrument."""

    name: str
    matches: dict[str, tuple[str, ...]]
    pixel_scale_arcsec: float | None
    # A static e-/ADU conversion, loadable for an instrument whose gain truly
    # is a fixed constant. ROBO43's profile omits it: the QHY600M's GAIN
    # header keyword is a vendor setting index, not a conversion factor, so
    # nothing may fall back to this field for it -- the physical value is
    # measured at runtime instead (see robo43.calib.gain). ``None`` here
    # means "not a fixed instrument property"; it is never silently assumed.
    gain_e_per_adu: float | None
    read_noise_e: float | None
    saturation_adu: float | None
    nonlinearity_adu: float | None
    binning_keyword: str
    layout: InstrumentLayout
    valid_filters: tuple[str, ...]
    filter_map: dict[str, str]
    keywords: dict[str, tuple[str, ...]]
    imagetyp_map: dict[str, str]
    site_lat_deg: float | None
    site_lon_deg: float | None
    site_elevation_m: float | None
    index_scales: tuple[int, ...] | None
    search_radius_deg: float
    # Optional Gaia-referenced absolute-calibration maps (robo43.photometry.gaia_zp):
    # canonical filter name -> gaiaxpy PhotometricSystem.JPLUS band name, and
    # canonical filter name -> robo43.photometry.gaia_zp.GAIA_TRANSFORMS key.
    # Empty dicts (the default, so every pre-existing InstrumentProfile(...)
    # call site keeps working unchanged) mean "no Gaia reference mapping for
    # any filter of this instrument" -- absolute calibration then reports
    # "unavailable" for every frame rather than raising.
    filter_gaia_xp: dict[str, str] = field(default_factory=dict)
    filter_gaia_transform: dict[str, str] = field(default_factory=dict)
    # Whether the raw header's WCS (CD matrix, or CDELT/PC pair) can be
    # trusted as a real plate solve. False for an instrument whose header
    # WCS is a known-nominal design value (see t80s.toml): derive_pixel_scale
    # then skips the header-WCS step, and reduced products get their WCS
    # rescaled to pixel_scale_arcsec instead of carrying the nominal one.
    trust_header_wcs: bool = True
    calib: InstrumentCalibOptions = field(default_factory=InstrumentCalibOptions)
    version: str = "1"
    source: str = "packaged"

    def canonical_filter(self, raw: str | None) -> str | None:
        """Map a raw ``FILTER`` header value onto its canonical name."""
        if raw is None:
            return None
        if raw in self.filter_map:
            return self.filter_map[raw]
        if not self.valid_filters:
            return raw
        if raw.lower() in {v.lower() for v in self.valid_filters}:
            return raw
        logger.debug("%s: unrecognised filter name %r", self.name, raw)
        return raw

    def canonical_frame_type(self, raw: str | None) -> str | None:
        """Map a raw ``IMAGETYP`` value onto ``bias``/``dark``/``flat``/``science``."""
        if raw is None:
            return None
        needle = raw.lower()
        for key, value in self.imagetyp_map.items():
            if key.lower() == needle:
                return value
        for key, value in DEFAULT_IMAGETYP_MAP.items():
            if key.lower() == needle:
                return value
        return None

    def aliases_for(self, field: str) -> tuple[str, ...]:
        """Return the ordered header-keyword aliases for a canonical field."""
        own = self.keywords.get(field, ())
        seen = {a.lower() for a in own}
        extra = tuple(
            a for a in DEFAULT_KEYWORD_ALIASES.get(field, ()) if a.lower() not in seen
        )
        return tuple(own) + extra

    def earth_location(self) -> EarthLocation | None:
        """Return the observatory location, or ``None`` if the site is unset."""
        if self.site_lat_deg is None or self.site_lon_deg is None:
            return None
        height = self.site_elevation_m if self.site_elevation_m is not None else 0.0
        return EarthLocation.from_geodetic(
            lon=self.site_lon_deg * u.deg,
            lat=self.site_lat_deg * u.deg,
            height=height * u.m,
        )


def _float_or_none(instrument: dict[str, Any], key: str, origin: str) -> float | None:
    if key not in instrument:
        return None
    value = instrument[key]
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        msg = f"{origin}: {key} must be a number, got {value!r}"
        raise ConfigError(msg) from exc


def _profile_from_toml_dict(
    data: dict[str, Any], *, path_for_errors: str, source: str
) -> InstrumentProfile:
    unknown_top = set(data) - {"instrument"}
    if unknown_top:
        msg = f"{path_for_errors}: unknown top-level key(s) {sorted(unknown_top)}"
        raise ConfigError(msg)
    if "instrument" not in data:
        msg = f"{path_for_errors}: missing top-level [instrument] table"
        raise ConfigError(msg)
    instrument = data["instrument"]
    if not isinstance(instrument, dict):
        msg = f"{path_for_errors}: [instrument] must be a table"
        raise ConfigError(msg)

    unknown = set(instrument) - _KNOWN_INSTRUMENT_KEYS
    if unknown:
        msg = f"{path_for_errors}: unknown key(s) {sorted(unknown)} in [instrument]"
        raise ConfigError(msg)

    name = instrument.get("name")
    if not name:
        msg = f"{path_for_errors}: [instrument] is missing required key 'name'"
        raise ConfigError(msg)

    raw_matches = instrument.get("matches", {})
    matches = {k: tuple(v) for k, v in raw_matches.items()}

    raw_scale = instrument.get("pixel_scale_arcsec")
    if raw_scale is None or raw_scale == "derive":
        pixel_scale_arcsec = None
    else:
        try:
            pixel_scale_arcsec = float(raw_scale)
        except (TypeError, ValueError) as exc:
            msg = (
                f"{path_for_errors}: pixel_scale_arcsec must be 'derive' or a "
                f"number, got {raw_scale!r}"
            )
            raise ConfigError(msg) from exc

    gain_e_per_adu = _float_or_none(instrument, "gain_e_per_adu", path_for_errors)
    read_noise_e = _float_or_none(instrument, "read_noise_e", path_for_errors)
    saturation_adu = _float_or_none(instrument, "saturation_adu", path_for_errors)
    nonlinearity_adu = _float_or_none(instrument, "nonlinearity_adu", path_for_errors)
    binning_keyword = instrument.get("binning_keyword", "XBINNING")
    trust_header_wcs = bool(instrument.get("trust_header_wcs", True))

    raw_calib = instrument.get("calib", {})
    calib_options = InstrumentCalibOptions(
        flat_pool=bool(raw_calib.get("flat_pool", False)),
        flat_pool_min_nights=int(raw_calib.get("flat_pool_min_nights", 10)),
        flat_pool_max_nights=int(raw_calib.get("flat_pool_max_nights", 20)),
        flat_pool_target_frames=int(raw_calib.get("flat_pool_target_frames", 20)),
        flat_pool_window_nights=int(raw_calib.get("flat_pool_window_nights", 0)),
        flat_master_roots=tuple(str(r) for r in raw_calib.get("flat_master_roots", ())),
        bias_pool=bool(raw_calib.get("bias_pool", False)),
        bias_pool_window_nights=int(raw_calib.get("bias_pool_window_nights", 10)),
        bias_pool_target_frames=int(raw_calib.get("bias_pool_target_frames", 20)),
        bias_pool_gain_max_frames=int(raw_calib.get("bias_pool_gain_max_frames", 12)),
    )
    for _key in (
        "flat_pool_window_nights",
        "bias_pool_window_nights",
        "bias_pool_gain_max_frames",
    ):
        if getattr(calib_options, _key) < 0:
            msg = f"[instrument.calib] {_key} must be >= 0, got {getattr(calib_options, _key)}"
            raise ConfigError(msg)
    for _key in ("flat_pool_target_frames", "bias_pool_target_frames"):
        if getattr(calib_options, _key) < 1:
            msg = f"[instrument.calib] {_key} must be >= 1, got {getattr(calib_options, _key)}"
            raise ConfigError(msg)

    version = instrument.get("version", "1")

    raw_layout = instrument.get("layout", {})
    layout = InstrumentLayout(
        science_hdu=raw_layout.get("science_hdu", 0),
        n_amplifiers=raw_layout.get("n_amplifiers", 1),
        overscan=raw_layout.get("overscan"),
        trim=raw_layout.get("trim"),
        flip_y=raw_layout.get("flip_y", False),
    )

    raw_filters = instrument.get("filters", {})
    valid_filters = tuple(raw_filters.get("valid", ()))
    filter_map = dict(raw_filters.get("map", {}))
    filter_gaia_xp = dict(raw_filters.get("gaia_xp", {}))
    filter_gaia_transform = dict(raw_filters.get("gaia_transform", {}))

    raw_keywords = instrument.get("keywords", {})
    keywords = {k: tuple(v) for k, v in raw_keywords.items()}

    raw_imagetyp_map = instrument.get("imagetyp_map", {})
    imagetyp_map = {str(k).lower(): v for k, v in raw_imagetyp_map.items()}

    raw_site = instrument.get("site", {})
    site_lat_deg = raw_site.get("lat_deg")
    site_lon_deg = raw_site.get("lon_deg")
    site_elevation_m = raw_site.get("elevation_m")

    raw_astrometry = instrument.get("astrometry", {})
    raw_index_scales = raw_astrometry.get("index_scales")
    if raw_index_scales is None or raw_index_scales == "derive":
        index_scales = None
    else:
        index_scales = tuple(int(v) for v in raw_index_scales)
    search_radius_deg = float(raw_astrometry.get("search_radius_deg", 0.5))

    return InstrumentProfile(
        name=name,
        matches=matches,
        pixel_scale_arcsec=pixel_scale_arcsec,
        gain_e_per_adu=gain_e_per_adu,
        read_noise_e=read_noise_e,
        saturation_adu=saturation_adu,
        nonlinearity_adu=nonlinearity_adu,
        binning_keyword=binning_keyword,
        layout=layout,
        valid_filters=valid_filters,
        filter_map=filter_map,
        filter_gaia_xp=filter_gaia_xp,
        filter_gaia_transform=filter_gaia_transform,
        keywords=keywords,
        imagetyp_map=imagetyp_map,
        site_lat_deg=site_lat_deg,
        site_lon_deg=site_lon_deg,
        site_elevation_m=site_elevation_m,
        index_scales=index_scales,
        search_radius_deg=search_radius_deg,
        trust_header_wcs=trust_header_wcs,
        calib=calib_options,
        version=str(version),
        source=source,
    )


def load_profile_file(
    path: str | os.PathLike[str], *, source: str = "packaged"
) -> InstrumentProfile:
    """Load and validate an instrument profile TOML file."""
    path = Path(path)
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        msg = f"cannot read instrument profile {path}: {exc}"
        raise ConfigError(msg) from exc
    except tomllib.TOMLDecodeError as exc:
        msg = f"malformed TOML in {path}: {exc}"
        raise ConfigError(msg) from exc
    return _profile_from_toml_dict(data, path_for_errors=str(path), source=source)


def available_profiles() -> dict[str, Path]:
    """Return ``{stem: path}`` for every packaged and user profile TOML.

    User profiles under :data:`USER_PROFILE_DIR` shadow packaged ones with
    the same stem.
    """
    from importlib.resources import files

    profiles: dict[str, Path] = {}
    packaged_dir = files("robo43.data.instruments")
    for entry in packaged_dir.iterdir():
        if entry.name.endswith(".toml"):
            profiles[Path(entry.name).stem] = Path(str(entry))

    if USER_PROFILE_DIR.is_dir():
        for entry in USER_PROFILE_DIR.iterdir():
            if entry.suffix == ".toml":
                profiles[entry.stem] = entry

    return profiles


def resolve_profile(
    *,
    name: str | os.PathLike[str] | None = None,
    header: fits.Header | Mapping[str, Any] | None = None,
    profile_toml: str | None = None,
) -> InstrumentProfile:
    """Resolve an :class:`InstrumentProfile` by explicit name, inline TOML, or header match."""
    profiles = available_profiles()

    if name is not None:
        candidate_path = Path(name)
        if candidate_path.suffix == ".toml" and candidate_path.is_file():
            logger.info("instrument profile loaded from filesystem path '%s'", candidate_path)
            return load_profile_file(candidate_path, source="path")
        name = str(name)
        if name not in profiles:
            known = ", ".join(sorted(profiles))
            msg = (
                f"instrument profile {name!r} not found as a registered name "
                f"(available: {known}) or as an existing .toml file path"
            )
            raise ConfigError(msg)
        logger.info("instrument profile '%s' selected explicitly by name", name)
        return load_profile_file(profiles[name])

    if profile_toml is not None:
        try:
            data = tomllib.loads(profile_toml)
        except tomllib.TOMLDecodeError as exc:
            msg = f"<profile_toml string>: malformed TOML: {exc}"
            raise ConfigError(msg) from exc
        profile = _profile_from_toml_dict(
            data, path_for_errors="<profile_toml string>", source="inline"
        )
        logger.info(
            "instrument profile selected from inline profile_toml (name='%s')",
            profile.name,
        )
        return profile

    if header is not None:
        for profile_name in sorted(profiles):
            if profile_name == "generic":
                continue
            candidate = load_profile_file(profiles[profile_name])
            if _header_matches(candidate, header):
                logger.info(
                    "instrument profile '%s' selected by header match (%s)",
                    candidate.name,
                    candidate.matches,
                )
                return candidate

    if "generic" not in profiles:
        msg = "no instrument profile matched and 'generic' fallback is not available"
        raise ConfigError(msg)
    logger.info("instrument profile 'generic' selected as fallback")
    return load_profile_file(profiles["generic"])


def _header_matches(
    profile: InstrumentProfile, header: fits.Header | Mapping[str, Any]
) -> bool:
    for keyword, accepted in profile.matches.items():
        if "*" in accepted:
            continue
        value = header.get(keyword)
        if value is None:
            return False
        value_str = str(value).strip().lower()
        if value_str not in {str(a).strip().lower() for a in accepted}:
            return False
    return True
