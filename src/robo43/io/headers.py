"""Header parsing and metadata resolution for a single FITS frame.

This module turns a raw FITS header into a :class:`~robo43.io.meta.FrameMeta`,
applying the pipeline's field-resolution precedence -- ``--set`` overrides,
per-glob overrides, header aliases, instrument-profile defaults, and derived
values -- and recording where every value came from.
"""

from __future__ import annotations

import logging
import math
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, Angle, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales

from robo43.exceptions import MetadataError
from robo43.instruments.keywords import OBJECT_TYPE_PATTERNS
from robo43.instruments.profile import InstrumentProfile, ReadoutModeVote
from robo43.io.fits_io import read_header
from robo43.io.meta import FieldOrigin, FrameMeta, Source

if TYPE_CHECKING:
    from astropy.io import fits

    from robo43.config import Settings
    from robo43.io.validate import OverrideSet

logger = logging.getLogger(__name__)

#: Tier-1 fields: always required, refused as ``DERIVED`` under ``--strict``.
_TIER1_FIELDS: tuple[str, ...] = ("naxis1", "naxis2", "frame_type", "exptime", "date_obs")

#: Tier-2 fields: profile-supplied values here log a warning unless
#: ``settings.runtime.assume_profile_defaults`` is set.
_TIER2_FIELDS: tuple[str, ...] = (
    "filter", "ra_deg", "dec_deg", "pixel_scale", "ccd_temp",
    "gain_setting", "read_noise", "saturation", "airmass",
)

#: canonical FrameMeta field name -> canonical alias-lookup field name.
_ALIAS_FIELD_MAP: dict[str, str] = {
    "naxis1": "naxis1",
    "naxis2": "naxis2",
    "exptime": "exptime",
    "date_obs": "date_obs",
    "filter": "filter",
    "ccd_temp": "ccd_temp",
    "gain_setting": "gain_setting",
    "read_noise": "read_noise",
    "saturation": "saturation",
    "object": "object",
    "telescope": "telescope",
    "instrument": "instrument",
    "observer": "observer",
    "binning": "binning",
    "readout_mode": "readout_mode",
    "focal_length_mm": "focal_length",
    "mjd": "mjd",
}


def _find_alias_value(
    header: fits.Header, profile: InstrumentProfile, field: str
) -> tuple[Any, FieldOrigin | None]:
    """Look up ``field`` in ``header`` via ``profile.aliases_for(field)``.

    Returns ``(value, origin)``; ``origin`` is ``None`` when nothing matched.
    The first alias yields ``Source.HEADER``, any later alias ``Source.ALIAS``.
    """
    aliases = profile.aliases_for(field)
    for index, alias in enumerate(aliases):
        value = header.get(alias)
        if value is not None and value != "":
            source = Source.HEADER if index == 0 else Source.ALIAS
            return value, FieldOrigin(source, alias)
    return None, None


def normalise_radec(ra: Any, dec: Any) -> tuple[float | None, float | None]:
    """Normalise an RA/Dec pair to plain-float decimal degrees, or ``(None, None)``.

    Accepts numeric values (already degrees), :class:`~astropy.units.Quantity`,
    or strings -- either bare decimal degrees (``"161.264"``) or sexagesimal
    (``"10:44:32.1"``, ``"10 44 32.1"``, ``"10h44m32.1s"`` for RA in hours;
    ``"-59:41:02.4"`` etc. in degrees for Dec). Never raises.
    """
    if ra is None or dec is None:
        return None, None

    try:
        ra_deg = _axis_to_degrees(ra, unit=u.hourangle)
        dec_deg = _axis_to_degrees(dec, unit=u.deg)
    except Exception as exc:
        logger.warning("could not parse RA/DEC value %r/%r as coordinates: %s", ra, dec, exc)
        return None, None

    if ra_deg is None or dec_deg is None:
        return None, None
    return ra_deg, dec_deg


def _axis_to_degrees(value: Any, *, unit: u.UnitBase) -> float | None:
    """Convert one coordinate axis to degrees. ``unit`` is the sexagesimal unit
    to use for a string that turns out to be sexagesimal (hourangle for RA,
    deg for Dec). Raises on genuinely unparseable input; caller catches."""
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, u.Quantity):
        return float(value.to(u.deg).value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return float(text)
        except ValueError:
            pass
        return float(Angle(text, unit=unit).degree)
    msg = f"unsupported coordinate value type {type(value)!r}: {value!r}"
    raise TypeError(msg)


def classify_from_object(object_text: str | None) -> str | None:
    """Classify a frame from its OBJECT value alone, ignoring IMAGETYP.

    Returns "bias"/"dark"/"flat"/"focus"/"science", or None when object_text
    is empty or None. Case-insensitive substring match against
    OBJECT_TYPE_PATTERNS; "science" is the default when nothing matches.
    """
    if not object_text:
        return None
    lowered = object_text.lower()
    for category in ("bias", "dark", "flat", "focus"):
        substrings = OBJECT_TYPE_PATTERNS.get(category, ())
        if any(sub in lowered for sub in substrings):
            return category
    return "science"


def classify_frame_type(
    header: fits.Header, profile: InstrumentProfile, *, object_value: str | None = None
) -> tuple[str | None, Source]:
    """Classify a frame as bias/dark/flat/focus/science.

    IMAGETYP (via the profile's alias-then-map lookup) wins whenever present
    and mappable. Only falls back to OBJECT substring matching (via
    classify_from_object) when IMAGETYP is absent or unmappable.
    """
    raw_imagetyp, imagetyp_origin = _find_alias_value(header, profile, "imagetyp")
    if raw_imagetyp is not None:
        mapped = profile.canonical_frame_type(str(raw_imagetyp))
        if mapped is not None:
            source = imagetyp_origin.source if imagetyp_origin is not None else Source.HEADER
            return mapped, source

    if object_value is not None:
        object_text = object_value
    else:
        raw_object, _origin = _find_alias_value(header, profile, "object")
        object_text = str(raw_object) if raw_object is not None else None

    category = classify_from_object(object_text)
    if category is None:
        return None, Source.DEFAULT
    return category, Source.DERIVED


def derive_pixel_scale(
    header: fits.Header, profile: InstrumentProfile
) -> tuple[float | None, FieldOrigin]:
    """Derive the plate scale in arcsec/pixel.

    Precedence: XPIXSZ+FOCALLEN+binning, then WCS (skipped entirely when
    ``profile.trust_header_wcs`` is ``False``, because the header WCS is a
    known-nominal design value rather than a plate solve), then the
    profile's static ``pixel_scale_arcsec``, then unavailable.
    """
    raw_pixsz, _ = _find_alias_value(header, profile, "pixel_size")
    raw_focallen, _ = _find_alias_value(header, profile, "focal_length")
    binning = header.get(profile.binning_keyword, 1)

    if raw_pixsz is not None and raw_focallen is not None:
        try:
            pixsz = float(raw_pixsz)
            focallen = float(raw_focallen)
            binning_val = float(binning) if binning else 1.0
        except (TypeError, ValueError):
            pixsz = focallen = binning_val = None  # type: ignore[assignment]
        if pixsz is not None and focallen not in (None, 0.0):
            scale = 206.265 * pixsz * binning_val / focallen
            return scale, FieldOrigin(Source.DERIVED, "xpixsz+focallen+binning")

    if profile.trust_header_wcs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                wcs = WCS(header)
            if wcs.has_celestial:
                scales_deg = proj_plane_pixel_scales(wcs)
                scale_arcsec = float(np.mean(scales_deg) * 3600.0)
                return scale_arcsec, FieldOrigin(Source.DERIVED, "wcs")
        except Exception as exc:
            logger.debug("could not derive pixel scale from WCS: %s", exc)

    if profile.pixel_scale_arcsec is not None:
        return profile.pixel_scale_arcsec, FieldOrigin(Source.PROFILE, profile.name)

    return None, FieldOrigin(Source.DEFAULT, "unavailable")


def derive_airmass(
    ra_deg: float | None, dec_deg: float | None, time: Time | None, location: EarthLocation | None
) -> float | None:
    """Compute a simple sec(z) airmass, or ``None`` if it cannot be computed."""
    if ra_deg is None or dec_deg is None or time is None or location is None:
        return None
    try:
        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        altaz_frame = AltAz(obstime=time, location=location)
        altaz = coord.transform_to(altaz_frame)
        if altaz.alt.deg <= 0:
            return None
        return float(altaz.secz)
    except Exception as exc:
        logger.warning("could not compute airmass: %s", exc)
        return None


#: ``"<profile name>:<status>"`` keys (status "disagree" / "default") whose
#: first readout-mode case in this process was already logged at WARNING;
#: later cases with the same key log at DEBUG only.
_READOUT_MODE_WARNED: set[str] = set()


def _card_number(value: Any) -> float | None:
    """A header card as a finite float, or ``None`` (never raises)."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _vote_mode(vote: ReadoutModeVote, number: float) -> str | None:
    if vote.threshold is not None:
        return vote.below if number < vote.threshold else vote.at_or_above
    for target, mode in vote.values:
        if math.isclose(number, target, rel_tol=0.0, abs_tol=1e-9):
            return mode
    return None


def derive_readout_mode(
    header: fits.Header, profile: InstrumentProfile
) -> tuple[str, str, str] | None:
    """Apply ``profile.readout_mode_rule`` to a frame's header.

    Returns ``None`` when the profile has no rule.  Otherwise
    ``(mode, detail, status)``:

    * ``status == "ok"``: the votes present agree; ``mode`` is their mode.
    * ``status == "disagree"``: the votes present name different modes;
      ``mode`` is the highest-priority vote's.
    * ``status == "default"``: no vote is present; ``mode`` is the rule's
      default.

    ``detail`` names the deciding card(s), e.g. ``"mode0 from PG0_50=-12.96"``.
    A missing, non-numeric or out-of-map card casts no vote.
    """
    rule = profile.readout_mode_rule
    if rule is None:
        return None
    cast: list[tuple[str, float, str]] = []
    for vote in rule.votes:
        number = _card_number(header.get(vote.keyword))
        if number is None:
            continue
        mode = _vote_mode(vote, number)
        if mode is not None:
            cast.append((vote.keyword, number, mode))
    if not cast:
        keywords = ", ".join(v.keyword for v in rule.votes)
        return rule.default, f"{rule.default} by default: no vote from {keywords}", "default"
    keyword, number, mode = cast[0]
    detail = f"{mode} from {keyword}={number:g}"
    if any(m != mode for _, _, m in cast[1:]):
        others = ", ".join(f"{k}={n:g} -> {m}" for k, n, m in cast[1:])
        return mode, f"{detail} (disagreeing: {others})", "disagree"
    return mode, detail, "ok"


def read_meta(
    path: str | os.PathLike[str],
    profile: InstrumentProfile,
    *,
    overrides: OverrideSet | None = None,
    settings: Settings | None = None,
    header: fits.Header | None = None,
) -> FrameMeta:
    """Resolve a frame's full metadata, honouring the field-resolution precedence.

    Precedence, per field: ``--set`` / overrides file (via ``overrides``) >
    header/alias > instrument profile > derived value.
    """
    path = Path(path)
    if header is None:
        header = read_header(path, profile)

    field_overrides: dict[str, tuple[Any, FieldOrigin]] = {}
    if overrides is not None:
        field_overrides = overrides.for_path(path)

    values: dict[str, Any] = {}
    origins: dict[str, FieldOrigin] = {}
    strict = settings is not None and settings.runtime.strict
    assume_profile_defaults = settings is not None and settings.runtime.assume_profile_defaults

    def _set(field: str, value: Any, origin: FieldOrigin | None) -> None:
        values[field] = value
        if value is not None and origin is not None:
            origins[field] = origin
            is_tier2_profile = field in _TIER2_FIELDS and origin.source is Source.PROFILE
            if is_tier2_profile and not assume_profile_defaults:
                logger.warning(
                    "using profile default for tier-2 field '%s' "
                    "(assume_profile_defaults is False); consider --set or --overrides",
                    field,
                )

    def _resolve_via_alias(field: str, alias_field: str) -> None:
        if field in field_overrides:
            value, origin = field_overrides[field]
            _set(field, value, origin)
            return
        value, origin = _find_alias_value(header, profile, alias_field)
        _set(field, value, origin)

    # naxis1 / naxis2 -- hard requirement, unconditional.
    _resolve_via_alias("naxis1", "naxis1")
    _resolve_via_alias("naxis2", "naxis2")
    if values.get("naxis1") is None or values.get("naxis2") is None:
        msg = f"{path}: NAXIS1/NAXIS2 could not be determined"
        raise MetadataError(msg)
    values["naxis1"] = int(values["naxis1"])
    values["naxis2"] = int(values["naxis2"])

    # exptime, date_obs, ccd_temp, gain, read_noise, saturation, object,
    # telescope, instrument, observer, binning, focal_length_mm, mjd.
    for field, alias_field in _ALIAS_FIELD_MAP.items():
        if field in ("naxis1", "naxis2"):
            continue
        _resolve_via_alias(field, alias_field)

    # exptime -> float
    if values.get("exptime") is not None:
        try:
            values["exptime"] = float(values["exptime"])
        except (TypeError, ValueError):
            values["exptime"] = None
            origins.pop("exptime", None)

    # date_obs -> Time
    if values.get("date_obs") is not None:
        try:
            values["date_obs"] = Time(str(values["date_obs"]), format="isot", scale="utc")
        except Exception:
            try:
                values["date_obs"] = Time(str(values["date_obs"]))
            except Exception as exc:
                logger.warning("could not parse DATE-OBS %r: %s", values["date_obs"], exc)
                values["date_obs"] = None
                origins.pop("date_obs", None)

    # simple float fields
    for field in ("ccd_temp", "gain_setting", "read_noise", "saturation", "focal_length_mm"):
        if values.get(field) is not None:
            try:
                values[field] = float(values[field])
            except (TypeError, ValueError):
                values[field] = None
                origins.pop(field, None)

    # binning -> int
    if values.get("binning") is not None:
        try:
            values["binning"] = int(values["binning"])
        except (TypeError, ValueError):
            values["binning"] = None
            origins.pop("binning", None)

    # readout_mode -> str (a free-text camera setting; no numeric cast)
    if values.get("readout_mode") is not None:
        values["readout_mode"] = str(values["readout_mode"])

    # readout_mode from the profile's [instrument.readout_mode_rule]: replaces a
    # value read from the header aliases, never an explicit --set / overrides
    # value. Done here, once, so every consumer of FrameMeta.readout_mode sees
    # it; the origin is DERIVED with the deciding card(s) as detail.
    if profile.readout_mode_rule is not None and "readout_mode" not in field_overrides:
        derived = derive_readout_mode(header, profile)
        if derived is not None:
            mode, mode_detail, mode_status = derived
            values["readout_mode"] = mode
            origins["readout_mode"] = FieldOrigin(Source.DERIVED, mode_detail)
            if mode_status != "ok":
                warned_key = f"{profile.name}:{mode_status}"
                if warned_key in _READOUT_MODE_WARNED:
                    logger.debug("%s: readout mode: %s", path, mode_detail)
                else:
                    _READOUT_MODE_WARNED.add(warned_key)
                    logger.warning(
                        "%s: readout mode: %s (further '%s' cases for profile "
                        "'%s' are logged at DEBUG only)",
                        path,
                        mode_detail,
                        mode_status,
                        profile.name,
                    )

    # mjd -> float (may be overwritten below by header value's own presence)
    if values.get("mjd") is not None:
        try:
            values["mjd"] = float(values["mjd"])
        except (TypeError, ValueError):
            values["mjd"] = None
            origins.pop("mjd", None)

    # read_noise / saturation profile fallback (tier 2). gain_setting has no
    # profile fallback: it is a raw per-frame GAIN index, not a physical
    # constant, and the physical e-/ADU conversion is never sourced from the
    # profile at all -- it is measured at runtime (see robo43.calib.gain) and
    # looked up from the CalibrationSet at reduction time, keyed by
    # (readout_mode, gain_setting, binning).
    if values.get("read_noise") is None and profile.read_noise_e is not None:
        _set("read_noise", profile.read_noise_e, FieldOrigin(Source.PROFILE, profile.name))
    if values.get("saturation") is None and profile.saturation_adu is not None:
        _set("saturation", profile.saturation_adu, FieldOrigin(Source.PROFILE, profile.name))

    # filter -- header/alias, then translated through the profile's filter map.
    if "filter" in field_overrides:
        value, origin = field_overrides["filter"]
        _set("filter", value, origin)
    else:
        raw_filter, filter_origin = _find_alias_value(header, profile, "filter")
        canonical = profile.canonical_filter(str(raw_filter)) if raw_filter is not None else None
        _set("filter", canonical, filter_origin)

    # ra_deg / dec_deg
    if "ra_deg" in field_overrides or "dec_deg" in field_overrides:
        if "ra_deg" in field_overrides:
            ra_value, ra_origin = field_overrides["ra_deg"]
            _set("ra_deg", ra_value, ra_origin)
        if "dec_deg" in field_overrides:
            dec_value, dec_origin = field_overrides["dec_deg"]
            _set("dec_deg", dec_value, dec_origin)
        if "ra_deg" not in field_overrides:
            values.setdefault("ra_deg", None)
        if "dec_deg" not in field_overrides:
            values.setdefault("dec_deg", None)
    else:
        raw_ra, ra_origin = _find_alias_value(header, profile, "ra")
        raw_dec, dec_origin = _find_alias_value(header, profile, "dec")
        ra_deg, dec_deg = normalise_radec(raw_ra, raw_dec)
        _set("ra_deg", ra_deg, ra_origin)
        _set("dec_deg", dec_deg, dec_origin)

    # pixel_scale
    if "pixel_scale" in field_overrides:
        value, origin = field_overrides["pixel_scale"]
        _set("pixel_scale", value, origin)
    else:
        scale, scale_origin = derive_pixel_scale(header, profile)
        _set("pixel_scale", scale, scale_origin)

    # site_lat_deg / site_lon_deg / site_elevation_m
    for field, keyword in (
        ("site_lat_deg", "site_lat_deg"),
        ("site_lon_deg", "site_lon_deg"),
        ("site_elevation_m", "site_elevation_m"),
    ):
        if field in field_overrides:
            value, origin = field_overrides[field]
            _set(field, value, origin)
            continue
        profile_value = getattr(profile, keyword)
        if profile_value is not None:
            _set(field, profile_value, FieldOrigin(Source.PROFILE, profile.name))
        else:
            values[field] = None

    # frame_type -- overrides win, else classify_frame_type.
    if "frame_type" in field_overrides:
        value, origin = field_overrides["frame_type"]
        _set("frame_type", value, origin)
    else:
        frame_type, frame_source = classify_frame_type(
            header, profile, object_value=values.get("object")
        )
        frame_origin = (
            FieldOrigin(frame_source, "imagetyp/object") if frame_type is not None else None
        )
        _set("frame_type", frame_type, frame_origin)

    # Keep the raw IMAGETYP string (e.g. T80S 'sky-flat') so later stages can
    # tell twilight flats from dome flats; not a FrameMeta field, but it
    # survives an inventory round trip through origins_json.
    raw_imagetyp, imagetyp_origin = _find_alias_value(header, profile, "imagetyp")
    if raw_imagetyp is not None and imagetyp_origin is not None:
        origins["imagetyp"] = FieldOrigin(imagetyp_origin.source, str(raw_imagetyp))

    # mjd -- derive from date_obs if not already found.
    if values.get("mjd") is None and values.get("date_obs") is not None:
        _set("mjd", float(values["date_obs"].mjd), FieldOrigin(Source.DERIVED, "date_obs"))

    # airmass -- must happen after ra_deg/dec_deg/date_obs are resolved.
    if "airmass" in field_overrides:
        value, origin = field_overrides["airmass"]
        _set("airmass", value, origin)
    else:
        raw_airmass, airmass_origin = _find_alias_value(header, profile, "airmass")
        if raw_airmass is not None:
            try:
                _set("airmass", float(raw_airmass), airmass_origin)
            except (TypeError, ValueError):
                values["airmass"] = None
        else:
            location = profile.earth_location()
            computed = derive_airmass(
                values.get("ra_deg"), values.get("dec_deg"), values.get("date_obs"), location
            )
            airmass_origin = (
                FieldOrigin(Source.DERIVED, "ra/dec/date_obs") if computed is not None else None
            )
            _set("airmass", computed, airmass_origin)

    # night -- date of date_obs shifted back by 12h.
    if values.get("date_obs") is not None:
        try:
            shifted = values["date_obs"] - 12 * u.hour
            values["night"] = shifted.to_datetime().date()
            origins["night"] = FieldOrigin(Source.DERIVED, "date_obs-12h")
        except Exception as exc:
            logger.warning("could not derive night from date_obs: %s", exc)
            values["night"] = None
    else:
        values["night"] = None

    # is_valid -- direct header read, default True, never overridable.
    raw_valid = header.get("VALID", True)
    values["is_valid"] = bool(raw_valid)

    # --strict: refuse a DERIVED value for any tier-1 field.
    if strict:
        for field in _TIER1_FIELDS:
            origin = origins.get(field)
            if origin is not None and origin.source is Source.DERIVED:
                msg = (
                    f"{path}: field '{field}' was derived rather than supplied "
                    f"({origin.detail}); --strict refuses this"
                )
                raise MetadataError(msg)

    return FrameMeta(
        path=path,
        naxis1=values["naxis1"],
        naxis2=values["naxis2"],
        frame_type=values.get("frame_type"),
        exptime=values.get("exptime"),
        date_obs=values.get("date_obs"),
        filter=values.get("filter"),
        ra_deg=values.get("ra_deg"),
        dec_deg=values.get("dec_deg"),
        pixel_scale=values.get("pixel_scale"),
        ccd_temp=values.get("ccd_temp"),
        gain_setting=values.get("gain_setting"),
        read_noise=values.get("read_noise"),
        saturation=values.get("saturation"),
        airmass=values.get("airmass"),
        object=values.get("object"),
        telescope=values.get("telescope"),
        instrument=values.get("instrument"),
        observer=values.get("observer"),
        binning=values.get("binning"),
        readout_mode=values.get("readout_mode"),
        focal_length_mm=values.get("focal_length_mm"),
        site_lat_deg=values.get("site_lat_deg"),
        site_lon_deg=values.get("site_lon_deg"),
        site_elevation_m=values.get("site_elevation_m"),
        mjd=values.get("mjd"),
        night=values.get("night"),
        is_valid=values["is_valid"],
        profile_name=profile.name,
        origins=origins,
    )
