"""Settings dataclasses and their loader.

Every value the legacy scripts hard-coded lives here, with one deliberate
exception: anything that varies from telescope to telescope belongs to the
*instrument profile* (:mod:`robo43.instruments.profile`), not to these global
settings.  Plate scale, gain, saturation and the filter map are therefore
absent from this module on purpose.

Precedence, highest first::

    CLI flag  >  ROBO43_* environment  >  --config file
              >  ~/.config/robo43/config.toml  >  packaged config.default.toml

An unknown key in any TOML file is an error rather than a typo that silently
does nothing.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args, get_origin

from robo43.exceptions import ConfigError

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "GAIA_CATALOGS",
    "SCIENCE_SECTIONS",
    "USER_CONFIG_PATH",
    "AstrometrySettings",
    "CalibSettings",
    "CoaddSettings",
    "DatabaseSettings",
    "DiagnosticsSettings",
    "PathSettings",
    "PhotometrySettings",
    "RuntimeSettings",
    "ServiceSettings",
    "Settings",
    "TargetSettings",
    "load_settings",
]

USER_CONFIG_PATH = Path.home() / ".config" / "robo43" / "config.toml"

DIAGNOSTIC_LEVELS = ("none", "summary", "full")
COMBINE_METHODS = ("mean", "median")
CATALOG_FORMATS = ("mef", "ldac", "both")
COADD_SCALINGS = ("exptime", "starflux", "none")
GAIA_CATALOGS = ("vizier", "esa")

#: The sections whose values can change the numbers in a product.  These, and
#: only these, feed :meth:`Settings.config_hash`.
SCIENCE_SECTIONS: tuple[str, ...] = (
    "astrometry",
    "calib",
    "coadd",
    "photometry",
    "targets",
)


@dataclass(frozen=True, slots=True)
class PathSettings:
    """Where inputs are found and products are written."""

    data_root: Path | None = None
    workdir: Path = Path()
    outdir: Path | None = None
    calibdir: Path | None = None
    fits_patterns: tuple[str, ...] = (
        "*.fits", "*.fit", "*.fts",
        "*.fits.fz", "*.fits.gz", "*.fit.fz", "*.fit.gz",
        "*.fts.fz", "*.fts.gz",
    )

    def resolved_outdir(self) -> Path:
        return self.outdir if self.outdir is not None else self.workdir / "reduced"

    def resolved_calibdir(self) -> Path:
        return self.calibdir if self.calibdir is not None else self.resolved_outdir()


@dataclass(frozen=True, slots=True)
class AstrometrySettings:
    """Solver hints, budgets, and the accuracy threshold."""

    index_dir: Path | None = None
    index_scales: tuple[int, ...] | None = None
    search_radius_deg: float = 0.5
    scale_tolerance: float = 0.03
    solve_timeout_s: float = 120.0
    frame_budget_s: float = 300.0
    max_sources: int = 150
    min_sources: int = 20
    rms_max_arcsec: float = 0.3
    refine_with_gaia: bool = True
    sip_order: int = 3
    use_web_fallback: bool = False
    astrometry_net_api_key: str | None = None
    cache_file: Path | None = None
    grid_sample: bool = True


@dataclass(frozen=True, slots=True)
class PhotometrySettings:
    """Detection, the quality-rejection chain, and zero-point calibration."""

    instrumental_zp: float = 20.0
    aperture_radii_arcsec: tuple[float, ...] = (2.0, 3.0, 4.0, 6.0, 8.0)
    detect_thresh: float = 1.5
    analysis_thresh: float = 1.5
    detect_minarea: int = 5
    detect_filter: str = "tophat_3.0_3x3.conv"
    deblend_nthresh: int = 32
    deblend_mincont: float = 0.005
    back_size: int = 64
    back_filtersize: int = 3
    seeing_fwhm_arcsec: float = 4.0
    snr_min: float = 5.0
    fwhm_min_arcsec: float = 0.5
    fwhm_max_arcsec: float = 8.0
    ellipticity_max: float = 0.3
    edge_margin_px: int = 20
    edge_margin_fwhm: float = 3.0
    saturation_fraction: float = 0.9
    nonlinearity_fraction: float = 0.8
    isolation_radius_arcsec: float = 3.0
    class_star_min: float = 0.8
    reference_catalog: str = "gaia"
    zp_sigma_clip: float = 3.0
    zp_min_stars: int = 10
    zp_rms_max: float = 0.05
    # Optional Gaia DR3 absolute (external) zero-point calibration -- see
    # robo43.photometry.gaia_zp. Off by default: it never alters
    # instrumental_zp or any instrumental column the relative-photometry
    # (exoplanet-detection) path reads, only adds calibrated columns/cards
    # alongside. Enabled by --gaia-zp on `robo43 photometry`.
    abs_calibration: bool = False
    gaia_match_radius_arcsec: float = 2.0
    gaia_ruwe_max: float = 1.4
    # Cone-query source ("vizier" or "esa" -- see robo43.photometry.gaia_zp)
    # and bounds: an unfiltered cone query holds ~1.6e6 sources per square
    # degree and can run unbounded. The configured service is tried first
    # and the other is used as a fallback on any failure or timeout.
    gaia_catalog: str = "vizier"
    gaia_gmax: float = 17.0
    gaia_query_timeout_s: float = 300.0
    # Gaia DR3 XP synthetic-photometry request bounds (robo43.photometry.
    # gaia_zp.resolve_reference_mags path (a)): at most this many
    # cross-matched, FLAGS==0 sources, brightest G first, are ever sent to
    # gaiaxpy, subject to their own timeout.
    gaia_xp_max_stars: int = 2000
    gaia_xp_timeout_s: float = 300.0
    # Only reference stars brighter than this REFMAG feed the zero-point
    # fit, per-tile offsets, and spatial ZP range; REFMAG/REFMAGERR are
    # still written for every cross-matched source regardless of this cut.
    gaia_zp_refmag_max: float = 16.0
    # Aperture magnitude the Gaia ZP fit (and MAG_CALIB/MAG_APER_CAL) uses,
    # matched by radius against aperture_radii_arcsec -- 0 fits on MAG_AUTO
    # instead (pre-aperture-fit back-compat). Chosen empirically on T80S
    # R-band data: MAG_AUTO's Kron aperture correlates with per-frame
    # FWHM/PSF-shape variation, which a fixed aperture avoids -- see
    # robo43.photometry.gaia_zp.fit_zero_point_model.
    gaia_zp_aperture_arcsec: float = 4.0
    # Fit a linear term in Gaia BP-RP colour alongside the zero point,
    # referenced to the fit sample's own median BP-RP. Stars with no BP-RP
    # cross-match are excluded from the fit (not from REFMAG/REFMAGERR).
    gaia_zp_color_term: bool = True
    # Spatial zero-point surface fit order alongside the colour term: 0 is a
    # single scalar (pre-surface-fit behaviour), 1 a plane in normalised
    # detector (x, y), 2 a full quadratic surface. See
    # robo43.photometry.gaia_zp.fit_zero_point_model.
    gaia_zp_surface_order: int = 1
    catalog_format: str = "mef"
    write_catalog: bool = True


@dataclass(frozen=True, slots=True)
class CalibSettings:
    """Master-frame construction: combination, level cuts, dark scaling, masks."""

    combine_method: str = "mean"
    sigma_clip_low: float = 3.0
    sigma_clip_high: float = 3.0
    sigma_clip_maxiters: int = 5
    max_memory_mb: float = 2048.0
    flat_min_level_frac: float = 0.20
    flat_max_level_frac: float = 0.70
    flat_norm_region_frac: float = 0.5
    dark_temp_tolerance_c: float = 0.5
    dark_exptime_tolerance_s: float = 0.5
    dark_doubling_temp_c: float = 6.5
    dark_temp_extrap_max_c: float = 5.0
    hot_pixel_nmad: float = 5.0
    dead_pixel_frac: float = 0.5
    # A pixel is cold (flagged DQ.DEAD) when the master flat's residual from
    # its local block-median level is below -cold_pixel_nsigma * MAD-sigma.
    cold_pixel_nsigma: float = 5.0
    # The cold mask is skipped when fewer flats than this were combined: with
    # few flats the master traces local/night-specific minima, not the sensor.
    cold_pixel_min_flats: int = 20
    # Block size (px) of the local-level (block-median) estimate.
    cold_pixel_box: int = 32
    unstable_pixel_nmad: float = 5.0
    # A line whose median departs from its immediate neighbours' by more than
    # this fraction is a persistent bad column/row.  A relative test is used
    # because it discriminates cleanly: a dead or hot column deviates by tens
    # of percent, while smooth vignetting changes the level by far less than
    # one percent across a few columns.
    bad_column_frac: float = 0.10
    cosmic_ray_rejection: bool = True
    cosmic_ray_sigclip: float = 4.5
    overscan_correct: bool = True
    # Run runtime overscan detection from FLAT frames; when False, use profile
    # region if set, otherwise skip.
    overscan_detect: bool = True
    # Fraction of flat level span within which a line median counts as bias-level
    overscan_bias_tol_frac: float = 0.10
    # Minimum fraction of axis length for a detected overscan run; shorter runs
    # are rejected to exclude isolated noisy lines
    overscan_min_line_frac: float = 0.0015
    # Convert the reduced pixels to electrons (BUNIT='electron') as the
    # last step of the reduction, per plan section 4.4.  Set false to keep
    # products in ADU; the unit is always recorded in BUNIT either way.
    apply_gain: bool = True
    nonlinearity_correct: bool = False
    separate_dome_and_sky_flats: bool = True
    # Measure the e-/ADU gain at runtime by photon transfer from that night's
    # own flats and biases (robo43.calib.gain), per (readout_mode,
    # gain_setting, binning) group; when False, or when a group has no
    # measurement, frames in that group are reduced in ADU rather than
    # guessing a conversion factor.
    gain_measure: bool = True
    # Central fraction of the (already overscan-trimmed) frame sampled for
    # the photon-transfer measurement, avoiding edge vignetting.
    gain_region_frac: float = 0.25
    # Minimum number of flat frames and of bias frames required in a
    # (readout_mode, gain_setting, binning) group before it can be measured.
    gain_min_frames_per_group: int = 2
    # Relative scatter of per-filter-pair K estimates within a group above
    # which a warning is logged (the measurement is still used).
    gain_scatter_warn_frac: float = 0.02
    # Guard factor for robo43.calib.gain's raw-vs-MAD-based bias variance
    # cross-check: a real bias difference's heavy read-noise tails alone
    # produce a modest raw/MAD-based ratio (~1.1-1.5x on real ROBO43 data);
    # a ratio further outside [1/factor, factor] is logged as a possible
    # cosmic-ray storm or light leak (the measurement is still used, and
    # the ratio is always recorded on GainMeasurement.bias_var_ratio).
    gain_var_disagreement_factor: float = 2.0
    # Two flats pair for photon-transfer measurement when their median
    # levels (post overscan/trim, pre-bias-subtraction) differ by no more
    # than this fraction of the brighter one -- see
    # robo43.calib.gain._level_matched_pairs. Needed because pooled T80S
    # flats span several nights at different lamp/sky levels, so a naive
    # "next frame in the list" pairing yields almost no usable pairs.
    gain_flat_pair_level_tol_frac: float = 0.05
    # Multi-night calibration fallback (T80S support): when a night
    # directory has zero raw frames of a calibration type -- bias, dark,
    # or a flat filter group -- search sibling YYYYMMDD directories
    # under the same parent for strictly earlier nights' raw frames of
    # that type (and, for flats, that filter) and build the master from
    # those instead. Only takes effect when the reduction input resolves
    # to a single directory whose name parses as YYYYMMDD; disabled
    # entirely otherwise. See the --no-calib-fallback CLI flag.
    calib_fallback: bool = True
    # Minimum bias frames required from a single (or, failing that,
    # accumulated nearest) earlier night before it is used to build a
    # fallback master bias.
    min_bias_frames: int = 3
    # Minimum dark frames required from a single (or accumulated) earlier
    # night before it is used to build a fallback master dark.
    min_dark_frames: int = 3
    # Minimum flat frames, per filter, required from a single (or
    # accumulated) earlier night before it is used to build a fallback
    # master flat for that filter.
    min_flat_frames: int = 5
    # How many calendar days back from the target night the bias and dark
    # fallback search is allowed to look before giving up.
    fallback_max_nights_bias: int = 30
    # How many calendar days back from the target night the flat fallback
    # search is allowed to look before giving up. Flats drift with dust
    # and mirror coatings more slowly than the bias level does, so this
    # window is wider than the bias/dark one. Also the outer bound for the
    # per-instrument flat-pooling policy's search of older nights once
    # nothing is found within its own (typically much shorter) pooling
    # window -- see InstrumentProfile.calib.flat_pool and
    # robo43.calib.fallback.pool_flat_frames -- hence the wide default.
    fallback_max_nights_flat: int = 365


@dataclass(frozen=True, slots=True)
class CoaddSettings:
    """Stacking: flux scaling, sky handling, and outlier rejection."""

    scale_by: str = "exptime"
    sky_subtract: bool = False
    reject_sigma: float = 3.0
    combine: str = "weighted_mean"
    min_frames: int = 2
    astrom_rms_max_arcsec: float | None = None


@dataclass(frozen=True, slots=True)
class DiagnosticsSettings:
    """How many PNGs a run is allowed to write."""

    level: str = "summary"
    decimate: int = 4
    figsize: tuple[float, float] = (10.0, 8.0)
    dpi: int = 100


@dataclass(frozen=True, slots=True)
class DatabaseSettings:
    """Optional PostgreSQL layer (phase 9)."""

    url: str | None = None
    enabled: bool = False
    statement_timeout_ms: int = 60000
    pool_pre_ping: bool = True
    pool_size: int = 5
    schema_raw: str = "raw"
    schema_reduced: str = "reduced"


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    """Long-running mode (phase 8)."""

    watch_dirs: tuple[Path, ...] = ()
    settle_seconds: float = 10.0
    rescan_seconds: float = 300.0
    workers: int = 4
    max_attempts: int = 3
    backoff_base_s: float = 30.0
    metrics_port: int = 9109
    metrics_enabled: bool = True
    strict: bool = True
    lock_key: int = 4304043


@dataclass(frozen=True, slots=True)
class TargetSettings:
    """Target-name aliasing and which OBJECT values are not science targets."""

    aliases: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "eta car": ("etacar", "eta carinae", "etacarnebula"),
            "m42": ("orion nebula", "ngc1976", "m42"),
            "eso 97-g13": ("circinus galaxy", "circinus", "cir galaxy"),
            "tyc 4663-974-1": ("saturn",),
        }
    )
    calibration_object_names: tuple[str, ...] = ("flat", "dark", "bias", "trash")
    comparison_magnitudes: dict[str, dict[str, float]] = field(default_factory=dict)

    def canonical_target(self, name: str | None) -> str | None:
        """Map an ``OBJECT`` value onto its canonical target name."""
        if not name:
            return None
        needle = name.strip().lower()
        for canonical, alternatives in self.aliases.items():
            if needle == canonical or needle in {a.lower() for a in alternatives}:
                return canonical
        return needle


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Per-invocation behaviour that is not science configuration."""

    nprocs: int = 1
    loglevel: str = "INFO"
    logfile: Path | None = None
    json_logs: bool = False
    strict: bool = False
    assume_profile_defaults: bool = False
    dry_run: bool = False
    clobber: bool = False
    pdb: bool = False
    memory_reserve_mb: float = 8192.0
    release_memory: bool = True


@dataclass(frozen=True, slots=True)
class Settings:
    """The whole configuration tree."""

    paths: PathSettings = field(default_factory=PathSettings)
    astrometry: AstrometrySettings = field(default_factory=AstrometrySettings)
    photometry: PhotometrySettings = field(default_factory=PhotometrySettings)
    calib: CalibSettings = field(default_factory=CalibSettings)
    coadd: CoaddSettings = field(default_factory=CoaddSettings)
    diagnostics: DiagnosticsSettings = field(default_factory=DiagnosticsSettings)
    db: DatabaseSettings = field(default_factory=DatabaseSettings)
    service: ServiceSettings = field(default_factory=ServiceSettings)
    targets: TargetSettings = field(default_factory=TargetSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)

    def to_dict(self) -> dict[str, Any]:
        return _as_plain(dataclasses.asdict(self))

    def config_hash(self) -> str:
        """Stable short digest of the science-relevant configuration.

        Only the sections that can change the *numbers* in a product are
        hashed (see :data:`SCIENCE_SECTIONS`).  Deployment details -- how many
        worker processes ran, where the log went, which database URL was used,
        how many diagnostic PNGs were written -- are excluded, so two runs that
        produce identical pixels share a ``PIPECFG`` and the phase-8
        idempotency key does the right thing.
        """
        payload = self.to_dict()
        blob = json.dumps(
            {name: payload[name] for name in SCIENCE_SECTIONS},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def replace(self, **sections: Any) -> Settings:
        return dataclasses.replace(self, **sections)


#: Section name -> dataclass, used by the loader to validate and coerce keys.
_SECTIONS: dict[str, type] = {
    "paths": PathSettings,
    "astrometry": AstrometrySettings,
    "photometry": PhotometrySettings,
    "calib": CalibSettings,
    "coadd": CoaddSettings,
    "diagnostics": DiagnosticsSettings,
    "db": DatabaseSettings,
    "service": ServiceSettings,
    "targets": TargetSettings,
    "runtime": RuntimeSettings,
}


def _as_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _as_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _field_types(section: type) -> dict[str, Any]:
    return {f.name: f.type for f in fields(section)}


def _coerce(section_name: str, key: str, annotation: Any, value: Any) -> Any:
    """Coerce a TOML/env/CLI value to the annotated type of a settings field."""
    if isinstance(annotation, str):
        annotation = _ANNOTATION_ALIASES.get(annotation, annotation)
    where = f"{section_name}.{key}"
    optional = False
    args = get_args(annotation)
    if get_origin(annotation) is not None and type(None) in args:
        optional = True
        remaining = [a for a in args if a is not type(None)]
        annotation = remaining[0] if len(remaining) == 1 else remaining
    if value is None:
        if not optional:
            msg = f"{where}: null is not allowed"
            raise ConfigError(msg)
        return None

    origin = get_origin(annotation)
    try:
        if annotation is Path or annotation == "Path":
            return Path(str(value)).expanduser()
        if annotation is bool:
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off"}:
                return False
            msg = f"{where}: {value!r} is not a boolean"
            raise ConfigError(msg)
        if annotation is int:
            return int(value)
        if annotation is float:
            return float(value)
        if annotation is str:
            return str(value)
        if origin is tuple:
            item = get_args(annotation)[0]
            items = _as_sequence(value)
            return tuple(_coerce(section_name, key, item, v) for v in items)
        if origin is dict or annotation is dict:
            if not isinstance(value, dict):
                msg = f"{where}: expected a table, got {type(value).__name__}"
                raise ConfigError(msg)
            return _coerce_mapping(annotation, value)
    except (TypeError, ValueError) as exc:
        msg = f"{where}: cannot interpret {value!r} as {annotation}"
        raise ConfigError(msg) from exc
    return value


def _coerce_mapping(annotation: Any, value: dict[str, Any]) -> dict[str, Any]:
    args = get_args(annotation)
    if len(args) != 2:
        return dict(value)
    _, val_type = args
    if get_origin(val_type) is tuple:
        return {str(k): tuple(_as_sequence(v)) for k, v in value.items()}
    if get_origin(val_type) is dict:
        return {str(k): dict(v) for k, v in value.items()}
    return dict(value)


def _as_sequence(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [value]


#: ``from __future__ import annotations`` makes every annotation a string, so
#: the coercer resolves the ones it cares about by name.
_ANNOTATION_ALIASES: dict[str, Any] = {
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
    "Path": Path,
    "int | None": int | None,
    "float | None": float | None,
    "bool | None": bool | None,
    "str | None": str | None,
    "Path | None": Path | None,
    "tuple[str, ...]": tuple[str, ...],
    "tuple[int, ...]": tuple[int, ...],
    "tuple[float, ...]": tuple[float, ...],
    "tuple[Path, ...]": tuple[Path, ...],
    "tuple[int, ...] | None": tuple[int, ...] | None,
    "tuple[float, float]": tuple[float, float],
    "dict[str, tuple[str, ...]]": dict[str, tuple[str, ...]],
    "dict[str, dict[str, float]]": dict[str, dict[str, float]],
}


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except OSError as exc:
        msg = f"cannot read configuration file {path}: {exc}"
        raise ConfigError(msg) from exc
    except tomllib.TOMLDecodeError as exc:
        msg = f"malformed TOML in {path}: {exc}"
        raise ConfigError(msg) from exc


def _packaged_defaults() -> dict[str, Any]:
    from importlib.resources import files

    resource = files("robo43.data").joinpath("config.default.toml")
    if not resource.is_file():
        return {}
    return tomllib.loads(resource.read_text(encoding="utf-8"))


def _merge(base: dict[str, Any], overlay: Mapping[str, Any], origin: str) -> None:
    """Merge one layer into the accumulating section/key mapping."""
    for section, values in overlay.items():
        if section not in _SECTIONS:
            known = ", ".join(sorted(_SECTIONS))
            msg = f"{origin}: unknown configuration section [{section}] (known: {known})"
            raise ConfigError(msg)
        if not isinstance(values, dict):
            msg = f"{origin}: section [{section}] must be a table"
            raise ConfigError(msg)
        allowed = _field_types(_SECTIONS[section])
        for key, value in values.items():
            if key not in allowed:
                known = ", ".join(sorted(allowed))
                msg = (
                    f"{origin}: unknown key {section}.{key} "
                    f"(known keys: {known})"
                )
                raise ConfigError(msg)
            base.setdefault(section, {})[key] = (value, origin)


def _env_layer(env: Mapping[str, str]) -> dict[str, Any]:
    """Read ``ROBO43_<SECTION>_<FIELD>`` plus the ``ROBO43_DB_URL`` shorthand."""
    layer: dict[str, dict[str, Any]] = {}
    for raw_key, raw_value in env.items():
        if not raw_key.startswith("ROBO43_"):
            continue
        remainder = raw_key[len("ROBO43_") :].lower()
        if remainder == "db_url":
            layer.setdefault("db", {})["url"] = raw_value
            continue
        if remainder == "config":
            continue
        for section, cls in _SECTIONS.items():
            prefix = f"{section}_"
            if remainder.startswith(prefix):
                key = remainder[len(prefix) :]
                if key in _field_types(cls):
                    layer.setdefault(section, {})[key] = raw_value
                break
    return layer


def _validate(settings: Settings) -> None:
    checks = (
        ("diagnostics.level", settings.diagnostics.level, DIAGNOSTIC_LEVELS),
        ("calib.combine_method", settings.calib.combine_method, COMBINE_METHODS),
        ("photometry.catalog_format", settings.photometry.catalog_format, CATALOG_FORMATS),
        ("photometry.gaia_catalog", settings.photometry.gaia_catalog, GAIA_CATALOGS),
        ("coadd.scale_by", settings.coadd.scale_by, COADD_SCALINGS),
    )
    for where, value, allowed in checks:
        if value not in allowed:
            msg = f"{where}: {value!r} is not one of {allowed}"
            raise ConfigError(msg)
    if settings.runtime.nprocs < 1:
        msg = "runtime.nprocs must be >= 1"
        raise ConfigError(msg)
    if settings.calib.flat_min_level_frac >= settings.calib.flat_max_level_frac:
        msg = "calib.flat_min_level_frac must be below calib.flat_max_level_frac"
        raise ConfigError(msg)
    if settings.calib.cold_pixel_nsigma <= 0:
        msg = "calib.cold_pixel_nsigma must be > 0"
        raise ConfigError(msg)
    if settings.calib.cold_pixel_min_flats < 1:
        msg = "calib.cold_pixel_min_flats must be >= 1"
        raise ConfigError(msg)
    if settings.calib.cold_pixel_box < 3:
        msg = "calib.cold_pixel_box must be >= 3"
        raise ConfigError(msg)
    gaia_zp_aperture = settings.photometry.gaia_zp_aperture_arcsec
    gaia_zp_radii = settings.photometry.aperture_radii_arcsec
    if gaia_zp_aperture != 0.0 and gaia_zp_aperture not in gaia_zp_radii:
        msg = (
            "photometry.gaia_zp_aperture_arcsec: "
            f"{gaia_zp_aperture!r} is not one of "
            f"{gaia_zp_radii} "
            "(use 0 to fit on MAG_AUTO instead)"
        )
        raise ConfigError(msg)
    if settings.photometry.gaia_zp_surface_order not in (0, 1, 2):
        msg = (
            "photometry.gaia_zp_surface_order must be 0, 1, or 2 "
            f"(got {settings.photometry.gaia_zp_surface_order!r})"
        )
        raise ConfigError(msg)
    if settings.runtime.strict and settings.runtime.assume_profile_defaults:
        msg = "--strict and --assume-profile-defaults are mutually exclusive"
        raise ConfigError(msg)


def load_settings(
    *,
    config_file: str | os.PathLike[str] | None = None,
    cli: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    use_user_config: bool = True,
) -> Settings:
    """Build a :class:`Settings` from all configuration layers.

    ``cli`` uses dotted keys (``{"astrometry.index_dir": "/data/idx"}``) and
    wins over everything; a ``None`` value in ``cli`` means "flag not given"
    and is ignored, so an unset argparse default never overrides a file.
    """
    env = os.environ if env is None else env
    merged: dict[str, dict[str, tuple[Any, str]]] = {}

    _merge(merged, _packaged_defaults(), "packaged default")

    if use_user_config and USER_CONFIG_PATH.is_file():
        _merge(merged, _load_toml(USER_CONFIG_PATH), str(USER_CONFIG_PATH))

    chosen = config_file if config_file is not None else env.get("ROBO43_CONFIG")
    if chosen is not None:
        path = Path(chosen).expanduser()
        if not path.is_file():
            msg = f"configuration file not found: {path}"
            raise ConfigError(msg)
        _merge(merged, _load_toml(path), str(path))

    _merge(merged, _env_layer(env), "environment")

    if cli:
        cli_layer: dict[str, dict[str, Any]] = {}
        for dotted, value in cli.items():
            if value is None:
                continue
            section, _, key = dotted.partition(".")
            if not key:
                msg = f"CLI override {dotted!r} must be of the form section.key"
                raise ConfigError(msg)
            cli_layer.setdefault(section, {})[key] = value
        _merge(merged, cli_layer, "command line")

    sections: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        annotations = _field_types(cls)
        kwargs = {
            key: _coerce(name, key, annotations[key], value)
            for key, (value, _origin) in merged.get(name, {}).items()
        }
        sections[name] = cls(**kwargs)

    settings = Settings(**sections)
    _validate(settings)
    return settings


def origin_report(
    *,
    config_file: str | os.PathLike[str] | None = None,
    cli: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """``{"astrometry.index_dir": "environment", ...}`` -- where each value came from.

    Used by ``robo43 config show`` so a surprising value can be traced to the
    layer that set it.
    """
    env = os.environ if env is None else env
    merged: dict[str, dict[str, tuple[Any, str]]] = {}
    _merge(merged, _packaged_defaults(), "packaged default")
    if USER_CONFIG_PATH.is_file():
        _merge(merged, _load_toml(USER_CONFIG_PATH), str(USER_CONFIG_PATH))
    chosen = config_file if config_file is not None else env.get("ROBO43_CONFIG")
    if chosen is not None and Path(chosen).expanduser().is_file():
        _merge(merged, _load_toml(Path(chosen).expanduser()), str(chosen))
    _merge(merged, _env_layer(env), "environment")
    if cli:
        layer: dict[str, dict[str, Any]] = {}
        for dotted, value in cli.items():
            if value is None:
                continue
            section, _, key = dotted.partition(".")
            if key:
                layer.setdefault(section, {})[key] = value
        _merge(merged, layer, "command line")
    return {
        f"{section}.{key}": origin
        for section, values in merged.items()
        for key, (_value, origin) in values.items()
    }
