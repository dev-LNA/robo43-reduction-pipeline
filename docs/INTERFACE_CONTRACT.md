# robo43 — interface contract

**Purpose.** This is the authoritative specification of the interfaces that the
`robo43` package is being refactored onto, per `REFACTORING_PLAN.md`. It was
written before the implementation so that work on different phases could
proceed in parallel without the modules drifting apart, and it remains the
document to consult before adding or changing a public signature.

**Status.** Sections marked "already implemented" in the addenda are landed and
verified; the rest are specifications for work not yet done. Where an
implementation deviates from a signature here, the addenda record it — the
addenda are newer than the body, so **when the body and an addendum disagree,
the addendum wins**.

**How to use it.** Read the body for the module layout and the core types
(`Settings`, `InstrumentProfile`, `FrameMeta`), then read the addenda in order:
they record what has landed, the product model, the phase 6–9 signatures, and
the cross-module rules. Do not change a signature here without checking which
modules already code against it.

---

# robo43 foundation interface contract (authoritative)

Repo: `/home/herpich/Dropbox/robo43-reduction-pipeline`
Plan: `REFACTORING_PLAN.md` (read the phases named in your brief).
Python: 3.13.5 venv at `/home/herpich/py313` (already the default `python3`).

This file is the **contract**. Every agent MUST conform to these names and
signatures exactly. If you believe a signature is wrong, implement it as written
and note the objection in your report — do NOT silently change it, because other
agents are coding against it in parallel.

## Layout

`src/robo43/` is the package (src-layout). Package data at `src/robo43/data/`
(SExtractor `.conv`/`.nnw`/`.sex`/`.param` flat in there, plus
`data/instruments/*.toml` and `data/config.default.toml`). Access package data
ONLY via `importlib.resources.files("robo43.data")` — never `__file__`
arithmetic. Legacy `src/*.py` scripts stay in place for now; they become
deprecation shims in the final wave.

## Global rules

- Python >= 3.12 syntax OK. `from __future__ import annotations` at the top of
  every module.
- Type-annotate every public function. Target `ruff check` clean and
  `mypy` clean-ish on the package.
- Logging: `logger = logging.getLogger(__name__)` at module level. Module names
  are under the `robo43.` root so the single handler picks them up. NEVER call
  `basicConfig` or `addHandler` outside `robo43/logging_setup.py`.
- NEVER `sys.exit()` outside `robo43/cli/`. Raise instead.
- NEVER return an `Exception` instance as data. Either return a valid value or
  raise (see `robo43/exceptions.py`).
- NEVER `pdb.set_trace()`. Anywhere.
- NEVER index `hdul[0]` directly outside `robo43/io/fits_io.py`; use
  `science_hdus()` / `read_image()`.
- Matplotlib: any module that plots does `import matplotlib; matplotlib.use("Agg")`
  before `pyplot`, uses the OO API (`Figure` + `FigureCanvasAgg`), and closes
  figures. No `plt.show()` anywhere.
- A metric that cannot be computed is **omitted** from a FITS header, never
  written as a sentinel.

## `robo43/__init__.py`

```python
__version__: str            # "0.2.0.dev0"
def git_sha() -> str | None # short SHA of the installed checkout, None if unknown (cached)
```

## `robo43/exceptions.py`

```python
class Robo43Error(Exception): ...
class ConfigError(Robo43Error): ...
class MetadataError(Robo43Error): ...        # keyword contract violations
class ValidationError(MetadataError): ...    # preflight failure
class CalibrationError(Robo43Error): ...
class ReductionError(Robo43Error): ...
class DetectionError(Robo43Error): ...
class AstrometryError(Robo43Error): ...
class PhotometryError(Robo43Error): ...
class ProductError(Robo43Error): ...
class DatabaseError(Robo43Error): ...
class ExternalToolError(Robo43Error): ...    # source-extractor missing/failed
```

## `robo43/logging_setup.py`

```python
def setup_logging(
    level: str | int = "INFO",
    logfile: str | os.PathLike[str] | None = None,
    *,
    json_format: bool = False,
    queue: "multiprocessing.Queue | None" = None,
) -> logging.Logger
```
Idempotent: configures the `robo43` logger, removing handlers it previously
installed (mark them with an attribute) so a second call does not duplicate
lines. `queue` installs a `QueueHandler` for worker processes.

```python
def worker_log_config(queue) -> None      # called in Pool initializer
def log_listener(queue) -> ...            # parent-side QueueListener helper
```

## `robo43/config.py`

Dataclasses (all `@dataclass(frozen=True, slots=True)`), nested under
`Settings`. Field names are the TOML keys.

```python
@dataclass(frozen=True, slots=True)
class PathSettings:
    data_root: Path | None = None      # products stored relative to this
    workdir: Path = Path(".")
    outdir: Path | None = None         # default: workdir / "reduced"
    calibdir: Path | None = None
    fits_patterns: tuple[str, ...] = ("*.fits", "*.fit", "*.fts",
                                      "*.fits.fz", "*.fits.gz",
                                      "*.fit.fz", "*.fit.gz")

class AstrometrySettings:
    index_dir: Path | None = None      # was hard-coded /home/herpich/Documents/.astrometry
    index_scales: tuple[int, ...] | None = None   # None => derive from FOV
    search_radius_deg: float = 0.5
    scale_tolerance: float = 0.03      # +/- 3% on pixel scale
    solve_timeout_s: float = 120.0
    frame_budget_s: float = 300.0
    max_sources: int = 150
    min_sources: int = 20
    rms_max_arcsec: float = 0.3        # pass/fail threshold
    refine_with_gaia: bool = True
    sip_order: int = 3
    use_web_fallback: bool = False
    cache_file: Path | None = None

class PhotometrySettings:
    instrumental_zp: float = 20.0
    aperture_radii_arcsec: tuple[float, ...] = (2.0, 3.0, 4.0, 6.0, 8.0)
    detect_thresh: float = 1.5
    detect_minarea: int = 5
    deblend_nthresh: int = 32
    deblend_mincont: float = 0.005
    back_size: int = 64
    back_filtersize: int = 3
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
    reference_catalog: str = "gaia"     # gaia | apass | none
    zp_sigma_clip: float = 3.0
    zp_min_stars: int = 10
    zp_rms_max: float = 0.05            # above this => non-photometric
    catalog_format: str = "mef"         # mef | ldac | both

class CalibSettings:
    combine_method: str = "mean"        # mean | median (after clipping)
    sigma_clip_low: float = 3.0
    sigma_clip_high: float = 3.0
    sigma_clip_maxiters: int = 5
    max_memory_mb: float = 2048.0       # drives chunked combination
    flat_min_level_frac: float = 0.20   # of saturation
    flat_max_level_frac: float = 0.70
    flat_norm_region_frac: float = 0.5  # central fraction used for normalisation
    dark_temp_tolerance_c: float = 0.5
    dark_exptime_tolerance_s: float = 0.5
    dark_doubling_temp_c: float = 6.5   # Arrhenius-ish scaling
    dark_temp_extrap_max_c: float = 5.0
    hot_pixel_nmad: float = 5.0
    dead_pixel_frac: float = 0.5
    unstable_pixel_nmad: float = 5.0
    cosmic_ray_rejection: bool = True

class CoaddSettings:
    scale_by: str = "exptime"           # exptime | starflux | none
    sky_subtract: bool = False          # opt-in; source-masked when on
    reject_sigma: float = 3.0
    combine: str = "weighted_mean"

class DiagnosticsSettings:
    level: str = "summary"              # none | summary | full
    decimate: int = 4

class DatabaseSettings:
    url: str | None = None              # from ROBO43_DB_URL
    enabled: bool = False
    statement_timeout_ms: int = 60000
    pool_pre_ping: bool = True

class ServiceSettings:
    watch_dirs: tuple[Path, ...] = ()
    settle_seconds: float = 10.0
    rescan_seconds: float = 300.0
    workers: int = 4
    max_attempts: int = 3
    metrics_port: int = 9109
    strict: bool = True

class RuntimeSettings:
    nprocs: int = 1
    loglevel: str = "INFO"
    strict: bool = False                # refuse derived values
    assume_profile_defaults: bool = False
    dry_run: bool = False
    clobber: bool = False
    pdb: bool = False

@dataclass(frozen=True, slots=True)
class Settings:
    paths: PathSettings
    astrometry: AstrometrySettings
    photometry: PhotometrySettings
    calib: CalibSettings
    coadd: CoaddSettings
    diagnostics: DiagnosticsSettings
    db: DatabaseSettings
    service: ServiceSettings
    runtime: RuntimeSettings
    targets: TargetSettings             # aliases: dict[str,str]; calibration_object_names: tuple[str,...]

    def config_hash(self) -> str        # stable sha256[:16] of the whole tree
    def to_dict(self) -> dict[str, Any]
```

Loader:
```python
def load_settings(
    *,
    config_file: str | os.PathLike[str] | None = None,
    cli: Mapping[str, Any] | None = None,   # dotted keys: {"astrometry.index_dir": "..."}
    env: Mapping[str, str] | None = None,   # defaults to os.environ
) -> Settings
```
Precedence, highest first: `cli` > env `ROBO43_<SECTION>_<FIELD>` (also plain
`ROBO43_DB_URL`) > `config_file` > `~/.config/robo43/config.toml` > packaged
`robo43/data/config.default.toml`. Unknown keys in TOML raise `ConfigError`.
Instrument-specific values are NOT here — they live in the profile.

## `robo43/instruments/profile.py`

```python
@dataclass(frozen=True, slots=True)
class InstrumentLayout:
    science_hdu: int | str = 0          # int, EXTNAME, or "all"
    n_amplifiers: int = 1
    overscan: str | None = None         # IRAF-style "[2049:2080,1:2048]"
    trim: str | None = None
    flip_y: bool = False

@dataclass(frozen=True, slots=True)
class InstrumentProfile:
    name: str
    matches: dict[str, tuple[str, ...]]        # {"TELESCOP": ("ROBO43",), ...}, "*" wildcard
    pixel_scale_arcsec: float | None           # None means "derive"
    gain_e_per_adu: float | None
    read_noise_e: float | None
    saturation_adu: float | None
    nonlinearity_adu: float | None
    binning_keyword: str = "XBINNING"
    layout: InstrumentLayout
    valid_filters: tuple[str, ...]
    filter_map: dict[str, str]                 # raw header value -> canonical
    keywords: dict[str, tuple[str, ...]]       # canonical field -> ordered aliases
    imagetyp_map: dict[str, str]               # raw -> bias|dark|flat|science
    site_lat_deg: float | None
    site_lon_deg: float | None
    site_elevation_m: float | None
    index_scales: tuple[int, ...] | None
    search_radius_deg: float
    version: str = "1"
    source: str = "packaged"                   # packaged | user | database

    def canonical_filter(self, raw: str | None) -> str | None
    def canonical_frame_type(self, raw: str | None) -> str | None
    def aliases_for(self, field: str) -> tuple[str, ...]   # profile aliases + built-in defaults
    def earth_location(self) -> astropy.coordinates.EarthLocation | None
```

Built-in default alias tables live in `robo43/instruments/keywords.py` as
`DEFAULT_KEYWORD_ALIASES: dict[str, tuple[str, ...]]`; a profile's `keywords`
block *extends and takes precedence over* them.

Loading / resolution:
```python
def load_profile_file(path) -> InstrumentProfile
def available_profiles() -> dict[str, Path]      # user dir shadows packaged
def resolve_profile(
    *,
    name: str | None = None,
    header: fits.Header | Mapping[str, Any] | None = None,
    profile_toml: str | None = None,             # from a DB row
) -> InstrumentProfile
```
Order, first match wins: `name` → `profile_toml` → match on
`TELESCOP`/`INSTRUME` → `generic`.

Packaged profiles required: `robo43.toml` (values from
REFACTORING_PLAN.md §1B.2) and `generic.toml` (assumes nothing; derives pixel
scale from `XPIXSZ`+`FOCALLEN` or an existing WCS, saturation from `BITPIX`;
warns per derived value).

## `robo43/io/meta.py`

```python
class Source(enum.StrEnum):
    HEADER = "header"; ALIAS = "alias"; OVERRIDE = "override"
    PROFILE = "profile"; DERIVED = "derived"; DEFAULT = "default"

@dataclass(frozen=True, slots=True)
class FieldOrigin:
    source: Source
    detail: str = ""        # e.g. "OBJCTRA", "overrides.toml", "radec+dateobs+site"

@dataclass(frozen=True, slots=True)
class FrameMeta:
    path: Path
    # tier 1
    naxis1: int
    naxis2: int
    frame_type: str | None       # bias | dark | flat | science | None
    exptime: float | None
    date_obs: astropy.time.Time | None
    # tier 2
    filter: str | None
    ra_deg: float | None
    dec_deg: float | None
    pixel_scale: float | None    # arcsec/px
    ccd_temp: float | None
    gain: float | None
    read_noise: float | None
    saturation: float | None
    airmass: float | None
    # tier 3
    object: str | None
    telescope: str | None
    instrument: str | None
    observer: str | None
    binning: int | None
    focal_length_mm: float | None
    site_lat_deg: float | None
    site_lon_deg: float | None
    site_elevation_m: float | None
    mjd: float | None
    night: datetime.date | None  # local night of observation
    is_valid: bool               # honours the VALID keyword invalidate_frames writes
    profile_name: str
    origins: Mapping[str, FieldOrigin]

    def get(self, field: str) -> Any
    def origin(self, field: str) -> FieldOrigin | None
    def missing(self, fields: Iterable[str]) -> tuple[str, ...]
    def provenance_records(self) -> list[str]   # ["gain=override:overrides.toml", ...] for KWSRC<n>
    def provenance_json(self) -> dict[str, dict[str, str]]  # for reduced.images.metadata_provenance
```

`night` rule: the date of `date_obs` shifted back by 12 h (so an evening and the
following morning share one night).

## `robo43/io/headers.py`

```python
def normalise_radec(ra: Any, dec: Any) -> tuple[float | None, float | None]
    # accepts "10:44:32.1"/"10 44 32.1"/"161.264"/float/Quantity; returns degrees.
    # NEVER guess by counting colons on a float — type-check first.

def classify_frame_type(header, profile, *, object_value=None) -> tuple[str | None, Source]
    # prefers IMAGETYP via profile.imagetyp_map, falls back to OBJECT substrings,
    # honours the VALID keyword.

def derive_pixel_scale(header, profile) -> tuple[float | None, FieldOrigin]
    # 206.265 * XPIXSZ[um] * binning / FOCALLEN[mm], or from a valid WCS.

def derive_airmass(ra_deg, dec_deg, time, location) -> float | None

def read_meta(
    path: str | os.PathLike[str],
    profile: InstrumentProfile,
    *,
    overrides: "OverrideSet | None" = None,
    settings: Settings | None = None,
    header: fits.Header | None = None,      # avoid a re-read when already loaded
) -> FrameMeta
```
Precedence inside `read_meta`, per §1B.4 rule 1:
`--set` > overrides (most specific glob) > header/alias > profile > derived.
Every non-header value records a `FieldOrigin`. Derived values are recorded as
`DERIVED` and are refused when `settings.runtime.strict`.

## `robo43/io/validate.py`

```python
TIER1: tuple[str, ...] = ("naxis1", "naxis2", "frame_type", "exptime", "date_obs")
STAGE_REQUIREMENTS: dict[str, tuple[str, ...]]
    # "bias": (), "dark": ("ccd_temp",), "flat": ("filter",),
    # "reduce": ("filter",), "astrometry": ("ra_deg","dec_deg","pixel_scale"),
    # "photometry": ("filter","gain","saturation"), "zeropoint": ("airmass",)

@dataclass(frozen=True, slots=True)
class OverrideSet:
    defaults: dict[str, Any]
    per_pattern: tuple[tuple[str, dict[str, Any]], ...]   # ordered, most specific last
    cli_set: dict[str, Any]
    @classmethod
    def from_toml(cls, path) -> OverrideSet
    @classmethod
    def from_cli(cls, set_args: Sequence[str]) -> OverrideSet    # ["gain=1.48", ...]
    def merge(self, other: OverrideSet) -> OverrideSet
    def for_path(self, path: Path, root: Path | None = None) -> dict[str, tuple[Any, FieldOrigin]]

@dataclass(frozen=True, slots=True)
class ValidationIssue:
    path: Path; field: str; stage: str; reason: str

@dataclass(frozen=True, slots=True)
class ValidationReport:
    metas: tuple[FrameMeta, ...]
    issues: tuple[ValidationIssue, ...]
    @property
    def ok(self) -> bool
    def blocking(self) -> tuple[ValidationIssue, ...]
    def as_table(self) -> pandas.DataFrame     # field x resolution, grouped
    def emit_overrides_template(self, path) -> None

def validate(metas, *, stages, strict=False) -> ValidationReport
```
Values parsed from `--set`/TOML are coerced to the `FrameMeta` field type;
a bad coercion raises `ValidationError` naming the key.

## `robo43/io/fits_io.py`

```python
FITS_SUFFIXES: tuple[str, ...]

def iter_fits_files(paths, patterns=None, *, recursive=False) -> Iterator[Path]
    # accepts a dir, a glob, an explicit file, or a text file of paths (one per line)

@contextmanager
def open_fits(path, mode="readonly") -> Iterator[fits.HDUList]

def science_hdus(hdul, profile) -> list[fits.ImageHDU | fits.PrimaryHDU | fits.CompImageHDU]

def read_image(path, profile, *, hdu=None) -> tuple[numpy.ndarray, fits.Header]
    # float32 output, BSCALE/BZERO applied once at read, CompImageHDU transparent

def read_header(path, profile=None) -> fits.Header     # the ONE header read per file

def parse_region(spec: str) -> tuple[slice, slice]     # IRAF "[x1:x2,y1:y2]" -> numpy slices

def write_fits(path, hdus, *, overwrite=False, checksum=True, compress=False) -> Path
    # atomic: write to <path>.tmp, fsync, os.replace
```

## `robo43/io/inventory.py`

Replaces `collect_night_info.py`. Reads **each header exactly once**.

```python
INVENTORY_COLUMNS: tuple[str, ...]   # path + every FrameMeta scalar field + "origins_json"

def build_inventory(
    paths, *, profile=None, overrides=None, settings=None, nprocs=1
) -> pandas.DataFrame
    # one row per file; object dtype for strings (NEVER fixed-width U20/U23);
    # per-file profile resolution when profile is None.

def metas_from_inventory(df) -> list[FrameMeta]
def write_inventory(df, path) -> None      # csv or parquet by suffix
def filter_inventory(df, *, frame_type=None, filter=None, object=None,
                     night=None, exptime=None, ccd_temp=None,
                     valid_only=True) -> pandas.DataFrame
```

## `robo43/cli/main.py`

```python
def build_parser() -> argparse.ArgumentParser
def main(argv: Sequence[str] | None = None) -> int
```
Subcommand registry so later phases can add theirs without touching `main`:

```python
# robo43/cli/registry.py
@dataclass
class Subcommand:
    name: str
    help: str
    add_arguments: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace], int]

def register(cmd: Subcommand) -> None
def all_subcommands() -> list[Subcommand]
```
Each `robo43/cli/<name>.py` defines `SUBCOMMAND: Subcommand` and `main.py`
imports them lazily (a module list, imported on dispatch, so an optional
dependency missing does not break `robo43 --help`).

Common flags shared by all subcommands (put them in
`robo43/cli/common.py::add_common_arguments`):
`--config`, `--loglevel {DEBUG,INFO,WARNING,ERROR}`, `--verbose` (alias for
`--loglevel DEBUG`), `--logfile`, `--instrument`, `--overrides`,
`--set KEY=VALUE` (repeatable), `--assume-profile-defaults`, `--strict`,
`--permissive`, `--diagnostics {none,summary,full}`, `--np N`, `--dry-run`,
`--clobber`, `--pdb`.
There is NO `--debug`-swallows-errors flag: errors propagate; `--pdb` wraps
`main` in `pdb.post_mortem()`.

`main()` returns an exit code; it catches `Robo43Error` and prints
`error: <message>` returning 2, and returns 130 on `KeyboardInterrupt`.
Subcommand names (wired as their phases land): `inventory`, `inspect`,
`fixheader`, `calib`, `reduce`, `coadd`, `photometry`, `catalog`, `colour`,
`thumb`, `serve`, `status`, `requeue`, `db`.

## Testing conventions

`tests/` at repo root, `pytest`. Synthetic fixtures only — no external data,
nothing over ~256x256. Fixture generator at `tests/fixtures/make_synthetic.py`,
shared fixtures in `tests/conftest.py`. Tests that need `source-extractor`,
a live PostgreSQL (`ROBO43_DB_URL`), or astrometry index files must
`pytest.importorskip` / `pytest.mark.skipif` rather than fail.

Run with: `cd <repo> && python3 -m pytest -q`
Lint with: `ruff check src tests`

---

# ADDENDUM 1 — already implemented, do not rewrite

The orchestrator has already written and verified these. They are authoritative.
Read them; do not modify them (add tests for them if your brief says so).

- `src/robo43/__init__.py`      — `__version__`, `git_sha()`
- `src/robo43/exceptions.py`    — the full hierarchy above
- `src/robo43/logging_setup.py` — `setup_logging`, `worker_log_config`,
  `log_listener`, `JsonFormatter`, `describe_handlers`
- `src/robo43/config.py`        — all the settings dataclasses, `load_settings`,
  `origin_report`, `SCIENCE_SECTIONS`
- `src/robo43/data/config.default.toml`
- `src/robo43/data/__init__.py`, `src/robo43/data/instruments/__init__.py`
- empty `__init__.py` in every subpackage

Two refinements to the contract above:

1. `Settings.config_hash()` hashes only `SCIENCE_SECTIONS = ("astrometry",
   "calib", "coadd", "photometry", "targets")` — the sections that can change
   the numbers in a product. `paths`, `db`, `service`, `diagnostics` and
   `runtime` are deployment detail and are excluded.
2. `TargetSettings.canonical_target(name)` exists and does the alias mapping;
   `aliases` is `dict[str, tuple[str, ...]]`.

# ADDENDUM 2 — subcommand discovery (removes all future conflicts on main.py)

`robo43/cli/main.py` must NOT hold a hand-maintained import list that later
phases have to edit. Instead it iterates a fixed tuple of *candidate module
names* and imports each inside `try/except ImportError`, logging a debug line
and skipping any that is absent or whose optional dependency is missing:

```python
SUBCOMMAND_MODULES: tuple[str, ...] = (
    "inventory", "inspect", "fixheader", "calib", "reduce", "coadd",
    "photometry", "catalog", "colour", "thumb", "serve", "status",
    "requeue", "db", "config",
)
```

So a later phase adds a subcommand by creating `robo43/cli/<name>.py` with a
module-level `SUBCOMMAND: Subcommand` and nothing else. `robo43 --help` must
work with any subset of those modules present, and must not fail because
`psycopg` or `watchdog` is not installed.

---

# ADDENDUM 3 — the product model (plan §4.4, §4.4.1, §5.1, Appendix B)

Authoritative for every phase from 4 onward. `products/mef.py` and
`products/catalog.py` own the writing; everyone else goes through them.

Two new `CalibSettings` keys were added by the orchestrator and exist already:
`apply_gain: bool = True` and `nonlinearity_correct: bool = False`.

## Units convention (pin this; it is otherwise the easiest thing to get wrong)

The reduction converts to **electrons** as its last pixel step when
`settings.calib.apply_gain` (the default), and writes `BUNIT = 'electron'`;
with `apply_gain = false` the data stay in ADU and `BUNIT = 'adu'`. Therefore:

- `InstrumentProfile.saturation_adu` / `nonlinearity_adu` are always **ADU**.
- The `SATLEVEL` card written into a product header is in the **data's own
  units** (electrons when gain was applied), and `GAIN` records the factor used.
- Every saturation test in the pipeline compares like with like. Convert once,
  at the point of comparison, never twice.

## `robo43/products/dq.py` — the data-quality bit assignment

The plan (§5.1 item 3) fixes bits 0–5; the rest are ours and must not be
renumbered later, because they end up in archived files.

```python
class DQ(enum.IntFlag):
    GOOD        = 0
    HOT         = 1 << 0   # hot pixel, from the normalised dark map
    DEAD        = 1 << 1   # dead / low-QE, from the master flat
    UNSTABLE    = 1 << 2   # high per-pixel variance across the bias stack
    SATURATED   = 1 << 3   # saturated in THIS frame
    COSMIC_RAY  = 1 << 4   # flagged by astroscrappy
    NO_DATA     = 1 << 5   # outside the detector after reprojection
    NONLINEAR   = 1 << 6   # above the non-linearity threshold in this frame
    BAD_COLUMN  = 1 << 7   # persistent column
    NO_FLAT     = 1 << 8   # no master flat was available for this filter
    CLIPPED     = 1 << 9   # rejected by the coadd outlier rejection

DQ_DESCRIPTIONS: dict[DQ, str]
def dq_bintable() -> fits.BinTableHDU     # the optional CATFLAGS HDU: bit -> meaning
def dq_summary(mask) -> dict[str, float]  # affected pixel fraction per class
```
`DQ` planes are `int32` images with `EXTNAME = 'DQ'`.

## `robo43/products/mef.py` — one reduced frame is ONE file

```
HDU 0  SCI       PrimaryHDU   reduced pixels; full instrument + provenance header
HDU 1  ERR        ImageHDU    1-sigma uncertainty, same shape, same units as SCI
HDU 2  DQ         ImageHDU    int32, bit-encoded (above)
HDU 3  CATALOG   BinTableHDU  the catalogue, WITH ITS OWN HEADER (below)
HDU 4  CATFLAGS  BinTableHDU  optional bit -> meaning legend
```
`ERR` is 1-sigma, not variance. Anything holding variance internally converts
at write time and the header says which (`EXTNAME='ERR'`).

```python
@dataclass(slots=True)
class ReducedProduct:
    sci: np.ndarray                 # float32
    err: np.ndarray                 # float32, 1-sigma
    dq: np.ndarray                  # int32
    header: fits.Header             # SCI header, incl. WCS
    catalog: astropy.table.Table | None = None
    catalog_header: fits.Header | None = None
    bunit: str = "electron"

    def to_hdulist(self, *, include_catflags: bool = True) -> fits.HDUList
    @classmethod
    def from_hdulist(cls, hdul) -> ReducedProduct
    def write(self, path, *, overwrite=False, compress=False) -> Path   # atomic
    @classmethod
    def read(cls, path) -> ReducedProduct

def calibration_cards(*, bias=None, dark=None, flat=None, mask=None,
                      dark_scale=None, crrej=False, overscan=False,
                      profile=None, meta=None, settings=None
                      ) -> list[tuple[str, Any, str]]
```
`calibration_cards` produces the Appendix B.1 block:
`PIPEVER PIPEGIT PIPECFG PIPEDATE INSTPROF INSTPVER BIASFIL DARKFIL DARKSCL
FLATFIL MASKFIL CRREJ OSCANCOR WCSSOLVD ASTSOLVR ASTRMS1 ASTRMS2 ASTNREF
ASTEPOCH NCOMBINE TEXPTIME` plus the `KWSRC<n>` provenance records from
`FrameMeta.provenance_records()`.

Rules, from the plan, that are not negotiable:
- A master that was **skipped** records the string `'NONE'` (§1B.5), e.g.
  `BIASFIL = 'NONE'`. A metric that could not be **computed** is omitted
  entirely (§4.4.1). Those are different cases; do not conflate them.
- `BIASFIL`/`DARKFIL`/`FLATFIL`/`MASKFIL` carry the master's **basename**, and
  its sha256 goes in a matching `HISTORY` line, so a product can be traced to
  the exact calibration file.
- Reuse `robo43.provenance.provenance_cards` and `apply_cards`; do not
  re-implement the pipeline-version block.

## `robo43/products/catalog.py` — the CATALOG HDU

Columns: the superset in §4.4.1. Every column carries a `TUNIT`. Per-frame
content depends on which stages ran; `MAG_CALIB`/`MAGERR_CALIB` are **absent**
when no zero point was derived, and `MAGSYS = 'INSTRUMENTAL'` says so.

```python
CATALOG_COLUMNS: dict[str, tuple[str, str]]    # name -> (fits format, unit)

@dataclass(slots=True)
class CatalogMetrics:
    # every field Optional; None means "omit the card", never a sentinel
    ...  # one field per Appendix B.2 keyword, snake_cased
    def to_header(self) -> fits.Header
    @classmethod
    def from_header(cls, header) -> CatalogMetrics

def build_catalog_hdu(table, metrics, *, aperture_radii=None) -> fits.BinTableHDU
def read_catalog(path_or_hdul) -> tuple[Table, CatalogMetrics]
def write_ldac(table, path, *, image_header) -> Path   # LDAC_IMHEAD + LDAC_OBJECTS
def extract_catalog(path, out, *, fmt="fits") -> Path  # fits | parquet | csv
```
Header keywords, grouped exactly as Appendix B.2:
`EXTNAME CATSOFT CATSOFTV CATCFG CATDATE PIPEVER PIPEGIT | DETTHRSH DETMINAR
DETFILT DEBLNTH DEBLMINC BACKSIZE BACKFILT WEIGHTTP | NSOURCES NSTARS NCLEAN
NREJECT NSATUR NBADPIX | ZP ZPERR ZPNSTAR ZPRMS ZPCAT ZPBAND MAGSYS EXTINCT
AIRMASS PHOTOMET APERRAD APERUNIT | FWHMPIX FWHMARC ELLIP THETA SKYLEV SKYRMS
SKYUNIT LIMMAG3 LIMMAG5 SNRLIM | GAIN RDNOISE SATLEVEL PIXSCALE EXPTIME FILTER |
ASTSOLVR ASTRMS1 ASTRMS2 ASTNREF ASTEPOCH RADESYS WCSSOLVD`

For coadd products `EXPTIME` is replaced by `TEXPTIME` and `NCOMBINE` appears.

**The phase-9 ingester reads its metrics out of this header rather than
recomputing them**, so `CatalogMetrics.from_header(to_header(m)) == m` must
round-trip exactly. That round-trip is a required test.

## `robo43/calib/masks.py`

```python
@dataclass(frozen=True, slots=True)
class MaskSet:
    dq: np.ndarray                     # int32, the static (per-night) DQ plane
    summary: dict[str, float]          # affected fraction per class
    def write(self, path, *, cards=None, overwrite=False) -> Path
    @classmethod
    def read(cls, path) -> MaskSet

def build_mask(*, dark_current_map=None, master_flat=None,
               bias_variance=None, settings, profile) -> MaskSet
def detect_bad_columns(image, settings) -> np.ndarray
def saturation_mask(data, saturation) -> np.ndarray     # per-frame, DQ.SATURATED
```

## `robo43/calib/set.py` — `CalibrationSet` (plan §3.3)

The legacy reducer calls `fits.getdata` on the masters **for every science
frame** (`src/process_robo43_frames.py:202`, `:250`). Load once, pass around.

```python
@dataclass(slots=True)
class CalibrationSet:
    bias: np.ndarray | None; bias_var: np.ndarray | None; bias_path: Path | None
    flats: dict[str, np.ndarray]; flat_var: dict[str, np.ndarray]
    flat_paths: dict[str, Path]
    darks: tuple[DarkEntry, ...]                 # see calib/dark.py
    dark_current_map: np.ndarray | None; dark_current_path: Path | None
    mask: MaskSet | None
    @classmethod
    def load(cls, calibdir, *, settings, profile, filters=None) -> CalibrationSet
    def flat_for(self, filter_name) -> tuple[np.ndarray | None, Path | None]
    def dark_for(self, exptime, ccd_temp, *, settings) -> DarkSelection
    def checksums(self) -> dict[str, str]
```
It must be picklable and cheap to send to a worker, or be built in the pool
initializer. Missing masters are `None`, never an exception (§1B.5).

## `robo43/reduce/frame.py` — order of operations and the status record

§4.4 order, each step skippable and each recorded:
**overscan → trim → bias → dark (scaled) → flat → bad-pixel mask → cosmic-ray
rejection → gain to electrons → variance and DQ planes**.

```python
@dataclass(slots=True)
class ReductionStatus:
    path: Path; raw_name: str; proc_code: int; proc_status: str
    steps_applied: tuple[str, ...]; steps_skipped: tuple[str, ...]
    error: str | None = None; product_path: Path | None = None
    wcs_solved: bool = False; astrom_rms_arcsec: float | None = None
    fwhm_arcsec: float | None = None; zeropoint: float | None = None
    is_photometric: bool | None = None; duration_s: float | None = None
    def as_row(self) -> dict[str, Any]

def reduce_frame(meta, calib, *, settings, profile) -> tuple[ReducedProduct, ReductionStatus]
```
Phase 3.7: workers **return** a `ReductionStatus`; the parent collects them from
`imap_unordered` and writes **one** CSV at the end. Never a per-frame status CSV
(Appendix A bug 59), never a mutated instance dict (bug 2).

Phase 3.6: the pool worker is a **module-level** function taking a small
picklable payload — never a bound method, which pickles the whole
`ProcessFrame` instance including its `Logger` for every task.

## Graceful degradation (§1B.5) — implement exactly this table

| Missing | Behaviour |
|---|---|
| master bias | skip; `BIASFIL='NONE'`; continue |
| master dark | skip; warn dark current uncorrected |
| master flat for the filter | skip; set `DQ.NO_FLAT`; `is_photometric = False`; catalogue still produced but flagged |
| bad-pixel mask | minimal mask from saturation + NaNs only |
| astrometric solution | reduced image still written; catalogue in pixel coordinates, `ra`/`dec` null, `WCSSOLVD = F` |
| photometric reference catalogue | instrumental magnitudes only; `ZP` absent, `MAGSYS = 'INSTRUMENTAL'` |

The rule: **always produce the reduced image and the catalogue**, and record
precisely which corrections were applied and which were not. A missing
calibration frame must never raise (the legacy code aborts at
`src/process_robo43_frames.py:223` and `:240`).

---

# ADDENDUM 4 — photometry, astrometry, coadd, database, service

Authoritative for plan phases 6, 7, 8, 9 and §4.5/§4.6.

## `robo43/photometry/quality.py` — the QUALITY_MASK bits

Retained as a column, **never applied destructively** (§6 item 3), so cuts can
be revisited without re-running SExtractor. These numbers reach archived
catalogues; do not renumber.

```python
class Reject(enum.IntFlag):
    NONE          = 0
    FLAGS         = 1 << 0   # SExtractor FLAGS above the allowed value
    BADPIX        = 1 << 1   # IMAFLAGS_ISO != 0 (hot/dead/unstable/CR/saturated in isophote)
    SATURATED     = 1 << 2   # FLUX_MAX above saturation_fraction * saturation
    NONLINEAR     = 1 << 3   # FLUX_MAX above nonlinearity_fraction * saturation
    EDGE          = 1 << 4   # centroid within max(edge_margin_fwhm*FWHM, edge_margin_px)
    FWHM_LOW      = 1 << 5   # narrower than fwhm_min_arcsec (cosmic-ray-like)
    FWHM_HIGH     = 1 << 6   # wider than fwhm_max_arcsec (blended/extended)
    ELONGATED     = 1 << 7   # ellipticity above ellipticity_max (trailed)
    LOW_SNR       = 1 << 8   # FLUX_AUTO/FLUXERR_AUTO below snr_min
    NOT_STELLAR   = 1 << 9   # CLASS_STAR below class_star_min
    BLENDED       = 1 << 10  # neighbour within isolation_radius_arcsec
    NO_GAIA_MATCH = 1 << 11  # no Gaia DR3 counterpart

@dataclass(frozen=True, slots=True)
class QualityReport:
    mask: np.ndarray                       # int32 per source, the Reject bitmask
    counts: dict[str, int]                 # per-cut attrition, IN ORDER APPLIED
    n_input: int; n_clean: int
    def clean_index(self, *, allow=Reject.NONE) -> np.ndarray
    def metric_cards(self) -> dict[str, int]   # NSOURCES/NCLEAN/NREJECT/NSATUR/NBADPIX

def assess(table, *, settings, profile, meta, dq=None, wcs=None,
           reference=None) -> QualityReport
def reference_sample(table, report, *, settings) -> Table   # FLAGS==0 & Reject.NONE
```
Every step is logged with a count so the attrition is auditable (§6 item 2).
Absolute FWHM windows in **arcsec** from config — never the legacy
80th-percentile flux cut, which throws away 80 % of the field regardless of
quality (Appendix A bug 20), and never `FLAGS < 4`, which admits flag 1
(blended) and 2 (deblended).

## `robo43/photometry/sextractor.py` — one wrapper, one config

Replaces the two divergent copies (`src/process_robo43_frames.py:587`,
`src/get_photometry.py:71`; Appendix A bugs 19, 56).

```python
def sextractor_config(*, settings, profile, meta, weight_path=None,
                      flag_path=None, catalog_path=None) -> dict[str, Any]
def run_sextractor(image_path, *, settings, profile, meta,
                   weight_path=None, flag_path=None, workdir=None) -> Table
```
Mandatory corrections to the legacy config:
- `WEIGHT_TYPE = MAP_WEIGHT` with the **real per-frame inverse-variance map**.
  The legacy `WEIGHT_TYPE = BACKGROUND` makes SExtractor **ignore**
  `WEIGHT_IMAGE` entirely, so the weight map is never used, and the path was
  resolved against the repo root rather than the working directory
  (Appendix A bug 18).
- `FLAG_IMAGE` = the DQ plane, `FLAG_TYPE = OR`, and `IMAFLAGS_ISO`,
  `NIMAFLAGS_ISO` added to the output parameters.
- `SATUR_KEY`/`SATUR_LEVEL`, `GAIN`, `PIXEL_SCALE`, `MAG_ZEROPOINT` all come
  from the profile/settings — never hard-coded, and never disagreeing between
  two call sites.
- Filter and NNW files come from `robo43.provenance.package_data_path`.
- The binary is found through `robo43.external.find_sextractor`, so a missing
  install is an `ExternalToolError` with a clear message, not a `sewpy`
  traceback.
- The catalogue is **returned and persisted** — the legacy
  `src/get_photometry.py:105` computes `sources` and never uses it, making the
  whole module a no-op (Appendix A bug 12).

## `robo43/photometry/aperture.py` (ex `get_relative_photometry.py`)

Fix all of §6 item 4 / Appendix A bugs 11, 40, 41, 42, 43:
- **No double background subtraction.** `Background2D` is subtracted from the
  image and then the annulus background is measured on the *already subtracted*
  image and subtracted again. Measure the annulus on the **original** data.
- Annulus background is `ApertureStats(...).median` with sigma clipping, not
  `aperture_sum/area`, which any star in the annulus biases.
- Errors from `photutils.utils.calc_total_error` with the **real** gain and the
  product's `ERR` plane — not `np.sqrt(abs(data))`, which assumes gain 1 when
  the configured gain is 10 — and the annulus uncertainty propagated into
  `flux_err`.
- Reject any aperture overlapping a `DQ`-flagged pixel or a saturated peak.
- Airmass/extinction correction, optional colour term.
- Comparison-star magnitudes from per-target config
  (`TargetSettings.comparison_magnitudes`) or the DB, never hard-coded.
- Filenames via `Path.with_suffix('')`/`removesuffix`, never
  `.strip('.fits')`, which strips *characters* from both ends
  (`first_flat.fits` -> `rst_fla`).
- The `refstar` argument is either genuinely used as a second comparison with
  an agreement check, or removed. Decide and say which.

## `robo43/photometry/zeropoint.py`

```python
@dataclass(frozen=True, slots=True)
class ZeroPoint:
    zp: float; zp_err: float; n_stars: int; rms: float
    catalog: str; band: str; extinction: float | None; airmass: float | None
    is_photometric: bool
def fit_zeropoint(table, report, *, settings, profile, meta, wcs) -> ZeroPoint | None
def query_reference(ra_deg, dec_deg, radius_arcmin, *, settings, epoch) -> Table
```
Sigma-clipped fit per frame per filter with an airmass term; frames whose ZP
scatter exceeds `settings.photometry.zp_rms_max` are flagged non-photometric —
this doubles as a cloud detector. Returns `None` (not an exception) when no
reference catalogue is reachable; the caller then writes
`MAGSYS='INSTRUMENTAL'` and omits `ZP` (§1B.5).
Gaia queries are cached on disk, proper-motion-propagated to the epoch of
observation, and network failure degrades rather than aborts.

## `robo43/astrometry/*`

```python
# indexes.py
def index_files(*, settings, profile, naxis1, naxis2) -> list[Path]
def derive_scales(naxis1, naxis2, pixel_scale) -> tuple[int, ...]
def get_solver(*, settings, profile, shape) -> Solver     # PROCESS-GLOBAL SINGLETON
def reset_solver() -> None                                # tests only
```
**§3.1 is the single largest performance win in the plan.** The legacy code
builds `astrometry.Solver(series_5200.index_files(...))` *inside* the per-frame
function, which is called from inside two nested retry loops, so a hard frame
re-reads gigabytes of index files up to ten times. Build it **once per worker
process** (module-level lazy singleton, or stashed in a process global by
`Pool(initializer=...)`) and keep it alive for the run.

```python
# detect.py
def detect_sources(data, *, settings, profile, dq=None, wcs=None) -> Table
```
Threshold from `astropy.stats.sigma_clipped_stats` (or `Background2D` +
`detect_threshold`), **never `sigma * np.std(whole_image)`**, which in a crowded
or nebulous field is set by the sources themselves — wildly wrong exactly where
solving is hard (Appendix A bug 17). Mask bad pixels and cosmic rays *before*
detection. **Grid-sample** 50–150 sources spread across the field rather than
pure brightest-first, which biases toward one corner in a nebula field. Reject
saturated ones. Never return `None`; raise `DetectionError` instead.

```python
# solver.py
@dataclass(frozen=True, slots=True)
class SolveResult:
    wcs: astropy.wcs.WCS; solver: str; n_ref: int
    rms1_arcsec: float | None; rms2_arcsec: float | None
    epoch: float | None; seeded_from: str | None
def solve(data, meta, *, settings, profile, dq=None, seed=None) -> SolveResult
def solution_cache(settings) -> SolutionCache   # keyed (object, filter, rounded pointing)
```
- `SizeHint` from the configured plate scale ±`scale_tolerance` (3 %), not the
  legacy 0.2–1.0 arcsec/px — a factor-5 window when the scale is known to be
  0.529.
- `PositionHint` radius from `settings.astrometry.search_radius_deg` (0.5°),
  not the legacy 2.0°.
- **One** detection pass at an adaptive threshold plus at most two retries at
  pre-computed levels — not the sigma-decrement ladder, which was partly driven
  by `TypeError`s from functions returning `Exception` objects (§2.3).
- Ordered fallback chain, each step recorded: cached/seeded → local index solve
  → local `solve-field` binary → astrometry.net web → give up and flag.
  `run_astrometry_solver2` (the web fallback) is fully written in the legacy
  code but **never called from anywhere** (Appendix A bug 26) — wire it in with
  a real backoff whose counter increments on every exception, not only
  `TimeoutError`. `solve-field` is NOT installed on this machine, so that step
  must degrade cleanly when the binary is absent.
- **Blind-solve reuse**: solve one frame per (target, dither-group), seed the
  rest from it and refine. Persist the cache to disk.
- Every failure raises `AstrometryError`; nothing returns `None` or an
  `Exception` instance.

```python
# refine.py
def refine_wcs(wcs, sources, *, settings, profile, meta) -> SolveResult
```
Cross-match the cleaned catalogue against Gaia DR3 propagated to the epoch of
observation, fit **SIP** with `astropy.wcs.utils.fit_wcs_from_points`, iterate
with sigma-clipped rejection. Record `ASTRMS1/2`, `ASTNREF`, `ASTSOLVR`,
`ASTEPOCH`. A solve whose RMS exceeds `settings.astrometry.rms_max_arcsec` is
marked **unsolved**, not silently coadded: `coadd_frames.py` only checks
`wcs.has_celestial`, which is true for any solution however bad.

RA/Dec plumbing (§7 item 6, Appendix A bug 3): normalise **once**, on ingest,
into `FrameMeta.ra_deg`/`dec_deg`; nothing downstream re-parses a header string.
SIMBAD lookups are cached by object name for the run and persisted — a night of
200 frames of one target currently makes 200 identical queries — and
`result['ra']`/`result['dec']` vs `RA`/`DEC` column naming across `astroquery`
versions must be handled.

## `robo43/products/coadd.py` (§4.5)

The legacy `coadd_frames.py:117-122` multiplies each frame by `1/std**2` and
then takes a **median** — neither a weighted median nor a weighted mean, biased
and on no consistent flux scale; and `1/std**2` over the whole frame is
dominated by the sources, not the noise. Correct sequence:
1. Scale each frame to a common flux scale (`EXPTIME`, or the median flux ratio
   of common stars when `settings.coadd.scale_by == "starflux"`).
2. Per-frame noise from a **source-masked, sigma-clipped background**.
3. Inverse-variance-weighted mean with per-pixel outlier rejection (this is
   what removes cosmic rays and satellite trails), marking rejects `DQ.CLIPPED`.
4. Write the weight/exposure map as an extension; `TEXPTIME` must include
   **every** contributing frame — the legacy `exptimes` list never records the
   seed frame, so `TEXPTIME` is short by one exposure (bug 34).

Also fix bugs 9, 32, 33, 35, 36: choose the reference frame and seed the stack
**consistently** (the legacy code seeds `files[0]` unreprojected while using
`files[j]` as the reference, and its `if i == j` skip test compares a running
counter to an index, so the reference is stacked twice); read `self.clobber`
not the module-level `args`, and honour it on the write instead of always
passing `overwrite=True`; tolerate a missing `OBJECT` card
(`.get('OBJECT', None).lower()` -> `AttributeError`); either implement
`--cutout` via `Cutout2D` on the output WCS or remove the flag; and make sky
subtraction **opt-in per target** using a source-masked `Background2D` — a
plain whole-frame `np.nanmedian` removes real signal for a nebula that fills
the field, which is exactly the Eta Car/M42 case this pipeline is used for.
Reuse `calib/combine.py` rather than writing a third combination routine.
Coadd products are `ReducedProduct`s with `TEXPTIME`/`NCOMBINE` and a
`CATALOG` HDU on the same convention.

## `robo43/products/colour.py`, `thumbnails.py` (§4.6)

`make_coloured_images.py:169` sums reprojected channels with `np.nansum`
ignoring exposure time, so a channel with more frames is simply brighter —
scale by exposure time (or the coadd's `TEXPTIME`) before summing, and use the
§4.5 coadd routine rather than the private duplicate. Fix bugs 13, 14, 37, 38:
`self.logger` is never defined so `--cutout` always raises `AttributeError`;
`output_path` is never defined so `--verbose` raises `NameError` at the very end
of a long run; `--degrade_psf` computes `fwhm_values` and discards them, making
the flag a no-op; and `fit_shape` is passed the **whole image** where it should
be a small cutout like `(11, 11)`.

## `robo43/db/*` (phase 9)

SQLAlchemy 2.0 ORM (`db/models.py`), Alembic migrations from the first commit
(`migrations/`), `psycopg`. Schema exactly as §9.1: schemas `raw` and
`reduced`, and **the `NOT NULL`/`CHECK` constraints ARE the phase-1B keyword
contract** — that is the whole point, so transcribe them faithfully.

```python
# session.py
def make_engine(settings) -> Engine        # pool_pre_ping, statement_timeout
def session_scope(settings) -> Iterator[Session]
def require_db(settings) -> None           # DatabaseError when [db] extra absent
# query.py
def list_work(settings, *, night=None, object=None, filter=None,
              state=None, instrument=None, limit=None) -> pandas.DataFrame
    # returns THE SAME in-memory inventory structure io/inventory.py produces,
    # so the reducer does not care where its work list came from
def load_profile_from_db(settings, name) -> InstrumentProfile
def register_instrument(settings, profile_path) -> int
def export_instrument(settings, name, out) -> Path
# ingest.py
def ingest_reduced(settings, product_path, status, *, meta, calib) -> int
def ingest_master(settings, path, *, cal_type, ...) -> int
def ingest_coadd(settings, path, members) -> int
def backfill(settings, root) -> int
```
Non-negotiable rules from §9.2:
- One transaction per logical product; write to a temp path, `fsync`, atomically
  rename, **then** commit the row — never a row pointing at a half-written file.
- Upsert on the natural keys (`ON CONFLICT ... DO UPDATE`) so re-ingest is
  idempotent.
- `pipeline_version`, git SHA and `config_hash` on **every** row.
- Read per-frame metrics **out of the `CATALOG` HDU header**, never recompute
  them, so file and database cannot disagree.
- Bulk-load `source_catalog` with `cursor.copy()`, not row-by-row inserts.
- Store paths **relative to `settings.paths.data_root`**, never absolute.
- Credentials only from `ROBO43_DB_URL` or a `~/.pgpass`-style file with a
  permission check. Never in the repo, never in a default.
- `--no-db` must keep the whole pipeline working; `[db]` is an optional extra.

**Environment limit:** PostgreSQL is running on this machine but the account has
no `CREATEDB` right, so live integration tests cannot create a scratch database.
Make the DB tests run against SQLite where the SQL is portable, and mark the
PostgreSQL-only ones `@pytest.mark.needs_db`, skipped unless `ROBO43_DB_URL` is
set. Verify the DDL by rendering it with `sqlalchemy.schema.CreateTable` against
the `postgresql` dialect and asserting on the SQL text.

## `robo43/service/*` (phase 8)

Same code paths as the CLI — one implementation, two front ends. Watcher
(`watchdog` + settling delay + periodic full rescan), scheduler with the
explicit dependency graph `PENDING -> READY -> RUNNING -> DONE | FAILED |
SKIPPED`, queue as a `reduced.jobs` table using `SELECT ... FOR UPDATE SKIP
LOCKED`, workers with attempt counts, exponential backoff and a dead-letter
state. **Idempotency key = (input checksum, pipeline version, config hash)** —
that is what makes the service restartable at any point. Graceful `SIGTERM`,
`--once` drain-and-exit, `/healthz` and `/metrics` (Prometheus text), JSON logs
in service mode, single-instance PostgreSQL advisory lock, `robo43 status` and
`robo43 requeue`, plus a `systemd` unit and a `Dockerfile`/compose file shipped
in the repo.

---

# ADDENDUM 5 — also already implemented

`src/robo43/products/dq.py` is written and verified by the orchestrator. Import
`DQ`, `DQ_DESCRIPTIONS`, `DQ_DTYPE`, `STATIC_FLAGS`, `FATAL_FLAGS`,
`dq_bintable`, `dq_summary`, `describe` from it. Do not rewrite it and do not
define a second bad-pixel encoding anywhere.

`CalibSettings` also already carries `apply_gain: bool = True` and
`nonlinearity_correct: bool = False`.

---

# ADDENDUM 6 — more landed modules, and two cross-module rules

Verified by the orchestrator and now authoritative. Import them; do not
reimplement, and do not define a competing version of anything they own.

- `src/robo43/calib/combine.py` — `CombineResult`, `combine_stack`,
  `combine_lazy`, `row_block_size`, `sigma_clip_stack`. Clips along `axis=0`;
  the read-noise variance is imposed as a genuine **floor** (`np.maximum`), and
  with no read noise supplied a variance below `1e-6 x` the frame's median is
  masked as unmeasurable rather than believed. `CombineResult` also carries
  `sigma_low`, `sigma_high`, `maxiters`, and `combine_stack`/`combine_lazy`
  take a keyword-only `fill`.
- `src/robo43/reduce/cosmics.py` — `CosmicRayResult`, `detect_cosmics`,
  `apply_to_dq`.
- `src/robo43/photometry/lightcurve.py` — `EpochMagnitude`,
  `average_magnitudes`, `read_photometry_csvs`, `combine_in_flux_space`.
- `src/robo43/products/dq.py`, `config.py`, `logging_setup.py`,
  `exceptions.py`, `provenance.py`, `external.py`, and the `cli/` framework.

## Rule A — construct result dataclasses by KEYWORD, never positionally

Several result types carry more fields than this contract's abbreviated
sketches showed, because `header_cards()` has to report the parameters actually
used (`CosmicRayResult` gained `enabled` and `sigclip`; `CombineResult` gained
`sigma_low`, `sigma_high`, `maxiters`). More will grow the same way. So:
**every construction and every unpacking of these dataclasses uses keyword
arguments and named attribute access.** Never positional construction, never
tuple-unpacking, never an assumption about field count or order. In particular
`products/mef.py` must read `result.mask` / `result.header_cards()` by name.

## Rule B — the photometry table schema is fixed here

`photometry/lightcurve.py` consumes a tidy long-format frame with exactly these
column names, and `photometry/aperture.py` **must emit them** (in its returned
frame and in any CSV it writes):

```
target   filter   jd   mag   mag_err
```
plus, optionally, `comp_star`, `aperture_arcsec`, `airmass`, `method` — which
are also the `reduced.photometry` columns in the §9.1 schema, so the CSV path
and the database path agree by construction. Do **not** reintroduce the legacy
`targetstar_mag` / `targetstar_mag_err` names.

## Note on testing cosmic-ray rejection

`astroscrappy` builds its own noise model from `gain` and `readnoise`, so a
synthetic test frame whose injected noise disagrees with the gain it is given
produces a huge false-positive rate that looks like a wrapper bug and is not.
On a frame with self-consistent Poisson-plus-read noise the wrapper gives 100 %
recall at a 0.03 % false-positive rate; with Gaussian noise 2.7x larger than
the gain implies, the same call flags 7 % of the frame. Build synthetic frames
as `(poisson(signal_adu * gain) + normal(0, read_noise_e)) / gain`.

---

# ADDENDUM 7 — masks landed; one new config key

`src/robo43/calib/masks.py` is written and verified: `MaskSet`, `build_mask`,
`detect_bad_columns`, `saturation_mask`, `nonlinearity_mask`, `minimal_mask`.
Verified against injected truth on a non-square frame: hot pixels, a dead
block, unstable pixels and bad columns all recovered; vignetting is not called
dead; a thermal gradient is not called hot; and a defective column is found at
every index including 0, 1 and N-1, with zero false positives on five clean
darks and a vignetted flat.

Two design points worth knowing, because they are easy to reintroduce:

1. **Column detection runs on every supplied source** (dark map, flat, bias
   variance) and ORs the results. A column can be visible in only one of them,
   so preferring one and ignoring the rest silently misses real defects.
2. **The column test is a fractional departure from the immediate
   neighbourhood**, governed by the new setting `CalibSettings.bad_column_frac`
   (default `0.10`). An absolute, scatter-based threshold does not work: the
   residual against a local baseline is dominated by the frame's own curvature
   rather than by noise, so its MAD is not a noise estimate and collapses
   towards zero on a low-noise frame, flagging everything. Do not "improve"
   this back into a sigma test.

`CalibSettings` therefore now carries `bad_column_frac`, alongside the earlier
`apply_gain` and `nonlinearity_correct`.

---

# ADDENDUM 8 — resolution of the `calibration_cards` gap

The product-model implementation correctly identified that the specified
`calibration_cards` signature has no parameter for eight Appendix B.1
keywords, so nothing currently writes them:

```
ASTSOLVR  ASTRMS1  ASTRMS2  ASTNREF  ASTEPOCH  WCSSOLVD  NCOMBINE  TEXPTIME
```

**Decision.** Do not widen `calibration_cards`. It records *calibration*
provenance, and the missing keywords belong to two other stages that run later
and would drag their types into `products/mef.py`. Add two sibling helpers in
`products/mef.py` instead, both taking plain scalars so `mef.py` never imports
from `robo43.astrometry` or `robo43.products.coadd`:

```python
def astrometry_cards(
    *,
    solver: str | None = None,
    rms1_arcsec: float | None = None,
    rms2_arcsec: float | None = None,
    n_ref: int | None = None,
    epoch: float | None = None,
    solved: bool = False,
) -> list[tuple[str, Any, str]]
    # ASTSOLVR ASTRMS1 ASTRMS2 ASTNREF ASTEPOCH WCSSOLVD

def coadd_cards(
    *, n_combine: int | None = None, total_exptime: float | None = None
) -> list[tuple[str, Any, str]]
    # NCOMBINE TEXPTIME
```

`WCSSOLVD` is the one exception to "omit what you cannot compute": it is always
written, as `F` when there is no accepted solution, because §1B.5 requires a
reduced product to state explicitly that its catalogue is in pixel coordinates.
Every other card here is omitted when its value is `None`.

Ownership: `astrometry_cards` is called by phase 7 (`astrometry/refine.py`
consumers) and `coadd_cards` by phase 4.5 (`products/coadd.py`). Whoever
implements either phase adds the corresponding helper and its test.

---

# ADDENDUM 9 — two findings and one decision from the pause checkpoint

## A 61st bug, not in the plan's Appendix A

The plan's §6 quotes the legacy quality cuts but does not spot that the third
one **overwrites** the mask instead of compounding it. Confirmed directly at
`src/process_robo43_frames.py:627-634`:

```python
mask = (sources['FLAGS'] < 4)
mask &= (sources['FWHM_IMAGE'] > np.percentile(
    sources['FWHM_IMAGE'][sources['FWHM_IMAGE'] > 0], 1))
# mask &= (sources['CLASS_STAR'] > 0.5)
sources = sources[mask]
mask = (sources['FLUX_AUTO'] > np.percentile(
    sources['FLUX_AUTO'], 80))
sources = sources[mask]
```

The second `mask = ...` uses plain `=`. Because the first two cuts are applied
via `sources = sources[mask]` before it, they do take effect — but the pattern
is a trap: the reassignment means the flux cut is computed on the *already
filtered* table and any further cut appended after it would silently drop the
earlier ones. `photometry/quality.py` replaces the whole idiom with an
accumulating bitmask, so this class of error cannot recur. Add it to the bug
list as **61** if Appendix A is ever renumbered.

## `generic.toml` cannot resolve a multi-extension foreign instrument

Verified empirically while designing the inventory work: the packaged
`generic` profile cannot resolve the `synth2` multi-extension fixture. That
matters because §1B.2 promises `generic` "assumes nothing beyond the required
keywords". Either `generic` must handle a multi-extension layout (deriving
`science_hdu` when the primary HDU carries no pixels), or foreign MEF data
genuinely needs its own profile. The plan's own acceptance criterion allows the
latter — it asks for "a new profile TOML and **no code changes**".

**Decision.** A test fixture's profile must NOT be shipped in
`robo43/data/instruments/`; packaged data is for real instruments. Instead,
`--instrument` and `resolve_profile(name=...)` must accept **either a
registered profile name or a filesystem path to a TOML**, so
`tests/fixtures/synth2.toml` can be used directly without writing into the
user's `~/.config/robo43/instruments/`. That is useful to real users too: it
lets someone try a profile before installing it. If `resolve_profile` does not
yet accept a path, adding that is part of finishing phase 1B.

Separately, `generic.toml` should still be improved to derive `science_hdu`
from the file when the primary HDU has no data — that is a real gap against the
§1B.5 graceful-degradation promise, and worth a test.
