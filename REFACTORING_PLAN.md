# ROBO43 Reduction Pipeline — Refactoring Plan

**Status:** in progress — see `docs/refactor-resume/README.md` for what has
landed, what is outstanding, and how to resume. Interface signatures for the
whole refactor are specified in `docs/INTERFACE_CONTRACT.md`.
**Date:** 2026-09-06 (plan), execution started 2026-09-08
**Scope:** `src/*.py` (10 modules, ~2 600 lines), `data/` (SExtractor config), packaging,
multi-telescope support, service mode, PostgreSQL integration.

This document is the working plan and remains the specification; it is deliberately not edited
as work lands, so that it stays a faithful record of the findings. Each work item lists
concrete findings with `file:line` anchors, the proposed change, and an acceptance criterion so
progress is checkable.

---

## 0. Where the code stands today

Ten independent CLI scripts under `src/`, each with its own `argparse`, its own logger, and its own
copy of shared constants. There is no package, no `pyproject.toml`, no tests, and no configuration
file — the plate scale, gain, saturation level, filter names, SExtractor parameters, and even an
absolute path to one developer's astrometry index cache are hard-coded in the source.

| Module | Lines | Role | Health |
|---|---:|---|---|
| `process_robo43_frames.py` | 1034 | bias/flat + astrometry driver | fragile; has a live `pdb.set_trace()` |
| `make_calibration_frames.py` | 272 | master bias / dark / flat | dark is a stub; combine statistics wrong |
| `coadd_frames.py` | 197 | stacking | reference-frame and weighting bugs |
| `make_coloured_images.py` | 367 | RGB composites | two `AttributeError`/`NameError` on live paths |
| `get_relative_photometry.py` | 247 | differential photometry | double background subtraction |
| `get_photometry.py` | 133 | SExtractor catalogues | discards its own results |
| `calculate_average_mags.py` | 124 | nightly averages | dead JD-matching logic, ragged output |
| `collect_night_info.py` | 62 | header inventory | silent string truncation |
| `invalidate_frames.py` | 57 | mark bad frames | OK |
| `fix_header.py` | 54 | filter-name normalisation | ignores its own argument |
| `make_thumb.py` | 51 | quick-look PNGs | OK |

The refactor is organised into eleven phases. Phases 1, 1B, 2, and 3 are prerequisites for
everything else; phases 4–9 carry the goals in the brief; phase 10 is the safety net that should be
built incrementally alongside all of them.

Two requirements cut across the whole plan and are worth stating up front, because they shape the
abstractions rather than sitting in one phase:

- **The pipeline is not ROBO43-only.** It must reduce data from any telescope whose frames carry the
  key set of keywords (phase 1B). Through the database those keywords are guaranteed, because the
  data model constrains the header format; through a user-supplied file list they are not, so
  anything missing must be supplied by the user before the run starts.
- **A reduced product is one file.** The reduced image and the photometric catalogue extracted from
  it ship together as a multi-extension FITS, with the catalogue in its own `CATALOG` HDU whose
  header carries the frame's photometric metrics (§4.4.1).

---

## Phase 1 — Package skeleton and configuration (goal 3, part one)

Everything else depends on this. Today `self.files_path` is computed as
`dirname(dirname(abspath(__file__)))` (`process_robo43_frames.py:112`, `get_photometry.py:62`) so the
SExtractor `.conv`/`.nnw` files are found only when running from a git checkout. That breaks the
moment the code is `pip install`ed.

### 1.1 Target layout

```
robo43-reduction-pipeline/
├── pyproject.toml
├── src/robo43/
│   ├── __init__.py            # __version__
│   ├── config.py              # Settings dataclasses + TOML/YAML loader
│   ├── logging_setup.py       # the single setup_logging()
│   ├── instruments/
│   │   ├── profile.py         # InstrumentProfile loader + resolution order
│   │   └── (profiles live in data/instruments/*.toml)
│   ├── io/
│   │   ├── fits_io.py         # open/write with provenance headers, MEF-aware
│   │   ├── headers.py         # keyword aliasing + value normalisation
│   │   ├── meta.py            # FrameMeta — the canonical metadata contract
│   │   ├── validate.py        # preflight keyword check + override resolution
│   │   └── inventory.py       # night info (ex collect_night_info)
│   ├── calib/
│   │   ├── combine.py         # stack statistics, shared by bias/dark/flat
│   │   ├── bias.py
│   │   ├── dark.py            # NEW — see phase 5
│   │   ├── flat.py
│   │   └── masks.py           # NEW — bad-pixel / hot-pixel masks
│   ├── reduce/
│   │   ├── frame.py           # per-frame reduction (ex ProcessFrame)
│   │   └── cosmics.py         # astroscrappy wrapper
│   ├── astrometry/
│   │   ├── detect.py          # source detection for solving
│   │   ├── solver.py          # local index solver, cached
│   │   ├── refine.py          # Gaia-anchored WCS refinement + SIP
│   │   └── indexes.py         # index-file discovery/caching
│   ├── photometry/
│   │   ├── sextractor.py      # one SEW wrapper, one config
│   │   ├── aperture.py        # ex get_relative_photometry
│   │   ├── quality.py         # NEW — bad-pixel/source rejection (goal 4)
│   │   └── zeropoint.py       # NEW — Gaia/APASS calibration
│   ├── products/
│   │   ├── mef.py             # SCI/ERR/DQ/CATALOG assembly and provenance
│   │   ├── catalog.py         # catalogue BINTABLE + its metrics header
│   │   ├── coadd.py
│   │   ├── colour.py
│   │   └── thumbnails.py
│   ├── db/                    # phase 7
│   │   ├── models.py
│   │   ├── session.py
│   │   ├── query.py
│   │   └── ingest.py
│   ├── service/               # phase 6
│   │   ├── daemon.py
│   │   ├── watcher.py
│   │   └── worker.py
│   ├── cli/
│   │   ├── main.py            # `robo43` dispatcher
│   │   └── ...                # one thin module per subcommand
│   └── data/                  # package data: *.conv, default.nnw, default.sex, default.param
│       └── instruments/       # robo43.toml, generic.toml, <telescope>.toml
└── tests/
```

Keep thin backwards-compatible shims in `src/` for one release cycle so existing shell scripts and
cron jobs do not break on the day of the switch; have them emit a `DeprecationWarning`.

### 1.2 Configuration object

Introduce `robo43/config.py` with dataclasses loaded from TOML, with this precedence:
CLI flag > environment variable (`ROBO43_*`) > user file (`~/.config/robo43/config.toml`) >
**instrument profile** (phase 1B) > packaged default. Every value currently hard-coded moves here;
everything in the `instrument.*` group below belongs to the *instrument profile* rather than the
global config, because it varies per telescope:

| Current hard-coded value | Location | New setting |
|---|---|---|
| `PIXEL_SCALE: 0.529` | `process_robo43_frames.py:611`, `get_photometry.py:93` | `instrument.pixel_scale_arcsec` |
| `GAIN: 10` | `process_robo43_frames.py:610` | `instrument.gain_e_per_adu` |
| `SATUR_LEVEL: 200000` / `55000` (inconsistent!) | `:606` / `get_photometry.py:89` | `instrument.saturation_adu` |
| `MAG_ZEROPOINT: 20` | both | `photometry.instrumental_zp` |
| `cache_directory="/home/herpich/Documents/.astrometry"` | `process_robo43_frames.py:398` | `astrometry.index_dir` |
| `scales={3,4,5,6}` | `:399` | derived from pixel scale + NAXIS, overridable |
| `lower/upper_arcsec_per_pixel=0.2/1.0` | `:415-416` | derived from `pixel_scale ± tolerance` |
| filter slot map | `fix_header.py:22-30` | `instrument.filter_map` |
| `valid_filters` list | `make_calibration_frames.py:131-132` | `instrument.valid_filters` |
| `object_names_to_guess` / `_to_exclude` | `process_robo43_frames.py:116-124` | `targets.aliases` / `targets.calibration_object_names` |
| comparison-star magnitudes | `get_relative_photometry.py:150` | per-target TOML or DB table |

The saturation-level disagreement between the two SExtractor configs (200 000 vs 55 000 ADU) is
itself a bug — one of the two catalogues is silently wrong about which sources are saturated. Pick
the true detector value once, in config.

### 1.3 Packaging

- `pyproject.toml` with `hatchling` or `setuptools>=64`, `src/` layout, `requires-python = ">=3.12"`
  (forced: `get_relative_photometry.py:75`, `:87`, `:180` and `calculate_average_mags.py:118` use
  newlines inside f-string expressions, a 3.12-only syntax).
- Package data: `[tool.hatch.build] include = ["src/robo43/data/*"]`; access at runtime through
  `importlib.resources.files("robo43.data")`, never `__file__` arithmetic.
- Console entry points:
  ```toml
  [project.scripts]
  robo43 = "robo43.cli.main:main"
  ```
  with subcommands `inventory`, `calib`, `reduce`, `coadd`, `photometry`, `colour`, `thumb`,
  `serve`, `db`.
- Optional dependency groups: `[db]` (SQLAlchemy, psycopg), `[service]` (watchdog, APScheduler),
  `[dev]` (pytest, ruff, mypy, pytest-cov).
- Fix `requirements.txt`: it lists only `photutils astropy numpy astroquery` but the code also
  imports `pandas`, `matplotlib`, `scipy`, `reproject`, `sewpy`, and `astrometry`. Pin lower bounds.
- Declare the external binary dependency (`source-extractor`) explicitly and fail with a clear
  message when it is absent, instead of the current opaque `sewpy` traceback.
- Replace `#!/bin/python3` shebangs (all modules) with `#!/usr/bin/env python3`, or drop them once
  entry points exist.

**Acceptance:** `pip install .` into a clean venv, then `robo43 --help` and
`robo43 reduce <dir>` both work from a directory that is not the checkout.

---

## Phase 1B — Instrument abstraction and the keyword contract

**Requirement.** The pipeline must reduce images from telescopes other than ROBO43, provided the
frames carry the key set of keywords that makes reduction possible. Two entry paths, with different
guarantees:

- **Database-driven** (phase 9): keywords are guaranteed compatible, because the data model puts
  strict constraints on the header format. No prompting, no guessing — a row that does not satisfy
  the contract never becomes a job.
- **User-supplied file list**: no such guarantee. If a required keyword is absent, the user must
  supply it. The pipeline never invents a value silently.

Guaranteed products in both cases: **the reduced images and the photometric catalogues** (§4.4).

### 1B.1 What is ROBO43-specific today

Nothing in the current code can process another telescope's data without editing source:

| Assumption | Location |
|---|---|
| plate scale 0.529 arcsec/px | `process_robo43_frames.py:611`, `get_photometry.py:93` |
| gain 10 e-/ADU, saturation 200 000 / 55 000 ADU | `:610`, `:606`, `get_photometry.py:89` |
| ROBO43 filter-wheel slot names (`Slot 0/Slot 1` → `U-Bessell`) | `fix_header.py:22-30` |
| `valid_filters` list of ROBO43 filters | `make_calibration_frames.py:131-132` |
| ROBO43 target aliases (`eta car`, `m42`, …) | `process_robo43_frames.py:116-121` |
| frame type inferred from `OBJECT` substrings only | `:122-124`, `make_calibration_frames.py:60-72` |
| single-extension data always at `hdul[0]` | every module |
| `*.fits` glob only (no `.fit`, `.fts`, `.fits.fz`, `.fits.gz`) | `:148`, `make_calibration_frames.py:54`, and others |
| astrometry index scales `{3,4,5,6}` fixed for one field of view | `:399` |

### 1B.2 Instrument profiles

A profile is a TOML file describing one telescope+camera combination. Packaged profiles live in
`robo43/data/instruments/`; user profiles in `~/.config/robo43/instruments/` and shadow them.

Resolution order, first match wins:
`--instrument <name>` CLI flag → `instrument` column from the database row → match on the frame's
`TELESCOP`/`INSTRUME` (with the profile declaring the values it matches) → `generic`.

A `generic` profile ships as the fallback: it assumes nothing beyond the required keywords and
derives what it can (pixel scale from `XPIXSZ`/`FOCALLEN` or from an existing WCS, saturation from
`BITPIX`), warning for each derived value.

Profile contents:

```toml
[instrument]
name = "robo43"
matches = { TELESCOP = ["ROBO43", "Robo-43"], INSTRUME = ["*"] }
pixel_scale_arcsec = 0.529        # or "derive"
gain_e_per_adu     = 10.0
read_noise_e       = 7.0
saturation_adu     = 55000
nonlinearity_adu   = 45000
binning_keyword    = "XBINNING"

[instrument.layout]
science_hdu = 0                   # int, name, or "all" for multi-extension
n_amplifiers = 1
overscan = "[2049:2080,1:2048]"   # optional, per amplifier
trim     = "[1:2048,1:2048]"
flip_y   = false

[instrument.filters]
valid = ["U-Bessell", "B-Bessell", "V-Bessell", "R-Bessell", "I-Bessell",
         "Clear", "Halpha", "Methane"]
[instrument.filters.map]          # raw header value -> canonical name
"Slot 0/Slot 1" = "U-Bessell"
"Slot 0/Slot 2" = "B-Bessell"

[instrument.keywords]             # canonical key <- ordered list of header aliases
exptime   = ["EXPTIME", "EXPOSURE"]
imagetyp  = ["IMAGETYP", "IMGTYPE", "OBSTYPE", "FRAMETYP"]
ccd_temp  = ["CCD-TEMP", "CCDTEMP", "SET-TEMP", "TEMPERAT"]
ra        = ["RA", "OBJCTRA", "TELRA", "CRVAL1"]
dec       = ["DEC", "OBJCTDEC", "TELDEC", "CRVAL2"]

[instrument.imagetyp_map]         # raw value -> canonical frame type
"Light Frame" = "science"; "Bias Frame" = "bias"; "Dark Frame" = "dark"
"Flat Field"  = "flat";    "zero" = "bias";       "object" = "science"

[instrument.site]
lat_deg = -22.53; lon_deg = -45.58; elevation_m = 1864

[instrument.astrometry]
index_scales = "derive"           # from FOV, or an explicit list
search_radius_deg = 0.5
```

### 1B.3 The canonical metadata contract

Every module stops reading raw header cards. `io/meta.py` defines a frozen `FrameMeta` dataclass and
`io/headers.py::read_meta(path, profile, overrides)` produces it — resolving aliases, normalising
values (filter names, image types, RA/Dec to degrees, dates to `Time`), and recording the *provenance
of each field* (`header` / `alias` / `override` / `profile` / `derived`).

Required keywords, by tier. A stage runs only if its tier-2 keywords resolve; otherwise it is
skipped with an explicit reason, not silently.

| Tier | Canonical field | FITS source | Needed for |
|---|---|---|---|
| **1 — always required** | `naxis1`, `naxis2` | `NAXIS1/2` | everything |
| | `frame_type` | `IMAGETYP` (or classifier) | selecting what to reduce |
| | `exptime` | `EXPTIME` | dark scaling, coadd, photometry |
| | `date_obs` | `DATE-OBS` | grouping by night, airmass, Gaia epoch |
| **2 — stage-gated** | `filter` | `FILTER` | flats, coadd, photometry (not bias/dark) |
| | `ra`, `dec` | `RA`/`OBJCTRA`/`CRVAL1` | astrometry position hint |
| | `pixel_scale` | profile, or `XPIXSZ`+`FOCALLEN`, or `SECPIX` | astrometry scale hint |
| | `ccd_temp` | `CCD-TEMP` | dark matching and scaling |
| | `gain`, `read_noise` | `GAIN`/`EGAIN`, `RDNOISE` | error model, catalogues |
| | `saturation` | `SATURATE` or profile | saturation rejection (goal 4) |
| | `airmass` | `AIRMASS`, or derived from RA/Dec/date/site | extinction, zero points |
| **3 — enriching** | `object`, `telescope`, `instrument`, `observer`, `binning`, `focal_length`, `site_*`, `mjd` | as named | provenance, grouping, reporting |

Derivation rules are explicit and recorded: `airmass` from RA/Dec + `DATE-OBS` + site coordinates;
`pixel_scale` from `206.265 · XPIXSZ[µm] · binning / FOCALLEN[mm]`, or from an existing valid WCS;
`mjd` from `DATE-OBS`. A derived value is never treated as if it came from the header.

### 1B.4 Manual file lists: preflight validation and overrides

New stage, run before any reduction when the work list did **not** come from the database:

```
robo43 inspect <dir|list>  [--instrument NAME]
    → a table of every canonical field × how it resolved (header / alias / profile / derived / MISSING)
    → grouped by (instrument, filter, exptime) so a night collapses to a few rows
    → --emit-overrides overrides.toml writes a pre-filled template with blanks
      only where something is missing, so the user edits rather than authors it
```

If anything required is missing, the run **fails fast** with a message naming the file, the missing
canonical field, and the stage it blocks. The user then supplies it by any of:

- `--set KEY=VALUE` (repeatable) — applies to every file in the run;
- `--instrument <name>` — a profile that supplies static values (pixel scale, gain, saturation,
  filter map, site);
- `--overrides overrides.toml` — per-file or per-glob values, the general case:
  ```toml
  [defaults]
  gain = 1.48
  saturation = 65000
  pixel_scale = 0.34

  ["2026-09-05/flat_*.fits"]
  imagetyp = "flat"
  filter = "V"

  ["2026-09-05/n1_0042.fits"]
  ra = 161.264; dec = -59.684
  ```
- `--assume-profile-defaults` — accept the profile's values for anything absent, with a warning per
  keyword. Off by default.

Rules that keep this honest:

1. Precedence: `--set` > `--overrides` (most specific glob wins) > header/alias > profile > derived.
2. No interactive prompting — it would break service mode (phase 8) and non-TTY runs.
3. Every non-header value is written into the product header as `KWSRC<n>` provenance records and
   into `reduced.images.metadata_provenance` (jsonb), so a catalogue can always be traced back to
   whether its gain was measured or asserted.
4. `--strict` (default in service mode) refuses derived values outright; `--permissive` allows them.
5. Overrides never modify the raw files. `fix_header.py`'s in-place rewriting becomes an explicit,
   backed-up, `HISTORY`-recording `robo43 fixheader` subcommand.

### 1B.5 Graceful degradation for foreign data

Foreign data will often lack part of the calibration set. Define which stages are skippable and what
the product looks like when they are, rather than crashing (today `flat_correction` returns `None`
and the caller falls over, `process_robo43_frames.py:240`, and a missing master bias raises
`FileNotFoundError` at `:223`):

| Missing | Behaviour |
|---|---|
| master bias | skip; record `BIASFIL = 'NONE'`; continue |
| master dark | skip; warn that dark current is uncorrected |
| master flat for the filter | skip; mark the frame `is_photometric = false`; catalogue still produced but flagged |
| bad-pixel mask | build a minimal mask from saturation + NaNs only |
| astrometric solution | reduced image still written; catalogue written in pixel coordinates with `ra`/`dec` null and `WCSSOLVD = F` |
| photometric reference catalogue | instrumental magnitudes only; `ZP` absent, `MAGSYS = 'INSTRUMENTAL'` |

The rule: **always produce the reduced image and the catalogue**, and record precisely which
corrections were applied and which were not.

### 1B.6 File-format generality

- Accept `.fits`, `.fit`, `.fts`, and the compressed `.fz`/`.gz` variants; drive this from a config
  list rather than the hard-coded `*.fits` globs.
- Read multi-extension and multi-amplifier data via the profile's `layout` block; never index
  `hdul[0]` directly (`io/fits_io.py` provides `science_hdus(hdul, profile)`).
- Handle `CompImageHDU` (tile-compressed) input transparently, and offer compressed output.
- Respect `BSCALE`/`BZERO` and integer input, converting to float32 once, at read.

**Acceptance:** a night of images from a second, non-ROBO43 telescope reduces end-to-end using only a
new profile TOML and no code changes; the same night with three keywords deliberately stripped fails
`robo43 inspect` with a precise report, and succeeds once an `--overrides` file supplies them, with
the provenance of those three values visible in the output headers.

---

## Phase 2 — Cross-cutting hygiene

### 2.1 Remove the debugger from production paths

`process_robo43_frames.py:984-985` contains an **unconditional** `import pdb; pdb.set_trace()` on
the main reduction path, immediately after RA/DEC is found. Any run that reaches it hangs forever
waiting on stdin — fatal under `nohup`, cron, or `multiprocessing`. There are a further ~14
`pdb.set_trace()` calls guarded by `self.debug`, plus one in `get_relative_photometry.py:231-232`
guarded by nothing.

Replace the whole `_brake_point` idiom with:
- `--debug` raising instead of swallowing (`raise` at the point of failure),
- `--pdb` (opt-in, off by default) wrapping the entry point in `pdb.post_mortem()`,
- structured error records written to the status table rather than interactive breakpoints.

### 2.2 One logger, configured once

`setup_logging()` is duplicated verbatim in `process_robo43_frames.py:23-41` and
`get_photometry.py:10-28`; `make_calibration_frames.py:31-39` has a third variant hard-wired to
`DEBUG`. All three call `addHandler` unconditionally, so calling them twice duplicates every line.

Also: `logger.setLevel(loglevel if verbose else logging.WARNING)` means `--loglevel DEBUG` does
nothing unless `--verbose` is also passed — the two flags are redundant and confusing. Keep
`--loglevel` only; make `--verbose` an alias for `--loglevel DEBUG`.

Move to a single `robo43/logging_setup.py` that is idempotent, configures the root `robo43` logger,
and supports a `QueueHandler` so worker processes log safely (needed for phases 6 and 7).

### 2.3 Error handling contract

Three functions return an `Exception` **object** as if it were data:
`run_sewpy` (`process_robo43_frames.py:655`), `run_daofinder` (`:579`), and by omission
`run_astrometry_solver1` (`:377-444`, which falls off the end returning `None` when there is no
match). Callers then do `len(sorted_sources)` (`:694`) or tuple-unpack the `None` (`:708`), so the
control flow is driven by `TypeError`s caught in a broad `except Exception`. That is why the
sigma-lowering retry loop "works": it is being driven by unpacking errors, not by detection logic.

Define a small exception hierarchy (`Robo43Error`, `CalibrationError`, `AstrometryError`,
`PhotometryError`, `DetectionError`) and make every helper either return a valid result or raise.

### 2.4 Unbound-variable and control-flow bugs

- `run_astrometry_solver1` (`:391-394`): `_ra_deg`/`_dec_deg` are assigned **only** inside
  `if ':' in hdul[0].header['RA']`. If RA is already numeric, the solver call at `:418` raises
  `UnboundLocalError`. Worse, `':' in <float>` raises `TypeError` when the header stores RA as a
  number — which is exactly what SIMBAD writes back at `:348-351` when it returns decimal degrees.
- `solver_astrometry` (`:657-850`): `solver_used` is assigned at `:711`, `:738`, `:804`; every
  `break` path before those leaves it unbound, and `:798` reads it. `stars_used` (`:708`, `:771`) and
  `_fwhm_median` (`:687`) have the same problem in the DAOStarFinder branch.
- `get_fits_files` (`:173`): `sys.exit(1)` inside what should be a library call, aborting the whole
  run because *one* processed file already exists.
- `flat_correction` (`:240`) returns `None` without setting `proc_code`, so `guess_ra_dec`'s
  `self.proc_status[raw_name]['proc_code']` lookup at `:309` raises `KeyError` (caught at `:313` and
  misreported as "previous processing steps failed").
- `process_frame` (`:981`) calls `guess_ra_dec(proc_path, ...)` which opens `proc_path` — but
  `proc_path` is only written when `--save_processed` is given (`:970`). Without that flag the whole
  astrometry branch dies on `FileNotFoundError`.
- `guess_ra_dec` (`:324`) opens with `mode='update'` and returns the open `HDUList` without ever
  closing it; the caller keeps using it. File-descriptor leak across thousands of frames.
- `coadd_frames.py:93-98`: the reference frame is chosen as `files[j]` (first frame with valid WCS)
  but the stack is seeded with `fits.getdata(files[0])`, and the skip test is `if i == j` where `i`
  is a running counter and `j` an index — so when `j > 0` the wrong frame is seeded unreprojected
  and the reference frame is stacked twice.
- `coadd_frames.py:101-113`: `exptimes` never records the seed frame, so `TEXPTIME` (`:129`) is
  short by one exposure.
- `coadd_frames.py:181`: reads the module-level global `args.clobber` from inside a method instead
  of `self.clobber`, and then calls `writeto(overwrite=True)` at `:186` regardless — `--clobber` is
  effectively ignored on the write.
- `coadd_frames.py:58`: `fits.getheader(f).get('OBJECT', None).lower()` → `AttributeError` on any
  frame without an `OBJECT` card.
- `make_coloured_images.py:222`: `self.logger.warning(...)` — `ColouredImageMaker` never defines
  `self.logger`, so `--cutout` always raises `AttributeError`.
- `make_coloured_images.py:361`: `print(f"... {output_path}")` — `output_path` is never defined;
  `--verbose` always raises `NameError` at the very end of a long run.
- `make_coloured_images.py:192-211`: `degrade_psf` computes `fwhm_values` and then discards them —
  the flag is a no-op. Also passes `fit_shape=img_shape` (the *whole image*) to `fit_fwhm`, which
  should be a small cutout shape like `(11, 11)`.
- `get_relative_photometry.py:117`: `image_file.strip('.fits')` strips *characters* from both ends,
  so `first_flat.fits` → `rst_fla`. Use `Path.with_suffix('')` / `removesuffix`.
- `calculate_average_mags.py:84-86`: `data['targetstar_mag'] == median_mag` compares an individual
  magnitude to a magnitude derived from the *median flux* — it essentially never matches, so the
  branch is dead and the fallback always fires.
- `calculate_average_mags.py:88`, `:60-71`: `jd` leaks across the filter loop, and when the CSV is
  missing no values are appended to any column, producing a ragged dict that `pd.DataFrame` rejects.
- `fix_header.py:33-42`: `fix_header(self, imagename)` ignores `imagename` and reads `self.filename`,
  which `main` sets as a loop variable (`:47`). Also `header['FILTER']` raises `KeyError` when the
  card is absent, and the function rewrites files in place with no `HISTORY` record and no backup.
- `collect_night_info.py:23-29`: fixed-width dtypes (`U20` for object, `U23` for `DATE-OBS`) silently
  truncate. Build the DataFrame from Python lists instead.
- `get_photometry.py:56`: `f.replace('.fits', '_proc.fits')` applied to a list that may already
  contain `_proc` names produces `_proc_proc.fits`.
- `get_photometry.py:66`, `:117`: references a module-level `logger` that only exists under
  `if __name__ == '__main__'` — `NameError` on import.

### 2.5 Matplotlib in worker processes

`plot_frame` (`process_robo43_frames.py:852-886`) uses the pyplot global state machine and is called
three times per frame (`:959`, `:968`, `:847`). Under `multiprocessing.Pool` this is both a
correctness hazard and a large slowdown, and when `--save_processed` is not set it calls
`plt.show()` — blocking every worker on a GUI window that will never be closed.

Set `matplotlib.use("Agg")` at import in any non-interactive path, use the object-oriented API
(`Figure`/`FigureCanvasAgg`), close figures explicitly, and gate diagnostic plots behind an explicit
`--diagnostics {none,summary,full}` flag defaulting to `summary`.

**Acceptance:** `ruff check` and `mypy --strict` clean on the new package; a grep for
`pdb.set_trace` returns nothing outside tests; a full reduction run under `nohup` completes without
touching stdin.

---

## Phase 3 — Performance (goal 1, second half)

Measured-by-inspection hot spots, roughly in order of expected win.

### 3.1 Astrometry index files reloaded per frame — the dominant cost

`run_astrometry_solver1` (`:396-410`) builds `astrometry.Solver(series_5200.index_files(...))`
**inside the per-frame function**, and that function is called from inside two nested retry loops
(`:708`, `:771`). Every attempt re-reads gigabytes of index files from disk. With sigma stepping down
in 0.5 increments from 5.0 to 1.0, a single hard frame can reload the indexes ten times.

Fix: build the solver **once per worker process** (module-level lazy singleton, or a
`Pool(initializer=...)` that stashes it in a process-global), and keep it alive for the run. Expect
the largest single speedup in the whole refactor.

### 3.2 Redundant header reads

`get_fits_files` (`:139-145`, `:176-181`) calls `fits.getheader(f)` **once per exclusion keyword per
file** — four disk reads per file for the `['flat','dark','bias','trash']` list — and then a further
`fits.getheader(f)` per file for the object filter at `:186-187`. `make_calibration_frames.py:153-154`
re-reads every header once per filter in the filter loop. `coadd_frames.py:55-59` reads each header
twice.

Fix: read each header exactly once into an in-memory inventory (a `pandas.DataFrame` keyed by path),
then filter that. This is also the natural bridge to the database in phase 7 — the inventory becomes
a DB query.

### 3.3 Master calibration frames re-read per science frame

`bias_subtraction` (`:202`) and `flat_correction` (`:250`) call `fits.getdata` on the master frames
for *every* science frame. Load masters once into a `CalibrationSet` object passed to the reducer;
in multiprocessing, load them in the pool initializer or share via `multiprocessing.shared_memory`.

### 3.4 Memory: whole stacks held in RAM

`create_master_bias` (`:88-94`), `create_master_flat` (`:161-176`), and `coadd_images`
(`coadd_frames.py:93-122`) all build a Python list of full 2-D arrays and then call `np.median`,
which internally allocates a further copy. For an N-frame stack of an H×W detector this is
`~2·N·H·W·8` bytes.

Fix: implement chunked (row-block) combination in `calib/combine.py` — process `n_rows` at a time,
with the block size derived from a `--max-memory` setting. Optionally use `ccdproc.Combiner` with
`mem_limit`, which already implements this.

### 3.5 Diagnostic plots

Three PNGs written per frame (pre-bias, post-flat, astrometry overlay) at `figsize=(10,8)`. Under
`--diagnostics summary` write one overlay per frame; under `none`, write none. Compute the
percentile stretch on a decimated view (`data[::4, ::4]`) rather than the full array.

### 3.6 Parallelism

`pool.map(self.process_frame, fits_files)` (`:1022-1023`) pickles the bound method, and therefore the
entire `ProcessFrame` instance including its `logging.Logger`, for every task. Beyond being slow,
this is why `self.proc_status` never comes back (see 3.7).

Fix: make the worker a module-level function taking a small picklable payload
`(path, config, calib_paths)`; use `Pool(initializer=...)` for the expensive per-process setup;
switch `map` → `imap_unordered` with a chunk size so results stream back and progress is visible.

### 3.7 Status collection is silently broken under `--np > 1`

`self.proc_status` is a plain instance dict mutated by the workers (`:898-907`). Under
`multiprocessing` each worker mutates its **own copy**; the parent's dict stays empty. Then
`organize_proc_status()` (`:998-1008`) runs `np.array([], dtype=...)` on the empty list and
`pd.DataFrame` produces an empty CSV — so every parallel run loses its processing record.
Independently, `process_frame` writes a *separate* per-frame CSV at `:992-996`, littering the output
directory with one file per frame.

Fix: have workers **return** a status record; the parent collects them from `imap_unordered` and
writes one CSV (and, in phase 7, one DB transaction) at the end.

**Acceptance:** a benchmark script over a fixed 100-frame night, recorded before and after; target
≥5× wall-clock improvement on the astrometry stage and a flat memory profile independent of stack
depth.

---

## Phase 4 — Correct the reduction science (goal 1, first half)

Bugs here change *numbers*, not just robustness, so each needs a regression test with synthetic data.

### 4.1 Stack combination statistics

`create_master_bias` (`make_calibration_frames.py:94-101`) first collapses the stack with
`np.median(bias_frames, axis=0)` and *then* applies `sigma_clip` to the resulting 2-D image. That
clips **spatially** — it rejects pixels that are outliers relative to their neighbours in the master,
which is not what sigma clipping in a stack means and will mask real structure (e.g. genuine bias
structure or a real gradient). The same pattern is repeated for flats at `:176-180`.

Correct order: sigma-clip **along the stack axis** (`axis=0`) across frames, then average the
surviving values per pixel. `astropy.stats.sigma_clip(np.array(frames), axis=0, ...)` followed by a
masked mean, or `ccdproc.Combiner.sigma_clipping()` + `average_combine()`.

### 4.2 Variance and weight maps

`weight_map = np.where(mask, 0, 1.0/_variance)` (`:100-101`, `:182-183`) divides by zero wherever the
per-pixel variance is zero (dead or perfectly stable pixels), producing `inf` in the weight map with
a `RuntimeWarning`. Guard with `np.errstate` and set those pixels to zero weight (or to the
read-noise floor). Also propagate a proper variance plane through bias/dark/flat rather than
recomputing it ad hoc.

Additionally, masked pixels in the master **bias** are not replaced (unlike the flat at `:185`), so
the master bias can carry a `MaskedArray` into `fits.PrimaryHDU`.

### 4.3 Flat-field construction

- Flats are bias-subtracted (`:166`) but **never dark-subtracted** — wrong for any non-trivial flat
  exposure time.
- No rejection of over- or under-exposed flats. Add a level cut (e.g. keep frames whose median lies
  between 20 % and 70 % of saturation) before combining.
- Normalisation uses `np.median(subtracted_frame)` over the full frame including any vignetted
  corners; normalise over a central region instead.
- Dead code: `if 'slot' in filter_name.lower(): filter_name = 'CLEAR'` (`:191-192`) can never fire,
  because `valid_filters` (`:131`) already rejected any name containing "Slot" at `:149`. The slot →
  filter translation belongs in `io/headers.py` and must run *before* selection.
- `self.master_filename` is `master_flat.fits` (`:49`), but flats are written per-filter as
  `master_flat_<FILTER>.fits` (`:196-197`), so the `--clobber` guard in `main` (`:246`) never
  protects the files actually being written.

### 4.4 Frame reduction order and the product model

`process_frame` (`:949-968`) does bias subtraction then flat division and nothing else. Missing, in
order: overscan correction and trim, bias, **dark** (see phase 5), flat, bad-pixel masking, cosmic-ray
rejection, gain application to electrons, and construction of variance and mask planes.

Restructure the output as a **single multi-extension FITS file per reduced frame**, so that the image
and the catalogue extracted from it can never drift apart or be separated in an archive:

| HDU | `EXTNAME` | Type | Contents |
|---:|---|---|---|
| 0 | `SCI` (primary) | image | reduced pixels; full instrument + provenance header |
| 1 | `ERR` | image | 1σ uncertainty (or `VAR`), same shape |
| 2 | `DQ` | image, integer | bit-encoded data quality (phase 5.1) |
| 3 | `CATALOG` | BINTABLE | the photometric catalogue, **with its own header** (§4.4.1) |
| 4 | `CATFLAGS` | BINTABLE, optional | bit → meaning legend for `quality_mask` |

Nothing downstream can do correct error propagation without `ERR`, and goal 4 cannot be met without
`DQ`. The primary header carries the provenance block (pipeline version, git SHA, config hash, names
and checksums of every master used, UTC of reduction, keyword-source records from phase 1B) —
see Appendix B.

#### 4.4.1 The `CATALOG` HDU and its metrics header

The photometric catalogue is stored as an additional HDU rather than a sidecar file, and its header
carries the main photometric metrics for that frame. This makes the catalogue self-describing: the
zero point, seeing, depth, and sky under which those magnitudes were measured travel with the
magnitudes themselves, and a reader needs no database and no companion file to interpret them.

`products/catalog.py` writes it; the same values are what phase 9 ingests, so header and database
agree by construction — the ingester **reads the metrics out of this header** rather than
recomputing them.

Table columns (superset; per-frame content depends on which stages ran):

```
NUMBER  X_IMAGE  Y_IMAGE  ERRX2_IMAGE  ERRY2_IMAGE
ALPHA_J2000  DELTA_J2000  ERRA_WORLD  ERRB_WORLD
FLUX_AUTO  FLUXERR_AUTO  MAG_AUTO  MAGERR_AUTO
FLUX_APER[]  FLUXERR_APER[]  MAG_APER[]  MAGERR_APER[]
MAG_CALIB  MAGERR_CALIB          # absent when no zero point was derived
BACKGROUND  FLUX_MAX  FLUX_RADIUS  FWHM_IMAGE  FWHM_WORLD
A_IMAGE  B_IMAGE  THETA_IMAGE  ELONGATION  ELLIPTICITY
FLAGS  IMAFLAGS_ISO  NIMAFLAGS_ISO  CLASS_STAR  SPREAD_MODEL
SNR  QUALITY_MASK          # phase 6 rejection bitmask, retained not applied
GAIA_SOURCE_ID  GAIA_SEP_ARCSEC   # when a cross-match was performed
```

Header of the `CATALOG` HDU — grouped by what it records:

```
EXTNAME = 'CATALOG'
# provenance of the extraction
CATSOFT , CATSOFTV , CATCFG  , CATDATE , PIPEVER , PIPEGIT
# detection parameters actually used
DETTHRSH, DETMINAR, DETFILT , DEBLNTH , DEBLMINC, BACKSIZE, BACKFILT, WEIGHTTP
# counts
NSOURCES, NSTARS  , NCLEAN  , NREJECT , NSATUR  , NBADPIX
# photometric calibration
ZP      , ZPERR   , ZPNSTAR , ZPRMS   , ZPCAT   , ZPBAND
MAGSYS  , EXTINCT , AIRMASS , PHOTOMET
APERRAD , APERUNIT                      # aperture radii used for MAG_APER
# image quality
FWHMPIX , FWHMARC , ELLIP   , THETA
SKYLEV  , SKYRMS  , SKYUNIT
LIMMAG3 , LIMMAG5 , SNRLIM
# detector terms the magnitudes depend on
GAIN    , RDNOISE , SATLEVEL, PIXSCALE , EXPTIME , FILTER
# astrometric frame the coordinates are on
ASTSOLVR, ASTRMS1 , ASTRMS2 , ASTNREF  , ASTEPOCH, RADESYS , WCSSOLVD
```

Rules:

- A metric that could not be computed is **omitted**, never written as a placeholder — absence of
  `ZP` means instrumental magnitudes, and `MAGSYS = 'INSTRUMENTAL'` says so explicitly.
- Units are declared: `TUNIT` on every column, and `APERUNIT`/`SKYUNIT` for the scalars.
- The same `CATALOG` HDU convention is used for coadd products, where `EXPTIME` is replaced by
  `TEXPTIME` and `NCOMBINE` appears.
- `--catalog-format {mef,ldac,both}`: the default `mef` is the in-file HDU above; `ldac` additionally
  writes FITS-LDAC (`LDAC_IMHEAD` + `LDAC_OBJECTS`) for SCAMP/PSFEx compatibility, which the phase-7
  astrometric refinement wants. `robo43 catalog extract <file>` pulls the HDU out to a standalone
  FITS/Parquet/CSV for users who want it separately.
- `--no-catalog` skips extraction for bulk reprocessing, and `robo43 catalog build <file>` adds or
  replaces the HDU on an existing reduced image without redoing the pixel reduction.

### 4.5 Coaddition

`coadd_frames.py:117-122`:
```python
weights = [1/np.nanstd(d)**2 ...]
coadded_data = np.nanmedian(np.array(data_stack).T * weights, axis=2).T
```
Two problems. First, multiplying each frame by its weight and then taking the **median** is not a
weighted median and not a weighted mean — the result is the median of rescaled frames, which is
biased and not on any consistent flux scale. Second, `1/std²` per whole frame is dominated by the
astronomical sources in the field, not by the noise.

Correct approach:
1. Scale each frame to a common flux scale (by `EXPTIME`, or better by the median flux ratio of
   common stars).
2. Estimate per-frame noise from a source-masked, sigma-clipped background, not the raw `std`.
3. Combine as an inverse-variance-weighted mean with per-pixel outlier rejection (this is what
   removes cosmic rays and satellite trails); or use `reproject.mosaicking.reproject_and_coadd`.
4. Write the accompanying weight/exposure map as a second extension.

Also: sky subtraction is applied per frame at `:111-113` with a plain `np.nanmedian` of the whole
frame — for a large nebula filling the field (Eta Car, M42) that removes real signal. Use a
source-masked background (`photutils.background.Background2D` with a source mask), and make sky
subtraction opt-in per target.

The `--cutout` argument (`coadd_frames.py:22-23`) is parsed and stored (`:40`) but never used.
Either implement it via `astropy.nddata.Cutout2D` on the output WCS, or remove the flag.

### 4.6 Colour composites

`make_coloured_images.py:169` combines channel frames with `np.nansum` of reprojected data, ignoring
exposure time entirely, so a channel with more frames simply gets brighter. Scale by exposure time
(or by the coadd's own `TEXPTIME`) before summing, and prefer the phase-4.5 coadd routine over the
private one here — the duplication is itself a maintenance problem.

**Acceptance:** synthetic-data regression tests where the injected truth is known — a synthetic bias
with a known hot column, a synthetic flat with a known gradient, and a synthetic star field with a
known total flux — recovering the input to within stated tolerances.

---

## Phase 5 — Complete the missing blocks (goal 2)

### 5.1 Dark frames — currently a stub

`create_master_dark` (`make_calibration_frames.py:116-125`) is explicitly marked `# TODO`, returns a
bare `ndarray` rather than an HDU, and is unreachable anyway: `main` (`:254-256`) logs
`"Not implemented. Use bias mode instead."` and returns before the call. The dead line at `:258`
would then fail, since `master_frame, _ = <2-D array>` unpacks rows, not a tuple.

`process_robo43_frames.py` accepts `--masterdark` (`:53-54`) and stores it (`:95`) but there is **no
dark-subtraction step anywhere in the class**.

Implement:

1. **`calib/dark.py`**
   - Select frames by `OBJECT`/`IMAGETYP` == dark.
   - Group by exposure time **and** CCD set-point temperature (`CCD-TEMP`/`SET-TEMP`), within a
     configurable tolerance (e.g. ±0.5 °C).
   - Bias-subtract each frame, then sigma-clip-combine along the stack axis (phase 4.1).
   - Write both `master_dark_<exptime>s_<temp>C.fits` and a **normalised dark current map**
     (e- / s / pixel) that can be scaled to arbitrary exposure times.
   - Emit variance and mask planes alongside.

2. **Dark scaling** (the README already lists this as a TODO)
   - Preferred: exact match on exptime and temperature.
   - Fallback: scale the normalised dark-current map by `EXPTIME`, with an Arrhenius-style
     temperature correction, doubling roughly every 6–7 °C, with the coefficient calibrated from the
     actual dark library and stored in config. Record the applied scaling in the header.
   - Refuse to extrapolate beyond a configurable temperature range; warn and fall back to
     bias-only if no usable dark exists.

3. **Hot-pixel mask** (also a README TODO) — `calib/masks.py`
   - Hot pixels: pixels in the normalised dark map above `N·MAD` of the map.
   - Dead / low-QE pixels: pixels in the master flat below a fractional threshold (e.g. < 0.5).
   - Unstable pixels: high per-pixel variance across the bias stack.
   - Persistent-column detection (whole columns whose median deviates).
   - Output a single bit-encoded `DQ` mask FITS (bit 0 hot, bit 1 dead, bit 2 unstable, bit 3
     saturated-in-this-frame, bit 4 cosmic ray, bit 5 outside detector after reprojection), plus a
     human-readable summary of the affected pixel fraction.

4. **`--cal_type all`** to build bias → dark → flat in dependency order in one invocation, and
   `--cal_type auto` to build only what the night's inventory actually supports.

### 5.2 Other missing image handling

- **Overscan / bias-strip** correction: currently absent. Add a `region` config
  (`instrument.overscan`, `instrument.trim`) and apply it as the first step.
- **Cosmic-ray rejection** on single frames: wrap `astroscrappy.detect_cosmics` (L.A.Cosmic) in
  `reduce/cosmics.py`, feeding the result into the `DQ` plane. Essential for the long unstacked
  exposures used for photometry.
- **Non-linearity correction** and **shutter-timing correction**: add hook points and config
  entries even if the coefficients are initially identity, so the ROBO43 characterisation can be
  dropped in without another refactor.
- **Frame classification**: today the type is inferred by substring matching on `OBJECT`
  (`process_robo43_frames.py:122-124`, `make_calibration_frames.py:60-72`), which misclassifies any
  science target whose name happens to contain "flat" or "dark". Add a proper classifier in
  `io/headers.py` that prefers `IMAGETYP`, falls back to `OBJECT` matching, and honours the `VALID`
  keyword that `invalidate_frames.py` writes (which the reduction path currently ignores entirely).
- **Twilight vs dome flats**: distinguish and combine separately; optionally derive an illumination
  correction from the ratio.

**Acceptance:** `robo43 calib <dir> --cal-type all` produces master bias, per-(exptime, temperature)
darks, per-filter flats, and a `DQ` mask, on a night that contains all three; the reducer applies the
correct dark by exptime/temperature and records the choice in `HISTORY`.

---

## Phase 6 — Photometry quality: eliminate bad pixels from source lists (goal 4)

Today nothing in the pipeline knows which pixels are untrustworthy. `get_photometry.py:61-105` runs
SExtractor and then **throws the result away** (`sources` is assigned at `:105` and never used or
written — the function is effectively a no-op). The filtering in `process_robo43_frames.py:627-634`
is the only quality cut anywhere, and it is crude:

```python
mask = (sources['FLAGS'] < 4)
mask &= (sources['FWHM_IMAGE'] > np.percentile(sources['FWHM_IMAGE'][sources['FWHM_IMAGE']>0], 1))
mask = (sources['FLUX_AUTO'] > np.percentile(sources['FLUX_AUTO'], 80))   # keeps only the top 20 %
```

`FLAGS < 4` admits flag 1 (blended/crowded) and 2 (deblended); the 80th-percentile flux cut is
data-dependent and throws away 80 % of the field regardless of quality.

Additionally `WEIGHT_TYPE: "BACKGROUND"` is set together with `WEIGHT_IMAGE` pointing at
`master_bias_weight.fits` (`:598-599`) — with `BACKGROUND`, SExtractor **ignores** `WEIGHT_IMAGE`
entirely, so the weight map that `make_calibration_frames.py` carefully builds is never used. The
path is also wrong: it resolves against the repo root (`self.files_path`), not the working directory
where the weight map is actually written.

### Plan

1. **Feed the DQ mask to the detector.** Set `WEIGHT_TYPE = MAP_WEIGHT` with the real per-frame
   inverse-variance map, and `FLAG_IMAGE` = the phase-5.1 `DQ` mask with
   `FLAG_TYPE = OR` so every catalogue row carries an `IMAFLAGS_ISO` column recording which bad-pixel
   classes touched that source's isophote. Add `IMAFLAGS_ISO` and `NIMAFLAGS_ISO` to `out_params`.
2. **`photometry/quality.py` — an explicit, configurable rejection chain**, each step logged with a
   count so the attrition is auditable:
   - `FLAGS == 0` for astrometry/zero-point reference sources; `FLAGS <= 3` for science catalogues
     with the flag value retained.
   - `IMAFLAGS_ISO == 0` — no hot, dead, unstable, cosmic-ray, or saturated pixel inside the source.
   - Saturation: peak (`FLUX_MAX`) below a configurable fraction of the true saturation level; and
     non-linearity guard below a second, lower threshold.
   - Edge rejection: centroid at least `max(3·FWHM, 20 px)` from any border, and beyond the
     vignetted radius when applicable.
   - Absolute FWHM window in arcsec (from config), not a percentile — reject both cosmic-ray-like
     PSFs narrower than the seeing and extended/blended sources much wider.
   - Ellipticity cut (`A_IMAGE/B_IMAGE`) to reject trailed sources during tracking excursions.
   - `SNR = FLUX_AUTO/FLUXERR_AUTO` above a configurable floor (e.g. 5), replacing the arbitrary
     80th-percentile flux cut.
   - `CLASS_STAR > 0.8` (or, better, `SPREAD_MODEL` if a PSF model is available) when a *stellar*
     sample is wanted — the existing `CLASS_STAR` cut is commented out at `:630`.
   - Nearest-neighbour isolation radius, to drop blends before they contaminate aperture photometry.
   - Cross-match against Gaia DR3 and drop sources with no counterpart when a clean reference sample
     is needed (also gives a spurious-detection rate as a free quality metric).
3. **Actually persist catalogues.** Write the catalogue into the reduced frame's `CATALOG` HDU
   (§4.4.1), with the rejection bitmask retained as a `QUALITY_MASK` column rather than applied
   destructively, so cuts can be revisited without re-running SExtractor. Optionally also emit
   FITS-LDAC for SCAMP/PSFEx. Every metric the quality chain computes — counts before and after each
   cut, seeing, depth, zero point — is written to that HDU's header, which is the single source the
   database ingester reads in phase 9.
4. **Aperture photometry fixes** in `get_relative_photometry.py`:
   - **Double background subtraction** (`:91-93` then `:130-135`): `Background2D` is subtracted from
     the image, and then the annulus background is measured on the *already subtracted* image and
     subtracted again. Pick one; if both, measure the annulus on the original data.
   - `bkg_mean = annulus_stats['aperture_sum']/annuli.area` (`:133`) is a plain mean, so any star
     inside the annulus biases it. Use `photutils.aperture.ApertureStats(...).median` with sigma
     clipping.
   - `error=np.sqrt(np.abs(data))` (`:125-126`) assumes gain = 1 e-/ADU and zero read noise; the
     configured gain is 10. Use `photutils.utils.calc_total_error` with the real gain and the
     `ERR` plane from phase 4.4, and propagate the annulus uncertainty into `flux_err` (currently
     `flux_err` at `:139` ignores the background term entirely).
   - Reject any measurement whose aperture overlaps a `DQ`-flagged pixel, or whose peak is
     saturated — the current code will happily report a magnitude for a saturated star.
   - The reference star (`refstar`) is measured (`:60`, `:142`) but never used in the ratio at
     `:145` — either use it as a second comparison and check the two agree, or drop the argument.
   - Add airmass/extinction correction and, optionally, a colour term.
   - Replace the hard-coded comparison-star magnitudes (`:150`) with a per-target config or a DB
     table (phase 8).
5. **Absolute zero points** — `photometry/zeropoint.py`: cross-match the cleaned catalogue against
   Gaia DR3 synthetic photometry or APASS, fit a sigma-clipped zero point per frame per filter with
   an airmass term, and record `ZP`, `ZPERR`, `ZPNSTAR`, and the extinction coefficient in the header
   and the database. Flag frames whose ZP scatter exceeds a threshold as non-photometric — this
   doubles as a cloud detector.
6. **Per-frame quality metrics** written to the header and the DB: median FWHM, ellipticity, sky
   level, sky RMS, limiting magnitude, source count, ZP scatter. These drive the automatic frame
   rejection that goals 6 and 7 need.

**Acceptance:** on a frame with a synthetic hot column and injected cosmic rays, zero surviving
catalogue rows overlap flagged pixels; the differential light curve's night-to-night scatter drops
measurably against the current output on the same input data.

---

## Phase 7 — Astrometry: faster and more accurate (goal 5)

### What is wrong now

`solver_astrometry` (`:657-850`) is two nested `while` loops that lower `sigma_clip` by 0.5 (or 1.0
at `:778`) after each failure, re-running full source detection each time; the loop is driven partly
by real failures and partly by the `TypeError`s described in 2.3. Both loops call
`run_astrometry_solver1`, which rebuilds the index solver from disk on every call (3.1). A hard frame
therefore reloads the index set up to ten times before giving up.

`run_astrometry_solver2` — the astrometry.net web fallback (`:446-557`) — is fully written but
**never called from anywhere**. Its retry counter `i` only increments on `TimeoutError`, so any other
exception exits after one attempt.

Detection quality is the deeper problem: `DAOStarFinder(threshold=sigma*np.std(data))` (`:561-563`)
uses the **standard deviation of the whole image**, which in a crowded or nebulous field is set by
the sources themselves, not the noise — so the threshold is wildly wrong exactly where solving is
hard. `sources` may also be `None` when nothing is found, and `np.argsort(sources['flux'])` at `:565`
then raises.

Finally, the solution is used as-is: `hdul[0].header.update(wcs_header.to_header())` (`:799`) writes
a plain TAN WCS with no distortion terms and no measure of its own accuracy. Nothing ever checks the
residuals.

### Plan

1. **Persistent solver, tight hints.**
   - Build the `astrometry.Solver` once per process (3.1).
   - Derive `SizeHint` from the configured plate scale ±3 % instead of the current 0.2–1.0 arcsec/px
     range (`:415-416`), which is a factor-5 window when the true scale is known to be 0.529.
   - Use the telescope pointing from the header for `PositionHint` with a radius matched to the
     mount's real pointing error (start at 0.5°, not 2.0° as at `:420`), falling back to a wider
     search only on failure.
   - Restrict index scales to those matching the actual field of view rather than the hard-coded
     `{3,4,5,6}` (`:399`); compute them from `NAXIS1/2 × pixel_scale`.
   - Add a solve timeout and a hard per-frame budget.
2. **Better source lists for solving.**
   - Replace `np.std` with `astropy.stats.sigma_clipped_stats` (or `photutils.Background2D` +
     `detect_threshold`) so the threshold tracks the background noise.
   - Mask bad pixels and cosmic rays (phase 5/6) *before* detection — bad columns are a classic cause
     of failed solves.
   - Select 50–150 sources spread across the field (grid-sample rather than pure brightest-first,
     which biases toward one corner in a nebula field) and reject saturated ones.
   - Replace the sigma-decrement retry ladder with a single detection pass at an adaptive threshold
     plus at most two retries at pre-computed levels.
3. **Blind-solve reuse across a night.** Consecutive frames of the same target share a pointing.
   Solve one frame per (target, dither-group), then seed the rest from that solution and refine —
   turning most of a night's frames from blind solves into a fast local fit. Cache successful
   solutions keyed by `(OBJECT, filter, rounded pointing)` and persist the cache to disk (and to the
   DB in phase 8).
4. **Accuracy: refine against Gaia.**
   - After the initial solve, cross-match the cleaned detection catalogue (phase 6) against Gaia DR3
     (proper-motion-propagated to the epoch of observation) and fit a **SIP** distortion solution with
     `astropy.wcs.utils.fit_wcs_from_points`, or run SCAMP if a heavier solution is warranted.
   - Iterate with sigma-clipped rejection of mismatches.
   - Record the quality of the fit in the header: `ASTRMS1`/`ASTRMS2` (RMS residuals in arcsec),
     `ASTNREF` (number of reference stars), `ASTSOLVR` (which solver succeeded), `ASTEPOCH`. Nothing
     downstream can currently distinguish a good solve from a marginal one, and `coadd_frames.py`
     only checks `wcs.has_celestial` (`:57`, `:84`, `:101`) — which is true for any solution, however
     bad.
   - Define a pass/fail threshold on the RMS and mark frames that fail as unsolved rather than
     silently coadding them.
5. **Ordered fallback chain**, each step recorded in the status table: cached/seeded solution →
   local index solve → local `solve-field` binary → astrometry.net web (`run_astrometry_solver2`,
   finally wired in and given a proper backoff) → give up and flag.
6. **Fix the RA/Dec plumbing.** `radec_to_degrees` (`:276-303`) guesses the format by counting
   colons/spaces and its result is discarded in favour of writing SIMBAD's raw string back into the
   header (`:348-351`); `run_astrometry_solver1` then re-parses it with `':' in ...` (`:391`), which
   raises `TypeError` on a numeric card. Normalise once, on ingest, into `RA_DEG`/`DEC_DEG` float
   keywords, and have everything downstream read only those.
7. **Cache SIMBAD lookups.** `guess_ra_dec` (`:346`) hits SIMBAD per frame; a night of 200 frames of
   the same target makes 200 identical queries. Cache by object name for the run and persist to the
   DB. Also note the column-name fragility: `result['ra']` / `result['dec']` (`:348-351`) are the
   modern `astroquery` names; older versions return `RA`/`DEC`. Pin `astroquery` and normalise.

**Acceptance:** median solve time per frame reduced by ≥5× on a benchmark night; ≥95 % solve rate on
frames that currently solve; astrometric RMS recorded for every frame and below 0.3 arcsec (roughly
0.5 px) for solved frames.

---

## Phase 8 — Service mode (goal 6)

Keep the user-run CLI exactly as it is — the interactive, per-directory workflow stays the primary
interface. Add a long-running mode on top of the same code paths, so there is one implementation and
two front ends.

### 8.1 Architecture

```
                ┌────────────────┐
  new files ──▶ │  Watcher       │──┐
  (inotify)     └────────────────┘  │
                                    ├──▶  Job queue  ──▶  Worker pool  ──▶  Products + DB
  DB poll   ──▶ ┌────────────────┐  │      (DB table       (N procs,          (phase 9)
  (phase 9)     │  Scheduler     │──┘    or Redis/RQ)     reuses reduce/)
                └────────────────┘
```

- **Watcher** (`service/watcher.py`): `watchdog` observer on one or more incoming directories, with
  a settling delay so partially written files are not picked up, plus a periodic full rescan to catch
  anything inotify missed.
- **Scheduler** (`service/daemon.py`): decides what is runnable now — a science frame needs its
  masters; masters need a complete calibration set; a coadd needs its members solved. Model this as
  an explicit dependency graph with states `PENDING → READY → RUNNING → DONE | FAILED | SKIPPED`.
- **Queue**: start with a `jobs` table in PostgreSQL using `SELECT ... FOR UPDATE SKIP LOCKED`
  (no extra infrastructure, transactional with the data, and it is the same database goal 7 needs).
  Keep the interface narrow enough to swap in Redis/RQ or Celery later if throughput demands it.
- **Workers** (`service/worker.py`): call exactly the same functions as the CLI. A job carries an
  attempt count, a `last_error`, and an exponential backoff; poison jobs land in a dead-letter state
  after N attempts instead of retrying forever.
- **Idempotency**: every job is keyed on `(input checksum, pipeline version, config hash)`. Re-running
  is safe and cheap — if the key matches an existing product, the job short-circuits. This is what
  makes the service restartable at any point.

### 8.2 Operational requirements

- `robo43 serve --config /etc/robo43/config.toml` as the entry point; `--once` runs a single drain
  pass and exits (useful under cron and in tests).
- Graceful shutdown on `SIGTERM`: stop accepting jobs, let in-flight frames finish, release DB locks.
- A `systemd` unit file plus a `Dockerfile`/compose file, both shipped in the repo.
- Health and metrics: a `/healthz` and `/metrics` HTTP endpoint (Prometheus text format) exposing
  queue depth, jobs/hour, failure rate, median solve time, and last-successful-run timestamp.
- Structured JSON logging in service mode, human-readable in CLI mode.
- Single-instance locking (advisory lock in PostgreSQL) so two daemons cannot process the same night.
- A `robo43 status` subcommand to inspect the queue, and `robo43 requeue --night ... --stage ...` to
  re-run a stage after a fix.
- Nightly summary report (per-night HTML/PDF: frame counts, solve rate, seeing trend, ZP trend,
  rejected frames with reasons), optionally emailed.

**Acceptance:** dropping a night's raw files into the watched directory produces the full set of
reduced products with no human intervention; killing the daemon mid-night and restarting resumes
without duplicating or losing work.

---

## Phase 9 — PostgreSQL query and ingest (goal 7)

The pipeline should be able to ask the database what to reduce, reduce it, and write the products
back into the tables that describe them.

### 9.1 Schema sketch

Two schemas: `raw` (populated by the acquisition system, read-mostly for us) and `reduced` (written
by this pipeline).

```sql
-- the instrument registry: the bridge to the phase-1B profiles
raw.instruments(
  id serial PK, name text UNIQUE, telescope text, instrument text,
  profile_toml text,                       -- the phase-1B profile, versioned here
  pixel_scale_arcsec real, gain_e_per_adu real, read_noise_e real,
  saturation_adu real, site_lat real, site_lon real, site_elev real)

-- raw side (existing or to be agreed with the acquisition system).
-- The NOT NULL constraints ARE the keyword contract of phase 1B: because the data
-- model enforces them, anything reachable through this table is reducible without
-- user-supplied overrides.
raw.images(
  id bigserial PK, file_path text, checksum_sha256 text UNIQUE,
  instrument_id int NOT NULL REFERENCES raw.instruments,
  object text, imagetyp text NOT NULL, filter text, exptime real NOT NULL,
  ra_deg double precision, dec_deg double precision,
  date_obs timestamptz NOT NULL, mjd double precision, night date NOT NULL,
  naxis1 int NOT NULL, naxis2 int NOT NULL,
  ccd_temp real, airmass real, binning smallint,
  is_valid boolean DEFAULT true, ingested_at timestamptz,
  CONSTRAINT imagetyp_known CHECK (imagetyp IN ('bias','dark','flat','science')),
  CONSTRAINT filter_when_needed CHECK (imagetyp IN ('bias','dark') OR filter IS NOT NULL),
  CONSTRAINT pointing_when_science CHECK (imagetyp <> 'science'
                                          OR (ra_deg IS NOT NULL AND dec_deg IS NOT NULL)))

-- reduced side
reduced.master_calib(
  id bigserial PK, cal_type text CHECK (cal_type IN ('bias','dark','flat','mask')),
  filter text NULL, exptime real NULL, ccd_temp real NULL, night date,
  file_path text, checksum_sha256 text, n_combine int,
  pipeline_version text, config_hash text, created_at timestamptz,
  UNIQUE (cal_type, filter, exptime, ccd_temp, night, config_hash))

reduced.images(
  id bigserial PK, raw_image_id bigint REFERENCES raw.images,
  instrument_id int REFERENCES raw.instruments,
  file_path text, checksum_sha256 text,
  metadata_provenance jsonb,     -- per-keyword: header | alias | override | profile | derived
  master_bias_id bigint REFERENCES reduced.master_calib,
  master_dark_id bigint REFERENCES reduced.master_calib,
  master_flat_id bigint REFERENCES reduced.master_calib,
  mask_id        bigint REFERENCES reduced.master_calib,
  proc_status text, proc_code int, wcs_solved boolean,
  astrom_rms_arcsec real, astrom_nref int, astrom_solver text,
  fwhm_arcsec real, ellipticity real, sky_level real, sky_rms real,
  zeropoint real, zeropoint_err real, zeropoint_nstar int,
  limiting_mag real, is_photometric boolean,
  pipeline_version text, config_hash text, created_at timestamptz,
  UNIQUE (raw_image_id, config_hash))

reduced.coadds(
  id bigserial PK, object text, filter text, night date,
  file_path text, checksum_sha256 text,
  n_combine int, total_exptime real, ra_deg, dec_deg, fov_arcmin real,
  pipeline_version text, config_hash text, created_at timestamptz)

reduced.coadd_members(coadd_id bigint, reduced_image_id bigint, weight real,
  PRIMARY KEY (coadd_id, reduced_image_id))

reduced.source_catalog(          -- partitioned by night
  id bigserial, reduced_image_id bigint REFERENCES reduced.images,
  number int, x_image real, y_image real,
  ra_deg double precision, dec_deg double precision,
  flux_auto real, fluxerr_auto real, mag_auto real, magerr_auto real,
  mag_calib real, magerr_calib real,
  fwhm_image real, elongation real, flags smallint, imaflags_iso int,
  class_star real, quality_mask int)

reduced.photometry(              -- differential / time-series results
  id bigserial PK, reduced_image_id bigint, target text,
  jd double precision, filter text, mag real, mag_err real,
  comp_star text, aperture_arcsec real, airmass real, method text)

reduced.jobs(                    -- the phase-8 queue
  id bigserial PK, job_type text, payload jsonb, state text,
  priority int, attempts int, last_error text,
  locked_by text, locked_at timestamptz, created_at, updated_at)
```

Indexes on `(night)`, `(object, filter)`, `checksum_sha256`, and a `q3c` or PostGIS spatial index on
`(ra_deg, dec_deg)` for cone searches over `source_catalog`.

### 9.2 Implementation

- **SQLAlchemy 2.0 ORM** in `db/models.py`, **Alembic** migrations from the first commit so the
  schema is versioned like the code. Connect with `psycopg[binary]`.
- **`db/query.py`** — the selection layer that replaces today's `glob` + repeated `fits.getheader`:
  `robo43 db list --night 2026-09-05 --object "ETA CAR" --filter V --state pending`, returning the
  same in-memory inventory structure that the filesystem path produces, so the reduction code does
  not care where its work list came from. Provide `--dry-run` everywhere.
- **Instrument round trip.** `raw.instruments.profile_toml` stores the phase-1B profile, so a
  database-driven run needs no local profile files: `db list` returns the instrument with the work
  list, and the reducer loads the profile from the row. `robo43 db register-instrument <profile.toml>`
  pushes a local profile into the registry; `robo43 db export-instrument <name>` pulls one back out.
  This is what makes "the keywords will be compatible" true in practice rather than by convention.
- **Contract validation at the boundary.** `db/query.py` runs the same `io/validate.py` check that
  the manual path runs, against the DB row rather than the header. It should always pass — the
  constraints above guarantee it — so a failure means the acquisition system's data model and the
  pipeline's contract have diverged, and it is reported as a schema problem rather than a per-frame
  one. Rows that fail land in a `raw.quarantine` view for a human to look at, and never enter the
  job queue.
- **`db/ingest.py`** — writes products back. Rules:
  - One transaction per logical product; never leave a row pointing at a half-written file. Write to
    a temporary path, `fsync`, atomically rename, then commit the row.
  - Upsert on the natural keys (`ON CONFLICT ... DO UPDATE`) so re-ingest is idempotent.
  - Record `pipeline_version`, git SHA, and `config_hash` on every row, so products from different
    pipeline versions are distinguishable and reproducible.
  - Read the per-frame quality and photometric metrics **out of the `CATALOG` HDU header** (§4.4.1)
    rather than recomputing them, so the file and the database can never disagree; the header is the
    source of truth and `reduced.images` is its index.
  - Bulk-load `source_catalog` straight from the `CATALOG` BINTABLE with `COPY` (psycopg's
    `cursor.copy()`), not row-by-row inserts — a single frame can contain tens of thousands of
    sources.
  - Store file paths relative to a configured data root, never absolute, so the archive can move.
- **Credentials**: never in the repo. Read from `ROBO43_DB_URL` or a `~/.pgpass`-style file with a
  permissions check; support read-only and read-write roles so `db list` can run with reduced
  privileges. Document the required grants.
- **Connection handling**: pooled engine, `pool_pre_ping=True`, retry with backoff on transient
  disconnects, and a statement timeout so a stuck query cannot wedge the daemon.
- **Filesystem fallback**: the pipeline must remain fully usable with no database (`--no-db`), which
  is how it works today; the DB layer is an optional dependency (`pip install robo43[db]`).
- **Backfill tool**: `robo43 db backfill <dir>` to register an existing archive of already-reduced
  products, so the database can be adopted without re-reducing everything.

**Acceptance:** `robo43 db list --night X --state pending | robo43 reduce --from-db` reduces exactly
the frames the DB reports as pending and, on completion, those rows are marked done with product
paths, checksums, and quality metrics populated; re-running the same command is a no-op.

---

## Phase 10 — Tests, CI, and documentation (built incrementally, not last)

- **Fixtures**: a synthetic ROBO43-like data generator (`tests/fixtures/make_synthetic.py`) producing
  a small night — bias, darks at two exposure times, flats in three filters, and science frames with
  a known star field, injected hot pixels, and injected cosmic rays. Small enough to commit
  (e.g. 256×256), so tests need no external data.
- **A second synthetic instrument** with different geometry, keyword spellings (`EXPOSURE`,
  `IMGTYPE`, `OBJCTRA`), and a multi-extension layout, plus a third variant with required keywords
  deliberately stripped — these are the fixtures that keep phase 1B honest and stop ROBO43
  assumptions creeping back in.
- **Unit tests** for keyword aliasing and `FrameMeta` resolution, override precedence, RA/Dec
  normalisation, filter mapping, dark scaling, stack combination statistics, quality cuts, coadd
  weighting, and `CATALOG` HDU round-tripping (write → read → ingest → read back, values identical).
- **Regression tests** with numerical tolerances on recovered flux, recovered zero point, and
  astrometric RMS from the synthetic field.
- **Integration test**: full CLI run end-to-end on the synthetic night, plus a `--once` service run
  against a `testcontainers`-launched PostgreSQL.
- **CI** (GitHub Actions): ruff, mypy, pytest with coverage, a build of the wheel, and an install
  smoke test in a clean container. Cache the SExtractor binary install.
- **Docs**: a real README (installation, quickstart, the CLI subcommands), a configuration reference
  generated from the dataclasses, **a keyword-contract page listing the required/optional keywords
  per tier and how to write an instrument profile**, a product-format page documenting the
  `SCI`/`ERR`/`DQ`/`CATALOG` layout and every `CATALOG` header keyword, a data-model page for the DB
  schema, an operations runbook for service mode, and an `ARCHITECTURE.md`. Update the README TODO list — "dark scaling"
  and "hot pixel mask" are addressed by phase 5, "improve astrometric calibration" by phase 7.
- **`CHANGELOG.md`** and semantic versioning, with `__version__` written into every product header.

---

## Suggested ordering

The dependency chain is real: packaging and config unblock everything; the DQ mask from phase 5 is a
prerequisite for the photometry cleanup in phase 6, which is in turn what makes the Gaia-based
astrometric refinement in phase 7 trustworthy; and both service mode and the database need the
idempotency and status-record work from phase 3.

| Order | Phase | Depends on | Rough size |
|---|---|---|---|
| 1 | 1 — package + config | — | M |
| 2 | 1B — instrument abstraction + keyword contract | 1 | M |
| 3 | 2 — hygiene / bug fixes | 1 | M |
| 4 | 3 — performance | 1, 2 | M |
| 5 | 4 — reduction correctness + product model | 1B, 2 | M |
| 6 | 5 — dark, masks, missing blocks | 4 | L |
| 7 | 6 — photometry quality + `CATALOG` HDU | 4, 5 | L |
| 8 | 7 — astrometry | 3, 6 | L |
| 9 | 9 — database (schema + query/ingest) | 1B, 3, 6 | L |
| 10 | 8 — service mode | 3, 9 | M |
| 11 | 10 — tests/CI/docs | continuous | M |

Phase 1B moves early deliberately: `FrameMeta` is the type every later module reads, and retrofitting
it after phases 4–7 are written would mean touching all of them twice. It also pays off immediately —
the keyword-alias layer subsumes `fix_header.py` and most of the ad-hoc header handling that phase 2
would otherwise have to clean up in place.

Phase 10 is listed last only because it never ends; the fixtures from it should be written during
phase 2, before any of the science-changing work in phases 4–7 starts, so those changes can be
verified rather than hoped at.

---

## Appendix A — Consolidated bug list

Numbered for tracking. Severity: **A** = wrong results or hard failure, **B** = fragile/incorrect
under some inputs, **C** = hygiene.

| # | Sev | Location | Issue |
|---|---|---|---|
| 1 | A | `process_robo43_frames.py:984` | unconditional `pdb.set_trace()` on the main path |
| 2 | A | `process_robo43_frames.py:1022`, `:898` | `proc_status` lost under `multiprocessing`; status CSV empty |
| 3 | A | `process_robo43_frames.py:391` | `_ra_deg` unbound / `TypeError` on numeric RA |
| 4 | A | `process_robo43_frames.py:798` | `solver_used` may be unbound |
| 5 | A | `process_robo43_frames.py:981` | `guess_ra_dec` needs `--save_processed` to exist |
| 6 | A | `process_robo43_frames.py` (all) | no dark subtraction despite `--masterdark` |
| 7 | A | `make_calibration_frames.py:116` | `create_master_dark` is a stub and unreachable |
| 8 | A | `make_calibration_frames.py:96`, `:178` | sigma clipping applied spatially, not along the stack |
| 9 | A | `coadd_frames.py:93-98` | wrong reference frame seeded; reference stacked twice |
| 10 | A | `coadd_frames.py:121` | "weighted median" is neither weighted nor a valid combine |
| 11 | A | `get_relative_photometry.py:91`,`:130` | background subtracted twice |
| 12 | A | `get_photometry.py:105` | catalogue computed and discarded; module is a no-op |
| 13 | A | `make_coloured_images.py:222` | `self.logger` undefined → `--cutout` always fails |
| 14 | A | `make_coloured_images.py:361` | `output_path` undefined → `--verbose` fails at the end |
| 15 | B | `process_robo43_frames.py:398` | astrometry index cache path hard-coded to one machine |
| 16 | B | `process_robo43_frames.py:396` | solver rebuilt from disk on every attempt |
| 17 | B | `process_robo43_frames.py:562` | detection threshold from whole-image `np.std` |
| 18 | B | `process_robo43_frames.py:598` | `WEIGHT_TYPE=BACKGROUND` silently ignores `WEIGHT_IMAGE`; wrong path |
| 19 | B | `process_robo43_frames.py:606` vs `get_photometry.py:89` | saturation level disagrees (200 000 vs 55 000) |
| 20 | B | `process_robo43_frames.py:632` | 80th-percentile flux cut discards 80 % of sources |
| 21 | B | `process_robo43_frames.py:579`,`:655` | functions return `Exception` objects as data |
| 22 | B | `process_robo43_frames.py:173` | `sys.exit(1)` inside library code |
| 23 | B | `process_robo43_frames.py:139`,`:176` | header re-read once per exclusion keyword per file |
| 24 | B | `process_robo43_frames.py:202`,`:250` | masters re-read per science frame |
| 25 | B | `process_robo43_frames.py:324` | `HDUList` opened `mode='update'` and never closed |
| 26 | B | `process_robo43_frames.py:446` | web astrometry fallback written but never called |
| 27 | B | `process_robo43_frames.py:25` | `--loglevel` inert unless `--verbose` also given |
| 28 | B | `make_calibration_frames.py:100` | divide-by-zero in weight map |
| 29 | B | `make_calibration_frames.py:166` | flats not dark-subtracted |
| 30 | B | `make_calibration_frames.py:191` | dead `'slot'` branch after `valid_filters` filter |
| 31 | B | `make_calibration_frames.py:49`,`:246` | flat `--clobber` guard checks the wrong filename |
| 32 | B | `coadd_frames.py:181` | reads global `args`, then `overwrite=True` regardless |
| 33 | B | `coadd_frames.py:58` | `AttributeError` when `OBJECT` is absent |
| 34 | B | `coadd_frames.py:101` | `TEXPTIME` misses the seed frame |
| 35 | B | `coadd_frames.py:22`,`:40` | `--cutout` parsed but never used |
| 36 | B | `coadd_frames.py:111` | whole-frame median sky subtraction removes extended emission |
| 37 | B | `make_coloured_images.py:192` | `--degrade_psf` computes FWHM then discards it |
| 38 | B | `make_coloured_images.py:208` | `fit_shape` set to the whole image |
| 39 | B | `make_coloured_images.py:169` | channels summed without exposure-time scaling |
| 40 | B | `get_relative_photometry.py:117` | `.strip('.fits')` mangles filenames |
| 41 | B | `get_relative_photometry.py:133` | annulus background is a mean, not a clipped median |
| 42 | B | `get_relative_photometry.py:125` | error model assumes gain 1, ignores read noise |
| 43 | B | `get_relative_photometry.py:142` | reference star measured but unused |
| 44 | B | `get_relative_photometry.py:231` | unguarded `pdb.set_trace()` |
| 45 | B | `calculate_average_mags.py:84` | dead JD-matching comparison |
| 46 | B | `calculate_average_mags.py:88` | `jd` leaks across filters |
| 47 | B | `calculate_average_mags.py:60` | ragged output dict when a CSV is missing |
| 48 | B | `get_photometry.py:56` | produces `_proc_proc.fits` names |
| 49 | B | `get_photometry.py:66` | module-level `logger` undefined on import |
| 50 | B | `fix_header.py:33` | ignores its `imagename` argument; `KeyError` without `FILTER` |
| 51 | B | `collect_night_info.py:23` | fixed-width dtypes truncate silently |
| 52 | B | `process_robo43_frames.py:122` | frame type inferred by substring on `OBJECT` |
| 53 | B | everywhere | `VALID` keyword written by `invalidate_frames.py` never read by the reducer |
| 53a | B | everywhere | science data assumed at `hdul[0]`; no multi-extension or multi-amplifier support |
| 53b | B | `:148`, `make_calibration_frames.py:54`, `coadd_frames.py:54` | only `*.fits` matched; `.fit`, `.fts`, `.fz`, `.gz` ignored |
| 53c | B | `fix_header.py:22`, `make_calibration_frames.py:131` | filter names and validity hard-coded to ROBO43's filter wheel |
| 53d | B | `process_robo43_frames.py:116` | ROBO43 target aliases hard-coded in the reducer |
| 53e | B | `process_robo43_frames.py:240`, `:223` | missing calibration frames abort instead of degrading gracefully |
| 53f | B | all photometry modules | catalogues, where written at all, are detached from the image they came from |
| 54 | C | all modules | `#!/bin/python3` shebang |
| 55 | C | `process_robo43_frames.py:23`, `get_photometry.py:10` | duplicated `setup_logging` |
| 56 | C | `process_robo43_frames.py:587`, `get_photometry.py:71` | duplicated SExtractor config |
| 57 | C | `requirements.txt` | missing pandas, matplotlib, scipy, reproject, sewpy, astrometry; unpinned |
| 58 | C | `process_robo43_frames.py:49` | `--output_file` is actually an output *directory* |
| 59 | C | `process_robo43_frames.py:992` | one status CSV written per frame |
| 60 | C | `process_robo43_frames.py:218` | `raise e from e` |

## Appendix B — Header keywords the pipeline should write

### B.1 Primary (`SCI`) header — provenance and calibration

On every reduced product, so that nothing downstream (including the database) has to guess:

```
PIPEVER   pipeline version            BIASFIL   master bias used ('NONE' if skipped)
PIPEGIT   git SHA                     DARKFIL   master dark used
PIPECFG   config hash                 DARKSCL   dark scaling factor applied
PIPEDATE  UTC of reduction            FLATFIL   master flat used
INSTPROF  instrument profile name     MASKFIL   bad-pixel mask used
INSTPVER  instrument profile version  CRREJ     T/F cosmic-ray rejection applied
                                      OSCANCOR  T/F overscan corrected
ASTSOLVR  solver that succeeded       NCOMBINE  frames combined (coadds)
ASTRMS1/2 astrometric RMS (arcsec)    TEXPTIME  total exposure time (coadds)
ASTNREF   reference stars used        WCSSOLVD  T/F astrometry solved
ASTEPOCH  reference catalogue epoch
```

Keyword-provenance records from phase 1B, one per canonical field that did **not** come straight
from the raw header — so an asserted gain is never mistaken for a measured one:

```
KWSRC1  = 'gain=override:overrides.toml'
KWSRC2  = 'pixel_scale=profile:robo43'
KWSRC3  = 'airmass=derived:radec+dateobs+site'
```

### B.2 `CATALOG` HDU header — photometric metrics

The catalogue is self-describing: the conditions under which its magnitudes were measured travel in
its own header (§4.4.1). Grouped: extraction provenance (`CATSOFT`, `CATSOFTV`, `CATCFG`, `CATDATE`),
detection parameters (`DETTHRSH`, `DETMINAR`, `DETFILT`, `DEBLNTH`, `DEBLMINC`, `BACKSIZE`,
`BACKFILT`, `WEIGHTTP`), counts (`NSOURCES`, `NSTARS`, `NCLEAN`, `NREJECT`, `NSATUR`, `NBADPIX`),
photometric calibration (`ZP`, `ZPERR`, `ZPNSTAR`, `ZPRMS`, `ZPCAT`, `ZPBAND`, `MAGSYS`, `EXTINCT`,
`AIRMASS`, `PHOTOMET`, `APERRAD`, `APERUNIT`), image quality (`FWHMPIX`, `FWHMARC`, `ELLIP`, `THETA`,
`SKYLEV`, `SKYRMS`, `SKYUNIT`, `LIMMAG3`, `LIMMAG5`, `SNRLIM`), detector terms the magnitudes depend
on (`GAIN`, `RDNOISE`, `SATLEVEL`, `PIXSCALE`, `EXPTIME`, `FILTER`), and the astrometric frame the
coordinates sit on (`ASTSOLVR`, `ASTRMS1/2`, `ASTNREF`, `ASTEPOCH`, `RADESYS`, `WCSSOLVD`).

A metric that could not be computed is omitted rather than filled with a sentinel; `MAGSYS =
'INSTRUMENTAL'` is how a catalogue says it has no zero point.
