# Phase 1B resume note — inventory + CLI subcommands

## Unit

Phase 1B: `src/robo43/io/inventory.py` plus the `inventory`, `inspect` and
`fixheader` CLI subcommands (`src/robo43/cli/inventory.py`,
`src/robo43/cli/inspect.py`, `src/robo43/cli/fixheader.py`), plus
`tests/conftest.py` and the test files listed in the brief
(`tests/test_inventory.py`, `tests/test_cli_inspect.py`,
`tests/test_headers.py`, `tests/test_meta.py`, `tests/test_validate.py`,
`tests/test_fits_io.py`, `tests/test_profile.py`).

Contract: `FOUNDATION_CONTRACT.md` (a copy was placed in the scratchpad for
this session; read the `io/inventory.py` and CLI sections, ADDENDUM 2, and
Rule A in ADDENDUM 6).

## Status at checkpoint

**No implementation work has been done yet.** This session was stopped
mid-investigation, before any executer agent was spawned and before any file
was written. `git status --short` shows no changes attributable to this
unit — the only local diffs are from other agents/phases already in flight
(`requirements.txt`, `src/coadd_frames.py`, the `data/*` -> `src/robo43/data/*`
renames, and a long list of untracked `src/robo43/**` files from other
phases). Do not attribute any of that to this unit.

## Files complete and passing

None yet.

## Files started but incomplete

None. No file under `src/robo43/io/inventory.py`,
`src/robo43/cli/inventory.py`, `src/robo43/cli/inspect.py`,
`src/robo43/cli/fixheader.py`, `src/robo43/data/instruments/synth2.toml`,
`tests/conftest.py`, `tests/test_inventory.py`, `tests/test_cli_inspect.py`,
`tests/test_headers.py`, `tests/test_meta.py`, `tests/test_validate.py`,
`tests/test_fits_io.py`, or `tests/test_profile.py` exists on disk.

## Files not started

All of the above. Everything in this unit remains to be written.

## Test status

Whole-suite baseline at checkpoint (files owned by other phases):
`python3 -m pytest -q`
→ **123 passed** (see verbatim output below). Collection is clean
(`pytest --collect-only -q` lists 11 test files, 123 items, no errors).
Note: mid-session, one interim run transiently showed two failures in
`tests/test_quality.py` (not owned by this unit) while another agent was
actively editing that phase's source concurrently; a re-run seconds later
was green. Treat any `test_quality.py`/`test_mef.py`/`test_catalog.py`/
`test_masks.py` failures as belonging to other phases, not this one, and
re-run before concluding anything is broken.

## Decisions made (binding — do not silently reverse)

These were worked out carefully this session via direct experimentation
against the synthetic fixtures (`tests/fixtures/make_synthetic.py`) and the
already-landed `io/meta.py`, `io/headers.py`, `io/fits_io.py`,
`io/validate.py`, `instruments/profile.py`. A successor should read this
before redesigning anything.

### 1. `INVENTORY_COLUMNS` includes a `read_error` column

The contract text lists `INVENTORY_COLUMNS` as "path + every FrameMeta scalar
field + `origins_json`", but the prose requirement immediately below it says
"An unreadable/corrupt FITS file is logged and recorded with a `read_error`
column, never allowed to abort a whole night's inventory." These two
statements are in tension. Resolution: **add `read_error` as the last column
in `INVENTORY_COLUMNS`.** For a successfully-read file it is `None`; for a
failed one, all `FrameMeta`-derived columns are `None` and `read_error` holds
`str(exc)` (never the exception object itself — that would violate the
global "never return an Exception instance as data" rule).
`metas_from_inventory` must skip rows where `read_error` is set (log a
warning, do not raise).

Full planned column list (finalize when writing the module):
```
path, naxis1, naxis2, frame_type, exptime, date_obs, filter, ra_deg, dec_deg,
pixel_scale, ccd_temp, gain, read_noise, saturation, airmass, object,
telescope, instrument, observer, binning, focal_length_mm, site_lat_deg,
site_lon_deg, site_elevation_m, mjd, night, is_valid, profile_name,
origins_json, read_error
```
`date_obs` is stored as an ISO-T string (`Time.isot`) or `None`; `night` as
`date.isoformat()` or `None`; `origins_json` is
`json.dumps({field: {"source": origin.source.value, "detail": origin.detail}
for field, origin in meta.origins.items()})` for **every** origin (not the
filtered subset `FrameMeta.provenance_json()` returns), so the full
`origins` mapping round-trips through `metas_from_inventory` and
`FrameMeta.provenance_records()` still works correctly after reload. Cast
all string-valued columns to `object` dtype explicitly after building the
DataFrame — do not rely on pandas' inference alone, and never use a
`numpy` structured/fixed-width dtype (`U20` etc. — this is exactly
Appendix A bug 51).

### 2. The once-only-read guarantee is real but has one documented scope limit

`fits_io.read_header(path, profile=None)` returns **only the primary HDU's
header** — it merges in an extension header only when `profile is not None`
(see `src/robo43/io/fits_io.py` lines ~154-169). This is existing code,
**not owned by this unit, must not be modified**.

Consequence, verified empirically this session: the `synth2` synthetic
fixture is a 2-HDU MEF (primary header has no data, `NAXIS1`/`NAXIS2` only
exist on the `SCI` extension's auto-generated header). If
`build_inventory(profile=None)` (per-file auto-resolution) tries to satisfy
"read the header exactly once" by calling `read_header(path)` once
(primary-only) and resolving the profile from *that*, it can never get a
correctly-merged header for a MEF instrument in the same call — getting the
merged header requires calling `read_header(path, profile)` a second time,
i.e. two `fits.open` calls for that file.

Decision: **`build_inventory`'s once-only-read guarantee is honoured by
reading only the primary header in per-file auto-resolution mode, i.e.
`read_header(path)` once, `resolve_profile(header=...)` from that, then
`read_meta(path, profile, header=<that same primary header>)` — never a
second `fits.open` call.** This works correctly for every single-HDU
instrument (`robo43`, `generic`, and any realistic instrument whose
canonical fields live in the primary header — the normal case). It does
**not** correctly resolve a MEF instrument's `NAXIS1`/`NAXIS2` in auto mode;
those files will land as `read_error` rows, which is the intended graceful
degradation, not a bug to silently paper over.

**`cli/inspect.py` and `cli/fixheader.py` do NOT route through
`build_inventory` for this reason.** They do their own per-file resolution:
if `--instrument NAME` is given, resolve once by name (no header probe
needed) then one `read_header(path, profile)` call (correctly merged, one
read). If `--instrument` is *not* given, they do a documented **two-read**
auto-resolution (`read_header(path)` for matching, then let `read_meta`
internally call `read_header(path, profile)` again for the correctly-merged
read) — acceptable because neither subcommand has a once-per-file contract;
only `io/inventory.py::build_inventory` does, and it is the one place with a
regression test asserting the exact call count.

Write the regression test (`test_build_inventory_reads_header_once_per_file`
in `tests/test_inventory.py`) against the **`robo43` synthetic night only**
(single-HDU, so the once-only path is both correct and exercised), with
`nprocs=1` (a multiprocessing pool would run workers in separate processes
where a test-process monkeypatch of `astropy.io.fits.open` does not apply).
Monkeypatch `astropy.io.fits.open` itself (not a copy) — `robo43.io.fits_io`
does `from astropy.io import fits` then calls `fits.open(...)`, so patching
the attribute on the real module object is visible to it.

### 3. A new packaged instrument profile is required: `src/robo43/data/instruments/synth2.toml`

Acceptance criterion 3 requires "the `synth2` night resolves end-to-end
using only its profile TOML and no code changes." Verified empirically that
this is **not achievable via the existing packaged `generic.toml`** (its
default `layout.science_hdu = 0` never merges the `SCI` extension, so
`NAXIS1`/`NAXIS2` are unreachable for the synth2 MEF fixture — confirmed by
direct reproduction, `MetadataError: NAXIS1/NAXIS2 could not be determined`).

There is already a **`tests/fixtures/synth2.toml`** file (created by the
same interrupted agent that landed `io/meta.py` etc. — timestamps line up,
10:26-10:29 — this is a test-fixture companion to
`tests/fixtures/make_synthetic.py`, not something a different concurrent
phase owns). It has `layout.science_hdu = "SCI"`, correct
`gain_e_per_adu`/`read_noise_e`/`saturation_adu` matching
`SyntheticTruth`, explicit `keywords` aliases for `EXPOSURE`/`IMGTYPE`/
`SET-TEMP`/`OBJCTRA`/`OBJCTDEC`, and an `imagetyp_map`. It does **not** have
an `[instrument.site]` block, so airmass/the `zeropoint` stage cannot
resolve from it as-is.

Decision: **do not modify `tests/fixtures/synth2.toml`.** Instead, create
`src/robo43/data/instruments/synth2.toml` as a new packaged profile (this
is additive — it only adds a new key to `available_profiles()`'s dict, it
touches nothing else) with the same core settings as the fixture copy, plus
a `[instrument.site]` block (e.g. `lat_deg = -23.0, lon_deg = -46.0,
elevation_m = 800` — verified empirically this gives a real, non-`None`
airmass for the synthetic science pointing/date). This makes
`--instrument synth2` (and automatic `TELESCOP=SYNTH2-TEL` header matching,
no flag needed) work through the real CLI. Verified manually (outside the
package, via `_profile_from_toml_dict` on equivalent TOML text) that with
`science_hdu = "SCI"` every canonical field resolves correctly for all four
synth2 frame categories (bias/dark/flat/science), and that with the site
block added, all of `STAGE_REQUIREMENTS` (including `zeropoint`) pass with
**zero** validation issues once stages are filtered per frame type (see
decision 4).

### 4. `robo43 inspect` filters each stage's requirements to relevant frame types

`STAGE_REQUIREMENTS` in `io/validate.py` (not owned by this unit) is
frame-type-agnostic: `validate()` checks every requested stage's required
fields against *every* meta passed in, regardless of `frame_type`. Checking
e.g. `"zeropoint"` (needs `airmass`) or `"photometry"` (needs
`filter`/`gain`/`saturation`) against a bias frame would spuriously report
it as broken (bias frames never have a scientifically meaningful filter or
airmass).

Decision: `cli/inspect.py` owns a **presentation-layer** mapping (not a
change to `validate()`, which is untouched) restricting which frame types
each stage is actually checked against:
```python
_STAGE_APPLIES_TO = {
    "bias": ("bias",), "dark": ("dark",), "flat": ("flat",),
    "reduce": ("flat", "science"), "astrometry": ("science",),
    "photometry": ("science",), "zeropoint": ("science",),
}
```
It calls `validate(subset, stages=(stage,))` once per requested stage with
`subset` pre-filtered by `_STAGE_APPLIES_TO`, then merges all the resulting
`ValidationIssue`s into one `ValidationReport(metas=tuple(all_metas),
issues=tuple(all_issues))` (constructed by keyword, per Rule A) so
`as_table()`/`emit_overrides_template()` still see the whole night. Default
`--stages` (when the flag is omitted) is **all** of
`STAGE_REQUIREMENTS.keys()` — safe now that each stage is scoped to its
relevant frame types.

### 5. The "three keywords" in acceptance criterion 3 are `filter`, `gain`, `saturation`

Verified empirically. `make_synthetic.py`'s `strip=("gain", "pixel_scale",
"filter")` does **not** map 1:1 onto three missing `FrameMeta` fields when
the auto-matched `robo43` profile is used, because `robo43.toml` supplies
static fallbacks for `gain_e_per_adu` (10.0) and `pixel_scale_arcsec`
(0.529) — those two fields resolve via `Source.PROFILE`, not `None`, so
`validate()` does not flag them. Only `filter` ends up genuinely `None`
under the `robo43`-profile path (the profile has no filter default).

For the acceptance-criterion test to produce exactly three missing,
blocking fields, run `robo43 inspect` on the **stripped** night with
**`--instrument generic`** explicitly (the `generic` profile has *no*
static fallbacks for `gain`/`saturation`, and `pixel_scale` also has no
usable WCS since it was stripped too). Verified empirically with
`--stages photometry` (`STAGE_REQUIREMENTS["photometry"] = ("filter",
"gain", "saturation")`) against the stripped night + `generic` profile:
exactly the three expected `ValidationIssue`s per science frame, one per
field, all `stage == "photometry"`. This is very likely the intended
design — `STAGE_REQUIREMENTS["photometry"]` is *exactly* three fields, and
they are *exactly* the three the strip parameter is named after minus one
substitution (`pixel_scale` strip's actual casualty for the `photometry`
stage check is nothing — it only affects the separate `astrometry` stage,
which the test should not need to invoke for this criterion).

Write the "succeeds once an `--overrides` TOML supplies the three keywords"
half of the test with a **hand-authored** `[defaults]` overrides TOML (not
literally the CLI's own `--emit-overrides` output, which is deliberately
just blank placeholders `key = ""` that fail `OverrideSet`'s type coercion
if used as-is — that blank-template file stands in for what a human is
supposed to fill in, and the test should still separately assert the
template was written and contains the three expected keys). After supplying
real values (e.g. `gain = 2.0`, `saturation = 55000`,
`filter = "V-Bessell"` in `[defaults]`), re-running should exit 0, and
`FrameMeta.provenance_records()` for an affected file should contain
entries of the form `"gain=override:defaults"`,
`"saturation=override:defaults"`, `"filter=override:defaults"` (the format
is `f"{field}={origin.source.value}:{origin.detail}"`, and
`OverrideSet.for_path`'s `[defaults]` table tags its origin detail as
literally the string `"defaults"`).

### 6. `filter_inventory`/`build_inventory` API shapes settled this session

- `build_inventory(paths, *, profile=None, overrides=None, settings=None,
  nprocs=1) -> pandas.DataFrame` — worker is a module-level function
  `_build_row(task: _InventoryTask) -> dict[str, Any]` where
  `_InventoryTask` is a frozen dataclass of `(path: str, profile:
  InstrumentProfile | None, overrides: OverrideSet | None, settings:
  Settings | None)`. Verified this session that `InstrumentProfile`,
  `Settings`, and `OverrideSet` are all picklable as-is (`pickle.dumps`/
  `pickle.loads` round-trip tested directly against `resolve_profile(name=
  "robo43")`, `load_settings(use_user_config=False, env={})`, and
  `OverrideSet.from_cli(["gain=1.5"])`), so passing the real objects through
  the payload is fine — no need to re-serialize them to TOML text per-task.
  `nprocs > 1` uses `multiprocessing.Pool(processes=nprocs).imap(_build_row,
  tasks)` (ordered, for deterministic test output); `nprocs <= 1` runs the
  loop directly in-process with no `Pool` at all (also gives the
  monkeypatch-based regression test a clean single-process call count).
- `filter_inventory(df, *, frame_type=None, filter=None, object=None,
  night=None, exptime=None, ccd_temp=None, valid_only=True) ->
  pandas.DataFrame` — `night` accepts a `datetime.date` or an ISO string;
  `exptime`/`ccd_temp` compare via `np.isclose` (not `==`) since these are
  floats reconstructed through a possible CSV round-trip.
- `write_inventory(df, path)` dispatches purely on `Path(path).suffix`
  (`.csv` -> `df.to_csv`, `.parquet` -> `df.to_parquet`; anything else
  raises `ValueError`). `pyarrow` (24.0.0) is installed in this venv, so
  Parquet works without extra handling.

### 7. `cli/inventory.py`, `cli/inspect.py`, `cli/fixheader.py` shapes settled

Full draft code for all three was written out in this session's working
notes (not yet committed to disk). Key points for whoever picks this up:

- All three subcommand modules must expose exactly `SUBCOMMAND: Subcommand`
  and call `register(SUBCOMMAND)` at import time, matching the pattern
  already established in `src/robo43/cli/config.py`.
- Both `inventory` and `inspect` build an `OverrideSet` from `--overrides`
  (`OverrideSet.from_toml`) merged with `--set` (`OverrideSet.from_cli`) via
  `.merge()`, with `--set` taking precedence (merge order:
  `toml_overrides.merge(cli_overrides)` — `OverrideSet.merge` documented as
  right-hand-side-wins per key, confirm this when writing the code by
  re-reading `io/validate.py::OverrideSet.merge`).
- `cli/fixheader.py` replaces `src/fix_header.py` (Appendix A bug 50): must
  operate on the actual per-file `Path` argument (never a mutated loop
  variable), must tolerate a header with no FILTER-like alias present
  (scan `profile.aliases_for("filter")`, skip with a logged message, never
  `KeyError`), must `shutil.copy2` a `.bak` backup **before** opening the
  file in `mode="update"` and refuse (raise `Robo43Error`, do not touch the
  file) if the backup copy fails, and must `header.add_history(...)`
  recording old value, new value, `robo43.__version__`, and a UTC ISO
  timestamp. `--dry-run` (via `settings.runtime.dry_run`, already wired by
  `cli/common.py`) must report the message but skip both the backup and the
  actual open-for-write. Use `open_fits(path, mode="update")` from
  `io/fits_io.py` for the write (already exists, do not reimplement), and
  `science_hdus(hdul, profile)` (never index `hdul[0]` directly — that is a
  hard global rule) as the fallback HDU if the alias keyword is not found on
  any HDU by direct scan.

## Problems found in files not owned by this unit

- `src/robo43/io/fits_io.py::read_header` (not owned): only merges an
  extension header into the returned copy when `profile is not None` (see
  decision 2 above). This is a real, load-bearing asymmetry — worth flagging
  upward, but out of scope to fix here since `fits_io.py` belongs to the
  interrupted agent's already-landed set, not this unit.
- `tests/test_quality.py` (owned by a different, concurrently-running
  phase) showed a transient failure mid-session
  (`test_one_row_per_class`, `test_large_table_performance_and_correctness`)
  that had resolved itself by the next run seconds later — almost certainly
  a concurrent-edit race from that other agent, not a real regression. Not
  investigated further; not this unit's file.

## Contract deviations

- `INVENTORY_COLUMNS` gains a `read_error` column beyond the contract's
  literal "path + every FrameMeta scalar field + origins_json" — see
  decision 1. Necessary to satisfy the explicit "never allowed to abort a
  whole night's inventory" requirement in the same contract section.
- A new file, `src/robo43/data/instruments/synth2.toml`, is added outside
  the brief's literal "files you own" list — see decision 3. Justified
  because acceptance criterion 3 is otherwise unsatisfiable with the
  packaged profiles that exist today, and the addition is purely additive
  (a new instrument TOML, touching no other agent's code or tests).
- `cli/inspect.py`'s per-stage frame-type filtering (decision 4) is an
  inspect-level policy layered on top of `validate()`, not a change to
  `io/validate.py` itself, which stays untouched.

## Next steps (ordered)

1. Re-run `python3 -m pytest -q` fresh to reconfirm the 123-passed baseline
   hasn't moved further while this session was paused (other agents are
   actively landing work in this repo).
2. Write `src/robo43/io/inventory.py` per decisions 1, 2, 6 above.
3. Write `src/robo43/data/instruments/synth2.toml` per decision 3 (base it
   on `tests/fixtures/synth2.toml`'s settings, add the site block; do not
   modify the fixture file itself).
4. Write `tests/conftest.py` exposing `synthetic_night`, `synthetic_night_synth2`,
   `synthetic_night_stripped` (all `tmp_path`-scoped, wrapping
   `make_synthetic_night` with `instrument="robo43"`/`"synth2"`/
   `instrument="robo43", strip=("gain","pixel_scale","filter")`
   respectively) and `default_settings` (`load_settings(use_user_config=False,
   env={})`). Every settings load anywhere in this unit's fixtures/tests must
   pass `use_user_config=False, env={}`. Follow the existing per-file
   `_reset_logging`/`_restore_robo43_logger` autouse-fixture pattern already
   established in `tests/test_logging_setup.py` and
   `tests/test_cli_framework.py` for any test file that drives the CLI
   (`main()`/`configure_from_args`) — do not centralize logging
   snapshot/restore in `conftest.py` itself, to match the established
   per-file convention and avoid interfering with other phases' test files.
5. Write `tests/test_inventory.py`, including the once-only-read regression
   test (decision 2) — this is the single most important test in this unit.
6. Write `src/robo43/cli/inventory.py`, `src/robo43/cli/inspect.py`,
   `src/robo43/cli/fixheader.py` per decisions 4 and 7.
7. Write `tests/test_cli_inspect.py` covering: (a) full unstripped
   `synthetic_night` → `robo43 inspect` exits 0; (b) `synthetic_night_synth2`
   with `--instrument synth2` → exits 0 across `reduce,astrometry,photometry`
   at minimum (verified empirically this session that all `STAGE_REQUIREMENTS`
   pass with zero issues once the synth2.toml site block is in place — safe
   to check all stages, not just three); (c) `synthetic_night_stripped` with
   `--instrument generic --stages photometry` → exits 1, issues name file +
   field (`filter`/`gain`/`saturation`) + stage `photometry`, and
   `--emit-overrides` writes a template containing those three keys; (d) a
   second run with a hand-authored filled `--overrides` TOML → exits 0, and
   `FrameMeta.provenance_records()` shows the three `override:defaults`
   entries (decision 5).
8. Write `tests/test_headers.py`, `tests/test_meta.py`, `tests/test_validate.py`,
   `tests/test_fits_io.py`, `tests/test_profile.py` — straightforward unit
   coverage of the already-landed modules; none of this was started, no
   design decisions pending, just needs writing.
9. Run `python3 -m pytest -q` (whole suite) and `ruff check src/robo43 tests`;
   paste real output in the final report. Fix only files owned by this unit.
10. Verify acceptance criteria 3-5 from the original brief explicitly
    (end-to-end synth2 resolution, stripped-night fail-then-succeed-via-
    overrides with provenance, `robo43 --help` listing `inventory`,
    `inspect`, `fixheader`, `config`) and paste verbatim output.
