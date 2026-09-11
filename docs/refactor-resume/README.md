# Refactor progress and resume notes

**Paused:** 2026-09-08. **Suite at pause:** 140 passing, 0 failing,
`ruff check src/robo43 tests` clean.
**Plan:** `REFACTORING_PLAN.md` (unchanged; it is the specification).
**Interface contract:** `docs/INTERFACE_CONTRACT.md` — read this before touching
any public signature. It was written up front so parallel work would not drift,
and its addenda record what has landed and any deviations. Where the body and an
addendum disagree, the addendum wins.

Per-unit notes from the agents that were working when the session paused live
beside this file, one markdown file per unit.

---

## Read this first if you are resuming

1. `docs/INTERFACE_CONTRACT.md` — the module layout and the three core types
   (`Settings`, `InstrumentProfile`, `FrameMeta`), then the addenda in order.
2. This file, for what is done and what is next.
3. The per-unit resume notes in this directory, for work that was mid-flight.

The suite runs from a bare checkout with no install step:

```
cd <repo> && python3 -m pytest -q      # pyproject sets pythonpath = ["src"]
ruff check src/robo43 tests
```

---

## Phase status

| Phase | Scope | Status |
|---|---|---|
| 1 | package skeleton, config, packaging | **done** |
| 1B | instrument profiles, keyword contract, `io/` | **mostly done** — `inventory.py` and the three CLI subcommands outstanding |
| 2 | cross-cutting hygiene | **done for new code**; legacy `src/*.py` not yet replaced by shims |
| 3 | performance | **partly** — 3.2 (single header read) and 3.4 (chunked combine) done; 3.1, 3.3, 3.6, 3.7 outstanding |
| 4 | reduction correctness + product model | **partly** — 4.1, 4.2, 4.4/4.4.1 done; 4.3 half done (bias+dark, no flat); 4.5, 4.6 outstanding |
| 5 | dark, masks, missing blocks | **partly** — 5.1 masks, darks and overscan done; `flat.py`, `set.py`, non-linearity outstanding |
| 6 | photometry quality + `CATALOG` HDU | **partly** — `quality.py` landed, needs review; sextractor/aperture/zeropoint outstanding |
| 7 | astrometry | **not started** |
| 8 | service mode | **not started** |
| 9 | PostgreSQL | **not started** |
| 10 | tests, CI, docs | **partly** — synthetic fixtures exist; no CI, no user docs |

---

## Landed and verified

Verified means: I ran the check myself against injected truth or a clean
environment, not that a subagent reported success.

**Foundation**
- `config.py` — every hard-coded value from plan §1.2 moved here. Precedence
  CLI > `ROBO43_*` env > `--config` > `~/.config/robo43/config.toml` > packaged
  default, all four layers verified. Unknown section or key raises `ConfigError`
  rather than silently doing nothing. `config_hash()` covers only
  `SCIENCE_SECTIONS` (astrometry, calib, coadd, photometry, targets), so `--np`
  or a log path cannot change a product's `PIPECFG`.
- `logging_setup.py` — idempotent: three calls leave one handler and emit each
  record once (verified). Queue handler for worker processes. Fixes Appendix A
  bug 27: `--loglevel` works alone, `--verbose` is just an alias for
  `--loglevel DEBUG`.
- `exceptions.py` — the hierarchy that ends "return an `Exception` as data".
- `provenance.py`, `external.py` — `PIPEVER`/`PIPEGIT`/`PIPECFG`/`PIPEDATE`
  cards, sha256, packaged-data access via `importlib.resources`, and a clear
  `ExternalToolError` when `source-extractor` is absent.
- `pyproject.toml` — hatchling, src layout, `requires-python >=3.12`, extras
  `[db]`/`[service]`/`[dev]`, `robo43` entry point. **Verified end to end:**
  `pip install` into a clean venv, then `robo43 --help` and `robo43 config show`
  both run from `/tmp`, and the packaged `.conv`/`.nnw`/`.sex`/`.param` files
  are reachable through `importlib.resources.files("robo43.data")`. The 27
  SExtractor files moved from the repo-root `data/` as git renames.
- `cli/` — `registry.py`/`common.py`/`main.py`, with subcommand auto-discovery:
  a new subcommand is added by dropping `cli/<name>.py` with a module-level
  `SUBCOMMAND`, and `robo43 --help` works with any subset present and does not
  break when `psycopg`/`watchdog` are missing.

**Science**
- `calib/combine.py` (§4.1, §4.2, §3.4) — the headline fix: sigma-clipping now
  runs **along the stack axis**, not spatially. Verified contrast on a stack
  carrying a real hot column and a real gradient: the corrected combination
  recovers the column at 410.1 against a truth of 410.0 with gradient RMS 1.140
  where the noise floor is 1.118, while the legacy order
  (`sigma_clip(np.median(stack, axis=0))`) masks **all 40 pixels of the real
  hot column** as spatial outliers. `combine_lazy` is bit-identical to
  `combine_stack` and its peak memory is independent of stack depth.
- `products/dq.py` — the DQ bit assignment. Bits 0–5 are fixed by plan §5.1;
  6–9 extend it. These reach archived files: **do not renumber them.**
- `calib/masks.py` (§5.1 item 3, a standing README TODO) — hot, dead, unstable
  and bad-column detection. Verified on a non-square frame: all injected
  defects recovered, a defective column found at every index including 0, 1 and
  N-1, zero false positives across five clean darks and a vignetted flat,
  vignetting not called dead, thermal gradient not called hot.
- `reduce/cosmics.py` (§5.2) — astroscrappy wrapper, optional import, ADU and
  electron paths agreeing to 99.99%. Verified 100% recall at a 0.03%
  false-positive rate on a frame with self-consistent Poisson noise, with
  saturated star cores correctly not flagged.
- `photometry/lightcurve.py` — replaces `calculate_average_mags.py`; fixes
  Appendix A bugs 45, 46, 47. Flux-space inverse-variance averaging verified
  against a hand computation to 1e-9.
- `io/` — `meta.py` (`FrameMeta`), `headers.py`, `fits_io.py`, `validate.py`,
  and `instruments/profile.py` + `keywords.py` with packaged `robo43.toml` and
  `generic.toml` profiles. Landed by an agent that was interrupted; the
  modules import and their tests pass, but **I have not independently verified
  these to the standard above** — see "What to check first" below.
- `photometry/quality.py` (16 tests), `products/mef.py` + `products/catalog.py`
  (21 tests) — landed complete just before the pause. `CatalogMetrics`
  round-trips through its FITS header exactly, for both a fully-populated and
  a sparse instance, which is what lets phase 9 read metrics out of the header
  rather than recomputing them. **Not independently re-verified by me.**
- `calib/master.py` (`MasterFrame`), `calib/bias.py`, `calib/dark.py`
  (17 tests) — master bias with overscan/trim, and the phase-5.1 dark library:
  grouping by exposure time and CCD temperature, the normalised dark-current
  map, and Arrhenius temperature scaling. I spot-checked the scaling myself:
  doubling the exposure doubles the current (ratio 2.000) and a temperature
  step of `dark_doubling_temp_c` also doubles it (ratio 2.000), and
  `select_dark` correctly refuses to extrapolate beyond
  `dark_temp_extrap_max_c`, returning `method="none"` with an explanatory
  reason. This closes the README's "dark scaling" TODO.
  **`calib/flat.py` and `calib/set.py` do not exist** — see below.

**Test fixtures** — `tests/fixtures/make_synthetic.py` generates a synthetic
night (bias, darks at two exposure times, flats in three filters, science
frames with a known star field, injected hot pixels and cosmic rays) at
256x256, plus a second instrument `synth2` with different keyword spellings and
a multi-extension layout, and a variant with keywords deliberately stripped.
There is no real FITS data in this repo, so these fixtures are what makes
anything verifiable.

---

## Appendix A bugs closed so far

Fixed and covered by a regression test: **8** (spatial sigma clipping),
**19** (saturation 200000 vs 55000 — the profile uses 55000 per §1B.2),
**27** (`--loglevel` inert without `--verbose`), **28** (divide-by-zero weight
map), **45**, **46**, **47** (nightly averages), **55** (duplicated
`setup_logging`), **57** (`requirements.txt` missing half the imports),
**15** (astrometry index path hard-coded to one machine — now
`astrometry.index_dir`). The README TODOs "create hot pixel mask" and
"dark scaling" are addressed by `calib/masks.py` and, once the master-frame
unit lands, `calib/dark.py`.

Not yet closed, and still live in `src/*.py`: everything else, including
**1** (the unconditional `pdb.set_trace()` on the main reduction path). The
legacy scripts are untouched so far — they are to become deprecation shims in
the final wave, per plan §1.1.

---

## Two bugs found during verification that the tests had passed over

Recorded because both are easy to reintroduce.

1. **`combine.py` read-noise floor.** It was written as a *substitute* for
   undefined variance rather than a lower bound, so a perfectly stable pixel —
   sample variance ~1e-28 from float rounding, not exactly zero — passed the
   `<= 0` guard and received weight **1.05e28**. Finite, so `isfinite` asserts
   passed; the test only checked `> 0`. One such pixel would dominate every
   inverse-variance combination downstream. Now imposed with `np.maximum`, with
   a separate guard for the no-read-noise path.
2. **`masks.py` bad-column detection.** It only ever examined the flat, so a
   column visible only in the dark map was never searched for; and the test
   itself was unsound. Three attempts failed before the right criterion: a
   global-median threshold is swamped by real structure; a running median
   filter must extrapolate past the array ends and *hid* a column at x=5; a
   clipped polynomial fit *invented* one at x=0; a leave-one-out local median
   over-flagged 21 columns because its residual is structured curvature, not
   noise, so the MAD collapses. The working criterion is a **fractional**
   departure from the immediate neighbourhood (`CalibSettings.bad_column_frac`,
   default 0.10). Do not "improve" this back into a sigma test.

---

## Known issues and deviations

- **`sewpy` is not on PyPI.** It is declared as a direct VCS reference
  (`sewpy @ git+https://github.com/megalut/sewpy.git`) with
  `[tool.hatch.metadata] allow-direct-references = true`. This makes
  `pip install .` require network and git, and **blocks publication to PyPI**,
  which rejects direct references. Worth deciding: either vendor the small
  amount of `sewpy` that is used, or call `source-extractor` through
  `subprocess` directly and drop the dependency. `robo43/external.py` already
  locates the binary, so the second option is close at hand.
- **Licence declared as `GPL-3.0-only`.** The `LICENSE` file is plain GPLv3
  with no "or later" clause. Correct this if "or later" was intended.
- `calib/masks.py` writes FITS directly instead of going through
  `io/fits_io.write_fits`, because it was built before that module existed. It
  is still atomic. There is a `# TODO(io):` marker; collapse the duplication.
- FITS cards must not exceed 80 columns. astropy only *warns* and silently
  truncates the comment, so an over-long card ships quietly into archived
  products. One such card (`UNSTNMAD`, 87 columns) was found and fixed, and the
  invariant is now asserted — keep asserting it as Appendix B keywords are added.

## Environment notes

- Python 3.13.5 venv at `/home/herpich/py313`, already the default `python3`.
- `source-extractor` is at `/usr/bin/source-extractor`.
- **`solve-field` is not installed**, so that step of the phase-7 fallback chain
  must degrade cleanly rather than assume the binary exists.
- **No astrometry.net index files are configured.** `astrometry.index_dir` has
  no default on purpose. Phase 7 tests must skip without them
  (`needs_astrometry_index` marker is registered).
- **PostgreSQL is running but this account has no `CREATEDB` right**, so phase
  9's live-database acceptance criterion cannot be met here. Plan for portable
  SQL tested against SQLite, DDL asserted by rendering against the
  `postgresql` dialect, and live tests behind the `needs_db` marker and
  `ROBO43_DB_URL`. Credentials for a scratch database would unblock it.
- Registered pytest markers: `needs_sextractor`, `needs_db`,
  `needs_astrometry_index`, `slow`.

---

## What to check first, before writing anything new

1. Independently verify the `io/` layer and `instruments/` profiles against the
   synthetic fixtures — especially **Appendix A bug 3**, the numeric-RA case
   (`':' in <float>` raises `TypeError`; SIMBAD writes decimal degrees back
   into the header), sexagesimal round-trips, `IMAGETYP` winning over an
   `OBJECT` containing "flat", and `parse_region` axis order and inclusivity.
2. Review `photometry/quality.py`, `products/mef.py` and `products/catalog.py`,
   which landed unreviewed. For the catalogue the critical property is that
   `CatalogMetrics.from_header(m.to_header()) == m` round-trips **exactly** —
   phase 9 treats that header as the source of truth and reads its metrics
   rather than recomputing them.
3. Confirm the once-only header read is actually asserted by a test that counts
   `astropy.io.fits` calls. §3.2 is one of the larger performance wins and is
   easy to regress.

## Open decisions recorded during the pause

- **Eight Appendix B.1 keywords are written by nothing yet**: `ASTSOLVR`,
  `ASTRMS1`, `ASTRMS2`, `ASTNREF`, `ASTEPOCH`, `WCSSOLVD`, `NCOMBINE`,
  `TEXPTIME`. Resolved in `docs/INTERFACE_CONTRACT.md` ADDENDUM 8: add
  `astrometry_cards()` and `coadd_cards()` as siblings of `calibration_cards`
  in `products/mef.py`, taking plain scalars so `mef.py` stays free of
  astrometry and coadd imports. `WCSSOLVD` is always written (`F` when
  unsolved); the rest are omitted when unknown. Phase 7 and phase 4.5 each own
  their helper.

- **`--instrument` must accept a path, not only a registered name.** A test
  fixture's profile must not be shipped in `robo43/data/instruments/`, which is
  for real instruments; so `resolve_profile` should take either a name or a
  filesystem path, letting `tests/fixtures/synth2.toml` be used directly. See
  `docs/INTERFACE_CONTRACT.md` ADDENDUM 9.
- **`generic.toml` cannot resolve a multi-extension instrument**, which is a
  real gap against the §1B.5 promise that `generic` assumes nothing beyond the
  required keywords. It should derive `science_hdu` when the primary HDU holds
  no pixels. Needs a test.
- **A 61st bug, not in Appendix A**: `src/process_robo43_frames.py:632` uses
  `mask = ...` where the surrounding code uses `mask &= ...`. It happens not to
  lose the earlier cuts today, because they are applied before it, but the
  pattern means any cut appended after it would silently discard them.
  `photometry/quality.py`'s accumulating bitmask removes the whole idiom.

- **`scale_dark` should repeat `select_dark`'s extrapolation guard.** The
  refusal to extrapolate beyond `CalibSettings.dark_temp_extrap_max_c` is
  implemented in `select_dark` (`calib/dark.py:276`), so the normal path is
  safe. But `scale_dark`'s `current_map` branch applies the Arrhenius factor
  with no bound of its own: handed a selection built elsewhere — a cached or
  persisted one, as service mode will want — it will happily return a scale of
  ~207x and subtract a wildly wrong dark. Not reachable today (nothing outside
  the tests calls `scale_dark`), so this is defence in depth to add when
  `reduce/frame.py` becomes its first real caller.
- **Appendix A bug 29 (flats not dark-subtracted) is still open**, because
  `calib/flat.py` was never started. This is the headline numerical fix of
  §4.3 and should be the next thing built after phase 1B's inventory.

## Next steps, in dependency order

1. Finish phase 1B: `io/inventory.py` and the `inventory`, `inspect`,
   `fixheader` subcommands. See `phase1b-inventory-cli.md`.
2. Finish phase 4.3: `calib/flat.py` (dark-subtracted, level-cut,
   centrally-normalised, per-filter clobber guard) and `calib/set.py`
   (`CalibrationSet`, masters loaded once). `bias.py` and `dark.py` are done.
   See `phase4-5-master-frames.md` for the settled field contracts and
   filename conventions.
3. Phase 4.4 integration: `reduce/frame.py`, the reduction order
   (overscan → trim → bias → dark → flat → mask → cosmic rays → gain), and the
   `ReductionStatus` record that workers **return** (§3.7 — the legacy
   `proc_status` dict is silently lost under `multiprocessing`).
4. Phase 3.1/3.3/3.6: the per-process astrometry solver singleton (the single
   largest performance win in the plan), `CalibrationSet` loaded once, and a
   module-level pool worker taking a small picklable payload.
5. Phase 6 remainder: `photometry/sextractor.py` (one wrapper, one config,
   `WEIGHT_TYPE = MAP_WEIGHT` with the real inverse-variance map and
   `FLAG_IMAGE` = the DQ plane), `aperture.py`, `zeropoint.py`.
6. Phase 7 astrometry, then phase 9 database, then phase 8 service mode, then
   phase 10 CI and user documentation.
7. Last: convert `src/*.py` into thin deprecation shims emitting
   `DeprecationWarning`, per plan §1.1, so existing cron jobs keep working for
   one release cycle.

## A note on delegation cost

This refactor was executed by parallel subagents. Measured over the first
session: ~1.56 M agent tokens produced ~6 400 lines, about 244 tokens per line.
The dominant costs were nested fan-out (42% — six supervising agents became
twenty-two) and repeated re-reading of the same two large documents (~29%).
Units given a tight import allowlist and no need to read the plan cost ~143
tokens per line; a unit whose acceptance criterion was a network
`pip install` loop cost 376. Keep briefs self-contained, quote the plan rather
than pointing at it, and keep network verification out of the inner loop.
