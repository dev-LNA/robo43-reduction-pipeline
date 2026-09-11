# Phase 4.3 / 5.1 resume note — master calibration frames (`calib/bias.py`, `calib/dark.py`, `calib/flat.py`, `calib/set.py`)

Written at a session checkpoint (paused mid-unit by the coordinator). Read
this before touching any of these four files again.

## Unit

Refactoring plan §4.3 (flat-field construction, bugs 29/30/31) and §5.1 (dark
frame support, currently a stub in `src/make_calibration_frames.py`). Goal:
replace `src/make_calibration_frames.py` with four new modules —
`src/robo43/calib/bias.py`, `dark.py`, `flat.py`, `set.py` — plus
`tests/test_bias.py`, `test_dark.py`, `test_flat.py`, `test_calibset.py`.
Spec source of truth: `docs/INTERFACE_CONTRACT.md` (this file now lives in the
repo; earlier sessions only had it in a session scratchpad) — read the body's
"Global rules", ADDENDUM 3 (units convention, masks, `CalibrationSet`),
ADDENDUM 6 Rule A (keyword-only dataclass construction), ADDENDUM 7 (masks),
and ADDENDUM 8 (confirms `products/mef.py::calibration_cards` is NOT this
unit's concern — see "Contract deviations" below).

## Per-module status

- **`calib/bias.py` — COMPLETE.** `select_frames`, `apply_overscan_trim`,
  `build_master_bias`. All required behaviour implemented: overscan
  subtraction then trim (each independently optional and recorded via header
  cards `OSCANCOR`/`OSCANLVL`/`OSCANREG`/`TRIMMED`/`TRIMREG`), sigma-clip
  combine via `combine_stack` (never a second combiner), graceful
  single-frame fallback, `master_bias.fits` write with `settings.runtime.clobber`
  honoured.
- **`calib/dark.py` — COMPLETE.** `DarkEntry`, `DarkSelection`,
  `build_master_darks`, `select_dark`, `scale_dark`. Grouping, the
  dark-current map (e-/s/pixel, gain applied exactly once), and the
  exact/scaled/current_map/none selection ladder with Arrhenius temperature
  scaling and extrapolation refusal are all implemented and tested.
- **`calib/master.py` — COMPLETE** (new shared file, not originally named as
  "owned" in the brief but explicitly permitted by it: "a shared `MasterFrame`
  dataclass ... or a small `calib/master.py` that you own"). `MasterFrame`
  dataclass + `to_hdulist()` + `write()`. Both `bias.py` and `dark.py` import
  it; `flat.py`/`set.py` must too (see below).
- **`calib/flat.py` — NOT STARTED.** Does not exist. This is the headline
  fix (Appendix A bug 29: flats not dark-subtracted) and it has **not been
  implemented in this session**. See "Next steps".
- **`calib/set.py` — NOT STARTED.** `CalibrationSet` does not exist. Not
  implemented in this session.
- **`tests/test_flat.py`, `tests/test_calibset.py` — NOT STARTED.** Do not
  exist.

## Files complete and passing

- `src/robo43/calib/master.py` — no dedicated test file (it's exercised
  indirectly through `test_bias.py`/`test_dark.py`, which both write and
  re-read master frames).
- `src/robo43/calib/bias.py` — covered by `tests/test_bias.py` (7 tests).
- `src/robo43/calib/dark.py` — covered by `tests/test_dark.py` (10 tests).

Verified directly (not just trusting the executer's report) from
`/home/herpich/Dropbox/robo43-reduction-pipeline`:

```
python3 -m pytest --collect-only -q
  -> collects cleanly, 13 test files, no errors (includes files from three
     other concurrently-checkpointed units: test_catalog.py, test_cli_framework.py,
     test_mef.py, test_quality.py)

python3 -m pytest -q
  -> 140 passed in 0.75s  (0 failed, 0 errors, 0 skipped)

ruff check src/robo43 tests
  -> All checks passed!

python3 -m mypy src/robo43/calib/master.py src/robo43/calib/bias.py src/robo43/calib/dark.py --ignore-missing-imports
  -> Success: no issues found in 3 source files
```

No regression versus the 123-passed baseline the coordinator quoted (140
includes this unit's 17 new tests plus other units' new test files that
landed in the same working tree).

One real bug was caught and fixed during this session, in a file this unit
owns: `tests/test_bias.py`'s first draft of the overscan/trim test applied
the synthetic pedestal only to the overscan *columns* rather than across the
whole readout, which does not model how a real overscan pedestal works
(the overscan region should represent the same electronic offset that
contaminates every pixel, not an offset unique to the overscan columns) —
the implementation in `bias.py::apply_overscan_trim` was correct throughout;
only the test fixture was wrong. Fixed; now passing.

## Files started but incomplete

None left mid-edit. `flat.py`/`set.py` were never started (zero bytes written,
not stubbed), so there is nothing to finish or delete — collection is clean.

## Whether flats are dark-subtracted yet (Appendix A bug 29)

**No.** `flat.py` does not exist yet, so bug 29 is not fixed in this session.
No contrast numbers were measured (there is no flat-building code to measure
them against). This is the single most important item for the next session —
see "Next steps".

## Whether the per-filter clobber guard checks the right filename (bug 31)

Not yet applicable — `flat.py` doesn't exist. However, the *mechanism* it
must use is already built and proven correct in this session:
`MasterFrame.write(path, overwrite=...)` delegates to
`robo43.io.fits_io.write_fits`, which raises `FileExistsError` naming the
**exact path passed to it** when `overwrite=False` and that path exists. So
as long as `flat.py` builds the path as
`outdir / f"master_flat_{filter_name}.fits"` per filter (not a single
`master_flat.fits`) and calls `.write(that_path, overwrite=settings.runtime.clobber)`,
bug 31 is fixed for free — do not reimplement the guard, and do not write a
single shared `master_flat.fits` path anywhere.

## Decisions made — a successor must not silently reverse these

1. **`settings` is the FULL `Settings` tree everywhere in this unit's public
   API**, not `CalibSettings`. Every public function (`build_master_bias`,
   `build_master_darks`, and — not yet written — `build_master_flats`,
   `CalibrationSet.load`/`dark_for`) takes `settings: Settings`. Internally,
   pass `settings.calib` into `combine_stack(...)` (which wants
   `CalibSettings`), and use `settings.runtime.clobber` for the write guard.
   This is a deliberate resolution of an ambiguity in the interface contract
   (the "## API" sketch just says `settings` with no type) — necessary
   because `RuntimeSettings.clobber` lives outside `CalibSettings` but the
   brief explicitly requires `--clobber`/`settings.runtime.clobber` to be
   honoured by these builders. Documented as a one-line comment at the top of
   both `bias.py` and `dark.py`; do the same in `flat.py`/`set.py`.

2. **`MasterFrame`** lives in `src/robo43/calib/master.py` (not inlined into
   `bias.py`), as:
   ```python
   @dataclass(slots=True)
   class MasterFrame:
       data: np.ndarray
       variance: np.ndarray | None = None
       weight: np.ndarray | None = None
       mask: np.ndarray | None = None
       header_cards: list[tuple[str, Any, str]] = field(default_factory=list)
       history: list[str] = field(default_factory=list)   # -> HISTORY cards
       bunit: str = "adu"
       path: Path | None = None

       def to_hdulist(self) -> fits.HDUList: ...   # SCI + optional VAR/WEIGHT/DQ
       def write(self, path, *, overwrite=False) -> Path: ...
   ```
   `flat.py` and `set.py` must import this same class from
   `robo43.calib.master` — do not define a second `MasterFrame`-like type.
   **`history` is the ONLY safe place for free-text explanations** (e.g.
   `DarkSelection.reason`) — `header.add_history()` auto-wraps long strings
   across multiple 80-column `HISTORY` cards with **no** `VerifyWarning`,
   whereas a long string in a card's `comment` field triggers
   `astropy.io.fits.verify.VerifyWarning` *at write time* (confirmed
   empirically this session) and is silently truncated. Keep every
   `header_cards` comment under ~45 characters. This constraint is enforced
   by a real test in `test_bias.py` (`warnings.catch_warnings` +
   `simplefilter("always")` around `.write()`, asserting zero
   `VerifyWarning`s) — replicate that test in `test_flat.py`/`test_calibset.py`.

3. **Output filename conventions** (fixed, do not change):
   - `master_bias.fits`
   - `master_dark_<mean_exptime:g>s_<mean_ccd_temp:.1f>C.fits`, or
     `master_dark_<exptime:g>s_unknownC.fits` when the group has no
     temperature at all. `<mean_exptime:g>`/`<mean_ccd_temp:.1f>` are the
     **group's own mean** exptime/temp (computed from its member frames),
     not the group's tolerance anchor.
   - `dark_current_map.fits` (singular, one composite map per calibration
     run — see point 5 below on how it's built).
   - **Not yet decided/built**: `master_flat_<FILTER>.fits` per filter (bug
     31 fix — literally required, see above) and whatever `set.py` names its
     optional mask file (`CalibrationSet.load` was never written, so nothing
     is fixed here yet; `mask.fits` was the plan but never implemented or
     tested — a successor is free to choose, but must document it here).

4. **`DarkEntry`/`DarkSelection` field names (fixed)**:
   ```python
   @dataclass(frozen=True, slots=True)
   class DarkEntry:
       path: Path
       exptime: float
       ccd_temp: float | None
       data: np.ndarray | None
       variance: np.ndarray | None
       n_combine: int

   @dataclass(frozen=True, slots=True)
   class DarkSelection:
       entry: DarkEntry | None
       scale: float
       method: str   # "exact" | "scaled" | "current_map" | "none"
       reason: str   # full sentence; caller puts this in MasterFrame.history
   ```
   The **current_map fallback is represented as a synthetic `DarkEntry`**
   wrapping the map: `DarkEntry(path=Path("dark_current_map.fits"),
   exptime=1.0, ccd_temp=<map's reference temp>, data=<map converted from
   e-/s to ADU/s by dividing by gain>, variance=None, n_combine=0)`. This lets
   `scale_dark(selection, exptime, ccd_temp, *, settings)` stay
   profile-agnostic (it never needs `gain` — the ADU/s conversion already
   happened once, inside `select_dark`, which does take `profile`).

5. **Dark-current map units and where the gain conversion happens.** The
   map is **e-/s/pixel** (`MasterFrame.bunit = "electron/s"`), built in
   `build_master_darks` as: for each (exptime, temp) group,
   `current_e_per_s_group = (group_master.data_ADU * profile.gain_e_per_adu) / group_mean_exptime`
   (gain applied **exactly once**, here); the **composite map is the
   `n_combine`-weighted mean of all groups' per-group current arrays**
   (`np.average(stack, axis=0, weights=n_combines)`), and its reference
   temperature `DCTREF` is the `n_combine`-weighted mean of the *temperatures
   of the groups that have one* (groups with `ccd_temp is None` contribute to
   the composite current array but not to `DCTREF`; `DCTREF` is omitted from
   the header entirely if no group has a temperature — never a sentinel).
   When `select_dark` later needs to turn this e-/s map back into ADU to
   subtract from a raw ADU frame, it converts back **once**, dividing by gain
   again (`map_adu_per_s = dark_current_map / gain`) — this is intentional
   and not a double-application: the map is genuinely stored in e-/s on disk,
   and needs converting to ADU/s only at the point of use.

6. **Arrhenius scaling formula, as implemented** (in `scale_dark`, method
   `"current_map"`):
   ```
   exp_scale   = target_exptime / entry.exptime         # entry.exptime == 1.0 for the map wrapper, so this is just target_exptime
   temp_factor = 2.0 ** ((target_ccd_temp - entry.ccd_temp) / settings.calib.dark_doubling_temp_c)
   scale       = exp_scale * temp_factor
   data        = entry.data * scale
   ```
   `entry.ccd_temp` here is the map's `DCTREF`. If either `target_ccd_temp`
   or `entry.ccd_temp` is `None`, `temp_factor = 1.0` (no correction
   attempted, only exposure-time scaling). Verified numerically: a target
   temperature exactly `dark_doubling_temp_c` warmer than the reference
   doubles the recovered signal (`pytest.approx`, exact by construction of
   the formula).

7. **Grouping tolerance semantics** (in `dark.py::_group_darks`, and it is
   the algorithm `select_dark`'s "exact"/"scaled" tiers also use for
   matching): **greedy, order-preserving, anchor-fixed-at-first-membership**.
   Each group's `anchor_exp`/`anchor_temp` is set by the FIRST frame placed
   into it and never recomputed as a running mean — only the group's
   *reported* `mean_exp`/`mean_temp` (used for the filename and `EXPTIME`/
   `CCDTEMP` cards) is a true mean over all members. A candidate frame joins
   the first existing group whose **anchor** (not running mean) is within
   `settings.calib.dark_exptime_tolerance_s` (exptime) AND
   `settings.calib.dark_temp_tolerance_c` (temperature) — both conditions
   required. `ccd_temp is None` only matches a group whose anchor temp is
   also `None`. Verified: 4 explicit (exptime, temp) combinations × 3 frames
   each → exactly 4 groups at default (0.5/0.5) tolerance, fewer at a loosened
   tolerance; a frame 0.4°C from a group's anchor joins it, one 0.6°C away
   does not (both exercised as literal tests).

8. **Selection ladder order, fixed**: exact (exptime AND temp within
   tolerance, `scale=1.0`) → scaled (temp within tolerance, exptime is not;
   `scale = target_exptime / entry.exptime`, linear, no temperature
   correction since temp already matches) → current_map (temperature-and-
   exposure-corrected fallback, refused with `method="none"` if the target
   temperature is more than `settings.calib.dark_temp_extrap_max_c` from the
   map's `DCTREF`) → none (nothing usable at all: no library entry and no
   map). **Never raises** for any of these four outcomes — `scale_dark` on a
   `method="none"` selection returns a **0-d** `np.zeros((), dtype=np.float32)`
   (deliberately 0-dimensional so it broadcasts safely against a 2-D frame of
   any shape via plain subtraction) plus `DARKFIL='NONE'`/`DARKMETH='none'`
   cards — this is how "warn and fall back to bias-only" is implemented
   without the caller needing to special-case a `None` return.

## Contract deviations from `docs/INTERFACE_CONTRACT.md`

- **None discovered that required deviating from the literal API sketch** in
  the orchestrator's brief for `bias.py`/`dark.py` — everything above is an
  *interpretation* of an underspecified part of the contract (the `settings`
  type; where `DarkSelection.reason` goes; how the current-map fallback is
  represented as a `DarkEntry`), not a violation of anything explicit.
- **ADDENDUM 8 does not apply here and required no change.** It resolves a
  gap in `products/mef.py::calibration_cards` (a function that stamps
  `BIASFIL`/`DARKFIL`/etc. into a *reduced science frame's* header,
  referencing a master's basename + sha256). `MasterFrame.header_cards` in
  this unit is a completely different thing: it's the set of cards written
  **into the master calibration FITS file itself** (`NCOMBINE`, `COMBMETH`,
  `EXPTIME`, `CCDTEMP`, `DARKSCL`, etc.), assembled by `bias.py`/`dark.py`/
  (future) `flat.py`, not by `mef.py`. No overlap, no widening needed. A
  successor should still call `robo43.products.mef.calibration_cards(...)`
  from wherever science frames get dark-subtracted (that's phase 4.4's
  `reduce/frame.py`, not this unit) to stamp `DARKFIL`/`DARKSCL` into the
  *reduced* product — this unit only builds the masters and the
  selection/scaling logic those cards will eventually quote.
- `CalibrationSet` (ADDENDUM 3's sketch) was never implemented in this
  session, so no deviation from it has been decided yet. The planned
  deviations (adding `profile: InstrumentProfile | None = None` and
  `dark_current_ref_temp: float | None = None` fields beyond ADDENDUM 3's
  literal field list, justified by Rule A and by `dark_for`/`scale_dark`
  needing gain access without widening their own signatures) are **only a
  plan** — see "Next steps" for the exact reasoning a successor can adopt or
  revisit.

## Test status

All 17 tests this unit owns pass; 0 skipped; 0 xfail. No `pytest.importorskip`
or `needs_*` markers were needed (everything here is pure numpy/astropy, no
external binaries or services). Full command:
```
cd /home/herpich/Dropbox/robo43-reduction-pipeline
python3 -m pytest tests/test_bias.py tests/test_dark.py -q
# -> 17 passed
```

## Next steps (ordered)

1. **Write `src/robo43/calib/flat.py`** — `select_well_exposed`,
   `normalise_central`, `build_master_flats`. This is the headline fix (bug
   29: dark-subtract every flat frame, scaled to *that frame's own* exptime/
   temp via `dark.py::select_dark`/`scale_dark`, BEFORE the level cut and
   before central-region normalisation). Order of operations per frame:
   overscan → trim → bias-subtract → dark-subtract (per-frame scaling, not
   per-filter-group) → compute a level metric (recommend: central-region
   median, same region as step 2, for consistency and to avoid vignetted
   corners corrupting the level cut too) → `select_well_exposed` (keep
   `flat_min_level_frac*saturation <= level <= flat_max_level_frac*saturation`,
   log every frame's disposition) → `normalise_central` on survivors →
   `combine_stack` → write `master_flat_<FILTER>.fits` (bug 31: literally
   this filename per filter, `settings.runtime.clobber` via
   `MasterFrame.write`, never a shared `master_flat.fits`). Missing `bias`
   or `dark` degrades gracefully (`BIASFIL='NONE'`/`DARKFIL='NONE'`, never
   raises) — reuse the exact same graceful-degradation pattern already
   proven in `dark.py::build_master_darks`'s `bias=None` handling.
2. **Write the bug-29 regression test** in `test_flat.py`: build flats with
   a *deliberately large* dark contribution relative to the illumination
   signal (recommend: dark ADU contribution ≥ ~10% of the flat's own flux,
   e.g. dark=5000 ADU against illum≈30000 ADU, so the shape leakage
   `D/(F+D) × (illum deviation range)` is a clearly measurable few percent,
   not a fraction of a percent), assert the dark-subtracted path recovers
   the injected multiplicative gradient to a tight RMS, and **compute inline
   in the test** (not by calling `build_master_flats` twice) what a
   bias-only path would give using the *same* central-region normalisation
   — isolate bug 29 specifically, don't conflate it with the normalisation-
   region fix. Report both RMS numbers.
3. **Write the central-vs-whole-frame normalisation contrast test**: reuse
   the vignetted-flat fixture pattern from `tests/test_masks.py`
   (`test_dead_pixels_and_vignetting`/`test_normalisation_invariance` —
   `1.0 - 0.4*r_norm`, corners ≈0.6), assert `normalise_central` divides by
   a divisor that makes the *central region* median exactly 1.0, and that
   dividing by the *whole-frame* median (computed inline in the test) would
   not.
4. **Write `src/robo43/calib/set.py`** — `CalibrationSet` exactly as
   ADDENDUM 3 specifies, PLUS the two extension fields reasoned about above
   (`profile`, `dark_current_ref_temp`) so `dark_for` can call
   `dark.py::select_dark` without widening `select_dark`'s own signature.
   `CalibrationSet.load(calibdir, *, settings, profile, filters=None)` reads
   `master_bias.fits`, `master_flat_*.fits`, `master_dark_*.fits` (parse
   `EXPTIME`/`CCDTEMP` **from the FITS header cards**, not by parsing the
   filename — the filename's `:.1f` temperature formatting is lossy and not
   meant to round-trip), and `dark_current_map.fits` (its `DCTREF` card) once
   each. Must be picklable (plain dataclass of numpy arrays / `Path` / dicts —
   already true of every type it references) and `flat_for`/`dark_for`/
   `checksums` implemented per the literal API. **Caution for the test**:
   these dataclasses (`CalibrationSet`, `DarkEntry`, `MaskSet`, `MasterFrame`)
   all use the dataclass-generated `__eq__` by default with numpy-array
   fields — comparing two instances with `==` raises
   `ValueError: truth value of an array is ambiguous`. Test the pickle
   round-trip via explicit per-field comparison (`np.array_equal`, dict key
   equality, etc.), never `loaded == original`.
5. **Write `test_calibset.py`**: build a small bias+dark+flat set to a
   `tmp_path` calibdir using this unit's own `build_master_bias`/
   `build_master_darks` plus the new `build_master_flats`, then
   `CalibrationSet.load` it back and assert round-trip; `flat_for("bogus")
   == (None, None)`; `pickle.dumps`/`loads` round-trip (per point 4's
   caution); `checksums()` returns a sha256 per master via
   `robo43.provenance.sha256_file`.
6. Re-run the full acceptance sequence from the original brief once all four
   files exist: `ruff check src/robo43/calib tests/test_bias.py
   tests/test_dark.py tests/test_flat.py tests/test_calibset.py`,
   `python3 -m pytest tests/test_bias.py tests/test_dark.py tests/test_flat.py
   tests/test_calibset.py -q`, full `python3 -m pytest -q` (must stay ≥140,
   report the new count), `python3 -m mypy src/robo43/calib
   --ignore-missing-imports`.

Nothing here needs `git add`/`commit` — the brief that started this unit
explicitly forbade it, and that has not changed.
