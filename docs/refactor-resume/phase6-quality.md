# Phase 6 resume note — `robo43/photometry/quality.py`

Written at a checkpoint pause. Readable with no memory of the working session
that produced it.

## Unit

Phase 6 (plan §6, ADDENDUM 4 of `FOUNDATION_CONTRACT.md`) — the configurable,
per-source quality-rejection chain (goal 4 of the refactor): replaces the
crude, buggy filtering in `process_robo43_frames.py` with an auditable,
config-driven `Reject` bitmask retained as a column, never destructively
applied.

## Status: COMPLETE for this unit

Both owned files are finished, lint-clean, type-clean, and fully tested.

## Files complete and passing

- `src/robo43/photometry/quality.py` — the full module: `Reject` (IntFlag),
  `QualityReport` (dataclass), `assess()`, `reference_sample()`, plus private
  helpers `_check_required_columns`, `_saturation_level`, `_sky_columns`.
  - `ruff check src/robo43/photometry/quality.py` — clean.
  - `python3 -m mypy src/robo43/photometry/quality.py --ignore-missing-imports`
    — `Success: no issues found in 1 source file`.
- `tests/test_quality.py` — 16 test functions, all passing, covering every
  rejection class individually, the saturation-unit-convention equivalence,
  blending (including a 20000-source performance/correctness check), the
  absolute FWHM window at the legacy pixel scale, Gaia cross-matching and the
  spurious-detection rate, the two-bits-in-one-row regression test for the
  `=`-vs-`&=` legacy bug, the "excellent catalogue rejects nobody" headline
  contrast, `ELLIPTICITY`-vs-`ELONGATION` agreement, skip-vs-clean honesty,
  empty-table/missing-column handling, `reference_sample`, and a non-square
  frame edge/transposition guard.
  - `ruff check tests/test_quality.py` — clean.
  - `python3 -m pytest tests/test_quality.py -q` — `16 passed`.

Full-repo verification at checkpoint time (from repo root):
```
$ python3 -m pytest --collect-only -q 2>&1 | tail -5
tests/test_masks.py: 20
tests/test_mef.py: 9
tests/test_packaging.py: 9
tests/test_quality.py: 16

$ python3 -m pytest 2>&1 | tail -5
........................................................................ [ 58%]
...................................................                      [100%]
123 passed in 0.64s
```
All 123 tests in the whole suite pass, including this unit's 16 and every
other already-landed module's tests. Nothing is broken; the repo is fully
collectable and green.

## Files started but incomplete

None. No other file was created or partially edited by this unit. Only the
two files above were touched (plus this resume note and, transiently, nothing
else). `REFACTORING_PLAN.md` and `FOUNDATION_CONTRACT.md` were read only, not
modified.

## Rejection steps implemented vs not — all ten (really twelve `Reject`
members) implemented and tested

| `Reject` member | Implemented | Tested |
|---|---|---|
| `FLAGS` | yes — `FLAGS != 0` when `stellar=True`, `FLAGS > 3` when `stellar=False` | yes (`test_one_row_per_class`, and mixed with `NOT_STELLAR` in the two-bit regression test) |
| `BADPIX` | yes — `IMAFLAGS_ISO != 0`, skipped when column absent | yes (row test; skip-honesty test) |
| `SATURATED` | yes — `FLUX_MAX > saturation_fraction * sat_level` in the data's own units | yes, isolated in its own `assess()` call with `nonlinearity_adu=None` so it doesn't co-trip `NONLINEAR`; also the unit-convention-equivalence test |
| `NONLINEAR` | yes — `FLUX_MAX > nonlinearity_fraction * nonlin_level`, independent bit from `SATURATED` | yes, isolated with a `FLUX_MAX` between the two thresholds |
| `EDGE` | yes — `margin = max(edge_margin_fwhm * frame_median_fwhm_px, edge_margin_px)`, `X_IMAGE` vs `naxis1`, `Y_IMAGE` vs `naxis2` (median FWHM computed once for the whole frame, not per-source) | yes (row test; dedicated non-square `naxis1=200, naxis2=100` transposition-guard test) |
| `FWHM_LOW` | yes — absolute `fwhm_arcsec = FWHM_IMAGE * meta.pixel_scale < fwhm_min_arcsec`, skipped when `pixel_scale is None` | yes (row test; dedicated absolute-window test at the legacy `pixel_scale=0.529`) |
| `FWHM_HIGH` | yes — same mechanism, `> fwhm_max_arcsec` | yes (same two tests) |
| `ELONGATED` | yes — prefers an `ELLIPTICITY` column, else derives `ellipticity = 1 - 1/ELONGATION` from `ELONGATION`, skipped if neither present | yes (row test; dedicated `ELLIPTICITY`-only vs `ELONGATION`-only agreement test) |
| `LOW_SNR` | yes — `FLUX_AUTO/FLUXERR_AUTO < snr_min`, with `FLUXERR_AUTO <= 0` also tripping the bit (guarded, no div-by-zero warnings) | yes (row test) |
| `NOT_STELLAR` | yes — only evaluated when `stellar=True`; prefers `SPREAD_MODEL` (`abs(SPREAD_MODEL) > 0.002`, a hardcoded SExtractor/PSFEx convention constant, NOT a config field) over `CLASS_STAR` (`< class_star_min`); skipped (not silently off) only when both columns are absent AND `stellar=True` — when `stellar=False` it is simply not evaluated and is NOT added to `skipped` (intentional, not broken) | yes (row test; two-bit regression test; skip-honesty test) |
| `BLENDED` | yes — `scipy.spatial.cKDTree` nearest-neighbour query (`k=2`), O(N log N), radius = `isolation_radius_arcsec / pixel_scale` in px, skipped when `pixel_scale is None` | yes (known close pair + isolated source; 20000-source performance test, empirically verified zero accidental collisions at the chosen density) |
| `NO_GAIA_MATCH` | yes — only when `reference` table supplied; sky coordinates from `ALPHA_J2000`/`DELTA_J2000`, `RA`/`DEC`, `ra`/`dec`, or `ra_deg`/`dec_deg` columns on either table (in that preference order), or via `wcs.pixel_to_world` on the source table if no sky columns present; `astropy.coordinates.SkyCoord.match_to_catalog_sky`; match radius is a **module constant** `_GAIA_MATCH_RADIUS_ARCSEC = 2.0`, NOT a `PhotometrySettings` field (flagged below) | yes (dedicated test with a 4-row table, 2 matched + 2 unmatched, asserting `spurious_rate == 0.5`) |

## Test status

All 16 tests in `tests/test_quality.py` pass. No skips, no xfails, nothing
owed. Full-suite run (123 tests across the whole repo) is green.

## Decisions made — a successor must not silently reverse these

1. **`Reject` bit numbers** (verbatim from `FOUNDATION_CONTRACT.md` ADDENDUM 4,
   confirmed by direct enum inspection): `NONE=0, FLAGS=1<<0, BADPIX=1<<1,
   SATURATED=1<<2, NONLINEAR=1<<3, EDGE=1<<4, FWHM_LOW=1<<5, FWHM_HIGH=1<<6,
   ELONGATED=1<<7, LOW_SNR=1<<8, NOT_STELLAR=1<<9, BLENDED=1<<10,
   NO_GAIA_MATCH=1<<11`. These reach archived catalogues — never renumber.
2. **Saturation units**: `profile.saturation_adu` / `profile.nonlinearity_adu`
   and `meta.saturation` are ALWAYS in ADU (verified against
   `src/robo43/io/headers.py` — `meta.saturation` is populated straight from
   the header or from `profile.saturation_adu`, never gain-converted). The
   catalogue's `FLUX_MAX` column is in the data's own units: electrons when
   `settings.calib.apply_gain` (the pipeline default) is `True`, ADU
   otherwise. The single conversion point is the private helper
   `_saturation_level(meta, profile, settings, *, level_adu)` in
   `quality.py`: it returns `level_adu` unchanged when `apply_gain` is
   `False`, or `level_adu * gain` (gain from `meta.gain`, falling back to
   `profile.gain_e_per_adu`) when `apply_gain` is `True`; returns `None`
   (meaning "skip the cut") when the level or, in the gain-applied case, the
   gain itself, cannot be determined. Called independently for `SATURATED`
   (level = `meta.saturation` or `profile.saturation_adu`) and `NONLINEAR`
   (level = `profile.nonlinearity_adu`, no per-frame override exists), so one
   can be skipped while the other still runs.
3. **Skipped vs "0 rejected"**: a cut that could not run (missing optional
   column, `pixel_scale is None`, `reference is None`, saturation level
   undeterminable, etc.) has its `Reject` member NAME placed in
   `QualityReport.skipped` and its name is **absent** as a key from
   `QualityReport.counts` (both the bare name and the `<name>_new` variant).
   A cut that ran and genuinely rejected nothing has its name present in
   `counts` with value `0`. These two states are only distinguishable by
   checking `skipped` — this is exactly the honesty requirement from the
   brief, and there is a dedicated test (`test_skipped_cuts_are_recorded...`)
   proving it. Exception: `NOT_STELLAR` when `stellar=False` is intentionally
   not evaluated and is **not** added to `skipped` either (it's off by
   request, not a missing-data skip) — a documented, deliberate special case.
4. **`counts` dict shape**: for every cut that ran, two keys:
   `counts["<NAME>"]` = total sources with that bit set (feeds
   `metric_cards()`'s `NSATUR`/`NBADPIX`), and `counts["<NAME>_new"]` =
   newly-rejected at that step in fixed rule order 1–11 (a source is "newly
   rejected" only the first time, at the first rule in order that flags it —
   `sum(*_new) + n_clean == n_input` holds exactly, verified in
   `test_one_row_per_class`).
5. **Required vs optional columns**: required (raise `PhotometryError` naming
   the missing column, checked before any row processing, even on an empty
   table): `X_IMAGE`, `Y_IMAGE`, `FLUX_AUTO`, `FLUXERR_AUTO`, `FLAGS`,
   `FWHM_IMAGE`. Optional (their cut skips, recorded per point 3 above):
   `IMAFLAGS_ISO`, `FLUX_MAX` (shared by `SATURATED`+`NONLINEAR`),
   `ELLIPTICITY`/`ELONGATION` (either satisfies `ELONGATED`),
   `CLASS_STAR`/`SPREAD_MODEL` (either satisfies `NOT_STELLAR`, only relevant
   when `stellar=True`), `reference` (a caller-supplied `Table`, satisfies
   `NO_GAIA_MATCH`).
6. **`clean_index(*, allow=Reject.NONE)`**: `np.flatnonzero((mask &
   ~int(allow)) == 0)` — a source is clean relative to `allow` if every bit
   set on it is among the allowed bits. `QualityReport.n_clean` is always
   computed with the default `allow=Reject.NONE`.
7. **`text_summary()`** — an additive convenience method on `QualityReport`
   (not part of the fixed contract, no exact-format test) added specifically
   to satisfy the orchestrator's acceptance-criterion instruction to "paste
   `report.text_summary()`". Safe to change formatting freely; nothing else
   depends on its exact text.

## Findings about the legacy code

Confirmed directly by reading `src/process_robo43_frames.py:627-634`:
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
Yes — the third executable `mask = ...` line (the `FLUX_AUTO` percentile cut)
uses plain `=`, not `&=`, silently discarding the `FLAGS` and `FWHM_IMAGE`
cuts computed just above it. It does not compound with them; it replaces
them. Also `FLAGS < 4` wrongly admits SExtractor flag values 1 (blended) and 2
(deblended) as if they were clean, and the surviving 80th-percentile
`FLUX_AUTO` cut discards 80% of every field regardless of actual source
quality — a relative, data-dependent threshold, not a physically motivated
one. `quality.py`'s headline test (`test_excellent_catalogue_rejects_nobody`)
demonstrates the fix directly: 50/50 synthetically excellent sources survive
the new absolute-threshold chain, where the legacy code would have kept only
the top 20% (10 of 50) regardless of quality.

## Contract deviations from `FOUNDATION_CONTRACT.md` ADDENDUM 4

1. **`assess()` gained one new keyword-only parameter: `stellar: bool =
   True`.** The contract's signature (`assess(table, *, settings, profile,
   meta, dq=None, wcs=None, reference=None) -> QualityReport`) has no way to
   express the "reference/astrometry source selection" (strict, `FLAGS==0`,
   `NOT_STELLAR` active) vs "science catalogue" (looser, `FLAGS<=3`,
   `NOT_STELLAR` off) distinction that the brief explicitly requires ("Drive
   this from the stellar/reference distinction, not a magic number inline";
   "applied only when `stellar=True`"). Adding a new keyword-only parameter
   with a default is backward compatible with any caller coding strictly
   against the documented signature. Default `True` (the stricter/safer
   behaviour) was chosen deliberately so an unmodified call matches
   "reference/astrometry" semantics; a science-catalogue caller must pass
   `stellar=False` explicitly. **Any other module calling `assess()` needs to
   know about this parameter and pass it deliberately** — this is the single
   most important thing a successor integrating `sextractor.py` or the
   reduce/product pipeline against this module needs to know.
2. **`QualityReport` gained fields beyond the contract's abbreviated sketch**:
   `skipped`, `median_fwhm_px`, `median_fwhm_arcsec`, `median_ellipticity`,
   `spurious_rate`, plus the `text_summary()` method. This follows the
   established pattern in this codebase (Rule A in the contract: dataclasses
   grow fields beyond the sketch, always constructed/read by keyword) — `mask`,
   `counts`, `n_input`, `n_clean`, `clean_index()`, `metric_cards()` all match
   the contract exactly.
3. **`dq` and `wcs` parameters are accepted per the contract's signature but
   `dq` is currently unused by every rule** (reserved for a future per-pixel
   DQ-plane check that isn't wired in — BADPIX currently only reads the
   catalogue's `IMAFLAGS_ISO` column, per the brief's explicit instruction).
   `wcs` IS used, opportunistically, by `NO_GAIA_MATCH` when the table lacks
   its own sky-coordinate columns.
4. **`_GAIA_MATCH_RADIUS_ARCSEC = 2.0`** is a module-level constant, not a
   `PhotometrySettings` field. There is no config field for a Gaia match
   tolerance anywhere in `PhotometrySettings` as currently defined. Flagged as
   a plausible missing config field — a successor wiring this into
   `zeropoint.py` or the CLI may want to promote it to
   `settings.photometry.gaia_match_radius_arcsec` with a matching default.
5. **`_SPREAD_MODEL_STELLAR_LIMIT = 0.002`** is likewise a hardcoded module
   constant (the standard SExtractor/PSFEx point-source convention), not a
   config field — `PhotometrySettings` has no `spread_model_max` equivalent.
6. Everything else (the 12 `Reject` bit values, `QualityReport`'s contract
   fields, `assess`'s other parameters, `reference_sample`'s signature) was
   implemented exactly as written in ADDENDUM 4 — no other deviations.

## Next steps

1. Wire `assess()` into whichever module ends up calling SExtractor
   (`robo43/photometry/sextractor.py`, per ADDENDUM 4 — NOT started as part of
   this unit and explicitly out of scope for `quality.py`) — the caller must
   decide and pass `stellar=` explicitly per call site (astrometry source
   selection vs science catalogue) per deviation #1 above.
2. `products/catalog.py`'s `build_catalog_hdu()` should consume
   `QualityReport.metric_cards()` for the `NSOURCES/NCLEAN/NREJECT/NSATUR/
   NBADPIX` header keywords, and `median_fwhm_px`/`median_fwhm_arcsec`/
   `median_ellipticity` for `FWHMPIX`/`FWHMARC`/`ELLIP`, and
   `spurious_rate` wherever the plan's per-frame quality metrics land in the
   database (`reduced` schema, phase 9) — none of that wiring exists yet,
   `quality.py` only produces the numbers.
3. Consider promoting `_GAIA_MATCH_RADIUS_ARCSEC` and
   `_SPREAD_MODEL_STELLAR_LIMIT` to real `PhotometrySettings` fields if a
   config-driven override turns out to be needed (currently hardcoded, see
   deviations #4/#5).
4. No known bugs, no failing tests, no TODOs left inside `quality.py` itself.
