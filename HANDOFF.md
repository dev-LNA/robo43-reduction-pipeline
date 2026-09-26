# Session handoff

Read this at the start of a new or cleared session. Newest entry first.

---

## 2026-09-25 (night) — ROBO43 night 20250911 reduced (WASP-145A); master-dark overscan bug fixed

**Branch:** `main`, uncommitted: `src/robo43/calib/dark.py`, `src/robo43/cli/reduce.py`, `tests/test_dark.py`.

- **Bug fixed:** `build_master_darks` ignored the runtime-detected overscan geometry and trimmed with the
  profile layout, so darks stayed (3211, 4800) while the master bias was (3194, 4788) and `reduce` aborted.
  It now takes `overscan=` (mirroring `build_master_bias`), and `cli/reduce.py` passes `overscan_geometry`.
  This path had never run on real data (T80S has no darks). One mock-based regression test was added.
- **Raw headers:** every frame of this night has `IMAGETYP='LIGHT'`. The bias/dark/flat frames were copied to
  `$O/raw_calib/` and fixed there with `robo43 fixheader` (raw data untouched); the science frames are read
  from the raw directory.
- Output `O=/ssdsto1/data/ROBO43_reduced/20250911`: 351 WASP-145A R-Bessell 30 s frames, masters in `$O/calib`,
  input list `$O/inputs_all.txt`. Measured gain 0.7056 e-/ADU, RN 6.95 e- (matches the profile note).
  CRFRAC ~0.17% (the README's 31.5% TODO no longer applies).
  Reduce 19 min (`--np 8`), astrometry 351/351 (RMS ~0.23"), photometry 351/351 in 13 min.

```
robo43 reduce --instrument robo43 --clobber --np 8 --outdir $O --calibdir $O/calib --logfile $O/reduce.log $(cat $O/inputs_all.txt)
robo43 astrometry --instrument robo43 --index-dir ~/Documents/.astrometry --in-place --max-sources 400 --np 8 --logfile $O/astrometry.log $O/*_proc.fits
robo43 photometry --instrument robo43 --in-place --catalog-csv --np 8 --logfile $O/photometry.log $O/*_proc.fits
```

### Open items

1. **Excess frame-to-frame noise in the reduced photometry.** Clean (FLAGS==0) unsaturated bright-star pairs
   scatter 7-12 ppt per 30 s frame against 2.3-2.8 ppt expected, in every aperture up to 8". Not cosmic rays
   (<1% of CR pixels are near stars) and not linear in seeing or x/y. The field drifts ~200 x 150 px over 3 h
   with 1.7 px frame-to-frame jitter (no autoguiding?) and stellar cores look boxy (image motion during the
   exposure). Suspects: flat-field error x drift, tracking glitches, scintillation well above Young's ~1.2 ppt.
2. Commit the dark fix after review.
3. `fixheader` writes into the raw directory; there is no `--outdir` option, hence the copy step above.

---

## 2026-09-25 (evening) — test warnings removed; phase 1B (inventory/inspect/fixheader) landed

**Branch:** `fix-imagetyp-headers` (uncommitted, on top of the Gaia ZP changes below). **Suite:** 487
passed, 0 warnings, 0 skips (`python3 -m pytest -o addopts="" -q -rs`). `ruff check` is clean on all
new files.

### Warnings (100 → 0)

- Header comments that were too long were shortened: `MEDGAIN` (`reduce/frame.py`), `FLATNGT`
  (`calib/flat.py`), `GAINRN` (`calib/gain.py`), `TRIMREG` (`calib/overscan.py`).
- Test fixtures wrote the non-standard `READNOISE`; they now write `RDNOISE` (the profile alias).
- photutils `NoDetectionsWarning` (an expected, already-handled empty-frame case) is suppressed locally
  in `astrometry/centroids.py` and `photometry/catalog.py`.
- astropy `FITSFixedWarning` ('datfix' filling MJD-OBS) is suppressed around each `WCS(...)`
  construction: `io/headers.py`, `astrometry/solve.py`, `products/nominal_wcs.py`, `cli/photometry.py`
  and `tests/test_cli_astrometry.py`.

### Phase 1B

The resume note `docs/refactor-resume/phase1b-inventory-cli.md` was stale. Where it disagreed with the
current code or with `docs/INTERFACE_CONTRACT.md`, the contract won:

- **No packaged `synth2.toml`** (ADDENDUM 9). Instead, `resolve_profile` / `--instrument` accept a
  path to a TOML file (`instruments/profile.py`).
- **Pre-existing bug fixed:** `io/validate.py` referenced a nonexistent `FrameMeta.gain`, so
  `inspect --stages photometry` raised `AttributeError`. It is now `gain_setting`. The override key is
  `gain_setting`; there is deliberately **no `gain=` alias**, since GAIN is a QHY setting index, not
  e-/ADU.
- **FOCUS added** to classification (`instruments/keywords.py`, `io/headers.py`). Focus frames
  previously fell through to science.

New code:

- `io/inventory.py`: `build_inventory`, `metas_from_inventory`, `write_inventory` and
  `filter_inventory`. There is an `imagetyp` column in the canonical vocabulary and a `read_error`
  column. In auto mode each file is read exactly once. The worker pool uses `spawn`. `filter=` never
  matches ZERO/DARK rows. `valid_only` drops only frames whose header says `VALID = False` and always
  keeps `read_error` rows.
- `cli/inventory.py`: the `--frame-type`, `--filter`, `--object` and `--include-invalid` options.
  It logs how many `VALID = False` frames were excluded.
- `cli/inspect.py`: each stage is checked only against the frame types it applies to
  (`_STAGE_APPLIES_TO`, keyed on ZERO/DARK/FLAT/OBJECT). FOCUS is never checked by science stages.
  `--emit-overrides` writes a template.
- `cli/fixheader.py`: derives IMAGETYP from OBJECT only (a wrong but mappable existing value such as
  "Light Frame" is still repaired). It never touches FILTER on ZERO/DARK. It writes a `.bak` backup
  before writing, plus a HISTORY card. `--dry-run` is a true no-op and reports "would change (dry-run)".
- Tests: `tests/conftest.py`, `test_inventory.py`, `test_cli_inventory.py`, `test_cli_inspect.py`,
  `test_cli_fixheader.py`, `test_headers_classify_focus.py`, `test_validate_gain_field.py` and
  `test_profile_resolve_path.py`. Real `mini_sample` frames are used where they exist (the tests skip
  when absent); synthetic data is used only for the MEF layout and the stripped-keyword night.

Checked by hand on copies of the real `mini_sample`: the inventory gives ZERO/FLAT/OBJECT;
`--filter U-Bessell` returns 0 rows (only biases carry it); the `BAD_…` frame (`VALID = False`) is
excluded with a log line; a truncated file lands as a `read_error` row. `fixheader` repaired
"Light Frame" to FLAT, with backup and HISTORY, and the dry run left the sha256 unchanged.

### Open items

1. **Commit** everything once reviewed: the Gaia ZP files from the entry below, the warning fixes and
   phase 1B.
2. ~~Delete the legacy `src/fix_header.py`~~ — done (deleted, superseded by `robo43 fixheader`).
3. `generic.toml` cannot derive `science_hdu` for a foreign MEF instrument (ADDENDUM 9's second
   finding). It needs a fix and a test.
4. Full unit suites for the already-landed `io/headers.py`, `io/meta.py`, `io/validate.py`,
   `io/fits_io.py` and `instruments/profile.py` (next step 8 of the old resume note) were not written.
   Only targeted regression tests exist.
5. Next in dependency order (`docs/refactor-resume/README.md`): phase 4.3 remainder, then the rest.
   Check the README against the current code first, because it is as stale as the phase 1B note was.
6. The open items from the entries below still stand.

---

## 2026-09-25 (later) — Gaia ZP spatial gradient diagnosed and fixed

**Branch:** `fix-imagetyp-headers` (4 files modified, **uncommitted**). **Suite:** 434 passed
(`python3 -m pytest -o addopts="" -q`); `ruff check` clean on the changed files.

### Diagnosis (3 T80S R frames of 20251104, ~4600 stars with REFMAG < 16)

The 0.13–0.15 mag `SPZPRNG` had three causes:

- **Kron-aperture/PSF bias (main cause).** FWHM varies smoothly across the field (focus/coma).
  The `MAG_AUTO` residual tracks FWHM (r = 0.45); with a fixed aperture the correlation is
  about 0.
- **A planar tilt fixed on the detector** (bottom-left +0.10 to +0.15, top-right about −0.10;
  identical in the dithered frames). **T80S has only twilight sky flats — dome flats will never
  exist.** The tilt is therefore most likely a twilight-sky gradient imprinted in the sky flat.
- **Colour term:** about +0.056 mag per mag of BP−RP.
- Ruled out: crowding (r = 0.07) and mismatches (median separation 0.15").

Analysis scripts and plots are in the session scratchpad (`zpgrad/`); that directory does not
persist.

### What changed

`photometry/gaia_zp.py`, `config.py`, `data/config.default.toml`, `tests/test_gaia_zp.py`.

- The ZP is fitted as REFMAG − MAG = a + b·(BP−RP − c0) + S(x,y). This is `fit_zero_point_model`:
  iterative OLS with sigma clipping, and S is a polynomial in detector coordinates normalised
  to the centre.
- New settings `gaia_zp_aperture_arcsec = 4.0` (0.0 = old `MAG_AUTO`),
  `gaia_zp_color_term = true` and `gaia_zp_surface_order = 1` (0 reproduces the old scalar fit
  exactly).
- `ZPABS` is the value at the detector centre and colour c0. The RMS gate uses the residuals
  after the full model.
- `MAG_CALIB` = the fitted magnitude + a + S(x,y) + b·(BP−RP − c0), with the colour term set
  to 0 for stars without Gaia BP−RP.
- New header cards `ZPAPER`, `ZPSURFO`, `ZPCOLB`, `ZPCOLC0`, `ZPSFnn` and `SPZPRES`
  (the range after the full model). `SPZPRNG` and `ZPTnn` still describe the raw diffs
  (the gradient diagnostic).

Aperture choice (pooled residual MAD after the full model): 2" 0.070, 3" 0.037,
**4" 0.033**, MAG_AUTO 0.044.

| Frame | ZPABS old→new | RMS old→new | SPZPRES | ZPABSCAL |
|---|---|---|---|---|
| 030335 | 28.120→28.108 | 0.079→0.045 | 0.044 | False→True |
| 034524 | 28.139→28.093 | 0.082→0.042 | 0.048 | False→True |
| 043150 | 28.143→28.142 | 0.077→0.042 | 0.046 | False→True |

The colour slope and surface coefficients agree frame to frame
(b ≈ 0.056, (sx, sy) ≈ (−0.024, −0.042)).

### Open items

1. **Commit** the four modified files once reviewed.
2. **`MAG_APER_CAL` applies the single 4"-anchored correction to every aperture column**, so the
   2/3/6/8" calibrated columns are off by their aperture correction relative to 4". This
   predates the change (the old code anchored to `MAG_AUTO`). Fix it with a per-aperture
   correction, or write `MAG_APER_CAL` only for the fitted aperture.
3. **Sky flats labelled `_dome`.** `calib/flat.py:62` classifies a flat as sky only when `OBJECT`
   contains "sky" or "twilight"; T80S marks twilight flats via `IMAGETYP = 'sky-flat'`, so its
   master is written as `master_flat_R_dome.fits`. This is a naming issue only (same pixels).
   Fix the classifier, for example by checking IMAGETYP or adding a per-instrument default
   flat kind of "sky".
4. The planar tilt is absorbed per frame by the surface. A permanent illumination/star-flat
   correction built from these residuals would remove it at the reduction stage instead.
5. `gaia_zp_surface_order = 2` is covered only by the unit tests and has not been run on real
   data. The XP path is still untested live, and the full 45-frame `--gaia-zp` run is still
   pending.
6. The items from the entry below still stand (`select_reference_stars` guard, T80S linearity,
   doubled log timestamps, ruff findings elsewhere).

---

## 2026-09-25 — Gaia zero point (`--gaia-zp`) fixed and tested on T80S

**Branch:** `fix-imagetyp-headers`. **Suite:** 428 passed (`python3 -m pytest -o addopts="" -q`
prints the summary line; the repo's `-q` addopts hides it otherwise).

### Why `--gaia-zp` hung

- The Gaia cone query had no magnitude limit. The T80S LMC cone (radius 1.06°) holds about 1.6M
  Gaia sources. The ESA archive never finishes that query, even with a G < 16 limit
  (still EXECUTING after 8 min).
- The XP step then asked `gaiaxpy` for spectra on every source in the cone.

### What changed

All in `photometry/gaia_zp.py`; the new settings are in `config.py` and `config.default.toml`.

- The cone query uses **VizieR `I/355/gaiadr3`** by default (`gaia_catalog = "vizier"`), with
  the ESA archive as fallback. It is limited to `gaia_gmax = 17.0` and has a `gaia_query_timeout_s`
  timeout. The cache key includes gmax. If both services fail, the frame is marked
  "unavailable" and nothing raises.
- XP synthetic photometry is requested only for the `gaia_xp_max_stars = 2000` brightest
  cross-matched `FLAGS==0` stars, with `gaia_xp_timeout_s`. Matching now runs before
  reference-magnitude resolution. On failure it falls back to the G/BP/RP colour transform.
- **ZP semantics bug fixed.** `MAG_AUTO` already includes `instrumental_zp` (20), so
  median(REFMAG − MAG_AUTO) is an *offset*. `ZPABS`/`zp_abs` now report offset + instrumental_zp,
  and `MAG_CALIB = MAG_AUTO + offset`. Previously ZPABS read about 8.1, and `MAG_CALIB` would have
  been 20 mag off whenever a frame passed the RMS gate.
- Bright-star cut: the ZP fit, the per-tile offsets and the spatial range use only stars with
  REFMAG < `gaia_zp_refmag_max = 16.0`. Fainter stars in this field scatter 0.13–0.18 mag and
  carry a blending bias of about +0.06 mag. REFMAG is still written for every match.

### Test on 3 frames of T80S 20251104 (copies; production outputs untouched)

| Frame | ZPABS | N | RMS | SPZPRNG |
|---|---|---|---|---|
| 030335 | 28.120 | 1444 | 0.079 | 0.134 |
| 034524 | 28.139 | 1551 | 0.082 | 0.126 |
| 043150 | 28.143 | 1284 | 0.077 | 0.145 |

About 3–4 min per frame, of which the VizieR query takes about 50 s. `ZPABSSRC = gaia_gbrp_sdss_r`,
because the ESA archive returned 503 during testing, so the XP path fell back.
`ZPABSCAL = False` on all three: RMS 0.08 > `zp_rms_max = 0.05`, so no `MAG_CALIB` is written.

### Open items

1. **ZP spatial gradient of 0.13–0.15 mag** across the T80S field. It dominates the RMS.
   Check whether it is a flat-field or illumination residual or a colour term. The options are
   to fix it, to raise `zp_rms_max` to about 0.10 for T80S, or to accept `ZPABSCAL = False`
   (Gaia is only a consistency check).
2. **XP path untested live.** The ESA archive returned 503 throughout. Retry once it is up
   and compare against the colour-transform ZP.
3. `--gaia-zp` has not yet been run on the full 45-frame night.
4. `calibrate_frame` does not guard `select_reference_stars` with a try/except (this
   predates the change).
5. The earlier open items below still stand: T80S linearity, the doubled log timestamps and the
   ruff findings.

---

## 2026-09-24 — saturation flagging fixed, reduce parallelised, T80S night re-reduced

**Branch:** `fix-imagetyp-headers`. **Suite:** 425 passed (`python3 -m pytest -q`).
`ruff check src/robo43 tests` reports 50 findings (mostly unused arguments,
import order, long lines). These existed before this session and have not been fixed.

### What changed

- **Saturation and non-linearity are now flagged.** The old mask compared
  bias-subtracted ADU with `saturation_adu`. Once the ~1000 ADU bias was
  removed, no pixel could ever reach 65535, so `SATURATED` never fired. When a
  master bad-pixel mask was present, saturation was not evaluated at all.
  `reduce/frame.py` now tests the **raw** ADC read, before overscan, trim,
  bias and dark. `DQ.SATURATED` is set at `>= saturation_adu`; `DQ.NONLINEAR`
  is set at `>= nonlinearity_adu`. The masks are trimmed with the new
  `calib.overscan.trim_to_science` (same block geometry as `apply_overscan`),
  and they are OR'd into the DQ plane in every branch.
- `photometry/catalog.py`: the new `FLAG_NONLINEAR = 64` comes from `DQ.NONLINEAR`
  inside the source footprint, alongside `FLAG_SATURATED = 4`. Both keep
  stars out of the `FLAGS==0` reference sample.
- `t80s.toml`: `nonlinearity_adu = 45000` (provisional, raw ADU including bias).
- **`robo43 reduce --np N`** reduces science frames in parallel. It uses the
  `spawn` context (fork is unsafe with `ProcessPoolExecutor`'s own threads) and a
  per-worker initializer that loads the `CalibrationSet` once. Change-log appends
  are serialised by a lock. The worker count is capped by
  `safe_worker_count` (48 B/px, `reduce.science`). `--np 1` keeps the old serial path.
  `tests/test_cli_reduce.py::test_reduce_parallel_matches_serial` covers it.

### Re-reduction of T80S night 20251104 (45 R-band frames, 90 s, LMC field)

Output: `/ssdsto1/data/T80S_reduced/20251104`; previous runs are in `superseded/`
(run5 = the run before this fix). Inputs list and config override:
`inputs_20251104.txt` and `t80s_20251104.toml` (`[calib] flat_min_level_frac = 0.15`).
They lived in the session scratchpad, so re-create them if needed: the inputs are
the raw 20251104 science, bias and flat frames.

```
robo43 reduce --instrument t80s --config t80s_20251104.toml --clobber --np 8 \
  --outdir $O --calibdir $O/calib --logfile $O/reduce.log $(cat inputs_20251104.txt)
robo43 astrometry --instrument t80s --index-dir ~/Documents/.astrometry --in-place \
  --max-sources 400 --rms-max 0.5 --np 6 --logfile $O/astrometry.log $O/*_proc.fits
robo43 photometry --instrument t80s --in-place --catalog-csv --np 4 \
  --logfile $O/photometry.log $O/*_proc.fits
```

- Reduce: 10 min with 8 workers (previously ~28 min serial); 45/45 are "degraded" only because
  there are no darks. Astrometry 45/45 solved. Photometry 45/45.
- Flags on frame TOO-20251105-030335: 4645 SATURATED px and 14709 NONLINEAR px. Every star
  brighter than instrumental mag 5 is flagged; before the fix, many carried no flag.
- Reliability (4" aperture, ensemble relative photometry): RMS is 1.25–1.46x the
  predicted error at mags 5.5–10.5; the best stars reach 2.1–3.0 mmag per 90 s exposure. The earlier
  1.7–2.5 mmag included mag 5–5.5 stars that are now flagged non-linear.
  Extinction k ≈ 0.17 mag/airmass; the ZP residual tracks seeing (r = 0.97).

### Open items

1. **Measure T80S linearity** and replace the provisional `nonlinearity_adu = 45000`
   and `saturation_adu = 65535`. If 45000 turns out too strict, mag 5–5.5 stars come back.
2. **Gaia zero point (`--gaia-zp`)**: the CLI flag exists, but an earlier run with it
   hung. The fix from the previous session was never verified, so this night's photometry
   ran without it. Test on a few frames before using it at scale.
3. Worker log lines from the parallel reduce show a doubled timestamp prefix
   (cosmetic; `worker_log_config` / the listener formatter both format).
4. Scatter is 1.3–1.5x the predicted error, likely from crowding in the LMC field. For transits, use
   FWHM-scaled apertures and nearby comparison stars.
5. Ruff findings (see above).
