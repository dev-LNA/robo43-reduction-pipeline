# Session handoff

Read this at the start of a new or cleared session. Newest entry first.

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
