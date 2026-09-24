# Session handoff

Read this at the start of a new or cleared session. Newest entry first.

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
