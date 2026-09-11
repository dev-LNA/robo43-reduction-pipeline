# Phase 4.4 / 4.4.1 resume note — product model (`products/mef.py`, `products/catalog.py`)

Written at a session checkpoint. Read this before touching either file again.

## Unit

Refactoring plan §4.4 / §4.4.1: the one-file-per-reduced-frame multi-extension
FITS (MEF) container (`SCI`/`ERR`/`DQ`/`CATALOG`/`CATFLAGS`), plus the
`CATALOG` HDU's self-describing header. Spec source of truth:
`FOUNDATION_CONTRACT.md` ADDENDUM 3 (this file lived only in the scratchpad
directory of the session that did this work, not in the repo — if it's not on
disk anymore, ask the orchestrator for it again before changing behavior).

## Status: COMPLETE, not committed

Both files are finished, independently verified, and all required tests
pass. Nothing was left mid-edit. No `git add`/`commit` has been run (the
brief forbade it) — everything here is working-tree-only.

## Files complete and passing

- `src/robo43/products/mef.py` — `ReducedProduct` (`to_hdulist`,
  `from_hdulist`, `write`, `read`) and `calibration_cards`. Covered by
  `tests/test_mef.py` (9 tests, all pass).
- `src/robo43/products/catalog.py` — `CATALOG_COLUMNS`, `CatalogMetrics`
  (`to_header`/`from_header`), `build_catalog_hdu`, `read_catalog`,
  `write_ldac`, `extract_catalog`. Covered by `tests/test_catalog.py`
  (12 tests, all pass).

Verified independently (not just trusting the executer reports) with, from
`/home/herpich/Dropbox/robo43-reduction-pipeline`:

```
ruff check src/robo43/products tests/test_mef.py tests/test_catalog.py
  -> All checks passed!
python3 -m pytest tests/test_mef.py tests/test_catalog.py -q
  -> 21 passed
python3 -m mypy src/robo43/products/mef.py src/robo43/products/catalog.py --ignore-missing-imports
  -> 2 errors, BOTH in src/robo43/products/dq.py (lines 107, 119; a file we
     do not own or edit, mypy reports them only because it follows the
     import). Zero errors attributable to mef.py or catalog.py themselves
     (confirmed by running mypy on dq.py alone with the same two errors
     appearing in isolation).
```

## Files started but incomplete

None. Nothing under `products/` from this unit was left partially written.

## Coverage of the header keyword set

### Appendix B.1 (`SCI` primary header, via `calibration_cards`)

Implemented: `PIPEVER PIPEGIT PIPECFG PIPEDATE` (delegated verbatim to
`robo43.provenance.provenance_cards`, not reimplemented), `INSTPROF INSTPVER`
(from `profile.name`/`profile.version`), `BIASFIL DARKFIL FLATFIL MASKFIL`
(+ matching `HISTORY` sha256 lines), `DARKSCL`, `CRREJ`, `OSCANCOR`, and
`KWSRC<n>` (from `FrameMeta.provenance_records()`).

**Missing, deliberately, and flagged in code + both executer reports**:
`ASTSOLVR ASTRMS1 ASTRMS2 ASTNREF ASTEPOCH WCSSOLVD NCOMBINE TEXPTIME`.
The contract's own `calibration_cards` **function signature** (which we were
told to implement exactly as written, not silently extend) has no parameter
that carries a `SolveResult` or coadd-membership data, so these 8 keywords
cannot be computed inside this function. `mef.py` has a code comment on
`calibration_cards` documenting this gap. **Next successor action**: whoever
implements astrometry solving (`astrometry/solver.py`) and coadd assembly
(`products/coadd.py`) must append these 8 cards themselves via
`robo43.provenance.apply_cards` on top of `calibration_cards(...)`'s output,
OR the orchestrator should decide to add `solve: SolveResult | None = None`
and `ncombine`/`texptime` kwargs to `calibration_cards` and have this file's
owner extend it then. Do not silently invent a different mechanism.

### Appendix B.2 (`CATALOG` header, via `CatalogMetrics`)

All 57 keywords from the addendum are implemented as `CatalogMetrics`
fields (see field/keyword table below) — this is a complete implementation,
nothing is missing from the header-keyword set itself. `EXTNAME` is not a
`CatalogMetrics` field; it's set directly by `build_catalog_hdu`/
`ReducedProduct.to_hdulist`.

Field name -> FITS keyword (the single source of truth is the
`_FIELD_KEYWORDS` dict at the top of `catalog.py`, reproduced here for a
successor who doesn't want to open the file):

```
cat_soft->CATSOFT   cat_softv->CATSOFTV  cat_cfg->CATCFG    cat_date->CATDATE
pipe_ver->PIPEVER   pipe_git->PIPEGIT    det_thrsh->DETTHRSH det_minar->DETMINAR
det_filt->DETFILT   debl_nth->DEBLNTH    debl_minc->DEBLMINC back_size->BACKSIZE
back_filt->BACKFILT weight_tp->WEIGHTTP  n_sources->NSOURCES n_stars->NSTARS
n_clean->NCLEAN     n_reject->NREJECT    n_satur->NSATUR     n_badpix->NBADPIX
zp->ZP              zp_err->ZPERR        zp_nstar->ZPNSTAR   zp_rms->ZPRMS
zp_cat->ZPCAT        zp_band->ZPBAND     magsys->MAGSYS      extinct->EXTINCT
airmass->AIRMASS    photomet->PHOTOMET   aper_rad->APERRAD   aper_unit->APERUNIT
fwhm_pix->FWHMPIX   fwhm_arc->FWHMARC    ellip->ELLIP        theta->THETA
sky_lev->SKYLEV     sky_rms->SKYRMS      sky_unit->SKYUNIT   lim_mag3->LIMMAG3
lim_mag5->LIMMAG5   snr_lim->SNRLIM      gain->GAIN          rdnoise->RDNOISE
sat_level->SATLEVEL pix_scale->PIXSCALE  exptime->EXPTIME    filter->FILTER
ast_solvr->ASTSOLVR ast_rms1->ASTRMS1    ast_rms2->ASTRMS2   ast_nref->ASTNREF
ast_epoch->ASTEPOCH radesys->RADESYS    wcssolvd->WCSSOLVD   texptime->TEXPTIME
ncombine->NCOMBINE
```

Every field defaults to `None`; `to_header()` omits the card entirely when a
field is `None` (never a sentinel). This mapping and the omission behavior
ARE the contract now — see "Decisions" below on why they must not be
silently changed.

## Test status

All 21 tests in `tests/test_mef.py` + `tests/test_catalog.py` pass, verified
independently. `mypy` on both files is clean except for the two pre-existing
`dq.py` errors noted above (not ours to fix; flag to whoever owns `dq.py` —
`dq.py` was already marked "authoritative, do not modify" in the contract, so
this is a pre-existing bug in already-accepted code, not new breakage).

**Flaky test observed in a file we do NOT own**: during the full-suite run
(`python3 -m pytest -q`), `tests/test_quality.py::test_one_row_per_class` and
`tests/test_quality.py::test_large_table_performance_and_correctness` failed
on one run and passed cleanly on an immediate rerun with no code changes in
between (123/123 passed on the clean rerun). This is `robo43.photometry.quality`
territory — outside this unit's ownership (the brief explicitly forbade
importing `robo43.photometry.*`) — flag to whoever owns that file; it smells
like a seed/ordering-dependent test rather than anything in `products/`.

## Decisions made — a successor must not silently reverse these

1. **`ERR` is 1-sigma, never variance**, and there is NO conversion step
   inside `mef.py` — `ReducedProduct.err` is assumed to already be 1-sigma by
   the time it reaches this dataclass. Per the contract, "anything holding
   variance internally converts at write time" — that conversion is the
   responsibility of whoever builds a `ReducedProduct` upstream (the reducer
   in `reduce/frame.py`, not yet written), NOT `mef.py` itself. `mef.py` only
   casts dtype (`float32`) and writes `BUNIT`; it does not know whether the
   value it was handed is variance or sigma, so it cannot enforce this — a
   successor writing `reduce/frame.py` must take the sqrt before constructing
   `ReducedProduct`.
2. **`CatalogMetrics` field names and FITS-keyword mapping** are exactly the
   table above (`_FIELD_KEYWORDS` in `catalog.py`). Renaming a field or
   keyword breaks the phase-9 ingester contract described in the addendum
   ("the ingester reads the metrics out of this header"). If a keyword must
   change, it is a cross-cutting decision for the orchestrator, not a local
   edit.
3. **Vector `FLUX_APER`/`FLUXERR_APER`/`MAG_APER`/`MAGERR_APER` columns**:
   represented as ordinary 2-D numpy arrays on the `astropy.table.Table`
   (shape `(n_sources, n_apertures)`). `build_catalog_hdu` does NOT hand-build
   `fits.Column` objects for these — it calls `astropy.io.fits.table_to_hdu`,
   which auto-derives the correct `TFORM='<n>E'`/`TDIM='(<n>)'` from the
   array's actual width. The aperture radii themselves are recorded once, in
   the header, as `APERRAD` — a **comma-joined string** (e.g.
   `"2.0,3.0,4.0,6.0,8.0"`), parsed back to a `tuple[float, ...]` by
   `CatalogMetrics.from_header`. This string-encoding is the only way a
   `tuple` field survives one FITS header card; if a successor needs
   per-aperture named columns instead, that's a breaking format change, not
   a bug fix.
4. **`'NONE'`-for-skipped vs. omitted-for-uncomputable** is implemented ONLY
   in `mef.py`'s `calibration_cards`, and ONLY for the four master-file
   keywords `BIASFIL`/`DARKFIL`/`FLATFIL`/`MASKFIL`: if the corresponding
   path argument is `None`, the card is **always written** with the literal
   string value `"NONE"` (never omitted). Every other field in both
   `calibration_cards` and `CatalogMetrics` (e.g. `DARKSCL`, `ZP`, all of
   Appendix B.2) follows the opposite rule — a `None` value means the card is
   **omitted entirely**. Do not generalize the `'NONE'` behavior to any other
   keyword; do not omit `BIASFIL` et al. when skipped.
5. **`CatalogMetrics.from_header(m.to_header()) == m` round-trips exactly,
   confirmed by test, for both a fully-populated instance and a sparse one**
   (`tests/test_catalog.py::test_round_trip_dense` and `::test_round_trip_sparse`).
   This works because `to_header`/`from_header` are pure field<->card
   mappings with NO injected business-rule side effects (in particular they
   do NOT auto-set `MAGSYS='INSTRUMENTAL'` when `zp is None` — that
   enforcement lives one layer up, in `build_catalog_hdu`, precisely so the
   round trip stays exact). A successor must keep that separation: any new
   business rule about header contents belongs in `build_catalog_hdu`
   (or `ReducedProduct`), never inside `CatalogMetrics.to_header`/
   `from_header`, or the round-trip test will start failing for sparse
   instances.
6. **`MAGSYS='INSTRUMENTAL'` enforcement location**: `build_catalog_hdu`
   forces `header["MAGSYS"] = "INSTRUMENTAL"` whenever `metrics.zp is None`,
   overriding whatever `metrics.magsys` said (or adding it if absent), and
   raises `robo43.exceptions.ProductError` if `MAG_CALIB`/`MAGERR_CALIB` are
   present in the table while `zp is None`.
7. **`ReducedProduct.catalog_header` is expected to already be the COMPLETE
   `CATALOG` HDU header** (structural `TTYPE`/`TFORM`/`TUNIT` cards for the
   table AND the metrics cards AND `EXTNAME='CATALOG'`), i.e. exactly
   `catalog.build_catalog_hdu(table, metrics).header`. `mef.py` deliberately
   does NOT import `products/catalog.py` (kept the two files decoupled for
   parallel implementation) — it just re-attaches a stored `(Table, Header)`
   pair on write via
   `fits.BinTableHDU(data=fits.table_to_hdu(self.catalog).data, header=self.catalog_header)`.
   Whoever writes the reducer that actually constructs a `ReducedProduct`
   with a populated `catalog` must call `catalog.build_catalog_hdu(...)`
   itself first and pass `.header` through — `ReducedProduct` will not do
   that for you, and passing a bare/incomplete header will produce a
   `CATALOG` HDU missing metrics cards (not a crash, just silently sparse).
8. **HDU access is always by `EXTNAME`, never positional** (`hdul["SCI"]`,
   `hdul["CATALOG"]`, `"CATALOG" in hdul`, etc.), including on the
   `PrimaryHDU`, which carries `EXTNAME='SCI'`. This was a deliberate choice
   to satisfy the repo-wide "never index `hdul[0]` outside `fits_io.py`" rule.
9. **`CATFLAGS` is only ever added when a catalogue is present** (test
   `test_hdu_order_without_catalog` locks this in) — `include_catflags=True`
   with `catalog=None` produces `["SCI","ERR","DQ"]`, not
   `["SCI","ERR","DQ","CATFLAGS"]`.
10. **Missing `CATALOG` HDU on read is graceful, not an exception** — but
    only in `mef.py`'s `ReducedProduct.read`/`from_hdulist` (returns
    `catalog=None, catalog_header=None`). `catalog.py`'s `read_catalog`
    function is the opposite on purpose: its return type is a non-Optional
    `tuple[Table, CatalogMetrics]`, so a missing `CATALOG` extension there
    raises `ProductError`. Do not blur this split — `read_catalog` is for
    "I know there's a catalogue, get it"; `ReducedProduct.read` is for
    "read whatever this product file has."

## Any card that exceeded 80 columns

None shipped. All `CatalogMetrics` card comments were deliberately kept
under ~35 characters specifically to leave headroom for long string values
(e.g. `zp_cat="GAIA-DR3"`, `cat_date` ISO timestamps) without truncation.
Verified by a dedicated test in both files
(`test_no_verify_warning_on_write` in `test_catalog.py`,
`test_calibration_cards_no_verify_warning` in `test_mef.py`) that wraps an
actual FITS write (`hdu.writeto(io.BytesIO())`) inside
`warnings.simplefilter("error", VerifyWarning)` — astropy's `VerifyWarning`
for a truncated-comment card only fires at write/verify time, NOT at
header-assignment time, so a naive test that only builds the header without
writing it would silently miss this class of bug. Both tests pass.

## Contract deviations from `FOUNDATION_CONTRACT.md` ADDENDUM 3

1. `calibration_cards` omits `ASTSOLVR ASTRMS1 ASTRMS2 ASTNREF ASTEPOCH
   WCSSOLVD NCOMBINE TEXPTIME` — see "Coverage" section above. This is a
   genuine signature gap in the contract as written (the listed keyword
   block and the listed Python signature disagree), not an implementation
   shortcut; implemented the signature literally as specified and flagged
   the gap rather than guessing new kwargs.
2. No other deviations. `DQ` (bits, dtype), `provenance.provenance_cards`/
   `apply_cards`, `fits_io.write_fits` were all reused verbatim, not
   reimplemented, per the contract's explicit instruction.

## Next steps (ordered)

1. Orchestrator decision needed on the `calibration_cards` gap above (item 1
   under "Coverage of Appendix B.1"): either extend the function's kwargs to
   accept a `SolveResult`/coadd data, or explicitly assign that job to
   `astrometry/solver.py` and `products/coadd.py`'s authors. Until decided,
   `WCSSOLVD` et al. will not appear in any product's `SCI` header.
2. Whoever writes `reduce/frame.py` (order of operations, builds the actual
   `ReducedProduct` per frame) needs to: (a) take the sqrt of variance before
   setting `ReducedProduct.err` — nothing enforces this for them; (b) call
   `catalog.build_catalog_hdu(table, metrics)` itself and pass `.header`
   into `ReducedProduct(catalog_header=...)` — `ReducedProduct` does not call
   into `catalog.py`.
3. Whoever owns `robo43/products/dq.py` should look at the two pre-existing
   mypy errors at lines 107/119 (`describe()` — `Optional` narrowing issue on
   `value & int(flag)` names list) — unrelated to this unit but currently the
   only mypy noise anywhere near `products/`.
4. If the `tests/test_quality.py` flakiness reproduces again, its owner
   should check whether any global RNG state leaks between test modules
   (unlikely, since the failing test uses a locally seeded
   `np.random.default_rng(0)`) or whether the two failing tests are simply
   too tight a tolerance for the RNG draw in question — not this unit's file,
   noted here only because it showed up in the same `pytest -q` run.
5. Nothing in `products/mef.py` or `products/catalog.py` itself is
   outstanding; this unit can be marked done in the plan tracker once the
   orchestrator has ruled on item 1.
