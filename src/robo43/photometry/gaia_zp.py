"""Optional Gaia DR3 absolute zero-point calibration.

This module adds an *optional*, off-by-default absolute photometric
calibration on top of the existing instrumental photometry
(:mod:`robo43.photometry.catalog`). It never alters ``FLUX_AUTO``,
``FLUXERR_AUTO``, ``MAG_AUTO``, ``MAGERR_AUTO``, any ``*_APER`` column, or
``settings.photometry.instrumental_zp`` -- the relative-photometry
(exoplanet-detection) path this pipeline exists for reads only those, and
must see identical numbers whether or not this module ever runs. Everything
here is additive: new ``REFMAG``/``REFMAGERR``/``GAIA_SOURCE_ID``/
``GAIA_SEP_ARCSEC`` columns for cross-matched sources, new
``MAG_CALIB``/``MAGERR_CALIB``/``MAG_APER_CAL``/``MAGERR_APER_CAL`` columns
*only* when a zero point was actually fit, and a handful of SCI-header cards
(``ZPABS*``, ``ZPTnn``).

Reference-magnitude order (design requirement 1):

    (a) Gaia DR3 XP low-resolution-spectra synthetic photometry, in the
        instrument's own filter system -- ``gaiaxpy``'s ``PhotometricSystem
        .JPLUS`` for T80S/T80N (S-PLUS and J-PLUS are twin surveys sharing
        the same 12-band Javalambre filter set), mapped per instrument via
        ``[instrument.filters.gaia_xp]``. Used only when ``gaiaxpy`` is
        installed (it is an optional dependency -- see the ``gaia`` extra in
        pyproject.toml) and the archive query succeeds; imported lazily so
        its absence never breaks anything else.
    (b) Fallback: a Gaia G/BP/RP colour transform to a Johnson-Cousins/SDSS
        magnitude (:data:`GAIA_TRANSFORMS`, coefficients from Riello et al.
        2021 -- see that dict's docstring for the exact citation), mapped
        per instrument via ``[instrument.filters.gaia_transform]``.

A filter with neither mapping degrades to "calibration unavailable"
(``GaiaZeroPoint.source == "unavailable"``), logged, never an error.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip
from astropy.table import Table

from robo43.io.meta import FrameMeta
from robo43.photometry import quality

if TYPE_CHECKING:
    from astropy.io import fits
    from astropy.wcs import WCS

    from robo43.config import Settings
    from robo43.instruments.profile import InstrumentProfile

logger = logging.getLogger(__name__)

__all__ = [
    "GAIA_TRANSFORMS",
    "GaiaTransform",
    "GaiaZeroPoint",
    "ZeroPointFit",
    "calibrate_frame",
    "fit_zero_point",
    "gaia_quality_mask",
    "header_cards",
    "match_to_gaia",
    "parse_tile_slices",
    "per_tile_offsets",
    "propagate_epoch",
    "query_gaia_cone",
    "resolve_reference_mags",
    "select_reference_stars",
    "spatial_zp_range",
    "transform_mag",
    "xp_synthetic_mags",
]

QueryFn = Callable[[float, float, float], Table]

#: Default sub-directory name for the on-disk Gaia cone/XP cache, created
#: under the reduction outdir by the CLI.
DEFAULT_GAIA_CACHE_DIRNAME = "gaia_cache"

#: Columns requested from gaiadr3.gaia_source; kept to what this module
#: actually uses (proper motion, photometry, and the three quality flags),
#: not a blind SELECT *.
_GAIA_QUERY_COLUMNS: tuple[str, ...] = (
    "source_id",
    "ra",
    "dec",
    "ref_epoch",
    "pmra",
    "pmdec",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "ruwe",
    "phot_bp_rp_excess_factor",
    "duplicated_source",
)


# ---------------------------------------------------------------------------
# Path (b): Gaia DR3 -> Johnson-Cousins/SDSS colour-transform polynomials
# ---------------------------------------------------------------------------
#
# Source: Riello, M., De Angeli, F., Evans, D. W., et al. 2021, A&A, 649, A3,
# "Gaia Early Data Release 3: Photometric content and validation"
# (arXiv:2012.01916v1, fetched and read directly -- not quoted from memory).
# Appendix C, Table C.1 gives the colour validity range and Table C.2 the
# polynomial coefficients and RMS, page 31 of the arXiv PDF. Only the
# "G - m_X = f(GBP-GRP)" rows are used (T80S/T80N have no native V-I or g-i
# colour of their own to drive the other rows in that table). Of that
# family, only three bands both appear in Table C.2 AND correspond to a
# T80S broad Javalambre filter with an SDSS-like bandpass (see the module
# docstring): G-g, G-r, G-i. No official Gaia relation exists for an
# SDSS z-band, or for any narrow Javalambre filter (F378/F395/F410/F430/
# F515/F660/F861) or uJAVA -- those simply have no entry here, and no
# [instrument.filters.gaia_transform] mapping in t80s.toml either.
@dataclass(frozen=True, slots=True)
class GaiaTransform:
    """One ``G - m_X = poly(GBP-GRP)`` relation from Riello et al. 2021, Table C.2."""

    #: c0, c1, c2, ... such that ``G - m_X = sum(c_i * (GBP-GRP)**i)``.
    coeffs: tuple[float, ...]
    #: Table C.1 validity range for GBP-GRP.
    color_min: float
    color_max: float
    #: Table C.2 sigma, mag.
    rms: float


GAIA_TRANSFORMS: dict[str, GaiaTransform] = {
    # G - g = 0.2199 - 0.6365 x - 0.1548 x^2 + 0.0064 x^3 (x = GBP-GRP)
    # valid 0.3 < x < 3.0 (Table C.2 footnote g: x > 2.0 valid for M giants
    # only), sigma = 0.0745.
    "sdss_g": GaiaTransform(
        coeffs=(0.2199, -0.6365, -0.1548, 0.0064),
        color_min=0.3,
        color_max=3.0,
        rms=0.0745,
    ),
    # G - r = -0.09837 + 0.08592 x + 0.1907 x^2 - 0.1701 x^3 + 0.02263 x^4
    # valid 0.0 < x < 3.0 (footnote f: x > 2.0 valid for M giants only),
    # sigma = 0.03776.
    "sdss_r": GaiaTransform(
        coeffs=(-0.09837, 0.08592, 0.1907, -0.1701, 0.02263),
        color_min=0.0,
        color_max=3.0,
        rms=0.03776,
    ),
    # G - i = -0.293 + 0.6404 x - 0.09609 x^2 - 0.002104 x^3
    # valid 0.5 < x < 2.0, sigma = 0.04092.
    "sdss_i": GaiaTransform(
        coeffs=(-0.293, 0.6404, -0.09609, -0.002104),
        color_min=0.5,
        color_max=2.0,
        rms=0.04092,
    ),
}


def transform_mag(
    g_mag: np.ndarray, bp_rp: np.ndarray, transform_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a :data:`GAIA_TRANSFORMS` polynomial.

    Returns ``(mag, valid_mask)``; ``mag = g_mag - poly(bp_rp)``.
    ``valid_mask`` is False outside ``[color_min, color_max]`` or wherever
    ``g_mag``/``bp_rp`` is not finite. Raises ``KeyError`` for an unknown
    ``transform_name`` -- callers check :data:`GAIA_TRANSFORMS` first.
    """
    xf = GAIA_TRANSFORMS[transform_name]
    g_mag = np.asarray(g_mag, dtype=float)
    bp_rp = np.asarray(bp_rp, dtype=float)
    poly = np.zeros_like(bp_rp)
    for power, c in enumerate(xf.coeffs):
        poly = poly + c * bp_rp**power
    mag = g_mag - poly
    valid = (
        np.isfinite(g_mag) & np.isfinite(bp_rp) & (bp_rp >= xf.color_min) & (bp_rp <= xf.color_max)
    )
    return mag, valid


# ---------------------------------------------------------------------------
# Gaia cone query, caching, proper motion
# ---------------------------------------------------------------------------


def _cache_key(ra_deg: float, dec_deg: float, radius_deg: float) -> str:
    """One cache key per rounded (ra, dec, radius) triple -- see the module
    docstring: 45 frames of the same field share one cache file."""
    return f"gaia_{ra_deg:.3f}_{dec_deg:.3f}_{radius_deg:.3f}"


def _default_gaia_cone_query(ra_deg: float, dec_deg: float, radius_deg: float) -> Table:
    """Live ADQL cone search against the Gaia archive. Imported lazily so
    tests never need astroquery or network access -- they inject their own
    ``query_fn`` instead."""
    from astroquery.gaia import Gaia

    cols = ", ".join(_GAIA_QUERY_COLUMNS)
    adql = (
        f"SELECT {cols} FROM gaiadr3.gaia_source WHERE "
        f"1=CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg}))"
    )
    job = Gaia.launch_job_async(adql)
    return job.get_results()


def query_gaia_cone(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    *,
    cache_dir: Path,
    query_fn: QueryFn | None = None,
) -> Table:
    """Cone search around ``(ra_deg, dec_deg)``, cached on disk under ``cache_dir``.

    One query per rounded ``(ra, dec, radius)`` triple, so 45 frames of one
    field query the archive exactly once. ``query_fn`` defaults to a live
    astroquery call; passing a fake makes this fully offline-testable.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_cache_key(ra_deg, dec_deg, radius_deg)}.fits"
    if cache_path.exists():
        logger.info("Gaia cone cache hit: %s", cache_path)
        return Table.read(cache_path)

    fn = query_fn or _default_gaia_cone_query
    table = fn(ra_deg, dec_deg, radius_deg)
    try:
        table.write(cache_path, format="fits", overwrite=True)
    except Exception as exc:
        # A cache-write failure must not fail the calibration.
        logger.warning("failed to write Gaia cone cache %s: %s", cache_path, exc)
    return table


def propagate_epoch(table: Table, target_epoch_jyear: float) -> Table:
    """Return a copy of ``table`` with ``ra_epoch``/``dec_epoch`` columns:
    positions propagated from each row's own ``ref_epoch`` to
    ``target_epoch_jyear`` using ``pmra``/``pmdec`` (mas/yr; ``pmra`` is
    already ``pm_ra * cos(dec)``, the Gaia convention). Rows with no
    ``ref_epoch``/``pmra``/``pmdec`` propagate with zero proper motion.
    """
    out = table.copy()
    n = len(out)
    ra = np.asarray(out["ra"], dtype=float)
    dec = np.asarray(out["dec"], dtype=float)
    ref_epoch = (
        np.nan_to_num(np.asarray(out["ref_epoch"], dtype=float), nan=target_epoch_jyear)
        if "ref_epoch" in out.colnames
        else np.full(n, target_epoch_jyear)
    )
    pmra = (
        np.nan_to_num(np.asarray(out["pmra"], dtype=float))
        if "pmra" in out.colnames
        else np.zeros(n)
    )
    pmdec = (
        np.nan_to_num(np.asarray(out["pmdec"], dtype=float))
        if "pmdec" in out.colnames
        else np.zeros(n)
    )

    dt = target_epoch_jyear - ref_epoch  # years
    ddec_deg = pmdec * dt / 3.6e6  # mas/yr * yr -> mas -> deg
    cosdec = np.cos(np.radians(dec))
    cosdec = np.where(np.abs(cosdec) > 1e-6, cosdec, 1e-6)
    dra_deg = (pmra * dt / 3.6e6) / cosdec

    out["ra_epoch"] = ra + dra_deg
    out["dec_epoch"] = dec + ddec_deg
    return out


def gaia_quality_mask(gaia: Table, *, ruwe_max: float = 1.4) -> np.ndarray:
    """Boolean mask: RUWE < ``ruwe_max``, not a duplicated source, and a
    sane BP/RP flux-excess factor.

    The excess-factor envelope (Evans et al. 2018, A&A 616, A4, eq. 1,
    refined by Riello et al. 2021 sec 9.4) is the standard published Gaia
    photometric-quality cut::

        1.0 + 0.015*bp_rp**2 < phot_bp_rp_excess_factor < 1.3 + 0.06*bp_rp**2

    Any of the three inputs missing from ``gaia`` counts as "passes" for
    that one cut, not "fails" -- a Gaia table lacking one column degrades
    gracefully rather than rejecting every star.
    """
    n = len(gaia)
    mask = np.ones(n, dtype=bool)
    if "ruwe" in gaia.colnames:
        ruwe = np.asarray(gaia["ruwe"], dtype=float)
        mask &= np.isfinite(ruwe) & (ruwe < ruwe_max)
    if "duplicated_source" in gaia.colnames:
        dup = np.asarray(gaia["duplicated_source"]).astype(bool)
        mask &= ~dup
    if {"phot_bp_rp_excess_factor", "phot_bp_mean_mag", "phot_rp_mean_mag"} <= set(gaia.colnames):
        bp_rp = np.asarray(gaia["phot_bp_mean_mag"], dtype=float) - np.asarray(
            gaia["phot_rp_mean_mag"], dtype=float
        )
        excess = np.asarray(gaia["phot_bp_rp_excess_factor"], dtype=float)
        lo = 1.0 + 0.015 * bp_rp**2
        hi = 1.3 + 0.06 * bp_rp**2
        mask &= np.isfinite(excess) & (excess > lo) & (excess < hi)
    return mask


# ---------------------------------------------------------------------------
# Path (a): Gaia XP synthetic photometry (gaiaxpy, optional dependency)
# ---------------------------------------------------------------------------


def _find_gaiaxpy_columns(df: Any, band: str) -> tuple[str, str] | None:
    """Locate ``(mag_col, err_col)`` for ``band`` in a ``gaiaxpy.generate()`` result.

    gaiaxpy's documented naming (``<system_label>_mag_<band>`` /
    ``<system_label>_mag_error_<band>``) is not hardcoded here in case a
    future release changes it: any column containing both ``band`` and
    "mag" (case-insensitive) is a candidate, split into magnitude/error by
    whether "error" also appears. Returns ``None`` -- never raises -- if
    that search is not exactly one-to-one.
    """
    cols = list(df.columns)
    mag_candidates = [
        c
        for c in cols
        if band.lower() in c.lower() and "mag" in c.lower() and "error" not in c.lower()
    ]
    err_candidates = [
        c for c in cols if band.lower() in c.lower() and "mag" in c.lower() and "error" in c.lower()
    ]
    if len(mag_candidates) != 1 or len(err_candidates) != 1:
        return None
    return mag_candidates[0], err_candidates[0]


def xp_synthetic_mags(
    source_ids: list[int],
    band: str,
    *,
    cache_dir: Path,
    cache_key: str,
    generate_fn: Callable[[list[int]], Any] | None = None,
) -> Table | None:
    """Gaia DR3 XP synthetic photometry in the JPLUS band ``band`` (path a).

    Lazily imports ``gaiaxpy``; returns ``None`` -- never raises -- whenever
    gaiaxpy is not installed, the archive query fails, or ``band`` cannot
    be found in the returned columns, so the caller always has a
    well-defined fallback to the colour-transform path (b). Cached on disk
    next to the cone-search cache, keyed the same way (one XP query per
    field).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}_xp_{band}.fits"
    if cache_path.exists():
        logger.info("Gaia XP synthetic photometry cache hit: %s", cache_path)
        return Table.read(cache_path)

    if not source_ids:
        return None

    fn = generate_fn
    if fn is None:
        try:
            from gaiaxpy import PhotometricSystem, generate
        except ImportError:
            logger.info("gaiaxpy not installed; falling back to the Gaia G/BP/RP colour transform")
            return None

        def fn(ids: list[int]) -> Any:
            return generate(
                list(ids),
                photometric_system=PhotometricSystem.JPLUS,
                save_file=False,
                truncation=False,
            )

    try:
        df = fn(source_ids)
    except Exception as exc:
        # Any archive/network/gaiaxpy failure degrades gracefully to path (b).
        logger.warning(
            "Gaia XP synthetic photometry query failed (%s); falling back to colour transform", exc
        )
        return None

    located = _find_gaiaxpy_columns(df, band)
    if located is None:
        logger.warning(
            "could not find band %r in gaiaxpy output columns %s; falling back",
            band,
            list(df.columns),
        )
        return None
    mag_col, err_col = located

    out = Table()
    out["source_id"] = np.asarray(df["source_id"], dtype=np.int64)
    out["xp_mag"] = np.asarray(df[mag_col], dtype=np.float64)
    out["xp_mag_err"] = np.asarray(df[err_col], dtype=np.float64)

    try:
        out.write(cache_path, format="fits", overwrite=True)
    except Exception as exc:
        logger.warning("failed to write Gaia XP cache %s: %s", cache_path, exc)
    return out


def resolve_reference_mags(
    gaia: Table,
    *,
    filter_name: str,
    profile: InstrumentProfile,
    cache_dir: Path,
    cache_key: str,
    allow_xp: bool = True,
    allow_transform: bool = True,
    xp_generate_fn: Callable[[list[int]], Any] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, str]:
    """Reference magnitude per row of ``gaia``, in the fixed design order
    (a) XP synthetic photometry, (b) colour transform.

    Returns ``(mag, mag_err, valid_mask, source_label)``; ``mag`` is
    ``None`` (``source_label == "unavailable"``) when neither path applies
    to ``filter_name`` -- an unmapped filter is logged, never an error.
    """
    xp_band = profile.filter_gaia_xp.get(filter_name)
    if allow_xp and xp_band:
        xp = xp_synthetic_mags(
            [int(v) for v in gaia["source_id"]],
            xp_band,
            cache_dir=cache_dir,
            cache_key=cache_key,
            generate_fn=xp_generate_fn,
        )
        if xp is not None:
            lookup = {
                int(sid): (float(m), float(e))
                for sid, m, e in zip(xp["source_id"], xp["xp_mag"], xp["xp_mag_err"], strict=True)
            }
            mag = np.full(len(gaia), np.nan)
            err = np.full(len(gaia), np.nan)
            for i, sid in enumerate(gaia["source_id"]):
                pair = lookup.get(int(sid))
                if pair is not None:
                    mag[i], err[i] = pair
            valid = np.isfinite(mag)
            if np.any(valid):
                return mag, err, valid, "gaia_xp_jplus"
            logger.info(
                "Gaia XP synthetic photometry returned no usable rows; "
                "falling back to colour transform"
            )

    transform_name = profile.filter_gaia_transform.get(filter_name)
    if allow_transform and transform_name and transform_name in GAIA_TRANSFORMS:
        g_mag = np.asarray(gaia["phot_g_mean_mag"], dtype=float)
        bp = np.asarray(gaia["phot_bp_mean_mag"], dtype=float)
        rp = np.asarray(gaia["phot_rp_mean_mag"], dtype=float)
        mag, valid = transform_mag(g_mag, bp - rp, transform_name)
        err = np.full(len(gaia), GAIA_TRANSFORMS[transform_name].rms)
        return mag, err, valid, f"gaia_gbrp_{transform_name}"

    logger.info(
        "filter %r has no [instrument.filters.gaia_xp] or [gaia_transform] mapping; "
        "absolute calibration unavailable",
        filter_name,
    )
    return None, None, None, "unavailable"


# ---------------------------------------------------------------------------
# Reference-star selection (reuses robo43.photometry.quality) and matching
# ---------------------------------------------------------------------------


def _minimal_frame_meta(
    *,
    path: Path,
    naxis1: int,
    naxis2: int,
    pixel_scale: float,
    saturation: float | None,
    profile_name: str,
) -> FrameMeta:
    """A :class:`~robo43.io.meta.FrameMeta` carrying only the four fields
    :func:`robo43.photometry.quality.assess` actually reads (naxis1, naxis2,
    pixel_scale, saturation); every other field is a harmless placeholder.
    Built here, not via :mod:`robo43.io.headers`, because this module never
    reads a raw frame -- it only re-derives quality cuts on an
    already-measured CATALOG table.
    """
    return FrameMeta(
        path=path,
        naxis1=naxis1,
        naxis2=naxis2,
        frame_type=None,
        exptime=None,
        date_obs=None,
        filter=None,
        ra_deg=None,
        dec_deg=None,
        pixel_scale=pixel_scale,
        ccd_temp=None,
        gain_setting=None,
        read_noise=None,
        saturation=saturation,
        airmass=None,
        object=None,
        telescope=None,
        instrument=None,
        observer=None,
        binning=None,
        readout_mode=None,
        focal_length_mm=None,
        site_lat_deg=None,
        site_lon_deg=None,
        site_elevation_m=None,
        mjd=None,
        night=None,
        is_valid=True,
        profile_name=profile_name,
        origins={},
    )


def select_reference_stars(
    table: Table,
    *,
    settings: Settings,
    profile: InstrumentProfile,
    header: fits.Header,
    wcs: WCS | None,
    scale: float,
    gain_e_per_adu: float | None,
) -> Table:
    """The subset of ``table`` usable as photometric ZP reference stars:
    unsaturated, isolated, well-shaped, ``FLAGS==0`` -- via
    :func:`robo43.photometry.quality.assess`/``reference_sample`` (stellar
    mode), reused rather than reimplemented.
    """
    nx = header.get("NAXIS1")
    ny = header.get("NAXIS2")
    naxis1 = int(nx) if nx else (int(np.max(table["X_IMAGE"])) if len(table) else 0)
    naxis2 = int(ny) if ny else (int(np.max(table["Y_IMAGE"])) if len(table) else 0)
    meta = _minimal_frame_meta(
        path=Path(),
        naxis1=naxis1,
        naxis2=naxis2,
        pixel_scale=scale,
        saturation=profile.saturation_adu,
        profile_name=profile.name,
    )
    report = quality.assess(
        table,
        settings=settings,
        profile=profile,
        meta=meta,
        wcs=wcs,
        reference=None,
        stellar=True,
        gain_e_per_adu=gain_e_per_adu,
    )
    return quality.reference_sample(table, report, settings=settings)


def match_to_gaia(
    table: Table,
    gaia_epoch: Table,
    *,
    radius_arcsec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-neighbour sky match of ``table`` (``ALPHA_J2000``/
    ``DELTA_J2000``) to ``gaia_epoch`` (``ra_epoch``/``dec_epoch``, already
    proper-motion corrected). Returns ``(cat_idx, gaia_idx, sep_arcsec)``
    for every pair within ``radius_arcsec``.
    """
    if len(table) == 0 or len(gaia_epoch) == 0:
        empty_i = np.array([], dtype=int)
        return empty_i, empty_i, np.array([], dtype=float)
    src = SkyCoord(
        ra=np.asarray(table["ALPHA_J2000"], dtype=float) * u.deg,
        dec=np.asarray(table["DELTA_J2000"], dtype=float) * u.deg,
    )
    ref = SkyCoord(
        ra=np.asarray(gaia_epoch["ra_epoch"], dtype=float) * u.deg,
        dec=np.asarray(gaia_epoch["dec_epoch"], dtype=float) * u.deg,
    )
    idx, sep2d, _ = src.match_to_catalog_sky(ref)
    sep_arcsec = sep2d.arcsec
    good = sep_arcsec <= radius_arcsec
    cat_idx = np.flatnonzero(good)
    return cat_idx, idx[good], sep_arcsec[good]


# ---------------------------------------------------------------------------
# Zero-point fit, per-tile offsets, spatial diagnostic
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ZeroPointFit:
    zp: float | None
    err: float | None
    rms: float | None
    n: int


def fit_zero_point(diffs: np.ndarray, *, sigma: float) -> ZeroPointFit:
    """Sigma-clipped median of ``diffs`` (= m_ref - m_inst), with a
    MAD-based RMS and standard error::

        rms = 1.4826 * MAD(clipped diffs)
        err = rms / sqrt(N)

    ``ZeroPointFit(None, None, None, 0)`` when nothing survives.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return ZeroPointFit(None, None, None, 0)
    clipped = sigma_clip(diffs, sigma=sigma, maxiters=5, masked=True)
    good = np.asarray(diffs[~clipped.mask])
    if good.size == 0:
        return ZeroPointFit(None, None, None, 0)
    zp = float(np.median(good))
    mad = float(np.median(np.abs(good - zp)))
    rms = 1.4826 * mad
    err = rms / np.sqrt(good.size)
    return ZeroPointFit(zp=zp, err=float(err), rms=float(rms), n=int(good.size))


_TILE_RE = re.compile(r"\[(\d+):(\d+),(\d+):(\d+)\]")


def parse_tile_slices(header: fits.Header, *, shape: tuple[int, int]) -> list[tuple[slice, slice]]:
    """Recover science-tile pixel boundaries from ``TILE00``.. header cards
    (written by
    :meth:`robo43.calib.overscan.OverscanGeometry.tile_header_cards` during
    reduction). ``shape`` is ``(ny, nx)`` of the array these slices index
    into; falls back to a single whole-frame tile when no such cards are
    present (a single-amplifier instrument, or a product reduced before
    these cards existed).
    """
    tiles: list[tuple[slice, slice]] = []
    idx = 0
    while True:
        key = f"TILE{idx:02d}"
        if key not in header:
            break
        match = _TILE_RE.match(str(header[key]).strip())
        if not match:
            break
        x1, x2, y1, y2 = (int(g) for g in match.groups())
        tiles.append((slice(y1 - 1, y2), slice(x1 - 1, x2)))
        idx += 1
    if tiles:
        return tiles
    ny, nx = shape
    return [(slice(0, ny), slice(0, nx))]


def per_tile_offsets(
    x: np.ndarray,
    y: np.ndarray,
    diffs: np.ndarray,
    tiles: list[tuple[slice, slice]],
    *,
    zp_global: float,
    sigma: float,
    min_stars: int = 3,
) -> dict[int, float]:
    """ZP offset (tile ZP - ``zp_global``) for each tile with at least
    ``min_stars`` usable stars -- a consistency diagnostic, never gated by
    ``zp_rms_max``. Tiles with too few stars are simply absent from the
    result. Returns ``{}`` for a single-tile geometry.
    """
    offsets: dict[int, float] = {}
    if len(tiles) <= 1:
        return offsets
    for i, (row_slice, col_slice) in enumerate(tiles):
        in_tile = (
            (y >= row_slice.start)
            & (y < row_slice.stop)
            & (x >= col_slice.start)
            & (x < col_slice.stop)
        )
        if np.count_nonzero(in_tile) < min_stars:
            continue
        fit = fit_zero_point(diffs[in_tile], sigma=sigma)
        if fit.zp is None or fit.n < min_stars:
            continue
        offsets[i] = fit.zp - zp_global
    return offsets


def spatial_zp_range(
    x: np.ndarray,
    y: np.ndarray,
    diffs: np.ndarray,
    *,
    naxis1: int,
    naxis2: int,
    grid: int = 4,
    min_stars: int = 3,
) -> float | None:
    """max-min of the per-cell median ZP across a coarse ``grid``x``grid``
    spatial mesh -- a single diagnostic scalar for spatially-varying
    illumination or flat-field residuals, not a persisted map. ``None``
    when fewer than two cells have ``min_stars`` stars.
    """
    if naxis1 <= 0 or naxis2 <= 0:
        return None
    cell_w = naxis1 / grid
    cell_h = naxis2 / grid
    cell_medians: list[float] = []
    for gy in range(grid):
        for gx in range(grid):
            in_cell = (
                (x >= gx * cell_w)
                & (x < (gx + 1) * cell_w)
                & (y >= gy * cell_h)
                & (y < (gy + 1) * cell_h)
                & np.isfinite(diffs)
            )
            if np.count_nonzero(in_cell) < min_stars:
                continue
            cell_medians.append(float(np.median(diffs[in_cell])))
    if len(cell_medians) < 2:
        return None
    return float(max(cell_medians) - min(cell_medians))


# ---------------------------------------------------------------------------
# Top-level result, header cards, orchestration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GaiaZeroPoint:
    """The outcome of one :func:`calibrate_frame` call."""

    zp: float | None
    zp_err: float | None
    zp_rms: float | None
    n: int
    source: str
    calibrated: bool
    tile_offsets: dict[int, float] = field(default_factory=dict)
    spatial_range: float | None = None
    filter: str | None = None


def header_cards(gzp: GaiaZeroPoint) -> list[tuple[str, Any, str]]:
    """SCI-header cards for one frame's absolute calibration (<=8-char keys)."""
    cards: list[tuple[str, Any, str]] = [
        (
            "ZPABS",
            gzp.zp if gzp.zp is not None else "NONE",
            "Gaia-calibrated absolute zero point, mag",
        ),
        ("ZPABSERR", gzp.zp_err if gzp.zp_err is not None else "NONE", "ZPABS standard error, mag"),
        (
            "ZPABSRMS",
            gzp.zp_rms if gzp.zp_rms is not None else "NONE",
            "ZPABS fit RMS (1.4826*MAD), mag",
        ),
        ("ZPABSN", gzp.n, "reference stars used for ZPABS"),
        ("ZPABSSRC", gzp.source, "reference-magnitude source"),
        ("ZPABSCAL", gzp.calibrated, "ZPABS meets zp_min_stars/zp_rms_max"),
    ]
    for idx, offset in sorted(gzp.tile_offsets.items()):
        cards.append((f"ZPT{idx:02d}", offset, f"tile {idx} ZP offset from ZPABS, mag"))
    if gzp.spatial_range is not None:
        cards.append(("SPZPRNG", gzp.spatial_range, "4x4 spatial ZP grid range (max-min), mag"))
    return cards


def _field_cone(table: Table, wcs: WCS | None) -> tuple[float, float, float]:
    """Cone ``(ra, dec, radius_deg)`` covering the frame's footprint.

    Prefers the WCS footprint (``wcs.calc_footprint()``); falls back to the
    detected sources' own RA/Dec extent when the WCS cannot report one (no
    ``NAXIS``/shape known to it).
    """
    if wcs is not None:
        try:
            corners = wcs.calc_footprint()
            ra_c = float(np.mean(corners[:, 0]))
            dec_c = float(np.mean(corners[:, 1]))
            center = SkyCoord(ra=ra_c * u.deg, dec=dec_c * u.deg)
            footprint = SkyCoord(ra=corners[:, 0] * u.deg, dec=corners[:, 1] * u.deg)
            radius_deg = float(np.max(center.separation(footprint).deg)) * 1.05
            if radius_deg > 0:
                return ra_c, dec_c, radius_deg
        except Exception as exc:
            # Fall through to the table-extent estimate below.
            logger.debug("wcs.calc_footprint() failed (%s); using source extent instead", exc)

    ra = np.asarray(table["ALPHA_J2000"], dtype=float)
    dec = np.asarray(table["DELTA_J2000"], dtype=float)
    ra = ra[np.isfinite(ra)]
    dec = dec[np.isfinite(dec)]
    if ra.size == 0:
        return 0.0, 0.0, 0.0
    ra_c, dec_c = float(np.median(ra)), float(np.median(dec))
    sep = (
        SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        .separation(SkyCoord(ra=ra_c * u.deg, dec=dec_c * u.deg))
        .deg
    )
    radius_deg = float(np.max(sep)) * 1.1 if sep.size else 0.7
    return ra_c, dec_c, radius_deg


def _frame_epoch_jyear(header: fits.Header) -> float:
    """Observation epoch as a Julian-year float, for Gaia proper-motion propagation."""
    from astropy.time import Time

    mjd = header.get("MJD-OBS")
    if mjd is not None:
        try:
            return float(Time(float(mjd), format="mjd").jyear)
        except (TypeError, ValueError):
            pass
    date_obs = header.get("DATE-OBS")
    if date_obs is not None:
        try:
            return float(Time(str(date_obs), format="isot").jyear)
        except Exception:
            pass
    return 2016.0  # Gaia DR3 reference epoch; last-resort fallback


def calibrate_frame(
    table: Table,
    *,
    header: fits.Header,
    wcs: WCS | None,
    settings: Settings,
    profile: InstrumentProfile,
    gain_e_per_adu: float | None,
    scale: float,
    filter_name: str | None,
    cache_dir: Path,
    query_fn: QueryFn | None = None,
    xp_generate_fn: Callable[[list[int]], Any] | None = None,
) -> tuple[GaiaZeroPoint, Table]:
    """Full absolute-calibration pipeline for one already-measured frame.

    Returns ``(GaiaZeroPoint, table)`` where ``table`` is a copy of the
    input CATALOG table with ``REFMAG``/``REFMAGERR``/``GAIA_SOURCE_ID``/
    ``GAIA_SEP_ARCSEC`` added for every cross-matched source, plus
    ``MAG_CALIB``/``MAGERR_CALIB``/``MAG_APER_CAL``/``MAGERR_APER_CAL``
    only when ``GaiaZeroPoint.calibrated`` is True -- never added at all
    otherwise, so :func:`robo43.products.catalog.build_catalog_hdu`'s own
    zp-required-for-MAG_CALIB guard (which only requires ``zp`` to be
    non-``None``) is honoured a fortiori. Never
    modifies any input column in place.
    """
    ps = settings.photometry
    out = table.copy()
    unavailable = GaiaZeroPoint(
        zp=None,
        zp_err=None,
        zp_rms=None,
        n=0,
        source="unavailable",
        calibrated=False,
        filter=filter_name,
    )

    if filter_name is None or len(table) == 0:
        return unavailable, out

    if not (
        profile.filter_gaia_xp.get(filter_name) or profile.filter_gaia_transform.get(filter_name)
    ):
        logger.info(
            "filter %r: no Gaia reference mapping in the instrument profile; "
            "absolute calibration unavailable",
            filter_name,
        )
        return unavailable, out

    ra_center, dec_center, radius_deg = _field_cone(out, wcs)
    if radius_deg <= 0:
        return unavailable, out
    epoch = _frame_epoch_jyear(header)

    gaia = query_gaia_cone(
        ra_center, dec_center, radius_deg, cache_dir=cache_dir, query_fn=query_fn
    )
    if len(gaia) == 0:
        return unavailable, out

    gaia = gaia[gaia_quality_mask(gaia, ruwe_max=ps.gaia_ruwe_max)]
    if len(gaia) == 0:
        return unavailable, out

    gaia_epoch = propagate_epoch(gaia, epoch)
    cache_key = _cache_key(ra_center, dec_center, radius_deg)
    ref_mag, ref_mag_err, ref_valid, source_label = resolve_reference_mags(
        gaia_epoch,
        filter_name=filter_name,
        profile=profile,
        cache_dir=cache_dir,
        cache_key=cache_key,
        xp_generate_fn=xp_generate_fn,
    )
    if ref_mag is None:
        return unavailable, out

    cat_idx, gaia_idx, sep_arcsec = match_to_gaia(
        out, gaia_epoch, radius_arcsec=ps.gaia_match_radius_arcsec
    )
    if len(cat_idx) == 0:
        return GaiaZeroPoint(
            zp=None,
            zp_err=None,
            zp_rms=None,
            n=0,
            source=source_label,
            calibrated=False,
            filter=filter_name,
        ), out

    keep = ref_valid[gaia_idx]
    cat_idx, gaia_idx, sep_arcsec = cat_idx[keep], gaia_idx[keep], sep_arcsec[keep]
    if len(cat_idx) == 0:
        return GaiaZeroPoint(
            zp=None,
            zp_err=None,
            zp_rms=None,
            n=0,
            source=source_label,
            calibrated=False,
            filter=filter_name,
        ), out

    # REFMAG/REFMAGERR/GAIA_SOURCE_ID/GAIA_SEP_ARCSEC: every cross-matched
    # source, independent of whether it also passes the stricter cuts used
    # for the ZP fit itself below -- useful for photometric-consistency
    # checks even on a frame whose ZP fit was rejected.
    n_rows = len(out)
    refmag_col = np.full(n_rows, np.nan, dtype=np.float32)
    refmagerr_col = np.full(n_rows, np.nan, dtype=np.float32)
    gaia_id_col = np.zeros(n_rows, dtype=np.int64)
    sep_col = np.full(n_rows, np.nan, dtype=np.float32)
    for ci, gi, sep in zip(cat_idx, gaia_idx, sep_arcsec, strict=True):
        refmag_col[ci] = ref_mag[gi]
        refmagerr_col[ci] = ref_mag_err[gi]
        gaia_id_col[ci] = int(gaia_epoch["source_id"][gi])
        sep_col[ci] = sep
    out["REFMAG"] = refmag_col
    out["REFMAGERR"] = refmagerr_col
    out["GAIA_SOURCE_ID"] = gaia_id_col
    out["GAIA_SEP_ARCSEC"] = sep_col

    # The zero point itself uses only the stricter reference-star subset
    # (unsaturated/isolated/FLAGS==0 -- quality.reference_sample) that also
    # has a valid cross-match, per the design ("reuse it for ZP stars").
    clean = select_reference_stars(
        out,
        settings=settings,
        profile=profile,
        header=header,
        wcs=wcs,
        scale=scale,
        gain_e_per_adu=gain_e_per_adu,
    )
    has_ref = (
        np.isfinite(np.asarray(clean["REFMAG"], dtype=float))
        if len(clean)
        else np.array([], dtype=bool)
    )
    clean = clean[has_ref]

    if len(clean) == 0:
        return GaiaZeroPoint(
            zp=None,
            zp_err=None,
            zp_rms=None,
            n=0,
            source=source_label,
            calibrated=False,
            filter=filter_name,
        ), out

    diffs = np.asarray(clean["REFMAG"], dtype=float) - np.asarray(clean["MAG_AUTO"], dtype=float)
    fit = fit_zero_point(diffs, sigma=ps.zp_sigma_clip)
    calibrated = (
        fit.zp is not None
        and fit.n >= ps.zp_min_stars
        and fit.rms is not None
        and fit.rms <= ps.zp_rms_max
    )

    tile_offsets: dict[int, float] = {}
    spatial_range: float | None = None
    if fit.zp is not None:
        naxis1 = int(header.get("NAXIS1", 0))
        naxis2 = int(header.get("NAXIS2", 0))
        tiles = parse_tile_slices(header, shape=(naxis2, naxis1))
        x = (
            np.asarray(clean["X_IMAGE"], dtype=float) - 1.0
        )  # 1-based -> 0-based, matches tile slices
        y = np.asarray(clean["Y_IMAGE"], dtype=float) - 1.0
        tile_offsets = per_tile_offsets(
            x, y, diffs, tiles, zp_global=fit.zp, sigma=ps.zp_sigma_clip
        )
        spatial_range = spatial_zp_range(x, y, diffs, naxis1=naxis1, naxis2=naxis2)

    gzp = GaiaZeroPoint(
        zp=fit.zp,
        zp_err=fit.err,
        zp_rms=fit.rms,
        n=fit.n,
        source=source_label,
        calibrated=bool(calibrated),
        tile_offsets=tile_offsets,
        spatial_range=spatial_range,
        filter=filter_name,
    )

    if gzp.calibrated:
        offset = gzp.zp - ps.instrumental_zp
        zp_err = gzp.zp_err or 0.0
        out["MAG_CALIB"] = (np.asarray(out["MAG_AUTO"], dtype=float) + offset).astype(np.float32)
        out["MAGERR_CALIB"] = np.sqrt(
            np.asarray(out["MAGERR_AUTO"], dtype=float) ** 2 + zp_err**2
        ).astype(np.float32)
        if "MAG_APER" in out.colnames:
            out["MAG_APER_CAL"] = (np.asarray(out["MAG_APER"], dtype=float) + offset).astype(
                np.float32
            )
            out["MAGERR_APER_CAL"] = np.sqrt(
                np.asarray(out["MAGERR_APER"], dtype=float) ** 2 + zp_err**2
            ).astype(np.float32)

    return gzp, out
