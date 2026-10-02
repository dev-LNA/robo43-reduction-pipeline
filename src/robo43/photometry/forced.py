"""Fixed-centroid ("forced") aperture photometry.

``robo43 photometry`` (:func:`robo43.photometry.catalog.measure_sources`) detects and
centroids sources independently on every frame.  For a blend, the per-frame
deblender may split or merge the components from one frame to the next, so the
aperture centre -- and with it the flux -- jumps between two levels ("extraction
flips"); in a light-curve search these are false transits.  This module removes the
per-frame centroiding:

1. **Frame quality** (:func:`summarize_frame`, :func:`measure_transparency`) from the
   existing per-frame catalogues: seeing (FWHM), ``n_useful`` (unflagged sources with
   ``SNR >= snr_useful``) and a relative **transparency** ``t`` (robust median
   flux ratio of bright, isolated, unflagged stars, matched on the sky, to a provisional
   reference frame -- the one with the most useful sources -- normalised so that
   the ``transparency_norm_percentile`` of the night is 1).
2. **Centroid frames** (:func:`score_frames`, :func:`select_centroid_frames`).  Seeing alone
   would pick the sharpest frame, which on a night with passing cloud can be a
   thin-cloud frame with few detectable stars; transparency alone would pick a
   photon-rich frame with poor centroids.  The compromise: a frame is *eligible*
   when astrometry is solved (RMS below ``astrometry_rms_max_arcsec``) and
   ``t >= transparency_min``; among the eligible frames

       score = n_useful / max(n_useful) * (fwhm_min / fwhm) ** seeing_power

   so the score is the fraction of the richest frame's useful sources, penalised by
   how much worse the seeing is than the best eligible frame (``seeing_power = 2``
   because centroid noise grows ~ FWHM / SNR and SNR itself falls with FWHM).  The top
   ``n_centroid_frames`` frames are used.  If no frame is eligible the transparency
   cut is dropped (with a warning) rather than failing.
3. **Master positions** (:func:`build_master`): the sources of the selected frames are
   matched on the sky; a master source is kept when detected in at least ``min_frames``
   of them (capped by how many selected frames cover its position), its RA/Dec is the
   median over the detections, near-duplicates are removed and the survivors are
   numbered 1..N in (RA, Dec) order, so the numbering is reproducible.
   (Gaia positions could replace the medians for stars with a Gaia counterpart; not done.)
4. **Forced measurement** (:func:`measure_forced`) on every frame: master RA/Dec are
   projected through *that frame's* WCS (drift, rotation and dithering are absorbed
   by the WCS, the apertures are fixed in the sky) and **not recentred**.  Sky model
   (:func:`~robo43.photometry.catalog.estimate_sky_background`), aperture radii in
   arcsec and ``ERR``-plane errors follow ``measure_sources``; unlike it, neighbouring
   sources are *not* masked or mirrored out of the aperture (that correction is what flips
   with the per-frame segmentation), they are only flagged through the master flags, and
   the local sky annulus is **smoothed over neighbouring sources**: each source's annulus
   estimate (3 sigma clipped ``SExtractorBackground``, other sources masked) is replaced by the
   median over the ``local_bkg_smooth_k`` nearest master sources.  Reason, measured on T80S night
   20251107 (12 frames, 452 bright comparison stars, median frame-to-frame MAD scatter [mmag] in
   apertures 2/3/4/6/8", after normalising by an ensemble of 5468 fainter stars):

       production ``measure_sources``        4.97  5.24  6.45  9.30  12.3
       forced, raw per-source annulus        5.80  6.15  7.85  13.9  22.7
       forced, no local sky                  5.71  5.54  6.29  8.79  12.0
       forced, annulus smoothed (k = 64)     5.58  5.23  5.90  7.75  10.2
       forced, annulus smoothed (k = 200)    5.60  5.24  5.89  7.69  9.9

   A per-source annulus (a few hundred pixels) is far noisier than the 2-D model, whose mesh
   averages ~36k pixels, and the noise grows with aperture area; dropping the local sky altogether
   leaves a seeing-correlated frame-level offset between bright stars and the ensemble.  The
   smoothed estimate keeps the local sky level without the per-source noise.  The table has
   exactly the columns of ``measure_sources``.

Per-source column semantics that differ from ``measure_sources``:

* ``NUMBER``: master source number (stable across frames); rows of a frame are the master
  sources whose centre falls on the frame (or all, with ``keep_offframe``, as NaN flagged rows).
* ``FLUX_APER``/``FLUXERR_APER``: sum of (SCI - 2-D sky - smoothed local annulus sky) in the
  circle, error from the ``ERR`` plane.  ``BACKGROUND``: that local sky relative to the 2-D
  sky (0 when ``local_bkg_width_arcsec`` is 0).
* ``SNR`` and the ``*_AUTO``/alias columns (``FLUX``, ``MAG`` ...): there is no Kron
  aperture; they repeat the aperture ``snr_aperture_arcsec``.
* ``FLAGS``: master ``FLAG_NEIGHBOR | FLAG_DEBLENDED`` (blend in the centroid frames),
  plus per frame ``FLAG_SATURATED``/``FLAG_NONLINEAR`` (DQ pixel within
  ``flag_radius_arcsec``), ``FLAG_TRUNCATED`` (centre within ``edge_margin_px`` of an edge),
  ``FLAG_APERTURE_INCOMPLETE`` (largest aperture leaves the frame), ``FLAG_BAD_PHOTOMETRY``
  (non-finite flux).  ``IMAFLAGS_ISO``/``NIMAFLAGS_ISO``: OR / count of fatal DQ pixels
  within the same radius.
* ``FWHM*``, ``FLUX_RADIUS``, ``A/B_IMAGE``, ``THETA_IMAGE``, ``ELONGATION``,
  ``ELLIPTICITY``: taken from the nearest per-frame detection within
  ``shape_match_radius_arcsec``, NaN if there is none.  (These are second-moment widths of the
  detection footprint, not the seeing FWHM; substituting the frame FWHM for the unmatched
  sources would mix two definitions, so the per-frame median of ``FWHM`` -- which relphot uses as
  a regressor and for master-frame choice -- would drift with the matched fraction.)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from astropy.stats import SigmaClip
from astropy.table import Table
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.background import SExtractorBackground
from scipy.spatial import cKDTree

from robo43.astrometry.centroids import estimate_background, measure_fwhm
from robo43.exceptions import PhotometryError
from robo43.photometry.catalog import (
    _BAD_PIXEL_FLAGS,
    _SCALAR_DTYPES,
    _VECTOR_COLUMNS,
    COLUMN_NAMES,
    FLAG_APERTURE_INCOMPLETE,
    FLAG_BAD_PHOTOMETRY,
    FLAG_DEBLENDED,
    FLAG_NEIGHBOR,
    FLAG_NONLINEAR,
    FLAG_SATURATED,
    FLAG_TRUNCATED,
    _magnitudes,
    estimate_sky_background,
    pixel_scale_arcsec,
)
from robo43.products.dq import DQ, FATAL_FLAGS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astropy.io import fits
    from astropy.wcs import WCS

    from robo43.config import ForcedSettings, Settings
    from robo43.instruments.profile import InstrumentProfile

logger = logging.getLogger(__name__)

__all__ = [
    "MASTER_COLUMNS",
    "ForcedResult",
    "FrameGeometry",
    "FrameInfo",
    "MasterCatalog",
    "build_master",
    "measure_forced",
    "measure_transparency",
    "nearest_radius_index",
    "read_master",
    "score_frames",
    "select_centroid_frames",
    "summarize_frame",
    "write_master",
]

#: Flag bits a master source passes on to every frame (a blend in the centroid frames).
_MASTER_FLAG_BITS = FLAG_NEIGHBOR | FLAG_DEBLENDED

#: Columns of the shape block that are copied from the nearest per-frame detection.
_SHAPE_COLUMNS = (
    "FLUX_RADIUS", "A_IMAGE", "B_IMAGE", "THETA_IMAGE", "ELONGATION", "ELLIPTICITY",
)

MASTER_COLUMNS: tuple[str, ...] = ("NUMBER", "RA", "DEC", "N_DET", "N_COVER", "SNR", "FLAGS")


# ---------------------------------------------------------------------------
# sky matching helpers
# ---------------------------------------------------------------------------


def _unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.radians(np.asarray(ra_deg, dtype=np.float64))
    dec = np.radians(np.asarray(dec_deg, dtype=np.float64))
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _chord(radius_arcsec: float) -> float:
    return 2.0 * np.sin(np.radians(radius_arcsec / 3600.0) / 2.0)


def _nearest(
    ra: np.ndarray, dec: np.ndarray, ref_ra: np.ndarray, ref_dec: np.ndarray, radius_arcsec: float
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest reference source of each ``(ra, dec)`` and whether it lies within the radius."""
    n = len(ra)
    if n == 0 or len(ref_ra) == 0:
        return np.zeros(n, dtype=np.int64), np.zeros(n, dtype=bool)
    tree = cKDTree(_unit_vectors(ref_ra, ref_dec))
    dist, idx = tree.query(_unit_vectors(ra, dec), k=1)
    return idx.astype(np.int64), dist <= _chord(radius_arcsec)


def nearest_radius_index(radii: Sequence[float], target: float) -> int:
    """Index of the configured aperture radius closest to ``target`` (arcsec)."""
    return int(np.argmin(np.abs(np.asarray(radii, dtype=np.float64) - target)))


# ---------------------------------------------------------------------------
# 1. frame quality
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrameInfo:
    """Per-frame summary taken from its ``measure_sources`` catalogue (no pixels)."""

    path: str
    fwhm_arcsec: float
    fwhm_px: float
    exptime: float
    astrometry_ok: bool
    astrometry_rms: float
    n_sources: int
    n_useful: int
    #: Aperture radii (arcsec) of the frame's own catalogue.
    aperture_radii_arcsec: tuple[float, ...]
    #: Bright, unflagged, isolated stars: sky position and exposure-normalised flux
    #: in the transparency aperture.
    bright_ra: np.ndarray
    bright_dec: np.ndarray
    bright_flux: np.ndarray


def summarize_frame(
    table: Table,
    *,
    path: str,
    fwhm_arcsec: float,
    fwhm_px: float,
    exptime: float,
    astrometry_ok: bool,
    astrometry_rms: float,
    aperture_radii_arcsec: Sequence[float],
    settings: Settings,
) -> FrameInfo:
    """Reduce one frame's catalogue to the numbers the frame-selection step needs."""
    fs = settings.forced
    ra = np.asarray(table["RA"], dtype=np.float64)
    dec = np.asarray(table["DEC"], dtype=np.float64)
    flags = np.asarray(table["FLAGS"])
    snr = np.asarray(table["SNR"], dtype=np.float64)
    finite = np.isfinite(ra) & np.isfinite(dec)
    clean = finite & (flags == 0)

    n_useful = int(np.count_nonzero(clean & (snr >= fs.snr_useful)))

    k = nearest_radius_index(aperture_radii_arcsec, fs.transparency_aperture_arcsec)
    flux_ap = np.asarray(table["FLUX_APER"], dtype=np.float64)
    flux_t = (flux_ap[:, k] if flux_ap.ndim == 2 else flux_ap)
    if exptime > 0:
        flux_t = flux_t / exptime

    bright = clean & (snr >= fs.transparency_snr_min) & np.isfinite(flux_t) & (flux_t > 0)
    if bright.any() and finite.sum() > 1:
        # Isolation is judged against every catalogue source, not just the bright ones.
        tree = cKDTree(_unit_vectors(ra[finite], dec[finite]))
        rows = np.nonzero(bright)[0]
        dist, _ = tree.query(_unit_vectors(ra[rows], dec[rows]), k=2)
        isolated = dist[:, 1] > _chord(settings.photometry.isolation_radius_arcsec)
        bright[rows[~isolated]] = False

    return FrameInfo(
        path=path,
        fwhm_arcsec=float(fwhm_arcsec),
        fwhm_px=float(fwhm_px),
        exptime=float(exptime),
        astrometry_ok=bool(astrometry_ok),
        astrometry_rms=float(astrometry_rms),
        n_sources=len(table),
        n_useful=n_useful,
        aperture_radii_arcsec=tuple(float(r) for r in aperture_radii_arcsec),
        bright_ra=ra[bright],
        bright_dec=dec[bright],
        bright_flux=flux_t[bright],
    )


def measure_transparency(
    infos: Sequence[FrameInfo], settings: Settings
) -> tuple[np.ndarray, np.ndarray, int]:
    """Relative transparency of each frame.

    The provisional reference is the astrometry-OK frame with the most useful
    sources.  Each frame's raw transparency is the median (after a 3 sigma MAD clip) of
    ``flux / flux_ref`` over bright isolated stars matched within
    ``match_radius_arcsec``; NaN when fewer than ``transparency_min_stars`` match (and for
    every frame when the reference itself has fewer bright stars than that).
    The result is divided by the ``transparency_norm_percentile`` of the finite
    raw values, so the clearest frames sit at 1.

    Returns ``(transparency, n_stars_used, reference_index)``.
    """
    fs = settings.forced
    n = len(infos)
    n_useful = np.array([i.n_useful if i.astrometry_ok else -1 for i in infos])
    ref = int(np.argmax(n_useful)) if n else 0
    raw = np.full(n, np.nan)
    used = np.zeros(n, dtype=np.int64)
    if n == 0:
        return raw, used, ref
    ref_info = infos[ref]
    if len(ref_info.bright_ra) < fs.transparency_min_stars:
        return raw, used, ref  # no usable yardstick: transparency unknown for every frame
    for i, info in enumerate(infos):
        if len(info.bright_ra) == 0 or len(ref_info.bright_ra) == 0:
            continue
        if i == ref:
            raw[i], used[i] = 1.0, len(info.bright_ra)
            continue
        idx, ok = _nearest(
            info.bright_ra, info.bright_dec, ref_info.bright_ra, ref_info.bright_dec,
            fs.match_radius_arcsec,
        )
        ratio = info.bright_flux[ok] / ref_info.bright_flux[idx[ok]]
        ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
        if ratio.size:
            med = np.median(ratio)
            sig = 1.4826 * np.median(np.abs(ratio - med))
            if sig > 0:
                ratio = ratio[np.abs(ratio - med) <= 3.0 * sig]
        used[i] = ratio.size
        if ratio.size >= fs.transparency_min_stars:
            raw[i] = float(np.median(ratio))
    finite = np.isfinite(raw)
    if not finite.any():
        return raw, used, ref
    norm = np.percentile(raw[finite], fs.transparency_norm_percentile)
    return raw / norm, used, ref


# ---------------------------------------------------------------------------
# 2. centroid-frame selection
# ---------------------------------------------------------------------------


def score_frames(
    fwhm_arcsec: np.ndarray,
    n_useful: np.ndarray,
    transparency: np.ndarray,
    astrometry_ok: np.ndarray,
    fs: ForcedSettings,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Eligibility and score of every frame (see the module docstring for the rationale).

    Returns ``(eligible, score, relaxed)``; ``score`` is ``-inf`` for ineligible
    frames and ``relaxed`` is True when no frame passed the transparency cut so it was
    dropped.
    """
    fwhm = np.asarray(fwhm_arcsec, dtype=np.float64)
    useful = np.asarray(n_useful, dtype=np.float64)
    trans = np.asarray(transparency, dtype=np.float64)
    usable = (
        np.asarray(astrometry_ok, dtype=bool) & np.isfinite(fwhm) & (fwhm > 0) & (useful > 0)
    )
    eligible = usable & np.isfinite(trans) & (trans >= fs.transparency_min)
    relaxed = False
    if not eligible.any() and usable.any():
        relaxed = True
        eligible = usable
    score = np.full(len(fwhm), -np.inf)
    if eligible.any():
        n_max = useful[eligible].max()
        fwhm_min = fwhm[eligible].min()
        score[eligible] = (
            useful[eligible] / n_max * (fwhm_min / fwhm[eligible]) ** fs.seeing_power
        )
    return eligible, score, relaxed


def select_centroid_frames(score: np.ndarray, fwhm_arcsec: np.ndarray, k: int) -> list[int]:
    """Indices of the ``k`` best-scoring frames, best first (ties: smaller FWHM, then index)."""
    score = np.asarray(score, dtype=np.float64)
    order = sorted(
        (i for i in range(len(score)) if np.isfinite(score[i])),
        key=lambda i: (-score[i], fwhm_arcsec[i], i),
    )
    return order[:k]


# ---------------------------------------------------------------------------
# 3. master positions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrameGeometry:
    """What :func:`build_master` needs to know about a frame's footprint."""

    wcs: WCS
    naxis1: int
    naxis2: int

    def covers(self, ra: np.ndarray, dec: np.ndarray, margin_px: float = 0.0) -> np.ndarray:
        x, y = _world_to_pixel(self.wcs, ra, dec)
        return (
            np.isfinite(x) & np.isfinite(y)
            & (x >= margin_px - 0.5) & (x < self.naxis1 - 0.5 - margin_px)
            & (y >= margin_px - 0.5) & (y < self.naxis2 - 0.5 - margin_px)
        )


def _world_to_pixel(wcs: WCS, ra: np.ndarray, dec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """0-based pixel coordinates (NaN where the inversion fails)."""
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    if ra.size == 0:
        return np.zeros(0), np.zeros(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, y = wcs.all_world2pix(ra, dec, 0, quiet=True)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class MasterCatalog:
    """Sky-fixed source positions for forced photometry."""

    number: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    n_det: np.ndarray
    n_cover: np.ndarray
    snr: np.ndarray
    flags: np.ndarray

    def __len__(self) -> int:
        return len(self.ra)


def build_master(
    tables: Sequence[Table], geometries: Sequence[FrameGeometry], fs: ForcedSettings
) -> MasterCatalog:
    """Master positions from the catalogues of the selected centroid frames (best first)."""
    k = len(tables)
    if k == 0 or k != len(geometries):
        msg = "build_master needs one geometry per selected catalogue (and at least one)"
        raise PhotometryError(msg)

    # --- gather the usable detections of each frame
    det_ra: list[np.ndarray] = []
    det_dec: list[np.ndarray] = []
    det_snr: list[np.ndarray] = []
    det_flags: list[np.ndarray] = []
    for t in tables:
        ra = np.asarray(t["RA"], dtype=np.float64)
        dec = np.asarray(t["DEC"], dtype=np.float64)
        flags = np.asarray(t["FLAGS"]).astype(np.int64)
        keep = np.isfinite(ra) & np.isfinite(dec) & ((flags & FLAG_BAD_PHOTOMETRY) == 0)
        det_ra.append(ra[keep])
        det_dec.append(dec[keep])
        det_snr.append(np.asarray(t["SNR"], dtype=np.float64)[keep])
        det_flags.append(flags[keep])

    # --- sky-match the frames, best frame first; closest claimant wins each group
    vec_sum = _unit_vectors(det_ra[0], det_dec[0])
    group_of: list[np.ndarray] = [np.arange(len(vec_sum))]
    chord = _chord(fs.match_radius_arcsec)
    for f in range(1, k):
        n_f = len(det_ra[f])
        gid = np.full(n_f, -1, dtype=np.int64)
        if n_f and len(vec_sum):
            rep = vec_sum / np.linalg.norm(vec_sum, axis=1)[:, None]
            dist, idx = cKDTree(rep).query(_unit_vectors(det_ra[f], det_dec[f]), k=1)
            cand = np.nonzero(dist <= chord)[0]
            order = cand[np.argsort(dist[cand], kind="stable")]
            _, first = np.unique(idx[order], return_index=True)
            won = order[first]
            gid[won] = idx[won]
        new = np.nonzero(gid < 0)[0]
        gid[new] = np.arange(len(vec_sum), len(vec_sum) + len(new))
        vec_new = _unit_vectors(det_ra[f], det_dec[f])
        vec_sum = np.vstack([vec_sum, np.zeros((len(new), 3))])
        np.add.at(vec_sum, gid, vec_new)
        # the representative of a group is the mean direction of its detections
        group_of.append(gid)
    n_groups = len(vec_sum)

    # --- (n_groups, k) tables of per-frame detections
    ra_t = np.full((n_groups, k), np.nan)
    dec_t = np.full((n_groups, k), np.nan)
    snr_t = np.full((n_groups, k), np.nan)
    flg_t = np.zeros((n_groups, k), dtype=np.int64)
    for f in range(k):
        g = group_of[f]
        ra_t[g, f], dec_t[g, f] = det_ra[f], det_dec[f]
        snr_t[g, f], flg_t[g, f] = det_snr[f], det_flags[f]
    n_det = np.isfinite(ra_t).sum(axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ra0 = np.nanmin(ra_t, axis=1)
        ra0 = np.where(np.isfinite(ra0), ra0, 0.0)
        dra = (ra_t - ra0[:, None] + 180.0) % 360.0 - 180.0  # offsets, safe across RA = 0
        dec_med = np.nanmedian(dec_t, axis=1)
        ra_med = (ra0 + np.nanmedian(dra, axis=1)) % 360.0
        snr_med = np.nanmedian(snr_t, axis=1)
    flags_or = np.bitwise_or.reduce(flg_t & _MASTER_FLAG_BITS, axis=1)

    # --- coverage and the detection requirement
    n_cover = np.zeros(n_groups, dtype=np.int64)
    for geom in geometries:
        n_cover += geom.covers(ra_med, dec_med).astype(np.int64)
    keep = (n_cover >= 1) & (n_det >= np.minimum(fs.min_frames, n_cover))

    ra_k, dec_k = ra_med[keep], dec_med[keep]
    nd_k, nc_k = n_det[keep], n_cover[keep]
    snr_k, fl_k = snr_med[keep], flags_or[keep]

    # --- remove near-duplicates (keep the better detected / brighter one)
    if len(ra_k) > 1:
        rank = np.empty(len(ra_k), dtype=np.int64)
        prio = np.lexsort((-np.nan_to_num(snr_k, nan=-1.0), -nd_k))
        rank[prio] = np.arange(len(ra_k))
        pairs = cKDTree(_unit_vectors(ra_k, dec_k)).query_pairs(
            _chord(fs.dedupe_radius_arcsec), output_type="ndarray"
        )
        if len(pairs):
            a, b = pairs[:, 0], pairs[:, 1]
            drop = np.where(rank[a] > rank[b], a, b)
            alive = np.ones(len(ra_k), dtype=bool)
            alive[drop] = False
            ra_k, dec_k, nd_k, nc_k, snr_k, fl_k = (
                v[alive] for v in (ra_k, dec_k, nd_k, nc_k, snr_k, fl_k)
            )

    order = np.lexsort((dec_k, ra_k))
    n_out = len(order)
    return MasterCatalog(
        number=np.arange(1, n_out + 1, dtype=np.int32),
        ra=ra_k[order],
        dec=dec_k[order],
        n_det=nd_k[order].astype(np.int32),
        n_cover=nc_k[order].astype(np.int32),
        snr=snr_k[order].astype(np.float32),
        flags=fl_k[order].astype(np.int32),
    )


def write_master(master: MasterCatalog, path) -> None:
    """Write ``forced_reference.csv`` (one row per master source)."""
    Table(
        {
            "NUMBER": master.number,
            "RA": master.ra,
            "DEC": master.dec,
            "N_DET": master.n_det,
            "N_COVER": master.n_cover,
            "SNR": master.snr,
            "FLAGS": master.flags,
        },
        names=MASTER_COLUMNS,
    ).write(path, format="csv", overwrite=True)


def read_master(path) -> MasterCatalog:
    """Inverse of :func:`write_master` (RA/Dec are written with full float64 precision)."""
    t = Table.read(path, format="ascii.csv")
    return MasterCatalog(
        number=np.asarray(t["NUMBER"], dtype=np.int32),
        ra=np.asarray(t["RA"], dtype=np.float64),
        dec=np.asarray(t["DEC"], dtype=np.float64),
        n_det=np.asarray(t["N_DET"], dtype=np.int32),
        n_cover=np.asarray(t["N_COVER"], dtype=np.int32),
        snr=np.asarray(t["SNR"], dtype=np.float32),
        flags=np.asarray(t["FLAGS"], dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# 4. forced measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ForcedResult:
    """The outcome of one :func:`measure_forced` call."""

    table: Table
    fwhm_px: float
    fwhm_arcsec: float
    sky_level: float
    sky_rms: float
    n_master: int
    n_measured: int
    n_offframe: int


def _source_mask(
    shape: tuple[int, int], x: np.ndarray, y: np.ndarray, radius_px: float
) -> np.ndarray:
    """Boolean image, True within ``radius_px`` of any (0-based) source position."""
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    reach = int(np.ceil(radius_px))
    dy, dx = np.mgrid[-reach : reach + 1, -reach : reach + 1]
    disk = np.hypot(dx, dy) <= radius_px
    dx, dy = dx[disk], dy[disk]
    xi = np.rint(x).astype(np.int64)
    yi = np.rint(y).astype(np.int64)
    for lo in range(0, len(xi), 20000):
        xs = xi[lo : lo + 20000, None] + dx[None, :]
        ys = yi[lo : lo + 20000, None] + dy[None, :]
        ok = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        mask[ys[ok], xs[ok]] = True
    return mask


def _local_background(
    sub: np.ndarray,
    exclude: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    r_in_px: float,
    r_out_px: float,
) -> np.ndarray:
    """Per-source local sky: SExtractor-style clipped estimate in a circular annulus.

    Same estimator as photutils' ``SourceCatalog.local_background`` (3 sigma clipped
    ``SExtractorBackground``) but on a circular annulus around the fixed position,
    with every second pixel in x and y (the sky noise is far below the aperture noise),
    excluding ``exclude`` pixels (bad pixels and other sources).  NaN where fewer than 10
    usable pixels remain (photutils uses 0 there; see :func:`_smooth_local_background`).
    """
    height, width = sub.shape
    reach = int(np.ceil(r_out_px))
    dy, dx = np.mgrid[-reach : reach + 1, -reach : reach + 1]
    rr = np.hypot(dx, dy)
    sel = (rr >= r_in_px) & (rr <= r_out_px) & (dx % 2 == 0) & (dy % 2 == 0)
    ox, oy = dx[sel], dy[sel]
    estimator = SExtractorBackground(SigmaClip(sigma=3.0, cenfunc="median", maxiters=10))
    out = np.full(len(x), np.nan)
    xi = np.rint(x).astype(np.int64)
    yi = np.rint(y).astype(np.int64)
    chunk = 4000
    for lo in range(0, len(x), chunk):
        xs = xi[lo : lo + chunk, None] + ox[None, :]
        ys = yi[lo : lo + chunk, None] + oy[None, :]
        inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        xs = np.clip(xs, 0, width - 1)
        ys = np.clip(ys, 0, height - 1)
        vals = sub[ys, xs].astype(np.float32)
        invalid = ~inside | exclude[ys, xs] | ~np.isfinite(vals)
        vals[invalid] = np.nan
        usable = (~invalid).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # NaN-padded rows: all-NaN slices, "clipped" notices
            est = np.asarray(estimator(vals, axis=1), dtype=np.float64)
        est[(usable < 10) | ~np.isfinite(est)] = np.nan
        out[lo : lo + chunk] = est
    return out


def _smooth_local_background(
    raw: np.ndarray, x: np.ndarray, y: np.ndarray, k: int
) -> np.ndarray:
    """Median of the raw annulus estimates over each source's ``k`` nearest sources (pixel space).

    Sources without a usable estimate (NaN) do not contribute; a source whose neighbours all
    lack one gets 0.  ``k <= 1`` skips the smoothing (NaN -> 0 only).
    """
    if k <= 1 or len(raw) < 2:
        return np.where(np.isfinite(raw), raw, 0.0)
    tree = cKDTree(np.column_stack([x, y]))
    k = min(k, len(raw))
    out = np.zeros(len(raw))
    for lo in range(0, len(raw), 20000):
        _, nb = tree.query(np.column_stack([x[lo : lo + 20000], y[lo : lo + 20000]]), k=k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(raw[nb], axis=1)
        out[lo : lo + 20000] = np.where(np.isfinite(med), med, 0.0)
    return out


def _dq_neighbourhood(
    dq: np.ndarray, x: np.ndarray, y: np.ndarray, radius_px: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Within ``radius_px``: saturated?, non-linear?, DQ-bit OR, fatal-pixel count per position."""
    n = len(x)
    sat = np.zeros(n, dtype=bool)
    nonlin = np.zeros(n, dtype=bool)
    ima = np.zeros(n, dtype=np.int32)
    nfatal = np.zeros(n, dtype=np.int32)
    rows, cols = np.nonzero(dq)
    if rows.size == 0 or n == 0:
        return sat, nonlin, ima, nfatal
    vals = dq[rows, cols].astype(np.int64)
    pix = np.column_stack([cols, rows]).astype(np.float64)
    pos = np.column_stack([x, y])
    for bit in range(int(DQ.CLIPPED).bit_length()):
        sel = (vals >> bit) & 1 == 1
        if not sel.any():
            continue
        d, _ = cKDTree(pix[sel]).query(pos, k=1, distance_upper_bound=radius_px)
        near = np.isfinite(d)
        ima[near] |= 1 << bit
        if (1 << bit) == int(DQ.SATURATED):
            sat = near
        elif (1 << bit) == int(DQ.NONLINEAR):
            nonlin = near
    fatal = (vals & int(FATAL_FLAGS)) != 0
    if fatal.any():
        nfatal = cKDTree(pix[fatal]).query_ball_point(pos, r=radius_px, return_length=True).astype(
            np.int32
        )
    return sat, nonlin, ima, nfatal


def _inherit_shape(
    frame_table: Table | None,
    ra: np.ndarray,
    dec: np.ndarray,
    radius_arcsec: float,
    scale: float,
) -> dict[str, np.ndarray]:
    """Shape/FWHM columns from the nearest per-frame detection (see the module docstring)."""
    n = len(ra)
    cols: dict[str, np.ndarray] = {c: np.full(n, np.nan) for c in _SHAPE_COLUMNS}
    fwhm_image = np.full(n, np.nan)
    if frame_table is not None and len(frame_table) and n:
        t_ra = np.asarray(frame_table["RA"], dtype=np.float64)
        t_dec = np.asarray(frame_table["DEC"], dtype=np.float64)
        good = np.isfinite(t_ra) & np.isfinite(t_dec)
        rows = np.nonzero(good)[0]
        idx, ok = _nearest(ra, dec, t_ra[good], t_dec[good], radius_arcsec)
        src = rows[idx[ok]]
        for c in _SHAPE_COLUMNS:
            cols[c][ok] = np.asarray(frame_table[c], dtype=np.float64)[src]
        fw = np.asarray(frame_table["FWHM_IMAGE"], dtype=np.float64)[src]
        fwhm_image[ok] = np.where(np.isfinite(fw) & (fw > 0), fw, np.nan)
    cols["FWHM_IMAGE"] = fwhm_image
    cols["FWHM_WORLD"] = fwhm_image * scale / 3600.0
    cols["FWHM"] = fwhm_image * scale
    return cols


def measure_forced(
    sci: np.ndarray,
    err: np.ndarray,
    dq: np.ndarray,
    *,
    master: MasterCatalog,
    header: fits.Header,
    settings: Settings,
    profile: InstrumentProfile,
    wcs: WCS,
    frame_table: Table | None = None,
    fwhm_px: float | None = None,
) -> ForcedResult:
    """Aperture photometry at the master sky positions of one reduced frame, no recentring.

    ``frame_table`` is the frame's own ``measure_sources`` catalogue (optional); it only
    supplies the shape columns.  ``fwhm_px`` skips the FWHM measurement when the
    caller already has it.  Raises :class:`~robo43.exceptions.PhotometryError` for
    inconsistent array shapes or a missing celestial WCS.
    """
    sci = np.asarray(sci)
    err = np.asarray(err)
    dq = np.asarray(dq)
    if sci.shape != err.shape or sci.shape != dq.shape:
        msg = f"sci/err/dq shape mismatch: sci={sci.shape}, err={err.shape}, dq={dq.shape}"
        raise PhotometryError(msg)
    if wcs is None or not wcs.has_celestial:
        msg = "forced photometry needs a celestial WCS"
        raise PhotometryError(msg)

    ps = settings.photometry
    fs = settings.forced
    scale = pixel_scale_arcsec(header, wcs, profile)
    height, width = sci.shape
    radii_px = [r / scale for r in ps.aperture_radii_arcsec]
    n_aper = len(radii_px)
    max_radius_px = max(radii_px)

    bad_mask = (dq.astype(np.int64) & _BAD_PIXEL_FLAGS) != 0
    bkg = estimate_sky_background(sci, bad_mask, ps)
    sky_level = float(np.median(bkg.background))
    sky_rms = float(np.median(bkg.background_rms))
    sub = sci.astype(np.float64)
    sub -= bkg.background
    del bkg

    if fwhm_px is None:
        med, sd = estimate_background(sci)
        fwhm_px = measure_fwhm(sci, med, sd, fallback_px=ps.seeing_fwhm_arcsec / scale)
    fwhm_px = float(fwhm_px)
    fwhm_arcsec = fwhm_px * scale

    # --- sky-fixed positions through this frame's WCS
    x_all, y_all = _world_to_pixel(wcs, master.ra, master.dec)
    on_frame = (
        np.isfinite(x_all) & np.isfinite(y_all)
        & (x_all >= -0.5) & (x_all < width - 0.5) & (y_all >= -0.5) & (y_all < height - 0.5)
    )
    row_sel = np.arange(len(master)) if fs.keep_offframe else np.nonzero(on_frame)[0]
    n_rows = len(row_sel)
    meas = on_frame[row_sel]  # rows that are actually measured
    x, y = x_all[row_sel], y_all[row_sel]
    xm, ym = x[meas], y[meas]

    flux = np.full((n_rows, n_aper), np.nan)
    fluxerr = np.full((n_rows, n_aper), np.nan)
    bkg_local = np.full(n_rows, np.nan)
    flux_max = np.full(n_rows, np.nan)
    flags = np.zeros(n_rows, dtype=np.int32)
    imaflags = np.zeros(n_rows, dtype=np.int32)
    nimaflags = np.zeros(n_rows, dtype=np.int32)

    if xm.size:
        finite_px = np.isfinite(sub)
        px_mask = None if finite_px.all() else ~finite_px
        err64 = err.astype(np.float64) if err.dtype != np.float64 else err
        positions = np.column_stack([xm, ym])
        near_edge = (
            (xm - max_radius_px < 0) | (xm + max_radius_px >= width)
            | (ym - max_radius_px < 0) | (ym + max_radius_px >= height)
        )

        # optional local sky (relative to the 2-D sky model): circular annulus, other sources masked
        if fs.local_bkg_width_arcsec > 0:
            exclude = bad_mask | _source_mask(
                (height, width), xm, ym, max(2.0, fs.source_mask_fwhm * fwhm_px)
            )
            r_in = max_radius_px + fs.local_bkg_gap_arcsec / scale
            r_out = r_in + fs.local_bkg_width_arcsec / scale
            lb = _smooth_local_background(
                _local_background(sub, exclude, xm, ym, r_in, r_out), xm, ym, fs.local_bkg_smooth_k
            )
            del exclude
        else:
            lb = np.zeros(xm.size)

        f_m = np.full((xm.size, n_aper), np.nan)
        e_m = np.full((xm.size, n_aper), np.nan)
        for j, radius_px in enumerate(radii_px):
            aper = CircularAperture(positions, radius_px)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tab = aperture_photometry(sub, aper, error=err64, mask=px_mask, method="exact")
            area = np.full(xm.size, np.pi * radius_px**2)
            if near_edge.any():
                area[near_edge] = CircularAperture(positions[near_edge], radius_px).area_overlap(
                    sub, mask=px_mask
                )
            f_m[:, j] = np.asarray(tab["aperture_sum"], dtype=np.float64) - lb * area
            e_m[:, j] = np.asarray(tab["aperture_sum_err"], dtype=np.float64)
        flux[meas], fluxerr[meas] = f_m, e_m
        bkg_local[meas] = lb

        # peak pixel of the sky-subtracted frame within +-1 pixel of the position
        xi = np.rint(xm).astype(np.int64)
        yi = np.rint(ym).astype(np.int64)
        peak = np.full(xm.size, -np.inf)
        for ddx in (-1, 0, 1):
            for ddy in (-1, 0, 1):
                peak = np.maximum(
                    peak, sub[np.clip(yi + ddy, 0, height - 1), np.clip(xi + ddx, 0, width - 1)]
                )
        flux_max[meas] = peak

        sat, nonlin, ima, nfatal = _dq_neighbourhood(
            dq, xm, ym, fs.flag_radius_arcsec / scale
        )
        fm = np.zeros(xm.size, dtype=np.int32)
        fm[sat] |= FLAG_SATURATED
        fm[nonlin] |= FLAG_NONLINEAR
        margin = ps.edge_margin_px
        fm[
            (xm < margin) | (xm > width - 1 - margin) | (ym < margin) | (ym > height - 1 - margin)
        ] |= FLAG_TRUNCATED
        fm[near_edge] |= FLAG_APERTURE_INCOMPLETE
        flags[meas] = fm
        imaflags[meas], nimaflags[meas] = ima, nfatal
    del sub

    flags |= (master.flags[row_sel] & _MASTER_FLAG_BITS).astype(np.int32)
    off = ~meas
    flags[off] |= FLAG_TRUNCATED | FLAG_APERTURE_INCOMPLETE | FLAG_BAD_PHOTOMETRY
    flags[meas & ~np.isfinite(flux[:, 0])] |= FLAG_BAD_PHOTOMETRY

    k_snr = nearest_radius_index(ps.aperture_radii_arcsec, fs.snr_aperture_arcsec)
    flux_k, err_k = flux[:, k_snr], fluxerr[:, k_snr]
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(err_k > 0, flux_k / err_k, 0.0)
    snr[off] = np.nan

    mag_k, magerr_k = _magnitudes(flux_k, err_k, ps.instrumental_zp)
    mag_aper, magerr_aper = _magnitudes(flux, fluxerr, ps.instrumental_zp)

    ra, dec = master.ra[row_sel], master.dec[row_sel]
    shape = _inherit_shape(frame_table, ra, dec, fs.shape_match_radius_arcsec, scale)

    data: dict[str, np.ndarray] = {
        "NUMBER": master.number[row_sel].astype(np.int32),
        "X_IMAGE": x + 1.0,
        "Y_IMAGE": y + 1.0,
        "ALPHA_J2000": ra,
        "DELTA_J2000": dec,
        "FLUX_AUTO": flux_k,
        "FLUXERR_AUTO": err_k,
        "MAG_AUTO": mag_k,
        "MAGERR_AUTO": magerr_k,
        "FLUX_APER": flux,
        "FLUXERR_APER": fluxerr,
        "MAG_APER": mag_aper,
        "MAGERR_APER": magerr_aper,
        "BACKGROUND": bkg_local,
        "FLUX_MAX": flux_max,
        "FLAGS": flags,
        "IMAFLAGS_ISO": imaflags,
        "NIMAFLAGS_ISO": nimaflags,
        "SNR": snr,
        "RA": ra.copy(),
        "DEC": dec.copy(),
        "FLUX": flux_k.copy(),
        "FLUX_ERR": err_k.copy(),
        "MAG": mag_k.copy(),
        "MAG_ERR": magerr_k.copy(),
        **shape,
    }
    columns: dict[str, np.ndarray] = {}
    for name in COLUMN_NAMES:
        values = np.asarray(data[name])
        dtype = np.float32 if name in _VECTOR_COLUMNS else np.dtype(_SCALAR_DTYPES[name])
        columns[name] = values.astype(dtype)
    table = Table(columns, names=COLUMN_NAMES)

    return ForcedResult(
        table=table,
        fwhm_px=fwhm_px,
        fwhm_arcsec=fwhm_arcsec,
        sky_level=sky_level,
        sky_rms=sky_rms,
        n_master=len(master),
        n_measured=int(meas.sum()),
        n_offframe=int(np.count_nonzero(~on_frame)),
    )
