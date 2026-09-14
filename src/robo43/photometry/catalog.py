"""Source-catalog photometry: detection and measurement on a reduced ``_proc.fits`` frame.

Given a ``SCI``/``ERR``/``DQ`` triple from :class:`robo43.products.mef.ReducedProduct`
(``SCI`` already background-included, in electrons -- ``BUNIT='electron'`` -- with
``ERR`` the 1-sigma uncertainty in the same units), this module runs photutils
detection and measurement and returns a source table plus a
:class:`~robo43.products.catalog.CatalogMetrics` header block, in the exact
column vocabulary :data:`robo43.products.catalog.CATALOG_COLUMNS` already
defines. ``ERR`` is passed straight to photutils as its ``error=`` array;
Poisson noise is never re-derived from gain here -- the ERR plane already
carries it, which is why ``gain_e_per_adu`` is accepted only to be recorded
on the metrics header (``GAIN``), never used in the noise model.

Detection uses **photutils** exclusively (2.0+; ``sep`` and SExtractor/sewpy
are not used here): :class:`~photutils.background.Background2D` for the sky,
:func:`~robo43.astrometry.centroids.measure_fwhm` (reused, not reimplemented)
for the seeing estimate, :func:`~photutils.segmentation.make_2dgaussian_kernel`
as the detection-convolution filter -- the photutils equivalent of a
SExtractor ``.conv`` file, which is why ``settings.photometry.detect_filter``
(a SExtractor filter-file *name*) is declared in :class:`~robo43.config.PhotometrySettings`
but not consumed anywhere in this module -- then
:func:`~photutils.segmentation.detect_sources`,
:func:`~photutils.segmentation.deblend_sources`, and
:class:`~photutils.segmentation.SourceCatalog` for Kron and circular-aperture
photometry.

Bad-pixel masking (background estimation and detection alike) uses only
``DQ.SATURATED | DQ.NO_DATA | DQ.DEAD | DQ.HOT | DQ.BAD_COLUMN``.
``DQ.COSMIC_RAY`` is deliberately excluded: :mod:`robo43.astrometry.centroids`
documents that bit as flagging roughly a third of all pixels spuriously, so
masking on it here would discard good sources, not bad pixels.

Centroid-uncertainty columns (``ERRX2_IMAGE``, ``ERRY2_IMAGE``,
``ERRA_WORLD``, ``ERRB_WORLD``) are deliberately **not emitted**. photutils'
``SourceCatalog`` has no centroid-uncertainty property -- ``covar_sigx2``/
``covar_sigy2`` are the source's own second moments (its spatial extent), not
the variance of the centroid estimate, and writing them into those columns
would silently record a value wrong by orders of magnitude while looking
plausible (the same class of silent unit/semantics error this pipeline was
already bitten by with the ``GAIN`` header keyword -- see
:mod:`robo43.calib.gain`). All four columns are optional in
:data:`~robo43.products.catalog.CATALOG_COLUMNS`; a later PSF-fitting stage
that actually fits a centroid covariance is the right place to fill them.

``NUMBER``, ``X_IMAGE``, ``Y_IMAGE`` are 1-based FITS pixel coordinates
(photutils' own ``xcentroid``/``ycentroid`` are 0-based; this module adds 1).

``RA``, ``DEC``, ``FWHM``, ``FLUX``, ``FLUX_ERR``, ``MAG``, ``MAG_ERR`` are
plain aliases -- exact copies of ``ALPHA_J2000``, ``DELTA_J2000``,
``FWHM_IMAGE`` (converted to arcsec), ``FLUX_AUTO``, ``FLUXERR_AUTO``,
``MAG_AUTO``, ``MAGERR_AUTO`` respectively. They are not independent
measurements; they exist only because a plainer name was requested
downstream.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import photutils
from astropy.convolution import convolve
from astropy.stats import SigmaClip
from astropy.table import Table
from astropy.wcs.utils import proj_plane_pixel_scales
from photutils.background import Background2D, MedianBackground
from scipy.ndimage import binary_dilation
from photutils.segmentation import (
    SourceCatalog,
    deblend_sources,
    detect_sources,
    make_2dgaussian_kernel,
)

import robo43
from robo43.astrometry.centroids import estimate_background, measure_fwhm
from robo43.exceptions import PhotometryError
from robo43.products.catalog import CatalogMetrics
from robo43.products.dq import DQ, FATAL_FLAGS

if TYPE_CHECKING:
    from astropy.io import fits
    from astropy.wcs import WCS

    from robo43.config import Settings
    from robo43.instruments.profile import InstrumentProfile

logger = logging.getLogger(__name__)

__all__ = [
    "FLAG_APERTURE_INCOMPLETE",
    "FLAG_BAD_PHOTOMETRY",
    "FLAG_DEBLENDED",
    "FLAG_NEIGHBOR",
    "FLAG_SATURATED",
    "FLAG_TRUNCATED",
    "PhotometryResult",
    "measure_sources",
    "pixel_scale_arcsec",
]

#: DQ bits masked out of background estimation and detection. DQ.COSMIC_RAY
#: is deliberately excluded -- see the module docstring.
_BAD_PIXEL_FLAGS = int(DQ.SATURATED | DQ.NO_DATA | DQ.DEAD | DQ.HOT | DQ.BAD_COLUMN)

#: SExtractor-compatible FLAGS bits. Bits 1 and 2 are always set together by
#: this module (a deblended child is, by definition, one of >1 siblings that
#: shared a single pre-deblend parent segment) since nothing here computes a
#: separate "has a neighbour" test independent of deblending.
FLAG_NEIGHBOR = 1
FLAG_DEBLENDED = 2
FLAG_SATURATED = 4
FLAG_TRUNCATED = 8
FLAG_APERTURE_INCOMPLETE = 16
FLAG_BAD_PHOTOMETRY = 32

#: Scalar output columns and their FITS-table-friendly numpy dtypes. Kept
#: separate from robo43.products.catalog.CATALOG_COLUMNS (which records FITS
#: format codes for the whole allow-list, most of which this module never
#: emits) so an empty/zero-row table can be built with exactly the right
#: dtype per column without depending on that module's format-code strings.
_SCALAR_DTYPES: dict[str, str] = {
    "NUMBER": "i4",
    "X_IMAGE": "f8",
    "Y_IMAGE": "f8",
    "ALPHA_J2000": "f8",
    "DELTA_J2000": "f8",
    "FLUX_AUTO": "f4",
    "FLUXERR_AUTO": "f4",
    "MAG_AUTO": "f4",
    "MAGERR_AUTO": "f4",
    "BACKGROUND": "f4",
    "FLUX_MAX": "f4",
    "FLUX_RADIUS": "f4",
    "FWHM_IMAGE": "f4",
    "FWHM_WORLD": "f4",
    "A_IMAGE": "f4",
    "B_IMAGE": "f4",
    "THETA_IMAGE": "f4",
    "ELONGATION": "f4",
    "ELLIPTICITY": "f4",
    "FLAGS": "i4",
    "IMAFLAGS_ISO": "i4",
    "NIMAFLAGS_ISO": "i4",
    "SNR": "f4",
    "RA": "f8",
    "DEC": "f8",
    "FWHM": "f4",
    "FLUX": "f4",
    "FLUX_ERR": "f4",
    "MAG": "f4",
    "MAG_ERR": "f4",
}

#: The four vector (N, n_aper) columns; always float32.
_VECTOR_COLUMNS: tuple[str, ...] = ("FLUX_APER", "FLUXERR_APER", "MAG_APER", "MAGERR_APER")

#: Every column this module emits, in write order. Every name here must be a
#: key of robo43.products.catalog.CATALOG_COLUMNS.
COLUMN_NAMES: tuple[str, ...] = (
    "NUMBER", "X_IMAGE", "Y_IMAGE", "ALPHA_J2000", "DELTA_J2000",
    "FLUX_AUTO", "FLUXERR_AUTO", "MAG_AUTO", "MAGERR_AUTO",
    "FLUX_APER", "FLUXERR_APER", "MAG_APER", "MAGERR_APER",
    "BACKGROUND", "FLUX_MAX", "FLUX_RADIUS", "FWHM_IMAGE", "FWHM_WORLD",
    "A_IMAGE", "B_IMAGE", "THETA_IMAGE", "ELONGATION", "ELLIPTICITY",
    "FLAGS", "IMAFLAGS_ISO", "NIMAFLAGS_ISO", "SNR",
    "RA", "DEC", "FWHM", "FLUX", "FLUX_ERR", "MAG", "MAG_ERR",
)


@dataclass(frozen=True, slots=True)
class PhotometryResult:
    """The outcome of one :func:`measure_sources` call."""

    table: Table
    metrics: CatalogMetrics
    fwhm_px: float
    fwhm_arcsec: float
    sky_level: float
    sky_rms: float
    n_sources: int
    #: False when deblend_sources could not run because scikit-image is not
    #: installed (a runtime dependency of its watershed algorithm that
    #: photutils does not declare itself) -- blended sources were then left
    #: merged into single entries instead of being split.
    deblended: bool = True


def pixel_scale_arcsec(header: fits.Header, wcs: WCS | None, profile: InstrumentProfile) -> float:
    """Plate scale in arcsec/pixel, preferring the astrometry stage's own measurement.

    Order: the ``ASTRSCL`` header card (written by
    :func:`robo43.astrometry.wcs_header.astrometry_cards` after a successful
    solve), then ``wcs`` via :func:`~astropy.wcs.utils.proj_plane_pixel_scales`,
    then ``profile.pixel_scale_arcsec``. Never returns 0 or NaN -- raises
    :class:`~robo43.exceptions.PhotometryError` if no source yields a finite,
    positive value.
    """
    raw = header.get("ASTRSCL")
    if raw is not None:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value) and value > 0:
            return value

    if wcs is not None:
        try:
            scales = proj_plane_pixel_scales(wcs) * 3600.0
            value = float(np.mean(scales))
        except Exception:  # noqa: BLE001 -- any WCS malformation just falls through
            value = float("nan")
        if np.isfinite(value) and value > 0:
            return value

    value = profile.pixel_scale_arcsec
    if value is not None and np.isfinite(value) and value > 0:
        return float(value)

    msg = (
        "cannot determine a pixel scale: no usable ASTRSCL header card, no "
        "usable WCS, and no profile.pixel_scale_arcsec"
    )
    raise PhotometryError(msg)


def _empty_table(n_aper: int) -> Table:
    columns: dict[str, np.ndarray] = {
        name: np.zeros(0, dtype=dtype) for name, dtype in _SCALAR_DTYPES.items()
    }
    for name in _VECTOR_COLUMNS:
        columns[name] = np.zeros((0, n_aper), dtype=np.float32)
    return Table(columns, names=COLUMN_NAMES)


def _magnitudes(flux: np.ndarray, fluxerr: np.ndarray, zp: float) -> tuple[np.ndarray, np.ndarray]:
    """MAG = zp - 2.5*log10(flux), MAGERR = 1.0857*fluxerr/flux, NaN where flux <= 0.

    Works elementwise on 1-D (AUTO) or 2-D (APER) arrays alike.
    """
    positive = np.isfinite(flux) & (flux > 0)
    safe_flux = np.where(positive, flux, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(positive, zp - 2.5 * np.log10(safe_flux), np.nan)
        magerr = np.where(positive, 1.0857 * fluxerr / safe_flux, np.nan)
    return mag.astype(np.float32), magerr.astype(np.float32)


def _build_metrics(
    *,
    settings: Settings,
    profile: InstrumentProfile,
    header: fits.Header,
    kernel_desc: str,
    sky_level: float,
    sky_rms: float,
    fwhm_px: float,
    fwhm_arcsec: float,
    scale: float,
    n_sources: int,
    n_satur: int,
    n_badpix: int,
    gain_e_per_adu: float | None,
    deblended: bool,
) -> CatalogMetrics:
    ps = settings.photometry
    airmass_raw = header.get("AIRMASS")
    try:
        airmass = float(airmass_raw) if airmass_raw is not None else None
    except (TypeError, ValueError):
        airmass = None

    # debl_nth=0 is a sentinel meaning "deblending was skipped" (scikit-image
    # unavailable; see PhotometryResult.deblended) rather than "0 levels were
    # configured" -- ps.deblend_nthresh is always >= 1 when deblending did
    # run. debl_minc is omitted (None) in that case since contrast was never
    # applied.
    debl_nth = ps.deblend_nthresh if deblended else 0
    debl_minc = ps.deblend_mincont if deblended else None

    return CatalogMetrics(
        cat_soft="photutils",
        cat_softv=photutils.__version__,
        cat_date=datetime.now(UTC).isoformat(timespec="seconds"),
        pipe_ver=robo43.__version__,
        det_thrsh=ps.detect_thresh,
        det_minar=ps.detect_minarea,
        det_filt=kernel_desc,
        debl_nth=debl_nth,
        debl_minc=debl_minc,
        back_size=ps.back_size,
        back_filt=ps.back_filtersize,
        n_sources=n_sources,
        n_satur=n_satur,
        n_badpix=n_badpix,
        fwhm_pix=fwhm_px,
        fwhm_arc=fwhm_arcsec,
        sky_lev=sky_level,
        sky_rms=sky_rms,
        pix_scale=scale,
        airmass=airmass,
        aper_rad=tuple(ps.aperture_radii_arcsec),
        magsys="instrumental",
        photomet=False,
        gain=gain_e_per_adu,
        rdnoise=profile.read_noise_e,
        sat_level=profile.saturation_adu,
    )


def measure_sources(
    sci: np.ndarray,
    err: np.ndarray,
    dq: np.ndarray,
    *,
    header: fits.Header,
    settings: Settings,
    profile: InstrumentProfile,
    wcs: WCS | None = None,
    gain_e_per_adu: float | None = None,
) -> PhotometryResult:
    """Detect and measure sources on one reduced (background-included) frame.

    Never raises on a degenerate frame -- a frame with no detections returns
    a zero-row table (every column present, correctly typed) and
    ``n_sources=0``. Raises :class:`~robo43.exceptions.PhotometryError` only
    for genuinely invalid input: ``sci``/``err``/``dq`` shape mismatch, or a
    pixel scale that cannot be determined at all (see
    :func:`pixel_scale_arcsec`).
    """
    sci = np.asarray(sci)
    err = np.asarray(err)
    dq = np.asarray(dq)
    if sci.shape != err.shape or sci.shape != dq.shape:
        msg = f"sci/err/dq shape mismatch: sci={sci.shape}, err={err.shape}, dq={dq.shape}"
        raise PhotometryError(msg)

    ps = settings.photometry
    scale = pixel_scale_arcsec(header, wcs, profile)
    height, width = sci.shape

    bad_mask = (dq.astype(np.int64) & _BAD_PIXEL_FLAGS) != 0

    bkg = Background2D(
        sci.astype(np.float64),
        box_size=ps.back_size,
        filter_size=ps.back_filtersize,
        sigma_clip=SigmaClip(sigma=3.0),
        bkg_estimator=MedianBackground(),
        mask=bad_mask,
        exclude_percentile=50.0,
    )
    sky_level = float(np.median(bkg.background))
    sky_rms = float(np.median(bkg.background_rms))

    med, sd = estimate_background(sci)
    fallback_px = ps.seeing_fwhm_arcsec / scale
    fwhm_px = measure_fwhm(sci, med, sd, fallback_px=fallback_px)
    fwhm_arcsec = fwhm_px * scale

    kernel_size = max(5, int(round(2.5 * fwhm_px)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = make_2dgaussian_kernel(fwhm=fwhm_px, size=kernel_size)
    kernel_desc = f"gauss_{fwhm_px:.2f}px"

    subtracted = sci.astype(np.float64) - bkg.background
    convolved = convolve(subtracted, kernel)

    segm = detect_sources(
        convolved,
        threshold=ps.detect_thresh * bkg.background_rms,
        npixels=ps.detect_minarea,
        mask=bad_mask,
    )

    n_aper = len(ps.aperture_radii_arcsec)

    if segm is None:
        metrics = _build_metrics(
            settings=settings, profile=profile, header=header, kernel_desc=kernel_desc,
            sky_level=sky_level, sky_rms=sky_rms, fwhm_px=fwhm_px, fwhm_arcsec=fwhm_arcsec,
            scale=scale, n_sources=0, n_satur=0, n_badpix=0, gain_e_per_adu=gain_e_per_adu,
            deblended=True,
        )
        return PhotometryResult(
            table=_empty_table(n_aper), metrics=metrics, fwhm_px=fwhm_px,
            fwhm_arcsec=fwhm_arcsec, sky_level=sky_level, sky_rms=sky_rms, n_sources=0,
            deblended=True,
        )

    # deblend_sources' watershed path imports scikit-image lazily, and only
    # when a source actually needs watershed splitting -- so this can fail well into
    # a batch run, on the first genuinely blended source, rather than at
    # import time. scikit-image is a runtime dependency of that watershed
    # step that photutils does not declare itself (see pyproject.toml).
    # Catching narrowly: only a missing dependency is degraded gracefully; a
    # genuine deblending failure on real data must still propagate.
    try:
        segm_deblend = deblend_sources(
            convolved,
            segm,
            npixels=ps.detect_minarea,
            nlevels=ps.deblend_nthresh,
            contrast=ps.deblend_mincont,
            progress_bar=False,
        )
        deblended = True
    except (ImportError, ModuleNotFoundError) as exc:
        logger.warning(
            "deblending skipped because scikit-image is not installed (%s); "
            "blended sources will be merged into single entries.",
            exc,
        )
        segm_deblend = segm
        deblended = False

    localbkg_width = int(min(50, max(5, round(2.5 * fwhm_px))))

    # Deliberately NOT passing convolved_data=convolved here (a deviation
    # from a literal photutils "pass the matched-filter image" recipe):
    # photutils computes every second-moment-derived shape property
    # (semimajor_sigma/semiminor_sigma/fwhm, and therefore FWHM_IMAGE,
    # A_IMAGE, B_IMAGE, ELONGATION, ELLIPTICITY) from convolved_data
    # unconditionally when it is given, which quadrature-broadens the
    # measured FWHM by roughly sqrt(2) whenever the detection kernel's own
    # FWHM matches the source's (exactly the case here, since the kernel is
    # sized from the same fwhm_px). Omitting it (falls back to `subtracted`
    # internally) reproduces SExtractor's own convention, where the
    # detection filter drives thresholding/segmentation only and isophotal
    # shape parameters are measured on the unfiltered image. Kron and
    # circular-aperture flux are unaffected either way -- both are always
    # integrated over `subtracted`, never over convolved_data.
    cat = SourceCatalog(
        subtracted,
        segm_deblend,
        error=err.astype(np.float64),
        wcs=wcs,
        localbkg_width=localbkg_width,
    )

    n = len(cat.labels)

    x_image = np.asarray(cat.xcentroid, dtype=np.float64) + 1.0
    y_image = np.asarray(cat.ycentroid, dtype=np.float64) + 1.0
    flux_auto = np.asarray(cat.kron_flux, dtype=np.float64)
    fluxerr_auto = np.asarray(cat.kron_fluxerr, dtype=np.float64)
    background = np.asarray(cat.local_background, dtype=np.float64)
    flux_max = np.asarray(cat.max_value, dtype=np.float64)
    flux_radius = np.asarray(cat.fluxfrac_radius(0.5).value, dtype=np.float64)
    fwhm_image = np.asarray(cat.fwhm.value, dtype=np.float64)
    a_image = np.asarray(cat.semimajor_sigma.value, dtype=np.float64)
    b_image = np.asarray(cat.semiminor_sigma.value, dtype=np.float64)
    theta_image = np.asarray(cat.orientation.value, dtype=np.float64)
    elongation = np.asarray(cat.elongation.value, dtype=np.float64)
    ellipticity = np.asarray(cat.ellipticity.value, dtype=np.float64)

    if wcs is not None:
        sky = cat.sky_centroid
        alpha = np.asarray(sky.ra.deg, dtype=np.float64)
        delta = np.asarray(sky.dec.deg, dtype=np.float64)
    else:
        alpha = np.full(n, np.nan)
        delta = np.full(n, np.nan)

    radii_px = [r / scale for r in ps.aperture_radii_arcsec]
    flux_aper = np.zeros((n, n_aper), dtype=np.float64)
    fluxerr_aper = np.zeros((n, n_aper), dtype=np.float64)
    for j, radius_px in enumerate(radii_px):
        f, ferr = cat.circular_photometry(radius_px)
        flux_aper[:, j] = f
        fluxerr_aper[:, j] = ferr

    fwhm_world = fwhm_image * scale / 3600.0
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(fluxerr_auto > 0, flux_auto / fluxerr_auto, 0.0)

    mag_auto, magerr_auto = _magnitudes(flux_auto, fluxerr_auto, ps.instrumental_zp)
    mag_aper, magerr_aper = _magnitudes(flux_aper, fluxerr_aper, ps.instrumental_zp)

    flags = np.zeros(n, dtype=np.int32)
    imaflags_iso = np.zeros(n, dtype=np.int32)
    nimaflags_iso = np.zeros(n, dtype=np.int32)

    margin = ps.edge_margin_px
    max_radius_px = radii_px[-1] if radii_px else 0.0
    bbox_xmin = np.asarray(cat.bbox_xmin)
    bbox_xmax = np.asarray(cat.bbox_xmax)
    bbox_ymin = np.asarray(cat.bbox_ymin)
    bbox_ymax = np.asarray(cat.bbox_ymax)

    parents: dict[int, list[int]] = {}
    for i, label in enumerate(cat.labels):
        sl = cat.slices[i]
        seg_sub = segm_deblend.data[sl]
        seg_here = seg_sub == label
        # A saturated (or otherwise bad-pixel-masked) pixel was excluded from
        # `bad_mask` at detection time, so it was never assigned this
        # source's label -- it shows up as a label-0 hole inside the
        # segment, not as part of it. One iteration of binary dilation
        # closes any such single/few-pixel hole so IMAFLAGS_ISO/
        # NIMAFLAGS_ISO/FLAG_SATURATED can actually see it; it also grows
        # the footprint by one pixel at the true (non-hole) segment
        # boundary, a deliberate, documented over-inclusion in exchange for
        # not silently missing saturation at a source's core.
        seg_here_dilated = binary_dilation(seg_here, iterations=1)
        dq_sub = dq[sl][seg_here_dilated]

        combined = int(np.bitwise_or.reduce(dq_sub)) if dq_sub.size else 0
        imaflags_iso[i] = combined
        nimaflags_iso[i] = int(np.count_nonzero(dq_sub.astype(np.int64) & int(FATAL_FLAGS)))
        if combined & int(DQ.SATURATED):
            flags[i] |= FLAG_SATURATED

        pre_sub = segm.data[sl][seg_here]  # parent lookup uses the true segment only
        parent = int(pre_sub[0]) if pre_sub.size else 0
        parents.setdefault(parent, []).append(i)

        if (
            bbox_xmin[i] < margin
            or bbox_xmax[i] > width - 1 - margin
            or bbox_ymin[i] < margin
            or bbox_ymax[i] > height - 1 - margin
        ):
            flags[i] |= FLAG_TRUNCATED

        if max_radius_px > 0:
            xi, yi = x_image[i] - 1.0, y_image[i] - 1.0
            if (
                xi - max_radius_px < 0
                or xi + max_radius_px >= width
                or yi - max_radius_px < 0
                or yi + max_radius_px >= height
            ):
                flags[i] |= FLAG_APERTURE_INCOMPLETE

    for members in parents.values():
        if len(members) > 1:
            for i in members:
                flags[i] |= FLAG_NEIGHBOR | FLAG_DEBLENDED

    bad_phot = ~np.isfinite(flux_auto) | (flux_auto <= 0)
    flags[bad_phot] |= FLAG_BAD_PHOTOMETRY

    n_satur = int(np.count_nonzero(flags & FLAG_SATURATED))
    # A source counts as touching a bad pixel when any pixel in its segment
    # carries a FATAL_FLAGS bit -- the same NIMAFLAGS_ISO this table also
    # reports per-source (mirrors quality.py's BADPIX cut, IMAFLAGS_ISO != 0).
    n_badpix = int(np.count_nonzero(nimaflags_iso > 0))

    table = Table(
        {
            "NUMBER": np.arange(1, n + 1, dtype=np.int32),
            "X_IMAGE": x_image,
            "Y_IMAGE": y_image,
            "ALPHA_J2000": alpha,
            "DELTA_J2000": delta,
            "FLUX_AUTO": flux_auto.astype(np.float32),
            "FLUXERR_AUTO": fluxerr_auto.astype(np.float32),
            "MAG_AUTO": mag_auto,
            "MAGERR_AUTO": magerr_auto,
            "FLUX_APER": flux_aper.astype(np.float32),
            "FLUXERR_APER": fluxerr_aper.astype(np.float32),
            "MAG_APER": mag_aper,
            "MAGERR_APER": magerr_aper,
            "BACKGROUND": background.astype(np.float32),
            "FLUX_MAX": flux_max.astype(np.float32),
            "FLUX_RADIUS": flux_radius.astype(np.float32),
            "FWHM_IMAGE": fwhm_image.astype(np.float32),
            "FWHM_WORLD": fwhm_world.astype(np.float32),
            "A_IMAGE": a_image.astype(np.float32),
            "B_IMAGE": b_image.astype(np.float32),
            "THETA_IMAGE": theta_image.astype(np.float32),
            "ELONGATION": elongation.astype(np.float32),
            "ELLIPTICITY": ellipticity.astype(np.float32),
            "FLAGS": flags,
            "IMAFLAGS_ISO": imaflags_iso,
            "NIMAFLAGS_ISO": nimaflags_iso,
            "SNR": snr.astype(np.float32),
            "RA": alpha.copy(),
            "DEC": delta.copy(),
            "FWHM": (fwhm_image * scale).astype(np.float32),
            "FLUX": flux_auto.astype(np.float32),
            "FLUX_ERR": fluxerr_auto.astype(np.float32),
            "MAG": mag_auto.copy(),
            "MAG_ERR": magerr_auto.copy(),
        },
        names=COLUMN_NAMES,
    )

    metrics = _build_metrics(
        settings=settings, profile=profile, header=header, kernel_desc=kernel_desc,
        sky_level=sky_level, sky_rms=sky_rms, fwhm_px=fwhm_px, fwhm_arcsec=fwhm_arcsec,
        scale=scale, n_sources=n, n_satur=n_satur, n_badpix=n_badpix,
        gain_e_per_adu=gain_e_per_adu, deblended=deblended,
    )

    return PhotometryResult(
        table=table, metrics=metrics, fwhm_px=fwhm_px, fwhm_arcsec=fwhm_arcsec,
        sky_level=sky_level, sky_rms=sky_rms, n_sources=n, deblended=deblended,
    )
