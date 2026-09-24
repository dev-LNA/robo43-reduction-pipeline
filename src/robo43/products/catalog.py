"""The photometric catalogue: the ``CATALOG`` HDU of a reduced product.

Every reduced frame is written as one multi-extension FITS file whose
``CATALOG`` extension holds the source table plus a header full of the
numbers that describe how it was produced (zero point, seeing, sky level,
and so on).  That header is consumed directly by the database ingester
instead of being recomputed, so :class:`CatalogMetrics` must round-trip
through a FITS header exactly: ``CatalogMetrics.from_header(m.to_header())
== m`` for any instance, including one where most fields are ``None``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astropy.io import fits
from astropy.table import Table

from robo43.exceptions import ProductError
from robo43.io.fits_io import write_fits

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "CATALOG_COLUMNS",
    "CatalogMetrics",
    "build_catalog_hdu",
    "extract_catalog",
    "read_catalog",
    "write_ldac",
]

#: column name -> (fits element format code, TUNIT). The four *_APER columns
#: are vectors; the actual TFORM/TDIM is derived by ``table_to_hdu`` from the
#: array's real width, this entry only documents the element type/unit.
CATALOG_COLUMNS: dict[str, tuple[str, str]] = {
    "NUMBER": ("J", ""),
    "X_IMAGE": ("D", "pix"),
    "Y_IMAGE": ("D", "pix"),
    "ERRX2_IMAGE": ("E", "pix**2"),
    "ERRY2_IMAGE": ("E", "pix**2"),
    "ALPHA_J2000": ("D", "deg"),
    "DELTA_J2000": ("D", "deg"),
    "ERRA_WORLD": ("E", "deg"),
    "ERRB_WORLD": ("E", "deg"),
    "FLUX_AUTO": ("E", "count"),
    "FLUXERR_AUTO": ("E", "count"),
    "MAG_AUTO": ("E", "mag"),
    "MAGERR_AUTO": ("E", "mag"),
    "FLUX_APER": ("E", "count"),
    "FLUXERR_APER": ("E", "count"),
    "MAG_APER": ("E", "mag"),
    "MAGERR_APER": ("E", "mag"),
    "MAG_CALIB": ("E", "mag"),
    "MAGERR_CALIB": ("E", "mag"),
    # Gaia-referenced absolute calibration (robo43.photometry.gaia_zp,
    # optional, off by default): MAG_CALIB/MAGERR_CALIB above are the
    # AUTO-flavoured calibrated magnitude; these two are their _APER
    # (vector, one value per aperture) counterparts. REFMAG/REFMAGERR are
    # the matched Gaia reference magnitude and its uncertainty, in the
    # frame's own filter -- populated for any cross-matched source, whether
    # or not that source also fed the zero-point fit itself.
    "MAG_APER_CAL": ("E", "mag"),
    "MAGERR_APER_CAL": ("E", "mag"),
    "REFMAG": ("E", "mag"),
    "REFMAGERR": ("E", "mag"),
    "BACKGROUND": ("E", "count"),
    "FLUX_MAX": ("E", "count"),
    "FLUX_RADIUS": ("E", "pix"),
    "FWHM_IMAGE": ("E", "pix"),
    "FWHM_WORLD": ("E", "deg"),
    "A_IMAGE": ("E", "pix"),
    "B_IMAGE": ("E", "pix"),
    "THETA_IMAGE": ("E", "deg"),
    "ELONGATION": ("E", ""),
    "ELLIPTICITY": ("E", ""),
    "FLAGS": ("J", ""),
    "IMAFLAGS_ISO": ("J", ""),
    "NIMAFLAGS_ISO": ("J", ""),
    "CLASS_STAR": ("E", ""),
    "SPREAD_MODEL": ("E", ""),
    "SNR": ("E", ""),
    "QUALITY_MASK": ("J", ""),
    "GAIA_SOURCE_ID": ("K", ""),
    "GAIA_SEP_ARCSEC": ("E", "arcsec"),
    # Convenience aliases of the canonical SExtractor-style columns above,
    # requested by name; each is an exact copy of its canonical column (or,
    # for FWHM, that column converted to arcsec), not an independent
    # measurement.
    "RA": ("D", "deg"),  # alias of ALPHA_J2000
    "DEC": ("D", "deg"),  # alias of DELTA_J2000
    "FWHM": ("E", "arcsec"),  # FWHM_IMAGE converted to arcsec
    "FLUX": ("E", "count"),  # alias of FLUX_AUTO
    "FLUX_ERR": ("E", "count"),  # alias of FLUXERR_AUTO
    "MAG": ("E", "mag"),  # alias of MAG_AUTO
    "MAG_ERR": ("E", "mag"),  # alias of MAGERR_AUTO
}


@dataclass(slots=True)
class CatalogMetrics:
    """Everything that goes into the ``CATALOG`` HDU header.

    Every field defaults to ``None``; ``None`` means "omit this card", it is
    never written as a sentinel value.
    """

    cat_soft: str | None = None
    cat_softv: str | None = None
    cat_cfg: str | None = None
    cat_date: str | None = None
    pipe_ver: str | None = None
    pipe_git: str | None = None
    det_thrsh: float | None = None
    det_minar: int | None = None
    det_filt: str | None = None
    debl_nth: int | None = None
    debl_minc: float | None = None
    back_size: int | None = None
    back_filt: int | None = None
    weight_tp: str | None = None
    n_sources: int | None = None
    n_stars: int | None = None
    n_clean: int | None = None
    n_reject: int | None = None
    n_satur: int | None = None
    n_badpix: int | None = None
    zp: float | None = None
    zp_err: float | None = None
    zp_nstar: int | None = None
    zp_rms: float | None = None
    zp_cat: str | None = None
    zp_band: str | None = None
    magsys: str | None = None
    extinct: float | None = None
    airmass: float | None = None
    photomet: bool | None = None
    aper_rad: tuple[float, ...] | None = None
    aper_unit: str | None = None
    fwhm_pix: float | None = None
    fwhm_arc: float | None = None
    ellip: float | None = None
    theta: float | None = None
    sky_lev: float | None = None
    sky_rms: float | None = None
    sky_unit: str | None = None
    lim_mag3: float | None = None
    lim_mag5: float | None = None
    snr_lim: float | None = None
    gain: float | None = None
    rdnoise: float | None = None
    sat_level: float | None = None
    pix_scale: float | None = None
    exptime: float | None = None
    filter: str | None = None
    ast_solvr: str | None = None
    ast_rms1: float | None = None
    ast_rms2: float | None = None
    ast_nref: int | None = None
    ast_epoch: float | None = None
    radesys: str | None = None
    wcssolvd: bool | None = None
    texptime: float | None = None
    ncombine: int | None = None

    def to_header(self) -> fits.Header:
        header = fits.Header()
        for field, (keyword, comment) in _FIELD_KEYWORDS.items():
            value = getattr(self, field)
            if value is None:
                continue
            if field == "aper_rad":
                header[keyword] = (_format_aper_rad(value), comment)
            else:
                header[keyword] = (value, comment)
        return header

    @classmethod
    def from_header(cls, header: fits.Header) -> CatalogMetrics:
        kwargs: dict[str, Any] = {}
        for field, (keyword, _comment) in _FIELD_KEYWORDS.items():
            if keyword not in header:
                continue
            value = header[keyword]
            if field == "aper_rad":
                kwargs[field] = tuple(float(x) for x in str(value).split(","))
            elif field in ("photomet", "wcssolvd"):
                kwargs[field] = bool(value)
            else:
                kwargs[field] = value
        return cls(**kwargs)


#: field name -> (FITS keyword, card comment). Single source of truth for
#: both ``CatalogMetrics.to_header`` and ``CatalogMetrics.from_header``.
_FIELD_KEYWORDS: dict[str, tuple[str, str]] = {
    "cat_soft": ("CATSOFT", "source-detection software"),
    "cat_softv": ("CATSOFTV", "software version"),
    "cat_cfg": ("CATCFG", "detection config identifier"),
    "cat_date": ("CATDATE", "UTC of cataloguing"),
    "pipe_ver": ("PIPEVER", "pipeline version"),
    "pipe_git": ("PIPEGIT", "pipeline git SHA"),
    "det_thrsh": ("DETTHRSH", "detection threshold, sigma"),
    "det_minar": ("DETMINAR", "detection min area, px"),
    "det_filt": ("DETFILT", "detection convolution filter"),
    "debl_nth": ("DEBLNTH", "deblend n thresholds"),
    "debl_minc": ("DEBLMINC", "deblend min contrast"),
    "back_size": ("BACKSIZE", "background mesh size, px"),
    "back_filt": ("BACKFILT", "background filter size"),
    "weight_tp": ("WEIGHTTP", "weight map type"),
    "n_sources": ("NSOURCES", "total sources detected"),
    "n_stars": ("NSTARS", "sources classed as stars"),
    "n_clean": ("NCLEAN", "sources passing all cuts"),
    "n_reject": ("NREJECT", "sources rejected by QC"),
    "n_satur": ("NSATUR", "saturated sources"),
    "n_badpix": ("NBADPIX", "sources touching bad pixels"),
    "zp": ("ZP", "photometric zero point, mag"),
    "zp_err": ("ZPERR", "zero point uncertainty, mag"),
    "zp_nstar": ("ZPNSTAR", "stars used for zero point"),
    "zp_rms": ("ZPRMS", "zero point fit RMS, mag"),
    "zp_cat": ("ZPCAT", "reference catalogue"),
    "zp_band": ("ZPBAND", "reference photometric band"),
    "magsys": ("MAGSYS", "magnitude system"),
    "extinct": ("EXTINCT", "atmospheric extinction, mag"),
    "airmass": ("AIRMASS", "airmass"),
    "photomet": ("PHOTOMET", "photometric conditions"),
    "aper_rad": ("APERRAD", "aperture radii used"),
    "aper_unit": ("APERUNIT", "aperture radius unit"),
    "fwhm_pix": ("FWHMPIX", "median FWHM, px"),
    "fwhm_arc": ("FWHMARC", "median FWHM, arcsec"),
    "ellip": ("ELLIP", "median ellipticity"),
    "theta": ("THETA", "median position angle, deg"),
    "sky_lev": ("SKYLEV", "sky background level"),
    "sky_rms": ("SKYRMS", "sky background RMS"),
    "sky_unit": ("SKYUNIT", "sky level/RMS unit"),
    "lim_mag3": ("LIMMAG3", "3-sigma limiting mag"),
    "lim_mag5": ("LIMMAG5", "5-sigma limiting mag"),
    "snr_lim": ("SNRLIM", "SNR threshold used for limits"),
    "gain": ("GAIN", "detector gain, e-/ADU"),
    "rdnoise": ("RDNOISE", "read noise, e-"),
    "sat_level": ("SATLEVEL", "saturation level, data units"),
    "pix_scale": ("PIXSCALE", "pixel scale, arcsec/px"),
    "exptime": ("EXPTIME", "exposure time, s"),
    "filter": ("FILTER", "filter name"),
    "ast_solvr": ("ASTSOLVR", "astrometric solver used"),
    "ast_rms1": ("ASTRMS1", "astrometric RMS axis 1, arcsec"),
    "ast_rms2": ("ASTRMS2", "astrometric RMS axis 2, arcsec"),
    "ast_nref": ("ASTNREF", "astrometric reference stars"),
    "ast_epoch": ("ASTEPOCH", "astrometric solution epoch"),
    "radesys": ("RADESYS", "WCS reference frame"),
    "wcssolvd": ("WCSSOLVD", "astrometric solution present"),
    "texptime": ("TEXPTIME", "total exposure time, s"),
    "ncombine": ("NCOMBINE", "frames combined"),
}


def _format_aper_rad(radii: Sequence[float]) -> str:
    return ",".join(str(float(r)) for r in radii)


def build_catalog_hdu(
    table: Table,
    metrics: CatalogMetrics,
    *,
    aperture_radii: Sequence[float] | None = None,
) -> fits.BinTableHDU:
    """Assemble the ``CATALOG`` HDU from a source table and its metrics."""
    for name in table.colnames:
        if name not in CATALOG_COLUMNS:
            msg = f"unknown catalog column: {name!r}"
            raise ProductError(msg)

    if metrics.zp is None and (
        "MAG_CALIB" in table.colnames or "MAGERR_CALIB" in table.colnames
    ):
        msg = "MAG_CALIB/MAGERR_CALIB present without a zero point (metrics.zp is None)"
        raise ProductError(msg)

    hdu = fits.table_to_hdu(table)
    hdu.name = "CATALOG"

    for index, name in enumerate(table.colnames, start=1):
        hdu.header[f"TUNIT{index}"] = CATALOG_COLUMNS[name][1]

    for card in metrics.to_header().cards:
        hdu.header[card.keyword] = (card.value, card.comment)

    if metrics.zp is None:
        hdu.header["MAGSYS"] = ("INSTRUMENTAL", "magnitude system")

    if aperture_radii is not None:
        hdu.header["APERRAD"] = (_format_aper_rad(aperture_radii), "aperture radii used")
        n_aper = len(aperture_radii)
        for name in ("FLUX_APER", "FLUXERR_APER", "MAG_APER", "MAGERR_APER"):
            if name in table.colnames:
                width = table[name].shape[-1] if table[name].ndim > 1 else 1
                if width != n_aper:
                    msg = (
                        f"{name} has {width} apertures, "
                        f"but aperture_radii has {n_aper}"
                    )
                    raise ProductError(msg)

    return hdu


def read_catalog(
    path_or_hdul: str | os.PathLike[str] | fits.HDUList,
) -> tuple[Table, CatalogMetrics]:
    """Read the ``CATALOG`` HDU of a reduced product."""
    if isinstance(path_or_hdul, fits.HDUList):
        return _read_catalog_hdul(path_or_hdul)
    with fits.open(path_or_hdul) as hdul:
        return _read_catalog_hdul(hdul)


def _read_catalog_hdul(hdul: fits.HDUList) -> tuple[Table, CatalogMetrics]:
    if "CATALOG" not in hdul:
        msg = "no CATALOG extension present"
        raise ProductError(msg)
    hdu = hdul["CATALOG"]
    return Table(hdu.data), CatalogMetrics.from_header(hdu.header)


def write_ldac(
    table: Table,
    path: str | os.PathLike[str],
    *,
    image_header: fits.Header,
) -> Path:
    """Write ``table`` as a FITS-LDAC file for SCAMP/PSFEx."""
    header_text = image_header.tostring(sep="", endcard=True, padding=True)
    imhead = fits.BinTableHDU.from_columns(
        [fits.Column(name="Field Header Card", format=f"{len(header_text)}A", array=[header_text])]
    )
    imhead.name = "LDAC_IMHEAD"

    objects = fits.table_to_hdu(table)
    objects.name = "LDAC_OBJECTS"

    hdul = fits.HDUList([fits.PrimaryHDU(), imhead, objects])
    return write_fits(path, hdul, overwrite=True)


def extract_catalog(
    path: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    fmt: str = "fits",
) -> Path:
    """Pull the catalogue out of a MEF product and write it standalone."""
    table, metrics = read_catalog(path)

    if fmt == "fits":
        hdu = build_catalog_hdu(table, metrics)
        hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
        return write_fits(out, hdul, overwrite=True)
    if fmt == "parquet":
        table.write(out, format="parquet", overwrite=True)
        return Path(out)
    if fmt == "csv":
        table.write(out, format="csv", overwrite=True)
        return Path(out)

    msg = f"unknown catalog format: {fmt!r}"
    raise ProductError(msg)
