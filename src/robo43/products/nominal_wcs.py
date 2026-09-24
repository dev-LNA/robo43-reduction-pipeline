"""Rescale a nominal, uncalibrated header WCS to an instrument profile's plate scale.

Some instrument profiles set ``trust_header_wcs = false`` because the raw
header's celestial WCS (CD matrix, or CDELT/PC pair) is a known-nominal
design value, not a real plate solve -- see
``src/robo43/data/instruments/t80s.toml``. For those profiles a reduced
product must not carry that nominal WCS forward unchanged: this module
rescales it to the profile's own ``pixel_scale_arcsec`` instead, preserving
the raw CD matrix's direction, sign, and rotation, and re-centring CRPIX on
the reduced (trimmed) output array. The astrometry stage
(:mod:`robo43.astrometry.solve`, applied via
:func:`robo43.astrometry.wcs_header.apply_solution`) overwrites this
placeholder WCS entirely on a successful solve; this module only keeps an
unsolved product's WCS from claiming a plate scale it does not have.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

if TYPE_CHECKING:
    from astropy.io import fits

    from robo43.instruments.profile import InstrumentProfile

logger = logging.getLogger(__name__)

__all__ = ["override_nominal_wcs"]

#: Keys dropped from a rescaled nominal WCS: the old CDELT/PC linear-transform
#: pair, SIP distortion polynomials, PV projection parameters, and CROTA.
#: CD1_1/CD1_2/CD2_1/CD2_2 are always (re)written immediately afterward
#: regardless of whether they were present before, so they are not listed
#: here.
_DROP_KEY_RE = re.compile(
    r"^(CDELT[12]|PC[0-9]+_[0-9]+|PV[0-9]+_[0-9]+|CROTA[12]|[AB]P?_ORDER|[AB]P?_[0-9]+_[0-9]+)$"
)


def override_nominal_wcs(
    header: fits.Header, profile: InstrumentProfile, *, ny: int, nx: int
) -> None:
    """Rescale ``header``'s celestial WCS in place to ``profile.pixel_scale_arcsec``.

    No-op when ``profile.trust_header_wcs`` is ``True``, when the profile has
    no static ``pixel_scale_arcsec``, when ``header`` has no celestial WCS, or
    when that WCS's own pixel scale cannot be measured. Otherwise rescales
    the raw CD matrix by a single scalar factor (preserving its direction,
    sign, and rotation), recentres CRPIX1/CRPIX2 on the ``(nx, ny)`` output
    array, drops CDELT/PC/SIP/PV/CROTA keys, and adds
    ``WCSSRC = 'nominal'``. ``CTYPE1``/``CTYPE2``/``CRVAL1``/``CRVAL2`` and
    any ``RADESYS``/``EQUINOX`` cards are left untouched.
    """
    if profile.trust_header_wcs or profile.pixel_scale_arcsec is None:
        return

    try:
        wcs = WCS(header)
    except Exception as exc:
        logger.debug("could not parse header WCS to override: %s", exc)
        return
    if not wcs.has_celestial:
        return

    if wcs.wcs.has_cd():
        cd = np.array(wcs.wcs.cd, dtype=float)
    else:
        pc = np.array(wcs.wcs.get_pc(), dtype=float)
        cdelt = np.array(wcs.wcs.cdelt, dtype=float)
        cd = pc * cdelt[:, np.newaxis]

    try:
        current_scale_deg = float(np.mean(proj_plane_pixel_scales(wcs)))
    except Exception as exc:
        logger.debug("could not measure current WCS pixel scale to rescale: %s", exc)
        return
    if not np.isfinite(current_scale_deg) or current_scale_deg <= 0:
        return

    target_scale_deg = profile.pixel_scale_arcsec / 3600.0
    factor = target_scale_deg / current_scale_deg
    new_cd = cd * factor

    ctype = tuple(wcs.wcs.ctype)
    crval = tuple(float(v) for v in wcs.wcs.crval)

    for key in [k for k in header if _DROP_KEY_RE.match(k)]:
        del header[key]

    header["CTYPE1"] = ctype[0]
    header["CTYPE2"] = ctype[1]
    header["CRVAL1"] = crval[0]
    header["CRVAL2"] = crval[1]
    header["CRPIX1"] = (nx + 1) / 2.0
    header["CRPIX2"] = (ny + 1) / 2.0
    header["CD1_1"] = float(new_cd[0, 0])
    header["CD1_2"] = float(new_cd[0, 1])
    header["CD2_1"] = float(new_cd[1, 0])
    header["CD2_2"] = float(new_cd[1, 1])
    header["WCSSRC"] = ("nominal", "Header WCS rescaled to profile pixel scale")
