"""Tests for robo43.io.headers.derive_pixel_scale's ``trust_header_wcs`` gate.

Builds its own synthetic FITS headers and InstrumentProfile objects; does not
use fixtures or conftest.py.
"""

from __future__ import annotations

import math

from astropy.io import fits

from robo43.instruments.profile import InstrumentLayout, InstrumentProfile
from robo43.io.headers import derive_pixel_scale
from robo43.io.meta import Source

#: A nominal T80S-like CD matrix: 0.500"/pix, no rotation, no XPIXSZ/FOCALLEN.
_NOMINAL_CD = -0.00013891756886253


def _profile(*, trust_header_wcs: bool, pixel_scale_arcsec: float = 0.55) -> InstrumentProfile:
    return InstrumentProfile(
        name="test-instrument",
        matches={},
        pixel_scale_arcsec=pixel_scale_arcsec,
        gain_e_per_adu=None,
        read_noise_e=None,
        saturation_adu=None,
        nonlinearity_adu=None,
        binning_keyword="XBINNING",
        layout=InstrumentLayout(),
        valid_filters=(),
        filter_map={},
        keywords={},
        imagetyp_map={},
        site_lat_deg=None,
        site_lon_deg=None,
        site_elevation_m=None,
        index_scales=None,
        search_radius_deg=0.5,
        trust_header_wcs=trust_header_wcs,
    )


def _header_with_nominal_wcs() -> fits.Header:
    header = fits.Header()
    header["NAXIS1"] = 4800
    header["NAXIS2"] = 3211
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 150.0
    header["CRVAL2"] = -30.0
    header["CRPIX1"] = 4607.0
    header["CRPIX2"] = 4615.0
    header["CD1_1"] = _NOMINAL_CD
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = _NOMINAL_CD
    header["RADESYS"] = "ICRS"
    return header


def test_derive_pixel_scale_uses_wcs_when_trusted() -> None:
    profile = _profile(trust_header_wcs=True)
    header = _header_with_nominal_wcs()
    scale, origin = derive_pixel_scale(header, profile)
    assert scale is not None
    assert math.isclose(scale, 0.5, rel_tol=1e-3)
    assert origin.source is Source.DERIVED
    assert origin.detail == "wcs"


def test_derive_pixel_scale_skips_wcs_when_not_trusted() -> None:
    profile = _profile(trust_header_wcs=False, pixel_scale_arcsec=0.55)
    header = _header_with_nominal_wcs()
    scale, origin = derive_pixel_scale(header, profile)
    assert scale == 0.55
    assert origin.source is Source.PROFILE
    assert origin.detail == "test-instrument"


def test_derive_pixel_scale_still_uses_xpixsz_focallen_when_not_trusted() -> None:
    # XPIXSZ+FOCALLEN precedes the WCS step and is unrelated to
    # trust_header_wcs; it must still fire even when the header WCS is
    # distrusted.
    profile = _profile(trust_header_wcs=False, pixel_scale_arcsec=0.55)
    header = _header_with_nominal_wcs()
    header["XPIXSZ"] = 9.0
    header["FOCALLEN"] = 1000.0
    scale, origin = derive_pixel_scale(header, profile)
    expected = 206.265 * 9.0 * 1.0 / 1000.0
    assert scale is not None
    assert math.isclose(scale, expected, rel_tol=1e-9)
    assert origin.source is Source.DERIVED
    assert origin.detail == "xpixsz+focallen+binning"
