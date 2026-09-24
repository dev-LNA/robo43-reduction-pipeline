"""Tests for robo43.products.nominal_wcs.override_nominal_wcs.

Builds its own synthetic FITS headers and InstrumentProfile objects; does not
use fixtures or conftest.py.
"""

from __future__ import annotations

import math

from astropy.io import fits

from robo43.instruments.profile import InstrumentLayout, InstrumentProfile
from robo43.products.nominal_wcs import override_nominal_wcs

_NOMINAL_CD = -0.00013891756886253


def _profile(*, trust_header_wcs: bool, pixel_scale_arcsec: float | None = 0.55) -> InstrumentProfile:
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
    header["EQUINOX"] = 2000.0
    return header


def test_noop_when_trust_header_wcs_true() -> None:
    profile = _profile(trust_header_wcs=True)
    header = _header_with_nominal_wcs()
    original = header.copy()
    override_nominal_wcs(header, profile, ny=2048, nx=2048)
    assert header == original


def test_noop_when_profile_has_no_pixel_scale() -> None:
    profile = _profile(trust_header_wcs=False, pixel_scale_arcsec=None)
    header = _header_with_nominal_wcs()
    original = header.copy()
    override_nominal_wcs(header, profile, ny=2048, nx=2048)
    assert header == original


def test_noop_when_no_celestial_wcs() -> None:
    profile = _profile(trust_header_wcs=False)
    header = fits.Header()
    header["NAXIS1"] = 2048
    header["NAXIS2"] = 2048
    original = header.copy()
    override_nominal_wcs(header, profile, ny=2048, nx=2048)
    assert header == original
    assert "WCSSRC" not in header


def test_rescales_cd_matrix_preserving_sign_and_recentres_crpix() -> None:
    profile = _profile(trust_header_wcs=False, pixel_scale_arcsec=0.55)
    header = _header_with_nominal_wcs()
    ny, nx = 3200, 4790  # a trimmed output shape, distinct from the raw one

    override_nominal_wcs(header, profile, ny=ny, nx=nx)

    target_scale_deg = 0.55 / 3600.0
    assert math.isclose(header["CD1_1"], -target_scale_deg, rel_tol=1e-9)
    assert math.isclose(header["CD2_2"], -target_scale_deg, rel_tol=1e-9)
    assert header["CD1_1"] < 0  # sign preserved
    assert header["CD2_2"] < 0  # sign preserved
    assert header["CD1_2"] == 0.0
    assert header["CD2_1"] == 0.0

    assert header["CRPIX1"] == (nx + 1) / 2.0
    assert header["CRPIX2"] == (ny + 1) / 2.0

    # CTYPE/CRVAL/RADESYS/EQUINOX are kept.
    assert header["CTYPE1"] == "RA---TAN"
    assert header["CTYPE2"] == "DEC--TAN"
    assert header["CRVAL1"] == 150.0
    assert header["CRVAL2"] == -30.0
    assert header["RADESYS"] == "ICRS"
    assert header["EQUINOX"] == 2000.0

    assert header["WCSSRC"] == "nominal"


def test_drops_cdelt_pc_sip_pv_crota_keys() -> None:
    profile = _profile(trust_header_wcs=False, pixel_scale_arcsec=0.55)
    header = fits.Header()
    header["CTYPE1"] = "RA---TAN-SIP"
    header["CTYPE2"] = "DEC--TAN-SIP"
    header["CRVAL1"] = 150.0
    header["CRVAL2"] = -30.0
    header["CRPIX1"] = 1024.0
    header["CRPIX2"] = 1024.0
    header["CDELT1"] = -_NOMINAL_CD * -1
    header["CDELT2"] = _NOMINAL_CD
    header["PC1_1"] = 1.0
    header["PC1_2"] = 0.0
    header["PC2_1"] = 0.0
    header["PC2_2"] = 1.0
    header["CROTA2"] = 0.0
    header["A_ORDER"] = 2
    header["A_0_2"] = 1e-6
    header["B_ORDER"] = 2
    header["B_2_0"] = 1e-6
    header["PV1_1"] = 0.0

    override_nominal_wcs(header, profile, ny=2048, nx=2048)

    for dropped in (
        "CDELT1", "CDELT2", "PC1_1", "PC1_2", "PC2_1", "PC2_2", "CROTA2",
        "A_ORDER", "A_0_2", "B_ORDER", "B_2_0", "PV1_1",
    ):
        assert dropped not in header
    assert "CD1_1" in header
    assert "CD2_2" in header
