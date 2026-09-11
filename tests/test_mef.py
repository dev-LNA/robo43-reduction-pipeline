"""Tests for robo43.products.mef.

Builds its own in-memory fixtures; does not use tests/conftest.py.
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.table import Table
from astropy.wcs import WCS
from numpy.testing import assert_array_equal

from robo43.config import load_settings
from robo43.instruments.profile import InstrumentLayout, InstrumentProfile
from robo43.io.meta import FieldOrigin, FrameMeta, Source
from robo43.products.dq import DQ_DTYPE
from robo43.products.mef import ReducedProduct, calibration_cards
from robo43.provenance import apply_cards, sha256_file


def _profile() -> InstrumentProfile:
    return InstrumentProfile(
        name="test-instrument",
        matches={},
        pixel_scale_arcsec=0.5,
        gain_e_per_adu=1.0,
        read_noise_e=5.0,
        saturation_adu=60000.0,
        nonlinearity_adu=None,
        binning_keyword="BINNING",
        layout=InstrumentLayout(),
        valid_filters=(),
        filter_map={},
        keywords={},
        imagetyp_map={},
        site_lat_deg=None,
        site_lon_deg=None,
        site_elevation_m=None,
        index_scales=None,
        search_radius_deg=1.0,
    )


def _frame_meta() -> FrameMeta:
    return FrameMeta(
        path=Path(__file__),
        naxis1=8,
        naxis2=8,
        frame_type="object",
        exptime=30.0,
        date_obs=None,
        filter="r",
        ra_deg=10.0,
        dec_deg=-20.0,
        pixel_scale=0.5,
        ccd_temp=-10.0,
        gain=1.2,
        read_noise=5.0,
        saturation=60000.0,
        airmass=1.1,
        object="M1",
        telescope="robo43",
        instrument="cam1",
        observer=None,
        binning=1,
        focal_length_mm=None,
        site_lat_deg=None,
        site_lon_deg=None,
        site_elevation_m=None,
        mjd=None,
        night=None,
        is_valid=True,
        profile_name="test-instrument",
        origins={
            "gain": FieldOrigin(source=Source.OVERRIDE, detail="overrides.toml"),
            "airmass": FieldOrigin(source=Source.DERIVED, detail="computed from date_obs"),
            "filter": FieldOrigin(source=Source.HEADER, detail=""),
        },
    )


def test_calibration_cards_skipped_masters():
    cards = calibration_cards()
    by_key = {}
    for key, value, _comment in cards:
        by_key.setdefault(key, []).append(value)
    assert by_key["BIASFIL"] == ["NONE"]
    assert by_key["DARKFIL"] == ["NONE"]
    assert by_key["FLATFIL"] == ["NONE"]
    assert by_key["MASKFIL"] == ["NONE"]
    assert "DARKSCL" not in by_key


def test_calibration_cards_real_master(tmp_path):
    bias_path = tmp_path / "master_bias_20240101.fits"
    bias_path.write_bytes(b"not really a fits file, just bytes for hashing")

    cards = calibration_cards(bias=bias_path)
    values = {key: value for key, value, _comment in cards if key == "BIASFIL"}
    assert values["BIASFIL"] == bias_path.name

    expected_sha = sha256_file(bias_path)
    history = [value for key, value, _comment in cards if key == "HISTORY"]
    assert any(
        v.startswith("BIASFIL sha256=") and v == f"BIASFIL sha256={expected_sha}" for v in history
    )


def test_calibration_cards_flags_always_present():
    cards = calibration_cards()
    by_key = {key: value for key, value, _comment in cards}
    assert by_key["CRREJ"] is False
    assert by_key["OSCANCOR"] is False


def test_calibration_cards_kwsrc_filtering():
    meta = _frame_meta()
    cards = calibration_cards(meta=meta)
    kwsrc = {key: value for key, value, _comment in cards if key.startswith("KWSRC")}
    assert len(kwsrc) == 2
    values = list(kwsrc.values())
    assert any(v.startswith("gain=override:overrides.toml") for v in values)
    assert any(v.startswith("airmass=derived:") for v in values)
    assert not any(v.startswith("filter=") for v in values)


def test_calibration_cards_no_verify_warning(tmp_path):
    bias_path = tmp_path / "master_bias.fits"
    bias_path.write_bytes(b"bytes")

    settings = load_settings(use_user_config=False)
    meta = _frame_meta()
    profile = _profile()

    cards = calibration_cards(bias=bias_path, profile=profile, settings=settings, meta=meta)
    header = fits.Header()
    apply_cards(header, cards)

    hdu = fits.PrimaryHDU(header=header)
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("error", VerifyWarning)
        hdu.writeto(buf)


def _wcs_header() -> fits.Header:
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [4.0, 4.0]
    wcs.wcs.crval = [10.0, -20.0]
    wcs.wcs.cdelt = [-0.0001, 0.0001]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = fits.Header()
    header.update(wcs.to_header())
    return header


def _small_product(*, with_catalog: bool) -> ReducedProduct:
    rng = np.random.default_rng(0)
    sci = rng.normal(size=(8, 8)).astype(np.float32)
    err = np.abs(rng.normal(size=(8, 8))).astype(np.float32)
    dq = np.zeros((8, 8), dtype=DQ_DTYPE)
    dq[0, 0] = 1
    dq[1, 1] = 4

    header = _wcs_header()

    catalog = None
    catalog_header = None
    if with_catalog:
        catalog = Table({"NUMBER": [1, 2, 3], "MAG_AUTO": [18.1, 19.2, 20.3]})
        catalog_header = fits.table_to_hdu(catalog).header.copy()
        catalog_header["EXTNAME"] = "CATALOG"
        catalog_header["ZP"] = 25.0

    return ReducedProduct(
        sci=sci, err=err, dq=dq, header=header, catalog=catalog, catalog_header=catalog_header
    )


def test_reduced_product_round_trip(tmp_path):
    product = _small_product(with_catalog=True)
    out_path = product.write(tmp_path / "out.fits")

    result = ReducedProduct.read(out_path)

    assert_array_equal(result.sci, product.sci)
    assert_array_equal(result.err, product.err)
    assert_array_equal(result.dq, product.dq)

    assert result.dq.dtype == np.int32
    assert result.err.dtype == np.float32
    assert result.sci.dtype == np.float32

    original_wcs = WCS(product.header)
    recovered_wcs = WCS(result.header)
    assert_array_equal(recovered_wcs.wcs.crval, original_wcs.wcs.crval)
    assert_array_equal(
        original_wcs.pixel_to_world_values([0, 4], [0, 4]),
        recovered_wcs.pixel_to_world_values([0, 4], [0, 4]),
    )

    assert result.catalog is not None
    assert_array_equal(result.catalog["NUMBER"], product.catalog["NUMBER"])
    assert_array_equal(result.catalog["MAG_AUTO"], product.catalog["MAG_AUTO"])
    assert result.catalog_header["ZP"] == 25.0


def test_hdu_order_with_catalog():
    product = _small_product(with_catalog=True)
    hdul = product.to_hdulist(include_catflags=True)
    assert [h.name for h in hdul] == ["SCI", "ERR", "DQ", "CATALOG", "CATFLAGS"]


def test_hdu_order_without_catalog():
    product = _small_product(with_catalog=False)
    hdul = product.to_hdulist(include_catflags=True)
    assert [h.name for h in hdul] == ["SCI", "ERR", "DQ"]


def test_missing_catalog_is_graceful(tmp_path):
    product = _small_product(with_catalog=False)
    out_path = product.write(tmp_path / "no_cat.fits")

    result = ReducedProduct.read(out_path)

    assert result.catalog is None
    assert result.catalog_header is None
