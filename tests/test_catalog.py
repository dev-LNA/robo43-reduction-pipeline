"""Tests for robo43.products.catalog.

Builds its own in-memory fixtures; does not use tests/conftest.py.
"""

from __future__ import annotations

import io
import warnings

import numpy as np
import pytest
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.table import Table

from robo43.exceptions import ProductError
from robo43.io.fits_io import write_fits
from robo43.products.catalog import (
    CatalogMetrics,
    build_catalog_hdu,
    extract_catalog,
    read_catalog,
    write_ldac,
)


def _dense_metrics() -> CatalogMetrics:
    return CatalogMetrics(
        cat_soft="sep",
        cat_softv="1.2.1",
        cat_cfg="default.cfg",
        cat_date="2026-09-08T03:00:00",
        pipe_ver="0.4.0",
        pipe_git="abc1234",
        det_thrsh=1.5,
        det_minar=5,
        det_filt="gauss_3.0_5x5",
        debl_nth=32,
        debl_minc=0.005,
        back_size=64,
        back_filt=3,
        weight_tp="MAP_WEIGHT",
        n_sources=100,
        n_stars=42,
        n_clean=90,
        n_reject=10,
        n_satur=2,
        n_badpix=1,
        zp=25.4321,
        zp_err=0.015,
        zp_nstar=30,
        zp_rms=0.021,
        zp_cat="GAIA-DR3",
        zp_band="G",
        magsys="AB",
        extinct=0.12,
        airmass=1.234,
        photomet=True,
        aper_rad=(2.0, 3.0, 4.0, 6.0, 8.0),
        aper_unit="pix",
        fwhm_pix=3.5,
        fwhm_arc=1.1,
        ellip=0.05,
        theta=12.3,
        sky_lev=100.5,
        sky_rms=5.2,
        sky_unit="count",
        lim_mag3=21.1,
        lim_mag5=20.5,
        snr_lim=5.0,
        gain=1.6,
        rdnoise=4.2,
        sat_level=60000.0,
        pix_scale=0.33,
        exptime=120.0,
        filter="r",
        ast_solvr="scamp",
        ast_rms1=0.05,
        ast_rms2=0.06,
        ast_nref=200,
        ast_epoch=2026.68,
        radesys="ICRS",
        wcssolvd=True,
        texptime=None,
        ncombine=None,
    )


def _small_table(n: int = 5) -> Table:
    rng = np.random.default_rng(0)
    return Table(
        {
            "NUMBER": np.arange(1, n + 1, dtype=np.int32),
            "X_IMAGE": rng.uniform(0, 1000, n).astype(np.float64),
            "Y_IMAGE": rng.uniform(0, 1000, n).astype(np.float64),
            "FLUX_AUTO": rng.uniform(100, 10000, n).astype(np.float32),
        }
    )


def test_round_trip_dense():
    m = _dense_metrics()
    assert CatalogMetrics.from_header(m.to_header()) == m


def test_round_trip_sparse():
    m = CatalogMetrics(zp_cat="GAIA-DR3", n_sources=10, gain=1.6, filter="g")
    assert CatalogMetrics.from_header(m.to_header()) == m
    assert "ZP" not in m.to_header()


def test_no_verify_warning_on_write():
    m = _dense_metrics()

    with warnings.catch_warnings():
        warnings.simplefilter("error", VerifyWarning)
        hdu = fits.PrimaryHDU()
        for card in m.to_header().cards:
            hdu.header[card.keyword] = (card.value, card.comment)
        hdu.writeto(io.BytesIO())

    with warnings.catch_warnings():
        warnings.simplefilter("error", VerifyWarning)
        table = _small_table()
        cat_hdu = build_catalog_hdu(table, m)
        fits.HDUList([fits.PrimaryHDU(), cat_hdu]).writeto(io.BytesIO())


def test_vector_columns_round_trip(tmp_path):
    n = 5
    n_aper = 5
    rng = np.random.default_rng(1)
    flux_aper = rng.uniform(1, 1000, (n, n_aper)).astype(np.float32)
    table = Table(
        {
            "NUMBER": np.arange(1, n + 1, dtype=np.int32),
            "FLUX_APER": flux_aper,
            "FLUXERR_APER": rng.uniform(0, 10, (n, n_aper)).astype(np.float32),
            "MAG_APER": rng.uniform(15, 25, (n, n_aper)).astype(np.float32),
            "MAGERR_APER": rng.uniform(0, 0.1, (n, n_aper)).astype(np.float32),
        }
    )
    m = _dense_metrics()
    hdu = build_catalog_hdu(table, m, aperture_radii=(2.0, 3.0, 4.0, 6.0, 8.0))

    path = tmp_path / "cat.fits"
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)

    with fits.open(path) as hdul:
        read_table = Table(hdul["CATALOG"].data)
        read_metrics = CatalogMetrics.from_header(hdul["CATALOG"].header)

    assert read_table["FLUX_APER"].shape == (n, n_aper)
    np.testing.assert_array_equal(read_table["FLUX_APER"], flux_aper)
    assert read_metrics.aper_rad == (2.0, 3.0, 4.0, 6.0, 8.0)
    assert read_metrics.aper_unit == m.aper_unit


def test_mag_calib_absence_rule():
    m = CatalogMetrics(zp=None)
    table = _small_table()
    hdu = build_catalog_hdu(table, m)
    assert hdu.header["MAGSYS"] == "INSTRUMENTAL"

    table_with_calib = _small_table()
    table_with_calib["MAG_CALIB"] = np.zeros(len(table_with_calib), dtype=np.float32)
    with pytest.raises(ProductError):
        build_catalog_hdu(table_with_calib, m)


def test_coadd_style_metrics():
    m = CatalogMetrics(texptime=1234.5, ncombine=7, exptime=None)
    header = m.to_header()
    assert "TEXPTIME" in header
    assert "NCOMBINE" in header
    assert "EXPTIME" not in header
    assert CatalogMetrics.from_header(header) == m


def test_write_ldac(tmp_path):
    table = _small_table()
    header = fits.Header()
    header["CRVAL1"] = 123.456
    header["CTYPE1"] = "RA---TAN"

    out_path = write_ldac(table, tmp_path / "out.ldac", image_header=header)

    with fits.open(out_path) as hdul:
        names = [hdu.name for hdu in hdul]
        assert "LDAC_IMHEAD" in names
        assert "LDAC_OBJECTS" in names
        imhead_text = hdul["LDAC_IMHEAD"].data["Field Header Card"][0]
        assert "CRVAL1" in imhead_text


def test_extract_catalog(tmp_path):
    table = _small_table()
    m = _dense_metrics()
    hdu = build_catalog_hdu(table, m)
    src_path = write_fits(
        tmp_path / "src.fits", fits.HDUList([fits.PrimaryHDU(), hdu]), overwrite=True
    )

    out_fits = tmp_path / "out.fits"
    out_csv = tmp_path / "out.csv"

    result_fits = extract_catalog(src_path, out_fits, fmt="fits")
    result_csv = extract_catalog(src_path, out_csv, fmt="csv")

    assert result_fits.exists()
    assert result_csv.exists()

    read_table, _ = read_catalog(src_path)
    csv_table = Table.read(out_csv, format="csv")
    np.testing.assert_array_equal(csv_table["NUMBER"], read_table["NUMBER"])


def test_extract_catalog_parquet(tmp_path):
    pytest.importorskip("pyarrow")

    table = _small_table()
    m = _dense_metrics()
    hdu = build_catalog_hdu(table, m)
    src_path = write_fits(
        tmp_path / "src.fits", fits.HDUList([fits.PrimaryHDU(), hdu]), overwrite=True
    )

    out_parquet = tmp_path / "out.parquet"
    result = extract_catalog(src_path, out_parquet, fmt="parquet")
    assert result.exists()


def test_extract_catalog_unknown_format(tmp_path):
    table = _small_table()
    m = _dense_metrics()
    hdu = build_catalog_hdu(table, m)
    src_path = write_fits(
        tmp_path / "src.fits", fits.HDUList([fits.PrimaryHDU(), hdu]), overwrite=True
    )

    with pytest.raises(ProductError):
        extract_catalog(src_path, tmp_path / "out.xyz", fmt="xyz")


def test_read_catalog_missing_extension(tmp_path):
    path = write_fits(tmp_path / "nocat.fits", fits.HDUList([fits.PrimaryHDU()]), overwrite=True)
    with pytest.raises(ProductError):
        read_catalog(path)


def test_unknown_column_raises():
    table = _small_table()
    table["BOGUS_COLUMN"] = np.zeros(len(table))
    m = CatalogMetrics()
    with pytest.raises(ProductError):
        build_catalog_hdu(table, m)
