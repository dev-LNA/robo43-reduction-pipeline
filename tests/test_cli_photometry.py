"""Tests for robo43.cli.photometry.

Builds a small synthetic MEF ``_proc.fits`` fixture directly (SCI/ERR/DQ, a
plain TAN WCS, and a few injected Gaussian stars); does not use
tests/conftest.py. The subcommand depends on
:func:`robo43.photometry.catalog.measure_sources`, a sibling module owned by
a parallel change; every test below that actually runs the subcommand will
only pass once that module exists and exports ``measure_sources``.
"""

from __future__ import annotations

import argparse
import csv
import logging

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from robo43.cli.main import build_parser, main
from robo43.logging_setup import ROOT_LOGGER_NAME
from robo43.products.dq import DQ_DTYPE


@pytest.fixture(autouse=True)
def _reset_logging():
    """Clean up logging state after each test to avoid pollution (mirrors
    tests/test_cli_reduce.py's fixture of the same name)."""
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    handlers = list(logger.handlers)
    level = logger.level
    propagate = logger.propagate
    yield
    for handler in logger.handlers:
        if handler not in handlers:
            handler.close()
    logger.handlers = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def _write_synthetic_proc(
    path,
    *,
    shape: tuple[int, int] = (100, 100),
    gainval: float | str = 0.706,
) -> None:
    rng = np.random.default_rng(42)
    sci = rng.normal(100.0, 5.0, size=shape).astype(np.float32)

    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    for xc, yc, amplitude in ((30, 40, 3000.0), (70, 60, 2500.0), (50, 20, 4000.0)):
        sigma = 2.0
        sci += amplitude * np.exp(-((xx - xc) ** 2 + (yy - yc) ** 2) / (2.0 * sigma**2))
    sci = sci.astype(np.float32)

    err = np.full(shape, 5.0, dtype=np.float32)
    dq = np.zeros(shape, dtype=DQ_DTYPE)

    pixel_scale_deg = 0.5 / 3600.0
    header = fits.Header()
    header["EXTNAME"] = "SCI"
    header["BUNIT"] = "electron"
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRVAL1"] = 150.0
    header["CRVAL2"] = -10.0
    header["CRPIX1"] = shape[1] / 2.0
    header["CRPIX2"] = shape[0] / 2.0
    header["CD1_1"] = -pixel_scale_deg
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = pixel_scale_deg
    header["RADESYS"] = "ICRS"
    header["EQUINOX"] = 2000.0
    header["GAINVAL"] = (gainval, "measured gain (e-/ADU)")

    primary = fits.PrimaryHDU(data=sci, header=header)
    err_hdu = fits.ImageHDU(data=err, name="ERR")
    dq_hdu = fits.ImageHDU(data=dq, name="DQ")
    fits.HDUList([primary, err_hdu, dq_hdu]).writeto(path, overwrite=True)


def _extnames(path) -> list[str]:
    with fits.open(path) as hdul:
        return [hdu.name for hdu in hdul]


def test_build_parser_exposes_photometry_subcommand():
    parser = build_parser()
    subparsers_action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    assert "photometry" in subparsers_action.choices


def test_photometry_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["photometry", "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "photometry" in captured.out


def test_photometry_writes_catalog_and_catflags_hdus_in_order(tmp_path):
    frame_dir = tmp_path / "proc"
    frame_dir.mkdir()
    proc_path = frame_dir / "frame_0001_proc.fits"
    _write_synthetic_proc(proc_path)

    outdir = tmp_path / "out"
    result = main(["photometry", str(proc_path), "--outdir", str(outdir)])
    assert result == 0

    out_path = outdir / proc_path.name
    assert out_path.exists()

    names = _extnames(out_path)
    assert names[:3] == ["SCI", "ERR", "DQ"]
    assert "CATALOG" in names
    assert "CATFLAGS" in names
    assert names.index("CATFLAGS") == names.index("CATALOG") + 1


def test_photometry_catalog_has_alias_columns(tmp_path):
    frame_dir = tmp_path / "proc"
    frame_dir.mkdir()
    proc_path = frame_dir / "frame_0001_proc.fits"
    _write_synthetic_proc(proc_path)

    outdir = tmp_path / "out"
    result = main(["photometry", str(proc_path), "--outdir", str(outdir)])
    assert result == 0

    out_path = outdir / proc_path.name
    with fits.open(out_path) as hdul:
        colnames = set(hdul["CATALOG"].columns.names)

    for name in ("RA", "DEC", "FWHM", "FLUX", "FLUX_ERR", "MAG", "MAG_ERR"):
        assert name in colnames


def test_photometry_rerun_on_output_is_idempotent_in_hdu_count(tmp_path):
    frame_dir = tmp_path / "proc"
    frame_dir.mkdir()
    proc_path = frame_dir / "frame_0001_proc.fits"
    _write_synthetic_proc(proc_path)

    result = main(["photometry", str(proc_path), "--in-place"])
    assert result == 0
    first_count = len(_extnames(proc_path))

    result = main(["photometry", str(proc_path), "--in-place"])
    assert result == 0
    second_count = len(_extnames(proc_path))

    assert second_count == first_count


def test_photometry_handles_literal_none_gainval(tmp_path):
    frame_dir = tmp_path / "proc"
    frame_dir.mkdir()
    proc_path = frame_dir / "frame_0001_proc.fits"
    _write_synthetic_proc(proc_path, gainval="NONE")

    outdir = tmp_path / "out"
    result = main(["photometry", str(proc_path), "--outdir", str(outdir)])
    assert result == 0

    csv_path = outdir / "photometry_status.csv"
    assert csv_path.exists()
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert not rows[0]["error"]


def test_photometry_dry_run_writes_no_output_file(tmp_path):
    frame_dir = tmp_path / "proc"
    frame_dir.mkdir()
    proc_path = frame_dir / "frame_0001_proc.fits"
    _write_synthetic_proc(proc_path)

    outdir = tmp_path / "out"
    result = main(["photometry", str(proc_path), "--outdir", str(outdir), "--dry-run"])
    assert result == 0
    assert not (outdir / proc_path.name).exists()


def test_photometry_catalog_csv_flattens_aperture_columns(tmp_path):
    frame_dir = tmp_path / "proc"
    frame_dir.mkdir()
    proc_path = frame_dir / "frame_0001_proc.fits"
    _write_synthetic_proc(proc_path)

    outdir = tmp_path / "out"
    result = main(["photometry", str(proc_path), "--outdir", str(outdir), "--catalog-csv"])
    assert result == 0

    out_path = outdir / proc_path.name
    with fits.open(out_path) as hdul:
        flux_aper = hdul["CATALOG"].data["FLUX_APER"]
    assert flux_aper.ndim == 2
    n_aper = flux_aper.shape[1]
    assert n_aper >= 2

    csv_path = outdir / f"{out_path.stem}_catalog.csv"
    assert csv_path.exists()
    csv_table = Table.read(csv_path, format="csv")
    assert "FLUX_APER" not in csv_table.colnames
    for i in range(1, n_aper + 1):
        assert f"FLUX_APER_{i}" in csv_table.colnames
        assert f"FLUXERR_APER_{i}" in csv_table.colnames
        assert f"MAG_APER_{i}" in csv_table.colnames
        assert f"MAGERR_APER_{i}" in csv_table.colnames


def test_photometry_catalog_csv_failure_does_not_fail_the_frame(tmp_path, monkeypatch):
    frame_dir = tmp_path / "proc"
    frame_dir.mkdir()
    proc_path = frame_dir / "frame_0001_proc.fits"
    _write_synthetic_proc(proc_path)

    def _boom(self, *args, **kwargs):
        raise OSError("disk full (synthetic failure injected by the test)")

    monkeypatch.setattr(Table, "write", _boom)

    outdir = tmp_path / "out"
    result = main(["photometry", str(proc_path), "--outdir", str(outdir), "--catalog-csv"])
    assert result == 0

    out_path = outdir / proc_path.name
    assert out_path.exists()
    with fits.open(out_path) as hdul:
        assert "CATALOG" in [hdu.name for hdu in hdul]
        n_sources_in_hdu = len(hdul["CATALOG"].data)

    status_csv = outdir / "photometry_status.csv"
    with status_csv.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert not rows[0]["error"]
    assert int(rows[0]["n_sources"]) == n_sources_in_hdu
    assert rows[0]["catalog_csv"] == ""
