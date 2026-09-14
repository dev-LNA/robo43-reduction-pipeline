"""Tests for robo43.cli.astrometry.

Builds small synthetic MEF ``_proc.fits`` fixtures directly; does not use
tests/conftest.py. The real solver needs local astrometry.net index files, so
the one test that expects an actual solved WCS is gated behind a skip when
the index directory is absent.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from robo43.cli.main import main
from robo43.logging_setup import ROOT_LOGGER_NAME
from robo43.products.dq import DQ_DTYPE

REAL_INDEX_DIR = Path("/home/herpich/Documents/.astrometry")
REAL_PROC_DIR = Path("/home/herpich/Documents/data/ROBO43/mini_out/proc")


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


def _write_synthetic_proc(path: Path, *, shape: tuple[int, int] = (100, 100)) -> None:
    rng = np.random.default_rng(0)
    sci = rng.normal(100.0, 5.0, size=shape).astype(np.float32)
    err = np.ones(shape, dtype=np.float32)
    dq = np.zeros(shape, dtype=DQ_DTYPE)
    weight = np.ones(shape, dtype=np.float32)

    header = fits.Header()
    header["EXTNAME"] = "SCI"
    header["BUNIT"] = "electron"
    header["RA"] = 150.0
    header["DEC"] = -10.0
    header["XPIXSZ"] = 7.52
    header["FOCALLEN"] = 2939.0
    header["XBINNING"] = 2

    primary = fits.PrimaryHDU(data=sci, header=header)
    err_hdu = fits.ImageHDU(data=err, name="ERR")
    dq_hdu = fits.ImageHDU(data=dq, name="DQ")
    weight_hdu = fits.ImageHDU(data=weight, name="WEIGHT")
    fits.HDUList([primary, err_hdu, dq_hdu, weight_hdu]).writeto(path, overwrite=True)


def test_astrometry_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["astrometry", "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "astrometry" in captured.out


def test_astrometry_writes_status_csv_and_does_not_modify_input_by_default(tmp_path):
    frame_dir = tmp_path / "proc"
    frame_dir.mkdir()
    proc_path = frame_dir / "frame_0001_proc.fits"
    _write_synthetic_proc(proc_path)
    original_bytes = proc_path.read_bytes()

    outdir = tmp_path / "out"
    result = main(
        [
            "astrometry",
            str(frame_dir),
            "--index-dir",
            str(tmp_path / "no_such_index_dir"),
            "--outdir",
            str(outdir),
        ]
    )
    assert result == 0

    # Nothing was found under the bogus index dir, so the solve fails, but
    # the run itself must not crash and must not touch the input file.
    assert proc_path.read_bytes() == original_bytes
    assert not (outdir / proc_path.name).exists()

    csv_path = outdir / "astrometry_status.csv"
    assert csv_path.exists()
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["solved"] == "False"
    assert rows[0]["raw_name"] == proc_path.name


def test_astrometry_dry_run_writes_no_output_file(tmp_path):
    frame_dir = tmp_path / "proc"
    frame_dir.mkdir()
    proc_path = frame_dir / "frame_0001_proc.fits"
    _write_synthetic_proc(proc_path)

    outdir = tmp_path / "out"
    result = main(
        [
            "astrometry",
            str(proc_path),
            "--index-dir",
            str(tmp_path / "no_such_index_dir"),
            "--outdir",
            str(outdir),
            "--dry-run",
        ]
    )
    assert result == 0
    assert not (outdir / proc_path.name).exists()


@pytest.mark.needs_astrometry_index
@pytest.mark.skipif(
    not REAL_INDEX_DIR.is_dir(), reason="local astrometry.net index cache not present"
)
def test_astrometry_solves_real_mini_sample_frames(tmp_path):
    paths = [str(p) for p in sorted(REAL_PROC_DIR.glob("*_proc.fits"))]
    if not paths:
        pytest.skip("ROBO43 mini-sample _proc.fits files not present on this machine")

    outdir = tmp_path / "out"
    result = main(
        [
            "astrometry",
            *paths,
            "--index-dir",
            str(REAL_INDEX_DIR),
            "--outdir",
            str(outdir),
        ]
    )
    assert result == 0

    csv_path = outdir / "astrometry_status.csv"
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(paths)

    for row in rows:
        assert row["solved"] == "True", row
        assert float(row["rms_arcsec"]) <= 0.3

        out_path = outdir / Path(row["raw_name"])
        assert out_path.exists()
        with fits.open(out_path) as hdul:
            for name in ("SCI", "ERR", "DQ", "WEIGHT"):
                assert WCS(hdul[name].header).has_celestial, name
