"""Tests for robo43.reduce.changelog.

Builds its own FrameMeta/ReductionStatus/fits.Header objects; does not use
tests/conftest.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from astropy.io import fits

from robo43.io.meta import FieldOrigin, FrameMeta, Source
from robo43.reduce.changelog import DEFAULT_CHANGE_LOG_NAME, append_change_log
from robo43.reduce.frame import ReductionStatus


def _meta(path: Path) -> FrameMeta:
    return FrameMeta(
        path=path,
        naxis1=64,
        naxis2=64,
        frame_type="science",
        exptime=10.0,
        date_obs=None,
        filter="R",
        ra_deg=None,
        dec_deg=None,
        pixel_scale=None,
        ccd_temp=None,
        gain_setting=2.0,
        read_noise=None,
        saturation=None,
        airmass=None,
        object=None,
        telescope=None,
        instrument=None,
        observer=None,
        binning=None,
        readout_mode=None,
        focal_length_mm=None,
        site_lat_deg=None,
        site_lon_deg=None,
        site_elevation_m=None,
        mjd=None,
        night=None,
        is_valid=True,
        profile_name="test-instrument",
        origins={"gain_setting": FieldOrigin(Source.PROFILE, "test-instrument")},
    )


def _status(path: Path, *, product_path: Path | None) -> ReductionStatus:
    return ReductionStatus(
        path=path,
        raw_name=path.name,
        proc_code=0,
        proc_status="reduced",
        steps_applied=("bias", "gain"),
        steps_skipped=("dark", "flat"),
        product_path=product_path,
    )


def test_append_change_log_writes_expected_record(tmp_path):
    raw_path = tmp_path / "raw.fits"
    product_path = tmp_path / "raw_proc.fits"

    raw_header = fits.Header()
    raw_header["NAXIS1"] = 64
    raw_header["NAXIS2"] = 64
    raw_header["GAIN"] = 2.0

    product_header = raw_header.copy()
    product_header["BIASFIL"] = ("master_bias.fits", "master bias used")
    product_header["GAINVAL"] = (0.706, "measured gain (e-/ADU)")
    product_header["GAINSRC"] = ("measured", "origin of GAINVAL: measured/none")
    product_header["GAINSET"] = (10.0, "camera GAIN header setting index")
    product_header["MEDOSC"] = 900.0
    product_header["MEDBIAS"] = 800.0
    product_header.add_history("BIASFIL sha256=deadbeef")

    meta = _meta(raw_path)
    status = _status(raw_path, product_path=product_path)

    append_change_log(tmp_path, meta, raw_header, product_header, status)

    log_path = tmp_path / DEFAULT_CHANGE_LOG_NAME
    assert log_path.exists()

    with log_path.open() as handle:
        lines = [line for line in handle if line.strip()]
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["raw_path"] == str(raw_path)
    assert record["product_path"] == str(product_path)
    assert record["proc_status"] == "reduced"
    assert record["history"] == ["BIASFIL sha256=deadbeef"]
    assert record["field_provenance"]["gain_setting"]["source"] == "profile"

    step_names = {s["step"] for s in record["steps"]}
    assert step_names == {"bias", "gain", "dark", "flat"}

    bias_step = next(s for s in record["steps"] if s["step"] == "bias")
    assert bias_step["applied"] is True
    assert bias_step["params"]["BIASFIL"] == "master_bias.fits"

    dark_step = next(s for s in record["steps"] if s["step"] == "dark")
    assert dark_step["applied"] is False

    keywords = {c["keyword"] for c in record["header_changes"]}
    assert "BIASFIL" in keywords
    assert "GAINVAL" in keywords
    assert "GAIN" not in keywords  # unchanged between raw and product headers


def test_append_change_log_appends_one_line_per_call(tmp_path):
    raw_path = tmp_path / "raw.fits"
    raw_header = fits.Header()
    product_header = raw_header.copy()
    meta = _meta(raw_path)
    status = _status(raw_path, product_path=None)

    append_change_log(tmp_path, meta, raw_header, product_header, status)
    append_change_log(tmp_path, meta, raw_header, product_header, status)

    log_path = tmp_path / DEFAULT_CHANGE_LOG_NAME
    with log_path.open() as handle:
        lines = [line for line in handle if line.strip()]
    assert len(lines) == 2


def test_append_change_log_never_raises_when_outdir_is_unusable(tmp_path, caplog):
    """A path that cannot be created as a directory must not raise."""
    blocked = tmp_path / "not_a_dir"
    blocked.write_text("i am a file, not a directory")

    raw_path = tmp_path / "raw.fits"
    raw_header = fits.Header()
    product_header = raw_header.copy()
    meta = _meta(raw_path)
    status = _status(raw_path, product_path=None)

    with caplog.at_level(logging.WARNING):
        append_change_log(blocked, meta, raw_header, product_header, status)

    assert not (blocked / DEFAULT_CHANGE_LOG_NAME).exists()
    assert any("change-log" in rec.message for rec in caplog.records)


def test_append_change_log_orders_steps_by_execution_order(tmp_path):
    """``steps`` reads in the pipeline's execution order, not applied-then-skipped."""
    raw_path = tmp_path / "raw.fits"
    raw_header = fits.Header()
    product_header = raw_header.copy()

    meta = _meta(raw_path)
    status = ReductionStatus(
        path=raw_path,
        raw_name=raw_path.name,
        proc_code=3,
        proc_status="reduced_degraded",
        steps_applied=("bias", "flat", "gain", "weight"),
        steps_skipped=("overscan", "trim", "dark", "mask", "cosmics"),
        product_path=None,
    )

    append_change_log(tmp_path, meta, raw_header, product_header, status)

    log_path = tmp_path / DEFAULT_CHANGE_LOG_NAME
    record = json.loads(log_path.read_text().splitlines()[0])

    assert [s["step"] for s in record["steps"]] == [
        "overscan",
        "trim",
        "bias",
        "dark",
        "flat",
        "mask",
        "cosmics",
        "gain",
        "weight",
    ]

    applied_by_step = {s["step"]: s["applied"] for s in record["steps"]}
    assert applied_by_step["overscan"] is False
    assert applied_by_step["trim"] is False
    assert applied_by_step["bias"] is True
    assert applied_by_step["dark"] is False
    assert applied_by_step["flat"] is True
    assert applied_by_step["mask"] is False
    assert applied_by_step["cosmics"] is False
    assert applied_by_step["gain"] is True
    assert applied_by_step["weight"] is True


def test_append_change_log_rejoins_wrapped_history_sha256(tmp_path):
    """A HISTORY value astropy wraps across cards rejoins into one 64-char sha256."""
    raw_path = tmp_path / "raw.fits"
    raw_header = fits.Header()
    product_header = raw_header.copy()

    sha256 = "4b0098d96334d9cc30c9e97c5f5b2046a892ed61dbef0a92fff4d09d95454941"
    assert len(sha256) == 64
    product_header.add_history(f"BIASFIL sha256={sha256}")

    meta = _meta(raw_path)
    status = _status(raw_path, product_path=None)

    append_change_log(tmp_path, meta, raw_header, product_header, status)

    log_path = tmp_path / DEFAULT_CHANGE_LOG_NAME
    record = json.loads(log_path.read_text().splitlines()[0])

    assert record["history"] == [f"BIASFIL sha256={sha256}"]
    rejoined_sha256 = record["history"][0].split("sha256=", 1)[1]
    assert len(rejoined_sha256) == 64
