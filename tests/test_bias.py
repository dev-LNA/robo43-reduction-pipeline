"""Tests for robo43.calib.bias.

Builds its own synthetic FITS files and FrameMeta objects; does not use
tests/conftest.py.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
import pytest
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning

from robo43.calib.bias import (
    apply_overscan_trim,
    build_master_bias,
    resolve_common_gain,
    select_frames,
)
from robo43.calib.gain import GainMeasurement, GainMeasurementSet
from robo43.calib.master import MasterFrame
from robo43.calib.overscan import OverscanGeometry
from robo43.config import CalibSettings, RuntimeSettings, Settings
from robo43.exceptions import CalibrationError
from robo43.instruments.profile import InstrumentLayout, InstrumentProfile
from robo43.io.meta import FrameMeta
from robo43.products.dq import DQ_DTYPE


def _profile(**layout_overrides) -> InstrumentProfile:
    layout = InstrumentLayout(**layout_overrides)
    return InstrumentProfile(
        name="test-instrument",
        matches={},
        pixel_scale_arcsec=0.5,
        gain_e_per_adu=2.0,
        read_noise_e=5.0,
        saturation_adu=60000.0,
        nonlinearity_adu=None,
        binning_keyword="XBINNING",
        layout=layout,
        valid_filters=(),
        filter_map={},
        keywords={},
        imagetyp_map={},
        site_lat_deg=None,
        site_lon_deg=None,
        site_elevation_m=None,
        index_scales=None,
        search_radius_deg=0.5,
    )


def _settings(**calib_overrides) -> Settings:
    return Settings(calib=CalibSettings(**calib_overrides), runtime=RuntimeSettings())


def _write_fits(path, data) -> None:
    hdu = fits.PrimaryHDU(data=data.astype(np.float32))
    hdu.writeto(path, overwrite=True)


def _meta(path, shape) -> FrameMeta:
    return FrameMeta(
        path=path,
        naxis1=shape[1],
        naxis2=shape[0],
        frame_type="bias",
        exptime=0.0,
        date_obs=None,
        filter=None,
        ra_deg=None,
        dec_deg=None,
        pixel_scale=None,
        ccd_temp=None,
        gain_setting=10.0,
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
        origins={},
    )


def test_select_frames_preserves_order_and_filters():
    metas = [
        _meta("a.fits", (5, 5)),
        dataclasses.replace(_meta("b.fits", (5, 5)), frame_type="dark"),
        _meta("c.fits", (5, 5)),
    ]
    result = select_frames(metas, "bias")
    assert [m.path for m in result] == ["a.fits", "c.fits"]


def test_recovery_of_constant_level_and_hot_column(tmp_path):
    rng = np.random.default_rng(101)
    n = 18
    h, w = 30, 30
    level = 1000.0
    sigma = 6.0

    truth = np.full((h, w), level)
    truth[:, 5] += 500.0  # hot column

    metas = []
    for i in range(n):
        frame = truth + rng.normal(0.0, sigma, size=(h, w))
        path = tmp_path / f"bias_{i:02d}.fits"
        _write_fits(path, frame)
        metas.append(_meta(path, (h, w)))

    profile = _profile()
    settings = _settings()
    master = build_master_bias(metas, settings=settings, profile=profile, outdir=tmp_path)

    se = sigma / math.sqrt(n)
    center_level = float(np.median(master.data[:, :4]))
    assert abs(center_level - level) < 5 * se

    hot_col_level = float(np.median(master.data[:, 5]))
    assert abs(hot_col_level - (level + 500.0)) < 5 * se

    assert master.data.dtype == np.float32
    assert not isinstance(master.data, np.ma.MaskedArray)
    assert np.isfinite(master.data).all()


def test_write_produces_readable_fits_with_expected_extensions(tmp_path):
    rng = np.random.default_rng(202)
    n = 10
    h, w = 20, 20
    metas = []
    for i in range(n):
        frame = 500.0 + rng.normal(0.0, 5.0, size=(h, w))
        path = tmp_path / f"bias_{i:02d}.fits"
        _write_fits(path, frame)
        metas.append(_meta(path, (h, w)))

    profile = _profile()
    settings = _settings()
    outdir = tmp_path / "out"
    master = build_master_bias(metas, settings=settings, profile=profile, outdir=outdir)

    with fits.open(master.path) as hdul:
        assert hdul[0].header["BUNIT"] == "adu"
        assert "NCOMBINE" in hdul[0].header
        assert "VAR" in hdul


def test_write_raises_no_verify_warning(tmp_path):
    rng = np.random.default_rng(303)
    n = 10
    h, w = 20, 20
    metas = []
    for i in range(n):
        frame = 500.0 + rng.normal(0.0, 5.0, size=(h, w))
        path = tmp_path / f"bias_{i:02d}.fits"
        _write_fits(path, frame)
        metas.append(_meta(path, (h, w)))

    profile = _profile()
    settings = _settings()
    outdir = tmp_path / "out"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_master_bias(metas, settings=settings, profile=profile, outdir=outdir)

    verify_warnings = [w for w in caught if issubclass(w.category, VerifyWarning)]
    assert not verify_warnings


def test_apply_overscan_trim_removes_pedestal_and_trims():
    rng = np.random.default_rng(404)
    # 50 columns x 20 rows, deliberately non-square to catch axis-order bugs.
    h, w = 20, 50
    overscan_pedestal = 300.0
    science = 200.0 + rng.normal(0.0, 1.0, size=(h, w))
    # The overscan pedestal is a bias signal present across the whole readout,
    # not just the overscan columns -- that's why subtracting it corrects the
    # science region too.
    data = science + overscan_pedestal

    profile = _profile(overscan="[46:50,1:20]", trim="[1:45,1:20]")
    settings = _settings()

    trimmed, cards = apply_overscan_trim(data, profile=profile, settings=settings)

    assert trimmed.shape == (20, 45)
    # The pedestal (bias level) has been removed; the remaining signal is
    # just the zero-mean science noise.
    assert abs(float(np.median(trimmed))) < 1.0

    card_dict = {c[0]: c[1] for c in cards}
    assert card_dict["OSCANCOR"] is True
    assert card_dict["OSCANREG"] == "[46:50,1:20]"
    assert card_dict["TRIMMED"] is True
    assert card_dict["TRIMREG"] == "[1:45,1:20]"


def test_apply_overscan_trim_disabled_by_setting():
    rng = np.random.default_rng(405)
    h, w = 20, 50
    science = 200.0 + rng.normal(0.0, 1.0, size=(h, w))
    data = science.copy()
    data[:, 45:50] += 300.0

    profile = _profile(overscan="[46:50,1:20]", trim="[1:45,1:20]")
    settings = _settings(overscan_correct=False)

    trimmed, cards = apply_overscan_trim(data, profile=profile, settings=settings)

    card_dict = {c[0]: c[1] for c in cards}
    assert card_dict["OSCANCOR"] is False
    # overscan not subtracted -- the pedestal in the (now trimmed-out) region
    # is irrelevant, but the science region itself is untouched.
    assert abs(float(np.median(trimmed[:, :44])) - 200.0) < 1.0


def test_build_master_bias_raises_on_empty(tmp_path):
    profile = _profile()
    settings = _settings()
    with pytest.raises(CalibrationError):
        build_master_bias([], settings=settings, profile=profile, outdir=tmp_path)


def test_build_master_bias_with_overscan_geometry(tmp_path):
    """Test build_master_bias with OverscanGeometry parameter.

    Verifies that:
    - When overscan is passed, the master's shape is trimmed to science region
    - The master header contains TRIMMED=True and OSCANLVL
    - When overscan is not passed, the master keeps the untrimmed shape
    """
    rng = np.random.default_rng(501)
    # Raw detector: 30 rows x 34 columns
    # Overscan: 4 columns at indices 30-33 (raw 0-based)
    # Science region: columns 0-29 (trimmed to 30 columns)
    h, w = 30, 34
    bias_level = 1000.0
    sigma = 5.0

    # Create synthetic bias frames with 4 overscan columns at bias level
    science_data = np.full((h, 30), bias_level, dtype=np.float32)
    overscan_cols = np.full((h, 4), bias_level, dtype=np.float32)

    n_frames = 5
    metas = []
    for i in range(n_frames):
        # Add noise
        science_frame = science_data + rng.normal(0.0, sigma, size=(h, 30))
        overscan_frame = overscan_cols + rng.normal(0.0, sigma, size=(h, 4))
        full_frame = np.hstack([science_frame, overscan_frame])

        path = tmp_path / f"bias_{i:02d}.fits"
        _write_fits(path, full_frame)
        metas.append(_meta(path, (h, w)))

    # Build with OverscanGeometry specifying the 4 overscan columns
    geom = OverscanGeometry(
        shape=(h, w),
        columns=(30, 31, 32, 33),
        rows=(),
        bias_level=bias_level,
        flat_level=np.nan,
        detected=True,
        source="test",
    )

    profile = _profile()
    settings = _settings()
    outdir_with = tmp_path / "with_overscan"

    # Build with overscan parameter
    master_with = build_master_bias(
        metas, settings=settings, profile=profile, outdir=outdir_with, overscan=geom
    )

    # Check shape is trimmed
    assert master_with.data.shape == (h, 30), \
        f"Master shape {master_with.data.shape} != trimmed shape (h={h}, w=30)"

    # Check header cards
    header_cards = {c[0]: c[1] for c in master_with.header_cards}
    assert header_cards.get("TRIMMED") is True, "TRIMMED should be True in header"
    assert "OSCANLVL" in header_cards, "OSCANLVL should be in header"
    assert header_cards["OSCANCOR"] is True, "OSCANCOR should be True"

    # Build without overscan parameter to verify default behavior is unchanged
    outdir_without = tmp_path / "without_overscan"
    master_without = build_master_bias(
        metas, settings=settings, profile=profile, outdir=outdir_without
    )

    # Check that default behavior still returns untrimmed shape
    assert master_without.data.shape == (h, w), \
        f"Master without overscan shape {master_without.data.shape} should be untrimmed ({h}, {w})"


def test_masterframe_roundtrip_with_all_extensions(tmp_path):
    """Test MasterFrame round-trip with data, variance, weight, and mask."""
    rng = np.random.default_rng(601)
    h, w = 10, 10

    # Create test data
    data = rng.normal(100.0, 5.0, size=(h, w)).astype(np.float32)
    variance = rng.exponential(2.0, size=(h, w)).astype(np.float32)
    weight = rng.uniform(0.5, 1.0, size=(h, w)).astype(np.float32)
    mask = rng.integers(0, 10, size=(h, w), dtype=DQ_DTYPE)

    # Create and write MasterFrame
    master = MasterFrame(
        data=data,
        variance=variance,
        weight=weight,
        mask=mask,
        bunit="electron",
    )
    path = tmp_path / "test_master.fits"
    master.write(path)

    # Load and verify
    loaded = MasterFrame.load(path)

    assert np.allclose(loaded.data, data)
    assert np.allclose(loaded.variance, variance)
    assert np.allclose(loaded.weight, weight)
    assert np.array_equal(loaded.mask, mask)
    assert loaded.bunit == "electron"
    assert loaded.path == path


def test_masterframe_roundtrip_data_only(tmp_path):
    """Test MasterFrame round-trip with data only (no variance/weight/mask)."""
    rng = np.random.default_rng(602)
    h, w = 10, 10

    # Create test data
    data = rng.normal(100.0, 5.0, size=(h, w)).astype(np.float32)

    # Create and write MasterFrame
    master = MasterFrame(data=data)
    path = tmp_path / "test_master_minimal.fits"
    master.write(path)

    # Load and verify
    loaded = MasterFrame.load(path)

    assert np.allclose(loaded.data, data)
    assert loaded.variance is None
    assert loaded.weight is None
    assert loaded.mask is None
    assert loaded.bunit == "adu"  # default


def test_masterframe_load_sets_path(tmp_path):
    """Test that MasterFrame.load() sets path to the file it read."""
    rng = np.random.default_rng(603)
    data = rng.normal(100.0, 5.0, size=(10, 10)).astype(np.float32)

    master = MasterFrame(data=data)
    path = tmp_path / "test_path_setting.fits"
    master.write(path)

    loaded = MasterFrame.load(path)
    assert loaded.path == path


def test_masterframe_load_nonexistent_raises_calibrationerror(tmp_path):
    """Test that MasterFrame.load() raises CalibrationError for nonexistent file."""
    path = tmp_path / "nonexistent.fits"
    with pytest.raises(CalibrationError, match="Master frame not found"):
        MasterFrame.load(path)


def test_resolve_common_gain_uses_header_setting_when_present(tmp_path):
    """When every meta agrees on the GAIN setting, that value is returned."""
    meta_a = dataclasses.replace(_meta(tmp_path / "a.fits", (5, 5)), gain_setting=10.0)
    meta_b = dataclasses.replace(_meta(tmp_path / "b.fits", (5, 5)), gain_setting=10.0)
    profile = _profile()

    setting, origin = resolve_common_gain([meta_a, meta_b], profile=profile)

    assert setting == 10.0
    assert origin == "header"


def test_resolve_common_gain_raises_when_no_frame_has_a_setting(tmp_path):
    """There is no profile fallback for a per-frame GAIN setting."""
    meta = dataclasses.replace(_meta(tmp_path / "a.fits", (5, 5)), gain_setting=None)
    profile = _profile()

    with pytest.raises(CalibrationError, match="no gain-setting value"):
        resolve_common_gain([meta], profile=profile)


def test_resolve_common_gain_raises_on_inconsistent_setting(tmp_path):
    meta_a = dataclasses.replace(_meta(tmp_path / "a.fits", (5, 5)), gain_setting=10.0)
    meta_b = dataclasses.replace(_meta(tmp_path / "b.fits", (5, 5)), gain_setting=30.0)
    profile = _profile()

    with pytest.raises(CalibrationError, match="inconsistent"):
        resolve_common_gain([meta_a, meta_b], profile=profile)


def test_build_master_bias_records_measured_gain_cards(tmp_path):
    """GAINVAL/GAINSRC come from the runtime measurement for this group's
    (readout_mode, gain_setting, binning), never from the header setting or
    the profile."""
    rng = np.random.default_rng(707)
    n = 6
    h, w = 16, 16
    metas = []
    for i in range(n):
        frame = 500.0 + rng.normal(0.0, 5.0, size=(h, w))
        path = tmp_path / f"bias_{i:02d}.fits"
        _write_fits(path, frame)
        metas.append(dataclasses.replace(_meta(path, (h, w)), gain_setting=10.0, binning=1))

    profile = _profile()  # gain_e_per_adu=2.0 -- must be ignored entirely
    settings = _settings()
    measurement = GainMeasurement(
        readout_mode="UNKNOWN",
        gain_setting=10.0,
        binning=1,
        gain_e_per_adu=0.706,
        read_noise_e=7.01,
        n_pairs=6,
        scatter_frac=0.005,
    )
    gain_measurements = GainMeasurementSet({("UNKNOWN", 10.0, 1): measurement})
    master = build_master_bias(
        metas, settings=settings, profile=profile, outdir=tmp_path,
        gain_measurements=gain_measurements,
    )

    card_dict = {c[0]: c[1] for c in master.header_cards}
    assert card_dict["GAINVAL"] == pytest.approx(0.706)
    assert card_dict["GAINSRC"] == "measured"
    assert card_dict["GAINSET"] == 10.0


def test_build_master_bias_with_no_measurement_records_none_cards(tmp_path):
    """A group with no runtime measurement records GAINVAL=NONE/GAINSRC=none,
    never a guessed value."""
    rng = np.random.default_rng(709)
    n = 6
    h, w = 16, 16
    metas = []
    for i in range(n):
        frame = 500.0 + rng.normal(0.0, 5.0, size=(h, w))
        path = tmp_path / f"bias_{i:02d}.fits"
        _write_fits(path, frame)
        metas.append(dataclasses.replace(_meta(path, (h, w)), gain_setting=30.0, binning=1))

    profile = _profile()
    settings = _settings()
    master = build_master_bias(
        metas, settings=settings, profile=profile, outdir=tmp_path,
        gain_measurements=GainMeasurementSet({}),
    )

    card_dict = {c[0]: c[1] for c in master.header_cards}
    assert card_dict["GAINVAL"] == "NONE"
    assert card_dict["GAINSRC"] == "none"
    assert card_dict["GAINSET"] == 30.0


def test_build_master_bias_raises_on_inconsistent_gain_setting(tmp_path):
    rng = np.random.default_rng(708)
    h, w = 16, 16
    metas = []
    for i, setting in enumerate([10.0, 10.0, 30.0]):
        frame = 500.0 + rng.normal(0.0, 5.0, size=(h, w))
        path = tmp_path / f"bias_{i:02d}.fits"
        _write_fits(path, frame)
        metas.append(dataclasses.replace(_meta(path, (h, w)), gain_setting=setting))

    profile = _profile()
    settings = _settings()

    with pytest.raises(CalibrationError, match="inconsistent"):
        build_master_bias(metas, settings=settings, profile=profile, outdir=tmp_path)
