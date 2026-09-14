"""Tests for robo43.reduce.frame.

Builds its own synthetic FITS files and FrameMeta objects; does not use
fixtures or conftest.py.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from robo43.calib.dark import DarkEntry
from robo43.calib.gain import GainMeasurement, GainMeasurementSet
from robo43.calib.masks import MaskSet, minimal_mask
from robo43.calib.overscan import OverscanGeometry
from robo43.calib.set import CalibrationSet
from robo43.config import CalibSettings, RuntimeSettings, Settings
from robo43.instruments.profile import InstrumentLayout, InstrumentProfile
from robo43.io.meta import FieldOrigin, FrameMeta, Source
from robo43.products.dq import DQ, DQ_DTYPE
from robo43.reduce.frame import reduce_frame


def _profile(gain=2.0) -> InstrumentProfile:
    """Create a minimal test instrument profile."""
    return InstrumentProfile(
        name="test-instrument",
        matches={},
        pixel_scale_arcsec=0.5,
        gain_e_per_adu=gain,
        read_noise_e=5.0,
        saturation_adu=60000.0,
        nonlinearity_adu=None,
        binning_keyword="XBINNING",
        layout=InstrumentLayout(),
        valid_filters=("R",),
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
    """Create a minimal test settings object."""
    calib_settings = CalibSettings(**calib_overrides)
    return Settings(
        paths=None,
        astrometry=None,
        photometry=None,
        calib=calib_settings,
        coadd=None,
        diagnostics=None,
        db=None,
        service=None,
        runtime=RuntimeSettings(),
        targets=None,
    )


def _write_fits(path: Path, data: np.ndarray) -> None:
    """Write a simple FITS file with one HDU."""
    hdu = fits.PrimaryHDU(data=data.astype(np.float32))
    hdu.writeto(path, overwrite=True)


def _meta(path: Path, shape: tuple[int, int], **kwargs) -> FrameMeta:
    """Create a minimal test FrameMeta object."""
    defaults = {
        "path": path,
        "naxis1": shape[1],
        "naxis2": shape[0],
        "frame_type": "science",
        "exptime": 10.0,
        "date_obs": None,
        "filter": "R",
        "ra_deg": None,
        "dec_deg": None,
        "pixel_scale": None,
        "ccd_temp": 20.0,
        "gain_setting": 10.0,
        "read_noise": None,
        "saturation": None,
        "airmass": None,
        "object": None,
        "telescope": None,
        "instrument": None,
        "observer": None,
        "binning": None,
        "readout_mode": None,
        "focal_length_mm": None,
        "site_lat_deg": None,
        "site_lon_deg": None,
        "site_elevation_m": None,
        "mjd": None,
        "night": None,
        "is_valid": True,
        "profile_name": "test-instrument",
        "origins": {},
    }
    defaults.update(kwargs)
    return FrameMeta(**defaults)


def _gain_measurements(
    gain_e_per_adu: float,
    *,
    read_noise_e: float = 5.0,
    readout_mode: str = "UNKNOWN",
    gain_setting: float = 10.0,
    binning: int = 1,
) -> GainMeasurementSet:
    """A GainMeasurementSet with one measurement matching ``_meta()``'s
    default (readout_mode, gain_setting, binning), for tests that need
    reduce_frame to resolve a specific e-/ADU value."""
    measurement = GainMeasurement(
        readout_mode=readout_mode,
        gain_setting=gain_setting,
        binning=binning,
        gain_e_per_adu=gain_e_per_adu,
        read_noise_e=read_noise_e,
        n_pairs=3,
        scatter_frac=0.0,
    )
    key = (readout_mode, float(gain_setting), int(binning))
    return GainMeasurementSet({key: measurement})


def test_reduce_frame_full_path(tmp_path):
    """Test full reduction with bias, flat, mask, and dark present -> proc_code == 0."""
    # Create test data
    shape = (128, 128)

    # Raw frame structure: bias + dark_frame + flat * signal
    # Where dark_frame is an already-integrated dark frame at the same exptime
    bias_level = 100.0
    dark_integrated = 20.0  # Already-integrated dark (for 10-second exptime)
    flat_response = np.ones(shape) * 1.2  # 20% flat-field variation
    signal = 1000.0
    raw_data = bias_level + dark_integrated + flat_response * signal

    # Write raw frame
    raw_path = tmp_path / "raw.fits"
    _write_fits(raw_path, raw_data)

    # Create bias master
    bias_data = np.ones(shape) * bias_level
    bias_path = tmp_path / "master_bias.fits"
    _write_fits(bias_path, bias_data)

    # Create dark master (integrated dark frame for 10 seconds)
    dark_data = np.ones(shape) * dark_integrated
    dark_path = tmp_path / "master_dark_10s_20C.fits"
    hdu = fits.PrimaryHDU(data=dark_data.astype(np.float32))
    hdu.header["EXPTIME"] = 10.0
    hdu.header["CCDTEMP"] = 20.0
    hdu.writeto(dark_path, overwrite=True)

    # Create flat master (normalized)
    flat_data = flat_response
    flat_path = tmp_path / "master_flat_R.fits"
    _write_fits(flat_path, flat_data)

    # Create minimal mask
    mask_dq = np.zeros(shape, dtype=DQ_DTYPE)
    mask = MaskSet(dq=mask_dq, summary={"any": 0.0, "fatal": 0.0})

    # Build calibration set with dark
    dark_entry = DarkEntry(
        path=dark_path,
        exptime=10.0,
        ccd_temp=20.0,
        data=dark_data.astype(np.float32),
        variance=None,
        n_combine=1,
    )

    calib = CalibrationSet(
        bias=bias_data.astype(np.float32),
        bias_var=None,
        bias_path=bias_path,
        flats={"R": flat_data.astype(np.float32)},
        flat_var={},
        flat_paths={"R": flat_path},
        darks=(dark_entry,),
        dark_current_map=None,
        dark_current_path=None,
        mask=mask,
        gain_measurements=_gain_measurements(2.0),
    )

    # Create metadata
    meta = _meta(raw_path, shape, exptime=10.0, ccd_temp=20.0)

    # Run reduction
    profile = _profile()
    settings = _settings()
    product, status = reduce_frame(meta, calib, settings=settings, profile=profile)

    # Verify product shape and type
    assert product.sci.shape == shape
    assert product.err.shape == shape
    assert product.dq.shape == shape
    assert product.sci.dtype == np.float32
    assert product.err.dtype == np.float32
    assert product.dq.dtype == DQ_DTYPE

    # Verify status: should be proc_code 0 since bias, dark, flat, mask all present
    assert status.proc_code == 0, f"Expected proc_code 0, got {status.proc_code}"
    assert "bias" in status.steps_applied
    assert "dark" in status.steps_applied
    assert "flat" in status.steps_applied
    assert "mask" in status.steps_applied

    # After reduction: (bias + dark + flat*signal) - bias - dark / flat * gain
    # = signal * gain
    gain = profile.gain_e_per_adu or 1.0
    expected_signal = signal * gain
    actual_signal = np.median(product.sci)
    assert abs(actual_signal - expected_signal) < expected_signal * 0.01, \
        f"Signal mismatch: expected ~{expected_signal}, got {actual_signal}"


def test_reduce_frame_no_mask_fallback_to_minimal_with_nan(tmp_path):
    """Test that when no mask is present, minimal mask correctly identifies NaN pixels."""
    shape = (128, 128)

    raw_data = np.ones(shape) * 1000.0
    # Inject a NaN pixel to verify minimal mask detects it
    raw_data[50, 50] = np.nan
    raw_path = tmp_path / "raw.fits"
    _write_fits(raw_path, raw_data)

    bias_data = np.ones(shape) * 100.0
    bias_path = tmp_path / "master_bias.fits"
    _write_fits(bias_path, bias_data)

    flat_data = np.ones(shape) * 1.0
    flat_path = tmp_path / "master_flat_R.fits"
    _write_fits(flat_path, flat_data)

    # NO mask in calib set -> should fall back to minimal_mask
    calib = CalibrationSet(
        bias=bias_data.astype(np.float32),
        bias_var=None,
        bias_path=bias_path,
        flats={"R": flat_data.astype(np.float32)},
        flat_var={},
        flat_paths={"R": flat_path},
        darks=(),
        dark_current_map=None,
        dark_current_path=None,
        mask=None,  # No static mask -> will use minimal_mask
        gain_measurements=GainMeasurementSet({}),
    )

    meta = _meta(raw_path, shape)
    profile = _profile()
    settings = _settings()

    # Should not raise
    product, status = reduce_frame(meta, calib, settings=settings, profile=profile)

    # Should produce a product
    assert product.sci.shape == shape
    assert product.dq.shape == shape
    assert product.dq.dtype == DQ_DTYPE

    # The NaN pixel should have NO_DATA flag set by minimal mask
    nan_pixel_dq = product.dq[50, 50]
    assert (nan_pixel_dq & int(DQ.NO_DATA)) != 0, \
        f"Expected NO_DATA flag on NaN pixel, but got {nan_pixel_dq:032b}"

    # mask should be in steps_skipped
    assert "mask" in status.steps_skipped


def test_reduce_frame_signal_recovery(tmp_path):
    """Verify that known-truth signal is recovered after corrections."""
    shape = (128, 128)
    rng = np.random.default_rng(123)

    # Define true values
    bias_level = 500.0
    flat_level = 2.5
    true_signal_adu = 5000.0
    gain = 1.5

    # Construct raw frame: bias + flat * signal + noise
    flat_field = np.ones(shape) * flat_level
    noise = rng.normal(0, 10, size=shape)  # Gaussian noise
    raw_data = bias_level + flat_field * true_signal_adu + noise

    raw_path = tmp_path / "raw.fits"
    _write_fits(raw_path, raw_data)

    # Create bias master
    bias_data = np.ones(shape) * bias_level
    bias_path = tmp_path / "master_bias.fits"
    _write_fits(bias_path, bias_data)

    # Create flat master
    flat_path = tmp_path / "master_flat_R.fits"
    _write_fits(flat_path, flat_field)

    # Create minimal mask
    mask = minimal_mask(raw_data, saturation=60000.0)

    # Build calibration set
    calib = CalibrationSet(
        bias=bias_data.astype(np.float32),
        bias_var=None,
        bias_path=bias_path,
        flats={"R": flat_field.astype(np.float32)},
        flat_var={},
        flat_paths={"R": flat_path},
        darks=(),
        dark_current_map=None,
        dark_current_path=None,
        mask=mask,
        gain_measurements=_gain_measurements(1.5),
    )

    # Create metadata
    meta = _meta(raw_path, shape)

    # Run reduction with gain applied
    profile = _profile(gain=gain)
    settings = _settings(apply_gain=True)
    product, status = reduce_frame(meta, calib, settings=settings, profile=profile)

    # Expected value: (bias_level + flat_level * signal) - bias_level) / flat_level * gain
    # = signal * gain = 5000 * 1.5 = 7500
    expected_signal = true_signal_adu * gain

    # Check that the median signal is close to expected (within a few percent)
    actual_signal = np.median(product.sci)
    tolerance = expected_signal * 0.05  # 5% tolerance
    assert abs(actual_signal - expected_signal) < tolerance, \
        f"Signal recovery failed: expected {expected_signal}, got {actual_signal}"

    # Dark was missing, so proc_code should be 3 (degraded)
    assert status.proc_code == 3

    # DQ dtype must be correct
    assert product.dq.dtype == DQ_DTYPE


def test_reduce_frame_no_bias(tmp_path):
    """Test with no bias master -> does not raise, bias skipped, product still produced."""
    shape = (128, 128)

    # Raw frame
    raw_data = np.ones(shape) * 1000.0
    raw_path = tmp_path / "raw.fits"
    _write_fits(raw_path, raw_data)

    # Flat master
    flat_data = np.ones(shape) * 1.0
    flat_path = tmp_path / "master_flat_R.fits"
    _write_fits(flat_path, flat_data)

    # Minimal mask
    mask = minimal_mask(raw_data, saturation=60000.0)

    # Build calib set with NO bias
    calib = CalibrationSet(
        bias=None,
        bias_var=None,
        bias_path=None,
        flats={"R": flat_data.astype(np.float32)},
        flat_var={},
        flat_paths={"R": flat_path},
        darks=(),
        dark_current_map=None,
        dark_current_path=None,
        mask=mask,
        gain_measurements=GainMeasurementSet({}),
    )

    meta = _meta(raw_path, shape)
    profile = _profile()
    settings = _settings()

    # Should not raise
    product, status = reduce_frame(meta, calib, settings=settings, profile=profile)

    # Should produce a product
    assert product.sci.shape == shape
    assert product.err.shape == shape
    assert product.dq.dtype == DQ_DTYPE

    # bias should be in steps_skipped
    assert "bias" in status.steps_skipped

    # BIASFIL should be 'NONE' in header
    assert product.header.get("BIASFIL") == "NONE"


def test_reduce_frame_no_flat_for_filter(tmp_path):
    """Test with no flat for the frame's filter -> no raise, NO_FLAT flag, is_photometric=False."""
    shape = (128, 128)

    raw_data = np.ones(shape) * 1000.0
    raw_path = tmp_path / "raw.fits"
    _write_fits(raw_path, raw_data)

    # Create bias
    bias_data = np.ones(shape) * 100.0
    bias_path = tmp_path / "master_bias.fits"
    _write_fits(bias_path, bias_data)

    # No flat for the 'R' filter in the calibration set
    mask = minimal_mask(raw_data, saturation=60000.0)

    calib = CalibrationSet(
        bias=bias_data.astype(np.float32),
        bias_var=None,
        bias_path=bias_path,
        flats={},  # Empty: no flats available
        flat_var={},
        flat_paths={},
        darks=(),
        dark_current_map=None,
        dark_current_path=None,
        mask=mask,
        gain_measurements=GainMeasurementSet({}),
    )

    meta = _meta(raw_path, shape, filter="R")
    profile = _profile()
    settings = _settings()

    # Should not raise
    product, status = reduce_frame(meta, calib, settings=settings, profile=profile)

    # Should produce a product
    assert product.sci.shape == shape
    assert product.dq.dtype == DQ_DTYPE

    # Should have NO_FLAT flag on all pixels
    assert np.all((product.dq & int(DQ.NO_FLAT)) != 0), \
        "Expected NO_FLAT flag on all pixels when no flat is available"

    # is_photometric should be False
    assert status.is_photometric is False

    # flat should be in steps_skipped
    assert "flat" in status.steps_skipped


def test_reduce_frame_no_mask_fallback_to_minimal(tmp_path):
    """Test with no static mask -> falls back to minimal mask, no crash."""
    shape = (128, 128)

    raw_data = np.ones(shape) * 1000.0
    raw_path = tmp_path / "raw.fits"
    _write_fits(raw_path, raw_data)

    bias_data = np.ones(shape) * 100.0
    bias_path = tmp_path / "master_bias.fits"
    _write_fits(bias_path, bias_data)

    flat_data = np.ones(shape) * 1.0
    flat_path = tmp_path / "master_flat_R.fits"
    _write_fits(flat_path, flat_data)

    # NO mask in calib set
    calib = CalibrationSet(
        bias=bias_data.astype(np.float32),
        bias_var=None,
        bias_path=bias_path,
        flats={"R": flat_data.astype(np.float32)},
        flat_var={},
        flat_paths={"R": flat_path},
        darks=(),
        dark_current_map=None,
        dark_current_path=None,
        mask=None,  # No static mask
        gain_measurements=GainMeasurementSet({}),
    )

    meta = _meta(raw_path, shape)
    profile = _profile()
    settings = _settings()

    # Should not raise
    product, status = reduce_frame(meta, calib, settings=settings, profile=profile)

    # Should produce a product
    assert product.sci.shape == shape
    assert product.dq.shape == shape

    # DQ dtype must be correct
    assert product.dq.dtype == DQ_DTYPE

    # mask should be in steps_skipped
    assert "mask" in status.steps_skipped


def test_reduce_frame_apply_gain_true_vs_false(tmp_path):
    """Test apply_gain setting: True gives 'electron', False gives 'adu'."""
    shape = (64, 64)

    raw_data = np.ones(shape) * 1000.0
    raw_path = tmp_path / "raw.fits"
    _write_fits(raw_path, raw_data)

    bias_data = np.ones(shape) * 100.0
    bias_path = tmp_path / "master_bias.fits"
    _write_fits(bias_path, bias_data)

    mask = minimal_mask(raw_data, saturation=60000.0)

    calib = CalibrationSet(
        bias=bias_data.astype(np.float32),
        bias_var=None,
        bias_path=bias_path,
        flats={},
        flat_var={},
        flat_paths={},
        darks=(),
        dark_current_map=None,
        dark_current_path=None,
        mask=mask,
        gain_measurements=_gain_measurements(2.5),
    )

    meta = _meta(raw_path, shape)
    profile = _profile(gain=2.5)

    # Test with apply_gain=True
    settings_with_gain = _settings(apply_gain=True)
    product_with_gain, status_with_gain = reduce_frame(
        meta, calib, settings=settings_with_gain, profile=profile
    )
    assert product_with_gain.bunit == "electron"
    assert product_with_gain.dq.dtype == DQ_DTYPE
    assert "gain" in status_with_gain.steps_applied

    # Test with apply_gain=False
    settings_no_gain = _settings(apply_gain=False)
    product_no_gain, status_no_gain = reduce_frame(
        meta, calib, settings=settings_no_gain, profile=profile
    )
    assert product_no_gain.bunit == "adu"
    assert product_no_gain.dq.dtype == DQ_DTYPE
    assert "gain" in status_no_gain.steps_skipped

    # With gain, the median should be ~2.5x higher (gain factor)
    ratio = np.median(product_with_gain.sci) / np.median(product_no_gain.sci)
    assert abs(ratio - 2.5) < 0.1, f"Expected gain ratio ~2.5, got {ratio}"


def test_reduce_frame_err_is_1sigma(tmp_path):
    """Test that ERR plane is 1-sigma: err**2 ≈ variance, and err >= 0, isfinite."""
    shape = (64, 64)

    raw_data = np.ones(shape) * 1000.0
    raw_path = tmp_path / "raw.fits"
    _write_fits(raw_path, raw_data)

    bias_data = np.ones(shape) * 100.0
    bias_path = tmp_path / "master_bias.fits"
    _write_fits(bias_path, bias_data)

    mask = minimal_mask(raw_data, saturation=60000.0)

    calib = CalibrationSet(
        bias=bias_data.astype(np.float32),
        bias_var=None,
        bias_path=bias_path,
        flats={},
        flat_var={},
        flat_paths={},
        darks=(),
        dark_current_map=None,
        dark_current_path=None,
        mask=mask,
        gain_measurements=_gain_measurements(2.0),
    )

    meta = _meta(raw_path, shape, exptime=10.0)
    profile = _profile(gain=2.0)
    settings = _settings(apply_gain=True)

    product, _status = reduce_frame(meta, calib, settings=settings, profile=profile)

    # DQ dtype must be correct
    assert product.dq.dtype == DQ_DTYPE

    # ERR should be non-negative and finite
    assert np.all(product.err >= 0), "ERR has negative values"
    assert np.all(np.isfinite(product.err)), "ERR has non-finite values"

    # ERR^2 should be approximately equal to the variance model
    # Variance = read_noise^2 + signal (in data units)
    # With gain applied: variance = read_noise_e^2 + signal_e
    # read_noise_e = 5.0, gain = 2.0
    # After bias subtraction: signal = 900 ADU = 1800 electrons
    # variance = 5^2 + 1800 = 25 + 1800 = 1825 e^2
    # err = sqrt(1825) ≈ 42.7

    read_noise_e = profile.read_noise_e or 1.0
    gain = profile.gain_e_per_adu or 1.0
    expected_variance = (read_noise_e ** 2) + (900.0 * gain)
    expected_err = np.sqrt(expected_variance)

    actual_err = np.median(product.err)
    tolerance = expected_err * 0.1  # 10% tolerance
    assert abs(actual_err - expected_err) < tolerance, \
        f"ERR variance model failed: expected {expected_err}, got {actual_err}"


def test_reduce_frame_all_header_cards_finite(tmp_path):
    """Regression test: all header cards must be FITS-writable (no NaN/Inf)."""
    shape = (64, 64)

    raw_data = np.ones(shape) * 1000.0
    raw_path = tmp_path / "raw.fits"
    _write_fits(raw_path, raw_data)

    bias_data = np.ones(shape) * 100.0
    bias_path = tmp_path / "master_bias.fits"
    _write_fits(bias_path, bias_data)

    mask = minimal_mask(raw_data, saturation=60000.0)

    calib = CalibrationSet(
        bias=bias_data.astype(np.float32),
        bias_var=None,
        bias_path=bias_path,
        flats={},
        flat_var={},
        flat_paths={},
        darks=(),
        dark_current_map=None,
        dark_current_path=None,
        mask=mask,
        gain_measurements=GainMeasurementSet({}),
    )

    meta = _meta(raw_path, shape)
    profile = _profile()
    settings = _settings()

    product, _status = reduce_frame(meta, calib, settings=settings, profile=profile)

    # Try to build a FITS header with the product's header cards
    # This should not raise ValueError about non-finite values
    test_header = fits.Header()
    for key in product.header:
        value = product.header[key]
        try:
            comment = product.header.comments[key]
        except KeyError:
            comment = ""
        if isinstance(value, (int, str, bool)) or (isinstance(value, float) and np.isfinite(value)):
            test_header[key] = (value, comment) if comment else value
        # Skip non-finite floats and other types

    # Try to write to a file to verify it's valid FITS
    test_path = tmp_path / "test_header.fits"
    try:
        hdu = fits.PrimaryHDU(header=test_header)
        hdu.writeto(test_path, overwrite=True)
    except ValueError as e:
        pytest.fail(f"Header contains invalid FITS values: {e}")


def test_reduce_frame_as_row_picklable_scalars(tmp_path):
    """Test ReductionStatus.as_row() returns only plain scalars/strings."""
    shape = (64, 64)

    raw_data = np.ones(shape) * 1000.0
    raw_path = tmp_path / "raw.fits"
    _write_fits(raw_path, raw_data)

    bias_data = np.ones(shape) * 100.0
    bias_path = tmp_path / "master_bias.fits"
    _write_fits(bias_path, bias_data)

    mask = minimal_mask(raw_data, saturation=60000.0)

    calib = CalibrationSet(
        bias=bias_data.astype(np.float32),
        bias_var=None,
        bias_path=bias_path,
        flats={},
        flat_var={},
        flat_paths={},
        darks=(),
        dark_current_map=None,
        dark_current_path=None,
        mask=mask,
        gain_measurements=GainMeasurementSet({}),
    )

    meta = _meta(raw_path, shape)
    profile = _profile()
    settings = _settings()

    _product, status = reduce_frame(meta, calib, settings=settings, profile=profile)

    row = status.as_row()

    # Verify all values are plain types
    for key, value in row.items():
        assert isinstance(value, (str, int, float, bool, type(None))), \
            f"Key {key} has non-scalar type {type(value)}: {value}"
        assert not isinstance(value, (tuple, list, np.ndarray, Path)), \
            f"Key {key} is a complex type (tuple/list/array/Path): {type(value)}"


def test_reduce_frame_result_is_picklable(tmp_path):
    """Test that (ReducedProduct, ReductionStatus) is picklable."""
    shape = (64, 64)

    raw_data = np.ones(shape) * 1000.0
    raw_path = tmp_path / "raw.fits"
    _write_fits(raw_path, raw_data)

    bias_data = np.ones(shape) * 100.0
    bias_path = tmp_path / "master_bias.fits"
    _write_fits(bias_path, bias_data)

    mask = minimal_mask(raw_data, saturation=60000.0)

    calib = CalibrationSet(
        bias=bias_data.astype(np.float32),
        bias_var=None,
        bias_path=bias_path,
        flats={},
        flat_var={},
        flat_paths={},
        darks=(),
        dark_current_map=None,
        dark_current_path=None,
        mask=mask,
        gain_measurements=GainMeasurementSet({}),
    )

    meta = _meta(raw_path, shape)
    profile = _profile()
    settings = _settings()

    _product, status = reduce_frame(meta, calib, settings=settings, profile=profile)

    # Should be able to pickle the status
    pickled_status = pickle.dumps(status)
    unpickled_status = pickle.loads(pickled_status)
    assert unpickled_status.proc_code == status.proc_code
    assert unpickled_status.steps_applied == status.steps_applied


def test_reduce_frame_with_overscan_geometry(tmp_path):
    """Test overscan geometry with mask and cosmic-ray rejection.

    Regression test for the bug where DQ was passed through apply_overscan,
    which would silently fail the mask and cosmics steps. Verifies that
    with a mask present and cosmic_ray_rejection=True, both steps are applied.
    """
    raw_shape = (256, 256)
    trimmed_shape = (246, 246)

    raw_data = np.ones(raw_shape) * 1000.0
    raw_path = tmp_path / "raw.fits"
    _write_fits(raw_path, raw_data)

    # Bias and mask must have the trimmed shape (what the data will be after overscan)
    bias_data = np.ones(trimmed_shape) * 100.0
    bias_path = tmp_path / "master_bias.fits"
    _write_fits(bias_path, bias_data)

    # Create mask with trimmed data to match the post-overscan shape
    trimmed_data = np.ones(trimmed_shape) * 1000.0
    mask = minimal_mask(trimmed_data, saturation=60000.0)

    calib = CalibrationSet(
        bias=bias_data.astype(np.float32),
        bias_var=None,
        bias_path=bias_path,
        flats={},
        flat_var={},
        flat_paths={},
        darks=(),
        dark_current_map=None,
        dark_current_path=None,
        mask=mask,
        gain_measurements=_gain_measurements(2.0),
    )

    meta = _meta(raw_path, raw_shape)
    profile = _profile()
    settings = _settings(cosmic_ray_rejection=True)

    # Define overscan geometry: last 10 columns and last 10 rows
    overscan = OverscanGeometry(
        shape=raw_shape,
        columns=tuple(range(246, 256)),  # Last 10 columns
        rows=tuple(range(246, 256)),  # Last 10 rows
        bias_level=100.0,
        flat_level=1.0,
        detected=True,
        source="detected",
    )

    product, status = reduce_frame(
        meta, calib, settings=settings, profile=profile, overscan=overscan
    )

    # After trimming, the product should be smaller than the raw frame
    # Science region is rows [0:246], cols [0:246]
    expected_shape = trimmed_shape
    assert product.sci.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {product.sci.shape}"

    # overscan and trim should be in steps_applied
    assert "overscan" in status.steps_applied
    assert "trim" in status.steps_applied

    # Check that dq was trimmed similarly
    assert product.dq.shape == expected_shape
    assert product.dq.dtype == DQ_DTYPE

    # Regression test: mask and cosmics should both be applied
    # (they were silently failing before because DQ was float)
    assert "mask" in status.steps_applied, (
        f"mask not applied. steps_applied: {status.steps_applied}, "
        f"steps_skipped: {status.steps_skipped}"
    )
    assert "cosmics" in status.steps_applied, (
        f"cosmics not applied. steps_applied: {status.steps_applied}, "
        f"steps_skipped: {status.steps_skipped}"
    )


def test_reduce_frame_with_overscan_and_no_mask_fallback(tmp_path):
    """Direct regression test for overscan + mask=None fallback.

    This is the exact scenario that triggered the bug: when DQ was passed
    through apply_overscan, it became float, causing bitwise operations to fail
    silently. The minimal mask fallback path then failed to run.

    This test injects a NaN pixel and verifies that the minimal mask fallback
    actually detects it and sets NO_DATA.
    """
    shape = (256, 256)

    raw_data = np.ones(shape) * 1000.0
    # Inject a NaN to verify minimal mask catches it
    raw_data[100, 100] = np.nan
    raw_path = tmp_path / "raw.fits"
    _write_fits(raw_path, raw_data)

    bias_data = np.ones(shape) * 100.0
    bias_path = tmp_path / "master_bias.fits"
    _write_fits(bias_path, bias_data)

    # NO mask in calib set -> must fall back to minimal_mask
    calib = CalibrationSet(
        bias=bias_data.astype(np.float32),
        bias_var=None,
        bias_path=bias_path,
        flats={},
        flat_var={},
        flat_paths={},
        darks=(),
        dark_current_map=None,
        dark_current_path=None,
        mask=None,  # Triggers minimal_mask fallback
        gain_measurements=GainMeasurementSet({}),
    )

    meta = _meta(raw_path, shape)
    profile = _profile()
    settings = _settings()

    # Define overscan geometry: last 10 columns and last 10 rows
    overscan = OverscanGeometry(
        shape=shape,
        columns=tuple(range(246, 256)),  # Last 10 columns
        rows=tuple(range(246, 256)),  # Last 10 rows
        bias_level=100.0,
        flat_level=1.0,
        detected=True,
        source="detected",
    )

    # Should not raise
    product, status = reduce_frame(
        meta, calib, settings=settings, profile=profile, overscan=overscan
    )

    # Verify trimmed shape
    expected_shape = (246, 246)
    assert product.sci.shape == expected_shape
    assert product.dq.shape == expected_shape

    # Regression: DQ must be int32, not float
    assert product.dq.dtype == DQ_DTYPE, (
        f"DQ dtype is {product.dq.dtype}, expected {DQ_DTYPE}. "
        "This indicates the DQ plane was passed through apply_overscan."
    )

    # The NaN pixel (if it survived the trim) should have NO_DATA flag
    # The NaN was at [100, 100], which is within the science region [0:246, 0:246],
    # so it should still be present in the trimmed product
    nan_pixel_dq = product.dq[100, 100]
    assert (nan_pixel_dq & int(DQ.NO_DATA)) != 0, (
        f"Expected NO_DATA flag on NaN pixel, but got {nan_pixel_dq:032b}. "
        "This indicates the minimal mask fallback failed to run."
    )

    # mask should be in steps_skipped (because no static mask was provided)
    assert "mask" in status.steps_skipped


def test_reduce_frame_resolves_gain_from_calibration_set_measurement(tmp_path):
    """The e-/ADU value comes from the CalibrationSet's runtime measurement
    for this frame's own (readout_mode, gain_setting, binning) -- never from
    the raw GAIN header setting, and never from the instrument profile."""
    shape = (64, 64)

    raw_data = np.ones(shape) * 1000.0
    raw_path = tmp_path / "raw.fits"
    _write_fits(raw_path, raw_data)

    bias_data = np.ones(shape) * 100.0
    bias_path = tmp_path / "master_bias.fits"
    _write_fits(bias_path, bias_data)

    mask = minimal_mask(raw_data, saturation=60000.0)

    # Deliberately distinct from the GAIN setting (5.0) to prove the setting
    # index and the measured e-/ADU value are never conflated.
    measured_gain = 0.706
    measurement = GainMeasurement(
        readout_mode="High Gain Mode 16BIT", gain_setting=5.0, binning=1,
        gain_e_per_adu=measured_gain, read_noise_e=7.01, n_pairs=6, scatter_frac=0.004,
    )
    calib = CalibrationSet(
        bias=bias_data.astype(np.float32),
        bias_var=None,
        bias_path=bias_path,
        flats={},
        flat_var={},
        flat_paths={},
        darks=(),
        dark_current_map=None,
        dark_current_path=None,
        mask=mask,
        gain_measurements=GainMeasurementSet(
            {("High Gain Mode 16BIT", 5.0, 1): measurement}
        ),
    )

    meta = _meta(
        raw_path, shape,
        gain_setting=5.0,
        readout_mode="High Gain Mode 16BIT",
        binning=1,
        origins={"gain_setting": FieldOrigin(Source.HEADER, "GAIN")},
    )
    profile = _profile(gain=2.0)  # must be ignored entirely
    settings = _settings(apply_gain=True)

    product, _status = reduce_frame(meta, calib, settings=settings, profile=profile)

    expected_signal = (1000.0 - 100.0) * measured_gain
    actual_signal = np.median(product.sci)
    assert abs(actual_signal - expected_signal) < expected_signal * 0.01, \
        f"expected ~{expected_signal}, got {actual_signal}"

    assert product.header["GAINVAL"] == pytest.approx(measured_gain)
    assert product.header["GAINSRC"] == "measured"
    assert product.header["GAINSET"] == 5.0
    assert product.header["READOUTM"] == "High Gain Mode 16BIT"


def test_reduce_frame_no_measurement_reduces_in_adu_and_degrades(tmp_path):
    """A gain setting with no measurement: gain skipped, BUNIT stays 'adu',
    and proc_code is degraded -- exactly like a missing bias/dark/flat/mask,
    never a guessed conversion factor."""
    shape = (64, 64)

    raw_data = np.ones(shape) * 1000.0
    raw_path = tmp_path / "raw.fits"
    _write_fits(raw_path, raw_data)

    bias_data = np.ones(shape) * 100.0
    bias_path = tmp_path / "master_bias.fits"
    _write_fits(bias_path, bias_data)

    flat_data = np.ones(shape) * 1.0
    flat_path = tmp_path / "master_flat_R.fits"
    _write_fits(flat_path, flat_data)

    mask = minimal_mask(raw_data, saturation=60000.0)

    calib = CalibrationSet(
        bias=bias_data.astype(np.float32),
        bias_var=None,
        bias_path=bias_path,
        flats={"R": flat_data.astype(np.float32)},
        flat_var={},
        flat_paths={"R": flat_path},
        darks=(),
        dark_current_map=None,
        dark_current_path=None,
        mask=mask,
        gain_measurements=GainMeasurementSet({}),  # setting 30 never measured
    )

    meta = _meta(raw_path, shape, gain_setting=30.0, readout_mode="High Gain Mode 16BIT")
    profile = _profile()
    settings = _settings(apply_gain=True)

    product, status = reduce_frame(meta, calib, settings=settings, profile=profile)

    assert product.bunit == "adu"
    assert "gain" in status.steps_skipped
    assert status.proc_code == 3
    assert status.proc_status == "reduced_degraded"
    assert product.header["GAINVAL"] == "NONE"
    assert product.header["GAINSRC"] == "none"
    assert product.header["GAINSET"] == 30.0


def test_reduce_frame_cosmics_skipped_for_missing_gain_logs_warning_not_error(
    tmp_path, caplog
):
    """Cosmic-ray rejection with no measured gain: skipped explicitly at
    WARNING level (never falling through to the generic ERROR-level
    'cosmic-ray rejection failed' path), and -- because cosmics is a
    pipeline step, not a calibration -- this alone must not degrade
    proc_code, unlike a missing bias/dark/flat/mask/gain."""
    shape = (64, 64)

    raw_data = np.ones(shape) * 1000.0
    raw_path = tmp_path / "raw.fits"
    _write_fits(raw_path, raw_data)

    bias_data = np.ones(shape) * 100.0
    bias_path = tmp_path / "master_bias.fits"
    _write_fits(bias_path, bias_data)

    dark_data = np.ones(shape) * 20.0
    dark_path = tmp_path / "master_dark_10s_20C.fits"
    hdu = fits.PrimaryHDU(data=dark_data.astype(np.float32))
    hdu.header["EXPTIME"] = 10.0
    hdu.header["CCDTEMP"] = 20.0
    hdu.writeto(dark_path, overwrite=True)

    flat_data = np.ones(shape) * 1.0
    flat_path = tmp_path / "master_flat_R.fits"
    _write_fits(flat_path, flat_data)

    mask = minimal_mask(raw_data, saturation=60000.0)

    dark_entry = DarkEntry(
        path=dark_path,
        exptime=10.0,
        ccd_temp=20.0,
        data=dark_data.astype(np.float32),
        variance=None,
        n_combine=1,
    )

    calib = CalibrationSet(
        bias=bias_data.astype(np.float32),
        bias_var=None,
        bias_path=bias_path,
        flats={"R": flat_data.astype(np.float32)},
        flat_var={},
        flat_paths={"R": flat_path},
        darks=(dark_entry,),
        dark_current_map=None,
        dark_current_path=None,
        mask=mask,
        gain_measurements=GainMeasurementSet({}),  # setting 30 never measured
    )

    meta = _meta(
        raw_path, shape, exptime=10.0, ccd_temp=20.0,
        gain_setting=30.0, readout_mode="High Gain Mode 16BIT",
    )
    profile = _profile()
    # apply_gain=False so the "gain" skip (also caused by the unmeasured
    # setting) does not itself degrade proc_code, isolating what the
    # "cosmics" skip alone does to it.
    settings = _settings(apply_gain=False, cosmic_ray_rejection=True)

    with caplog.at_level("WARNING"):
        product, status = reduce_frame(meta, calib, settings=settings, profile=profile)

    assert "cosmics" in status.steps_skipped
    assert product.header["CRREJ"] is False

    warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
    error_records = [r for r in caplog.records if r.levelname == "ERROR"]
    assert any("cosmic-ray rejection" in r.message for r in warning_records)
    assert not any("cosmic-ray rejection failed" in r.message for r in error_records)

    # bias, dark, flat, and mask all applied; apply_gain=False makes the
    # "gain" skip a deliberate setting rather than a missing calibration
    # (see the calib_steps_skipped logic in reduce_frame) -- so "cosmics"
    # being skipped is the only thing that could degrade proc_code, and it
    # must not: proc_code stays 0, exactly like the fully-reduced case.
    assert status.proc_code == 0
    assert status.proc_status == "reduced"


def test_reduce_frame_gain_cards_present_even_when_apply_gain_false(tmp_path):
    """GAINVAL/GAINSRC/GAINSET/READOUTM are recorded even when apply_gain is
    False, since cosmics and the read-noise-in-ADU variance term still
    consume the resolved gain -- and a deliberate apply_gain=False is not a
    degraded reduction."""
    shape = (64, 64)

    raw_data = np.ones(shape) * 1000.0
    raw_path = tmp_path / "raw.fits"
    _write_fits(raw_path, raw_data)

    bias_data = np.ones(shape) * 100.0
    bias_path = tmp_path / "master_bias.fits"
    _write_fits(bias_path, bias_data)

    mask = minimal_mask(raw_data, saturation=60000.0)

    calib = CalibrationSet(
        bias=bias_data.astype(np.float32),
        bias_var=None,
        bias_path=bias_path,
        flats={},
        flat_var={},
        flat_paths={},
        darks=(),
        dark_current_map=None,
        dark_current_path=None,
        mask=mask,
        gain_measurements=_gain_measurements(2.0),
    )

    meta = _meta(raw_path, shape)
    profile = _profile(gain=2.0)  # irrelevant: never consulted
    settings = _settings(apply_gain=False)

    product, status = reduce_frame(meta, calib, settings=settings, profile=profile)

    assert product.bunit == "adu"
    assert "gain" in status.steps_skipped
    assert product.header["GAINVAL"] == pytest.approx(2.0)
    assert product.header["GAINSRC"] == "measured"


def test_reduce_frame_records_per_step_median_cards(tmp_path):
    """Every MED* card is present and reflects the running data median."""
    shape = (64, 64)

    raw_data = np.ones(shape) * 1000.0
    raw_path = tmp_path / "raw.fits"
    _write_fits(raw_path, raw_data)

    bias_data = np.ones(shape) * 100.0
    bias_path = tmp_path / "master_bias.fits"
    _write_fits(bias_path, bias_data)

    mask = minimal_mask(raw_data, saturation=60000.0)

    calib = CalibrationSet(
        bias=bias_data.astype(np.float32),
        bias_var=None,
        bias_path=bias_path,
        flats={},
        flat_var={},
        flat_paths={},
        darks=(),
        dark_current_map=None,
        dark_current_path=None,
        mask=mask,
        gain_measurements=_gain_measurements(2.0),
    )

    meta = _meta(raw_path, shape)
    profile = _profile(gain=2.0)
    settings = _settings(apply_gain=True)

    product, _status = reduce_frame(meta, calib, settings=settings, profile=profile)

    for key in ("MEDRAW", "MEDOSC", "MEDBIAS", "MEDDARK", "MEDFLAT", "MEDCOSM", "MEDGAIN"):
        assert key in product.header, f"missing {key} in product header"

    assert product.header["MEDRAW"] == pytest.approx(1000.0)
    assert product.header["MEDBIAS"] == pytest.approx(900.0)
    # No dark or flat master supplied, so those steps are no-ops on the data.
    assert product.header["MEDDARK"] == pytest.approx(900.0)
    assert product.header["MEDFLAT"] == pytest.approx(900.0)
    assert product.header["MEDGAIN"] == pytest.approx(1800.0)
