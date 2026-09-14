"""Tests for robo43.calib.set.

Builds its own synthetic FITS files and FrameMeta objects; does not use
tests/conftest.py.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from robo43.calib.gain import GainMeasurement, GainMeasurementSet
from robo43.calib.masks import MaskSet
from robo43.calib.set import CalibrationSet
from robo43.config import CalibSettings, RuntimeSettings, Settings
from robo43.instruments.profile import InstrumentLayout, InstrumentProfile


def _profile(gain: float = 2.0) -> InstrumentProfile:
    """Create a test instrument profile."""
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
    """Create a test settings object."""
    return Settings(calib=CalibSettings(**calib_overrides), runtime=RuntimeSettings())


def _write_fits(path: Path, data: np.ndarray, var: np.ndarray | None = None) -> None:
    """Write a FITS file with optional VAR extension."""
    primary = fits.PrimaryHDU(data=data.astype(np.float32))
    hdul = fits.HDUList([primary])
    if var is not None:
        var_hdu = fits.ImageHDU(data=var.astype(np.float32), name="VAR")
        hdul.append(var_hdu)
    hdul.writeto(path, overwrite=True)


def test_empty_directory(tmp_path):
    """Empty calibdir: all fields None/empty, no raise, flat_for returns (None, None)."""
    calset = CalibrationSet.load(
        tmp_path,
        settings=_settings(),
        profile=_profile(),
    )
    assert calset.bias is None
    assert calset.bias_var is None
    assert calset.bias_path is None
    assert calset.flats == {}
    assert calset.flat_var == {}
    assert calset.flat_paths == {}
    assert calset.darks == ()
    assert calset.dark_current_map is None
    assert calset.dark_current_path is None
    assert calset.mask is None
    assert calset.gain_measurements.measurements == {}

    flat, path = calset.flat_for("R-Bessell")
    assert flat is None
    assert path is None


def test_none_calibdir():
    """calibdir=None: all fields None/empty."""
    calset = CalibrationSet.load(
        None,
        settings=_settings(),
        profile=_profile(),
    )
    assert calset.bias is None
    assert calset.flats == {}
    assert calset.darks == ()


def test_nonexistent_calibdir(tmp_path):
    """Non-existent calibdir: all fields None/empty."""
    fake_dir = tmp_path / "does_not_exist"
    calset = CalibrationSet.load(
        fake_dir,
        settings=_settings(),
        profile=_profile(),
    )
    assert calset.bias is None
    assert calset.flats == {}
    assert calset.darks == ()


def test_only_bias(tmp_path):
    """Directory with only master_bias.fits: bias loaded, others empty."""
    bias_data = np.full((10, 10), 100.0)
    bias_path = tmp_path / "master_bias.fits"
    _write_fits(bias_path, bias_data)

    calset = CalibrationSet.load(
        tmp_path,
        settings=_settings(),
        profile=_profile(),
    )
    assert calset.bias is not None
    assert calset.bias.shape == (10, 10)
    assert float(np.mean(calset.bias)) == pytest.approx(100.0, abs=1.0)
    assert calset.bias_var is None
    assert calset.bias_path == bias_path
    assert calset.flats == {}
    assert calset.darks == ()
    assert calset.mask is None


def test_bias_with_variance(tmp_path):
    """Master bias with VAR extension."""
    bias_data = np.full((10, 10), 100.0)
    bias_var = np.full((10, 10), 5.0)
    bias_path = tmp_path / "master_bias.fits"
    _write_fits(bias_path, bias_data, var=bias_var)

    calset = CalibrationSet.load(
        tmp_path,
        settings=_settings(),
        profile=_profile(),
    )
    assert calset.bias is not None
    assert calset.bias_var is not None
    assert float(np.mean(calset.bias_var)) == pytest.approx(5.0, abs=0.1)


def test_two_filter_flats(tmp_path):
    """Two flats for different filters, with _dome suffix."""
    flat_r = np.full((10, 10), 1.0)
    flat_h = np.full((10, 10), 1.0)

    path_r = tmp_path / "master_flat_R-Bessell_dome.fits"
    path_h = tmp_path / "master_flat_Halpha_dome.fits"
    _write_fits(path_r, flat_r)
    _write_fits(path_h, flat_h)

    calset = CalibrationSet.load(
        tmp_path,
        settings=_settings(),
        profile=_profile(),
    )
    assert set(calset.flats.keys()) == {"R-Bessell", "Halpha"}
    assert calset.flats["R-Bessell"] is not None
    assert calset.flats["Halpha"] is not None
    assert calset.flat_paths["R-Bessell"] == path_r
    assert calset.flat_paths["Halpha"] == path_h


def test_dome_preferred_over_sky(tmp_path):
    """When both dome and sky flats exist for same filter, dome is preferred."""
    flat_dome = np.full((10, 10), 1.0)
    flat_sky = np.full((10, 10), 0.9)

    path_dome = tmp_path / "master_flat_V-Bessell_dome.fits"
    path_sky = tmp_path / "master_flat_V-Bessell_sky.fits"
    _write_fits(path_dome, flat_dome)
    _write_fits(path_sky, flat_sky)

    calset = CalibrationSet.load(
        tmp_path,
        settings=_settings(),
        profile=_profile(),
    )
    assert "V-Bessell" in calset.flats
    # Should have loaded dome (mean ~1.0) not sky (mean ~0.9)
    assert float(np.mean(calset.flats["V-Bessell"])) == pytest.approx(1.0, abs=0.1)
    assert calset.flat_paths["V-Bessell"] == path_dome


def test_bare_flat_beats_suffixed(tmp_path):
    """Bare flat name (no suffix) takes precedence over suffixed versions."""
    flat_bare = np.full((10, 10), 1.2)
    flat_dome = np.full((10, 10), 1.0)

    path_bare = tmp_path / "master_flat_I-Bessell.fits"
    path_dome = tmp_path / "master_flat_I-Bessell_dome.fits"
    _write_fits(path_bare, flat_bare)
    _write_fits(path_dome, flat_dome)

    calset = CalibrationSet.load(
        tmp_path,
        settings=_settings(),
        profile=_profile(),
    )
    # Should have loaded bare (mean ~1.2) not dome (mean ~1.0)
    assert float(np.mean(calset.flats["I-Bessell"])) == pytest.approx(1.2, abs=0.1)
    assert calset.flat_paths["I-Bessell"] == path_bare


def test_hyphenated_filter_name_preserved(tmp_path):
    """Filter names with hyphens round-trip correctly (suffix parsing edge case)."""
    flat_data = np.full((10, 10), 1.0)

    # I-Bessell with _dome suffix should parse correctly
    path = tmp_path / "master_flat_I-Bessell_dome.fits"
    _write_fits(path, flat_data)

    calset = CalibrationSet.load(
        tmp_path,
        settings=_settings(),
        profile=_profile(),
    )
    # Filter name should be "I-Bessell", not "I" or something else
    assert "I-Bessell" in calset.flats
    assert len(calset.flats) == 1


def test_filters_parameter_loads_only_requested(tmp_path):
    """filters parameter: load only those filters, skip others."""
    flat_r = np.full((10, 10), 1.0)
    flat_h = np.full((10, 10), 1.0)
    flat_v = np.full((10, 10), 1.0)

    _write_fits(tmp_path / "master_flat_R-Bessell_dome.fits", flat_r)
    _write_fits(tmp_path / "master_flat_Halpha_dome.fits", flat_h)
    _write_fits(tmp_path / "master_flat_V-Bessell_dome.fits", flat_v)

    calset = CalibrationSet.load(
        tmp_path,
        settings=_settings(),
        profile=_profile(),
        filters=["Halpha"],
    )
    assert "Halpha" in calset.flats
    assert "R-Bessell" not in calset.flats
    assert "V-Bessell" not in calset.flats


def test_dark_with_headers(tmp_path):
    """Dark frame with EXPTIME and CCD-TEMP headers."""
    dark_data = np.full((10, 10), 50.0)
    path = tmp_path / "master_dark_10s_-10C.fits"

    hdu = fits.PrimaryHDU(data=dark_data.astype(np.float32))
    hdu.header["EXPTIME"] = 10.0
    hdu.header["CCD-TEMP"] = -10.0
    hdu.header["NCOMBINE"] = 5
    hdu.writeto(path, overwrite=True)

    calset = CalibrationSet.load(
        tmp_path,
        settings=_settings(),
        profile=_profile(),
    )
    assert len(calset.darks) == 1
    entry = calset.darks[0]
    assert entry.exptime == 10.0
    assert entry.ccd_temp == -10.0
    assert entry.n_combine == 5
    assert entry.data is not None
    assert entry.path == path


def test_dark_missing_exptime_skipped(tmp_path):
    """Dark without EXPTIME header is skipped with WARNING."""
    dark_data = np.full((10, 10), 50.0)
    path = tmp_path / "master_dark_noexp.fits"

    hdu = fits.PrimaryHDU(data=dark_data.astype(np.float32))
    # No EXPTIME
    hdu.header["CCD-TEMP"] = -10.0
    hdu.writeto(path, overwrite=True)

    calset = CalibrationSet.load(
        tmp_path,
        settings=_settings(),
        profile=_profile(),
    )

    assert len(calset.darks) == 0


def test_dark_current_map_from_exact_name(tmp_path):
    """dark_current_map.fits is loaded."""
    dc_data = np.full((10, 10), 100.0)
    path = tmp_path / "dark_current_map.fits"
    _write_fits(path, dc_data)

    calset = CalibrationSet.load(
        tmp_path,
        settings=_settings(),
        profile=_profile(),
    )
    assert calset.dark_current_map is not None
    assert float(np.mean(calset.dark_current_map)) == pytest.approx(100.0, abs=1.0)
    assert calset.dark_current_path == path


def test_dark_current_map_fallback_to_master_dark_current(tmp_path):
    """master_dark_current.fits is loaded if dark_current_map.fits absent."""
    dc_data = np.full((10, 10), 100.0)
    path = tmp_path / "master_dark_current.fits"
    _write_fits(path, dc_data)

    calset = CalibrationSet.load(
        tmp_path,
        settings=_settings(),
        profile=_profile(),
    )
    assert calset.dark_current_map is not None
    assert calset.dark_current_path == path


def test_mask_loaded_from_mask_fits(tmp_path):
    """mask.fits is loaded via MaskSet.read()."""
    # Build a minimal mask
    dq_data = np.zeros((10, 10), dtype=np.int32)
    dq_data[2:4, 3:5] = 1 << 0  # Set some HOT pixels

    mask = MaskSet(dq=dq_data, summary={"hot": 0.01, "any": 0.01})
    mask_path = tmp_path / "mask.fits"
    mask.write(mask_path, overwrite=True)

    calset = CalibrationSet.load(
        tmp_path,
        settings=_settings(),
        profile=_profile(),
    )
    assert calset.mask is not None
    assert calset.mask.dq is not None
    assert calset.mask.dq.shape == (10, 10)


def test_mask_missing_does_not_raise(tmp_path):
    """Absent mask.fits: mask field is None, no raise."""
    calset = CalibrationSet.load(
        tmp_path,
        settings=_settings(),
        profile=_profile(),
    )
    assert calset.mask is None


def test_mask_corrupt_handled_gracefully(tmp_path):
    """Corrupt mask.fits: logged as WARNING, mask=None, no raise."""
    mask_path = tmp_path / "mask.fits"
    # Write an invalid FITS file
    mask_path.write_text("not a fits file")

    calset = CalibrationSet.load(
        tmp_path,
        settings=_settings(),
        profile=_profile(),
    )

    assert calset.mask is None


def test_flat_for_returns_correct_pair(tmp_path):
    """flat_for() returns (array, path) for known filters and (None, None) for unknown."""
    flat_r = np.full((10, 10), 1.0)
    path_r = tmp_path / "master_flat_R-Bessell_dome.fits"
    _write_fits(path_r, flat_r)

    calset = CalibrationSet.load(
        tmp_path,
        settings=_settings(),
        profile=_profile(),
    )

    # Known filter
    arr, path = calset.flat_for("R-Bessell")
    assert arr is not None
    assert path == path_r

    # Unknown filter
    arr, path = calset.flat_for("Unknown-Filter")
    assert arr is None
    assert path is None


def test_dark_for_with_empty_darks_returns_none_method(tmp_path):
    """dark_for() with no darks returns method='none', entry=None."""
    calset = CalibrationSet.load(
        tmp_path,
        settings=_settings(),
        profile=_profile(),
    )

    selection = calset.dark_for(
        exptime=10.0,
        ccd_temp=-10.0,
        settings=_settings(),
        profile=_profile(),
    )
    assert selection.entry is None
    assert selection.method == "none"
    assert selection.scale == 0.0
    assert "no darks available" in selection.reason


def test_checksums_includes_loaded_files(tmp_path):
    """checksums() returns sha256 for each loaded master."""
    bias_data = np.full((10, 10), 100.0)
    flat_data = np.full((10, 10), 1.0)

    bias_path = tmp_path / "master_bias.fits"
    flat_path = tmp_path / "master_flat_R-Bessell_dome.fits"
    _write_fits(bias_path, bias_data)
    _write_fits(flat_path, flat_data)

    calset = CalibrationSet.load(
        tmp_path,
        settings=_settings(),
        profile=_profile(),
    )

    checksums = calset.checksums()
    assert "bias" in checksums
    assert "flat_R-Bessell" in checksums
    assert len(checksums["bias"]) == 64  # SHA256 hex is 64 chars
    assert len(checksums["flat_R-Bessell"]) == 64


def test_checksums_omits_absent_files(tmp_path):
    """checksums() omits files that were not loaded."""
    calset = CalibrationSet.load(
        tmp_path,
        settings=_settings(),
        profile=_profile(),
    )
    checksums = calset.checksums()
    assert checksums == {}


def test_picklable_and_round_trip(tmp_path):
    """CalibrationSet is picklable; arrays compare equal after pickle round-trip."""
    bias_data = np.full((10, 10), 100.0)
    flat_data = np.full((10, 10), 1.0)

    _write_fits(tmp_path / "master_bias.fits", bias_data)
    _write_fits(tmp_path / "master_flat_R-Bessell_dome.fits", flat_data)

    original = CalibrationSet.load(
        tmp_path,
        settings=_settings(),
        profile=_profile(),
    )

    # Pickle and unpickle
    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)

    # Verify arrays match
    assert np.array_equal(original.bias, restored.bias)
    assert np.array_equal(
        original.flats["R-Bessell"],
        restored.flats["R-Bessell"],
    )
    assert original.bias_path == restored.bias_path
    assert original.flat_paths == restored.flat_paths


def test_no_logger_in_slots():
    """CalibrationSet has only expected slots, no logger or file handles."""
    expected_slots = {
        "bias",
        "bias_var",
        "bias_path",
        "flats",
        "flat_var",
        "flat_paths",
        "darks",
        "dark_current_map",
        "dark_current_path",
        "mask",
        "gain_measurements",
    }
    actual_slots = set(CalibrationSet.__slots__)
    assert actual_slots == expected_slots


def test_gain_measurements_absent_defaults_empty(tmp_path):
    """No gain_measurements.json in calibdir: empty set, gain_for() is None."""
    calset = CalibrationSet.load(tmp_path, settings=_settings(), profile=_profile())
    assert calset.gain_measurements.measurements == {}
    assert calset.gain_for("UNKNOWN", 10.0, 1) is None


def test_gain_measurements_loaded_from_cache_file(tmp_path):
    """gain_measurements.json in calibdir is loaded and queryable via gain_for()."""
    measurement = GainMeasurement(
        readout_mode="High Gain Mode 16BIT", gain_setting=10.0, binning=2,
        gain_e_per_adu=0.706, read_noise_e=7.01, n_pairs=6, scatter_frac=0.004,
    )
    GainMeasurementSet({("High Gain Mode 16BIT", 10.0, 2): measurement}).write(
        tmp_path / "gain_measurements.json"
    )

    calset = CalibrationSet.load(tmp_path, settings=_settings(), profile=_profile())

    found = calset.gain_for("High Gain Mode 16BIT", 10.0, 2)
    assert found is not None
    assert found.gain_e_per_adu == pytest.approx(0.706)
    assert calset.gain_for("High Gain Mode 16BIT", 30.0, 2) is None
