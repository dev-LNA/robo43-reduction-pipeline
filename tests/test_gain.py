"""Tests for robo43.calib.gain.

Builds its own synthetic FITS files and FrameMeta objects; does not use
tests/conftest.py. Synthetic flats/biases are generated directly in
electrons (Poisson photon noise + Gaussian read noise) and converted to ADU
by a chosen true gain, so photon-transfer recovery of that gain is a
genuine test of the measurement, not a tautology.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from astropy.io import fits

from robo43.calib.bias import resolve_common_gain
from robo43.calib.gain import (
    GainMeasurement,
    GainMeasurementSet,
    measure_gain,
    no_measurement_cards,
)
from robo43.calib.set import CalibrationSet
from robo43.config import CalibSettings, RuntimeSettings, Settings
from robo43.exceptions import CalibrationError
from robo43.instruments.profile import InstrumentLayout, InstrumentProfile
from robo43.io.meta import FrameMeta
from robo43.reduce.frame import reduce_frame

TRUE_K = 0.706  # e-/ADU, the reference value measured on real ROBO43 data
TRUE_RN_E = 7.0  # e-
BIAS_LEVEL_ADU = 500.0


def _profile() -> InstrumentProfile:
    return InstrumentProfile(
        name="test-instrument",
        matches={},
        pixel_scale_arcsec=0.5,
        gain_e_per_adu=None,  # never a fallback for this instrument
        read_noise_e=99.0,  # deliberately wrong: proves the measured RN wins
        saturation_adu=60000.0,
        nonlinearity_adu=55000.0,
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
    return Settings(calib=CalibSettings(**calib_overrides), runtime=RuntimeSettings())


def _write_fits(path, data) -> None:
    hdu = fits.PrimaryHDU(data=data.astype(np.float32))
    hdu.writeto(path, overwrite=True)


def _meta(path, shape, *, frame_type, filter=None, exptime=5.0,
          gain_setting=10.0, readout_mode="High Gain Mode 16BIT", binning=1) -> FrameMeta:
    return FrameMeta(
        path=path,
        naxis1=shape[1],
        naxis2=shape[0],
        frame_type=frame_type,
        exptime=exptime,
        date_obs=None,
        filter=filter,
        ra_deg=None,
        dec_deg=None,
        pixel_scale=None,
        ccd_temp=None,
        gain_setting=gain_setting,
        read_noise=None,
        saturation=None,
        airmass=None,
        object="DomeFlat" if frame_type == "flat" else None,
        telescope=None,
        instrument=None,
        observer=None,
        binning=binning,
        readout_mode=readout_mode,
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


def _read_noise_with_tails(rng, shape, read_noise_e, tail_frac=0.0, tail_ratio=5.0):
    """Read-noise realization with an optional heavy non-Gaussian tail.

    With ``tail_frac=0`` (the default) this is plain Gaussian read noise, as
    before. With ``tail_frac>0``, a ``tail_frac`` fraction of pixels instead
    draw from a much wider Gaussian (``tail_ratio`` times the core sigma) --
    reproducing the hot-pixel/random-telegraph-noise tails measured in real
    ROBO43 bias differences (see robo43.calib.gain's module docstring). Core
    and tail sigmas are solved so the *total* variance still equals
    ``read_noise_e ** 2`` -- only the shape of the distribution changes, not
    its nominal read noise.
    """
    if tail_frac <= 0.0:
        return rng.normal(0.0, read_noise_e, size=shape)
    sigma_core = read_noise_e / np.sqrt((1.0 - tail_frac) + tail_frac * tail_ratio**2)
    sigma_tail = tail_ratio * sigma_core
    core = rng.normal(0.0, sigma_core, size=shape)
    tail = rng.normal(0.0, sigma_tail, size=shape)
    is_tail = rng.random(shape) < tail_frac
    return np.where(is_tail, tail, core)


def _bias_frame_adu(rng, shape, *, bias_adu=BIAS_LEVEL_ADU, read_noise_e=TRUE_RN_E, k=TRUE_K,
                     tail_frac=0.0, tail_ratio=5.0):
    read_noise_e_realized = _read_noise_with_tails(rng, shape, read_noise_e, tail_frac, tail_ratio)
    return bias_adu + read_noise_e_realized / k


def _flat_frame_adu(rng, shape, *, signal_e, bias_adu=BIAS_LEVEL_ADU, read_noise_e=TRUE_RN_E, k=TRUE_K,
                     tail_frac=0.0, tail_ratio=5.0):
    photons_e = rng.poisson(signal_e, size=shape).astype(np.float64)
    read_noise_e_realized = _read_noise_with_tails(rng, shape, read_noise_e, tail_frac, tail_ratio)
    return bias_adu + (photons_e + read_noise_e_realized) / k


def test_measure_gain_recovers_known_k_from_synthetic_ptc(tmp_path):
    """PTC on synthetic flats/biases (real Poisson + read noise, with a heavy
    non-Gaussian read-noise tail matching real ROBO43 bias frames) recovers
    the true injected gain and read noise to within a few percent.

    The injected tail (2% of pixels at 5x the core read-noise sigma, total
    variance still exactly ``read_noise_e ** 2``) reproduces the ~2% masked /
    ~30% variance-drop behaviour measured on real ROBO43 biases under 3-sigma
    clipping (see robo43.calib.gain's module docstring). Against the old
    sigma-clipped-variance estimator this same synthetic data recovers K
    biased ~3% high and read noise biased ~14% low -- confirmed by running
    this exact test against the pre-fix estimator, which failed the
    read_noise_e assertion below (see the change spec / PR description for
    the reverted-and-confirmed run). This test was strengthened specifically
    to catch that regression.
    """
    rng = np.random.default_rng(2024)
    shape = (400, 400)
    tail_kwargs = {"tail_frac": 0.02, "tail_ratio": 5.0}

    metas = []
    for i in range(4):
        path = tmp_path / f"bias_{i}.fits"
        _write_fits(path, _bias_frame_adu(rng, shape, **tail_kwargs))
        metas.append(_meta(path, shape, frame_type="bias"))

    # Six "filters" at different signal levels, like the real reference
    # measurement (15550-30767 ADU span, six filters).
    signal_levels_e = [10000.0, 13000.0, 16000.0, 19000.0, 22000.0, 25000.0]
    for i, signal_e in enumerate(signal_levels_e):
        for j in range(2):
            path = tmp_path / f"flat_{i}_{j}.fits"
            _write_fits(path, _flat_frame_adu(rng, shape, signal_e=signal_e, **tail_kwargs))
            metas.append(_meta(path, shape, frame_type="flat", filter=f"FILT{i}"))

    profile = _profile()
    settings = _settings()

    result = measure_gain(metas, settings=settings, profile=profile)
    measurement = result.get("High Gain Mode 16BIT", 10.0, 1)

    assert measurement is not None
    assert measurement.gain_e_per_adu == pytest.approx(TRUE_K, rel=0.025)
    assert measurement.read_noise_e == pytest.approx(TRUE_RN_E, rel=0.05)
    assert measurement.n_pairs == len(signal_levels_e)
    assert measurement.source == "measured"


def test_measure_gain_group_with_too_few_frames_yields_no_measurement(tmp_path):
    """A gain setting with fewer than the minimum flats/biases (e.g. all of
    a setting's frames are science, no calibration frames at all) has no
    measurement -- never a guess."""
    rng = np.random.default_rng(7)
    shape = (50, 50)

    # Only ONE bias and ONE flat at setting 30 -- below the minimum of 2.
    metas = [
        _meta(tmp_path / "b0.fits", shape, frame_type="bias", gain_setting=30.0),
    ]
    _write_fits(metas[0].path, _bias_frame_adu(rng, shape))
    flat_meta = _meta(tmp_path / "f0.fits", shape, frame_type="flat", gain_setting=30.0)
    _write_fits(flat_meta.path, _flat_frame_adu(rng, shape, signal_e=10000.0))
    metas.append(flat_meta)

    profile = _profile()
    settings = _settings()

    result = measure_gain(metas, settings=settings, profile=profile)
    assert result.get("High Gain Mode 16BIT", 30.0, 1) is None
    assert result.measurements == {}


def test_measure_gain_disabled_by_setting(tmp_path):
    """calib.gain_measure=False skips measurement entirely, even with ample data."""
    rng = np.random.default_rng(9)
    shape = (50, 50)
    metas = []
    for i in range(3):
        path = tmp_path / f"b_{i}.fits"
        _write_fits(path, _bias_frame_adu(rng, shape))
        metas.append(_meta(path, shape, frame_type="bias"))
    for i in range(3):
        path = tmp_path / f"f_{i}.fits"
        _write_fits(path, _flat_frame_adu(rng, shape, signal_e=10000.0))
        metas.append(_meta(path, shape, frame_type="flat"))

    profile = _profile()
    settings = _settings(gain_measure=False)

    result = measure_gain(metas, settings=settings, profile=profile)
    assert result.measurements == {}


def test_gain_measurement_set_json_roundtrip(tmp_path):
    """write()/read() round-trips a measurement through JSON, as CalibrationSet
    caches it in calibdir."""
    measurement = GainMeasurement(
        readout_mode="High Gain Mode 16BIT", gain_setting=10.0, binning=1,
        gain_e_per_adu=0.706, read_noise_e=7.01, n_pairs=6, scatter_frac=0.004,
    )
    original = GainMeasurementSet({("High Gain Mode 16BIT", 10.0, 1): measurement})
    path = tmp_path / "gain_measurements.json"
    original.write(path)

    restored = GainMeasurementSet.read(path)
    found = restored.get("High Gain Mode 16BIT", 10.0, 1)
    assert found is not None
    assert found.gain_e_per_adu == pytest.approx(0.706)
    assert found.read_noise_e == pytest.approx(7.01)


def test_gain_measurement_set_read_missing_file_returns_empty(tmp_path):
    result = GainMeasurementSet.read(tmp_path / "does_not_exist.json")
    assert result.measurements == {}


def test_masters_refuse_mixing_gain_settings(tmp_path):
    """resolve_common_gain raises rather than silently combining frames shot
    at different GAIN settings into one master."""
    meta_10 = _meta(tmp_path / "a.fits", (5, 5), frame_type="bias", gain_setting=10.0)
    meta_30 = _meta(tmp_path / "b.fits", (5, 5), frame_type="bias", gain_setting=30.0)
    profile = _profile()

    with pytest.raises(CalibrationError, match="inconsistent"):
        resolve_common_gain([meta_10, meta_30], profile=profile)


def test_no_measurement_cards_never_report_a_guessed_value():
    cards = no_measurement_cards(30.0, "High Gain Mode 16BIT")
    card_dict = {c[0]: c[1] for c in cards}
    assert card_dict["GAINVAL"] == "NONE"
    assert card_dict["GAINSRC"] == "none"
    assert card_dict["GAINSET"] == 30.0
    assert card_dict["READOUTM"] == "High Gain Mode 16BIT"


def test_frame_meta_has_no_gain_e_per_adu_field():
    """Structural proof that the GAIN header keyword has no field to flow
    into as an e-/ADU value: FrameMeta carries only gain_setting, never a
    field named 'gain'."""
    field_names = {f.name for f in dataclasses.fields(FrameMeta)}
    assert "gain_setting" in field_names
    assert "gain" not in field_names


def test_gain_header_value_never_used_as_e_per_adu_in_reduction(tmp_path):
    """End-to-end: a frame whose GAIN header reads 10 (a setting index) is
    reduced using the *measured* 0.706 e-/ADU for that setting, never 10."""
    rng = np.random.default_rng(4242)
    shape = (200, 200)

    ptc_metas = []
    for i in range(2):
        path = tmp_path / f"bias_{i}.fits"
        _write_fits(path, _bias_frame_adu(rng, shape))
        ptc_metas.append(_meta(path, shape, frame_type="bias"))
    for i, signal_e in enumerate([10000.0, 15000.0, 20000.0]):
        for j in range(2):
            path = tmp_path / f"flat_{i}_{j}.fits"
            _write_fits(path, _flat_frame_adu(rng, shape, signal_e=signal_e))
            ptc_metas.append(_meta(path, shape, frame_type="flat", filter=f"FILT{i}"))

    profile = _profile()
    settings = _settings()
    gain_measurements = measure_gain(ptc_metas, settings=settings, profile=profile)
    measurement = gain_measurements.get("High Gain Mode 16BIT", 10.0, 1)
    assert measurement is not None
    assert measurement.gain_e_per_adu == pytest.approx(TRUE_K, rel=0.05)
    assert measurement.gain_e_per_adu != pytest.approx(10.0, rel=0.5)

    # Now reduce a science frame whose GAIN header is the same setting (10).
    sci_shape = (32, 32)
    raw_adu = BIAS_LEVEL_ADU + 1000.0
    raw_path = tmp_path / "science.fits"
    _write_fits(raw_path, np.full(sci_shape, raw_adu))
    bias_master = np.full(sci_shape, BIAS_LEVEL_ADU, dtype=np.float32)

    calib = CalibrationSet(
        bias=bias_master,
        bias_var=None,
        bias_path=None,
        flats={},
        flat_var={},
        flat_paths={},
        darks=(),
        dark_current_map=None,
        dark_current_path=None,
        mask=None,
        gain_measurements=gain_measurements,
    )
    sci_meta = _meta(raw_path, sci_shape, frame_type="science")
    product, _status = reduce_frame(
        sci_meta, calib, settings=_settings(apply_gain=True), profile=profile
    )

    expected_signal_e = 1000.0 * measurement.gain_e_per_adu
    actual_signal_e = float(np.median(product.sci))
    assert abs(actual_signal_e - expected_signal_e) < expected_signal_e * 0.01

    # The GAIN header value (10) must never appear as the applied conversion.
    unmeasured_signal_e = 1000.0 * 10.0
    assert abs(actual_signal_e - unmeasured_signal_e) > expected_signal_e
    assert product.header["GAINVAL"] == pytest.approx(measurement.gain_e_per_adu)
    assert product.header["GAINSET"] == 10.0
