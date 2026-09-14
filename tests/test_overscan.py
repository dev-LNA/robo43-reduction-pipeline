"""Tests for robo43.calib.overscan.

Builds synthetic FITS files and FrameMeta objects; does not use conftest.py.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from robo43.calib.overscan import apply_overscan, detect_overscan
from robo43.config import CalibSettings, RuntimeSettings, Settings
from robo43.instruments.profile import InstrumentLayout, InstrumentProfile
from robo43.io.meta import FrameMeta


def _profile(
    gain: float = 2.0, overscan: str | None = None, trim: str | None = None
) -> InstrumentProfile:
    return InstrumentProfile(
        name="test-instrument",
        matches={},
        pixel_scale_arcsec=0.5,
        gain_e_per_adu=gain,
        read_noise_e=5.0,
        saturation_adu=60000.0,
        nonlinearity_adu=None,
        binning_keyword="XBINNING",
        layout=InstrumentLayout(
            science_hdu=0,
            n_amplifiers=1,
            overscan=overscan,
            trim=trim,
            flip_y=False,
        ),
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
    return Settings(
        calib=CalibSettings(**calib_overrides),
        runtime=RuntimeSettings(),
    )


def _write_fits(path, data) -> None:
    hdu = fits.PrimaryHDU(data=data.astype(np.float32))
    hdu.writeto(path, overwrite=True)


def _meta(path, shape, *, frame_type: str) -> FrameMeta:
    return FrameMeta(
        path=path,
        naxis1=shape[1],
        naxis2=shape[0],
        frame_type=frame_type,
        exptime=10.0,
        date_obs=None,
        filter=None,
        ra_deg=None,
        dec_deg=None,
        pixel_scale=None,
        ccd_temp=None,
        gain_setting=None,
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


def test_detect_columns_only(tmp_path):
    """Eight overscan columns on the right at bias level; illuminated at ~40%."""
    h, w = 100, 128
    bias_level = 1000.0
    full_well = 60000.0
    flat_level = full_well * 0.40

    bias_data = np.full((h, w), bias_level, dtype=np.float32)
    bias_path = tmp_path / "bias.fits"
    _write_fits(bias_path, bias_data)
    bias_meta = _meta(bias_path, (h, w), frame_type="bias")

    flat_data = np.full((h, w), flat_level, dtype=np.float32)
    flat_data[:, -8:] = bias_level
    flat_path = tmp_path / "flat.fits"
    _write_fits(flat_path, flat_data)
    flat_meta = _meta(flat_path, (h, w), frame_type="flat")

    profile = _profile()
    settings = _settings()
    geom = detect_overscan(
        [bias_meta, flat_meta],
        settings=settings,
        profile=profile,
    )

    assert geom.detected
    assert geom.source == "detected"
    assert len(geom.columns) == 8
    assert geom.columns == tuple(range(w - 8, w))
    assert len(geom.rows) == 0


def test_detect_rows_and_columns(tmp_path):
    """Detect both overscan rows (top/bottom) and columns (right)."""
    h, w = 100, 128
    bias_level = 1000.0
    flat_level = 24000.0

    bias_data = np.full((h, w), bias_level, dtype=np.float32)
    bias_path = tmp_path / "bias.fits"
    _write_fits(bias_path, bias_data)
    bias_meta = _meta(bias_path, (h, w), frame_type="bias")

    flat_data = np.full((h, w), flat_level, dtype=np.float32)
    flat_data[:5, :] = bias_level
    flat_data[-4:, :] = bias_level
    flat_data[:, -8:] = bias_level
    flat_path = tmp_path / "flat.fits"
    _write_fits(flat_path, flat_data)
    flat_meta = _meta(flat_path, (h, w), frame_type="flat")

    profile = _profile()
    settings = _settings()
    geom = detect_overscan(
        [bias_meta, flat_meta],
        settings=settings,
        profile=profile,
    )

    assert geom.detected
    assert len(geom.rows) == 9
    assert len(geom.columns) == 8


def test_no_overscan_detected(tmp_path):
    """No overscan regions: everything illuminated at flat level."""
    h, w = 100, 128
    bias_level = 1000.0
    flat_level = 24000.0

    bias_data = np.full((h, w), bias_level, dtype=np.float32)
    bias_path = tmp_path / "bias.fits"
    _write_fits(bias_path, bias_data)
    bias_meta = _meta(bias_path, (h, w), frame_type="bias")

    flat_data = np.full((h, w), flat_level, dtype=np.float32)
    flat_path = tmp_path / "flat.fits"
    _write_fits(flat_path, flat_data)
    flat_meta = _meta(flat_path, (h, w), frame_type="flat")

    profile = _profile()
    settings = _settings()
    geom = detect_overscan(
        [bias_meta, flat_meta],
        settings=settings,
        profile=profile,
    )

    assert not geom.detected
    assert geom.source == "detected"
    assert len(geom.columns) == 0
    assert len(geom.rows) == 0

    data = np.full((h, w), 2000.0, dtype=np.float32)
    corrected, cards = apply_overscan(data, geom, settings=settings)

    assert corrected.shape == data.shape
    card_dict = {c[0]: c[1] for c in cards}
    assert card_dict["OSCANCOR"] is False
    assert card_dict["TRIMMED"] is False


def test_apply_overscan_subtracts_and_trims(tmp_path):
    """apply_overscan subtracts median level and trims to science region."""
    h, w = 100, 128
    bias_level = 1000.0
    flat_level = 24000.0

    bias_data = np.full((h, w), bias_level, dtype=np.float32)
    bias_path = tmp_path / "bias.fits"
    _write_fits(bias_path, bias_data)
    bias_meta = _meta(bias_path, (h, w), frame_type="bias")

    flat_data = np.full((h, w), flat_level, dtype=np.float32)
    flat_data[:, -8:] = bias_level
    flat_path = tmp_path / "flat.fits"
    _write_fits(flat_path, flat_data)
    flat_meta = _meta(flat_path, (h, w), frame_type="flat")

    profile = _profile()
    settings = _settings()
    geom = detect_overscan(
        [bias_meta, flat_meta],
        settings=settings,
        profile=profile,
    )

    frame_level = 2000.0
    frame_data = np.full((h, w), frame_level, dtype=np.float32)
    frame_data[:, -8:] = bias_level + 50.0

    corrected, cards = apply_overscan(frame_data, geom, settings=settings)

    card_dict = {c[0]: c[1] for c in cards}
    assert card_dict["OSCANCOR"] is True
    assert card_dict["TRIMMED"] is True

    expected_overscan_level = bias_level + 50.0
    expected_frame_after_oscan = frame_level - expected_overscan_level
    expected_trimmed_shape = (h, w - 8)

    assert corrected.shape == expected_trimmed_shape
    assert np.allclose(corrected, expected_frame_after_oscan)


def test_noisy_flat_intersection_rule(tmp_path):
    """One noisy flat marks a column as bias-level; intersection rule rejects it."""
    h, w = 100, 128
    bias_level = 1000.0
    flat_level = 24000.0

    bias_data = np.full((h, w), bias_level, dtype=np.float32)
    bias_path = tmp_path / "bias.fits"
    _write_fits(bias_path, bias_data)
    bias_meta = _meta(bias_path, (h, w), frame_type="bias")

    flat1_data = np.full((h, w), flat_level, dtype=np.float32)
    flat1_data[:, -8:] = bias_level
    flat1_path = tmp_path / "flat1.fits"
    _write_fits(flat1_path, flat1_data)
    flat1_meta = _meta(flat1_path, (h, w), frame_type="flat")

    flat2_data = np.full((h, w), flat_level, dtype=np.float32)
    flat2_data[:, -8:] = bias_level
    flat2_data[:, 50] = bias_level
    flat2_path = tmp_path / "flat2.fits"
    _write_fits(flat2_path, flat2_data)
    flat2_meta = _meta(flat2_path, (h, w), frame_type="flat")

    profile = _profile()
    settings = _settings()
    geom = detect_overscan(
        [bias_meta, flat1_meta, flat2_meta],
        settings=settings,
        profile=profile,
    )

    assert geom.detected
    assert len(geom.columns) == 8
    assert 50 not in geom.columns


def test_overscan_detect_false_with_profile_region(tmp_path):
    """overscan_detect=False with profile region set returns source='profile'."""
    h, w = 100, 128
    bias_data = np.full((h, w), 1000.0, dtype=np.float32)
    bias_path = tmp_path / "bias.fits"
    _write_fits(bias_path, bias_data)
    bias_meta = _meta(bias_path, (h, w), frame_type="bias")

    flat_data = np.full((h, w), 24000.0, dtype=np.float32)
    flat_path = tmp_path / "flat.fits"
    _write_fits(flat_path, flat_data)
    flat_meta = _meta(flat_path, (h, w), frame_type="flat")

    profile = _profile(overscan="[121:128,1:100]")
    settings = _settings(overscan_detect=False)
    geom = detect_overscan(
        [bias_meta, flat_meta],
        settings=settings,
        profile=profile,
    )

    assert geom.source == "profile"
    assert len(geom.columns) == 8
    assert geom.columns == tuple(range(120, 128))


def test_overscan_detect_false_no_profile_region(tmp_path):
    """overscan_detect=False with no profile region returns source='none'."""
    h, w = 100, 128
    bias_data = np.full((h, w), 1000.0, dtype=np.float32)
    bias_path = tmp_path / "bias.fits"
    _write_fits(bias_path, bias_data)
    bias_meta = _meta(bias_path, (h, w), frame_type="bias")

    flat_data = np.full((h, w), 24000.0, dtype=np.float32)
    flat_path = tmp_path / "flat.fits"
    _write_fits(flat_path, flat_data)
    flat_meta = _meta(flat_path, (h, w), frame_type="flat")

    profile = _profile(overscan=None)
    settings = _settings(overscan_detect=False)
    geom = detect_overscan(
        [bias_meta, flat_meta],
        settings=settings,
        profile=profile,
    )

    assert geom.source == "none"
    assert not geom.detected
    assert len(geom.columns) == 0


def test_missing_bias_frames_returns_empty(tmp_path):
    """Missing bias frames: returns empty geometry and does not raise."""
    h, w = 100, 128
    flat_data = np.full((h, w), 24000.0, dtype=np.float32)
    flat_path = tmp_path / "flat.fits"
    _write_fits(flat_path, flat_data)
    flat_meta = _meta(flat_path, (h, w), frame_type="flat")

    profile = _profile()
    settings = _settings()
    geom = detect_overscan(
        [flat_meta],
        settings=settings,
        profile=profile,
    )

    assert not geom.detected
    assert geom.source == "none"
    assert len(geom.columns) == 0
    assert len(geom.rows) == 0


def test_missing_flat_frames_returns_empty(tmp_path):
    """Missing flat frames: returns empty geometry and does not raise."""
    h, w = 100, 128
    bias_data = np.full((h, w), 1000.0, dtype=np.float32)
    bias_path = tmp_path / "bias.fits"
    _write_fits(bias_path, bias_data)
    bias_meta = _meta(bias_path, (h, w), frame_type="bias")

    profile = _profile()
    settings = _settings()
    geom = detect_overscan(
        [bias_meta],
        settings=settings,
        profile=profile,
    )

    assert not geom.detected
    assert geom.source == "none"


def test_all_header_cards_finite_fits_writable(tmp_path):
    """Every card value from apply_overscan is finite and FITS-writable."""
    h, w = 100, 128
    bias_level = 1000.0
    flat_level = 24000.0

    bias_data = np.full((h, w), bias_level, dtype=np.float32)
    bias_path = tmp_path / "bias.fits"
    _write_fits(bias_path, bias_data)
    bias_meta = _meta(bias_path, (h, w), frame_type="bias")

    flat_data = np.full((h, w), flat_level, dtype=np.float32)
    flat_data[:, -8:] = bias_level
    flat_path = tmp_path / "flat.fits"
    _write_fits(flat_path, flat_data)
    flat_meta = _meta(flat_path, (h, w), frame_type="flat")

    profile = _profile()
    settings = _settings()
    geom = detect_overscan(
        [bias_meta, flat_meta],
        settings=settings,
        profile=profile,
    )

    frame_data = np.full((h, w), 2000.0, dtype=np.float32)
    frame_data[:, -8:] = bias_level + 50.0

    _corrected, cards = apply_overscan(frame_data, geom, settings=settings)

    header = fits.Header()
    for keyword, value, comment in cards:
        try:
            header[keyword] = (value, comment)
        except (ValueError, TypeError) as exc:
            raise AssertionError(
                f"Card {keyword}={value!r} is not FITS-writable: {exc}"
            ) from exc

    for keyword, value, _comment in cards:
        if isinstance(value, (int, float)):
            assert np.isfinite(value), f"Card {keyword} has non-finite value {value}"


def test_isolated_column_rejected_by_min_line_frac(tmp_path):
    """Single isolated bias-level column discarded by overscan_min_line_frac."""
    h, w = 100, 128
    bias_level = 1000.0
    flat_level = 24000.0

    bias_data = np.full((h, w), bias_level, dtype=np.float32)
    bias_path = tmp_path / "bias.fits"
    _write_fits(bias_path, bias_data)
    bias_meta = _meta(bias_path, (h, w), frame_type="bias")

    flat_data = np.full((h, w), flat_level, dtype=np.float32)
    flat_data[:, 50] = bias_level
    flat_path = tmp_path / "flat.fits"
    _write_fits(flat_path, flat_data)
    flat_meta = _meta(flat_path, (h, w), frame_type="flat")

    profile = _profile()
    settings = _settings(overscan_min_line_frac=0.016)
    geom = detect_overscan(
        [bias_meta, flat_meta],
        settings=settings,
        profile=profile,
    )

    assert not geom.detected
    assert 50 not in geom.columns
