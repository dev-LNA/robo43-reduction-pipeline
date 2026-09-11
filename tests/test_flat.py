"""Tests for robo43.calib.flat.

Builds its own synthetic FITS files and FrameMeta objects; does not use
tests/conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from robo43.calib.flat import build_master_flats, group_flats_by_filter, normalise_flat
from robo43.calib.overscan import OverscanGeometry
from robo43.config import CalibSettings, RuntimeSettings, Settings
from robo43.exceptions import CalibrationError
from robo43.instruments.profile import InstrumentLayout, InstrumentProfile
from robo43.io.meta import FrameMeta
from robo43.products.dq import DQ


def _profile(**overrides) -> InstrumentProfile:
    # Extract InstrumentProfile-level overrides
    saturation_adu = overrides.pop("saturation_adu", 60000.0)
    # Remaining kwargs are for InstrumentLayout
    layout = InstrumentLayout(**overrides)
    return InstrumentProfile(
        name="test-instrument",
        matches={},
        pixel_scale_arcsec=0.5,
        gain_e_per_adu=2.0,
        read_noise_e=5.0,
        saturation_adu=saturation_adu,
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


def _meta(path, shape, *, filter, exptime=1.0, object=None) -> FrameMeta:
    return FrameMeta(
        path=path,
        naxis1=shape[1],
        naxis2=shape[0],
        frame_type="flat",
        exptime=exptime,
        date_obs=None,
        filter=filter,
        ra_deg=None,
        dec_deg=None,
        pixel_scale=None,
        ccd_temp=None,
        gain=None,
        read_noise=None,
        saturation=None,
        airmass=None,
        object=object,
        telescope=None,
        instrument=None,
        observer=None,
        binning=None,
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


def _make_flat_group(tmp_path, prefix, *, n, shape, level, sigma, filter, object=None, rng=None):
    """Create a group of synthetic flat frames with a known illumination level."""
    if rng is None:
        rng = np.random.default_rng()
    metas = []
    for i in range(n):
        frame = level + rng.normal(0.0, sigma, size=shape)
        path = tmp_path / f"{prefix}_{i:02d}.fits"
        _write_fits(path, frame)
        metas.append(_meta(path, shape, filter=filter, object=object))
    return metas


def test_two_filters_four_frames_each_distinct_levels(tmp_path):
    """Test basic master flat building: two filters, distinct levels, correct normalization."""
    rng = np.random.default_rng(42)
    h, w = 32, 32
    shape = (h, w)

    # Build two groups with distinct illumination levels
    # saturation is 60000, so valid range is [12000, 42000]
    metas_r = _make_flat_group(
        tmp_path,
        "flat_r",
        n=4,
        shape=shape,
        level=20000.0,
        sigma=50.0,
        filter="R-Bessell",
        object="DomeFlat",
        rng=rng,
    )
    metas_i = _make_flat_group(
        tmp_path,
        "flat_i",
        n=4,
        shape=shape,
        level=25000.0,
        sigma=40.0,
        filter="I-Bessell",
        object="DomeFlat",
        rng=rng,
    )

    metas = metas_r + metas_i

    profile = _profile()
    settings = _settings()
    outdir = tmp_path / "masters"
    masters = build_master_flats(metas, settings=settings, profile=profile, outdir=outdir)

    assert len(masters) == 2
    assert "R-Bessell:dome" in masters
    assert "I-Bessell:dome" in masters

    # Check that both files were written
    assert (outdir / "master_flat_R-Bessell_dome.fits").exists()
    assert (outdir / "master_flat_I-Bessell_dome.fits").exists()

    # Check that normalized flats have central median ~1.0
    master_r = masters["R-Bessell:dome"]
    norm_r = float(np.median(master_r.data[8:24, 8:24]))
    assert abs(norm_r - 1.0) < 1e-6

    master_i = masters["I-Bessell:dome"]
    norm_i = float(np.median(master_i.data[8:24, 8:24]))
    assert abs(norm_i - 1.0) < 1e-6


def test_bias_subtraction_applied(tmp_path):
    """Test that bias subtraction is actually applied and doesn't change normalized result."""
    rng = np.random.default_rng(43)
    h, w = 32, 32
    shape = (h, w)

    # Create two identical groups of flats, one with bias offset in raw data
    bias_offset = 100.0
    level = 20000.0  # Within [12000, 42000] range

    # Group A: no bias offset
    metas_a = _make_flat_group(
        tmp_path,
        "flat_a",
        n=3,
        shape=shape,
        level=level,
        sigma=30.0,
        filter="R-Bessell",
        object="DomeFlat",
        rng=rng,
    )

    # Group B: with bias offset (will be subtracted via bias.data)
    metas_b_raw = []
    for i in range(3):
        frame = (level + bias_offset) + rng.normal(0.0, 30.0, size=shape)
        path = tmp_path / f"flat_b_{i:02d}.fits"
        _write_fits(path, frame)
        metas_b_raw.append(_meta(path, shape, filter="R-Bessell", object="DomeFlat"))

    # Create a "master bias" with the offset
    from robo43.calib.master import MasterFrame

    bias_data = np.full(shape, bias_offset, dtype=np.float32)
    bias_master = MasterFrame(data=bias_data, bunit="adu")

    # Build masters
    profile = _profile()
    settings = _settings()

    # Without bias subtraction
    outdir_a = tmp_path / "masters_no_bias"
    masters_a = build_master_flats(
        metas_a, settings=settings, profile=profile, outdir=outdir_a, bias=None
    )

    # With bias subtraction
    outdir_b = tmp_path / "masters_with_bias"
    masters_b = build_master_flats(
        metas_b_raw, settings=settings, profile=profile, outdir=outdir_b, bias=bias_master
    )

    # After normalization, the two masters should be nearly identical
    master_a = masters_a["R-Bessell:dome"]
    master_b = masters_b["R-Bessell:dome"]

    # Allow some tolerance due to different random noise
    assert np.allclose(master_a.data, master_b.data, atol=0.01)


def test_level_cut_rejects_outliers(tmp_path):
    """Test that frames outside the level-cut range are rejected."""
    rng = np.random.default_rng(44)
    h, w = 32, 32
    shape = (h, w)
    profile = _profile()
    settings = _settings()

    # Frame at a good level (between 12000 and 42000)
    metas = _make_flat_group(
        tmp_path,
        "good",
        n=2,
        shape=shape,
        level=20000.0,  # Between 0.2*60000=12000 and 0.7*60000=42000
        sigma=100.0,
        filter="R-Bessell",
        rng=rng,
    )

    # Frame far below minimum level (0.2 * 60000 = 12000)
    low_frame = 5000.0 + rng.normal(0.0, 50.0, size=shape)
    low_path = tmp_path / "low.fits"
    _write_fits(low_path, low_frame)
    metas.append(_meta(low_path, shape, filter="R-Bessell"))

    # Frame far above maximum level (0.7 * 60000 = 42000)
    high_frame = 50000.0 + rng.normal(0.0, 50.0, size=shape)
    high_path = tmp_path / "high.fits"
    _write_fits(high_path, high_frame)
    metas.append(_meta(high_path, shape, filter="R-Bessell"))

    outdir = tmp_path / "masters"
    masters = build_master_flats(metas, settings=settings, profile=profile, outdir=outdir)

    master = masters["R-Bessell:dome"]

    # Find the FLATNACC and FLATNREJ cards
    card_dict = {c[0]: c[1] for c in master.header_cards}
    assert card_dict["FLATNACC"] == 2
    assert card_dict["FLATNREJ"] == 2


def test_empty_group_skipped_no_file_written(tmp_path):
    """Test that a group where all frames fail level cut is skipped without raising."""
    rng = np.random.default_rng(45)
    h, w = 32, 32
    shape = (h, w)
    profile = _profile()
    settings = _settings()

    # All frames far below minimum level (< 12000)
    metas = []
    for i in range(3):
        frame = 8000.0 + rng.normal(0.0, 50.0, size=shape)
        path = tmp_path / f"bad_{i:02d}.fits"
        _write_fits(path, frame)
        metas.append(_meta(path, shape, filter="R-Bessell"))

    outdir = tmp_path / "masters"
    masters = build_master_flats(metas, settings=settings, profile=profile, outdir=outdir)

    # Should return empty dict
    assert len(masters) == 0
    # File should not exist
    assert not (outdir / "master_flat_R-Bessell_dome.fits").exists()


def test_group_flats_by_filter_separate_dome_and_sky(tmp_path):
    """Test grouping with separate_dome_and_sky=True."""
    h, w = 16, 16
    shape = (h, w)

    # Create dome and sky flats for the same filter
    dome_path = tmp_path / "dome.fits"
    _write_fits(dome_path, np.full(shape, 10000.0))
    dome_meta = _meta(dome_path, shape, filter="R-Bessell", object="DomeFlat")

    sky_path = tmp_path / "sky.fits"
    _write_fits(sky_path, np.full(shape, 10000.0))
    sky_meta = _meta(sky_path, shape, filter="R-Bessell", object="SkyFlat")

    twilight_path = tmp_path / "twilight.fits"
    _write_fits(twilight_path, np.full(shape, 10000.0))
    twilight_meta = _meta(twilight_path, shape, filter="R-Bessell", object="Twilight")

    metas = [dome_meta, sky_meta, twilight_meta]

    groups = group_flats_by_filter(metas, separate_dome_and_sky=True)

    assert len(groups) == 2
    assert "R-Bessell:dome" in groups
    assert "R-Bessell:sky" in groups

    assert len(groups["R-Bessell:dome"]) == 1
    assert groups["R-Bessell:dome"][0].path == dome_path

    assert len(groups["R-Bessell:sky"]) == 2
    assert groups["R-Bessell:sky"][0].path == sky_path
    assert groups["R-Bessell:sky"][1].path == twilight_path


def test_group_flats_by_filter_merge_dome_and_sky(tmp_path):
    """Test grouping with separate_dome_and_sky=False."""
    h, w = 16, 16
    shape = (h, w)

    # Create dome and sky flats for the same filter
    dome_path = tmp_path / "dome.fits"
    _write_fits(dome_path, np.full(shape, 10000.0))
    dome_meta = _meta(dome_path, shape, filter="R-Bessell", object="DomeFlat")

    sky_path = tmp_path / "sky.fits"
    _write_fits(sky_path, np.full(shape, 10000.0))
    sky_meta = _meta(sky_path, shape, filter="R-Bessell", object="SkyFlat")

    metas = [dome_meta, sky_meta]

    groups = group_flats_by_filter(metas, separate_dome_and_sky=False)

    assert len(groups) == 1
    assert "R-Bessell" in groups
    assert len(groups["R-Bessell"]) == 2


def test_empty_flat_selection_raises(tmp_path):
    """Test that selecting from an empty list raises CalibrationError."""
    profile = _profile()
    settings = _settings()
    outdir = tmp_path / "masters"

    with pytest.raises(CalibrationError, match="no flat frames found"):
        build_master_flats([], settings=settings, profile=profile, outdir=outdir)


def test_mask_applied_and_fatal_pixels_set_to_one(tmp_path):
    """Test that mask is applied and fatal pixels are set to 1.0."""
    rng = np.random.default_rng(46)
    h, w = 32, 32
    shape = (h, w)
    profile = _profile()
    settings = _settings()

    # Create flat frames with valid level
    metas = _make_flat_group(
        tmp_path,
        "flat",
        n=3,
        shape=shape,
        level=20000.0,  # Within [12000, 42000]
        sigma=100.0,
        filter="R-Bessell",
        rng=rng,
    )

    # Create a mask with a known dead pixel flagged
    from robo43.calib.masks import MaskSet

    dq = np.zeros(shape, dtype=np.int32)
    dq[10, 10] = int(DQ.DEAD)  # Flag one pixel as dead
    mask = MaskSet(dq=dq, summary={"any": 1.0 / (h * w), "fatal": 1.0 / (h * w)})

    outdir = tmp_path / "masters"
    masters = build_master_flats(
        metas, settings=settings, profile=profile, outdir=outdir, mask=mask
    )

    master = masters["R-Bessell:dome"]

    # Check that the dead pixel is set to 1.0
    assert master.data[10, 10] == 1.0

    # Check that mask is present in the MasterFrame
    assert master.mask is not None

    # Check that the mask HDU is in the written FITS file
    written_path = outdir / "master_flat_R-Bessell_dome.fits"
    with fits.open(written_path) as hdul:
        assert "DQ" in hdul
        assert hdul[0].header["BUNIT"] == "1"


def test_normalise_flat_computes_correct_level():
    """Test that normalise_flat computes the median correctly."""
    # Create a flat with a known level
    h, w = 32, 32
    data = np.full((h, w), 100.0, dtype=np.float32)

    settings = _settings()
    norm_data, cards = normalise_flat(data, settings.calib)

    # After normalization, the data should be all 1.0
    assert np.allclose(norm_data, 1.0)

    # Check the cards
    card_dict = {c[0]: c[1] for c in cards}
    assert card_dict["FLATNORM"] == 100.0
    assert card_dict["FLATNREG"] == 0.5


def test_normalise_flat_raises_on_bad_level():
    """Test that normalise_flat raises on non-finite or <= 0 level."""
    settings = _settings()

    # All NaN
    data_nan = np.full((32, 32), np.nan, dtype=np.float32)
    with pytest.raises(CalibrationError):
        normalise_flat(data_nan, settings.calib)

    # All zero
    data_zero = np.zeros((32, 32), dtype=np.float32)
    with pytest.raises(CalibrationError):
        normalise_flat(data_zero, settings.calib)


def test_master_flat_bunit_is_one(tmp_path):
    """Test that the written master flat has bunit='1'."""
    rng = np.random.default_rng(47)
    h, w = 32, 32
    shape = (h, w)

    metas = _make_flat_group(
        tmp_path,
        "flat",
        n=3,
        shape=shape,
        level=20000.0,  # Within [12000, 42000]
        sigma=100.0,
        filter="R-Bessell",
        rng=rng,
    )

    profile = _profile()
    settings = _settings()
    outdir = tmp_path / "masters"

    masters = build_master_flats(metas, settings=settings, profile=profile, outdir=outdir)
    master = masters["R-Bessell:dome"]

    assert master.bunit == "1"

    # Check the written file
    written_path = outdir / "master_flat_R-Bessell_dome.fits"
    with fits.open(written_path) as hdul:
        assert hdul[0].header["BUNIT"] == "1"


def test_build_master_flats_with_overscan_geometry(tmp_path):
    """Test build_master_flats with OverscanGeometry parameter.

    Verifies that:
    - When overscan is passed, each master's shape is trimmed to science region
    - Each master header contains TRIMMED=True
    - When overscan is not passed, masters keep the untrimmed shape
    """
    rng = np.random.default_rng(502)
    # Raw detector: 36 rows x 38 columns
    # Overscan: 4 columns at indices 34-37 (raw 0-based) at bias level
    # Science region: columns 0-33 (trimmed to 34 columns) at illuminated level
    h, w = 36, 38
    bias_level = 500.0
    illuminated_level = 20000.0  # Within [12000, 42000]
    sigma = 100.0

    # Create synthetic flat frames with illuminated science region and bias-level overscan
    science_data = np.full((h, 34), illuminated_level, dtype=np.float32)
    overscan_cols = np.full((h, 4), bias_level, dtype=np.float32)

    n_frames = 3
    metas = []
    for i in range(n_frames):
        # Add noise
        science_frame = science_data + rng.normal(0.0, sigma, size=(h, 34))
        overscan_frame = overscan_cols + rng.normal(0.0, sigma, size=(h, 4))
        full_frame = np.hstack([science_frame, overscan_frame])

        path = tmp_path / f"flat_{i:02d}.fits"
        _write_fits(path, full_frame)
        metas.append(_meta(path, (h, w), filter="R-Bessell", object="DomeFlat"))

    # Build with OverscanGeometry specifying the 4 overscan columns
    geom = OverscanGeometry(
        shape=(h, w),
        columns=(34, 35, 36, 37),
        rows=(),
        bias_level=bias_level,
        flat_level=illuminated_level,
        detected=True,
        source="test",
    )

    profile = _profile()
    settings = _settings()
    outdir_with = tmp_path / "with_overscan"

    # Build with overscan parameter
    masters_with = build_master_flats(
        metas, settings=settings, profile=profile, outdir=outdir_with, overscan=geom
    )

    # Check that we got a master
    assert "R-Bessell:dome" in masters_with
    master_with = masters_with["R-Bessell:dome"]

    # Check shape is trimmed (columns only removed, rows unchanged)
    assert master_with.data.shape == (h, 34), \
        f"Master shape {master_with.data.shape} != trimmed shape (h={h}, w=34)"

    # Check header cards
    header_cards = {c[0]: c[1] for c in master_with.header_cards}
    assert header_cards.get("TRIMMED") is True, "TRIMMED should be True in header"

    # Verify the file was written
    written_path = outdir_with / "master_flat_R-Bessell_dome.fits"
    assert written_path.exists(), f"Master flat not written to {written_path}"

    # Check file header
    with fits.open(written_path) as hdul:
        assert hdul[0].header.get("TRIMMED") is True

    # Build without overscan parameter to verify default behavior is unchanged
    outdir_without = tmp_path / "without_overscan"
    masters_without = build_master_flats(
        metas, settings=settings, profile=profile, outdir=outdir_without
    )

    # Check that default behavior still returns untrimmed shape
    assert "R-Bessell:dome" in masters_without
    master_without = masters_without["R-Bessell:dome"]
    assert master_without.data.shape == (h, w), \
        f"Master without overscan shape {master_without.data.shape} should be untrimmed ({h}, {w})"


def test_level_cut_skipped_when_saturation_adu_is_none(tmp_path):
    """Test that level cut is skipped when saturation_adu is None.

    When a profile declares no saturation level (saturation_adu=None),
    the level cut should be skipped entirely, all frames accepted,
    FLATNACC should equal the frame count, FLATNREJ should be 0,
    and FLATLCUT header card should be False.
    """
    rng = np.random.default_rng(48)
    h, w = 32, 32
    shape = (h, w)

    # Create flat frames with varying levels that would normally be rejected
    metas = []

    # Frame at a low level (would be rejected if saturation was enforced)
    low_frame = 8000.0 + rng.normal(0.0, 50.0, size=shape)
    low_path = tmp_path / "low.fits"
    _write_fits(low_path, low_frame)
    metas.append(_meta(low_path, shape, filter="R-Bessell", object="DomeFlat"))

    # Frame at a high level (would be rejected if saturation was enforced)
    high_frame = 50000.0 + rng.normal(0.0, 50.0, size=shape)
    high_path = tmp_path / "high.fits"
    _write_fits(high_path, high_frame)
    metas.append(_meta(high_path, shape, filter="R-Bessell", object="DomeFlat"))

    # Frame at a normal level
    normal_frame = 20000.0 + rng.normal(0.0, 50.0, size=shape)
    normal_path = tmp_path / "normal.fits"
    _write_fits(normal_path, normal_frame)
    metas.append(_meta(normal_path, shape, filter="R-Bessell", object="DomeFlat"))

    # Create profile with saturation_adu=None
    profile = _profile(saturation_adu=None)
    settings = _settings()
    outdir = tmp_path / "masters"

    # Build master flats -- should NOT raise even though frames are at extreme levels
    masters = build_master_flats(metas, settings=settings, profile=profile, outdir=outdir)

    master = masters["R-Bessell:dome"]

    # Check that all frames were accepted
    card_dict = {c[0]: c[1] for c in master.header_cards}
    assert card_dict["FLATNACC"] == 3, \
        f"Expected FLATNACC=3 but got {card_dict['FLATNACC']}"
    assert card_dict["FLATNREJ"] == 0, \
        f"Expected FLATNREJ=0 but got {card_dict['FLATNREJ']}"

    # Check that FLATLCUT is False
    assert card_dict["FLATLCUT"] is False, \
        f"Expected FLATLCUT=False but got {card_dict['FLATLCUT']}"
