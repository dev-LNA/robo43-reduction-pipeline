"""Tests for robo43.calib.overscan.

Builds synthetic FITS files and FrameMeta objects; does not use conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from robo43.calib.overscan import OverscanGeometry, apply_overscan, detect_overscan
from robo43.config import CalibSettings, RuntimeSettings, Settings
from robo43.exceptions import CalibrationError
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


# ---------------------------------------------------------------------------
# Multi-block (interior overscan) tests: OverscanGeometry constructed
# directly, matching a multi-amplifier mosaic (e.g. T80S T80Cam) where
# overscan strips sit inside the array and split the retained region on one
# or both axes into more than one contiguous block.
# ---------------------------------------------------------------------------


def test_science_blocks_single_region_matches_science_slice():
    """Single-block case: science_blocks has exactly one block per axis and
    science_slice keeps returning that same span (unchanged behaviour)."""
    h, w = 100, 128
    geom = OverscanGeometry(
        shape=(h, w),
        columns=tuple(range(w - 8, w)),
        rows=(),
        bias_level=1000.0,
        flat_level=24000.0,
        detected=True,
        source="detected",
    )
    row_blocks, col_blocks = geom.science_blocks
    assert row_blocks == [(0, h)]
    assert col_blocks == [(0, w - 8)]

    row_slice, col_slice = geom.science_slice
    assert (row_slice, col_slice) == (slice(0, h), slice(0, w - 8))
    assert geom.science_tile_slices == [(slice(0, h), slice(0, w - 8))]


def test_science_tile_slices_multi_block_is_row_major_stitched_space():
    """2 row blocks x 4 col blocks (see the stitching test below): tile
    slices are in row-major order, in the *stitched* (post apply_overscan)
    coordinate space -- cumulative block sizes, not the raw block offsets."""
    ny, nx = 12, 22
    geom = OverscanGeometry(
        shape=(ny, nx),
        columns=tuple(sorted([0, 1, 5, 6, 10, 11, 15, 16, 20, 21])),
        rows=(5, 6),
        bias_level=1000.0,
        flat_level=24000.0,
        detected=True,
        source="detected",
    )
    row_blocks, col_blocks = geom.science_blocks
    assert row_blocks == [(0, 5), (7, 12)]
    assert col_blocks == [(2, 5), (7, 10), (12, 15), (17, 20)]

    tiles = geom.science_tile_slices
    assert len(tiles) == 8
    expected = [
        (slice(r * 5, (r + 1) * 5), slice(c * 3, (c + 1) * 3))
        for r in range(2)
        for c in range(4)
    ]
    assert tiles == expected


def test_science_tile_slices_empty_when_no_science_region():
    """No retained science block on either axis: no tiles, matching
    science_blocks' own empty-list behaviour."""
    geom = OverscanGeometry(
        shape=(0, 0),
        columns=(),
        rows=(),
        bias_level=float("nan"),
        flat_level=float("nan"),
        detected=False,
        source="none",
    )
    assert geom.science_tile_slices == []


def test_apply_overscan_single_block_path_unchanged(tmp_path):
    """Direct-construction regression: single-block apply_overscan output and
    header cards are byte-for-byte what the pre-tiling implementation gave."""
    h, w = 100, 128
    bias_level = 1000.0
    frame_level = 2000.0

    frame_data = np.full((h, w), frame_level, dtype=np.float32)
    frame_data[:, -8:] = bias_level + 50.0

    geom = OverscanGeometry(
        shape=(h, w),
        columns=tuple(range(w - 8, w)),
        rows=(),
        bias_level=bias_level,
        flat_level=24000.0,
        detected=True,
        source="detected",
    )
    settings = _settings()

    corrected, cards = apply_overscan(frame_data, geom, settings=settings)
    card_dict = {c[0]: c[1] for c in cards}

    assert corrected.shape == (h, w - 8)
    assert np.allclose(corrected, frame_level - (bias_level + 50.0))
    assert card_dict["OSCANCOR"] is True
    assert card_dict["TRIMMED"] is True
    assert card_dict["TRIMREG"] == f"[1:{w - 8},1:{h}]"
    assert "TRIMNSEG" not in card_dict
    assert "OSCANLMN" not in card_dict
    assert "OSCANLMX" not in card_dict


def test_sliver_block_dropped_from_science_blocks(caplog):
    """A one-pixel retained sliver (<5% of the longest retained block) is
    dropped, leaving a single science block, and is logged as a WARNING."""
    h, w = 5, 30
    # Overscan columns 24-28 leave retained blocks 0-23 (24 px) and 29 (1 px).
    # 1 px < 5% of 24 px (1.2 px), so the 1 px block must be dropped.
    geom = OverscanGeometry(
        shape=(h, w),
        columns=tuple(range(24, 29)),
        rows=(),
        bias_level=1000.0,
        flat_level=24000.0,
        detected=True,
        source="detected",
    )

    with caplog.at_level("WARNING"):
        row_blocks, col_blocks = geom.science_blocks

    assert row_blocks == [(0, h)]
    assert col_blocks == [(0, 24)]
    assert any(
        "spurious retained block" in record.message for record in caplog.records
    )


def test_apply_overscan_stitches_2x4_tiles_with_distinct_levels():
    """2 row blocks x 4 col blocks, each with its own bias level from its
    adjacent overscan-column run(s): stitched shape, tile order, and
    per-tile subtraction must all be exact."""
    ny, nx = 12, 22
    row_block_defs = [(0, 5), (7, 12)]
    col_block_defs = [(2, 5), (7, 10), (12, 15), (17, 20)]

    def level(r: int, c: int) -> float:
        return 1000.0 + 10 * r + c

    def signal(r: int, c: int) -> float:
        return 500.0 + 100 * r + 10 * c

    data = np.full((ny, nx), -99999.0)
    for r, row_block in enumerate(row_block_defs):
        row_slice = slice(*row_block)
        for c, col_block in enumerate(col_block_defs):
            col_slice = slice(*col_block)
            data[row_slice, col_slice] = signal(r, c) + level(r, c)
        # Overscan-column assignment: edge runs go whole to the block they
        # touch; each interior run splits in half between its neighbours.
        data[row_slice, [0, 1, 5]] = level(r, 0)
        data[row_slice, [6, 10]] = level(r, 1)
        data[row_slice, [11, 15]] = level(r, 2)
        data[row_slice, [16, 20, 21]] = level(r, 3)

    geom = OverscanGeometry(
        shape=(ny, nx),
        columns=tuple(sorted([0, 1, 5, 6, 10, 11, 15, 16, 20, 21])),
        rows=(5, 6),
        bias_level=1000.0,
        flat_level=24000.0,
        detected=True,
        source="detected",
    )
    row_blocks, col_blocks = geom.science_blocks
    assert row_blocks == row_block_defs
    assert col_blocks == col_block_defs

    settings = _settings()
    corrected, cards = apply_overscan(data, geom, settings=settings)
    card_dict = {c[0]: c[1] for c in cards}

    assert corrected.shape == (10, 12)
    expected = np.zeros((10, 12))
    for r in range(2):
        for c in range(4):
            expected[r * 5:(r + 1) * 5, c * 3:(c + 1) * 3] = signal(r, c)
    assert np.allclose(corrected, expected)

    assert card_dict["OSCANCOR"] is True
    assert card_dict["TRIMMED"] is True
    assert card_dict["TRIMNSEG"] == 8
    assert card_dict["OSCANLMN"] == 1000.0
    assert card_dict["OSCANLMX"] == 1013.0
    assert card_dict["OSCANLVL"] == pytest.approx(1006.5)
    assert card_dict["TRIMREG"] == "[3:20,1:12]"

    with pytest.raises(CalibrationError):
        _ = geom.science_slice


def test_apply_overscan_no_overscan_columns_uses_row_runs():
    """No overscan columns at all: multi-block row splitting falls back to
    adjacent row runs restricted to each tile's column range."""
    ny, nx = 12, 5
    signal0, signal1 = 500.0, 600.0
    level0, level1 = 1000.0, 1010.0

    data = np.full((ny, nx), -99999.0)
    data[0:5, :] = signal0 + level0
    data[7:12, :] = signal1 + level1
    data[5, :] = level0  # left half of the shared row-overscan run
    data[6, :] = level1  # right half of the shared row-overscan run

    geom = OverscanGeometry(
        shape=(ny, nx),
        columns=(),
        rows=(5, 6),
        bias_level=1000.0,
        flat_level=24000.0,
        detected=True,
        source="detected",
    )
    row_blocks, col_blocks = geom.science_blocks
    assert row_blocks == [(0, 5), (7, 12)]
    assert col_blocks == [(0, 5)]

    settings = _settings()
    corrected, cards = apply_overscan(data, geom, settings=settings)
    card_dict = {c[0]: c[1] for c in cards}

    assert corrected.shape == (10, 5)
    assert np.allclose(corrected[0:5, :], signal0)
    assert np.allclose(corrected[5:10, :], signal1)
    assert card_dict["TRIMREG"] == "[1:5,1:12]"
    assert card_dict["TRIMNSEG"] == 2


def test_apply_overscan_tiled_non_finite_tile_skips_whole_frame():
    """A non-finite per-tile level (e.g. NaN pixels in the source overscan
    columns) skips correction and trimming for the whole frame, the same
    fallback the single-block path uses."""
    ny, nx = 12, 22
    row_block_defs = [(0, 5), (7, 12)]
    col_block_defs = [(2, 5), (7, 10), (12, 15), (17, 20)]

    def level(r: int, c: int) -> float:
        return 1000.0 + 10 * r + c

    def signal(r: int, c: int) -> float:
        return 500.0 + 100 * r + 10 * c

    data = np.full((ny, nx), -99999.0)
    for r, row_block in enumerate(row_block_defs):
        row_slice = slice(*row_block)
        for c, col_block in enumerate(col_block_defs):
            col_slice = slice(*col_block)
            data[row_slice, col_slice] = signal(r, c) + level(r, c)
        data[row_slice, [0, 1, 5]] = level(r, 0)
        data[row_slice, [6, 10]] = level(r, 1)
        data[row_slice, [11, 15]] = level(r, 2)
        data[row_slice, [16, 20, 21]] = level(r, 3)

    # Break the overscan source for row-block 0 / col-block 0's tile.
    data[0:5, [0, 1, 5]] = np.nan

    geom = OverscanGeometry(
        shape=(ny, nx),
        columns=tuple(sorted([0, 1, 5, 6, 10, 11, 15, 16, 20, 21])),
        rows=(5, 6),
        bias_level=1000.0,
        flat_level=24000.0,
        detected=True,
        source="detected",
    )
    settings = _settings()
    corrected, cards = apply_overscan(data, geom, settings=settings)
    card_dict = {c[0]: c[1] for c in cards}

    assert corrected.shape == data.shape
    assert card_dict["OSCANCOR"] is False
    assert card_dict["TRIMMED"] is False
