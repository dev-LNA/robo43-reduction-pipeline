"""Tests for robo43.calib.masks.

Builds its own in-memory numpy fixtures; does not use tests/conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from robo43.calib.combine import CombineResult, combine_stack
from robo43.calib.masks import (
    MaskSet,
    build_mask,
    detect_bad_columns,
    minimal_mask,
    nonlinearity_mask,
    saturation_mask,
)
from robo43.config import CalibSettings
from robo43.exceptions import CalibrationError
from robo43.products.dq import DQ


def _settings(**overrides) -> CalibSettings:
    return CalibSettings(**overrides)


def test_hot_pixels():
    rng = np.random.default_rng(1)
    dark = rng.normal(loc=1.0, scale=0.05, size=(50, 50))
    coords = [(5, 5), (10, 20), (30, 30), (45, 2), (0, 49)]
    sigma = np.median(np.abs(dark - np.median(dark))) * 1.4826
    hot_value = np.median(dark) + 50 * sigma
    for y, x in coords:
        dark[y, x] = hot_value

    settings = _settings(hot_pixel_nmad=6.0)
    result = build_mask(dark_current_map=dark, settings=settings)

    for y, x in coords:
        assert int(result.dq[y, x]) & int(DQ.HOT)

    hot_count = int(np.count_nonzero(result.dq & int(DQ.HOT)))
    assert hot_count == 5


def test_dead_pixels_and_vignetting():
    h, w = 60, 60
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / r.max()
    flat = 1.0 - 0.4 * r_norm  # bottoms out at 0.6 in the corners

    flat[10:15, 10:15] = 0.1  # genuine dead block

    settings = _settings()
    result = build_mask(master_flat=flat, settings=settings)

    assert np.all(result.dq[10:15, 10:15] & int(DQ.DEAD))

    corner_coords = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    for y, x in corner_coords:
        assert not (int(result.dq[y, x]) & int(DQ.DEAD))


def test_normalisation_invariance():
    h, w = 60, 60
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / r.max()
    flat = 1.0 - 0.4 * r_norm
    flat[10:15, 10:15] = 0.1

    settings = _settings()
    result_norm = build_mask(master_flat=flat, settings=settings)
    result_raw = build_mask(master_flat=flat * 30000, settings=settings)

    assert np.array_equal(result_norm.dq, result_raw.dq)


def test_unstable_pixels():
    rng = np.random.default_rng(2)
    variance = rng.normal(loc=10.0, scale=0.5, size=(40, 40))
    coords = [(3, 3), (20, 20), (39, 0)]
    sigma = np.median(np.abs(variance - np.median(variance))) * 1.4826
    unstable_value = np.median(variance) + 50 * sigma
    for y, x in coords:
        variance[y, x] = unstable_value

    settings = _settings(unstable_pixel_nmad=6.0)
    result = build_mask(bias_variance=variance, settings=settings)

    unstable_flagged = {
        (y, x)
        for y in range(40)
        for x in range(40)
        if int(result.dq[y, x]) & int(DQ.UNSTABLE)
    }
    assert unstable_flagged == set(coords)

    # CombineResult duck-typing
    stack = rng.normal(loc=100.0, scale=5.0, size=(5, 40, 40))
    combine_result = combine_stack(list(stack), settings)
    modified_variance = combine_result.variance.copy()
    for y, x in coords:
        modified_variance[y, x] = unstable_value
    combine_result_modified = CombineResult(
        data=combine_result.data,
        variance=modified_variance,
        weight=combine_result.weight,
        mask=combine_result.mask,
        n_used=combine_result.n_used,
        n_input=combine_result.n_input,
        method=combine_result.method,
        rejected_fraction=combine_result.rejected_fraction,
        sigma_low=combine_result.sigma_low,
        sigma_high=combine_result.sigma_high,
        maxiters=combine_result.maxiters,
    )
    result_from_combine = build_mask(bias_variance=combine_result_modified, settings=settings)
    result_from_array = build_mask(bias_variance=modified_variance, settings=settings)
    assert np.array_equal(result_from_combine.dq, result_from_array.dq)


def test_bad_column_and_row_non_square():
    rng = np.random.default_rng(3)
    image = rng.normal(loc=100.0, scale=1.0, size=(40, 60))
    bad_x = 25
    image[:, bad_x] += 500.0

    settings = _settings()
    mask = detect_bad_columns(image, settings, axis=0)
    assert np.all(mask[:, bad_x])
    assert not np.any(np.all(mask, axis=1))

    image_row = rng.normal(loc=100.0, scale=1.0, size=(40, 60))
    bad_y = 12
    image_row[bad_y, :] += 500.0
    mask_row = detect_bad_columns(image_row, settings, axis=1)
    assert np.all(mask_row[bad_y, :])
    assert not np.any(np.all(mask_row, axis=0))


def test_overlapping_defects():
    rng = np.random.default_rng(4)
    dark = rng.normal(loc=1.0, scale=0.05, size=(40, 40))
    sigma = np.median(np.abs(dark - np.median(dark))) * 1.4826
    hot_value = np.median(dark) + 50 * sigma

    bad_x = 15
    hot_y = 20
    dark[:, bad_x] += 50 * sigma  # offset the whole column enough to trip bad-column detection
    dark[hot_y, bad_x] = hot_value + 50 * sigma  # ensure this pixel is also hot

    settings = _settings(hot_pixel_nmad=6.0, unstable_pixel_nmad=6.0)
    result = build_mask(dark_current_map=dark, settings=settings)

    value = int(result.dq[hot_y, bad_x])
    assert value & int(DQ.HOT)
    assert value & int(DQ.BAD_COLUMN)


def test_build_mask_single_inputs():
    rng = np.random.default_rng(5)
    dark = rng.normal(loc=1.0, scale=0.05, size=(30, 30))
    flat = np.ones((30, 30))
    variance = rng.normal(loc=10.0, scale=0.5, size=(30, 30))

    settings = _settings()

    dark_only = build_mask(dark_current_map=dark, settings=settings)
    assert "hot" in dark_only.summary
    assert "dead" not in dark_only.summary
    assert "bad_column" in dark_only.summary
    assert "unstable" not in dark_only.summary

    flat_only = build_mask(master_flat=flat, settings=settings)
    assert "dead" in flat_only.summary
    assert "hot" not in flat_only.summary
    assert "bad_column" in flat_only.summary
    assert "unstable" not in flat_only.summary

    variance_only = build_mask(bias_variance=variance, settings=settings)
    assert "unstable" in variance_only.summary
    assert "hot" not in variance_only.summary
    assert "dead" not in variance_only.summary
    # Column detection DOES run on a bias-variance map: a persistent column
    # shows up there as excess variance just as readily as it shows up in a
    # dark or a flat.  It ran and legitimately found nothing, which the
    # summary reports as 0.0 -- distinct from the key being absent, which is
    # how "this detector did not run" is expressed.
    assert variance_only.summary["bad_column"] == 0.0


def test_build_mask_with_none():
    with pytest.raises(CalibrationError):
        build_mask(settings=CalibSettings())

    result = build_mask(settings=CalibSettings(), shape=(10, 10))
    assert np.array_equal(result.dq, np.zeros((10, 10), dtype=np.int32))
    assert result.summary == {"any": 0.0, "fatal": 0.0}


def test_mismatched_shapes():
    dark = np.ones((10, 10))
    flat = np.ones((10, 11))
    with pytest.raises(CalibrationError):
        build_mask(dark_current_map=dark, master_flat=flat, settings=CalibSettings())


def test_saturation_and_nonlinearity_boundary():
    data = np.array([9.0, 10.0, 10.5])
    sat = saturation_mask(data, 10.0)
    assert list(sat) == [False, True, True]

    nonlin = nonlinearity_mask(data, 10.0)
    assert list(nonlin) == [False, True, True]


def _build_full_mask() -> tuple[MaskSet, CalibSettings]:
    rng = np.random.default_rng(6)
    h, w = 50, 50
    dark = rng.normal(loc=1.0, scale=0.05, size=(h, w))
    sigma = np.median(np.abs(dark - np.median(dark))) * 1.4826
    hot_value = np.median(dark) + 50 * sigma
    dark[5, 5] = hot_value
    dark[:, 20] += 50 * sigma  # bad column

    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / r.max()
    flat = 1.0 - 0.4 * r_norm
    flat[30:35, 30:35] = 0.1

    variance = rng.normal(loc=10.0, scale=0.5, size=(h, w))
    var_sigma = np.median(np.abs(variance - np.median(variance))) * 1.4826
    variance[40, 40] = np.median(variance) + 50 * var_sigma

    settings = _settings(hot_pixel_nmad=6.0, unstable_pixel_nmad=6.0)
    result = build_mask(
        dark_current_map=dark, master_flat=flat, bias_variance=variance, settings=settings
    )
    assert result.summary["hot"] > 0
    assert result.summary["dead"] > 0
    assert result.summary["unstable"] > 0
    assert result.summary["bad_column"] > 0
    return result, settings


def test_round_trip(tmp_path):
    result, _settings_obj = _build_full_mask()
    path = tmp_path / "mask.fits"
    result.write(path)

    read_back = MaskSet.read(path)
    assert np.array_equal(result.dq, read_back.dq)

    for key in ("any", "fatal", "hot", "dead", "unstable", "bad_column"):
        assert read_back.summary[key] == pytest.approx(result.summary[key])
    for key in ("hot_nmad", "dead_frac", "unstable_nmad"):
        assert read_back.summary[key] == pytest.approx(result.summary[key])

    with fits.open(path) as hdul:
        names = [hdu.name for hdu in hdul]
        assert "DQ" in names
        assert "CATFLAGS" in names


def test_overwrite_protection(tmp_path):
    result, _settings_obj = _build_full_mask()
    path = tmp_path / "mask.fits"
    result.write(path)

    with pytest.raises(CalibrationError):
        result.write(path, overwrite=False)

    read_back = MaskSet.read(path)
    assert np.array_equal(result.dq, read_back.dq)


def test_text_summary():
    result, _settings_obj = _build_full_mask()
    text = result.text_summary()
    assert text
    expected = f"{result.summary['any'] * 100:.4f}"
    assert expected in text


def test_minimal_mask():
    data = np.array([1.0, np.nan, 5.0, 5.0])
    result = minimal_mask(data, saturation=5.0)
    assert int(result.dq[1]) & int(DQ.NO_DATA)
    assert int(result.dq[2]) & int(DQ.SATURATED)
    assert int(result.dq[3]) & int(DQ.SATURATED)
    assert not (int(result.dq[0]) & int(DQ.SATURATED))

    result_no_sat = minimal_mask(data)
    assert int(result_no_sat.dq[1]) & int(DQ.NO_DATA)
    assert "saturated" not in result_no_sat.summary
    assert int(result_no_sat.dq[2]) == 0


def test_no_header_card_exceeds_the_fits_80_column_limit():
    """Every card must fit in 80 columns or astropy silently truncates it.

    A truncated comment is a warning, not an error, so an over-long card ships
    quietly into archived products.  Appendix B of the plan adds dozens more
    keywords, so this invariant is asserted once here rather than discovered
    per keyword.
    """
    from astropy.io import fits

    mask, _ = _build_full_mask()
    over_long = [
        (key, str(fits.Card(key, value, comment)))
        for key, value, comment in mask.header_cards()
        if len(str(fits.Card(key, value, comment))) > 80
    ]
    assert not over_long, f"cards exceeding 80 columns: {over_long}"


def test_write_emits_no_verify_warnings(tmp_path):
    """Writing a mask must not trip an astropy VerifyWarning."""
    import warnings

    from astropy.io.fits.verify import VerifyWarning

    mask, _ = _build_full_mask()
    with warnings.catch_warnings():
        warnings.simplefilter("error", VerifyWarning)
        mask.write(tmp_path / "mask.fits")


def _vignetted_flat(h=40, w=60, level=32000.0):
    """A flat with a strong radial vignette and no defects."""
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2)
    return (1.0 - 0.45 * r / r.max()) * level


def test_bad_column_found_at_every_position_including_the_edges():
    """A defective column must be found wherever it sits, edges included.

    Two earlier implementations failed here.  A running median filter and a
    polynomial fit both have to extrapolate past the array ends: the filter
    flattened the baseline while the real vignette kept curving, which hid a
    dead column a few lines in from the edge, and the polynomial left its
    largest residuals at the extremes, which invented a column at index 0.
    Every index is therefore checked explicitly.
    """
    settings = _settings()
    h, w = 40, 60
    rng = np.random.default_rng(4)
    for x_bad in (0, 1, 5, w // 2, w - 2, w - 1):
        dark = (
            1.0
            + np.linspace(0, 0.3, h)[:, None]
            + np.zeros((h, w))
            + rng.normal(0, 0.02, (h, w))
        )
        dark[:, x_bad] += 1.5
        mask = build_mask(dark_current_map=dark, settings=settings)
        found = sorted(set(np.where((mask.dq & int(DQ.BAD_COLUMN)).any(axis=0))[0].tolist()))
        assert found == [x_bad], f"column at x={x_bad}: found {found}"


def test_smooth_structure_is_never_mistaken_for_a_bad_column():
    """Vignetting and thermal gradients are real structure, not defects."""
    settings = _settings()
    h, w = 40, 60

    flat = _vignetted_flat(h, w)
    mask = build_mask(master_flat=flat, settings=settings)
    assert int((mask.dq & int(DQ.BAD_COLUMN)).any(axis=0).sum()) == 0
    # ... and the vignetted corners are not "dead" either, despite sitting at
    # ~55% of the central level.
    assert not mask.dq[0, 0] & int(DQ.DEAD)
    assert not mask.dq[-1, -1] & int(DQ.DEAD)

    for seed in range(5):
        rng = np.random.default_rng(seed)
        dark = rng.normal(1.0, 0.02, (h, w)) + np.linspace(0, 0.3, h)[:, None]
        clean = build_mask(dark_current_map=dark, settings=settings)
        assert int((clean.dq & int(DQ.BAD_COLUMN)).any(axis=0).sum()) == 0, f"seed {seed}"
        assert int((clean.dq & int(DQ.HOT)).sum()) == 0, f"seed {seed}"


def test_bad_column_detection_runs_on_every_supplied_source():
    """A column may be visible in only one of the calibration products.

    Preferring one source and ignoring the others silently misses real
    defects, which is what the first implementation did.
    """
    settings = _settings()
    h, w = 40, 60
    rng = np.random.default_rng(5)

    dark = rng.normal(1.0, 0.02, (h, w))
    dark[:, 37] += 1.5                      # visible ONLY in the dark
    flat = _vignetted_flat(h, w)
    flat[:, 5] *= 0.2                       # visible ONLY in the flat

    mask = build_mask(dark_current_map=dark, master_flat=flat, settings=settings)
    found = sorted(set(np.where((mask.dq & int(DQ.BAD_COLUMN)).any(axis=0))[0].tolist()))
    assert found == [5, 37], found


def test_flat_normalisation_does_not_change_the_mask():
    """The same flat, normalised or in ADU, must give an identical mask."""
    settings = _settings()
    flat = _vignetted_flat()
    flat[15:18, 20:23] = 0.05 * 32000.0
    raw = build_mask(master_flat=flat, settings=settings)
    normalised = build_mask(master_flat=flat / np.median(flat), settings=settings)
    np.testing.assert_array_equal(raw.dq, normalised.dq)
