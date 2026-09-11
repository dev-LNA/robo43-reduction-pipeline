"""Tests for robo43.calib.combine.

Builds its own in-memory numpy stacks; does not use tests/conftest.py.
"""

from __future__ import annotations

import itertools
import math
import warnings

import numpy as np
import pytest
from astropy.stats import sigma_clip as _sc

from robo43.calib.combine import (
    CombineResult,
    combine_lazy,
    combine_stack,
    row_block_size,
    sigma_clip_stack,
)
from robo43.config import CalibSettings
from robo43.exceptions import CalibrationError


def _settings(**overrides) -> CalibSettings:
    return CalibSettings(**overrides)


def test_bug8_contrast_gradient_and_hot_column():
    rng = np.random.default_rng(42)
    n = 20
    h, w = 40, 50
    noise_sigma = 5.0

    x = np.arange(w)
    gradient = 1000 + 5 * x
    truth = np.broadcast_to(gradient, (h, w)).astype(np.float64).copy()
    truth[:, 10] += 2000.0  # hot column

    frames = []
    for _i in range(n):
        frame = truth + rng.normal(0.0, noise_sigma, size=(h, w))
        frames.append(frame)

    # inject a cosmic ray in one frame only, away from the hot column
    frames[3] = frames[3].copy()
    frames[3][19:22, 29:32] += 500.0

    settings = _settings(combine_method="mean")
    result = combine_stack(frames, settings)

    rms_corrected = float(np.sqrt(np.mean((result.data - truth) ** 2)))
    assert rms_corrected < 3.0

    # legacy buggy order: collapse first, then clip spatially
    legacy_master = np.median(np.stack(frames), axis=0)
    legacy_clipped = _sc(
        legacy_master, sigma=5, maxiters=5, cenfunc="median", stdfunc="mad_std", masked=True
    )
    legacy_filled = np.ma.filled(legacy_clipped, np.nan)

    truth_col10 = truth[:, 10]
    legacy_col10 = legacy_filled[:, 10]
    corrected_col10 = result.data[:, 10]

    legacy_valid = np.isfinite(legacy_col10)
    if legacy_valid.any():
        legacy_rms_col10 = float(
            np.sqrt(np.mean((legacy_col10[legacy_valid] - truth_col10[legacy_valid]) ** 2))
        )
    else:
        legacy_rms_col10 = float("inf")
    corrected_rms_col10 = float(np.sqrt(np.mean((corrected_col10 - truth_col10) ** 2)))

    # either legacy masked out the hot column entirely, or its error there
    # is dramatically larger than the corrected combine's
    assert (not legacy_valid.all()) or (legacy_rms_col10 > 3 * corrected_rms_col10)

    # outlier rejected at the cosmic-ray pixels
    assert result.n_used[20, 30] == n - 1
    expected_se = noise_sigma / math.sqrt(n - 1)
    assert abs(result.data[20, 30] - truth[20, 30]) < 5 * expected_se


def test_recovery_of_constant_level_standard_error():
    rng = np.random.default_rng(7)
    n = 20
    h, w = 30, 30
    c = 1000.0
    sigma = 8.0

    frames = [c + rng.normal(0.0, sigma, size=(h, w)) for _ in range(n)]
    settings = _settings()
    result = combine_stack(frames, settings)

    se = sigma / math.sqrt(n)
    assert abs(float(result.data.mean()) - c) < 5 * se


def test_outlier_rejection_isolated():
    rng = np.random.default_rng(11)
    n = 20
    h, w = 25, 25
    level = 500.0
    sigma = 5.0

    frames = [level + rng.normal(0.0, sigma, size=(h, w)) for _ in range(n)]
    frames[0] = frames[0].copy()
    frames[0][12, 12] += 50.0  # 10-sigma spike

    settings = _settings()
    result = combine_stack(frames, settings)

    assert result.n_used[12, 12] == n - 1
    expected_se = sigma / math.sqrt(n - 1)
    assert abs(result.data[12, 12] - level) < 5 * expected_se


def test_zero_variance_pixel_no_warnings():
    rng = np.random.default_rng(13)
    n = 20
    h, w = 20, 20
    frames = [500.0 + rng.normal(0.0, 5.0, size=(h, w)) for _ in range(n)]
    for f in frames:
        f[5, 5] = 500.0  # identical every frame -> zero variance

    settings = _settings()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = combine_stack(
            frames, settings, read_noise_e=7.0, gain_e_per_adu=2.0
        )

    assert np.isfinite(result.variance[5, 5])
    assert result.variance[5, 5] > 0
    assert np.isfinite(result.weight[5, 5])

    assert np.isfinite(result.variance).all()
    assert np.isfinite(result.weight).all()
    assert np.isfinite(result.data).all()


def test_fully_clipped_pixel_produces_no_nan():
    rng = np.random.default_rng(2)
    n = 20
    h, w = 60, 60
    sigma = 5.0

    frames = [500.0 + rng.normal(0.0, sigma, size=(h, w)) for _ in range(n)]
    settings = _settings(sigma_clip_low=0.5, sigma_clip_high=0.5, sigma_clip_maxiters=1)

    result = combine_stack(frames, settings)

    assert np.isfinite(result.data).all()
    assert np.isfinite(result.variance).all()
    assert np.isfinite(result.weight).all()

    zero_survivors = result.n_used == 0
    assert zero_survivors.any(), (
        "test did not exercise the fully-clipped branch; adjust seed/threshold"
    )
    assert result.mask[zero_survivors].all()
    assert (result.weight[zero_survivors] == 0).all()


def test_combine_lazy_matches_combine_stack_bit_for_bit():
    rng = np.random.default_rng(29)
    n = 12
    h, w = 37, 29
    frames = [500.0 + rng.normal(0.0, 5.0, size=(h, w)) for _ in range(n)]
    arrays = [np.asarray(f) for f in frames]

    settings = _settings(max_memory_mb=0.001)
    block = row_block_size((h, w), n, settings.max_memory_mb)
    assert block < h  # forces many chunks

    result_stack = combine_stack(frames, _settings())

    def _make_loader(arr):
        return lambda rc: arr[rc[0], rc[1]]

    loaders = [_make_loader(arr) for arr in arrays]
    result_lazy = combine_lazy(loaders, (h, w), settings)

    assert np.array_equal(result_lazy.data, result_stack.data)
    assert np.array_equal(result_lazy.variance, result_stack.variance)
    assert np.array_equal(result_lazy.weight, result_stack.weight)
    assert np.array_equal(result_lazy.mask, result_stack.mask)
    assert np.array_equal(result_lazy.n_used, result_stack.n_used)
    assert result_lazy.rejected_fraction == pytest.approx(result_stack.rejected_fraction)


def test_row_block_size_monotonic_and_bounded():
    shape = (2000, 2000)
    max_memory_mb = 64.0

    values = [row_block_size(shape, n, max_memory_mb) for n in (2, 10, 100, 1000)]
    for a, b in itertools.pairwise(values):
        assert b <= a
    for v in values:
        assert v >= 1

    huge = row_block_size(shape, 100000, max_memory_mb)
    assert huge == 1


def test_method_median_vs_mean():
    rng = np.random.default_rng(31)
    n = 20
    h, w = 15, 15
    level = 1000.0
    sigma = 6.0

    frames = [level + rng.normal(0.0, sigma, size=(h, w)) for _ in range(n)]

    result_mean = combine_stack(frames, _settings(combine_method="mean"))
    result_median = combine_stack(frames, _settings(combine_method="median"))

    se = sigma / math.sqrt(n)
    assert abs(float(result_mean.data.mean()) - level) < 5 * se
    assert abs(float(result_median.data.mean()) - level) < 5 * se

    full_n = np.logical_and(
        result_mean.n_used == n, result_median.n_used == n
    )
    assert full_n.any()

    mean_var = result_mean.variance[full_n]
    median_var = result_median.variance[full_n]
    ratio = float(np.mean(median_var / mean_var))
    assert ratio == pytest.approx(math.pi / 2.0, rel=0.5)


def test_error_paths():
    settings = _settings()

    with pytest.raises(CalibrationError):
        combine_stack([], settings)

    a = np.ones((5, 5))
    b = np.ones((6, 6))
    with pytest.raises(CalibrationError):
        combine_stack([a, b], settings)

    with pytest.raises(CalibrationError):
        combine_stack([np.ones(5), np.ones(5)], settings)

    with pytest.raises(CalibrationError):
        combine_stack([a, a], settings, method="bogus")

    with pytest.raises(CalibrationError):
        combine_stack([a], settings)

    with pytest.raises(CalibrationError):
        combine_lazy([], (5, 5), settings)

    def loader(rc):
        return np.ones((rc[0].stop - rc[0].start, rc[1].stop - rc[1].start))

    with pytest.raises(CalibrationError):
        combine_lazy([loader], (5, 5), settings)


def test_scales_bring_frames_to_common_level():
    rng = np.random.default_rng(41)
    n = 15
    h, w = 20, 20
    base_pattern = 1000 + rng.normal(0.0, 5.0, size=(h, w))

    factors = np.linspace(0.5, 2.0, n)
    frames = [base_pattern * f for f in factors]
    scales = 1.0 / factors

    settings = _settings()
    result = combine_stack(frames, settings, scales=scales)

    rms = float(np.sqrt(np.mean((result.data - base_pattern) ** 2)))
    assert rms < 5.0


def test_sigma_clip_stack_single_frame_passthrough():
    stack = np.ones((1, 4, 4))
    settings = _settings()
    clipped = sigma_clip_stack(stack, settings)
    assert not np.ma.getmaskarray(clipped).any()


def test_combine_result_header_cards():
    rng = np.random.default_rng(53)
    n = 16
    frames = [500.0 + rng.normal(0.0, 5.0, size=(10, 10)) for _ in range(n)]
    settings = _settings()
    result = combine_stack(frames, settings)
    assert isinstance(result, CombineResult)

    cards = result.header_cards()
    keys = [c[0] for c in cards]
    assert keys == [
        "NCOMBINE",
        "COMBMETH",
        "CLIPLO",
        "CLIPHI",
        "CLIPITER",
        "REJFRAC",
        "NUSEDMIN",
        "NUSEDMAX",
    ]


def _stack_with_perfectly_stable_pixel(n=20, shape=(32, 24), seed=11):
    """A stack whose pixel (5, 5) is bit-identical in every frame."""
    rng = np.random.default_rng(seed)
    truth = np.full(shape, 100.0)
    frames = [truth + rng.normal(0.0, 5.0, shape) for _ in range(n)]
    for frame in frames:
        frame[5, 5] = 500.0
    return frames


def test_read_noise_variance_is_a_floor_not_a_substitute():
    """A stable pixel must not receive a runaway inverse-variance weight.

    The sample variance of a perfectly stable pixel is zero in exact
    arithmetic but comes out as float rounding noise (~1e-28) in practice, so
    a guard that only replaces variances ``<= 0`` lets it through and hands
    that pixel a weight of ~1e28.  One such pixel would dominate every
    inverse-variance-weighted combination downstream, which is precisely the
    class of error Appendix A bug 28 is about.
    """
    frames = _stack_with_perfectly_stable_pixel()
    result = combine_stack(
        frames, _settings(), read_noise_e=7.0, gain_e_per_adu=10.0
    )

    expected_floor = (7.0 / 10.0) ** 2
    assert result.variance[5, 5] == pytest.approx(expected_floor, rel=1e-6)
    assert result.weight[5, 5] == pytest.approx(1.0 / expected_floor, rel=1e-6)

    # No pixel may dominate: the largest weight stays within an order of
    # magnitude of the typical one.
    usable = result.weight > 0
    typical = float(np.median(result.weight[usable]))
    assert result.weight.max() < 10.0 * typical


def test_unmeasurable_variance_is_masked_when_no_read_noise():
    """With no read noise there is no floor to impose, so mask instead.

    A variance far below the frame's own noise level cannot be measured from a
    handful of frames; inventing a value would be dishonest, and believing it
    reproduces the runaway weight.  Such pixels are marked unusable.
    """
    frames = _stack_with_perfectly_stable_pixel()
    result = combine_stack(frames, _settings())

    assert result.mask[5, 5]
    assert result.weight[5, 5] == 0.0
    usable = result.weight > 0
    assert result.weight.max() < 10.0 * float(np.median(result.weight[usable]))
    assert np.isfinite(result.variance).all()
    assert np.isfinite(result.weight).all()
