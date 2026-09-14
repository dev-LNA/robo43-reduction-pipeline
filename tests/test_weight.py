"""Tests for robo43.products.weight.

Builds its own in-memory numpy fixtures; does not use tests/conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from robo43.products.dq import DQ, DQ_DTYPE, FATAL_FLAGS
from robo43.products.mef import ReducedProduct
from robo43.products.weight import build_weight, weight_summary


def test_inverse_variance_value():
    err = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    dq = np.zeros(3, dtype=DQ_DTYPE)
    weight = build_weight(err, dq)
    assert_array_equal(weight, np.array([1.0, 0.25, 1.0 / 16.0], dtype=np.float32))


@pytest.mark.parametrize(
    "flag",
    [
        DQ.HOT,
        DQ.DEAD,
        DQ.UNSTABLE,
        DQ.SATURATED,
        DQ.COSMIC_RAY,
        DQ.NO_DATA,
        DQ.BAD_COLUMN,
    ],
)
def test_each_fatal_flag_zeros_weight(flag):
    err = np.array([1.0, 1.0], dtype=np.float32)
    dq = np.array([int(flag), 0], dtype=DQ_DTYPE)
    weight = build_weight(err, dq)
    assert weight[0] == 0.0
    assert weight[1] > 0.0


def test_non_fatal_flags_alone_keep_weight():
    err = np.array([1.0, 1.0], dtype=np.float32)
    for flag in (DQ.NONLINEAR, DQ.NO_FLAT):
        dq = np.array([int(flag), 0], dtype=DQ_DTYPE)
        weight = build_weight(err, dq)
        assert weight[0] > 0.0, f"{flag.name} incorrectly zeroed the weight"


def test_zero_and_negative_err_zeroed():
    err = np.array([0.0, -1.0, 1.0], dtype=np.float32)
    dq = np.zeros(3, dtype=DQ_DTYPE)
    weight = build_weight(err, dq)
    assert weight[0] == 0.0
    assert weight[1] == 0.0
    assert weight[2] > 0.0


def test_non_finite_err_zeroed():
    err = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)
    dq = np.zeros(4, dtype=DQ_DTYPE)
    weight = build_weight(err, dq)
    assert_array_equal(weight[:3], [0.0, 0.0, 0.0])
    assert weight[3] > 0.0


def test_no_nan_or_inf_in_output():
    rng = np.random.default_rng(0)
    err = rng.normal(size=100).astype(np.float32)  # includes negatives and near-zero
    err[5] = np.nan
    err[6] = np.inf
    err[7] = 1e-30  # extreme: would overflow float32 on naive 1/err**2
    dq = np.zeros(100, dtype=DQ_DTYPE)
    dq[10] = int(FATAL_FLAGS)
    weight = build_weight(err, dq)
    assert np.all(np.isfinite(weight))
    assert not np.any(np.isnan(weight))


def test_shape_mismatch_raises():
    err = np.ones((4, 4), dtype=np.float32)
    dq = np.zeros((4, 5), dtype=DQ_DTYPE)
    with pytest.raises(ValueError):
        build_weight(err, dq)


def test_inputs_not_mutated():
    err = np.array([1.0, 0.0, np.nan, -1.0], dtype=np.float32)
    dq = np.array([0, int(DQ.HOT), 0, 0], dtype=DQ_DTYPE)
    err_copy = err.copy()
    dq_copy = dq.copy()
    build_weight(err, dq)
    assert_array_equal(err, err_copy)
    assert_array_equal(dq, dq_copy)


def test_weight_summary_all_zero():
    weight = np.zeros((5, 5), dtype=np.float32)
    summary = weight_summary(weight)
    assert summary == {"zero_fraction": 1.0, "median": 0.0, "mean": 0.0}
    assert not any(np.isnan(v) for v in summary.values())


def test_weight_summary_mixed():
    weight = np.array([0.0, 0.0, 2.0, 4.0], dtype=np.float32)
    summary = weight_summary(weight)
    assert summary["zero_fraction"] == 0.5
    assert summary["median"] == pytest.approx(3.0)
    assert summary["mean"] == pytest.approx(3.0)


def _small_product(*, weight) -> ReducedProduct:
    import astropy.io.fits as fits

    sci = np.zeros((4, 4), dtype=np.float32)
    err = np.ones((4, 4), dtype=np.float32)
    dq = np.zeros((4, 4), dtype=DQ_DTYPE)
    return ReducedProduct(sci=sci, err=err, dq=dq, header=fits.Header(), weight=weight)


def test_reduced_product_round_trip_with_weight(tmp_path):
    weight = np.full((4, 4), 1.0, dtype=np.float32)
    weight[0, 0] = 0.0
    product = _small_product(weight=weight)

    out_path = product.write(tmp_path / "with_weight.fits")
    result = ReducedProduct.read(out_path)

    assert result.weight is not None
    assert_array_equal(result.weight, weight)
    assert result.weight.dtype == np.float32

    from astropy.io import fits

    with fits.open(out_path) as hdul:
        assert [h.name for h in hdul] == ["SCI", "ERR", "DQ", "WEIGHT"]


def test_reduced_product_round_trip_without_weight(tmp_path):
    product = _small_product(weight=None)

    out_path = product.write(tmp_path / "no_weight.fits")
    result = ReducedProduct.read(out_path)

    assert result.weight is None

    from astropy.io import fits

    with fits.open(out_path) as hdul:
        names = [h.name for h in hdul]
        assert "WEIGHT" not in names
        assert names == ["SCI", "ERR", "DQ"]
