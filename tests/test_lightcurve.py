from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from robo43.exceptions import PhotometryError
from robo43.photometry.lightcurve import (
    REQUIRED_COLUMNS,
    average_magnitudes,
    combine_in_flux_space,
    read_photometry_csvs,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_recovers_known_mean_weighted_and_sigma_clipped():
    rng = np.random.default_rng(0)
    true_mag = 12.5
    sigma_true = 0.05
    n = 30
    mags = true_mag + rng.normal(0, sigma_true, n)
    jds = 2460000.0 + np.arange(n) * 0.01

    rows = [
        {"target": "T1", "filter": "V", "jd": jd, "mag": m, "mag_err": sigma_true}
        for jd, m in zip(jds, mags, strict=True)
    ]
    frame = _frame(rows)

    tol = 3 * sigma_true / np.sqrt(n)

    result_wm = average_magnitudes(frame, method="weighted_mean")
    assert len(result_wm) == 1
    assert abs(result_wm["mag"].iloc[0] - true_mag) < tol

    result_scm = average_magnitudes(frame, method="sigma_clipped_mean")
    assert len(result_scm) == 1
    assert abs(result_scm["mag"].iloc[0] - true_mag) < tol


def test_bug46_jd_does_not_leak_across_filters():
    rng = np.random.default_rng(0)
    v_jds = 2460000.0 + rng.uniform(0, 1, 10)
    b_jds = 2460010.0 + rng.uniform(0, 1, 10)

    rows = []
    for jd in v_jds:
        rows.append({"target": "T1", "filter": "V", "jd": jd, "mag": 12.0, "mag_err": 0.05})
    for jd in b_jds:
        rows.append({"target": "T1", "filter": "B", "jd": jd, "mag": 13.0, "mag_err": 0.05})

    frame = _frame(rows)
    result = average_magnitudes(frame, method="weighted_mean")

    assert len(result) == 2

    jd_v = result.loc[result["filter"] == "V", "jd"].iloc[0]
    jd_b = result.loc[result["filter"] == "B", "jd"].iloc[0]

    assert abs(jd_v - v_jds.mean()) < 0.1
    assert abs(jd_b - b_jds.mean()) < 0.1
    assert abs(jd_b - jd_v) > 5


def test_bug47_all_missing_group_skipped_not_raised():
    rows = [
        {"target": "T1", "filter": "V", "jd": 2460000.0, "mag": 12.0, "mag_err": 0.05},
        {"target": "T1", "filter": "V", "jd": 2460000.1, "mag": 12.1, "mag_err": 0.05},
        {"target": "T1", "filter": "B", "jd": 2460000.0, "mag": np.nan, "mag_err": np.nan},
        {"target": "T1", "filter": "B", "jd": 2460000.1, "mag": np.nan, "mag_err": np.nan},
    ]
    frame = _frame(rows)

    result = average_magnitudes(frame, method="weighted_mean")

    assert len(result) == 1
    assert list(result.columns) == [
        "target",
        "filter",
        "jd",
        "mag",
        "mag_err",
        "n_points",
        "method",
    ]

    all_nan_rows = [
        {"target": "T1", "filter": "V", "jd": np.nan, "mag": np.nan, "mag_err": np.nan},
        {"target": "T1", "filter": "B", "jd": np.nan, "mag": np.nan, "mag_err": np.nan},
    ]
    all_nan_frame = _frame(all_nan_rows)
    empty_result = average_magnitudes(all_nan_frame, method="weighted_mean")

    assert len(empty_result) == 0
    assert list(empty_result.columns) == [
        "target",
        "filter",
        "jd",
        "mag",
        "mag_err",
        "n_points",
        "method",
    ]


def test_bug45_no_float_equality_needed_for_median():
    mag_values = [10.123456, 10.654321, 10.234567, 10.543210]
    rows = [
        {"target": "T1", "filter": "V", "jd": 2460000.0 + i, "mag": m, "mag_err": 0.02}
        for i, m in enumerate(mag_values)
    ]
    frame = _frame(rows)

    result = average_magnitudes(frame, method="median")

    assert len(result) == 1
    mag = result["mag"].iloc[0]
    assert np.isfinite(mag)
    assert np.isclose(mag, np.median(mag_values))


def test_combine_in_flux_space_hand_computed():
    mags = np.array([10.0, 10.5])
    mag_errs = np.array([0.05, 0.10])

    flux = 10 ** (-0.4 * mags)
    flux_err = flux * np.log(10) / 2.5 * mag_errs
    weights = 1.0 / flux_err**2
    flux_mean = np.sum(weights * flux) / np.sum(weights)
    flux_mean_err = np.sqrt(1.0 / np.sum(weights))
    expected_mag = -2.5 * np.log10(flux_mean)
    expected_mag_err = 2.5 / np.log(10) * flux_mean_err / flux_mean

    mag, mag_err = combine_in_flux_space(mags, mag_errs)

    assert np.isclose(mag, expected_mag, rtol=1e-9)
    assert np.isclose(mag_err, expected_mag_err, rtol=1e-9)


def test_flux_space_vs_magnitude_space_differ_for_noisy_input():
    rows = [
        {"target": "T1", "filter": "V", "jd": 2460000.0, "mag": 12.00, "mag_err": 0.01},
        {"target": "T1", "filter": "V", "jd": 2460000.1, "mag": 12.01, "mag_err": 0.01},
        {"target": "T1", "filter": "V", "jd": 2460000.2, "mag": 13.5, "mag_err": 0.5},
        {"target": "T1", "filter": "V", "jd": 2460000.3, "mag": 13.6, "mag_err": 0.5},
        {"target": "T1", "filter": "V", "jd": 2460000.4, "mag": 13.7, "mag_err": 0.5},
    ]
    frame = _frame(rows)

    result_median = average_magnitudes(frame, method="median")
    result_weighted = average_magnitudes(frame, method="weighted_mean")

    mag_median = result_median["mag"].iloc[0]
    mag_weighted = result_weighted["mag"].iloc[0]

    assert abs(mag_median - mag_weighted) > 0.05


def test_read_photometry_csvs(tmp_path):
    file1 = tmp_path / "target1.csv"
    file2 = tmp_path / "target2.csv"
    missing_file = tmp_path / "does_not_exist.csv"

    df1 = pd.DataFrame(
        {
            "target": ["T1", "T1"],
            "filter": ["V", "V"],
            "jd": [2460000.0, 2460000.1],
            "mag": [12.0, 12.1],
            "mag_err": [0.05, 0.05],
        }
    )
    df2 = pd.DataFrame(
        {
            "target": ["T2", "T2", "T2"],
            "filter": ["B", "B", "B"],
            "jd": [2460001.0, 2460001.1, 2460001.2],
            "mag": [13.0, 13.1, 13.2],
            "mag_err": [0.03, 0.03, 0.03],
            "image_file": ["a.fits", "b.fits", "c.fits"],
        }
    )
    df1.to_csv(file1, index=False)
    df2.to_csv(file2, index=False)

    result = read_photometry_csvs([file1, file2, missing_file])

    assert len(result) == len(df1) + len(df2)
    assert "source_path" in result.columns
    assert set(result["source_path"]) == {str(file1), str(file2)}

    with pytest.raises(PhotometryError):
        read_photometry_csvs([file1, file2, missing_file], missing_ok=False)

    empty_result = read_photometry_csvs([])
    assert list(empty_result.columns)[:5] == list(REQUIRED_COLUMNS)

    empty_result2 = read_photometry_csvs([missing_file], missing_ok=True)
    assert list(empty_result2.columns)[:5] == list(REQUIRED_COLUMNS)


def test_unknown_method_raises_photometry_error():
    frame = _frame(
        [{"target": "T1", "filter": "V", "jd": 2460000.0, "mag": 12.0, "mag_err": 0.05}]
    )

    with pytest.raises(PhotometryError, match="bogus"):
        average_magnitudes(frame, method="bogus")


def test_missing_required_column_raises_photometry_error():
    frame = pd.DataFrame(
        {
            "target": ["T1"],
            "filter": ["V"],
            "jd": [2460000.0],
            "mag": [12.0],
        }
    )

    with pytest.raises(PhotometryError, match="mag_err"):
        average_magnitudes(frame)
