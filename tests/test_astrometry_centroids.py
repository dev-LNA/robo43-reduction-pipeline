"""Tests for robo43.astrometry.centroids.

Builds small synthetic arrays directly; does not use tests/conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from robo43.astrometry.centroids import (
    DetectionRung,
    detect_stars_for_rung,
    estimate_background,
    measure_fwhm,
)
from robo43.products.dq import DQ, DQ_DTYPE


def _gaussian_field(
    shape: tuple[int, int],
    centers: list[tuple[float, float]],
    *,
    sigma: float,
    amplitude: float,
    background: float = 100.0,
    noise_sigma: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """A background + Gaussian-noise field with Gaussian PSF stars at ``centers``
    (x, y), each with the same ``sigma`` (pixels) and ``amplitude`` (peak counts
    above background)."""
    rng = np.random.default_rng(seed)
    height, width = shape
    data = rng.normal(background, noise_sigma, size=shape).astype(np.float64)
    yy, xx = np.mgrid[0:height, 0:width]
    for cx, cy in centers:
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        data += amplitude * np.exp(-r2 / (2.0 * sigma**2))
    return data.astype(np.float32)


def _grid_centers(
    *, n_per_side: int = 4, spacing: float = 35.0, offset: float = 40.0
) -> list[tuple[float, float]]:
    """``n_per_side ** 2`` well-separated star centers, comfortably clear of a
    10-pixel edge margin on a field sized to fit them."""
    coords = [offset + i * spacing for i in range(n_per_side)]
    return [(x, y) for x in coords for y in coords]


def test_estimate_background_recovers_known_level():
    rng = np.random.default_rng(1)
    data = rng.normal(500.0, 5.0, size=(200, 200)).astype(np.float32)
    med, sd = estimate_background(data)
    assert med == pytest.approx(500.0, abs=1.0)
    assert sd == pytest.approx(5.0, rel=0.15)


def test_measure_fwhm_recovers_known_sigma_within_10_percent():
    sigma_px = 2.2
    expected_fwhm = 2.3548 * sigma_px
    centers = _grid_centers(n_per_side=3, spacing=40.0, offset=30.0)
    data = _gaussian_field((140, 140), centers, sigma=sigma_px, amplitude=400.0, noise_sigma=1.0)

    med, sd = estimate_background(data)
    fwhm = measure_fwhm(data, med, sd, fallback_px=999.0)

    assert fwhm != 999.0  # the fallback must not have been used
    assert fwhm == pytest.approx(expected_fwhm, rel=0.10)


def test_measure_fwhm_falls_back_when_too_few_peaks():
    rng = np.random.default_rng(2)
    data = rng.normal(500.0, 5.0, size=(60, 60)).astype(np.float32)  # no stars at all
    med, sd = estimate_background(data)
    fwhm = measure_fwhm(data, med, sd, fallback_px=4.5)
    assert fwhm == 4.5


_RUNG = DetectionRung(
    threshold_sigma=5.0, sharplo=0.1, sharphi=1.5, round_limit=0.75, max_stars=250
)
_FIELD_SHAPE = (180, 180)


def _field_and_background(centers: list[tuple[float, float]]) -> tuple[np.ndarray, float, float]:
    data = _gaussian_field(_FIELD_SHAPE, centers, sigma=2.0, amplitude=500.0)
    med, sd = estimate_background(data)
    return data, med, sd


def test_detect_stars_for_rung_finds_every_clean_star():
    centers = _grid_centers()  # 16 stars, well above the 10-raw-detection floor
    data, med, sd = _field_and_background(centers)
    dq = np.zeros(data.shape, dtype=DQ_DTYPE)

    stars = detect_stars_for_rung(
        data, dq, fwhm_px=4.7, background_med=med, background_sd=sd, rung=_RUNG
    )
    assert len(stars) == len(centers)


def test_detect_stars_for_rung_rejects_saturated_pixel():
    centers = _grid_centers()
    data, med, sd = _field_and_background(centers)
    target_x, target_y = centers[0]

    dq = np.zeros(data.shape, dtype=DQ_DTYPE)
    dq[round(target_y), round(target_x)] = int(DQ.SATURATED)

    stars = detect_stars_for_rung(
        data, dq, fwhm_px=4.7, background_med=med, background_sd=sd, rung=_RUNG
    )
    assert len(stars) == len(centers) - 1
    assert all(
        abs(s.x - (target_x + 1.0)) > 0.5 or abs(s.y - (target_y + 1.0)) > 0.5 for s in stars
    )


def test_detect_stars_for_rung_rejects_edge_sources():
    centers = _grid_centers()
    centers_with_edge = [*centers, (5.0, 5.0)]  # within the default 10px edge margin
    data = _gaussian_field(_FIELD_SHAPE, centers_with_edge, sigma=2.0, amplitude=500.0)
    med, sd = estimate_background(data)
    dq = np.zeros(data.shape, dtype=DQ_DTYPE)

    stars = detect_stars_for_rung(
        data, dq, fwhm_px=4.7, background_med=med, background_sd=sd, rung=_RUNG, edge_margin_px=10
    )
    assert len(stars) == len(centers)
    assert all(s.x > 10.0 and s.y > 10.0 for s in stars)


def test_detect_stars_for_rung_rejects_non_finite_pixel():
    centers = _grid_centers()
    data = _gaussian_field(_FIELD_SHAPE, centers, sigma=2.0, amplitude=500.0)
    target_x, target_y = centers[0]
    data[round(target_y), round(target_x)] = np.nan
    med, sd = estimate_background(np.nan_to_num(data, nan=100.0))
    dq = np.zeros(data.shape, dtype=DQ_DTYPE)

    stars = detect_stars_for_rung(
        data, dq, fwhm_px=4.7, background_med=med, background_sd=sd, rung=_RUNG
    )
    assert len(stars) == len(centers) - 1


def test_detect_stars_for_rung_returns_empty_below_minimum_raw_detections():
    rng = np.random.default_rng(3)
    data = rng.normal(500.0, 5.0, size=(60, 60)).astype(np.float32)  # pure noise
    dq = np.zeros(data.shape, dtype=DQ_DTYPE)
    stars = detect_stars_for_rung(
        data, dq, fwhm_px=4.0, background_med=500.0, background_sd=5.0, rung=_RUNG
    )
    assert stars == ()


def test_detect_stars_falls_back_to_second_rung(monkeypatch):
    """detect_stars() tries ladder rungs in order and reports which one won."""
    import robo43.astrometry.centroids as centroids_mod

    calls: list[DetectionRung] = []

    def fake_detect(*_args, rung, **_kwargs):
        calls.append(rung)
        if rung.threshold_sigma == 5.0:
            return ()  # rung 0 fails
        return (
            centroids_mod.DetectedStar(
                x=1.0, y=1.0, flux=1.0, peak=1.0, sharpness=0.5, roundness=0.0
            ),
        )

    monkeypatch.setattr(centroids_mod, "detect_stars_for_rung", fake_detect)

    data = np.zeros((10, 10), dtype=np.float32)
    dq = np.zeros((10, 10), dtype=DQ_DTYPE)
    result = centroids_mod.detect_stars(data, dq, pixel_scale_arcsec=0.5, seeing_fwhm_arcsec=2.0)

    assert result.rung == 1
    assert len(result.stars) == 1
    assert len(calls) == 2


def test_detect_stars_reports_rung_minus_one_when_every_rung_fails(monkeypatch):
    import robo43.astrometry.centroids as centroids_mod

    def _always_empty(*_args, **_kwargs):
        return ()

    monkeypatch.setattr(centroids_mod, "detect_stars_for_rung", _always_empty)

    data = np.zeros((10, 10), dtype=np.float32)
    dq = np.zeros((10, 10), dtype=DQ_DTYPE)
    result = centroids_mod.detect_stars(data, dq, pixel_scale_arcsec=0.5, seeing_fwhm_arcsec=2.0)

    assert result.rung == -1
    assert result.stars == ()
