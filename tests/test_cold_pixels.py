"""Tests for the cold-pixel mask (robo43.calib.masks.cold_pixel_mask and friends).

Builds its own in-memory numpy fixtures; does not use tests/conftest.py.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from astropy.io import fits

from robo43.calib.masks import (
    MaskSet,
    build_mask,
    cold_pixel_mask,
    read_cold_pixel_masks,
    write_cold_pixel_mask,
)
from robo43.config import CalibSettings
from robo43.exceptions import CalibrationError
from robo43.products.dq import DQ


def _settings(**overrides) -> CalibSettings:
    return CalibSettings(**overrides)


def _vignetted_flat(rng, h=256, w=256, noise=0.005, n_cold=20, cold_level=0.8):
    """A vignetted, noisy normalised flat with ``n_cold`` injected cold pixels."""
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - (h - 1) / 2, xx - (w - 1) / 2) / np.hypot(h / 2, w / 2)
    smooth = 1.0 - 0.3 * r**2 + 0.05 * (xx / w)
    flat = smooth * (1.0 + rng.normal(0.0, noise, size=(h, w)))
    idx = rng.choice(h * w, size=n_cold, replace=False)
    ys, xs = np.unravel_index(idx, (h, w))
    flat[ys, xs] = cold_level * smooth[ys, xs]
    injected = np.zeros((h, w), dtype=bool)
    injected[ys, xs] = True
    return flat, injected


def test_injected_cold_pixels_found_with_negligible_false_positives():
    rng = np.random.default_rng(11)
    flat, injected = _vignetted_flat(rng)

    cold = cold_pixel_mask(flat, 20, _settings())

    assert cold is not None
    assert cold.shape == flat.shape
    assert cold.dtype == bool
    assert np.all(cold[injected])
    false_positives = int(np.count_nonzero(cold & ~injected))
    assert false_positives / flat.size < 1e-4


def test_ragged_shape_and_non_finite_pixels():
    rng = np.random.default_rng(12)
    flat, injected = _vignetted_flat(rng, h=200, w=237, n_cold=10)
    flat[3, 4] = np.nan
    flat[50:60, 50:60] = np.nan  # a whole region with no data
    injected[3, 4] = False
    injected[50:60, 50:60] = False

    cold = cold_pixel_mask(flat, 25, _settings())

    assert cold is not None
    assert cold.shape == (200, 237)
    assert not cold[3, 4]
    assert not cold[50:60, 50:60].any()
    assert np.all(cold[injected])


def test_too_few_flats_skips_with_warning(caplog):
    rng = np.random.default_rng(13)
    flat, _ = _vignetted_flat(rng)

    with caplog.at_level(logging.WARNING, logger="robo43.calib.masks"):
        assert cold_pixel_mask(flat, 19, _settings()) is None
    assert any("cold-pixel mask skipped" in rec.message for rec in caplog.records)

    assert cold_pixel_mask(flat, 20, _settings()) is not None
    assert cold_pixel_mask(flat, 5, _settings(cold_pixel_min_flats=5)) is not None


def test_constant_flat_returns_none():
    flat = np.ones((64, 64))
    assert cold_pixel_mask(flat, 20, _settings()) is None


def test_build_mask_cold_only_sets_dead_and_round_trips(tmp_path):
    cold = np.zeros((40, 50), dtype=bool)
    cold[3, 4] = cold[10, 20] = cold[39, 49] = True
    settings = _settings(cold_pixel_nsigma=6.0)

    result = build_mask(cold_pixel_mask=cold, settings=settings)

    assert np.array_equal((result.dq & int(DQ.DEAD)) != 0, cold)
    assert not (result.dq & ~int(DQ.DEAD)).any()
    assert result.summary["cold"] == pytest.approx(3 / cold.size)
    assert result.summary["dead"] == pytest.approx(3 / cold.size)
    assert result.summary["cold_nsigma"] == 6.0
    assert "cold_nsigma" not in result.text_summary()
    assert "cold:" in result.text_summary()

    path = tmp_path / "mask.fits"
    result.write(path)
    read_back = MaskSet.read(path)
    assert np.array_equal(read_back.dq, result.dq)
    assert read_back.summary["cold"] == pytest.approx(result.summary["cold"])
    assert read_back.summary["cold_nsigma"] == pytest.approx(6.0)
    assert read_back.summary["dead"] == pytest.approx(result.summary["dead"])
    with fits.open(path) as hdul:
        assert "FRCOLD" in hdul["DQ"].header
        assert "COLDNSIG" in hdul["DQ"].header
    assert all(len(str(fits.Card(k, v, c))) <= 80 for k, v, c in result.header_cards())


def test_build_mask_hot_and_cold_are_or_ed():
    rng = np.random.default_rng(14)
    dark = rng.normal(1.0, 0.05, size=(50, 50))
    dark[5, 5] = np.median(dark) + 50 * 0.05
    cold = np.zeros((50, 50), dtype=bool)
    cold[30, 30] = True

    result = build_mask(dark_current_map=dark, cold_pixel_mask=cold, settings=_settings())

    assert int(result.dq[5, 5]) & int(DQ.HOT)
    assert int(result.dq[30, 30]) & int(DQ.DEAD)
    assert not int(result.dq[30, 30]) & int(DQ.HOT)
    assert result.summary["hot"] > 0
    assert result.summary["cold"] == pytest.approx(1 / cold.size)


def test_build_mask_cold_shape_mismatch_raises():
    with pytest.raises(CalibrationError):
        build_mask(
            dark_current_map=np.ones((10, 10)),
            cold_pixel_mask=np.zeros((10, 11), dtype=bool),
            settings=_settings(),
        )
    with pytest.raises(CalibrationError):
        build_mask(
            cold_pixel_mask=np.zeros((10, 10), dtype=bool), shape=(9, 9), settings=_settings()
        )


def test_write_and_read_cold_pixel_masks_or_all_files(tmp_path):
    a = np.zeros((20, 30), dtype=bool)
    a[1, 2] = True
    b = np.zeros((20, 30), dtype=bool)
    b[5, 6] = True
    settings = _settings()

    write_cold_pixel_mask(tmp_path / "coldpix_R_dome.fits", a, n_flats=21, settings=settings)
    write_cold_pixel_mask(tmp_path / "coldpix_G_dome.fits", b, n_flats=22, settings=settings)

    combined, names = read_cold_pixel_masks(tmp_path)

    assert names == ["coldpix_G_dome.fits", "coldpix_R_dome.fits"]
    assert combined is not None
    assert np.array_equal(combined, a | b)
    with fits.open(tmp_path / "coldpix_R_dome.fits") as hdul:
        assert hdul["COLDPIX"].data.dtype == np.uint8
        header = hdul["COLDPIX"].header
        assert header["NFLATS"] == 21
        assert header["NCOLD"] == 1
        assert header["COLDBOX"] == 32
        assert header["COLDNSIG"] == 5.0
        assert header["FRCOLD"] == pytest.approx(1 / a.size)


def test_read_cold_pixel_masks_empty_and_mismatched(tmp_path):
    assert read_cold_pixel_masks(tmp_path) == (None, [])

    settings = _settings()
    write_cold_pixel_mask(
        tmp_path / "coldpix_a.fits", np.zeros((4, 4), dtype=bool), n_flats=20, settings=settings
    )
    write_cold_pixel_mask(
        tmp_path / "coldpix_b.fits", np.zeros((4, 5), dtype=bool), n_flats=20, settings=settings
    )
    with pytest.raises(CalibrationError):
        read_cold_pixel_masks(tmp_path)
