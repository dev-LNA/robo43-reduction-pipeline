"""Tests for robo43.reduce.cosmics.

Builds its own in-memory numpy arrays; does not use tests/conftest.py.
"""

from __future__ import annotations

import dataclasses
import sys

import numpy as np
import pytest

import robo43.exceptions
from robo43.config import CalibSettings
from robo43.products.dq import DQ
from robo43.reduce.cosmics import apply_to_dq, detect_cosmics


def _add_gauss(img: np.ndarray, x0: float, y0: float, amp: float, sigma: float = 3.0) -> None:
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    img += amp * np.exp(-(((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma**2)))


def _make_frame() -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]], float]:
    rng = np.random.default_rng(0)
    ny, nx = 120, 120
    bg = 500.0
    sat_adu = 55000.0

    data = rng.normal(bg, np.sqrt(bg), (ny, nx)).astype(np.float32)

    for x0, y0, amp in [(30, 30, 2500), (80, 40, 3500), (60, 90, 1800)]:
        _add_gauss(data, x0, y0, amp, sigma=3.0)

    sat_x, sat_y = 100, 100
    _add_gauss(data, sat_x, sat_y, 200000, sigma=3.0)
    data = np.clip(data, None, sat_adu).astype(np.float32)

    cr_coords = [(10, 10), (50, 15), (20, 70), (70, 20), (15, 110)]
    for cx, cy in cr_coords:
        data[cy, cx] += 4000

    return data, (sat_x, sat_y), cr_coords, sat_adu


def test_recall_and_false_positives() -> None:
    data, (sat_x, sat_y), cr_coords, sat_adu = _make_frame()
    settings = CalibSettings()
    gain = 2.0
    rn_e = 8.0

    result = detect_cosmics(
        data,
        settings=settings,
        gain_e_per_adu=gain,
        read_noise_e=rn_e,
        saturation=sat_adu,
        in_electrons=False,
    )

    for cx, cy in cr_coords:
        assert result.mask[cy, cx]

    assert not result.mask[sat_y, sat_x]

    flagged = int(result.mask.sum())
    false_positives = flagged - len(cr_coords)
    assert false_positives <= 40

    assert result.cleaned.dtype == np.float32
    assert np.all(np.isfinite(result.cleaned))


def test_electrons_adu_equivalence() -> None:
    data_adu, _, _, sat_adu = _make_frame()
    gain = 2.0
    rn_e = 8.0
    settings = CalibSettings()

    data_e = (data_adu * gain).astype(np.float32)

    result1 = detect_cosmics(
        data_adu,
        settings=settings,
        gain_e_per_adu=gain,
        read_noise_e=rn_e,
        saturation=sat_adu,
        in_electrons=False,
    )
    result2 = detect_cosmics(
        data_e,
        settings=settings,
        gain_e_per_adu=None,
        read_noise_e=rn_e,
        saturation=sat_adu * gain,
        in_electrons=True,
    )

    assert np.array_equal(result1.mask, result2.mask)
    assert np.allclose(result2.cleaned, result1.cleaned * gain, rtol=1e-3)


def test_apply_to_dq() -> None:
    data, _, cr_coords, sat_adu = _make_frame()
    settings = CalibSettings()
    result = detect_cosmics(
        data,
        settings=settings,
        gain_e_per_adu=2.0,
        read_noise_e=8.0,
        saturation=sat_adu,
        in_electrons=False,
    )

    dq = np.zeros(data.shape, dtype=np.int32)
    hot_px = (5, 5)
    sat_px = (6, 6)
    assert not result.mask[hot_px]
    assert not result.mask[sat_px]
    dq[hot_px] = int(DQ.HOT)
    dq[sat_px] = int(DQ.SATURATED)
    dq_before = dq.copy()

    new_dq = apply_to_dq(dq, result.mask)

    assert new_dq.dtype == np.int32
    assert np.array_equal(dq, dq_before)

    for cx, cy in cr_coords:
        assert new_dq[cy, cx] == int(DQ.COSMIC_RAY)

    assert new_dq[hot_px] == int(DQ.HOT)
    assert new_dq[sat_px] == int(DQ.SATURATED)


def test_disabled_path_is_noop() -> None:
    data, _, _, sat_adu = _make_frame()
    settings = dataclasses.replace(CalibSettings(), cosmic_ray_rejection=False)

    result = detect_cosmics(
        data,
        settings=settings,
        gain_e_per_adu=2.0,
        read_noise_e=8.0,
        saturation=sat_adu,
        in_electrons=False,
    )

    assert not result.mask.any()
    assert result.enabled is False
    assert result.n_flagged == 0
    assert np.array_equal(result.cleaned, data.astype(np.float32))
    assert result.cleaned is not data


def test_missing_astroscrappy_raises_external_tool_error(monkeypatch: pytest.MonkeyPatch) -> None:
    data, _, _, sat_adu = _make_frame()
    settings = CalibSettings()

    monkeypatch.setitem(sys.modules, "astroscrappy", None)

    with pytest.raises(robo43.exceptions.ExternalToolError) as excinfo:
        detect_cosmics(
            data,
            settings=settings,
            gain_e_per_adu=2.0,
            read_noise_e=8.0,
            saturation=sat_adu,
            in_electrons=False,
        )

    msg = str(excinfo.value)
    assert "astroscrappy" in msg
    assert "pip install" in msg


def test_header_cards() -> None:
    data, _, _, sat_adu = _make_frame()
    settings = CalibSettings()

    enabled_result = detect_cosmics(
        data,
        settings=settings,
        gain_e_per_adu=2.0,
        read_noise_e=8.0,
        saturation=sat_adu,
        in_electrons=False,
    )
    disabled_settings = dataclasses.replace(settings, cosmic_ray_rejection=False)
    disabled_result = detect_cosmics(
        data,
        settings=disabled_settings,
        gain_e_per_adu=2.0,
        read_noise_e=8.0,
        saturation=sat_adu,
        in_electrons=False,
    )

    expected_keywords = {"CRREJ", "CRNPIX", "CRFRAC", "CRSIGCLP", "CRSOFT", "CRSOFTV"}

    for result, enabled in [(enabled_result, True), (disabled_result, False)]:
        cards = result.header_cards()
        keywords = {kw for kw, _, _ in cards}
        assert expected_keywords <= keywords
        cards_by_kw = {kw: value for kw, value, _ in cards}
        assert cards_by_kw["CRREJ"] is enabled
        assert cards_by_kw["CRSOFT"] == "astroscrappy"


def test_satlevel_scaled_to_astroscrappys_internal_electron_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """astroscrappy compares its ``satlevel`` against its own internal
    ``data * gain`` array (always electrons; see its docstring: "Saturation
    of level of the image (electrons)"). ``detect_cosmics`` must therefore
    forward ``saturation * gain`` as ``satlevel``, not ``saturation``
    unchanged -- regardless of whether the caller's gain is above or below
    1.0.
    """
    import astroscrappy

    data = np.full((10, 10), 500.0, dtype=np.float32)
    captured: dict = {}

    def fake_detect_cosmics(indat: np.ndarray, **kwargs):
        captured.update(kwargs)
        return np.zeros(indat.shape, dtype=bool), indat.astype(np.float32).copy()

    monkeypatch.setattr(astroscrappy, "detect_cosmics", fake_detect_cosmics)

    settings = CalibSettings()
    sat_adu = 55000.0

    # Sub-unity gain, matching ROBO43's real photon-transfer-measured regime.
    gain = 0.7
    detect_cosmics(
        data,
        settings=settings,
        gain_e_per_adu=gain,
        read_noise_e=7.0,
        saturation=sat_adu,
        in_electrons=False,
    )
    assert captured["satlevel"] == pytest.approx(sat_adu * gain)

    # in_electrons=True forces astroscrappy's own gain to 1.0, so saturation
    # (already in electrons) must be passed through unchanged.
    captured.clear()
    data_e = (data * gain).astype(np.float32)
    detect_cosmics(
        data_e,
        settings=settings,
        gain_e_per_adu=None,
        read_noise_e=7.0,
        saturation=sat_adu * gain,
        in_electrons=True,
    )
    assert captured["satlevel"] == pytest.approx(sat_adu * gain)


def test_saturated_star_not_flagged_as_cosmic_ray_with_subunity_gain() -> None:
    """Regression test for the satlevel unit bug, using real astroscrappy
    (not mocked).

    ROBO43's real photon-transfer-measured gain is sub-unity (~0.7 e-/ADU).
    With an unscaled ADU-space ``satlevel``, astroscrappy's internal
    ``data * gain`` array never reaches it, so saturated-star protection
    never triggers, and pixels in the flat-topped core of a saturated star
    get treated as candidate cosmic rays and cleaned away. With the fix,
    none of the saturated core's pixels are flagged.
    """
    rng = np.random.default_rng(1)
    ny, nx = 80, 80
    bg = 500.0
    sat_adu = 55000.0
    gain = 0.7
    rn_e = 7.0

    data = rng.normal(bg, np.sqrt(bg) * gain, (ny, nx)).astype(np.float32)
    yy, xx = np.mgrid[0:ny, 0:nx]
    star = 200000 * np.exp(-((xx - 40) ** 2 + (yy - 40) ** 2) / (2 * 4.0**2))
    data += star
    data = np.clip(data, None, sat_adu).astype(np.float32)

    settings = CalibSettings()
    result = detect_cosmics(
        data,
        settings=settings,
        gain_e_per_adu=gain,
        read_noise_e=rn_e,
        saturation=sat_adu,
        in_electrons=False,
    )

    core = result.mask[36:45, 36:45]
    assert core.sum() == 0
