"""Tests for robo43.calib.dark.

Builds its own synthetic FITS files and FrameMeta objects; does not use
tests/conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from robo43.calib.dark import DarkEntry, build_master_darks, scale_dark, select_dark
from robo43.config import CalibSettings, RuntimeSettings, Settings
from robo43.instruments.profile import InstrumentLayout, InstrumentProfile
from robo43.io.meta import FrameMeta


def _profile(gain=2.0) -> InstrumentProfile:
    return InstrumentProfile(
        name="test-instrument",
        matches={},
        pixel_scale_arcsec=0.5,
        gain_e_per_adu=gain,
        read_noise_e=5.0,
        saturation_adu=60000.0,
        nonlinearity_adu=None,
        binning_keyword="XBINNING",
        layout=InstrumentLayout(),
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


def _meta(path, shape, *, exptime, ccd_temp) -> FrameMeta:
    return FrameMeta(
        path=path,
        naxis1=shape[1],
        naxis2=shape[0],
        frame_type="dark",
        exptime=exptime,
        date_obs=None,
        filter=None,
        ra_deg=None,
        dec_deg=None,
        pixel_scale=None,
        ccd_temp=ccd_temp,
        gain=None,
        read_noise=None,
        saturation=None,
        airmass=None,
        object=None,
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


def _make_group(
    tmp_path,
    prefix,
    *,
    n,
    shape,
    exptime,
    ccd_temp,
    level,
    sigma,
    rng,
):
    metas = []
    for i in range(n):
        frame = level + rng.normal(0.0, sigma, size=shape)
        path = tmp_path / f"{prefix}_{i:02d}.fits"
        _write_fits(path, frame)
        metas.append(_meta(path, shape, exptime=exptime, ccd_temp=ccd_temp))
    return metas


def test_grouping_four_groups_tight_tolerance(tmp_path):
    rng = np.random.default_rng(1)
    h, w = 16, 16
    metas = []
    combos = [(10.0, -10.0), (10.0, -5.0), (30.0, -10.0), (30.0, -5.0)]
    for exp, temp in combos:
        metas.extend(
            _make_group(
                tmp_path,
                f"d_{exp}_{temp}",
                n=3,
                shape=(h, w),
                exptime=exp,
                ccd_temp=temp,
                level=100.0,
                sigma=3.0,
                rng=rng,
            )
        )

    profile = _profile()
    settings = _settings()
    masters, _dc = build_master_darks(
        metas, settings=settings, profile=profile, outdir=tmp_path / "tight"
    )
    assert len(masters) == 4


def test_grouping_fewer_groups_loose_tolerance(tmp_path):
    rng = np.random.default_rng(2)
    h, w = 16, 16
    metas = []
    combos = [(10.0, -10.0), (10.0, -5.0), (30.0, -10.0), (30.0, -5.0)]
    for exp, temp in combos:
        metas.extend(
            _make_group(
                tmp_path,
                f"d_{exp}_{temp}",
                n=3,
                shape=(h, w),
                exptime=exp,
                ccd_temp=temp,
                level=100.0,
                sigma=3.0,
                rng=rng,
            )
        )

    profile = _profile()
    settings = _settings(dark_exptime_tolerance_s=25.0, dark_temp_tolerance_c=10.0)
    masters, _dc = build_master_darks(
        metas, settings=settings, profile=profile, outdir=tmp_path / "loose"
    )
    assert len(masters) < 4


def test_boundary_join_within_tolerance(tmp_path):
    rng = np.random.default_rng(3)
    h, w = 16, 16
    metas = _make_group(
        tmp_path,
        "base",
        n=4,
        shape=(h, w),
        exptime=10.0,
        ccd_temp=-10.0,
        level=100.0,
        sigma=3.0,
        rng=rng,
    )
    extra_frame = 100.0 + rng.normal(0.0, 3.0, size=(h, w))
    extra_path = tmp_path / "extra.fits"
    _write_fits(extra_path, extra_frame)
    metas.append(_meta(extra_path, (h, w), exptime=10.0, ccd_temp=-10.4))

    profile = _profile()
    settings = _settings()  # defaults: 0.5/0.5
    masters, _dc = build_master_darks(
        metas, settings=settings, profile=profile, outdir=tmp_path / "join"
    )
    assert len(masters) == 1


def test_boundary_no_join_outside_tolerance(tmp_path):
    rng = np.random.default_rng(4)
    h, w = 16, 16
    metas = _make_group(
        tmp_path,
        "base",
        n=4,
        shape=(h, w),
        exptime=10.0,
        ccd_temp=-10.0,
        level=100.0,
        sigma=3.0,
        rng=rng,
    )
    extra_frame = 100.0 + rng.normal(0.0, 3.0, size=(h, w))
    extra_path = tmp_path / "extra.fits"
    _write_fits(extra_path, extra_frame)
    metas.append(_meta(extra_path, (h, w), exptime=10.0, ccd_temp=-9.4))

    profile = _profile()
    settings = _settings()  # defaults: 0.5/0.5
    masters, _dc = build_master_darks(
        metas, settings=settings, profile=profile, outdir=tmp_path / "nojoin"
    )
    assert len(masters) == 2


def test_dark_current_map_recovery(tmp_path):
    rng = np.random.default_rng(5)
    h, w = 20, 20
    gain = 2.0
    exptime = 20.0
    current_e_per_s = 50.0
    adu_level = current_e_per_s * exptime / gain

    metas = _make_group(
        tmp_path,
        "dc",
        n=8,
        shape=(h, w),
        exptime=exptime,
        ccd_temp=-10.0,
        level=adu_level,
        sigma=2.0,
        rng=rng,
    )

    profile = _profile(gain=gain)
    settings = _settings()
    _masters, dark_current_master = build_master_darks(
        metas, settings=settings, profile=profile, outdir=tmp_path / "dcmap", bias=None
    )

    assert dark_current_master.bunit == "electron/s"
    assert float(np.mean(dark_current_master.data)) == pytest.approx(
        current_e_per_s, rel=0.02
    )


def test_select_dark_exact():
    entry = DarkEntry(
        path=tmp_path_placeholder(),
        exptime=30.0,
        ccd_temp=-10.0,
        data=np.full((10, 10), 100.0, dtype=np.float32),
        variance=None,
        n_combine=5,
    )
    profile = _profile()
    settings = _settings()
    selection = select_dark(
        [entry],
        None,
        None,
        exptime=entry.exptime,
        ccd_temp=entry.ccd_temp,
        settings=settings,
        profile=profile,
    )
    assert selection.method == "exact"
    assert selection.scale == 1.0
    assert selection.reason


def tmp_path_placeholder():
    from pathlib import Path

    return Path("master_dark_30s_-10.0C.fits")


def test_select_dark_current_map_double_exptime():
    profile = _profile(gain=1.0)
    settings = _settings()
    ref_temp = -10.0
    dark_current_map = np.full((10, 10), 5.0, dtype=np.float32)

    T = 15.0
    selection_t = select_dark(
        [], dark_current_map, ref_temp, exptime=T, ccd_temp=ref_temp,
        settings=settings, profile=profile,
    )
    assert selection_t.method == "current_map"
    data_t, _cards_t = scale_dark(selection_t, T, ref_temp, settings=settings)

    selection_2t = select_dark(
        [], dark_current_map, ref_temp, exptime=2 * T, ccd_temp=ref_temp,
        settings=settings, profile=profile,
    )
    data_2t, _cards_2t = scale_dark(selection_2t, 2 * T, ref_temp, settings=settings)

    assert data_2t == pytest.approx(2.0 * data_t)


def test_select_dark_arrhenius_doubling():
    profile = _profile(gain=1.0)
    settings = _settings(dark_temp_extrap_max_c=10.0)
    ref_temp = -10.0
    dark_current_map = np.full((10, 10), 5.0, dtype=np.float32)
    exptime = 20.0

    selection_ref = select_dark(
        [], dark_current_map, ref_temp, exptime=exptime, ccd_temp=ref_temp,
        settings=settings, profile=profile,
    )
    data_ref, _ = scale_dark(selection_ref, exptime, ref_temp, settings=settings)

    hotter_temp = ref_temp + settings.calib.dark_doubling_temp_c
    selection_hot = select_dark(
        [], dark_current_map, ref_temp, exptime=exptime, ccd_temp=hotter_temp,
        settings=settings, profile=profile,
    )
    data_hot, _ = scale_dark(selection_hot, exptime, hotter_temp, settings=settings)

    assert data_hot == pytest.approx(2.0 * data_ref)


def test_select_dark_extrapolation_refusal():
    profile = _profile(gain=1.0)
    settings = _settings()
    ref_temp = -10.0
    dark_current_map = np.full((10, 10), 5.0, dtype=np.float32)
    far_temp = ref_temp + settings.calib.dark_temp_extrap_max_c + 5.0

    selection = select_dark(
        [], dark_current_map, ref_temp, exptime=20.0, ccd_temp=far_temp,
        settings=settings, profile=profile,
    )
    assert selection.method == "none"
    assert selection.reason

    data, cards = scale_dark(selection, 20.0, far_temp, settings=settings)
    assert data.shape == ()
    assert float(data) == 0.0
    card_dict = {c[0]: c[1] for c in cards}
    assert card_dict["DARKFIL"] == "NONE"


def test_missing_bias_degrades_gracefully(tmp_path):
    rng = np.random.default_rng(6)
    h, w = 12, 12
    metas = _make_group(
        tmp_path,
        "nobias",
        n=4,
        shape=(h, w),
        exptime=10.0,
        ccd_temp=-10.0,
        level=100.0,
        sigma=3.0,
        rng=rng,
    )
    profile = _profile()
    settings = _settings()
    masters, _dc = build_master_darks(
        metas, settings=settings, profile=profile, outdir=tmp_path / "nobiasout", bias=None
    )
    master = next(iter(masters.values()))
    card_dict = {c[0]: c[1] for c in master.header_cards}
    assert card_dict["BIASFIL"] == "NONE"
