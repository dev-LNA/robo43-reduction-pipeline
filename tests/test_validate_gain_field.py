"""Regression tests for gain/gain_setting fix (P3)."""

from __future__ import annotations

from pathlib import Path

from robo43.io.meta import FrameMeta
from robo43.io.validate import OverrideSet, validate


class TestGainSettingValidation:
    """Test that gain_setting is used instead of gain."""

    def test_photometry_requires_gain_setting_not_gain(self):
        """Photometry validation uses gain_setting field."""
        meta = FrameMeta(
            path=Path("test.fits"),
            naxis1=2048,
            naxis2=2048,
            frame_type="science",
            exptime=60.0,
            date_obs=None,
            filter="V",
            ra_deg=180.0,
            dec_deg=0.0,
            pixel_scale=0.5,
            ccd_temp=None,
            gain_setting=2.0,  # This is the right field name
            read_noise=None,
            saturation=55000.0,
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
            profile_name=None,
            origins={},
        )
        # Should pass without AttributeError
        report = validate([meta], stages=["photometry"])
        assert report.ok

    def test_gain_setting_missing_raises_issue(self):
        """Missing gain_setting is flagged as an issue."""
        meta = FrameMeta(
            path=Path("test.fits"),
            naxis1=2048,
            naxis2=2048,
            frame_type="science",
            exptime=60.0,
            date_obs=None,
            filter="V",
            ra_deg=180.0,
            dec_deg=0.0,
            pixel_scale=0.5,
            ccd_temp=None,
            gain_setting=None,  # Missing
            read_noise=None,
            saturation=55000.0,
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
            profile_name=None,
            origins={},
        )
        report = validate([meta], stages=["photometry"])
        assert not report.ok
        # Check that the issue is about gain_setting
        assert any(issue.field == "gain_setting" for issue in report.issues)

    def test_override_set_gain_setting_key(self):
        """Override key 'gain_setting' is used directly (no friendly alias)."""
        overrides = OverrideSet.from_cli(["gain_setting=7.0"])
        result = overrides.for_path(Path("x.fits"))
        # Result should have gain_setting as the key
        assert "gain_setting" in result
        assert result["gain_setting"][0] == 7.0
