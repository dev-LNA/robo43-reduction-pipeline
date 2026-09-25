"""Tests for io.inventory module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from robo43.instruments.profile import resolve_profile
from robo43.io.headers import classify_from_object
from robo43.io.inventory import (
    INVENTORY_COLUMNS,
    build_inventory,
    filter_inventory,
    metas_from_inventory,
    write_inventory,
)


class TestClassifyFromObject:
    """Unit tests for classify_from_object."""

    def test_focus(self):
        assert classify_from_object("focus run 3") == "focus"

    def test_science(self):
        assert classify_from_object("M31") == "science"

    def test_none(self):
        assert classify_from_object(None) is None

    def test_empty(self):
        assert classify_from_object("") is None

    def test_bias_uppercase(self):
        assert classify_from_object("BIAS FRAME") == "bias"

    def test_flat(self):
        assert classify_from_object("flat field") == "flat"

    def test_dark(self):
        assert classify_from_object("dark") == "dark"


class TestBuildInventory:
    """Tests for build_inventory and related functions."""

    def test_build_inventory_columns_and_dtypes(self, synthetic_night):
        df = build_inventory(synthetic_night, profile=None)
        assert list(df.columns) == list(INVENTORY_COLUMNS)
        # Check string columns
        for col in ("path", "frame_type", "imagetyp", "filter"):
            assert df[col].dtype == object

    def test_build_inventory_canonical_imagetyp(self, synthetic_night):
        df = build_inventory(synthetic_night, profile=None)
        valid_imagetyps = {"ZERO", "DARK", "FLAT", "OBJECT", "FOCUS"}
        for val in df["imagetyp"].dropna():
            assert val in valid_imagetyps

    def test_build_inventory_read_error_never_raises(self, synthetic_night, tmp_path):
        """Corrupt file is returned as a read_error row, not raised."""
        corrupt_path = tmp_path / "corrupt.fits"
        corrupt_path.write_bytes(b"not a fits file")

        df = build_inventory([synthetic_night, corrupt_path], profile=None)
        # Find the corrupt row
        corrupt_rows = df[df["read_error"].notna()]
        assert len(corrupt_rows) >= 1
        assert all(corrupt_rows["read_error"].str.len() > 0)

    def test_metas_from_inventory_round_trip(self, synthetic_night):
        """Reconstructed metas have identical provenance records."""
        df = build_inventory(synthetic_night, profile=None)
        metas = metas_from_inventory(df)
        assert len(metas) > 0
        # Check one meta's provenance round-trip
        meta = metas[0]
        records = list(meta.provenance_records())
        assert len(records) > 0

    def test_build_inventory_synth2_via_path_profile(self, synthetic_night_synth2):
        """Synth2 MEF can be read when profile is passed explicitly."""
        # Find the synth2.toml fixture
        synth2_toml = Path(__file__).parent / "fixtures" / "synth2.toml"
        if synth2_toml.exists():
            profile = resolve_profile(name=str(synth2_toml))
            df = build_inventory(synthetic_night_synth2, profile=profile)
            assert len(df) > 0
            assert all(df["read_error"].isna())
            assert all(df["naxis1"] > 0)

    def test_build_inventory_nprocs_matches_serial(self, synthetic_night):
        """Multiprocess inventory matches single-process."""
        df1 = build_inventory(synthetic_night, nprocs=1)
        df2 = build_inventory(synthetic_night, nprocs=2)
        # Sort by path and reset index for comparison
        df1 = df1.sort_values("path").reset_index(drop=True)
        df2 = df2.sort_values("path").reset_index(drop=True)
        pd.testing.assert_frame_equal(df1, df2)

    def test_write_inventory_csv(self, synthetic_night, tmp_path):
        """CSV round-trip preserves data."""
        df = build_inventory(synthetic_night, profile=None)
        csv_path = tmp_path / "inventory.csv"
        write_inventory(df, csv_path)
        assert csv_path.exists()
        df2 = pd.read_csv(csv_path)
        assert len(df2) == len(df)

    def test_write_inventory_parquet(self, synthetic_night, tmp_path):
        """Parquet round-trip preserves data (skip if pyarrow absent)."""
        pytest.importorskip("pyarrow")
        df = build_inventory(synthetic_night, profile=None)
        parquet_path = tmp_path / "inventory.parquet"
        write_inventory(df, parquet_path)
        assert parquet_path.exists()
        df2 = pd.read_parquet(parquet_path)
        assert len(df2) == len(df)

    def test_write_inventory_rejects_unknown_extension(self, synthetic_night, tmp_path):
        """Unknown extension raises ValueError."""
        df = build_inventory(synthetic_night, profile=None)
        bad_path = tmp_path / "inventory.txt"
        with pytest.raises(ValueError, match="unsupported inventory file extension"):
            write_inventory(df, bad_path)

    def test_filter_inventory_never_matches_bias_dark_by_filter(self, synthetic_night):
        """ZERO/DARK rows are excluded from filter filtering."""
        df = build_inventory(synthetic_night, profile=None)
        science_filters = set(
            df[df["imagetyp"] == "OBJECT"]["filter"].dropna().unique()
        )
        if science_filters:
            test_filter = science_filters.pop()
            filtered_by_filter = filter_inventory(
                df, filter=test_filter
            )
            # No ZERO rows should be in the result
            assert not any(filtered_by_filter["imagetyp"] == "ZERO")

    def test_filter_inventory_frame_type_matches_canonical_vocabulary(self, synthetic_night):
        """frame_type parameter matches canonical IMAGETYP (case-insensitive)."""
        df = build_inventory(synthetic_night, profile=None)
        # Test both uppercase and lowercase
        filtered_upper = filter_inventory(df, frame_type="OBJECT")
        filtered_lower = filter_inventory(df, frame_type="object")
        pd.testing.assert_frame_equal(filtered_upper, filtered_lower)

    def test_build_inventory_real_mini_sample(self):
        """Real mini_sample inventory has expected frame types."""
        mini_sample = Path("/home/herpich/Documents/data/ROBO43/mini_sample")
        if not mini_sample.exists():
            pytest.skip("mini_sample not found")

        df = build_inventory(mini_sample, profile=None)
        # Should have zero read_error rows
        assert df["read_error"].isna().all()
        # imagetyp should contain ZERO, FLAT, OBJECT
        imagetyps = set(df["imagetyp"].dropna().unique())
        assert "ZERO" in imagetyps
        assert "FLAT" in imagetyps
        assert "OBJECT" in imagetyps

    def test_build_inventory_truncated_fits_and_filter_skip(self, synthetic_night, tmp_path):
        """Truncated FITS is unreadable; filter_inventory keeps error row; metas skips it."""
        # Get the first FITS file from synthetic_night
        fits_files = sorted(synthetic_night.glob("*.fits"))
        assert len(fits_files) > 0

        good_file = fits_files[0]
        truncated_path = tmp_path / "truncated.fits"

        # Create a completely corrupted file (not a FITS file at all)
        truncated_path.write_bytes(b"not a fits file" * 100)

        # Build inventory with both good and truncated files
        df = build_inventory([good_file, truncated_path], profile=None)

        # Find the truncated frame's row
        truncated_rows = df[df["path"] == str(truncated_path)]
        assert len(truncated_rows) == 1
        assert truncated_rows.iloc[0]["read_error"] is not None

        # filter_inventory with default args should keep read_error rows
        df_filtered = filter_inventory(df)
        assert len(df_filtered[df_filtered["path"] == str(truncated_path)]) == 1

        # metas_from_inventory should skip it without raising
        metas = metas_from_inventory(df)
        # The good frame should be in metas, but not the truncated one
        meta_paths = [str(m.path) for m in metas]
        assert str(good_file) in meta_paths
        assert str(truncated_path) not in meta_paths
