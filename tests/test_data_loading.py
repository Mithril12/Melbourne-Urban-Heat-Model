"""Tests for data loading (src.heat_model.load_data)."""

import pytest
import pandas as pd

from src.heat_model import load_data, REQUIRED_COLUMNS


class TestLoadData:
    def test_returns_dataframe(self, real_csv_path):
        df = load_data(real_csv_path)
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self, real_csv_path):
        df = load_data(real_csv_path)
        assert len(df) == 15

    def test_all_required_columns_present(self, real_csv_path):
        df = load_data(real_csv_path)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"Expected column '{col}' not found"

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_data(str(tmp_path / "nonexistent.csv"))

    def test_missing_column_raises_value_error(self, tmp_path):
        # Write a CSV that is missing the 'ndvi' column
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("Suburb,Latitude,Longitude\nFoo,-37.8,144.9\n")
        with pytest.raises(ValueError, match="ndvi"):
            load_data(str(bad_csv))

    def test_loads_correct_suburb_names(self, real_csv_path):
        df = load_data(real_csv_path)
        expected_suburbs = {
            "Footscray", "Brunswick", "Dandenong", "Sunshine", "Fitzroy",
            "Werribee", "Box Hill", "Frankston", "Coburg", "Preston",
            "St Kilda", "Carlton", "Heidelberg", "Glen Waverley", "Broadmeadows",
        }
        assert set(df["Suburb"]) == expected_suburbs

    def test_numeric_columns_have_correct_dtypes(self, real_csv_path):
        df = load_data(real_csv_path)
        numeric_cols = [
            "Latitude", "Longitude", "ndvi",
            "impervious_surface_pct", "building_density", "surface_temp_C",
        ]
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df[col]), (
                f"Column '{col}' should be numeric, got {df[col].dtype}"
            )
