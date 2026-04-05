"""Tests for data validation (src.heat_model.validate_data)."""

import pytest

from src.heat_model import validate_data, MELBOURNE_BBOX


class TestValidateData:
    # ------------------------------------------------------------------
    # Happy-path
    # ------------------------------------------------------------------

    def test_valid_df_does_not_raise(self, valid_df):
        validate_data(valid_df)  # should not raise

    def test_real_csv_passes_validation(self, real_df):
        validate_data(real_df)  # should not raise

    # ------------------------------------------------------------------
    # Null / missing values
    # ------------------------------------------------------------------

    def test_null_in_feature_raises(self, df_with_nulls):
        with pytest.raises(ValueError, match="Null values"):
            validate_data(df_with_nulls)

    # ------------------------------------------------------------------
    # NDVI
    # ------------------------------------------------------------------

    def test_ndvi_above_1_raises(self, df_invalid_ndvi):
        with pytest.raises(ValueError, match="ndvi"):
            validate_data(df_invalid_ndvi)

    def test_ndvi_below_minus_1_raises(self, valid_df):
        df = valid_df.copy()
        df.loc[0, "ndvi"] = -1.1
        with pytest.raises(ValueError, match="ndvi"):
            validate_data(df)

    def test_ndvi_at_boundary_values_passes(self, valid_df):
        df = valid_df.copy()
        df.loc[0, "ndvi"] = -1.0
        df.loc[1, "ndvi"] = 1.0
        validate_data(df)  # should not raise

    # ------------------------------------------------------------------
    # Impervious surface %
    # ------------------------------------------------------------------

    def test_impervious_above_100_raises(self, df_invalid_impervious):
        with pytest.raises(ValueError, match="impervious_surface_pct"):
            validate_data(df_invalid_impervious)

    def test_impervious_below_0_raises(self, valid_df):
        df = valid_df.copy()
        df.loc[0, "impervious_surface_pct"] = -5.0
        with pytest.raises(ValueError, match="impervious_surface_pct"):
            validate_data(df)

    # ------------------------------------------------------------------
    # Building density
    # ------------------------------------------------------------------

    def test_building_density_above_1_raises(self, df_invalid_building_density):
        with pytest.raises(ValueError, match="building_density"):
            validate_data(df_invalid_building_density)

    def test_building_density_below_0_raises(self, valid_df):
        df = valid_df.copy()
        df.loc[0, "building_density"] = -0.1
        with pytest.raises(ValueError, match="building_density"):
            validate_data(df)

    # ------------------------------------------------------------------
    # Land use type
    # ------------------------------------------------------------------

    def test_unknown_land_use_type_raises(self, df_invalid_land_use):
        with pytest.raises(ValueError, match="land_use_type"):
            validate_data(df_invalid_land_use)

    def test_all_valid_land_use_types_pass(self, valid_df):
        df = valid_df.copy()
        df["land_use_type"] = [0, 1, 2, 0, 1]
        validate_data(df)  # should not raise

    # ------------------------------------------------------------------
    # Coordinates / bounding box
    # ------------------------------------------------------------------

    def test_latitude_outside_bbox_raises(self, df_out_of_bbox):
        with pytest.raises(ValueError, match="Latitude"):
            validate_data(df_out_of_bbox)

    def test_longitude_outside_bbox_raises(self, valid_df):
        df = valid_df.copy()
        df.loc[0, "Longitude"] = 151.21  # Sydney longitude
        with pytest.raises(ValueError, match="Longitude"):
            validate_data(df)

    def test_real_data_coords_within_melbourne_bbox(self, real_df):
        assert real_df["Latitude"].between(
            MELBOURNE_BBOX["lat_min"], MELBOURNE_BBOX["lat_max"]
        ).all()
        assert real_df["Longitude"].between(
            MELBOURNE_BBOX["lon_min"], MELBOURNE_BBOX["lon_max"]
        ).all()
