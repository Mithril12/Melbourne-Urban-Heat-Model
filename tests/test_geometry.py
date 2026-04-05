"""Tests for GeoDataFrame creation (src.heat_model.create_geodataframe)."""

import pytest
import geopandas as gpd
from shapely.geometry import Point

from src.heat_model import create_geodataframe


class TestCreateGeoDataFrame:
    def test_returns_geodataframe(self, valid_df):
        gdf = create_geodataframe(valid_df)
        assert isinstance(gdf, gpd.GeoDataFrame)

    def test_row_count_preserved(self, valid_df):
        gdf = create_geodataframe(valid_df)
        assert len(gdf) == len(valid_df)

    def test_crs_is_wgs84(self, valid_df):
        gdf = create_geodataframe(valid_df)
        assert gdf.crs is not None
        assert gdf.crs.to_epsg() == 4326

    def test_geometry_column_exists(self, valid_df):
        gdf = create_geodataframe(valid_df)
        assert "geometry" in gdf.columns

    def test_all_geometries_are_points(self, valid_df):
        gdf = create_geodataframe(valid_df)
        for geom in gdf.geometry:
            assert isinstance(geom, Point), f"Expected Point, got {type(geom)}"

    def test_longitude_maps_to_geometry_x(self, valid_df):
        gdf = create_geodataframe(valid_df)
        for _, row in gdf.iterrows():
            assert row.geometry.x == pytest.approx(row["Longitude"])

    def test_latitude_maps_to_geometry_y(self, valid_df):
        gdf = create_geodataframe(valid_df)
        for _, row in gdf.iterrows():
            assert row.geometry.y == pytest.approx(row["Latitude"])

    def test_original_dataframe_not_mutated(self, valid_df):
        original_columns = set(valid_df.columns)
        create_geodataframe(valid_df)
        assert set(valid_df.columns) == original_columns

    def test_all_original_columns_retained(self, valid_df):
        gdf = create_geodataframe(valid_df)
        for col in valid_df.columns:
            assert col in gdf.columns

    def test_real_csv_creates_valid_geodataframe(self, real_df):
        gdf = create_geodataframe(real_df)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert gdf.crs.to_epsg() == 4326
        assert len(gdf) == 15
        assert all(isinstance(g, Point) for g in gdf.geometry)
