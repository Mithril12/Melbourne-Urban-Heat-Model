"""Shared pytest fixtures for Melbourne Urban Heat Model tests."""

import os
import pytest
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(REPO_ROOT, "melbourne_heat_model.csv")


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def real_csv_path():
    """Path to the real CSV shipped with the repository."""
    return CSV_PATH


@pytest.fixture(scope="session")
def real_df():
    """DataFrame loaded directly from the real CSV."""
    return pd.read_csv(CSV_PATH)


@pytest.fixture
def valid_df():
    """Minimal synthetic DataFrame that passes all validation rules."""
    return pd.DataFrame(
        {
            "Suburb": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
            "Latitude": [-37.80, -37.77, -37.98, -37.78, -37.79],
            "Longitude": [144.90, 144.96, 145.22, 144.83, 144.98],
            "ndvi": [0.36, 0.77, 0.61, 0.52, 0.21],
            "impervious_surface_pct": [49.2, 55.2, 66.2, 61.6, 54.6],
            "building_density": [0.59, 0.31, 0.24, 0.82, 0.83],
            "land_use_type": [1, 1, 1, 1, 0],
            "surface_temp_C": [51.0, 42.6, 47.9, 52.6, 55.7],
        }
    )


@pytest.fixture
def df_missing_column(valid_df):
    """DataFrame with the ndvi column removed."""
    return valid_df.drop(columns=["ndvi"])


@pytest.fixture
def df_with_nulls(valid_df):
    """DataFrame that has a NaN injected into a feature column."""
    df = valid_df.copy()
    df.loc[0, "ndvi"] = np.nan
    return df


@pytest.fixture
def df_invalid_ndvi(valid_df):
    """DataFrame with an out-of-range NDVI value."""
    df = valid_df.copy()
    df.loc[0, "ndvi"] = 1.5
    return df


@pytest.fixture
def df_invalid_impervious(valid_df):
    """DataFrame with an out-of-range impervious_surface_pct value."""
    df = valid_df.copy()
    df.loc[0, "impervious_surface_pct"] = 110.0
    return df


@pytest.fixture
def df_invalid_building_density(valid_df):
    """DataFrame with a building_density value above 1."""
    df = valid_df.copy()
    df.loc[0, "building_density"] = 1.5
    return df


@pytest.fixture
def df_invalid_land_use(valid_df):
    """DataFrame with an unknown land_use_type value."""
    df = valid_df.copy()
    df.loc[0, "land_use_type"] = 99
    return df


@pytest.fixture
def df_out_of_bbox(valid_df):
    """DataFrame with a coordinate outside Melbourne's bounding box."""
    df = valid_df.copy()
    df.loc[0, "Latitude"] = -33.87   # Sydney
    df.loc[0, "Longitude"] = 151.21
    return df
