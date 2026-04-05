"""
Melbourne Urban Heat Model

Functions for loading, validating, modelling, and visualising
surface temperature data across Melbourne suburbs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


REQUIRED_COLUMNS = [
    "Suburb",
    "Latitude",
    "Longitude",
    "ndvi",
    "impervious_surface_pct",
    "building_density",
    "land_use_type",
    "surface_temp_C",
]

FEATURES = ["ndvi", "impervious_surface_pct", "building_density", "land_use_type"]

TARGET = "surface_temp_C"

# Approximate bounding box for Greater Melbourne
MELBOURNE_BBOX = {
    "lat_min": -38.5,
    "lat_max": -37.5,
    "lon_min": 144.5,
    "lon_max": 145.6,
}

VALID_LAND_USE_TYPES = {0, 1, 2}

# Plausible surface temperature range for Melbourne (°C)
TEMP_MIN = 20.0
TEMP_MAX = 80.0


def load_data(filepath: str) -> pd.DataFrame:
    """Load suburb data from a CSV file and validate column presence.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with suburb heat data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(filepath)
    _check_required_columns(df)
    return df


def _check_required_columns(df: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")


def validate_data(df: pd.DataFrame) -> None:
    """Validate feature values are within expected ranges.

    Args:
        df: DataFrame returned by load_data().

    Raises:
        ValueError: On the first validation failure found.
    """
    if df[FEATURES + [TARGET]].isnull().any().any():
        null_cols = df[FEATURES + [TARGET]].columns[df[FEATURES + [TARGET]].isnull().any()].tolist()
        raise ValueError(f"Null values found in columns: {null_cols}")

    if not df["ndvi"].between(-1.0, 1.0).all():
        raise ValueError("ndvi values must be in the range [-1, 1]")

    if not df["impervious_surface_pct"].between(0.0, 100.0).all():
        raise ValueError("impervious_surface_pct values must be in the range [0, 100]")

    if not df["building_density"].between(0.0, 1.0).all():
        raise ValueError("building_density values must be in the range [0, 1]")

    invalid_lu = set(df["land_use_type"].unique()) - VALID_LAND_USE_TYPES
    if invalid_lu:
        raise ValueError(f"Unexpected land_use_type values: {sorted(invalid_lu)}")

    if not df["Latitude"].between(MELBOURNE_BBOX["lat_min"], MELBOURNE_BBOX["lat_max"]).all():
        raise ValueError(
            f"Latitude values must be in [{MELBOURNE_BBOX['lat_min']}, {MELBOURNE_BBOX['lat_max']}]"
        )

    if not df["Longitude"].between(MELBOURNE_BBOX["lon_min"], MELBOURNE_BBOX["lon_max"]).all():
        raise ValueError(
            f"Longitude values must be in [{MELBOURNE_BBOX['lon_min']}, {MELBOURNE_BBOX['lon_max']}]"
        )


def create_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert a suburb DataFrame to a GeoDataFrame using WGS-84 coordinates.

    Args:
        df: DataFrame with Latitude and Longitude columns.

    Returns:
        GeoDataFrame with Point geometry and CRS EPSG:4326.
    """
    df = df.copy()
    df["geometry"] = [Point(lon, lat) for lon, lat in zip(df["Longitude"], df["Latitude"])]
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


def train_model(
    df: pd.DataFrame,
    features: list = None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
) -> tuple:
    """Train a RandomForestRegressor with a held-out test split.

    Args:
        df: DataFrame containing feature and target columns.
        features: List of feature column names (defaults to FEATURES).
        test_size: Fraction of data to reserve for testing.
        random_state: Random seed for reproducibility.
        n_estimators: Number of trees in the forest.

    Returns:
        Tuple of (fitted model, X_test, y_test).
    """
    if features is None:
        features = FEATURES

    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    return model, X_test, y_test


def predict(model: RandomForestRegressor, df: pd.DataFrame, features: list = None) -> np.ndarray:
    """Generate surface temperature predictions for all rows.

    Args:
        model: A fitted RandomForestRegressor.
        df: DataFrame with feature columns.
        features: List of feature column names (defaults to FEATURES).

    Returns:
        Array of predicted temperatures in °C.
    """
    if features is None:
        features = FEATURES
    return model.predict(df[features])


def evaluate_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute evaluation metrics on the held-out test set.

    Args:
        model: A fitted RandomForestRegressor.
        X_test: Test feature matrix.
        y_test: True target values.

    Returns:
        Dict with keys 'r2' and 'mae'.
    """
    preds = model.predict(X_test)
    return {
        "r2": r2_score(y_test, preds),
        "mae": float(np.mean(np.abs(preds - y_test))),
    }


def plot_predictions(gdf: gpd.GeoDataFrame, output_path: str = None):
    """Plot predicted surface temperatures on a map.

    Args:
        gdf: GeoDataFrame with a predicted_temp_C column.
        output_path: If provided, save the figure here instead of displaying it.

    Returns:
        Tuple of (fig, ax).
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(
        column="predicted_temp_C",
        ax=ax,
        legend=True,
        cmap="inferno",
        edgecolor="black",
        markersize=100,
    )

    for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf["Suburb"]):
        ax.text(x + 0.005, y, label, fontsize=9)

    plt.title("Predicted Urban Surface Temperature by Suburb \u2013 Melbourne")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    return fig, ax
