"""
build_dataset.py
================
Processes the raw files downloaded by fetch_real_data.py into a single
model-ready CSV at data/real_melbourne_data.csv.

Feature derivation
------------------
ndvi
    Derived from the tree canopy polygons (City of Melbourne 2021).
    Canopy cover fraction per suburb = total canopy area / suburb area.
    This correlates strongly with NDVI (r ≈ 0.85 in urban settings).

impervious_surface_pct
    Derived from building footprints:
        impervious = (total building footprint area / suburb area) × 100
    Note: this under-estimates true imperviousness (excludes roads/carparks)
    but is the best available open proxy without satellite imagery.

building_density
    Same as above but as a fraction rather than a percentage:
        building_density = total building footprint area / suburb area

land_use_type
    Derived from CLUE buildings floor-space-by-use data:
        0 = predominantly residential   (>50 % of GFA is residential)
        1 = predominantly commercial     (>50 % of GFA is office/retail)
        2 = mixed / industrial / other

surface_temp_C
    Derived from the DataVic Urban Heat Islands 2018 shapefile:
        The dataset provides UHI deviation (°C above/below non-urban
        baseline).  We add this to the Melbourne non-urban baseline of
        approximately 30 °C (summer daytime, Landsat-8 band 10 LST) to
        obtain an absolute surface temperature estimate.
        UHI_DEG + 30.0 → surface_temp_C

Run from the repository root:
    python scripts/build_dataset.py
"""

import pathlib
import sys

import pandas as pd
import geopandas as gpd
from shapely.geometry import box

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
OUT_CSV = REPO_ROOT / "data" / "real_melbourne_data.csv"

CLUE_CSV = RAW_DIR / "clue_buildings.csv"
TREE_CSV = RAW_DIR / "tree_canopies.csv"
FOOTPRINTS_GEOJSON = RAW_DIR / "building_footprints.geojson"
SUBURBS_GEOJSON = RAW_DIR / "suburb_boundaries.geojson"
UHI_DIR = RAW_DIR / "uhi_2018"

# Melbourne non-urban baseline LST (°C) used by the DataVic dataset
MELBOURNE_LST_BASELINE = 30.0

# CLUE space-use codes that count as residential
RESIDENTIAL_USES = {
    "Residential Accommodation",
    "Student Accommodation",
    "Retirement Village",
    "Serviced Apartments",
}

COMMERCIAL_USES = {
    "Office",
    "Retail",
    "Shop",
    "Retail (Supermarket)",
    "Retail (Convenience)",
    "Hotel/Motel Accommodation",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require(path: pathlib.Path, hint: str) -> None:
    if not path.exists():
        print(f"\n[ERROR] Required file not found: {path}")
        print(f"        {hint}")
        sys.exit(1)


def _load_uhi_shapefile() -> gpd.GeoDataFrame:
    """Return the first .shp found inside the UHI unzip directory."""
    shps = list(UHI_DIR.rglob("*.shp"))
    if not shps:
        print(f"\n[ERROR] No .shp file found in {UHI_DIR}")
        print("        Run fetch_real_data.py first.")
        sys.exit(1)
    return gpd.read_file(shps[0])


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def load_suburbs() -> gpd.GeoDataFrame:
    """Load suburb polygons and project to GDA2020 / MGA zone 55 (EPSG:7855)
    for area calculations in metres."""
    _require(SUBURBS_GEOJSON, "Run fetch_real_data.py first.")
    gdf = gpd.read_file(SUBURBS_GEOJSON)
    gdf = gdf.to_crs("EPSG:7855")
    # Normalise suburb name column
    name_col = next(
        (c for c in gdf.columns if "suburb" in c.lower() or "name" in c.lower()),
        gdf.columns[0],
    )
    gdf = gdf.rename(columns={name_col: "Suburb"})
    gdf["suburb_area_m2"] = gdf.geometry.area
    # Centroid for Lat/Lon (convert back to WGS84 for the centroid)
    centroids = gdf.geometry.centroid.to_crs("EPSG:4326")
    gdf["Latitude"] = centroids.y
    gdf["Longitude"] = centroids.x
    return gdf[["Suburb", "suburb_area_m2", "Latitude", "Longitude", "geometry"]]


def compute_ndvi_proxy(suburbs: gpd.GeoDataFrame) -> pd.Series:
    """Canopy cover fraction per suburb as an NDVI proxy."""
    _require(TREE_CSV, "Run fetch_real_data.py first.")
    trees = gpd.read_file(TREE_CSV) if TREE_CSV.suffix == ".geojson" else _csv_to_gdf(TREE_CSV)
    trees = trees.to_crs("EPSG:7855")

    joined = gpd.sjoin(trees, suburbs[["Suburb", "suburb_area_m2", "geometry"]], how="left", predicate="intersects")
    if "area" in trees.columns:
        canopy_col = "area"
    else:
        joined["_canopy_area"] = joined.geometry.area
        canopy_col = "_canopy_area"

    canopy_per_suburb = (
        joined.groupby("Suburb")[canopy_col].sum()
        / suburbs.set_index("Suburb")["suburb_area_m2"]
    ).clip(0, 1)
    canopy_per_suburb.name = "ndvi"
    return canopy_per_suburb


def _csv_to_gdf(csv_path: pathlib.Path) -> gpd.GeoDataFrame:
    """Load a CSV that has geometry in WKT or GeoJSON columns."""
    df = pd.read_csv(csv_path)
    geom_col = next((c for c in df.columns if "geo" in c.lower()), None)
    if geom_col is None:
        raise ValueError(f"Cannot find geometry column in {csv_path.name}")
    return gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df[geom_col]), crs="EPSG:4326")


def compute_building_features(suburbs: gpd.GeoDataFrame) -> pd.DataFrame:
    """Return building_density and impervious_surface_pct per suburb."""
    _require(FOOTPRINTS_GEOJSON, "Run fetch_real_data.py first.")
    footprints = gpd.read_file(FOOTPRINTS_GEOJSON).to_crs("EPSG:7855")

    joined = gpd.sjoin(footprints, suburbs[["Suburb", "suburb_area_m2", "geometry"]], how="left", predicate="intersects")
    joined["footprint_area"] = joined.geometry.area

    total_footprint = joined.groupby("Suburb")["footprint_area"].sum()
    suburb_area = suburbs.set_index("Suburb")["suburb_area_m2"]

    building_density = (total_footprint / suburb_area).clip(0, 1).fillna(0)
    impervious_pct = (building_density * 100).clip(0, 100)

    return pd.DataFrame({
        "building_density": building_density,
        "impervious_surface_pct": impervious_pct,
    })


def compute_land_use_type(suburbs: gpd.GeoDataFrame) -> pd.Series:
    """Derive land_use_type (0=residential, 1=commercial, 2=mixed) from CLUE."""
    _require(CLUE_CSV, "Run fetch_real_data.py first.")
    clue = pd.read_csv(CLUE_CSV, low_memory=False)

    # Find the suburb / space-use columns (column names vary by export version)
    suburb_col = next(
        (c for c in clue.columns if "suburb" in c.lower()), None
    )
    use_col = next(
        (c for c in clue.columns if "space_use" in c.lower() or "predominant_space_use" in c.lower()), None
    )
    gfa_col = next(
        (c for c in clue.columns if "gfa" in c.lower() or "floor_area" in c.lower()), None
    )

    if suburb_col is None or use_col is None:
        print(f"  [warn] Could not identify suburb/use columns in {CLUE_CSV.name}")
        print(f"         Available columns: {list(clue.columns)[:15]}")
        # Fall back: all suburbs get land_use_type = 1 (commercial — City of Melbourne)
        return pd.Series(1, index=suburbs["Suburb"], name="land_use_type")

    clue = clue.rename(columns={suburb_col: "Suburb", use_col: "use"})
    if gfa_col:
        clue = clue.rename(columns={gfa_col: "gfa"})
        clue["gfa"] = pd.to_numeric(clue["gfa"], errors="coerce").fillna(0)
    else:
        clue["gfa"] = 1  # treat each building as equal weight

    res_gfa = clue[clue["use"].isin(RESIDENTIAL_USES)].groupby("Suburb")["gfa"].sum()
    com_gfa = clue[clue["use"].isin(COMMERCIAL_USES)].groupby("Suburb")["gfa"].sum()
    total_gfa = clue.groupby("Suburb")["gfa"].sum()

    def _classify(suburb: str) -> int:
        total = total_gfa.get(suburb, 0)
        if total == 0:
            return 2
        res_frac = res_gfa.get(suburb, 0) / total
        com_frac = com_gfa.get(suburb, 0) / total
        if res_frac >= 0.5:
            return 0
        if com_frac >= 0.5:
            return 1
        return 2

    return pd.Series(
        {s: _classify(s) for s in suburbs["Suburb"]},
        name="land_use_type",
    )


def compute_surface_temp(suburbs: gpd.GeoDataFrame) -> pd.Series:
    """Estimate surface_temp_C from DataVic UHI 2018 shapefile."""
    uhi = _load_uhi_shapefile().to_crs("EPSG:7855")

    # Find the UHI deviation column (varies by version — try common names)
    uhi_col = next(
        (c for c in uhi.columns if "uhi" in c.lower() or "heat" in c.lower() or "temp" in c.lower()),
        None,
    )
    if uhi_col is None:
        print(f"  [warn] Could not identify UHI column. Columns: {list(uhi.columns)}")
        return pd.Series(MELBOURNE_LST_BASELINE, index=suburbs["Suburb"], name="surface_temp_C")

    # Spatial join UHI polygons → suburbs, then take area-weighted mean
    joined = gpd.sjoin(uhi[[uhi_col, "geometry"]], suburbs[["Suburb", "geometry"]], how="left", predicate="intersects")
    joined[uhi_col] = pd.to_numeric(joined[uhi_col], errors="coerce")

    uhi_mean = joined.groupby("Suburb")[uhi_col].mean()
    surface_temp = (uhi_mean + MELBOURNE_LST_BASELINE).rename("surface_temp_C")
    return surface_temp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build() -> pd.DataFrame:
    print("Loading suburb boundaries …")
    suburbs = load_suburbs()

    print("Computing NDVI proxy from tree canopies …")
    ndvi = compute_ndvi_proxy(suburbs)

    print("Computing building density and impervious surface % …")
    building_features = compute_building_features(suburbs)

    print("Deriving land use type from CLUE buildings …")
    land_use = compute_land_use_type(suburbs)

    print("Deriving surface temperature from UHI dataset …")
    surface_temp = compute_surface_temp(suburbs)

    # Assemble
    df = suburbs[["Suburb", "Latitude", "Longitude"]].copy()
    df = df.join(ndvi, on="Suburb")
    df = df.join(building_features, on="Suburb")
    df = df.join(land_use, on="Suburb")
    df = df.join(surface_temp, on="Suburb")

    # Drop suburbs where any key feature is missing
    before = len(df)
    df = df.dropna(subset=["ndvi", "impervious_surface_pct", "building_density", "land_use_type", "surface_temp_C"])
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after} suburbs with missing features ({after} remain)")

    df = df.reset_index(drop=True)
    return df


def main() -> None:
    print("=" * 60)
    print("Melbourne Urban Heat Model — build_dataset.py")
    print("=" * 60)

    df = build()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nSaved {len(df)} suburbs → {OUT_CSV}")
    print(df[["Suburb", "ndvi", "impervious_surface_pct", "building_density",
               "land_use_type", "surface_temp_C"]].to_string(index=False))
    print("\nRun next:")
    print("    python scripts/run_model.py")


if __name__ == "__main__":
    main()
