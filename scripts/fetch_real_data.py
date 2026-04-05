"""
fetch_real_data.py
==================
Downloads all required raw data from publicly accessible open-data sources
(no login or API key required) and saves them to data/raw/.

Run from the repository root:
    python scripts/fetch_real_data.py

Sources
-------
1. City of Melbourne CLUE Buildings (land use type + building density)
   https://data.melbourne.vic.gov.au/
2. City of Melbourne Tree Canopies 2021 (NDVI proxy via canopy cover)
   https://data.melbourne.vic.gov.au/
3. City of Melbourne Suburb Boundaries (GeoJSON)
   https://data.melbourne.vic.gov.au/
4. Victorian Government DataVic — Urban Heat Islands 2018 (surface_temp_C proxy)
   https://discover.data.vic.gov.au/
5. ABS SAL (Suburbs and Localities) boundary file — used to expand beyond
   the inner-city LGA to Greater Melbourne where possible
   https://www.abs.gov.au/
"""

import os
import sys
import pathlib
import urllib.request
import urllib.error
import zipfile
import json

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download(url: str, dest: pathlib.Path, desc: str) -> bool:
    """Download *url* to *dest*.  Returns True on success."""
    if dest.exists():
        print(f"  [skip] {desc} — already exists at {dest.name}")
        return True
    print(f"  [fetch] {desc}")
    print(f"          {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as fh:
            fh.write(resp.read())
        size_kb = dest.stat().st_size // 1024
        print(f"          saved {dest.name} ({size_kb} KB)")
        return True
    except urllib.error.HTTPError as exc:
        print(f"  [error] HTTP {exc.code}: {exc.reason}")
        return False
    except Exception as exc:
        print(f"  [error] {exc}")
        return False


def unzip(archive: pathlib.Path, dest_dir: pathlib.Path) -> None:
    """Extract a zip archive into dest_dir."""
    print(f"  [unzip] {archive.name} → {dest_dir.name}/")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest_dir)


# ---------------------------------------------------------------------------
# Download tasks
# ---------------------------------------------------------------------------

DATASETS = [
    # --- City of Melbourne Open Data (CC BY 4.0, no login) -----------------
    {
        "id": "clue_buildings",
        "desc": "CLUE Buildings (land use, floors, GFA) — City of Melbourne",
        "url": (
            "https://data.melbourne.vic.gov.au/api/v2/catalog/datasets/"
            "buildings-with-name-age-size-accessibility-and-bicycle-facilities/"
            "exports/csv?limit=-1&timezone=Australia%2FMelbourne"
        ),
        "dest": RAW_DIR / "clue_buildings.csv",
    },
    {
        "id": "tree_canopies",
        "desc": "Tree Canopies 2021 (canopy cover, NDVI proxy) — City of Melbourne",
        "url": (
            "https://data.melbourne.vic.gov.au/api/v2/catalog/datasets/"
            "tree-canopies-2021-urban-forest/exports/csv?limit=-1"
        ),
        "dest": RAW_DIR / "tree_canopies.csv",
    },
    {
        "id": "suburb_boundaries",
        "desc": "Suburb Boundaries (GeoJSON) — City of Melbourne",
        "url": (
            "https://data.melbourne.vic.gov.au/api/v2/catalog/datasets/"
            "suburb-boundaries/exports/geojson"
        ),
        "dest": RAW_DIR / "suburb_boundaries.geojson",
    },
    {
        "id": "building_footprints",
        "desc": "2023 Building Footprints (GeoJSON) — City of Melbourne",
        "url": (
            "https://data.melbourne.vic.gov.au/api/v2/catalog/datasets/"
            "2023-building-footprints/exports/geojson"
        ),
        "dest": RAW_DIR / "building_footprints.geojson",
    },
    # --- DataVic — Urban Heat Islands 2018 (CC BY 4.0, no login) -----------
    # The Shapefile download is a zip; we unpack it after downloading.
    {
        "id": "uhi_2018",
        "desc": "Urban Heat Islands & Vegetation 2018 (SHP zip) — DataVic",
        "url": (
            "https://s3.ap-southeast-2.amazonaws.com/cl-isd-prod-datashare-s3-delivery/"
            "Group=Open/Category=Environment/Dataset=Metropolitan%20Melbourne%20Urban"
            "%20Heat%20Islands%20and%20Urban%20Vegetation%202018/"
            "MetroMelb_UHI_Veg_2018.zip"
        ),
        "dest": RAW_DIR / "uhi_2018.zip",
        "unzip_to": RAW_DIR / "uhi_2018",
    },
]


def main() -> None:
    print("=" * 60)
    print("Melbourne Urban Heat Model — fetch_real_data.py")
    print("=" * 60)

    ok_count = 0
    fail_count = 0

    for ds in DATASETS:
        ok = download(ds["url"], ds["dest"], ds["desc"])
        if ok and "unzip_to" in ds:
            unzip_dir: pathlib.Path = ds["unzip_to"]
            unzip_dir.mkdir(parents=True, exist_ok=True)
            unzip(ds["dest"], unzip_dir)
        if ok:
            ok_count += 1
        else:
            fail_count += 1

    print()
    print(f"Done: {ok_count} succeeded, {fail_count} failed.")
    if fail_count:
        print(
            "\nFor failed downloads, visit the dataset pages manually and save "
            "the files to data/raw/ with the filenames shown above.\n"
            "  City of Melbourne Open Data:  https://data.melbourne.vic.gov.au/\n"
            "  DataVic:                      https://discover.data.vic.gov.au/"
        )
        sys.exit(1)

    print("\nAll raw files are in data/raw/.  Run next:")
    print("    python scripts/build_dataset.py")


if __name__ == "__main__":
    main()
