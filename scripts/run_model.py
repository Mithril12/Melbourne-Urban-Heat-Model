"""
run_model.py
============
Trains the Melbourne Urban Heat model on data/real_melbourne_data.csv
(produced by build_dataset.py) and outputs predictions + evaluation metrics.

If real data is not yet available, falls back to the simulated CSV so the
pipeline can be exercised immediately.

Run from the repository root:
    python scripts/run_model.py [--data PATH]

Options
-------
--data PATH   Path to input CSV  (default: data/real_melbourne_data.csv,
              fallback: melbourne_heat_model.csv)
--save-map    Save the map to data/predicted_temps.png instead of displaying it
"""

import argparse
import pathlib
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; remove if you want a pop-up window

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.heat_model import (
    load_data,
    validate_data,
    create_geodataframe,
    train_model,
    predict,
    evaluate_model,
    plot_predictions,
    FEATURES,
)

REPO_ROOT = pathlib.Path(__file__).parent.parent
REAL_CSV = REPO_ROOT / "data" / "real_melbourne_data.csv"
SIMULATED_CSV = REPO_ROOT / "melbourne_heat_model.csv"
MAP_OUT = REPO_ROOT / "data" / "predicted_temps.png"


def resolve_data_path(cli_path: str | None) -> pathlib.Path:
    if cli_path:
        p = pathlib.Path(cli_path)
        if not p.exists():
            print(f"[error] Specified data file not found: {p}")
            sys.exit(1)
        return p
    if REAL_CSV.exists():
        print(f"[info]  Using real data: {REAL_CSV}")
        return REAL_CSV
    print(f"[warn]  Real data not found at {REAL_CSV}")
    print(f"        Falling back to simulated data: {SIMULATED_CSV}")
    print("        Run  python scripts/fetch_real_data.py  then")
    print("             python scripts/build_dataset.py  to get real data.\n")
    return SIMULATED_CSV


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Melbourne Urban Heat model.")
    parser.add_argument("--data", default=None, help="Path to input CSV")
    parser.add_argument("--save-map", action="store_true", help="Save map to PNG instead of displaying")
    args = parser.parse_args()

    data_path = resolve_data_path(args.data)

    # ------------------------------------------------------------------
    # 1. Load & validate
    # ------------------------------------------------------------------
    print("\n── 1. Loading data ──────────────────────────────────────────")
    df = load_data(str(data_path))
    print(f"    {len(df)} suburbs loaded from {data_path.name}")
    print(df[["Suburb", *FEATURES, "surface_temp_C"]].to_string(index=False))

    print("\n── 2. Validating data ───────────────────────────────────────")
    validate_data(df)
    print("    All validation checks passed.")

    print("\n── 3. Creating GeoDataFrame ─────────────────────────────────")
    gdf = create_geodataframe(df)
    print(f"    CRS: {gdf.crs}  |  {len(gdf)} features")

    # ------------------------------------------------------------------
    # 2. Train
    # ------------------------------------------------------------------
    print("\n── 4. Training RandomForestRegressor ────────────────────────")
    print("    Features:", FEATURES)
    # With only 15 rows (simulated) the split gives 3 test samples which is
    # too small for a stable R².  With real data (>30 suburbs) use 0.2.
    test_size = 0.2 if len(df) >= 20 else 0.33
    model, X_test, y_test = train_model(df, test_size=test_size)
    n_train = len(df) - len(X_test)
    print(f"    Train: {n_train} suburbs  |  Test: {len(X_test)} suburbs")

    # ------------------------------------------------------------------
    # 3. Evaluate
    # ------------------------------------------------------------------
    print("\n── 5. Evaluation (held-out test set) ────────────────────────")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"    R²  = {metrics['r2']:+.3f}")
    print(f"    MAE = {metrics['mae']:.2f} °C")

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\n    Feature importances:")
    for feat, imp in importances.items():
        bar = "█" * int(imp * 40)
        print(f"      {feat:<25} {imp:.3f}  {bar}")

    # ------------------------------------------------------------------
    # 4. Predict all suburbs & show results
    # ------------------------------------------------------------------
    print("\n── 6. Predictions (all suburbs) ─────────────────────────────")
    df["predicted_temp_C"] = predict(model, df)
    df["error_C"] = df["predicted_temp_C"] - df["surface_temp_C"]

    results = df[["Suburb", "surface_temp_C", "predicted_temp_C", "error_C"]].copy()
    results = results.sort_values("predicted_temp_C", ascending=False)
    print(results.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))

    hottest = results.iloc[0]["Suburb"]
    coolest = results.iloc[-1]["Suburb"]
    print(f"\n    Hottest predicted suburb:  {hottest}")
    print(f"    Coolest predicted suburb:  {coolest}")

    # ------------------------------------------------------------------
    # 5. Map
    # ------------------------------------------------------------------
    print("\n── 7. Map ───────────────────────────────────────────────────")
    gdf["predicted_temp_C"] = df["predicted_temp_C"]
    save_path = str(MAP_OUT) if args.save_map else None
    MAP_OUT.parent.mkdir(parents=True, exist_ok=True)
    plot_predictions(gdf, output_path=str(MAP_OUT))
    print(f"    Map saved to {MAP_OUT}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
