# =============================================================
# src/features.py
# GujEstateAI — Phase 3: Feature Engineering
# =============================================================
# Same logic as 03_feature_engineering.ipynb
# but as a reusable function.
#
# Usage:
#   from src.features import build_features
#   df = build_features("data/processed/cleaned.csv")
#
# Or run directly:
#   python src/features.py
# =============================================================

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder


# =============================================================
# FEATURE SETS — used by all ML modules
# =============================================================

FEATURES_DURATION = [
    "projectType_enc",
    "distName_enc",
    "promoter_type_simple_enc",
    "totalUnits",
    "log_cost",
    "totalLandCost",
    "is_redevelop",
    "startProjectYear",
    "start_quarter",
    "land_cost_ratio",
    "avgCostPerUnit",
]

FEATURES_COST = [
    "projectType_enc",
    "distName_enc",
    "promoter_type_simple_enc",
    "totalUnits",
    "noOfInventory",
    "avgUnits",
    "totalLandCost",
    "totalCarpetArea_form3A",
    "totalBuiltupArea_form3A",
    "totalSquareFootBuild",
    "AvgSquareFootBuild",
    "booking_rate",
    "log_units",
    "is_redevelop",
    "startProjectYear",
    "startProjectMonth",
    "start_quarter",
    "duration_months",
]

FEATURES_CLUSTER = [
    "log_cost",
    "log_units",
    "duration_months",
    "avgCostPerSqFt",
    "booking_rate",
    "land_cost_ratio",
    "projectType_enc",
    "distName_enc",
]

FEATURES_ANOMALY = [
    "totalEstimatedCost",
    "avgCostPerSqFt",
    "duration_months",
    "booking_rate",
    "totalLandCost",
    "land_cost_ratio",
    "totalUnits",
    "cost_per_unit",
]


# =============================================================
# HELPER: simplify promoter type into 4 groups
# =============================================================

def simplify_promoter(pt):
    pt = str(pt).upper()
    if "PARTNERSHIP" in pt:
        return "Partnership"
    elif "COMPANY" in pt or "LLP" in pt or "LIABILITY" in pt:
        return "Company"
    elif "INDIVIDUAL" in pt or "PROPRIET" in pt:
        return "Individual"
    else:
        return "Other"


# =============================================================
# MAIN FUNCTION
# =============================================================

def build_features(input_path, output_path=None, encoders_path="models/encoders.pkl"):
    """
    Load cleaned CSV, engineer all features, encode categoricals.

    Parameters:
        input_path    : path to cleaned.csv
        output_path   : (optional) path to save features.csv
        encoders_path : path to save/load label encoders

    Returns:
        df       : DataFrame with all features added
        encoders : dict of fitted LabelEncoders
    """

    print("=" * 50)
    print("  GujEstateAI — Feature Engineering")
    print("=" * 50)

    # ----------------------------------------------------------
    # Load cleaned data
    # ----------------------------------------------------------
    df = pd.read_csv(input_path)
    print(f"\nLoaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # ----------------------------------------------------------
    # Feature 1: Log transforms
    # ----------------------------------------------------------
    df["log_cost"]  = np.log1p(df["totalEstimatedCost"])
    df["log_units"] = np.log1p(df["totalUnits"])
    print("\nCreated: log_cost, log_units")

    # ----------------------------------------------------------
    # Feature 2: Cost per unit
    # ----------------------------------------------------------
    df["cost_per_unit"] = df["totalEstimatedCost"] / df["totalUnits"]
    df["cost_per_unit"] = df["cost_per_unit"].replace([np.inf, -np.inf], np.nan)
    df["cost_per_unit"] = df["cost_per_unit"].fillna(df["cost_per_unit"].median())
    print("Created: cost_per_unit")

    # ----------------------------------------------------------
    # Feature 3: Land cost ratio
    # ----------------------------------------------------------
    df["land_cost_ratio"] = df["totalLandCost"] / df["totalEstimatedCost"]
    df["land_cost_ratio"] = df["land_cost_ratio"].clip(0, 1)
    print("Created: land_cost_ratio")

    # ----------------------------------------------------------
    # Feature 4: Sell to develop ratio
    # ----------------------------------------------------------
    df["sell_dev_ratio"] = df["totalSellingAmount"] / df["totalDevelopCost"]
    df["sell_dev_ratio"] = df["sell_dev_ratio"].replace([np.inf, -np.inf], np.nan)
    df["sell_dev_ratio"] = df["sell_dev_ratio"].fillna(df["sell_dev_ratio"].median())
    upper = df["sell_dev_ratio"].quantile(0.99)
    df["sell_dev_ratio"] = df["sell_dev_ratio"].clip(0, upper)
    print("Created: sell_dev_ratio")

    # ----------------------------------------------------------
    # Feature 5: Is redevelopment (binary)
    # ----------------------------------------------------------
    df["is_redevelop"] = (df["underRedevelopment"] == "YES").astype(int)
    print("Created: is_redevelop")

    # ----------------------------------------------------------
    # Feature 6: Start quarter
    # ----------------------------------------------------------
    df["start_quarter"] = df["startProjectMonth"].apply(
        lambda m: int((int(m) - 1) // 3 + 1)
    )
    print("Created: start_quarter")

    # ----------------------------------------------------------
    # Feature 7: Simplified promoter type
    # ----------------------------------------------------------
    df["promoter_type_simple"] = df["promoterType"].apply(simplify_promoter)
    print("Created: promoter_type_simple")

    # ----------------------------------------------------------
    # Encode categoricals
    # ----------------------------------------------------------
    encoders = {}
    cat_cols = ["projectType", "distName", "promoter_type_simple"]

    os.makedirs(os.path.dirname(encoders_path) if os.path.dirname(encoders_path) else ".", exist_ok=True)

    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col]    = le

    joblib.dump(encoders, encoders_path)
    print(f"\nCategoricals encoded: {cat_cols}")
    print(f"Encoders saved to   : {encoders_path}")

    # ----------------------------------------------------------
    # Reset index
    # ----------------------------------------------------------
    df.reset_index(drop=True, inplace=True)

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    new_cols = [
        "log_cost", "log_units", "cost_per_unit", "land_cost_ratio",
        "sell_dev_ratio", "is_redevelop", "start_quarter",
        "promoter_type_simple", "projectType_enc",
        "distName_enc", "promoter_type_simple_enc"
    ]
    print(f"\n{'=' * 50}")
    print(f"  FEATURE ENGINEERING COMPLETE")
    print(f"{'=' * 50}")
    print(f"  Rows              : {df.shape[0]:,}")
    print(f"  Total columns     : {df.shape[1]}")
    print(f"  New features added: {len(new_cols)}")

    # ----------------------------------------------------------
    # Save (optional)
    # ----------------------------------------------------------
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"  Saved to          : {output_path}")

    return df, encoders


# =============================================================
# Run directly: python src/features.py
# =============================================================
if __name__ == "__main__":
    df, encoders = build_features(
        input_path    = "data/processed/cleaned.csv",
        output_path   = "data/processed/features.csv",
        encoders_path = "models/encoders.pkl"
    )
    print(f"\nDone! Shape: {df.shape}")
    print("\nFeature sets ready for ML modules:")
    print(f"  FEATURES_DURATION : {len(FEATURES_DURATION)} features")
    print(f"  FEATURES_COST     : {len(FEATURES_COST)} features")
    print(f"  FEATURES_CLUSTER  : {len(FEATURES_CLUSTER)} features")
    print(f"  FEATURES_ANOMALY  : {len(FEATURES_ANOMALY)} features")