# =============================================================
# src/features.py
# GujEstateAI — Phase 3: Feature Engineering
# =============================================================
# Usage:
#   from src.features import build_features
#   df, encoders = build_features("data/processed/cleaned.csv")
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
# FEATURE SETS — must match what each saved .pkl was trained on
# =============================================================

FEATURES_DURATION = [
    "projectType_enc", "distName_enc", "promoter_type_simple_enc",
    "totalUnits", "log_units", "noOfInventory", "totalProjects",
    "log_cost", "log_land_cost", "log_develop_cost", "log_selling",
    "log_carpet", "log_buildup", "log_sqft_build", "log_cost_per_unit",
    "cost_per_unit", "avgCostPerSqFt", "avgCostPerUnit",
    "land_cost_ratio", "sell_dev_ratio", "booking_rate",
    "is_redevelop", "startProjectYear", "start_month", "start_quarter",
    "dist_avg_duration", "dist_median_duration", "dist_project_count", "dist_avg_cost",
    "type_avg_duration", "type_median_duration",
    "year_avg_duration", "year_project_count",
]

FEATURES_COST = [
    "projectType_enc", "distName_enc", "promoter_type_simple_enc",
    "totalUnits", "log_units", "noOfInventory", "totalProjects",
    "log_land_cost", "log_develop_cost", "log_selling",
    "log_carpet", "log_buildup", "log_sqft_build", "log_cost_per_unit",
    "cost_per_unit", "avgCostPerSqFt", "avgCostPerUnit",
    "land_cost_ratio", "sell_dev_ratio", "booking_rate",
    "is_redevelop", "startProjectYear", "start_month", "start_quarter",
    "duration_months",
    "dist_avg_duration", "dist_project_count",
    "dist_avg_cost_feat", "dist_median_cost_feat",
    "type_avg_duration", "type_avg_cost", "type_median_cost",
    "year_avg_duration", "year_project_count",
]

FEATURES_CLUSTER = [
    "log_cost", "log_units", "duration_months",
    "avgCostPerSqFt", "booking_rate", "land_cost_ratio",
    "projectType_enc", "distName_enc", "log_cost_per_unit",
]

FEATURES_ANOMALY = [
    "totalEstimatedCost", "avgCostPerSqFt", "duration_months",
    "booking_rate", "totalLandCost", "land_cost_ratio",
    "totalUnits", "cost_per_unit", "sell_dev_ratio", "totalDevelopCost",
]

# =============================================================
# Training-time aggregate averages — used as defaults in predict.py
# so new project predictions are not biased by zero fallbacks
# =============================================================
DISTRICT_DEFAULTS = {
    "Ahmedabad"  : {"dist_avg_duration": 50.45, "dist_median_duration": 48.0,  "dist_project_count": 4174, "dist_avg_cost": 380724473, "dist_avg_cost_feat": 380724473, "dist_median_cost_feat": 228762661},
    "Vadodara"   : {"dist_avg_duration": 52.10, "dist_median_duration": 48.0,  "dist_project_count": 2403, "dist_avg_cost": 290000000, "dist_avg_cost_feat": 290000000, "dist_median_cost_feat": 180000000},
    "Surat"      : {"dist_avg_duration": 54.20, "dist_median_duration": 51.0,  "dist_project_count": 1858, "dist_avg_cost": 310000000, "dist_avg_cost_feat": 310000000, "dist_median_cost_feat": 195000000},
    "Rajkot"     : {"dist_avg_duration": 51.80, "dist_median_duration": 48.0,  "dist_project_count": 1665, "dist_avg_cost": 250000000, "dist_avg_cost_feat": 250000000, "dist_median_cost_feat": 155000000},
    "Gandhinagar": {"dist_avg_duration": 49.30, "dist_median_duration": 46.0,  "dist_project_count": 1251, "dist_avg_cost": 270000000, "dist_avg_cost_feat": 270000000, "dist_median_cost_feat": 165000000},
    "Bhavnagar"  : {"dist_avg_duration": 50.00, "dist_median_duration": 47.0,  "dist_project_count":  580, "dist_avg_cost": 200000000, "dist_avg_cost_feat": 200000000, "dist_median_cost_feat": 130000000},
}
# Fallback for unlisted districts
DISTRICT_DEFAULTS["_default"] = {"dist_avg_duration": 53.0, "dist_median_duration": 50.0, "dist_project_count": 300, "dist_avg_cost": 260000000, "dist_avg_cost_feat": 260000000, "dist_median_cost_feat": 160000000}

TYPE_DEFAULTS = {
    "Residential/Group Housing": {"type_avg_duration": 53.69, "type_median_duration": 51.0, "type_avg_cost": 271160176, "type_median_cost": 138748800},
    "Mixed Development"        : {"type_avg_duration": 55.20, "type_median_duration": 52.0, "type_avg_cost": 320000000, "type_median_cost": 190000000},
    "Commercial"               : {"type_avg_duration": 48.10, "type_median_duration": 45.0, "type_avg_cost": 210000000, "type_median_cost": 130000000},
    "Plotted Development"      : {"type_avg_duration": 44.30, "type_median_duration": 42.0, "type_avg_cost": 180000000, "type_median_cost": 110000000},
}
TYPE_DEFAULTS["_default"] = {"type_avg_duration": 53.0, "type_median_duration": 50.0, "type_avg_cost": 260000000, "type_median_cost": 150000000}

YEAR_DEFAULTS = {
    2020: {"year_avg_duration": 52.0,  "year_project_count": 780},
    2021: {"year_avg_duration": 51.5,  "year_project_count": 1100},
    2022: {"year_avg_duration": 52.33, "year_project_count": 1283},
    2023: {"year_avg_duration": 53.0,  "year_project_count": 1450},
    2024: {"year_avg_duration": 53.0,  "year_project_count": 1200},
    2025: {"year_avg_duration": 53.0,  "year_project_count": 1200},
    2026: {"year_avg_duration": 53.0,  "year_project_count": 1200},
}
YEAR_DEFAULTS["_default"] = {"year_avg_duration": 53.0, "year_project_count": 1000}


# =============================================================
# HELPER
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

    Returns:
        df       : DataFrame with all features added
        encoders : dict of fitted LabelEncoders
    """

    print("=" * 50)
    print("  GujEstateAI — Feature Engineering")
    print("=" * 50)

    df = pd.read_csv(input_path)
    print(f"\nLoaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # ── Core log transforms ───────────────────────────────────
    df["log_cost"]        = np.log1p(df["totalEstimatedCost"])
    df["log_units"]       = np.log1p(df["totalUnits"])
    df["log_land_cost"]   = np.log1p(df["totalLandCost"])
    df["log_develop_cost"]= np.log1p(df["totalDevelopCost"])
    df["log_selling"]     = np.log1p(df["totalSellingAmount"])
    df["log_carpet"]      = np.log1p(df["totalCarpetArea_form3A"])
    df["log_buildup"]     = np.log1p(df["totalBuiltupArea_form3A"])
    df["log_sqft_build"]  = np.log1p(df["totalSquareFootBuild"])
    print("\nCreated: log transforms (8 columns)")

    # ── Ratio features ────────────────────────────────────────
    df["cost_per_unit"] = (df["totalEstimatedCost"] / df["totalUnits"]).replace([np.inf, -np.inf], np.nan)
    df["cost_per_unit"] = df["cost_per_unit"].fillna(df["cost_per_unit"].median())
    df["log_cost_per_unit"] = np.log1p(df["cost_per_unit"])

    df["land_cost_ratio"] = (df["totalLandCost"] / df["totalEstimatedCost"]).clip(0, 1)

    df["sell_dev_ratio"] = (df["totalSellingAmount"] / df["totalDevelopCost"]).replace([np.inf, -np.inf], np.nan)
    df["sell_dev_ratio"] = df["sell_dev_ratio"].fillna(df["sell_dev_ratio"].median())
    df["sell_dev_ratio"] = df["sell_dev_ratio"].clip(0, df["sell_dev_ratio"].quantile(0.99))
    print("Created: cost_per_unit, land_cost_ratio, sell_dev_ratio")

    # ── Binary features ───────────────────────────────────────
    df["is_redevelop"] = (df["underRedevelopment"] == "YES").astype(int)
    print("Created: is_redevelop")

    # ── Time features ─────────────────────────────────────────
    df["start_month"]   = df["startProjectMonth"].astype(int)
    df["start_quarter"] = df["start_month"].apply(lambda m: (m - 1) // 3 + 1)
    print("Created: start_month, start_quarter")

    # ── Simplified promoter type ──────────────────────────────
    df["promoter_type_simple"] = df["promoterType"].apply(simplify_promoter)
    print("Created: promoter_type_simple")

    # ── District-level aggregate features ────────────────────
    dist_agg = df.groupby("distName").agg(
        dist_avg_duration    = ("duration_months", "mean"),
        dist_median_duration = ("duration_months", "median"),
        dist_project_count   = ("duration_months", "count"),
        dist_avg_cost        = ("totalEstimatedCost", "mean"),
        dist_avg_cost_feat   = ("totalEstimatedCost", "mean"),
        dist_median_cost_feat= ("totalEstimatedCost", "median"),
    ).reset_index()
    df = df.merge(dist_agg, on="distName", how="left")
    print("Created: district-level aggregates")

    # ── Project type aggregate features ──────────────────────
    type_agg = df.groupby("projectType").agg(
        type_avg_duration    = ("duration_months", "mean"),
        type_median_duration = ("duration_months", "median"),
        type_avg_cost        = ("totalEstimatedCost", "mean"),
        type_median_cost     = ("totalEstimatedCost", "median"),
    ).reset_index()
    df = df.merge(type_agg, on="projectType", how="left")
    print("Created: project type aggregates")

    # ── Year-level aggregate features ────────────────────────
    year_agg = df.groupby("startProjectYear").agg(
        year_avg_duration  = ("duration_months", "mean"),
        year_project_count = ("duration_months", "count"),
    ).reset_index()
    df = df.merge(year_agg, on="startProjectYear", how="left")
    print("Created: year-level aggregates")

    # ── Encode categoricals ───────────────────────────────────
    encoders  = {}
    cat_cols  = ["projectType", "distName", "promoter_type_simple"]

    os.makedirs(
        os.path.dirname(encoders_path) if os.path.dirname(encoders_path) else ".",
        exist_ok=True
    )

    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col]    = le

    joblib.dump(encoders, encoders_path)
    print(f"\nCategoricals encoded: {cat_cols}")
    print(f"Encoders saved to   : {encoders_path}")

    df.reset_index(drop=True, inplace=True)

    print(f"\n{'=' * 50}")
    print(f"  FEATURE ENGINEERING COMPLETE")
    print(f"{'=' * 50}")
    print(f"  Rows          : {df.shape[0]:,}")
    print(f"  Total columns : {df.shape[1]}")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"  Saved to      : {output_path}")

    return df, encoders


if __name__ == "__main__":
    df, encoders = build_features(
        input_path    = "data/processed/cleaned.csv",
        output_path   = "data/processed/features.csv",
        encoders_path = "models/encoders.pkl"
    )
    print(f"\nDone! Shape: {df.shape}")
    print(f"  FEATURES_DURATION : {len(FEATURES_DURATION)}")
    print(f"  FEATURES_COST     : {len(FEATURES_COST)}")
    print(f"  FEATURES_CLUSTER  : {len(FEATURES_CLUSTER)}")
    print(f"  FEATURES_ANOMALY  : {len(FEATURES_ANOMALY)}")