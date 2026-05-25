# =============================================================
# src/preprocess.py
# GujEstateAI — Phase 2: Data Cleaning
# =============================================================
# This script does the same thing as 02_cleaning.ipynb
# but as a clean reusable function.
#
# Usage:
#   from src.preprocess import clean_data
#   df = clean_data("data/raw/ProjectInfo_Gujarat.csv")
#
# Or run directly:
#   python src/preprocess.py
# =============================================================

import pandas as pd
import numpy as np
import os


# =============================================================
# District name correction map
# =============================================================
DISTRICT_NAME_FIXES = {
    "AHmedabad": "Ahmedabad",
    "RAJKOT": "Rajkot",
    "SURAT": "Surat",
    "VADODARA": "Vadodara",
    "Chhota Udepur": "Chhota Udaipur",
    "Chhota udepur": "Chhota Udaipur",
    "Sabar Kantha": "Sabarkantha",
    "Sabarkantha": "Sabarkantha",
}


def standardize_district_names(df, column="distName"):
    """
    Fix duplicate / misspelled district names in-place.

    Steps:
        1. Replace non-breaking spaces (\\xa0) with regular spaces
        2. Strip leading/trailing whitespace
        3. Apply hardcoded correction map for known misspellings
    """
    if column not in df.columns:
        return df
    df = df.copy()
    # Fix non-breaking spaces and strip
    df[column] = (
        df[column]
        .astype(str)
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
    )
    # Apply correction map
    df[column] = df[column].replace(DISTRICT_NAME_FIXES)
    return df


def clean_data(input_path, output_path=None):
    """
    Load raw Gujarat real estate CSV, clean it, return cleaned DataFrame.

    Parameters:
        input_path  : path to raw CSV file
        output_path : (optional) path to save cleaned CSV

    Returns:
        df : cleaned pandas DataFrame
    """

    print("=" * 50)
    print("  GujEstateAI — Data Cleaning")
    print("=" * 50)

    # ----------------------------------------------------------
    # STEP 1 — Load raw data
    # ----------------------------------------------------------
    df = pd.read_csv(input_path)
    print(f"\nStep 1 — Loaded raw data")
    print(f"  Shape : {df.shape[0]:,} rows x {df.shape[1]} columns")

    # ----------------------------------------------------------
    # STEP 2 — Drop useless columns (too many nulls or not useful)
    # ----------------------------------------------------------
    cols_to_drop = [
        "pinCode",          # 47% nulls
        "tPNo",             # 51% nulls
        "architect_name",   # 32% nulls, not an ML feature
        "eng_name",         # 34% nulls, not an ML feature
        "projectAddress2",  # duplicate address column
        "totalAreaOfLand",  # 40% nulls
        "AvgAreaOfLand",    # 40% nulls
    ]

    # Only drop columns that actually exist (safety check)
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    print(f"\nStep 2 — Dropped {len(cols_to_drop)} useless columns")
    print(f"  Shape : {df.shape[0]:,} rows x {df.shape[1]} columns")

    # ----------------------------------------------------------
    # STEP 2.5 — Standardize district names
    # ----------------------------------------------------------
    df = standardize_district_names(df, "distName")
    unique_before = 35  # known raw count
    unique_after = df["distName"].nunique()
    print(f"\nStep 2.5 — Standardized district names")
    print(f"  Unique districts : {unique_after}")

    # ----------------------------------------------------------
    # STEP 3 — Drop rows with missing essential values
    # ----------------------------------------------------------
    rows_before = len(df)

    essential_cols = [
        "distName",
        "startProjectYear",
        "startProjectMonth",
        "EndProjectYear",
        "EndProjectMonth",
        "totalEstimatedCost",
        "projectType",
    ]
    # Only check columns that exist
    essential_cols = [c for c in essential_cols if c in df.columns]
    df.dropna(subset=essential_cols, inplace=True)

    print(f"\nStep 3 — Dropped rows with missing essential values")
    print(f"  Removed : {rows_before - len(df):,} rows")
    print(f"  Shape   : {len(df):,} rows")

    # ----------------------------------------------------------
    # STEP 4 — Create duration_months column
    # ----------------------------------------------------------
    df["duration_months"] = (
        (df["EndProjectYear"]  - df["startProjectYear"])  * 12 +
        (df["EndProjectMonth"] - df["startProjectMonth"])
    )
    print(f"\nStep 4 — Created duration_months column")

    # ----------------------------------------------------------
    # STEP 5 — Remove bad duration values
    # ----------------------------------------------------------
    rows_before = len(df)
    df = df[
        (df["duration_months"] >= 3) &
        (df["duration_months"] <= 240)
    ].copy()
    print(f"\nStep 5 — Removed bad duration values")
    print(f"  Removed : {rows_before - len(df):,} rows")
    print(f"  Range   : {df['duration_months'].min()} to {df['duration_months'].max()} months")

    # ----------------------------------------------------------
    # STEP 6 — Remove cost outliers (top 0.5%)
    # ----------------------------------------------------------
    rows_before    = len(df)
    cost_threshold = df["totalEstimatedCost"].quantile(0.995)
    df = df[df["totalEstimatedCost"] <= cost_threshold].copy()
    print(f"\nStep 6 — Removed cost outliers (above 99.5th percentile)")
    print(f"  Threshold : Rs {cost_threshold:,.0f}")
    print(f"  Removed   : {rows_before - len(df):,} rows")

    # ----------------------------------------------------------
    # STEP 7 — Remove avgCostPerSqFt outliers (top 1%)
    # ----------------------------------------------------------
    rows_before    = len(df)
    sqft_threshold = df["avgCostPerSqFt"].quantile(0.99)
    mask = df["avgCostPerSqFt"].notna() & (df["avgCostPerSqFt"] > sqft_threshold)
    df   = df[~mask].copy()
    print(f"\nStep 7 — Removed cost/sqft outliers (above 99th percentile)")
    print(f"  Threshold : Rs {sqft_threshold:,.0f}")
    print(f"  Removed   : {rows_before - len(df):,} rows")

    # ----------------------------------------------------------
    # STEP 8 — Fill remaining nulls
    # ----------------------------------------------------------

    # Fill with median per projectType group
    for col in ["noOfInventory", "avgCostPerSqFt"]:
        if col in df.columns:
            df[col] = df.groupby("projectType")[col] \
                        .transform(lambda x: x.fillna(x.median()))

    # Fill with overall median
    fill_median_cols = [
        "totalUnits", "bookedUnits", "avgUnits",
        "totalSellingAmount", "bookedSellingAmount",
        "totalReceivedAmount", "bookedReceivedAmount",
        "totalCarpetArea_form3A", "totalBuiltupArea_form3A",
        "totalSquareFootBuild", "AvgSquareFootBuild", "avgCostPerUnit"
    ]
    for col in fill_median_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    print(f"\nStep 8 — Filled remaining nulls with medians")

    # ----------------------------------------------------------
    # STEP 9 — Create booking_rate column
    # ----------------------------------------------------------
    df["booking_rate"] = (df["bookedUnits"] / df["totalUnits"]).clip(0, 1)
    print(f"\nStep 9 — Created booking_rate column")

    # ----------------------------------------------------------
    # STEP 10 — Reset index
    # ----------------------------------------------------------
    df.reset_index(drop=True, inplace=True)

    # ----------------------------------------------------------
    # DONE — Print summary
    # ----------------------------------------------------------
    print(f"\n{'=' * 50}")
    print(f"  CLEANING COMPLETE")
    print(f"{'=' * 50}")
    print(f"  Original rows  : 14,507")
    print(f"  Cleaned rows   : {len(df):,}")
    print(f"  Rows removed   : {14507 - len(df):,}")
    print(f"  Final columns  : {df.shape[1]}")

    # ----------------------------------------------------------
    # SAVE (optional)
    # ----------------------------------------------------------
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n  Saved to : {output_path}")

    return df


# =============================================================
# Run directly: python src/preprocess.py
# =============================================================
if __name__ == "__main__":
    df_clean = clean_data(
        input_path  = "data/raw/ProjectInfo_Gujarat.csv",
        output_path = "data/processed/cleaned.csv"
    )
    print(f"\nDone! Shape: {df_clean.shape}")