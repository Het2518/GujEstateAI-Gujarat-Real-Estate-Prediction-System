"""
GujEstateAI — src/preprocess.py
================================
Reusable data-cleaning pipeline for ProjectInfo_Gujarat.csv.
Called directly by 02_cleaning.ipynb and by any downstream script
that needs a clean DataFrame without re-running the notebook.

Usage
-----
    from src.preprocess import run_cleaning, COLS_TO_DROP

    df_clean = run_cleaning(
        input_path  = "data/raw/ProjectInfo_Gujarat.csv",
        output_path = "data/processed/cleaned.csv",   # pass None to skip saving
        verbose     = True
    )

EDA findings that drive every decision are documented inline.
"""

import os
import pandas as pd
import numpy as np


# ── Column lists ──────────────────────────────────────────────────────────────

# Finding #1  — >30 % null in EDA → drop entirely
COLS_TO_DROP = [
    'pinCode',          # 46.9 % null
    'tPNo',             # 51.4 % null
    'totalAreaOfLand',  # 39.9 % null
    'AvgAreaOfLand',    # 39.9 % null
    'architect_name',   # 32.4 % null
    'eng_name',         # 33.5 % null
    'projectAddress2',  #  7.7 % null (low information, duplicate address field)
]

# Finding #2  — ~6 % null; small enough to drop rows safely
ROWS_REQUIRED = ['distName', 'startDate', 'completionDate']

# ── Cleaning thresholds (derived from EDA quantile analysis) ──────────────────
COST_OUTLIER_QUANTILE  = 0.995   # Finding #8 → 73 rows removed
SQFT_OUTLIER_QUANTILE  = 0.990   # Finding #7 → 136 rows removed
DURATION_MIN_MONTHS    = 3       # Finding #5 → below this = data-entry error
DURATION_MAX_MONTHS    = 240     # Finding #5 → above this = extreme outlier (1 row)


# ─────────────────────────────────────────────────────────────────────────────
# Step-level functions  (importable individually for unit tests)
# ─────────────────────────────────────────────────────────────────────────────

def drop_high_null_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Step 2.1a — Drop columns identified as >30 % null in EDA.
    EDA Finding #1.
    """
    cols_present = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_present)
    if verbose:
        print(f"  [2.1a] Dropped {len(cols_present)} high-null columns: {cols_present}")
        print(f"         Shape after: {df.shape}")
    return df


def drop_required_null_rows(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Step 2.1b — Drop rows where critical columns are null.
    EDA Finding #2  (~6 % of rows).
    """
    before = len(df)
    cols_present = [c for c in ROWS_REQUIRED if c in df.columns]
    df = df.dropna(subset=cols_present)
    if verbose:
        print(f"  [2.1b] Dropped {before - len(df):,} rows with null in {cols_present}")
        print(f"         Shape after: {df.shape}")
    return df


def impute_noOfInventory(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Step 2.1c — Fill noOfInventory nulls with per-projectType median.
    EDA Finding #3  (17 % null, 2,432 rows).
    Medians from EDA:
        Commercial                   86
        Mixed Development           117
        Plotted Development          52
        Residential/Group Housing    45
    """
    before = df['noOfInventory'].isna().sum()
    df['noOfInventory'] = df.groupby('projectType')['noOfInventory'] \
                            .transform(lambda x: x.fillna(x.median()))
    after = df['noOfInventory'].isna().sum()
    if verbose:
        print(f"  [2.1c] noOfInventory: filled {before - after:,} nulls via per-type median "
              f"({after} remain)")
    return df


def parse_dates(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Step 2.2 — Parse startDate / completionDate to datetime;
    extract start_year and start_month helper columns.
    """
    df['startDate']      = pd.to_datetime(df['startDate'],      errors='coerce')
    df['completionDate'] = pd.to_datetime(df['completionDate'], errors='coerce')

    df['start_year']  = df['startDate'].dt.year
    df['start_month'] = df['startDate'].dt.month

    if verbose:
        bad_start = df['startDate'].isna().sum()
        bad_comp  = df['completionDate'].isna().sum()
        print(f"  [2.2]  Dates parsed. Unparseable startDate: {bad_start}, "
              f"completionDate: {bad_comp}")
    return df


def remove_cost_outliers(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Step 2.3a — Remove totalEstimatedCost rows beyond 99.5th percentile.
    EDA Finding #8  (73 extreme-value rows; skewness confirms heavy right tail).
    Threshold computed from data: ~₹346 Cr.
    """
    before = len(df)
    upper  = df['totalEstimatedCost'].quantile(COST_OUTLIER_QUANTILE)
    df     = df[df['totalEstimatedCost'] <= upper]
    if verbose:
        print(f"  [2.3a] Cost outlier cap  : ₹{upper/1e7:.1f} Cr  "
              f"(removed {before - len(df):,} rows)")
    return df


def remove_sqft_outliers(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Step 2.3b — Remove avgCostPerSqFt rows beyond 99th percentile.
    EDA Finding #7  (136 rows; boxplot showed extreme right-tail fliers).
    """
    before = len(df)
    upper  = df['avgCostPerSqFt'].quantile(SQFT_OUTLIER_QUANTILE)
    df     = df[df['avgCostPerSqFt'].isna() | (df['avgCostPerSqFt'] <= upper)]
    if verbose:
        print(f"  [2.3b] SqFt outlier cap  : ₹{upper:,.0f}/sqft  "
              f"(removed {before - len(df):,} rows)")
    return df


def remove_duration_outliers(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Step 2.3c — Derive duration_months; keep only [3, 240] month range.
    EDA Finding #5  (1 row >240 months; no rows <3 months in this dataset).
    """
    df['duration_months'] = (
        (df['EndProjectYear']  - df['startProjectYear'])  * 12 +
        (df['EndProjectMonth'] - df['startProjectMonth'])
    )
    before = len(df)
    df = df[
        (df['duration_months'] >= DURATION_MIN_MONTHS) &
        (df['duration_months'] <= DURATION_MAX_MONTHS)
    ]
    if verbose:
        print(f"  [2.3c] Duration filter [{DURATION_MIN_MONTHS}–{DURATION_MAX_MONTHS} months]: "
              f"removed {before - len(df):,} rows  "
              f"(mean={df['duration_months'].mean():.1f}, "
              f"median={df['duration_months'].median():.1f})")
    return df


def clip_booking_rate(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Step 2.3d — Derive booking_rate and clip to [0, 1].
    EDA Finding #6  (no values >1 in this dataset, but clip is applied
    defensively for data robustness).
    """
    df['booking_rate'] = (df['bookedUnits'] / df['totalUnits']).clip(0, 1)
    if verbose:
        print(f"  [2.3d] booking_rate derived and clipped to [0,1].  "
              f"Mean={df['booking_rate'].mean():.1%}, "
              f"Median={df['booking_rate'].median():.1%}")
    return df


def derive_margin(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Step 2.3e — Derive profit margin proxy.
    EDA Finding Step 1.3  (margin = (selling - develop) / selling).
    """
    df['margin'] = (
        (df['totalSellingAmount'] - df['totalDevelopCost']) /
        df['totalSellingAmount'].replace(0, np.nan)
    )
    if verbose:
        neg = (df['margin'] < 0).sum()
        print(f"  [2.3e] margin derived.  "
              f"Mean={df['margin'].mean():.1%}, "
              f"Negative margins: {neg:,}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_cleaning(
    input_path:  str = "data/raw/ProjectInfo_Gujarat.csv",
    output_path: str = "data/processed/cleaned.csv",
    verbose:     bool = True
) -> pd.DataFrame:
    """
    Run the full Phase 2 cleaning pipeline in order.

    Steps
    -----
    2.1a  Drop high-null columns  (EDA Finding #1)
    2.1b  Drop rows with null in critical columns  (EDA Finding #2)
    2.1c  Impute noOfInventory per projectType median  (EDA Finding #3)
    2.2   Parse dates, extract start_year / start_month
    2.3a  Remove totalEstimatedCost outliers >99.5th pct  (EDA Finding #8)
    2.3b  Remove avgCostPerSqFt outliers >99th pct  (EDA Finding #7)
    2.3c  Remove duration_months outside [3, 240]  (EDA Finding #5)
    2.3d  Clip booking_rate to [0, 1]  (EDA Finding #6)
    2.3e  Derive margin  (EDA Step 1.3)
    2.4   Save to output_path

    Returns
    -------
    pd.DataFrame  — cleaned DataFrame
    """
    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(input_path)
    raw_shape = df.shape
    if verbose:
        print("=" * 60)
        print("  GujEstateAI — Phase 2: Data Cleaning")
        print("=" * 60)
        print(f"  Input  : {input_path}")
        print(f"  Raw    : {raw_shape[0]:,} rows × {raw_shape[1]} columns")
        print()

    # ── Steps ─────────────────────────────────────────────────────────────────
    df = drop_high_null_columns(df,    verbose=verbose)
    df = drop_required_null_rows(df,   verbose=verbose)
    df = impute_noOfInventory(df,      verbose=verbose)
    df = parse_dates(df,               verbose=verbose)
    df = remove_cost_outliers(df,      verbose=verbose)
    df = remove_sqft_outliers(df,      verbose=verbose)
    df = remove_duration_outliers(df,  verbose=verbose)
    df = clip_booking_rate(df,         verbose=verbose)
    df = derive_margin(df,             verbose=verbose)

    # ── Final report ──────────────────────────────────────────────────────────
    if verbose:
        print()
        print(f"  Raw    : {raw_shape[0]:,} rows × {raw_shape[1]} columns")
        print(f"  Clean  : {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"  Removed: {raw_shape[0] - df.shape[0]:,} rows  "
              f"({(raw_shape[0] - df.shape[0]) / raw_shape[0] * 100:.1f} %)")
        remaining_nulls = df.isnull().sum()
        remaining_nulls = remaining_nulls[remaining_nulls > 0]
        if len(remaining_nulls):
            print(f"\n  Remaining nulls:")
            for col, n in remaining_nulls.items():
                print(f"    {col:<35} {n:>5} ({n/len(df)*100:.1f}%)")
        else:
            print("  No remaining nulls ✔")

    # ── Save ──────────────────────────────────────────────────────────────────
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"\n  ✅  Saved → {output_path}")

    if verbose:
        print("=" * 60)

    return df


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GujEstateAI — Phase 2 Data Cleaning")
    parser.add_argument("--input",   default="data/raw/ProjectInfo_Gujarat.csv")
    parser.add_argument("--output",  default="data/processed/cleaned.csv")
    parser.add_argument("--quiet",   action="store_true")
    args = parser.parse_args()

    run_cleaning(
        input_path  = args.input,
        output_path = args.output,
        verbose     = not args.quiet
    )