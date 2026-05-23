"""Feature engineering pipeline for GujEstateAI.

Builds model-ready features from cleaned project-level data and persists:
1) encoded categorical mappings in models/encoders.pkl
2) engineered dataset in data/processed/features.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Module 1 — Duration Prediction features
FEATURES_DURATION: List[str] = [
    "projectType_enc",
    "distName_enc",
    "promoterType_enc",
    "totalUnits",
    "log_cost",
    "totalLandCost",
    "is_redevelop",
    "start_year",
    "start_quarter",
    "land_cost_ratio",
    "avgCostPerUnit",
]

# Module 2 — Cost Prediction features
FEATURES_COST: List[str] = [
    "projectType_enc",
    "distName_enc",
    "promoterType_enc",
    "totalUnits",
    "duration_months",
    "totalLandCost",
    "totalCarpetArea_form3A",
    "avgCostPerSqFt",
    "is_redevelop",
    "start_year",
]

# Module 4 — Clustering features
FEATURES_CLUSTER: List[str] = [
    "log_cost",
    "log_units",
    "duration_months",
    "avgCostPerSqFt",
    "booking_rate",
    "land_cost_ratio",
    "projectType_enc",
    "distName_enc",
]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise divide that avoids inf and keeps NaN for invalid rows."""
    result = numerator.astype(float) / denominator.replace({0: np.nan}).astype(float)
    return result.replace([np.inf, -np.inf], np.nan)


def create_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create core features and targets used by downstream modules."""
    out = df.copy()

    # Target for Module 1: duration in months.
    out["duration_months"] = (
        (out["EndProjectYear"] - out["startProjectYear"]) * 12
        + (out["EndProjectMonth"] - out["startProjectMonth"])
    )

    # Derived features.
    out["booking_rate"] = _safe_divide(out["bookedUnits"], out["totalUnits"])
    out["cost_per_unit"] = _safe_divide(out["totalEstimatedCost"], out["totalUnits"])
    out["land_cost_ratio"] = _safe_divide(out["totalLandCost"], out["totalEstimatedCost"])
    out["sell_dev_ratio"] = _safe_divide(out["totalSellingAmount"], out["totalDevelopCost"])
    out["is_redevelop"] = (
        out["underRedevelopment"].astype(str).str.strip().str.upper() == "YES"
    ).astype(int)
    out["log_cost"] = np.log1p(out["totalEstimatedCost"].clip(lower=0))
    out["log_units"] = np.log1p(out["totalUnits"].clip(lower=0))
    out["start_quarter"] = out["start_month"].apply(
        lambda m: (int(m) - 1) // 3 + 1 if pd.notna(m) else np.nan
    )

    return out


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Label-encode selected categorical columns and return fitted encoders."""
    out = df.copy()
    encoders: Dict[str, LabelEncoder] = {}

    for col in ["projectType", "distName", "promoterType"]:
        encoder = LabelEncoder()
        out[f"{col}_enc"] = encoder.fit_transform(out[col].astype(str))
        encoders[col] = encoder

    return out, encoders


def run_feature_engineering(
    input_csv: Path,
    output_csv: Path,
    encoders_path: Path,
) -> pd.DataFrame:
    """Run full feature engineering pipeline and persist outputs."""
    df = pd.read_csv(input_csv)
    df = create_core_features(df)
    df, encoders = encode_categoricals(df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    encoders_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_csv, index=False)
    joblib.dump(encoders, encoders_path)

    return df


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    input_csv = root / "data" / "processed" / "cleaned.csv"
    output_csv = root / "data" / "processed" / "features.csv"
    encoders_path = root / "models" / "encoders.pkl"

    df = run_feature_engineering(input_csv, output_csv, encoders_path)
    print(f"Feature engineering completed. Shape: {df.shape}")
    print(f"Saved features -> {output_csv}")
    print(f"Saved encoders -> {encoders_path}")


if __name__ == "__main__":
    main()
