"""predict.py

Reusable inference helpers for the Streamlit dashboard and CLI usage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from features import FEATURES_COST, FEATURES_DURATION, simplify_promoter
except ModuleNotFoundError:
    from src.features import FEATURES_COST, FEATURES_DURATION, simplify_promoter


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"


def _resolve_models_dir(path: str | Path | None = None) -> Path:
    return Path(path) if path is not None else MODELS_DIR


def load_models(path: str | Path | None = None) -> dict[str, Any]:
    """Load the trained model artifacts that exist in the repository."""

    models_dir = _resolve_models_dir(path)
    model_files = {
        "duration": models_dir / "duration_model.pkl",
        "cost": models_dir / "cost_model.pkl",
        "clustering": models_dir / "clustering_model.pkl",
        "anomaly": models_dir / "anomaly_model.pkl",
        "encoders": models_dir / "encoders.pkl",
    }

    artifacts: dict[str, Any] = {}
    for name, file_path in model_files.items():
        if file_path.exists():
            artifacts[name] = joblib.load(file_path)

    return artifacts


def _encode_value(encoder, value: Any) -> int:
    value = str(value)
    if hasattr(encoder, "classes_") and value in encoder.classes_:
        return int(encoder.transform([value])[0])
    return 0


def _apply_encoders(frame: pd.DataFrame, encoders: dict[str, Any] | None) -> pd.DataFrame:
    if not encoders:
        return frame

    frame = frame.copy()
    frame["projectType_enc"] = _encode_value(encoders["projectType"], frame.loc[0, "projectType"])
    frame["distName_enc"] = _encode_value(encoders["distName"], frame.loc[0, "distName"])
    frame["promoter_type_simple_enc"] = _encode_value(
        encoders["promoter_type_simple"], frame.loc[0, "promoter_type_simple"]
    )
    return frame


def _feature_payload_values(payload: dict[str, Any]) -> dict[str, Any]:
    total_units = max(int(payload.get("totalUnits", 1) or 1), 1)
    total_estimated_cost = float(payload.get("totalEstimatedCost", 0) or 0)
    total_land_cost = float(payload.get("totalLandCost", 0) or 0)
    total_develop_cost = float(payload.get("totalDevelopCost", total_estimated_cost or 1.0) or 1.0)
    total_selling_amount = float(payload.get("totalSellingAmount", max(total_estimated_cost, 1.0)) or max(total_estimated_cost, 1.0))
    total_carpet_area = float(payload.get("totalCarpetArea_form3A", 0) or 0)
    total_builtup_area = float(payload.get("totalBuiltupArea_form3A", 0) or 0)
    total_sqft_build = float(payload.get("totalSquareFootBuild", 0) or 0)
    start_year = int(payload.get("startProjectYear", 2021) or 2021)
    start_month = int(payload.get("startProjectMonth", 1) or 1)
    start_quarter = int((start_month - 1) // 3 + 1)
    duration_months = float(payload.get("duration_months", 0) or 0)
    cost_per_unit = total_estimated_cost / total_units if total_units else 0.0
    avg_cost_per_sqft = total_estimated_cost / total_sqft_build if total_sqft_build else 0.0

    feature_values = {
        "projectType_enc": 0,
        "distName_enc": 0,
        "promoter_type_simple_enc": 0,
        "totalUnits": total_units,
        "log_units": np.log1p(total_units),
        "noOfInventory": float(payload.get("noOfInventory", total_units) or total_units),
        "totalProjects": float(payload.get("totalProjects", 1) or 1),
        "log_cost": np.log1p(max(total_estimated_cost, 0.0)),
        "log_land_cost": np.log1p(max(total_land_cost, 0.0)),
        "log_develop_cost": np.log1p(max(total_develop_cost, 0.0)),
        "log_selling": np.log1p(max(total_selling_amount, 0.0)),
        "log_carpet": np.log1p(max(total_carpet_area, 0.0)),
        "log_buildup": np.log1p(max(total_builtup_area, 0.0)),
        "log_sqft_build": np.log1p(max(total_sqft_build, 0.0)),
        "log_cost_per_unit": np.log1p(max(cost_per_unit, 0.0)),
        "cost_per_unit": cost_per_unit,
        "avgCostPerSqFt": avg_cost_per_sqft,
        "avgCostPerUnit": cost_per_unit,
        "land_cost_ratio": total_land_cost / total_estimated_cost if total_estimated_cost > 0 else 0.0,
        "sell_dev_ratio": total_selling_amount / total_develop_cost if total_develop_cost > 0 else 1.0,
        "booking_rate": float(payload.get("booking_rate", 0) or 0),
        "is_redevelop": 1 if str(payload.get("underRedevelopment", "NO")).upper() == "YES" else 0,
        "startProjectYear": start_year,
        "start_month": start_month,
        "start_quarter": start_quarter,
        "duration_months": duration_months,
        "dist_avg_duration": float(payload.get("dist_avg_duration", duration_months) or duration_months),
        "dist_median_duration": float(payload.get("dist_median_duration", duration_months) or duration_months),
        "dist_project_count": float(payload.get("dist_project_count", 0) or 0),
        "dist_avg_cost": float(payload.get("dist_avg_cost", total_estimated_cost) or total_estimated_cost),
        "dist_avg_cost_feat": float(payload.get("dist_avg_cost_feat", total_estimated_cost) or total_estimated_cost),
        "dist_median_cost_feat": float(payload.get("dist_median_cost_feat", total_estimated_cost) or total_estimated_cost),
        "type_avg_duration": float(payload.get("type_avg_duration", duration_months) or duration_months),
        "type_median_duration": float(payload.get("type_median_duration", duration_months) or duration_months),
        "type_avg_cost": float(payload.get("type_avg_cost", total_estimated_cost) or total_estimated_cost),
        "type_median_cost": float(payload.get("type_median_cost", total_estimated_cost) or total_estimated_cost),
        "year_avg_duration": float(payload.get("year_avg_duration", duration_months) or duration_months),
        "year_project_count": float(payload.get("year_project_count", 0) or 0),
        "projectType": payload.get("projectType", "Residential/Group Housing"),
        "distName": payload.get("distName", "Ahmedabad"),
        "promoter_type_simple": payload.get(
            "promoter_type_simple",
            simplify_promoter(payload.get("promoterType", "Partnership")),
        ),
    }

    return feature_values


def _build_feature_frame(payload: dict[str, Any], feature_names: list[str], encoders: dict[str, Any] | None) -> pd.DataFrame:
    values = _feature_payload_values(payload)
    frame = pd.DataFrame([{name: values.get(name, 0.0) for name in feature_names}])
    frame = _apply_encoders(frame.assign(
        projectType=values.get("projectType", "Residential/Group Housing"),
        distName=values.get("distName", "Ahmedabad"),
        promoter_type_simple=values.get("promoter_type_simple", "Partnership"),
    ), encoders)

    for name in feature_names:
        if name not in frame.columns:
            frame[name] = values.get(name, 0.0)

    return frame[feature_names]


def build_duration_features(payload: dict[str, Any], encoders: dict[str, Any] | None = None) -> pd.DataFrame:
    """Build a single-row feature frame for the duration model."""

    models = load_models()
    duration_model = models.get("duration")
    feature_names = list(getattr(duration_model, "feature_names_in_", FEATURES_DURATION))
    return _build_feature_frame(payload, feature_names, encoders)


def build_cost_features(payload: dict[str, Any], encoders: dict[str, Any] | None = None) -> pd.DataFrame:
    """Build a single-row feature frame for the cost model."""

    models = load_models()
    cost_model = models.get("cost")
    feature_names = list(getattr(cost_model, "feature_names_in_", FEATURES_COST))
    return _build_feature_frame(payload, feature_names, encoders)


def predict_duration(payload: dict[str, Any], models: dict[str, Any] | None = None) -> float | None:
    """Predict project duration in months when the duration model is available."""

    models = models or load_models()
    model = models.get("duration")
    if model is None:
        return None

    encoders = models.get("encoders")
    features = build_duration_features(payload, encoders)
    return float(model.predict(features)[0])


def predict_cost(payload: dict[str, Any], models: dict[str, Any] | None = None) -> float | None:
    """Predict total estimated cost when the cost model is available."""

    models = models or load_models()
    model = models.get("cost")
    if model is None:
        return None

    encoders = models.get("encoders")
    features = build_cost_features(payload, encoders)
    return float(model.predict(features)[0])


def predict_bundle(payload: dict[str, Any], models: dict[str, Any] | None = None) -> dict[str, float | None]:
    """Predict duration and cost in one call for dashboard use."""

    models = models or load_models()
    duration = predict_duration(payload, models)

    cost_payload = dict(payload)
    if duration is not None:
        cost_payload["duration_months"] = duration

    cost = predict_cost(cost_payload, models)
    return {"duration_months": duration, "totalEstimatedCost": cost}


if __name__ == "__main__":
    artifacts = load_models()
    print(f"Loaded artifacts: {', '.join(sorted(artifacts)) if artifacts else 'none'}")
