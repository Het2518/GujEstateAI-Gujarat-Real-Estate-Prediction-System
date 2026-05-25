"""predict.py

Reusable inference helpers for the Streamlit dashboard and CLI usage.
Supports all 5 modules: duration, cost, clustering, anomaly detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from features import FEATURES_COST, FEATURES_DURATION, FEATURES_CLUSTER, FEATURES_ANOMALY, simplify_promoter
except ModuleNotFoundError:
    from src.features import FEATURES_COST, FEATURES_DURATION, FEATURES_CLUSTER, FEATURES_ANOMALY, simplify_promoter


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"


def _resolve_models_dir(path: str | Path | None = None) -> Path:
    return Path(path) if path is not None else MODELS_DIR


def load_models(path: str | Path | None = None) -> dict[str, Any]:
    """Load the trained model artifacts that exist in the repository.

    Handles tuple-packed models (clustering and anomaly are stored as
    ``(model, scaler)`` tuples) and unpacks them into separate keys.
    """

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
        if not file_path.exists():
            continue
        try:
            obj = joblib.load(file_path)
        except Exception:
            continue

        # Unpack tuple-stored models
        if name == "clustering" and isinstance(obj, tuple) and len(obj) == 2:
            artifacts["clustering_km"] = obj[0]
            artifacts["clustering_scaler"] = obj[1]
        elif name == "anomaly" and isinstance(obj, tuple) and len(obj) == 2:
            artifacts["anomaly_iso"] = obj[0]
            artifacts["anomaly_scaler"] = obj[1]
        else:
            artifacts[name] = obj

    return artifacts


# ------------------------------------------------------------------
# Encoder helpers
# ------------------------------------------------------------------

def _encode_value(encoder, value: Any) -> int:
    value = str(value)
    if hasattr(encoder, "classes_") and value in encoder.classes_:
        return int(encoder.transform([value])[0])
    return 0


def _apply_encoders(frame: pd.DataFrame, encoders: dict[str, Any] | None) -> pd.DataFrame:
    if not encoders:
        return frame

    frame = frame.copy()
    for col_name in ("projectType", "distName", "promoter_type_simple"):
        enc_col = f"{col_name}_enc"
        if col_name in encoders and col_name in frame.columns:
            frame[enc_col] = _encode_value(encoders[col_name], frame.loc[0, col_name])
    return frame


# ------------------------------------------------------------------
# Feature engineering from a user payload
# ------------------------------------------------------------------

def _feature_payload_values(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw user payload dict into all possible feature values."""

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
        "startProjectMonth": start_month,
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
        # Raw categorical values (used for encoding, not as model features)
        "projectType": payload.get("projectType", "Residential/Group Housing"),
        "distName": payload.get("distName", "Ahmedabad"),
        "promoter_type_simple": payload.get(
            "promoter_type_simple",
            simplify_promoter(payload.get("promoterType", "Partnership")),
        ),
        # Anomaly-specific features
        "totalEstimatedCost": total_estimated_cost,
        "totalLandCost": total_land_cost,
        "totalDevelopCost": total_develop_cost,
    }

    return feature_values


def _build_feature_frame(payload: dict[str, Any], feature_names: list[str], encoders: dict[str, Any] | None) -> pd.DataFrame:
    """Build a single-row DataFrame with the exact features a model expects."""

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


# ------------------------------------------------------------------
# Individual predictors
# ------------------------------------------------------------------

def predict_duration(payload: dict[str, Any], models: dict[str, Any] | None = None) -> float | None:
    """Predict project duration in months when the duration model is available."""

    models = models or load_models()
    model = models.get("duration")
    if model is None:
        return None

    try:
        encoders = models.get("encoders")
        feature_names = list(getattr(model, "feature_names_in_", FEATURES_DURATION))
        features = _build_feature_frame(payload, feature_names, encoders)
        prediction = float(model.predict(features)[0])
        return max(prediction, 0.0)  # duration cannot be negative
    except Exception:
        return None


def predict_cost(payload: dict[str, Any], models: dict[str, Any] | None = None) -> float | None:
    """Predict total estimated cost when the cost model is available."""

    models = models or load_models()
    model = models.get("cost")
    if model is None:
        return None

    try:
        encoders = models.get("encoders")
        feature_names = list(getattr(model, "feature_names_in_", FEATURES_COST))
        features = _build_feature_frame(payload, feature_names, encoders)
        prediction = float(model.predict(features)[0])
        # The cost model is a Ridge regression trained on log1p(cost)
        actual_cost = np.expm1(prediction)
        return max(actual_cost, 0.0)  # cost cannot be negative
    except Exception:
        return None


def predict_cluster(payload: dict[str, Any], models: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Predict which cluster a project belongs to."""

    models = models or load_models()
    km = models.get("clustering_km")
    scaler = models.get("clustering_scaler")
    if km is None or scaler is None:
        return None

    try:
        encoders = models.get("encoders")
        values = _feature_payload_values(payload)

        # Use the scaler's actual feature names (authoritative source)
        feature_names = list(getattr(scaler, "feature_names_in_", FEATURES_CLUSTER))

        # Build a single-row frame with the exact features
        row_data = {}
        for feat in feature_names:
            row_data[feat] = values.get(feat, 0.0)

        frame = pd.DataFrame([row_data])

        # Apply encoders for encoded columns
        if encoders:
            frame = _apply_encoders(frame.assign(
                projectType=values.get("projectType", "Residential/Group Housing"),
                distName=values.get("distName", "Ahmedabad"),
                promoter_type_simple=values.get("promoter_type_simple", "Partnership"),
            ), encoders)

        cluster_frame = frame[feature_names]
        scaled = scaler.transform(cluster_frame)
        cluster_id = int(km.predict(scaled)[0])

        # Cluster names (from the training notebook)
        cluster_names = {
            0: "Mixed Development Projects",
            1: "Mid-Range Residential",
            2: "Plotted Development Schemes",
            3: "Large High-Budget Residential",
            4: "Affordable Small Residential",
        }
        cluster_label = cluster_names.get(cluster_id, f"Cluster {cluster_id}")

        return {"cluster_id": cluster_id, "cluster_label": cluster_label}
    except Exception:
        return None


def predict_anomaly(payload: dict[str, Any], models: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Predict whether a project is anomalous (risky)."""

    models = models or load_models()
    iso = models.get("anomaly_iso")
    scaler = models.get("anomaly_scaler")
    if iso is None or scaler is None:
        return None

    try:
        values = _feature_payload_values(payload)

        # Use the scaler's actual feature names (authoritative source)
        feature_names = list(getattr(scaler, "feature_names_in_", FEATURES_ANOMALY))

        row_data = {}
        for feat in feature_names:
            row_data[feat] = values.get(feat, 0.0)

        frame = pd.DataFrame([row_data])
        scaled = scaler.transform(frame)

        raw_score = float(iso.score_samples(scaled)[0])
        prediction = int(iso.predict(scaled)[0])
        risk_flag = prediction == -1

        # Normalize to 0-100 risk scale (higher = riskier)
        # Based on data distribution: max ~ -0.35 (normal), min ~ -0.75 (anomalous)
        calibrated_min = -0.75
        calibrated_max = -0.35
        risk = (calibrated_max - raw_score) / (calibrated_max - calibrated_min) * 100.0
        risk_score = max(0.0, min(100.0, risk))

        # Assign risk category
        if risk_score >= 75:
            risk_category = "Critical"
        elif risk_score >= 50:
            risk_category = "High"
        elif risk_score >= 25:
            risk_category = "Medium"
        else:
            risk_category = "Low"

        return {
            "risk_flag": risk_flag,
            "risk_score": round(risk_score, 1),
            "risk_category": risk_category,
            "raw_score": round(raw_score, 4),
        }
    except Exception:
        return None


# ------------------------------------------------------------------
# Bundle predictor — all modules at once
# ------------------------------------------------------------------

def predict_bundle(payload: dict[str, Any], models: dict[str, Any] | None = None) -> dict[str, Any]:
    """Predict duration, cost, cluster, and risk in one call for dashboard use."""

    models = models or load_models()

    # 1. Duration
    duration = predict_duration(payload, models)

    # 2. Cost (use predicted duration if available)
    cost_payload = dict(payload)
    if duration is not None:
        cost_payload["duration_months"] = duration
    cost = predict_cost(cost_payload, models)

    # 3. Cluster
    cluster_payload = dict(payload)
    if duration is not None:
        cluster_payload["duration_months"] = duration
    if cost is not None:
        cluster_payload["totalEstimatedCost"] = cost
    cluster_result = predict_cluster(cluster_payload, models)

    # 4. Anomaly / Risk
    anomaly_payload = dict(payload)
    if duration is not None:
        anomaly_payload["duration_months"] = duration
    if cost is not None:
        anomaly_payload["totalEstimatedCost"] = cost
    anomaly_result = predict_anomaly(anomaly_payload, models)

    return {
        "duration_months": duration,
        "totalEstimatedCost": cost,
        "cluster": cluster_result,
        "anomaly": anomaly_result,
    }


if __name__ == "__main__":
    artifacts = load_models()
    print(f"Loaded artifacts: {', '.join(sorted(artifacts)) if artifacts else 'none'}")
