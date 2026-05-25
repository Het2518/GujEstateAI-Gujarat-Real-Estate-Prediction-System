"""
src/predict.py
GujEstateAI — Unified Inference Pipeline

Loads all 4 trained models and returns a complete intelligence
report for any new project in a single call.

Usage:
    from src.predict import load_models, predict_bundle
    models = load_models()
    result = predict_bundle(payload, models)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from features import (
        FEATURES_COST, FEATURES_DURATION, FEATURES_CLUSTER, FEATURES_ANOMALY,
        simplify_promoter,
        DISTRICT_DEFAULTS, TYPE_DEFAULTS, YEAR_DEFAULTS,
    )
except ModuleNotFoundError:
    from src.features import (
        FEATURES_COST, FEATURES_DURATION, FEATURES_CLUSTER, FEATURES_ANOMALY,
        simplify_promoter,
        DISTRICT_DEFAULTS, TYPE_DEFAULTS, YEAR_DEFAULTS,
    )


ROOT_DIR   = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"


# =============================================================
# MODEL LOADING
# =============================================================

def _resolve_models_dir(path: str | Path | None = None) -> Path:
    return Path(path) if path is not None else MODELS_DIR


def load_models(path: str | Path | None = None) -> dict[str, Any]:
    """
    Load all trained model artifacts from the models/ folder.
    Handles tuple-packed models (clustering and anomaly are stored
    as (model, scaler) tuples).
    """
    models_dir = _resolve_models_dir(path)
    model_files = {
        "duration"  : models_dir / "duration_model.pkl",
        "cost"      : models_dir / "cost_model.pkl",
        "clustering": models_dir / "clustering_model.pkl",
        "anomaly"   : models_dir / "anomaly_model.pkl",
        "encoders"  : models_dir / "encoders.pkl",
    }

    artifacts: dict[str, Any] = {}
    for name, file_path in model_files.items():
        if not file_path.exists():
            print(f"  WARNING: {file_path.name} not found — skipping")
            continue
        try:
            obj = joblib.load(file_path)
        except Exception as e:
            print(f"  WARNING: could not load {file_path.name}: {e}")
            continue

        # Unpack tuple-stored models
        if name == "clustering" and isinstance(obj, tuple) and len(obj) == 2:
            artifacts["clustering_km"]     = obj[0]
            artifacts["clustering_scaler"] = obj[1]
        elif name == "anomaly" and isinstance(obj, tuple) and len(obj) == 2:
            artifacts["anomaly_iso"]    = obj[0]
            artifacts["anomaly_scaler"] = obj[1]
        else:
            artifacts[name] = obj

    return artifacts


# =============================================================
# ENCODER HELPERS
# =============================================================

def _encode_value(encoder, value: Any) -> int:
    value = str(value)
    if hasattr(encoder, "classes_") and value in encoder.classes_:
        return int(encoder.transform([value])[0])
    return 0   # unknown category → class 0 (safe fallback)


def _apply_encoders(frame: pd.DataFrame, encoders: dict[str, Any] | None) -> pd.DataFrame:
    if not encoders:
        return frame
    frame = frame.copy()
    for col_name in ("projectType", "distName", "promoter_type_simple"):
        enc_col = f"{col_name}_enc"
        if col_name in encoders and col_name in frame.columns:
            frame[enc_col] = _encode_value(encoders[col_name], frame.loc[0, col_name])
    return frame


# =============================================================
# FEATURE PAYLOAD BUILDER
# FIX: aggregate features (dist_avg_duration etc.) now use
#      training-time lookup tables instead of defaulting to 0
# =============================================================

def _feature_payload_values(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a raw user payload dict into all feature values needed
    by all four models.
    """

    # ── Raw inputs ────────────────────────────────────────────
    total_units           = max(int(payload.get("totalUnits", 1) or 1), 1)
    total_estimated_cost  = float(payload.get("totalEstimatedCost", 0) or 0)
    total_land_cost       = float(payload.get("totalLandCost", 0) or 0)
    total_develop_cost    = float(payload.get("totalDevelopCost", max(total_estimated_cost, 1.0)) or 1.0)
    total_selling_amount  = float(payload.get("totalSellingAmount", max(total_estimated_cost, 1.0)) or max(total_estimated_cost, 1.0))
    total_carpet_area     = float(payload.get("totalCarpetArea_form3A", 0) or 0)
    total_builtup_area    = float(payload.get("totalBuiltupArea_form3A", 0) or 0)
    total_sqft_build      = float(payload.get("totalSquareFootBuild", 0) or 0)
    start_year            = int(payload.get("startProjectYear", 2022) or 2022)
    start_month           = int(payload.get("startProjectMonth", 1) or 1)
    start_quarter         = (start_month - 1) // 3 + 1
    duration_months       = float(payload.get("duration_months", 0) or 0)
    cost_per_unit         = total_estimated_cost / total_units if total_units else 0.0

    district      = str(payload.get("distName", "Ahmedabad"))
    project_type  = str(payload.get("projectType", "Residential/Group Housing"))
    promoter_raw  = str(payload.get("promoterType", payload.get("promoter_type_simple", "Partnership")))
    promoter_simp = simplify_promoter(promoter_raw)

    # ── Aggregate lookups (use real training-time averages) ───
    dist_d  = DISTRICT_DEFAULTS.get(district, DISTRICT_DEFAULTS["_default"])
    type_d  = TYPE_DEFAULTS.get(project_type, TYPE_DEFAULTS["_default"])
    year_d  = YEAR_DEFAULTS.get(start_year, YEAR_DEFAULTS["_default"])

    values = {
        # Categorical (filled in after encoding)
        "projectType_enc"          : 0,
        "distName_enc"             : 0,
        "promoter_type_simple_enc" : 0,

        # Size / units
        "totalUnits"    : total_units,
        "log_units"     : np.log1p(total_units),
        "noOfInventory" : float(payload.get("noOfInventory", total_units) or total_units),
        "totalProjects" : float(payload.get("totalProjects", 1) or 1),

        # Cost features
        "log_cost"          : np.log1p(max(total_estimated_cost, 0.0)),
        "log_land_cost"     : np.log1p(max(total_land_cost, 0.0)),
        "log_develop_cost"  : np.log1p(max(total_develop_cost, 0.0)),
        "log_selling"       : np.log1p(max(total_selling_amount, 0.0)),
        "log_carpet"        : np.log1p(max(total_carpet_area, 0.0)),
        "log_buildup"       : np.log1p(max(total_builtup_area, 0.0)),
        "log_sqft_build"    : np.log1p(max(total_sqft_build, 0.0)),
        "log_cost_per_unit" : np.log1p(max(cost_per_unit, 0.0)),
        "cost_per_unit"     : cost_per_unit,
        "avgCostPerSqFt"    : total_estimated_cost / total_sqft_build if total_sqft_build else 0.0,
        "avgCostPerUnit"    : cost_per_unit,

        # Ratio features
        "land_cost_ratio" : total_land_cost / total_estimated_cost if total_estimated_cost > 0 else 0.0,
        "sell_dev_ratio"  : total_selling_amount / total_develop_cost if total_develop_cost > 0 else 1.0,
        "booking_rate"    : float(payload.get("booking_rate", 0.5) or 0.5),

        # Binary / time
        "is_redevelop"     : 1 if str(payload.get("underRedevelopment", "NO")).upper() == "YES" else 0,
        "startProjectYear" : start_year,
        "startProjectMonth": start_month,
        "start_month"      : start_month,
        "start_quarter"    : start_quarter,
        "duration_months"  : duration_months,

        # District aggregates — from lookup table
        "dist_avg_duration"     : dist_d.get("dist_avg_duration", 53.0),
        "dist_median_duration"  : dist_d.get("dist_median_duration", 50.0),
        "dist_project_count"    : dist_d.get("dist_project_count", 300.0),
        "dist_avg_cost"         : dist_d.get("dist_avg_cost", 260000000.0),
        "dist_avg_cost_feat"    : dist_d.get("dist_avg_cost_feat", 260000000.0),
        "dist_median_cost_feat" : dist_d.get("dist_median_cost_feat", 160000000.0),

        # Project type aggregates — from lookup table
        "type_avg_duration"     : type_d.get("type_avg_duration", 53.0),
        "type_median_duration"  : type_d.get("type_median_duration", 50.0),
        "type_avg_cost"         : type_d.get("type_avg_cost", 260000000.0),
        "type_median_cost"      : type_d.get("type_median_cost", 150000000.0),

        # Year aggregates — from lookup table
        "year_avg_duration"  : year_d.get("year_avg_duration", 53.0),
        "year_project_count" : year_d.get("year_project_count", 1000.0),

        # Raw categorical (used for encoding step)
        "projectType"           : project_type,
        "distName"              : district,
        "promoter_type_simple"  : promoter_simp,

        # Anomaly-specific raw values
        "totalEstimatedCost" : total_estimated_cost,
        "totalLandCost"      : total_land_cost,
        "totalDevelopCost"   : total_develop_cost,
    }

    return values


def _build_feature_frame(
    payload: dict[str, Any],
    feature_names: list[str],
    encoders: dict[str, Any] | None,
) -> pd.DataFrame:
    """Build a single-row DataFrame with exactly the features a model expects."""

    values = _feature_payload_values(payload)
    frame  = pd.DataFrame([{name: values.get(name, 0.0) for name in feature_names}])

    # Apply encoders
    frame = _apply_encoders(
        frame.assign(
            projectType          = values.get("projectType", "Residential/Group Housing"),
            distName             = values.get("distName", "Ahmedabad"),
            promoter_type_simple = values.get("promoter_type_simple", "Partnership"),
        ),
        encoders,
    )

    # Ensure all required columns are present
    for name in feature_names:
        if name not in frame.columns:
            frame[name] = values.get(name, 0.0)

    return frame[feature_names]


# =============================================================
# INDIVIDUAL PREDICTORS
# =============================================================

def predict_duration(
    payload: dict[str, Any],
    models: dict[str, Any] | None = None,
) -> float | None:
    """Predict project duration in months."""
    models = models or load_models()
    model  = models.get("duration")
    if model is None:
        return None
    try:
        encoders      = models.get("encoders")
        feature_names = list(getattr(model, "feature_names_in_", FEATURES_DURATION))
        features      = _build_feature_frame(payload, feature_names, encoders)
        return max(float(model.predict(features)[0]), 0.0)
    except Exception:
        return None


def predict_cost(
    payload: dict[str, Any],
    models: dict[str, Any] | None = None,
) -> float | None:
    """Predict total estimated cost (returns actual rupees, not log scale)."""
    models = models or load_models()
    model  = models.get("cost")
    if model is None:
        return None
    try:
        encoders      = models.get("encoders")
        feature_names = list(getattr(model, "feature_names_in_", FEATURES_COST))
        features      = _build_feature_frame(payload, feature_names, encoders)
        prediction    = float(model.predict(features)[0])
        # Ridge cost model was trained on log1p(cost) so convert back
        actual_cost = np.expm1(prediction)
        return max(actual_cost, 0.0)
    except Exception:
        return None


def predict_cluster(
    payload: dict[str, Any],
    models: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Predict which cluster a project belongs to."""
    models  = models or load_models()
    km      = models.get("clustering_km")
    scaler  = models.get("clustering_scaler")
    if km is None or scaler is None:
        return None
    try:
        encoders      = models.get("encoders")
        feature_names = list(getattr(scaler, "feature_names_in_", FEATURES_CLUSTER))
        values        = _feature_payload_values(payload)

        row_data = {feat: values.get(feat, 0.0) for feat in feature_names}
        frame    = pd.DataFrame([row_data])
        if encoders:
            frame = _apply_encoders(
                frame.assign(
                    projectType          = values.get("projectType"),
                    distName             = values.get("distName"),
                    promoter_type_simple = values.get("promoter_type_simple"),
                ),
                encoders,
            )

        scaled     = scaler.transform(frame[feature_names])
        cluster_id = int(km.predict(scaled)[0])

        cluster_names = {
            0: "Mixed Development Projects",
            1: "Mid-Range Residential",
            2: "Plotted Development Schemes",
            3: "Large High-Budget Residential",
            4: "Affordable Small Residential",
        }

        return {
            "cluster_id"   : cluster_id,
            "cluster_label": cluster_names.get(cluster_id, f"Cluster {cluster_id}"),
        }
    except Exception:
        return None


def predict_anomaly(
    payload: dict[str, Any],
    models: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Predict risk score and flag for a project."""
    models = models or load_models()
    iso    = models.get("anomaly_iso")
    scaler = models.get("anomaly_scaler")
    if iso is None or scaler is None:
        return None
    try:
        values        = _feature_payload_values(payload)
        feature_names = list(getattr(scaler, "feature_names_in_", FEATURES_ANOMALY))
        row_data      = {feat: values.get(feat, 0.0) for feat in feature_names}
        frame         = pd.DataFrame([row_data])
        scaled        = scaler.transform(frame)

        raw_score  = float(iso.score_samples(scaled)[0])
        prediction = int(iso.predict(scaled)[0])
        risk_flag  = prediction == -1

        # Normalize raw score to 0-100 (higher = riskier)
        calibrated_min = -0.75
        calibrated_max = -0.35
        risk = (calibrated_max - raw_score) / (calibrated_max - calibrated_min) * 100.0
        risk_score = round(max(0.0, min(100.0, risk)), 1)

        if risk_score >= 75:    risk_category = "Critical"
        elif risk_score >= 50:  risk_category = "High"
        elif risk_score >= 25:  risk_category = "Medium"
        else:                   risk_category = "Low"

        return {
            "risk_flag"    : risk_flag,
            "risk_score"   : risk_score,
            "risk_category": risk_category,
            "raw_score"    : round(raw_score, 4),
        }
    except Exception:
        return None


# =============================================================
# BUNDLE PREDICTOR — all modules at once
# =============================================================

def predict_bundle(
    payload: dict[str, Any],
    models: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run all 4 ML modules and return a complete project report.

    Returns:
        {
            "duration_months"    : float | None,
            "totalEstimatedCost" : float | None,
            "cluster"            : {"cluster_id", "cluster_label"} | None,
            "anomaly"            : {"risk_flag", "risk_score", "risk_category"} | None,
        }
    """
    models = models or load_models()

    # 1. Duration
    duration = predict_duration(payload, models)

    # 2. Cost — feed predicted duration into cost model
    cost_payload = dict(payload)
    if duration is not None:
        cost_payload["duration_months"] = duration
    cost = predict_cost(cost_payload, models)

    # 3. Cluster — feed predicted cost so log_cost is accurate
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
        "duration_months"    : duration,
        "totalEstimatedCost" : cost,
        "cluster"            : cluster_result,
        "anomaly"            : anomaly_result,
    }


# =============================================================
# CLI test
# =============================================================
if __name__ == "__main__":
    artifacts = load_models()
    print(f"Loaded: {', '.join(sorted(artifacts)) if artifacts else 'none'}")

    sample = {
        "projectType"        : "Residential/Group Housing",
        "distName"           : "Ahmedabad",
        "promoterType"       : "PARTNERSHIP FIRM",
        "underRedevelopment" : "NO",
        "totalUnits"         : 120,
        "totalEstimatedCost" : 50_000_000,
        "totalLandCost"      : 15_000_000,
        "totalDevelopCost"   : 35_000_000,
        "totalSellingAmount" : 60_000_000,
        "startProjectYear"   : 2022,
        "startProjectMonth"  : 3,
        "booking_rate"       : 0.55,
        "avgCostPerSqFt"     : 4500,
    }

    print("\nRunning predict_bundle on sample project...")
    result = predict_bundle(sample, artifacts)

    print("\n=== PREDICTION REPORT ===")
    if result["duration_months"] is not None:
        print(f"  Duration  : {result['duration_months']:.1f} months  ({result['duration_months']/12:.1f} years)")
    if result["totalEstimatedCost"] is not None:
        print(f"  Cost      : Rs {result['totalEstimatedCost']/1e7:.2f} Crores")
    if result["cluster"]:
        print(f"  Cluster   : {result['cluster']['cluster_label']}")
    if result["anomaly"]:
        a = result["anomaly"]
        print(f"  Risk      : {a['risk_category']}  (score: {a['risk_score']}/100)")