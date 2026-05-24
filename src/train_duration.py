# =============================================================
# src/train_duration.py
# GujEstateAI — Phase 4: Module 1 — Duration Prediction
# =============================================================
# Same logic as 04_module1_duration.ipynb
# but as a clean reusable script.
#
# Usage:
#   from src.train_duration import train_duration_model
#   model, results = train_duration_model("data/processed/features.csv")
#
# Or run directly:
#   python src/train_duration.py
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model    import Ridge
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import mean_absolute_error, r2_score
from xgboost                 import XGBRegressor


# =============================================================
# Feature list for this module
# =============================================================

FEATURES = [
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

TARGET = "duration_months"


# =============================================================
# MAIN FUNCTION
# =============================================================

def train_duration_model(
    features_path  = "data/processed/features.csv",
    model_out_path = "models/duration_model.pkl",
    reports_dir    = "reports"
):
    """
    Train duration prediction models, evaluate, save best model.

    Returns:
        best_model : fitted XGBRegressor
        results    : dict with MAE and R2 for all models
    """

    print("=" * 50)
    print("  Module 1 — Duration Prediction")
    print("=" * 50)

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------
    df = pd.read_csv(features_path)
    print(f"\nLoaded: {df.shape[0]:,} rows")

    X = df[FEATURES]
    y = df[TARGET]

    # ----------------------------------------------------------
    # Train / test split
    # ----------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train : {X_train.shape[0]:,} rows")
    print(f"Test  : {X_test.shape[0]:,} rows")

    # ----------------------------------------------------------
    # Train all 3 models
    # ----------------------------------------------------------
    models = {
        "Ridge"        : Ridge(),
        "RandomForest" : RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost"      : XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbosity=0),
    }

    results = {}
    print("\nTraining models...")

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        r2    = r2_score(y_test, preds)
        results[name] = {"MAE": round(mae, 2), "R2": round(r2, 4), "model": model}
        print(f"  {name:<15} MAE = {mae:.2f} months   R2 = {r2:.4f}")

    # ----------------------------------------------------------
    # Best model = XGBoost
    # ----------------------------------------------------------
    best_model  = models["XGBoost"]
    best_preds  = best_model.predict(X_test)
    best_mae    = results["XGBoost"]["MAE"]
    best_r2     = results["XGBoost"]["R2"]

    # ----------------------------------------------------------
    # Cross validation
    # ----------------------------------------------------------
    cv = cross_val_score(
        XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbosity=0),
        X, y, cv=5, scoring="neg_mean_absolute_error"
    )
    cv_mae = -cv
    print(f"\n5-Fold CV MAE : {cv_mae.mean():.2f} +/- {cv_mae.std():.2f} months")

    # ----------------------------------------------------------
    # Feature importances plot
    # ----------------------------------------------------------
    os.makedirs(reports_dir, exist_ok=True)

    importances = pd.Series(
        best_model.feature_importances_,
        index=FEATURES
    ).sort_values(ascending=False)

    plt.figure(figsize=(9, 5))
    colors = ["#1565C0" if i == 0 else "#64B5F6" for i in range(len(importances))]
    plt.barh(importances.index[::-1], importances.values[::-1], color=colors[::-1])
    plt.title("Feature Importance — Duration Prediction", fontsize=12, fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(f"{reports_dir}/15_feature_importance_duration.png", dpi=150)
    plt.close()

    # ----------------------------------------------------------
    # Actual vs Predicted plot
    # ----------------------------------------------------------
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, best_preds, alpha=0.3, color="#2196F3", s=10)
    lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
    plt.plot(lims, lims, "r--", linewidth=2, label="Perfect prediction")
    plt.title("Actual vs Predicted Duration", fontsize=12, fontweight="bold")
    plt.xlabel("Actual (months)")
    plt.ylabel("Predicted (months)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{reports_dir}/16_actual_vs_predicted_duration.png", dpi=150)
    plt.close()

    # ----------------------------------------------------------
    # Save best model
    # ----------------------------------------------------------
    os.makedirs(os.path.dirname(model_out_path) if os.path.dirname(model_out_path) else ".", exist_ok=True)
    joblib.dump(best_model, model_out_path)

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print(f"\n{'=' * 50}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Best Model : XGBoost")
    print(f"  MAE        : {best_mae} months")
    print(f"  R2 Score   : {best_r2}")
    print(f"  CV MAE     : {cv_mae.mean():.2f} +/- {cv_mae.std():.2f} months")
    print(f"  Saved to   : {model_out_path}")

    return best_model, results


# =============================================================
# Run directly: python src/train_duration.py
# =============================================================
if __name__ == "__main__":
    model, results = train_duration_model(
        features_path  = "data/processed/features.csv",
        model_out_path = "models/duration_model.pkl",
        reports_dir    = "reports"
    )
    print("\nDone!")