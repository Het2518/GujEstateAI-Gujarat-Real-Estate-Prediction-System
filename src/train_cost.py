"""train_cost.py

Training script for the Module 2 cost prediction model.
"""

import os
import sys
import types

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRegressor

try:
    from features import FEATURES_COST
except ModuleNotFoundError:
    from src.features import FEATURES_COST


TARGET = "totalEstimatedCost"


def load_cost_data(features_path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(features_path)

    available_features = [feature for feature in FEATURES_COST if feature in df.columns]
    required_columns = available_features + [TARGET]
    df = df.dropna(subset=required_columns).copy()

    X = df[available_features].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    y = df[TARGET].astype(float)

    return X, y


def _build_models() -> dict[str, TransformedTargetRegressor]:
    return {
        "Ridge": TransformedTargetRegressor(
            regressor=Ridge(),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "Random Forest": TransformedTargetRegressor(
            regressor=RandomForestRegressor(
                n_estimators=250,
                random_state=42,
                n_jobs=-1,
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "XGBoost": TransformedTargetRegressor(
            regressor=XGBRegressor(
                n_estimators=600,
                learning_rate=0.03,
                max_depth=5,
                min_child_weight=2,
                subsample=0.9,
                colsample_bytree=0.85,
                reg_alpha=0.0,
                reg_lambda=1.5,
                random_state=42,
                objective="reg:squarederror",
                verbosity=0,
                n_jobs=-1,
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
    }


def train_cost_model(
    features_path: str = "data/processed/features.csv",
    model_out_path: str = "models/cost_model.pkl",
    reports_dir: str = "reports",
):
    print("=" * 50)
    print("  Module 2 — Cost Prediction")
    print("=" * 50)

    X, y = load_cost_data(features_path)
    print(f"\nLoaded: {X.shape[0]:,} rows")
    print(f"Features used: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train : {X_train.shape[0]:,} rows")
    print(f"Test  : {X_test.shape[0]:,} rows")

    models = _build_models()
    results = {}

    print("\nTraining models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        results[name] = {"MAE": round(mae, 2), "R2": round(r2, 4), "model": model}
        print(f"  {name:<15} MAE = ₹{mae:,.0f}   R2 = {r2:.4f}")

    results_df = pd.DataFrame(
        {
            "Model": list(results.keys()),
            "MAE (INR)": [v["MAE"] for v in results.values()],
            "R2 Score": [v["R2"] for v in results.values()],
        }
    )

    best_idx = results_df["MAE (INR)"].idxmin()
    best_name = results_df.loc[best_idx, "Model"]
    best_model = results[best_name]["model"]
    best_preds = best_model.predict(X_test)

    print("\nBest model:", best_name)
    print(f"Best MAE : ₹{results[best_name]['MAE']:,.0f}")
    print(f"Best R2  : {results[best_name]['R2']:.4f}")

    cv_estimator = _build_models()["XGBoost"]
    cv_scores = cross_val_score(
        cv_estimator,
        X,
        y,
        cv=5,
        scoring="neg_mean_absolute_error",
    )
    cv_mae = -cv_scores
    print(f"\n5-Fold CV MAE : ₹{cv_mae.mean():,.0f} +/- ₹{cv_mae.std():,.0f}")

    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(os.path.join(reports_dir, "shap_plots"), exist_ok=True)

    xgb_model = results["XGBoost"]["model"]
    xgb_regressor = xgb_model.regressor_

    importances = pd.Series(
        xgb_regressor.feature_importances_,
        index=X.columns,
    ).sort_values(ascending=False)

    plt.figure(figsize=(9, 5))
    colors = ["#1565C0" if i == 0 else "#64B5F6" for i in range(len(importances))]
    plt.barh(importances.index[::-1], importances.values[::-1], color=colors[::-1])
    plt.title("Feature Importance — Cost Prediction (XGBoost)", fontsize=12, fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "20_feature_importance_cost.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, best_preds, alpha=0.25, color="#2196F3", s=10)
    min_val = min(y_test.min(), best_preds.min())
    max_val = max(y_test.max(), best_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linewidth=2, linestyle="--")
    plt.title("Actual vs Predicted Cost (Best Model)", fontsize=12, fontweight="bold")
    plt.xlabel("Actual Cost (INR)")
    plt.ylabel("Predicted Cost (INR)")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "21_actual_vs_predicted_cost.png"), dpi=150)
    plt.close()

    errors = y_test.values - best_preds
    plt.figure(figsize=(9, 4))
    plt.hist(errors, bins=60, color="#4CAF50", edgecolor="white")
    plt.axvline(0, color="red", linewidth=2, linestyle="--", label="Zero error")
    plt.title("Prediction Error Distribution (Cost)", fontsize=12, fontweight="bold")
    plt.xlabel("Error (Actual - Predicted) in INR")
    plt.ylabel("Number of Predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "22_error_distribution_cost.png"), dpi=150)
    plt.close()

    if "torch" not in sys.modules:
        sys.modules["torch"] = types.ModuleType("torch")

    try:
        import shap

        shap_values = shap.TreeExplainer(xgb_regressor).shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        plt.figure()
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=X.columns.tolist(),
            show=False,
        )
        plt.title("SHAP Summary — Cost Prediction", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, "shap_plots", "20_shap_cost.png"), dpi=150, bbox_inches="tight")
        plt.close()

        sample_idx = 0
        shap_sample = shap_values[sample_idx]
        top_idx = np.argsort(np.abs(shap_sample))[::-1][:8]
        top_features = X_test.columns.tolist()
        top_names = [top_features[i] for i in top_idx]
        top_values = shap_sample[top_idx]
        colors = ["#EF5350" if value > 0 else "#4CAF50" for value in top_values]

        plt.figure(figsize=(9, 5))
        plt.barh(top_names[::-1], top_values[::-1], color=colors[::-1])
        plt.axvline(0, color="black", linewidth=0.8)
        plt.title("SHAP Waterfall — One Project Explained\nCost Prediction", fontsize=11, fontweight="bold")
        plt.xlabel("SHAP Value (impact on predicted log-cost)")
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, "shap_plots", "21_shap_waterfall_cost.png"), dpi=150)
        plt.close()
    except Exception as exc:
        print(f"SHAP plot skipped: {exc}")

    sample_values = X_train.median(numeric_only=True).to_dict()
    encoders_path = os.path.join("models", "encoders.pkl")
    if os.path.exists(encoders_path):
        encoders = joblib.load(encoders_path)

        if "projectType_enc" in X.columns:
            sample_values["projectType_enc"] = encoders["projectType"].transform(["Residential/Group Housing"])[0]
        if "distName_enc" in X.columns:
            sample_values["distName_enc"] = encoders["distName"].transform(["Ahmedabad"])[0]
        if "promoter_type_simple_enc" in X.columns:
            sample_values["promoter_type_simple_enc"] = encoders["promoter_type_simple"].transform(["Partnership"])[0]
        if "totalUnits" in X.columns:
            sample_values["totalUnits"] = 120
        if "duration_months" in X.columns:
            sample_values["duration_months"] = 48
        if "totalLandCost" in X.columns:
            sample_values["totalLandCost"] = 15_000_000
        if "startProjectYear" in X.columns:
            sample_values["startProjectYear"] = 2021
        if "startProjectMonth" in X.columns:
            sample_values["startProjectMonth"] = 1
        if "start_quarter" in X.columns:
            sample_values["start_quarter"] = 1

        sample_df = pd.DataFrame([{feature: sample_values.get(feature, 0) for feature in X.columns}])
        sample_pred = best_model.predict(sample_df)[0]

        print("\nMANUAL PREDICTION TEST:")
        print("-" * 38)
        print("  Project Type  : Residential/Group Housing")
        print("  District      : Ahmedabad")
        print("  Total Units   : 120")
        print("  Duration      : 48 months")
        print(f"  Predicted Cost : ₹{sample_pred:,.0f}")

    joblib.dump(best_model, model_out_path)
    print(f"\nModel saved to {model_out_path}")

    print(f"\n{'=' * 50}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Best Model : {best_name}")
    print(f"  MAE        : ₹{results[best_name]['MAE']:,.0f}")
    print(f"  R2 Score   : {results[best_name]['R2']:.4f}")
    print(f"  CV MAE     : ₹{cv_mae.mean():,.0f} +/- ₹{cv_mae.std():,.0f}")
    print(f"  Saved to   : {model_out_path}")

    return best_model, results


def main():
    train_cost_model()


if __name__ == "__main__":
    main()
