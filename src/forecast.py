"""forecast.py

Helpers for loading the saved forecast and score tables used by the dashboard.
Applies district name standardization on load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:
    from preprocess import standardize_district_names
except ModuleNotFoundError:
    try:
        from src.preprocess import standardize_district_names
    except ModuleNotFoundError:
        # Fallback: define inline if preprocess is not importable
        _FIXES = {
            "AHmedabad": "Ahmedabad", "RAJKOT": "Rajkot", "SURAT": "Surat",
            "VADODARA": "Vadodara", "Chhota Udepur": "Chhota Udaipur",
            "Chhota udepur": "Chhota Udaipur", "Sabar Kantha": "Sabarkantha",
        }

        def standardize_district_names(df, column="distName"):
            if column not in df.columns:
                return df
            df = df.copy()
            df[column] = df[column].astype(str).str.replace("\xa0", " ", regex=False).str.strip()
            df[column] = df[column].replace(_FIXES)
            return df


ROOT_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = ROOT_DIR / "data" / "predictions"
REPORTS_DIR = ROOT_DIR / "reports"


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _clean_district_col(df: pd.DataFrame) -> pd.DataFrame:
    """Apply district name standardization on any district column present."""
    for col in ("distName", "district"):
        if col in df.columns:
            df = standardize_district_names(df, col)
    return df


def load_forecast_tables(base_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Load the prediction tables saved by the notebook pipeline."""

    base_path = Path(base_dir) if base_dir is not None else PREDICTIONS_DIR
    table_paths = {
        "forecast_summary": base_path / "forecast_summary.csv",
        "annual_investment_forecast": base_path / "annual_investment_forecast.csv",
        "district_investment_forecasts": base_path / "district_investment_forecasts.csv",
        "project_count_forecast": base_path / "project_count_forecast.csv",
        "investment_scores": base_path / "investment_scores.csv",
        "risk_scores": base_path / "risk_scores.csv",
        "forecasts": base_path / "forecasts.csv",
        "project_clusters": base_path / "project_clusters.csv",
        "district_model_evaluation": base_path / "district_model_evaluation.csv",
    }

    tables: dict[str, pd.DataFrame] = {}
    for name, path in table_paths.items():
        table = _safe_read_csv(path)
        if table is not None:
            table = _clean_district_col(table)
            tables[name] = table

    return tables


def load_report_images(reports_dir: str | Path | None = None) -> list[Path]:
    """Return the report image paths that exist in the repository."""

    base_path = Path(reports_dir) if reports_dir is not None else REPORTS_DIR
    if not base_path.exists():
        return []
    return sorted(base_path.rglob("*.png"))


def summarize_forecast_tables(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Build a few dashboard-friendly summary values from the loaded tables."""

    summary: dict[str, Any] = {}

    if "investment_scores" in tables and not tables["investment_scores"].empty:
        inv = tables["investment_scores"]
        if "final_score" in inv.columns and "distName" in inv.columns:
            top = inv.sort_values("final_score", ascending=False).iloc[0]
            summary["top_investment_district"] = top.get("distName")
            summary["top_investment_score"] = float(top.get("final_score", 0))

    if "risk_scores" in tables and not tables["risk_scores"].empty:
        risk = tables["risk_scores"]
        if "risk_flag" in risk.columns:
            summary["flagged_projects"] = int((risk["risk_flag"] == 1).sum())
        if "risk_category" in risk.columns:
            summary["risk_categories"] = risk["risk_category"].value_counts().to_dict()

    if "annual_investment_forecast" in tables and not tables["annual_investment_forecast"].empty:
        forecast = tables["annual_investment_forecast"]
        if "startProjectYear" in forecast.columns and "total_investment_forecast" in forecast.columns:
            summary["latest_year"] = int(forecast["startProjectYear"].max())
            summary["latest_forecast"] = float(
                forecast.sort_values("startProjectYear").iloc[-1]["total_investment_forecast"]
            )

    return summary


def forecast(args=None):
    """CLI compatibility wrapper."""

    tables = load_forecast_tables()
    print(f"Loaded forecast tables: {', '.join(sorted(tables)) if tables else 'none'}")


if __name__ == "__main__":
    forecast()
