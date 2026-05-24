# =============================================================
# src/train_clustering.py
# GujEstateAI — Phase 7: Module 4 — Project Clustering
#                         & Investment Hotspot Scoring
# =============================================================
# Same logic as 07_module4_clustering.ipynb
# but as a clean reusable script.
#
# Usage:
#   from src.train_clustering import train_clustering_model
#   km, scaler, names, scores = train_clustering_model(
#       "data/processed/features.csv"
#   )
#
# Or run directly:
#   python src/train_clustering.py
# =============================================================

import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")                           # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from features import FEATURES_CLUSTER
except ModuleNotFoundError:
    from src.features import FEATURES_CLUSTER


TARGET_K = 5                                     # optimal K from elbow + silhouette


# =============================================================
# HELPERS
# =============================================================

def _norm_0_10(series: pd.Series) -> pd.Series:
    """Normalise a numeric Series to the 0–10 range."""
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return pd.Series(5.0, index=series.index)
    return ((series - s_min) / (s_max - s_min) * 10).round(2)


def _assign_cluster_names(cluster_means: pd.DataFrame, k: int) -> dict:
    """
    Derive human-readable cluster labels from centroid statistics.

    Strategy — rank each cluster on cost, units and booking; map the
    extreme / distinctive ones first, then fill the remaining with a
    sensible default.
    """
    cm = cluster_means.copy()
    names: dict[int, str] = {}
    available_labels = [
        "Luxury High-Rise",
        "Affordable Residential",
        "Commercial Zones",
        "Large Plotted Schemes",
        "Mixed Development Mid-Range",
    ]

    # 1. Highest cost + high units → Luxury High-Rise
    max_cost_c = cm["log_cost"].idxmax()
    if cm.loc[max_cost_c, "log_units"] >= cm["log_units"].median():
        names[max_cost_c] = "Luxury High-Rise"

    # 2. Lowest cost → Affordable Residential
    min_cost_c = cm["log_cost"].idxmin()
    if min_cost_c not in names:
        names[min_cost_c] = "Affordable Residential"

    # 3. High avgCostPerSqFt → Commercial Zones
    remaining = [c for c in range(k) if c not in names]
    if remaining:
        high_sqft = cm.loc[remaining, "avgCostPerSqFt"].idxmax()
        if cm.loc[high_sqft, "avgCostPerSqFt"] >= cm["avgCostPerSqFt"].quantile(0.6):
            names[high_sqft] = "Commercial Zones"

    # 4. High duration → Large Plotted Schemes
    remaining = [c for c in range(k) if c not in names]
    if remaining:
        high_dur = cm.loc[remaining, "duration_months"].idxmax()
        if cm.loc[high_dur, "duration_months"] >= cm["duration_months"].quantile(0.5):
            names[high_dur] = "Large Plotted Schemes"

    # 5. Fill the rest
    for c in range(k):
        if c not in names:
            names[c] = "Mixed Development Mid-Range"

    return names


# =============================================================
# PLOTTING
# =============================================================

def _plot_elbow(K_range, inertias, reports_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(K_range, inertias, "o-", color="#1565C0", linewidth=2.5, markersize=8)
    ax.fill_between(K_range, inertias, alpha=0.1, color="#1565C0")
    ax.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax.set_ylabel("Inertia", fontsize=12)
    ax.set_title("Elbow Method — Finding Optimal K", fontsize=14, fontweight="bold")
    ax.set_xticks(list(K_range))
    ax.annotate(
        f"Elbow (K={TARGET_K})",
        xy=(TARGET_K, inertias[TARGET_K - 2]),
        xytext=(TARGET_K + 2, inertias[0]),
        arrowprops=dict(arrowstyle="->", color="#EF5350", lw=2),
        fontsize=12, color="#EF5350", fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "30_elbow_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_silhouette(K_range, sil_scores, reports_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(K_range, sil_scores, color="#64B5F6", edgecolor="#1565C0", linewidth=1.2)
    best_idx = int(np.argmax(sil_scores))
    bars[best_idx].set_color("#1565C0")
    ax.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_title("Silhouette Score by K", fontsize=14, fontweight="bold")
    ax.set_xticks(list(K_range))
    for i, (k, s) in enumerate(zip(K_range, sil_scores)):
        ax.text(k, s + 0.005, f"{s:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "31_silhouette_scores.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_pca(X_scaled, labels, km, k, cluster_names, colors, reports_dir):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(12, 8))
    for c in range(k):
        mask = labels == c
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=colors[c], label=cluster_names[c], alpha=0.4, s=15, edgecolors="none",
        )
    centroids_pca = pca.transform(km.cluster_centers_)
    ax.scatter(
        centroids_pca[:, 0], centroids_pca[:, 1],
        c="red", marker="X", s=200, edgecolors="black", linewidth=2, zorder=10, label="Centroids",
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
    ax.set_title("Project Clusters — PCA 2D Projection", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "32_cluster_pca_2d.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_cluster_heatmap(cluster_means, k, reports_dir):
    cluster_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        cluster_norm,
        annot=cluster_means.round(2).values, fmt="",
        cmap="YlOrRd", linewidths=1, linecolor="white",
        xticklabels=[f.replace("_", " ").title() for f in FEATURES_CLUSTER],
        yticklabels=[f"Cluster {i}" for i in range(k)],
        ax=ax, cbar_kws={"label": "Normalised Value"},
    )
    ax.set_title("Cluster Profiles — Feature Means", fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "33_cluster_profiles_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_cluster_by_project_type(df_clust, colors, reports_dir):
    ct = pd.crosstab(df_clust["cluster_label"], df_clust["projectType"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(12, 6))
    ct.plot(kind="barh", stacked=True, ax=ax,
            color=["#1565C0", "#4CAF50", "#FF9800", "#E91E63"], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Percentage (%)", fontsize=12)
    ax.set_ylabel("")
    ax.set_title("Project Type Distribution by Cluster", fontsize=14, fontweight="bold")
    ax.legend(title="Project Type", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_xlim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "34_cluster_by_project_type.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_cluster_by_district(df_clust, k, cluster_names, colors, reports_dir):
    top10 = df_clust["distName"].value_counts().head(10).index.tolist()
    df_top10 = df_clust[df_clust["distName"].isin(top10)]
    ct = pd.crosstab(df_top10["distName"], df_top10["cluster_label"])
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=(12, 7))
    ct.plot(kind="barh", stacked=True, ax=ax, color=colors[:k], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Number of Projects", fontsize=12)
    ax.set_ylabel("")
    ax.set_title("Cluster Distribution — Top 10 Districts", fontsize=14, fontweight="bold")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "35_cluster_by_district.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_radar(cluster_means, k, cluster_names, colors, reports_dir):
    radar_data = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
    feature_labels = [f.replace("_", "\n").title() for f in FEATURES_CLUSTER]
    angles = np.linspace(0, 2 * np.pi, len(FEATURES_CLUSTER), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for c in range(k):
        values = radar_data.loc[c].tolist() + [radar_data.loc[c].tolist()[0]]
        ax.plot(angles, values, "o-", linewidth=2, color=colors[c], label=cluster_names[c])
        ax.fill(angles, values, alpha=0.1, color=colors[c])

    ax.set_thetagrids(np.degrees(angles[:-1]), feature_labels, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title("Cluster Profiles — Radar Chart", fontsize=14, fontweight="bold", pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "36_cluster_radar.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_investment_bar(district_scores, reports_dir):
    top15 = district_scores.head(15).copy()
    fig, ax = plt.subplots(figsize=(12, 7))
    norm_s = (top15["final_score"] - top15["final_score"].min()) / \
             (top15["final_score"].max() - top15["final_score"].min() + 1e-9)
    cmap = plt.cm.RdYlGn
    bar_colors = [cmap(0.3 + 0.7 * v) for v in norm_s]

    bars = ax.barh(
        top15["distName"][::-1], top15["final_score"][::-1],
        color=bar_colors[::-1], edgecolor="white", linewidth=0.8, height=0.7,
    )
    for bar, score in zip(bars, top15["final_score"][::-1]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Investment Score (0–10)", fontsize=12)
    ax.set_title("Top 15 Districts — Investment Hotspot Scores", fontsize=14, fontweight="bold")
    ax.set_xlim(0, top15["final_score"].max() * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "37_investment_hotspot_scores.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _plot_investment_heatmap(district_scores, reports_dir):
    top15 = district_scores.head(15).set_index("distName")[
        ["growth_score", "demand_score", "invest_score", "value_score", "final_score"]
    ]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        top15, annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=1, linecolor="white",
        xticklabels=["Growth", "Demand", "Investment", "Value", "FINAL"],
        ax=ax, cbar_kws={"label": "Score (0–10)"},
    )
    ax.set_title("Investment Score Breakdown — Top 15 Districts", fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "38_investment_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================
# MAIN FUNCTION
# =============================================================

def train_clustering_model(
    features_path:   str = "data/processed/features.csv",
    model_out_path:  str = "models/clustering_model.pkl",
    reports_dir:     str = "reports/module4",
    predictions_dir: str = "data/predictions",
):
    """
    Train K-Means clustering model, generate cluster profiles,
    compute district investment scores, and save everything.

    Returns:
        km_model          : fitted KMeans
        scaler            : fitted StandardScaler
        cluster_names     : dict  {cluster_id: label}
        district_scores   : DataFrame of district investment scores
    """

    print("=" * 50)
    print("  Module 4 — Clustering & Investment Scoring")
    print("=" * 50)

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------
    df = pd.read_csv(features_path)
    print(f"\nLoaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    df_clust = df.dropna(subset=FEATURES_CLUSTER).copy()
    X = df_clust[FEATURES_CLUSTER]
    print(f"After dropping NaN: {df_clust.shape[0]:,} rows")
    print(f"Clustering features: {len(FEATURES_CLUSTER)}")

    # ----------------------------------------------------------
    # Scale
    # ----------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("\nStandardScaler fitted")

    # ----------------------------------------------------------
    # Elbow + Silhouette scan
    # ----------------------------------------------------------
    K_range = range(2, 11)
    inertias = []
    sil_scores = []

    print("\nScanning K = 2 … 10:")
    for k in K_range:
        km_temp = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels_temp = km_temp.fit_predict(X_scaled)
        inertias.append(km_temp.inertia_)
        sil = silhouette_score(X_scaled, labels_temp)
        sil_scores.append(sil)
        print(f"  K={k:2d}  |  Inertia = {km_temp.inertia_:>12,.0f}  |  Silhouette = {sil:.4f}")

    # ----------------------------------------------------------
    # Train final model
    # ----------------------------------------------------------
    km = KMeans(n_clusters=TARGET_K, random_state=42, n_init=10, max_iter=300)
    df_clust["cluster"] = km.fit_predict(X_scaled)
    final_sil = silhouette_score(X_scaled, df_clust["cluster"])

    print(f"\nFinal K-Means (K={TARGET_K})")
    print(f"  Inertia   : {km.inertia_:,.0f}")
    print(f"  Silhouette: {final_sil:.4f}")

    # ----------------------------------------------------------
    # Cluster profiling & naming
    # ----------------------------------------------------------
    cluster_means = df_clust.groupby("cluster")[FEATURES_CLUSTER].mean()
    cluster_names = _assign_cluster_names(cluster_means, TARGET_K)
    df_clust["cluster_label"] = df_clust["cluster"].map(cluster_names)

    print("\nCluster distribution:")
    for c, name in cluster_names.items():
        count = (df_clust["cluster"] == c).sum()
        pct = count / len(df_clust) * 100
        print(f"  {c}. {name:<30s} {count:>5,} ({pct:.1f}%)")

    # ----------------------------------------------------------
    # Investment hotspot scoring
    # ----------------------------------------------------------
    district_scores = df.groupby("distName").agg(
        project_count=("projectType", "count"),
        avg_booking=("booking_rate", "mean"),
        total_invest=("totalEstimatedCost", "sum"),
        avg_cost_sqft=("avgCostPerSqFt", "mean"),
        avg_duration=("duration_months", "mean"),
        total_units=("totalUnits", "sum"),
    ).reset_index()

    district_scores["growth_score"]  = _norm_0_10(district_scores["project_count"])
    district_scores["demand_score"]  = _norm_0_10(district_scores["avg_booking"])
    district_scores["invest_score"]  = _norm_0_10(district_scores["total_invest"])
    district_scores["value_score"]   = _norm_0_10(
        district_scores["avg_cost_sqft"].max() - district_scores["avg_cost_sqft"]
    )

    district_scores["final_score"] = (
        district_scores["growth_score"]  * 0.30
        + district_scores["demand_score"]  * 0.30
        + district_scores["invest_score"]  * 0.20
        + district_scores["value_score"]   * 0.20
    ).round(2)

    district_scores.sort_values("final_score", ascending=False, inplace=True)
    district_scores.reset_index(drop=True, inplace=True)

    print("\nTop 5 investment districts:")
    for i, (_, row) in enumerate(district_scores.head(5).iterrows()):
        print(f"  {i + 1}. {row['distName']:<20s} Score: {row['final_score']:.1f}/10")

    # ----------------------------------------------------------
    # Generate all plots
    # ----------------------------------------------------------
    os.makedirs(reports_dir, exist_ok=True)
    colors_cluster = ["#1565C0", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

    print("\nGenerating plots …")
    _plot_elbow(K_range, inertias, reports_dir)
    print("  ✔ 30_elbow_plot.png")
    _plot_silhouette(K_range, sil_scores, reports_dir)
    print("  ✔ 31_silhouette_scores.png")
    _plot_pca(X_scaled, df_clust["cluster"].values, km, TARGET_K, cluster_names, colors_cluster, reports_dir)
    print("  ✔ 32_cluster_pca_2d.png")
    _plot_cluster_heatmap(cluster_means, TARGET_K, reports_dir)
    print("  ✔ 33_cluster_profiles_heatmap.png")
    _plot_cluster_by_project_type(df_clust, colors_cluster, reports_dir)
    print("  ✔ 34_cluster_by_project_type.png")
    _plot_cluster_by_district(df_clust, TARGET_K, cluster_names, colors_cluster, reports_dir)
    print("  ✔ 35_cluster_by_district.png")
    _plot_radar(cluster_means, TARGET_K, cluster_names, colors_cluster, reports_dir)
    print("  ✔ 36_cluster_radar.png")
    _plot_investment_bar(district_scores, reports_dir)
    print("  ✔ 37_investment_hotspot_scores.png")
    _plot_investment_heatmap(district_scores, reports_dir)
    print("  ✔ 38_investment_heatmap.png")

    # ----------------------------------------------------------
    # Save model & outputs
    # ----------------------------------------------------------
    os.makedirs(os.path.dirname(model_out_path) if os.path.dirname(model_out_path) else ".", exist_ok=True)
    joblib.dump((km, scaler), model_out_path)

    os.makedirs(predictions_dir, exist_ok=True)

    cluster_output = df_clust[
        ["projectRegId", "projectName", "distName", "projectType", "cluster", "cluster_label"]
    ].copy()
    cluster_output.to_csv(os.path.join(predictions_dir, "cluster_assignments.csv"), index=False)

    district_scores.to_csv(os.path.join(predictions_dir, "investment_scores.csv"), index=False)

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print(f"\n{'=' * 50}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Algorithm       : K-Means")
    print(f"  Clusters (K)    : {TARGET_K}")
    print(f"  Silhouette      : {final_sil:.4f}")
    print(f"  Total Projects  : {len(df_clust):,}")
    print(f"  Model saved     : {model_out_path}")
    print(f"  Clusters CSV    : {os.path.join(predictions_dir, 'cluster_assignments.csv')}")
    print(f"  Scores CSV      : {os.path.join(predictions_dir, 'investment_scores.csv')}")
    print(f"  Reports         : {reports_dir}/ (9 plots)")

    return km, scaler, cluster_names, district_scores


# =============================================================
# Run directly: python src/train_clustering.py
# =============================================================
if __name__ == "__main__":
    km_model, scaler, names, scores = train_clustering_model(
        features_path   = "data/processed/features.csv",
        model_out_path  = "models/clustering_model.pkl",
        reports_dir     = "reports/module4",
        predictions_dir = "data/predictions",
    )
    print("\nDone!")
