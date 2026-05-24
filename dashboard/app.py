"""Streamlit dashboard for GujEstateAI."""

from __future__ import annotations

from pathlib import Path
import sys

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from src.forecast import load_forecast_tables, load_report_images, summarize_forecast_tables
from src.predict import load_models, predict_bundle


st.set_page_config(
	page_title="GujEstateAI Dashboard",
	page_icon="🏗️",
	layout="wide",
	initial_sidebar_state="expanded",
)


st.markdown(
	"""
	<style>
	.stApp {
		background:
			radial-gradient(circle at top left, rgba(255, 193, 7, 0.18), transparent 30%),
			radial-gradient(circle at top right, rgba(0, 150, 136, 0.12), transparent 25%),
			linear-gradient(180deg, #091521 0%, #0e1e2d 48%, #f7f5ef 48%, #f7f5ef 100%);
	}
	.block-container {
		padding-top: 1.2rem;
		padding-bottom: 2rem;
	}
	.hero {
		padding: 1.4rem 1.5rem;
		border-radius: 22px;
		background: linear-gradient(135deg, rgba(6, 26, 38, 0.96), rgba(18, 61, 82, 0.92));
		color: white;
		box-shadow: 0 18px 48px rgba(0, 0, 0, 0.18);
		border: 1px solid rgba(255, 255, 255, 0.1);
	}
	.hero h1 { margin: 0; font-size: 2.2rem; }
	.hero p { margin: 0.45rem 0 0 0; opacity: 0.9; font-size: 1rem; }
	.section-title {
		margin-top: 0.2rem;
		margin-bottom: 0.6rem;
		color: #11263b;
		font-size: 1.05rem;
		font-weight: 700;
		letter-spacing: 0.02em;
		text-transform: uppercase;
	}
	</style>
	""",
	unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_tables() -> dict[str, pd.DataFrame]:
	return load_forecast_tables()


@st.cache_resource(show_spinner=False)
def load_artifacts() -> tuple[dict, dict]:
	return load_tables(), load_models()


def money_fmt(value: float) -> str:
	return f"₹{value:,.0f}"


def render_header() -> None:
	st.markdown(
		"""
		<div class="hero">
			<h1>GujEstateAI Dashboard</h1>
			<p>Forecasts, investment scoring, risk signals, and model outputs for Gujarat real estate.</p>
		</div>
		""",
		unsafe_allow_html=True,
	)


def build_sidebar(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
	st.sidebar.title("GujEstateAI")
	st.sidebar.caption("Gujarat real estate intelligence")

	districts = []
	for key in ("investment_scores", "risk_scores", "district_investment_forecasts", "forecast_summary"):
		if key in tables and "distName" in tables[key].columns:
			districts.extend(tables[key]["distName"].dropna().astype(str).tolist())
		if key in tables and "district" in tables[key].columns:
			districts.extend(tables[key]["district"].dropna().astype(str).tolist())
	district_options = ["All Gujarat"] + sorted(set(districts))

	selected_district = st.sidebar.selectbox("District", district_options)
	top_n = st.sidebar.slider("Top N", 5, 20, 10)

	st.sidebar.divider()
	st.sidebar.write(f"Prediction tables: {len(tables)}")
	st.sidebar.write(f"Report images: {len(load_report_images())}")

	return {"district": selected_district, "top_n": top_n}


def render_overview(tables: dict[str, pd.DataFrame], filters: dict[str, object]) -> None:
	st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)

	col1, col2, col3, col4 = st.columns(4)
	with col1:
		projects = 0
		if "forecasts" in tables:
			projects = int(tables["forecasts"].query("type == 'historical'")["project_count"].sum())
		st.metric("Historical projects", f"{projects:,}")
	with col2:
		districts = tables["investment_scores"]["distName"].nunique() if "investment_scores" in tables else 0
		st.metric("Districts scored", f"{districts:,}")
	with col3:
		if "investment_scores" in tables and not tables["investment_scores"].empty:
			top = tables["investment_scores"].sort_values("final_score", ascending=False).iloc[0]
			st.metric("Top investment district", str(top["distName"]), f"Score {top['final_score']:.1f}")
		else:
			st.metric("Top investment district", "N/A")
	with col4:
		if "risk_scores" in tables and not tables["risk_scores"].empty:
			flagged = int((tables["risk_scores"]["risk_flag"] == 1).sum())
			st.metric("Flagged projects", f"{flagged:,}")
		else:
			st.metric("Flagged projects", "N/A")

	col_left, col_right = st.columns([1.25, 0.85])
	with col_left:
		if "investment_scores" in tables and not tables["investment_scores"].empty:
			inv = tables["investment_scores"].head(int(filters["top_n"]))
			fig = px.bar(
				inv.sort_values("final_score", ascending=True),
				x="final_score",
				y="distName",
				orientation="h",
				color="final_score",
				color_continuous_scale=["#F59E0B", "#0F766E"],
				title="Top district investment scores",
			)
			fig.update_layout(height=460, margin=dict(l=10, r=10, t=60, b=10), coloraxis_showscale=False)
			st.plotly_chart(fig, use_container_width=True)
	with col_right:
		if "risk_scores" in tables and not tables["risk_scores"].empty:
			risk = tables["risk_scores"]
			category_counts = risk["risk_category"].value_counts().reset_index()
			category_counts.columns = ["risk_category", "count"]
			fig = px.pie(
				category_counts,
				names="risk_category",
				values="count",
				hole=0.52,
				color_discrete_sequence=["#0F766E", "#F59E0B", "#DC2626", "#2563EB"],
				title="Risk distribution",
			)
			fig.update_layout(height=460, margin=dict(l=10, r=10, t=60, b=10))
			st.plotly_chart(fig, use_container_width=True)


def render_forecasts(tables: dict[str, pd.DataFrame], filters: dict[str, object]) -> None:
	st.markdown('<div class="section-title">Forecasts</div>', unsafe_allow_html=True)

	col1, col2 = st.columns(2)
	with col1:
		if "annual_investment_forecast" in tables and not tables["annual_investment_forecast"].empty:
			annual = tables["annual_investment_forecast"]
			fig = px.line(annual, x="startProjectYear", y="total_investment_forecast", markers=True, title="Gujarat annual investment forecast")
			fig.update_traces(line=dict(color="#0F766E", width=3))
			fig.update_layout(height=380, margin=dict(l=10, r=10, t=60, b=10), yaxis_tickprefix="₹")
			st.plotly_chart(fig, use_container_width=True)
	with col2:
		if "project_count_forecast" in tables and not tables["project_count_forecast"].empty:
			counts = tables["project_count_forecast"]
			fig = px.bar(
				counts,
				x="startProjectYear",
				y="project_count_forecast",
				title="Project count forecast",
				color="project_count_forecast",
				color_continuous_scale=["#F59E0B", "#0F766E"],
			)
			fig.update_layout(height=380, margin=dict(l=10, r=10, t=60, b=10), coloraxis_showscale=False)
			st.plotly_chart(fig, use_container_width=True)

	selected_district = filters["district"]
	if selected_district != "All Gujarat" and "district_investment_forecasts" in tables:
		district_forecasts = tables["district_investment_forecasts"]
		district_forecasts = district_forecasts[district_forecasts["distName"].astype(str) == str(selected_district)]
		if not district_forecasts.empty:
			fig = px.line(district_forecasts, x="startProjectYear", y="forecast_total_investment", color="distName", markers=True, title=f"District investment forecast - {selected_district}")
			fig.update_layout(height=380, margin=dict(l=10, r=10, t=60, b=10), yaxis_tickprefix="₹")
			st.plotly_chart(fig, use_container_width=True)

	if "forecast_summary" in tables and not tables["forecast_summary"].empty:
		summary = tables["forecast_summary"]
		summary = summary.loc[:, [c for c in summary.columns if c in ["district", "actual_2023", "forecast_2024", "forecast_2025", "forecast_2026"]]]
		if not summary.empty:
			show = summary.head(int(filters["top_n"]))
			fig = go.Figure()
			for col, color in zip(["actual_2023", "forecast_2024", "forecast_2025", "forecast_2026"], ["#0F766E", "#F59E0B", "#2563EB", "#DC2626"]):
				if col in show.columns:
					fig.add_trace(go.Bar(name=col.replace("_", " ").title(), x=show["district"], y=show[col], marker_color=color))
			fig.update_layout(barmode="group", height=420, margin=dict(l=10, r=10, t=60, b=10), title="Forecast summary snapshot")
			st.plotly_chart(fig, use_container_width=True)


def render_scores_and_clusters(tables: dict[str, pd.DataFrame], filters: dict[str, object]) -> None:
	st.markdown('<div class="section-title">Scores and clusters</div>', unsafe_allow_html=True)

	col1, col2 = st.columns(2)
	with col1:
		if "investment_scores" in tables and not tables["investment_scores"].empty:
			scores = tables["investment_scores"].head(int(filters["top_n"]))
			fig = px.scatter(
				scores,
				x="avg_cost_sqft",
				y="final_score",
				size="project_count",
				color="final_score",
				hover_name="distName",
				color_continuous_scale=["#F59E0B", "#0F766E"],
				title="Investment score profile",
			)
			fig.update_layout(height=390, margin=dict(l=10, r=10, t=60, b=10), coloraxis_showscale=False)
			st.plotly_chart(fig, use_container_width=True)
	with col2:
		if "project_clusters" in tables and not tables["project_clusters"].empty:
			clusters = tables["project_clusters"].copy()
			if "cluster_label" in clusters.columns:
				counts = clusters["cluster_label"].value_counts().reset_index()
				counts.columns = ["cluster_label", "count"]
				fig = px.bar(
					counts,
					x="count",
					y="cluster_label",
					orientation="h",
					title="Cluster distribution",
					color="count",
					color_continuous_scale=["#2563EB", "#0F766E"],
				)
				fig.update_layout(height=390, margin=dict(l=10, r=10, t=60, b=10), coloraxis_showscale=False)
				st.plotly_chart(fig, use_container_width=True)

	if "district_model_evaluation" in tables and not tables["district_model_evaluation"].empty:
		evaluation = tables["district_model_evaluation"]
		best = evaluation.sort_values("R2", ascending=False).head(int(filters["top_n"]))
		st.dataframe(best, use_container_width=True, hide_index=True)


def render_risk(tables: dict[str, pd.DataFrame], filters: dict[str, object]) -> None:
	st.markdown('<div class="section-title">Risk Analysis</div>', unsafe_allow_html=True)

	if "risk_scores" not in tables or tables["risk_scores"].empty:
		st.info("No risk score table found.")
		return

	risk = tables["risk_scores"].copy()
	risk["risk_category"] = risk["risk_category"].fillna("Unknown")
	risk["risk_flag"] = risk["risk_flag"].fillna(0)

	col1, col2 = st.columns([1.2, 0.8])
	with col1:
		fig = px.histogram(
			risk,
			x="risk_score",
			nbins=30,
			color_discrete_sequence=["#0F766E"],
			title="Risk score distribution",
		)
		fig.update_layout(height=390, margin=dict(l=10, r=10, t=60, b=10))
		st.plotly_chart(fig, use_container_width=True)
	with col2:
		category_counts = risk["risk_category"].value_counts().reset_index()
		category_counts.columns = ["risk_category", "count"]
		fig = go.Figure(
			data=[
				go.Bar(
					x=category_counts["risk_category"],
					y=category_counts["count"],
					marker_color=["#0F766E", "#F59E0B", "#DC2626", "#2563EB"][: len(category_counts)],
				),
			]
		)
		fig.update_layout(
			title="Risk category breakdown",
			height=390,
			margin=dict(l=10, r=10, t=60, b=10),
			xaxis_title="Risk category",
			yaxis_title="Count",
		)
		st.plotly_chart(fig, use_container_width=True)

	top_risk = risk.sort_values("risk_score", ascending=False).head(int(filters["top_n"]))
	st.dataframe(top_risk, use_container_width=True, hide_index=True)


def render_predictions(models: dict[str, object]) -> None:
	st.markdown('<div class="section-title">Prediction Lab</div>', unsafe_allow_html=True)
	st.caption("Enter a project profile to get a duration and cost estimate from the saved models.")

	if not models:
		st.warning("Model artifacts are not available in the models folder.")
		return

	col1, col2 = st.columns(2)
	with col1:
		st.subheader("Project details")
		project_type = st.selectbox("Project type", ["Residential/Group Housing", "Commercial", "Mixed Development", "Plotted Development"], key="pred_project_type")
		district = st.selectbox("District", ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Gandhinagar", "Bhavnagar", "Other"], key="pred_district")
		promoter_type = st.selectbox("Promoter type", ["Partnership", "Company", "Individual", "Other"], key="pred_promoter_type")
		under_redevelopment = st.selectbox("Under redevelopment", ["NO", "YES"], key="pred_redevelop")
		total_units = st.number_input("Total units", min_value=1, value=120, step=1)
		total_land_cost = st.number_input("Total land cost", min_value=0.0, value=15000000.0, step=100000.0, format="%.0f")
		total_estimated_cost = st.number_input("Estimated total cost", min_value=0.0, value=250000000.0, step=5000000.0, format="%.0f")
		start_year = st.selectbox("Start year", [2021, 2022, 2023, 2024, 2025, 2026], index=2)
		start_month = st.selectbox("Start month", list(range(1, 13)), index=0)

	with col2:
		st.subheader("Operational inputs")
		no_of_inventory = st.number_input("Inventory count", min_value=1.0, value=float(total_units), step=1.0)
		avg_units = st.number_input("Average units", min_value=1.0, value=float(total_units), step=1.0)
		total_carpet_area = st.number_input("Total carpet area", min_value=0.0, value=0.0, step=100.0)
		total_builtup_area = st.number_input("Total built-up area", min_value=0.0, value=0.0, step=100.0)
		total_sqft_build = st.number_input("Total square foot build", min_value=0.0, value=0.0, step=100.0)
		avg_sqft_build = st.number_input("Average square foot build", min_value=0.0, value=0.0, step=10.0)
		booking_rate = st.slider("Booking rate", 0.0, 1.0, 0.5, 0.01)
		duration_months = st.number_input("Duration months", min_value=0.0, value=48.0, step=1.0)

	payload = {
		"projectType": project_type,
		"distName": district,
		"promoter_type_simple": promoter_type,
		"underRedevelopment": under_redevelopment,
		"totalUnits": total_units,
		"totalLandCost": total_land_cost,
		"totalEstimatedCost": total_estimated_cost,
		"startProjectYear": start_year,
		"startProjectMonth": start_month,
		"noOfInventory": no_of_inventory,
		"avgUnits": avg_units,
		"totalCarpetArea_form3A": total_carpet_area,
		"totalBuiltupArea_form3A": total_builtup_area,
		"totalSquareFootBuild": total_sqft_build,
		"AvgSquareFootBuild": avg_sqft_build,
		"booking_rate": booking_rate,
		"duration_months": duration_months,
	}

	if st.button("Run prediction", type="primary"):
		result = predict_bundle(payload, models)
		pred_col1, pred_col2 = st.columns(2)
		with pred_col1:
			if result["duration_months"] is not None:
				st.success(f"Predicted duration: {result['duration_months']:.1f} months")
			else:
				st.warning("Duration model not available.")
		with pred_col2:
			if result["totalEstimatedCost"] is not None:
				st.success(f"Predicted cost: {money_fmt(result['totalEstimatedCost'])}")
			else:
				st.warning("Cost model not available.")


def render_reports() -> None:
	st.markdown('<div class="section-title">Reports Gallery</div>', unsafe_allow_html=True)
	image_paths = load_report_images()
	if not image_paths:
		st.info("No report images found.")
		return

	gallery = [path for path in image_paths if path.name.startswith(("01_", "03_", "09_", "15_", "16_", "20_", "21_", "28_", "30_", "37_", "38_", "39_", "40_", "41_", "42_"))]
	if not gallery:
		gallery = image_paths[:8]

	cols = st.columns(2)
	for idx, image_path in enumerate(gallery[:8]):
		with cols[idx % 2]:
			st.image(str(image_path), caption=image_path.name, use_column_width=True)


def main() -> None:
	tables, models = load_artifacts()
	filters = build_sidebar(tables)
	render_header()

	summary = summarize_forecast_tables(tables)
	if summary:
		cols = st.columns(4)
		with cols[0]:
			if "top_investment_district" in summary:
				st.metric("Top district", str(summary["top_investment_district"]))
		with cols[1]:
			if "top_investment_score" in summary:
				st.metric("Top score", f"{summary['top_investment_score']:.1f}")
		with cols[2]:
			if "latest_year" in summary:
				st.metric("Latest forecast year", str(summary["latest_year"]))
		with cols[3]:
			if "latest_forecast" in summary:
				st.metric("Latest investment forecast", money_fmt(summary["latest_forecast"]))

	tabs = st.tabs(["Overview", "Forecasts", "Scores", "Risk", "Prediction Lab", "Reports"])
	with tabs[0]:
		render_overview(tables, filters)
	with tabs[1]:
		render_forecasts(tables, filters)
	with tabs[2]:
		render_scores_and_clusters(tables, filters)
	with tabs[3]:
		render_risk(tables, filters)
	with tabs[4]:
		render_predictions(models)
	with tabs[5]:
		render_reports()


if __name__ == "__main__":
	main()
