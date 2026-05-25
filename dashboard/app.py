"""GujEstateAI — Redesigned Premium Dashboard
A luxury dark-theme real estate intelligence dashboard for Gujarat.
"""

from __future__ import annotations

import warnings
from pathlib import Path
import sys
from typing import Any

warnings.filterwarnings("ignore")

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.forecast import load_forecast_tables, load_report_images, summarize_forecast_tables
from src.predict import load_models, predict_bundle

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="GujEstateAI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DESIGN SYSTEM ─────────────────────────────────────────────
# Theme: Luxury Property Intelligence — Obsidian & Champagne Gold
# Fonts: Cormorant Garamond (display) + DM Sans (body)
# Palette: Deep charcoal #0D0D0D, Champagne #D4A853, Ivory #F5F0E8
#          Steel Blue #2563EB, Emerald #059669, Rose #E11D48

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;0,700;1,300;1,400&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* ─── GLOBAL RESET ────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    font-family: 'DM Sans', sans-serif;
    background: #0D0D0D;
    color: #E8E3DA;
}

/* Texture overlay via pseudo-element simulation */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 0%, rgba(212, 168, 83, 0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(37, 99, 235, 0.05) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

.block-container {
    padding: 1.5rem 2rem 3rem 2rem;
    max-width: 1600px;
}

/* ─── SIDEBAR ─────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 1px solid rgba(212, 168, 83, 0.15) !important;
}

[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'Cormorant Garamond', serif;
    font-weight: 600;
    font-size: 1.4rem;
    color: #D4A853;
    letter-spacing: 0.04em;
    margin-bottom: 0;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stCaption {
    color: #6B6560 !important;
    font-size: 0.75rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(212, 168, 83, 0.12) !important;
    margin: 1rem 0;
}

[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: #9E9890 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ─── SIDEBAR LOGO BADGE ─────────────────────────── */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.3rem;
}

.sidebar-icon {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #D4A853, #B8860B);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    box-shadow: 0 4px 12px rgba(212, 168, 83, 0.3);
    flex-shrink: 0;
}

/* ─── HERO HEADER ─────────────────────────────────── */
.hero-wrapper {
    position: relative;
    padding: 2.8rem 3rem 2.4rem 3rem;
    margin-bottom: 2rem;
    border-radius: 20px;
    overflow: hidden;
    background:
        linear-gradient(135deg, #161310 0%, #1A1510 40%, #131A1F 100%);
    border: 1px solid rgba(212, 168, 83, 0.2);
    box-shadow:
        0 0 0 1px rgba(255,255,255,0.03),
        0 30px 80px rgba(0,0,0,0.5),
        inset 0 1px 0 rgba(212, 168, 83, 0.1);
}

.hero-wrapper::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(212, 168, 83, 0.12) 0%, transparent 65%);
    pointer-events: none;
}

.hero-wrapper::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(212, 168, 83, 0.4), transparent);
}

.hero-eyebrow {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #D4A853;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 24px; height: 1px;
    background: #D4A853;
}

.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.4rem;
    font-weight: 600;
    line-height: 1.1;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.01em;
    color: #F5F0E8;
}

.hero-title em {
    font-style: italic;
    color: #D4A853;
}

.hero-subtitle {
    font-size: 0.92rem;
    color: #6B6560;
    font-weight: 400;
    letter-spacing: 0.02em;
    max-width: 540px;
    line-height: 1.6;
    margin: 0.6rem 0 0 0;
}

.hero-badges {
    display: flex;
    gap: 10px;
    margin-top: 1.6rem;
    flex-wrap: wrap;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    border: 1px solid;
}

.badge-gold {
    background: rgba(212, 168, 83, 0.1);
    border-color: rgba(212, 168, 83, 0.35);
    color: #D4A853;
}

.badge-blue {
    background: rgba(37, 99, 235, 0.1);
    border-color: rgba(37, 99, 235, 0.35);
    color: #60A5FA;
}

.badge-green {
    background: rgba(5, 150, 105, 0.1);
    border-color: rgba(5, 150, 105, 0.35);
    color: #34D399;
}

.hero-stat-row {
    position: absolute;
    right: 3rem;
    top: 50%;
    transform: translateY(-50%);
    display: flex;
    flex-direction: column;
    gap: 1.2rem;
    text-align: right;
}

.hero-stat-num {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.2rem;
    font-weight: 600;
    color: #F5F0E8;
    line-height: 1;
}

.hero-stat-label {
    font-size: 0.65rem;
    color: #6B6560;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ─── KPI CARDS ───────────────────────────────────── */
div[data-testid="stMetric"] {
    background: #141414 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 16px !important;
    padding: 20px 22px !important;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}

div[data-testid="stMetric"]:hover {
    border-color: rgba(212, 168, 83, 0.25) !important;
}

div[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #D4A853, transparent);
    opacity: 0.5;
}

div[data-testid="stMetric"] label {
    color: #5A5550 !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #F5F0E8 !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 2rem !important;
    font-weight: 600 !important;
    line-height: 1.1 !important;
}

div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
    font-size: 0.75rem !important;
    font-weight: 500 !important;
}

/* ─── SECTION HEADINGS ────────────────────────────── */
.section-head {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 2rem 0 1.2rem 0;
}

.section-head-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(212, 168, 83, 0.3), transparent);
}

.section-head-text {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #D4A853;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    white-space: nowrap;
}

.section-head-icon {
    font-size: 1rem;
    color: #D4A853;
}

/* ─── CHART CONTAINERS ────────────────────────────── */
.chart-card {
    background: #111111;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}

.chart-card::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 18px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
    pointer-events: none;
}

.chart-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #5A5550;
    margin-bottom: 0.3rem;
}

.chart-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.25rem;
    font-weight: 600;
    color: #E8E3DA;
    margin-bottom: 1rem;
}

/* ─── TABS ────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #111111 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
    padding: 5px !important;
    gap: 2px !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: #5A5550 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.03em !important;
    padding: 8px 18px !important;
    transition: all 0.2s !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #D4A853 !important;
    background: rgba(212, 168, 83, 0.06) !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #1C1710, #1A1508) !important;
    color: #D4A853 !important;
    border: 1px solid rgba(212, 168, 83, 0.25) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
}

/* ─── PREDICTION RESULT CARDS ─────────────────────── */
.pred-card {
    background: #111111;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 24px 28px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}

.pred-card:hover {
    border-color: rgba(212, 168, 83, 0.2);
}

.pred-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 60px; height: 3px;
    background: linear-gradient(90deg, #D4A853, #B8860B);
}

.pred-card.success::before { background: linear-gradient(90deg, #059669, #34D399); }
.pred-card.risk::before    { background: linear-gradient(90deg, #E11D48, #FB7185); }
.pred-card.info::before    { background: linear-gradient(90deg, #2563EB, #60A5FA); }

.pred-card-icon {
    font-size: 1.8rem;
    margin-bottom: 10px;
    display: block;
}

.pred-card-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5A5550;
    margin-bottom: 6px;
}

.pred-card-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.4rem;
    font-weight: 600;
    color: #F5F0E8;
    line-height: 1.1;
}

.pred-card-sub {
    font-size: 0.78rem;
    color: #5A5550;
    margin-top: 6px;
}

/* ─── RISK STATUS PILL ────────────────────────────── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 8px;
}

.pill-danger { background: rgba(225, 29, 72, 0.15); color: #FB7185; border: 1px solid rgba(225, 29, 72, 0.3); }
.pill-safe   { background: rgba(5, 150, 105, 0.15); color: #34D399; border: 1px solid rgba(5, 150, 105, 0.3); }
.pill-warn   { background: rgba(245, 158, 11, 0.15); color: #FCD34D; border: 1px solid rgba(245, 158, 11, 0.3); }

/* ─── INFO BOX ────────────────────────────────────── */
.stInfo {
    background: rgba(37, 99, 235, 0.08) !important;
    border: 1px solid rgba(37, 99, 235, 0.2) !important;
    border-radius: 12px !important;
    color: #93C5FD !important;
}

/* ─── DATAFRAME ───────────────────────────────────── */
.stDataFrame {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

/* ─── BUTTON ──────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #D4A853 0%, #B8860B 100%) !important;
    color: #0D0D0D !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    box-shadow: 0 6px 24px rgba(212, 168, 83, 0.3) !important;
    transition: all 0.2s !important;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 32px rgba(212, 168, 83, 0.4) !important;
}

/* ─── INPUTS / SELECTS ────────────────────────────── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: #1A1A1A !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #E8E3DA !important;
}

.stSelectbox label,
.stNumberInput label,
.stSlider label {
    color: #5A5550 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
}

/* ─── DIVIDER ─────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.06) !important;
    margin: 1.5rem 0 !important;
}

/* ─── SCROLLBAR ───────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0D0D0D; }
::-webkit-scrollbar-thumb { background: #2A2520; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #3A3028; }

/* ─── KPI STRIP ───────────────────────────────────── */
.kpi-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 2rem;
}

.kpi-item {
    background: #111111;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 14px;
    padding: 16px 20px;
    position: relative;
    overflow: hidden;
}

.kpi-item::before {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, rgba(212, 168, 83, 0.25), transparent);
}

.kpi-item-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4A4540;
    margin-bottom: 6px;
}

.kpi-item-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.7rem;
    font-weight: 600;
    color: #D4A853;
    line-height: 1;
}

.kpi-item-sub {
    font-size: 0.7rem;
    color: #4A4540;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)


# ── PLOTLY TEMPLATE ──────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#9E9890", size=11),
    title_font=dict(family="Cormorant Garamond", size=18, color="#F5F0E8"),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        linecolor="rgba(255,255,255,0.06)",
        tickcolor="rgba(255,255,255,0.06)",
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        linecolor="rgba(255,255,255,0.06)",
        tickcolor="rgba(255,255,255,0.06)",
    ),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.06)"),
    margin=dict(l=10, r=10, t=60, b=10),
)

GOLD_SCALE = ["#1A1508", "#2E2210", "#8B6914", "#D4A853", "#F0CC85"]
CHART_COLORS = ["#D4A853", "#2563EB", "#059669", "#E11D48", "#8B5CF6", "#F59E0B", "#06B6D4", "#F472B6"]


# ── Helpers ──────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_tables() -> dict[str, pd.DataFrame]:
    return load_forecast_tables()

@st.cache_resource(show_spinner=False)
def _load_models() -> dict[str, Any]:
    return load_models()

@st.cache_data(show_spinner=False)
def _load_images() -> list[Path]:
    return load_report_images()

def _safe_col(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and not df[col].dropna().empty

def money_fmt(value: float) -> str:
    if abs(value) >= 1e9:  return f"₹{value/1e9:,.2f}B"
    if abs(value) >= 1e7:  return f"₹{value/1e7:,.2f} Cr"
    if abs(value) >= 1e5:  return f"₹{value/1e5:,.1f}L"
    return f"₹{value:,.0f}"

def section_head(icon: str, title: str) -> None:
    st.markdown(f"""
    <div class="section-head">
        <span class="section-head-icon">{icon}</span>
        <span class="section-head-text">{title}</span>
        <div class="section-head-line"></div>
    </div>""", unsafe_allow_html=True)


# ── SIDEBAR ──────────────────────────────────────────────────
def build_sidebar(tables: dict[str, pd.DataFrame]) -> dict[str, object]:
    st.sidebar.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-icon">🏛️</div>
        <div>
            <div style="font-family:'Cormorant Garamond',serif;font-weight:600;font-size:1.3rem;color:#D4A853;line-height:1">GujEstateAI</div>
            <div style="font-size:0.65rem;color:#4A4540;letter-spacing:0.08em;text-transform:uppercase">Intelligence Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.divider()

    districts: set[str] = set()
    for df in tables.values():
        for col in ("distName", "district"):
            if col in df.columns:
                districts.update(df[col].dropna().astype(str).unique())

    selected_district = st.sidebar.selectbox(
        "🗺️ District", ["All Gujarat"] + sorted(districts)
    )
    top_n = st.sidebar.slider("Top N Items", 5, 25, 10)

    st.sidebar.divider()

    images = _load_images()
    st.sidebar.markdown(f"""
    <div style="background:#161413;border:1px solid rgba(255,255,255,0.05);border-radius:12px;padding:14px 16px;">
        <div style="font-size:0.62rem;color:#4A4540;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;font-weight:700">Data Status</div>
        <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <span style="font-size:0.78rem;color:#9E9890">Tables Loaded</span>
            <span style="font-size:0.78rem;color:#D4A853;font-weight:600">{len(tables)}</span>
        </div>
        <div style="display:flex;justify-content:space-between">
            <span style="font-size:0.78rem;color:#9E9890">Report Images</span>
            <span style="font-size:0.78rem;color:#D4A853;font-weight:600">{len(images)}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    return {"district": selected_district, "top_n": top_n}


# ── HERO HEADER ──────────────────────────────────────────────
def render_header(summary: dict) -> None:
    top_dist = summary.get("top_investment_district", "—")
    top_score = summary.get("top_investment_score", 0)
    latest_yr = summary.get("latest_year", "—")
    latest_fc = money_fmt(summary.get("latest_forecast", 0)) if summary.get("latest_forecast") else "—"

    st.markdown(f"""
    <div class="hero-wrapper">
        <div class="hero-eyebrow">Gujarat Real Estate Intelligence</div>
        <h1 class="hero-title">Property <em>Analytics</em><br>Powered by AI</h1>
        <p class="hero-subtitle">
            AI-driven forecasts, investment scoring, cluster intelligence,
            and risk detection across all Gujarat districts — in one command centre.
        </p>
        <div class="hero-badges">
            <span class="hero-badge badge-gold">⚡ Live Prediction</span>
            <span class="hero-badge badge-blue">📊 5-Year Forecast</span>
            <span class="hero-badge badge-green">🔬 ML Clustering</span>
        </div>

        <div class="hero-stat-row">
            <div>
                <div class="hero-stat-num">{top_dist}</div>
                <div class="hero-stat-label">Top District</div>
            </div>
            <div>
                <div class="hero-stat-num">{latest_fc}</div>
                <div class="hero-stat-label">Latest Forecast</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════
def render_overview(tables: dict, filters: dict) -> None:
    section_head("◈", "Market Overview")

    # KPIs
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        count = 0
        if "forecasts" in tables and _safe_col(tables["forecasts"], "project_count"):
            try:
                count = int(tables["forecasts"].query("type == 'historical'")["project_count"].sum())
            except Exception:
                count = int(tables["forecasts"]["project_count"].sum())
        st.metric("Total Projects", f"{count:,}")

    with c2:
        n_dist = 0
        if "investment_scores" in tables and _safe_col(tables["investment_scores"], "distName"):
            n_dist = tables["investment_scores"]["distName"].nunique()
        st.metric("Districts Scored", f"{n_dist:,}")

    with c3:
        if "investment_scores" in tables and _safe_col(tables["investment_scores"], "final_score"):
            top = tables["investment_scores"].sort_values("final_score", ascending=False).iloc[0]
            st.metric("Top Investment District", str(top["distName"]), f"Score {top['final_score']:.1f}")
        else:
            st.metric("Top Investment District", "—")

    with c4:
        if "risk_scores" in tables and _safe_col(tables["risk_scores"], "risk_flag"):
            flagged = int((tables["risk_scores"]["risk_flag"] == 1).sum())
            total = len(tables["risk_scores"])
            st.metric("Risk-Flagged Projects", f"{flagged:,}", f"of {total:,} total")
        else:
            st.metric("Risk-Flagged Projects", "—")

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1.4, 0.6])

    with col_left:
        if "investment_scores" in tables and _safe_col(tables["investment_scores"], "final_score"):
            inv = tables["investment_scores"].head(int(filters["top_n"])).copy()
            inv = inv.sort_values("final_score", ascending=True)
            fig = go.Figure(go.Bar(
                x=inv["final_score"], y=inv["distName"],
                orientation="h",
                marker=dict(
                    color=inv["final_score"],
                    colorscale=[[0, "#1A1508"], [0.5, "#8B6914"], [1, "#D4A853"]],
                    line=dict(width=0),
                ),
                hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>",
            ))
            fig.update_layout(
                **PLOT_LAYOUT,
                height=480,
                title="District Investment Rankings",
                xaxis_title="Investment Score",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        if "risk_scores" in tables and _safe_col(tables["risk_scores"], "risk_category"):
            risk = tables["risk_scores"]
            cat_counts = risk["risk_category"].value_counts().reset_index()
            cat_counts.columns = ["risk_category", "count"]
            color_map = {
                "Critical": "#E11D48", "High": "#F59E0B",
                "Medium": "#2563EB", "Low": "#059669", "Unknown": "#4A4540"
            }
            colors = [color_map.get(c, "#4A4540") for c in cat_counts["risk_category"]]
            fig = go.Figure(go.Pie(
                labels=cat_counts["risk_category"],
                values=cat_counts["count"],
                hole=0.65,
                marker=dict(colors=colors, line=dict(color="#0D0D0D", width=2)),
                hovertemplate="<b>%{label}</b><br>%{value} projects (%{percent})<extra></extra>",
            ))
            fig.add_annotation(
                text="Risk<br>Map", x=0.5, y=0.5, showarrow=False,
                font=dict(family="Cormorant Garamond", size=16, color="#D4A853"),
            )
            fig.update_layout(**PLOT_LAYOUT, height=480, title="Risk Distribution")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 2: FORECASTS
# ══════════════════════════════════════════════════════════════
def render_forecasts(tables: dict, filters: dict) -> None:
    section_head("◈", "Market Forecasts")

    c1, c2 = st.columns(2)

    with c1:
        if "annual_investment_forecast" in tables:
            annual = tables["annual_investment_forecast"]
            if _safe_col(annual, "startProjectYear") and _safe_col(annual, "total_investment_forecast"):
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=annual["startProjectYear"],
                    y=annual["total_investment_forecast"],
                    mode="lines+markers",
                    line=dict(color="#D4A853", width=3),
                    marker=dict(size=8, color="#D4A853", line=dict(color="#0D0D0D", width=2)),
                    fill="tozeroy",
                    fillcolor="rgba(212,168,83,0.06)",
                    hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>",
                ))
                fig.update_layout(
                    **PLOT_LAYOUT, height=400,
                    title="Annual Investment Forecast",
                    yaxis_tickprefix="₹",
                )
                st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "project_count_forecast" in tables:
            counts = tables["project_count_forecast"]
            if _safe_col(counts, "startProjectYear") and _safe_col(counts, "project_count_forecast"):
                fig = go.Figure(go.Bar(
                    x=counts["startProjectYear"],
                    y=counts["project_count_forecast"],
                    marker=dict(
                        color=counts["project_count_forecast"],
                        colorscale=[[0, "#1A1508"], [1, "#D4A853"]],
                        line=dict(width=0),
                    ),
                    hovertemplate="<b>%{x}</b><br>%{y:,} projects<extra></extra>",
                ))
                fig.update_layout(**PLOT_LAYOUT, height=400, title="Project Count Forecast")
                st.plotly_chart(fig, use_container_width=True)

    # District-specific
    selected = filters["district"]
    if selected != "All Gujarat" and "district_investment_forecasts" in tables:
        df_dist = tables["district_investment_forecasts"]
        if _safe_col(df_dist, "distName"):
            df_f = df_dist[df_dist["distName"].astype(str) == str(selected)]
            if not df_f.empty and _safe_col(df_f, "forecast_total_investment"):
                fig = go.Figure(go.Scatter(
                    x=df_f["startProjectYear"], y=df_f["forecast_total_investment"],
                    mode="lines+markers",
                    line=dict(color="#2563EB", width=3),
                    marker=dict(size=8, color="#60A5FA"),
                    fill="tozeroy", fillcolor="rgba(37,99,235,0.06)",
                ))
                fig.update_layout(
                    **PLOT_LAYOUT, height=380,
                    title=f"{selected} — Investment Trajectory",
                    yaxis_tickprefix="₹",
                )
                st.plotly_chart(fig, use_container_width=True)

    if "forecast_summary" in tables and not tables["forecast_summary"].empty:
        summary = tables["forecast_summary"]
        year_cols = [c for c in summary.columns if c.startswith(("actual_", "forecast_"))]
        dist_col = next((c for c in ("district", "distName") if c in summary.columns), None)
        if dist_col and year_cols:
            show = summary.head(int(filters["top_n"]))
            fig = go.Figure()
            palette = [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(year_cols[:4]))]
            for i, col in enumerate(year_cols[:4]):
                if col in show.columns:
                    fig.add_trace(go.Bar(
                        name=col.replace("_", " ").title(),
                        x=show[dist_col], y=show[col],
                        marker_color=palette[i],
                    ))
            fig.update_layout(
                **PLOT_LAYOUT, barmode="group", height=440,
                title="Forecast Summary by District",
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3: INVESTMENT SCORES & CLUSTERS
# ══════════════════════════════════════════════════════════════
def render_scores_and_clusters(tables: dict, filters: dict) -> None:
    section_head("◈", "Investment Intelligence")

    c1, c2 = st.columns(2)

    with c1:
        if "investment_scores" in tables and not tables["investment_scores"].empty:
            scores = tables["investment_scores"].head(int(filters["top_n"])).copy()
            size_col = "project_count" if _safe_col(scores, "project_count") else None
            if _safe_col(scores, "avg_cost_sqft") and _safe_col(scores, "final_score"):
                fig = px.scatter(
                    scores, x="avg_cost_sqft", y="final_score",
                    size=size_col,
                    color="final_score",
                    hover_name="distName" if _safe_col(scores, "distName") else None,
                    color_continuous_scale=GOLD_SCALE,
                )
                fig.update_traces(marker=dict(line=dict(color="#0D0D0D", width=1.5)))
                fig.update_layout(
                    **PLOT_LAYOUT, height=420,
                    title="Investment Score Profile",
                    coloraxis_showscale=False,
                    xaxis_tickprefix="₹",
                )
                st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "project_clusters" in tables and not tables["project_clusters"].empty:
            clusters = tables["project_clusters"]
            if _safe_col(clusters, "cluster_label"):
                counts = clusters["cluster_label"].value_counts().reset_index()
                counts.columns = ["cluster_label", "count"]
                counts = counts.sort_values("count", ascending=True)
                fig = go.Figure(go.Bar(
                    x=counts["count"], y=counts["cluster_label"],
                    orientation="h",
                    marker=dict(
                        color=CHART_COLORS[:len(counts)],
                        line=dict(width=0),
                    ),
                    hovertemplate="<b>%{y}</b><br>%{x:,} projects<extra></extra>",
                ))
                fig.update_layout(**PLOT_LAYOUT, height=420, title="Project Cluster Distribution")
                st.plotly_chart(fig, use_container_width=True)

    # Score table
    if "investment_scores" in tables and not tables["investment_scores"].empty:
        inv = tables["investment_scores"].head(int(filters["top_n"])).copy()
        display_cols = [c for c in [
            "distName", "project_count", "avg_booking", "avg_cost_sqft",
            "growth_score", "demand_score", "invest_score", "final_score"
        ] if c in inv.columns]
        if display_cols:
            section_head("◈", "Score Breakdown")
            st.dataframe(
                inv[display_cols].style.format({
                    "avg_booking": "{:.2%}", "avg_cost_sqft": "₹{:,.0f}",
                    "growth_score": "{:.1f}", "demand_score": "{:.1f}",
                    "invest_score": "{:.1f}", "final_score": "{:.1f}",
                }, na_rep="—"),
                use_container_width=True, hide_index=True,
            )

    if "district_model_evaluation" in tables and not tables["district_model_evaluation"].empty:
        ev = tables["district_model_evaluation"]
        if _safe_col(ev, "R2"):
            best = ev.sort_values("R2", ascending=False).head(int(filters["top_n"]))
            section_head("◈", "Model Evaluation")
            st.dataframe(best, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 4: RISK ANALYSIS
# ══════════════════════════════════════════════════════════════
def render_risk(tables: dict, filters: dict) -> None:
    section_head("◈", "Risk & Anomaly Intelligence")

    if "risk_scores" not in tables or tables["risk_scores"].empty:
        st.info("No risk score data available.")
        return

    risk = tables["risk_scores"].copy()
    if not _safe_col(risk, "risk_score"):
        st.warning("Risk score column not found.")
        return

    if _safe_col(risk, "risk_category"):
        risk["risk_category"] = risk["risk_category"].fillna("Unknown")
    if _safe_col(risk, "risk_flag"):
        risk["risk_flag"] = risk["risk_flag"].fillna(0).astype(int)

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Total Projects", f"{len(risk):,}")
    with k2:
        flagged = int(risk["risk_flag"].sum()) if _safe_col(risk, "risk_flag") else 0
        st.metric("Risk-Flagged", f"{flagged:,}")
    with k3: st.metric("Avg Risk Score", f"{risk['risk_score'].mean():.1f}")
    with k4:
        critical = int((risk["risk_category"] == "Critical").sum()) if _safe_col(risk, "risk_category") else 0
        st.metric("Critical Projects", f"{critical:,}")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.3, 0.7])

    with c1:
        fig = go.Figure(go.Histogram(
            x=risk["risk_score"], nbinsx=40,
            marker=dict(color="#D4A853", line=dict(color="#0D0D0D", width=0.5)),
            hovertemplate="Score %{x:.1f}<br>Count: %{y}<extra></extra>",
        ))
        fig.update_layout(**PLOT_LAYOUT, height=400, title="Risk Score Distribution",
                          bargap=0.04)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if _safe_col(risk, "risk_category"):
            cat_counts = risk["risk_category"].value_counts().reset_index()
            cat_counts.columns = ["risk_category", "count"]
            cat_colors = {"Critical": "#E11D48", "High": "#F59E0B", "Medium": "#2563EB", "Low": "#059669", "Unknown": "#4A4540"}
            fig = go.Figure(go.Bar(
                x=cat_counts["risk_category"], y=cat_counts["count"],
                marker_color=[cat_colors.get(c, "#4A4540") for c in cat_counts["risk_category"]],
                marker_line_width=0,
                hovertemplate="<b>%{x}</b><br>%{y:,} projects<extra></extra>",
            ))
            fig.update_layout(**PLOT_LAYOUT, height=400, title="Risk Categories")
            st.plotly_chart(fig, use_container_width=True)

    if _safe_col(risk, "risk_category"):
        categories = ["All"] + sorted(risk["risk_category"].unique().tolist())
        sel = st.selectbox("Filter Category", categories, key="risk_cat_filter")
        if sel != "All":
            risk = risk[risk["risk_category"] == sel]

    display_cols = [c for c in [
        "distName", "projectType", "cost_cr", "duration_months",
        "booking_rate", "risk_score", "risk_category", "risk_flag"
    ] if c in risk.columns]

    if display_cols:
        section_head("◈", "Highest Risk Projects")
        top_risk = risk.sort_values("risk_score", ascending=False).head(int(filters["top_n"]))
        st.dataframe(top_risk[display_cols], use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 5: PREDICTION LAB
# ══════════════════════════════════════════════════════════════
def render_predictions(models: dict) -> None:
    section_head("◈", "AI Prediction Lab")

    st.markdown("""
    <p style="color:#5A5550;font-size:0.88rem;margin-bottom:1.5rem;max-width:600px;line-height:1.6">
        Enter a project profile below and run all five AI modules simultaneously —
        duration, cost, cluster type, and risk assessment are computed in seconds.
    </p>
    """, unsafe_allow_html=True)

    if not models:
        st.warning("⚠️ No trained model artifacts found in the `models/` folder. Train models first.")
        return

    available = []
    if "duration" in models:      available.append("⏱ Duration")
    if "cost" in models:          available.append("₹ Cost")
    if "clustering_km" in models: available.append("⬡ Clustering")
    if "anomaly_iso" in models:   available.append("⚡ Risk")
    st.info(f"**Active modules:** {'  ·  '.join(available)}")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div style="font-family:Cormorant Garamond,serif;font-size:1.05rem;color:#D4A853;margin-bottom:1rem;letter-spacing:0.04em">Project Details</div>', unsafe_allow_html=True)
        project_type = st.selectbox("Project Type", ["Residential/Group Housing", "Commercial", "Mixed Development", "Plotted Development"])
        district = st.selectbox("District", [
            "Ahmedabad", "Surat", "Vadodara", "Rajkot", "Gandhinagar", "Bhavnagar",
            "Anand", "Jamnagar", "Junagadh", "Mehsana", "Valsad", "Bharuch",
            "Navsari", "Kutch", "Banaskantha", "Kheda", "Patan", "Other",
        ])
        promoter_type = st.selectbox("Promoter Type", ["Partnership", "Company", "Individual", "Other"])
        under_redev = st.selectbox("Under Redevelopment", ["NO", "YES"])
        total_units = st.number_input("Total Units", min_value=1, value=120, step=1)
        total_land_cost = st.number_input("Land Cost (₹)", min_value=0.0, value=15_000_000.0, step=100_000.0, format="%.0f")
        total_estimated_cost = st.number_input("Estimated Total Cost (₹)", min_value=0.0, value=250_000_000.0, step=5_000_000.0, format="%.0f")

    with c2:
        st.markdown('<div style="font-family:Cormorant Garamond,serif;font-size:1.05rem;color:#D4A853;margin-bottom:1rem;letter-spacing:0.04em">Operational Parameters</div>', unsafe_allow_html=True)
        start_year = st.selectbox("Start Year", [2020, 2021, 2022, 2023, 2024, 2025, 2026], index=3)
        start_month = st.selectbox("Start Month", list(range(1, 13)), index=0)
        no_of_inventory = st.number_input("Inventory Count", min_value=1.0, value=float(total_units), step=1.0)
        total_carpet_area = st.number_input("Carpet Area (sq ft)", min_value=0.0, value=50000.0, step=100.0)
        total_builtup_area = st.number_input("Built-up Area (sq ft)", min_value=0.0, value=65000.0, step=100.0)
        total_sqft_build = st.number_input("Total Built (sq ft)", min_value=0.0, value=60000.0, step=100.0)
        booking_rate = st.slider("Booking Rate", 0.0, 1.0, 0.5, 0.01)
        duration_months_input = st.number_input("Duration Months (for cost model)", min_value=0.0, value=48.0, step=1.0)

    payload = {
        "projectType": project_type, "distName": district,
        "promoter_type_simple": promoter_type, "underRedevelopment": under_redev,
        "totalUnits": total_units, "totalLandCost": total_land_cost,
        "totalEstimatedCost": total_estimated_cost, "startProjectYear": start_year,
        "startProjectMonth": start_month, "noOfInventory": no_of_inventory,
        "totalCarpetArea_form3A": total_carpet_area, "totalBuiltupArea_form3A": total_builtup_area,
        "totalSquareFootBuild": total_sqft_build, "booking_rate": booking_rate,
        "duration_months": duration_months_input,
    }

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⚡  Run Full AI Prediction", type="primary", use_container_width=True):
        with st.spinner("Running all prediction modules..."):
            try:
                result = predict_bundle(payload, models)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                return

        st.markdown("<br>", unsafe_allow_html=True)
        section_head("◈", "Prediction Results")

        r1, r2, r3, r4 = st.columns(4)

        with r1:
            if result.get("duration_months") is not None:
                st.markdown(f"""
                <div class="pred-card">
                    <span class="pred-card-icon">⏱</span>
                    <div class="pred-card-label">Predicted Duration</div>
                    <div class="pred-card-value">{result['duration_months']:.1f}</div>
                    <div class="pred-card-sub">months to completion</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.warning("Duration model unavailable.")

        with r2:
            if result.get("totalEstimatedCost") is not None:
                st.markdown(f"""
                <div class="pred-card info">
                    <span class="pred-card-icon">₹</span>
                    <div class="pred-card-label">Predicted Cost</div>
                    <div class="pred-card-value">{money_fmt(result['totalEstimatedCost'])}</div>
                    <div class="pred-card-sub">estimated project value</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.warning("Cost model unavailable.")

        with r3:
            cluster = result.get("cluster")
            if cluster is not None:
                st.markdown(f"""
                <div class="pred-card success">
                    <span class="pred-card-icon">⬡</span>
                    <div class="pred-card-label">Project Cluster</div>
                    <div class="pred-card-value" style="font-size:1.6rem">{cluster['cluster_label']}</div>
                    <div class="pred-card-sub">market segment classification</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.warning("Cluster model unavailable.")

        with r4:
            anomaly = result.get("anomaly")
            if anomaly is not None:
                is_risk = anomaly["risk_flag"]
                card_class = "risk" if is_risk else "success"
                pill_class = "pill-danger" if is_risk else "pill-safe"
                pill_text = "HIGH RISK" if is_risk else "SAFE"
                st.markdown(f"""
                <div class="pred-card {card_class}">
                    <span class="pred-card-icon">{'🔴' if is_risk else '🟢'}</span>
                    <div class="pred-card-label">Risk Assessment</div>
                    <div class="pred-card-value" style="font-size:1.4rem">{anomaly['risk_category']}</div>
                    <span class="status-pill {pill_class}">{pill_text} · {anomaly['risk_score']}</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.warning("Risk model unavailable.")


# ══════════════════════════════════════════════════════════════
# TAB 6: REPORTS GALLERY
# ══════════════════════════════════════════════════════════════
def render_reports() -> None:
    section_head("◈", "Reports Gallery")

    image_paths = _load_images()
    if not image_paths:
        st.info("No report images found in the `reports/` directory.")
        return

    key_prefixes = ("01_", "03_", "09_", "15_", "16_", "20_", "21_", "28_", "30_", "33_", "34_", "37_", "38_", "39_", "40_", "42_")
    gallery = [p for p in image_paths if p.name.startswith(key_prefixes)] or image_paths[:12]

    for i in range(0, min(len(gallery), 12), 2):
        cols = st.columns(2)
        for j in range(2):
            idx = i + j
            if idx < len(gallery):
                with cols[j]:
                    try:
                        img = Image.open(gallery[idx])
                        st.image(img, caption=gallery[idx].name, use_container_width=True)
                    except Exception as e:
                        st.caption(f"⚠️ Could not load: {gallery[idx].name} ({e})")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main() -> None:
    tables = _load_tables()
    models = _load_models()
    filters = build_sidebar(tables)
    summary = summarize_forecast_tables(tables)

    render_header(summary or {})

    # Summary KPI strip
    if summary:
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            if "top_investment_district" in summary:
                st.metric("🏆 Top District", str(summary["top_investment_district"]))
        with kpi2:
            if "top_investment_score" in summary:
                st.metric("⭐ Top Score", f"{summary['top_investment_score']:.1f}")
        with kpi3:
            if "latest_year" in summary:
                st.metric("📅 Forecast Year", str(summary["latest_year"]))
        with kpi4:
            if "latest_forecast" in summary:
                st.metric("💹 Latest Forecast", money_fmt(summary["latest_forecast"]))

    st.markdown("<br>", unsafe_allow_html=True)

    tabs = st.tabs([
        "◈  Overview",
        "◈  Forecasts",
        "◈  Scores & Clusters",
        "◈  Risk Analysis",
        "◈  Prediction Lab",
        "◈  Reports",
    ])

    with tabs[0]: render_overview(tables, filters)
    with tabs[1]: render_forecasts(tables, filters)
    with tabs[2]: render_scores_and_clusters(tables, filters)
    with tabs[3]: render_risk(tables, filters)
    with tabs[4]: render_predictions(models)
    with tabs[5]: render_reports()


if __name__ == "__main__":
    main()