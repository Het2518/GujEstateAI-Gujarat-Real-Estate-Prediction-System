# GujEstateAI — Complete Build Guide
### Machine Learning-Based Construction Intelligence System for Gujarat

---

## Project Folder Structure

```
GujEstateAI/
│
├── data/
│   ├── raw/
│   │   └── ProjectInfo_Gujarat.csv          ← your original file
│   ├── processed/
│   │   ├── cleaned.csv                      ← after Step 2
│   │   └── features.csv                     ← after Step 3
│   └── predictions/
│       └── *.csv                            ← model outputs
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_module1_duration.ipynb
│   ├── 05_module2_cost.ipynb
│   ├── 06_module3_forecasting.ipynb
│   ├── 07_module4_clustering.ipynb
│   └── 08_module5_anomaly.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── train_duration.py
│   ├── train_cost.py
│   ├── train_clustering.py
│   ├── train_anomaly.py
│   ├── forecast.py
│   └── predict.py                           ← unified inference pipeline
│
├── models/
│   ├── duration_model.pkl
│   ├── cost_model.pkl
│   ├── clustering_model.pkl
│   ├── anomaly_model.pkl
│   └── encoders.pkl
│
├── dashboard/
│   └── app.py                               ← Streamlit app (later phase)
│
├── reports/
│   └── shap_plots/
│
├── requirements.txt
└── README.md
```

---

## Phase 0 — Setup (Day 1)

### Step 0.1 — Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate          # Mac/Linux
venv\Scripts\activate             # Windows
```

### Step 0.2 — Install Dependencies

Create `requirements.txt`:

```
pandas==2.1.0
numpy==1.24.0
scikit-learn==1.3.0
xgboost==2.0.0
lightgbm==4.1.0
prophet==1.1.4
matplotlib==3.7.0
seaborn==0.12.0
plotly==5.17.0
shap==0.43.0
joblib==1.3.0
streamlit==1.27.0
jupyter==1.0.0
```

```bash
pip install -r requirements.txt
```

---

## Phase 1 — Exploratory Data Analysis (Days 2–3)

**Notebook:** `notebooks/01_eda.ipynb`

### Step 1.1 — Load and Inspect

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/raw/ProjectInfo_Gujarat.csv")

print(df.shape)            # (14507, 44)
print(df.dtypes)
print(df.isnull().sum() / len(df) * 100)   # null percentages
print(df.describe())
```

### Step 1.2 — Key EDA Questions to Answer

Answer each of these with a plot:

1. **Distribution of projectType** — bar chart
2. **Projects per year (2010–2023)** — line chart (shows growth trend)
3. **Top 10 districts by project count** — horizontal bar
4. **Distribution of totalEstimatedCost** — histogram with log scale
5. **Distribution of project duration (months)** — histogram
6. **Booking rate distribution** — histogram
7. **avgCostPerSqFt by projectType** — boxplot (reveals outliers)
8. **Correlation heatmap** — of all numeric columns

### Step 1.3 — Key EDA Findings to Document

```python
# Project duration
df['duration_months'] = (df['EndProjectYear'] - df['startProjectYear']) * 12 \
                       + (df['EndProjectMonth'] - df['startProjectMonth'])

# Booking rate
df['booking_rate'] = df['bookedUnits'] / df['totalUnits']

# Profit margin indicator
df['margin'] = (df['totalSellingAmount'] - df['totalDevelopCost']) \
               / df['totalSellingAmount']

print("Avg duration:", df['duration_months'].mean(), "months")
print("Avg booking rate:", df['booking_rate'].mean())
print("Avg margin:", df['margin'].mean())
```

> **Save all EDA plots to `reports/` folder.**

---

## Phase 2 — Data Cleaning (Day 3–4)

**Notebook:** `notebooks/02_cleaning.ipynb`  
**Script:** `src/preprocess.py`

### Step 2.1 — Handle Nulls

```python
# Drop columns with >40% nulls — not useful as features
cols_to_drop = ['pinCode', 'tPNo', 'totalAreaOfLand', 'AvgAreaOfLand',
                'architect_name', 'eng_name', 'projectAddress2']
df.drop(columns=cols_to_drop, inplace=True)

# Drop rows where distName is null (only 6% — safe to drop)
df.dropna(subset=['distName', 'startDate', 'completionDate'], inplace=True)

# Fill noOfInventory nulls with median per projectType
df['noOfInventory'] = df.groupby('projectType')['noOfInventory'] \
                        .transform(lambda x: x.fillna(x.median()))
```

### Step 2.2 — Parse Dates

```python
df['startDate'] = pd.to_datetime(df['startDate'], errors='coerce')
df['completionDate'] = pd.to_datetime(df['completionDate'], errors='coerce')

df['start_year'] = df['startDate'].dt.year
df['start_month'] = df['startDate'].dt.month
```

### Step 2.3 — Remove Outliers

```python
# Remove extreme cost outliers (beyond 99.5th percentile)
upper = df['totalEstimatedCost'].quantile(0.995)
df = df[df['totalEstimatedCost'] <= upper]

# Remove extreme avgCostPerSqFt outliers
upper_sqft = df['avgCostPerSqFt'].quantile(0.99)
df = df[df['avgCostPerSqFt'] <= upper_sqft]

# Remove projects with duration < 3 months or > 240 months
df = df[(df['duration_months'] >= 3) & (df['duration_months'] <= 240)]
```

### Step 2.4 — Save Cleaned Data

```python
df.to_csv("data/processed/cleaned.csv", index=False)
print("Cleaned shape:", df.shape)
```

---

## Phase 3 — Feature Engineering (Day 4–5)

**Notebook:** `notebooks/03_feature_engineering.ipynb`  
**Script:** `src/features.py`

### Step 3.1 — Create Core Features

```python
df = pd.read_csv("data/processed/cleaned.csv")

# === Target for Module 1 ===
# Duration in months (regression target)
df['duration_months'] = (df['EndProjectYear'] - df['startProjectYear']) * 12 \
                       + (df['EndProjectMonth'] - df['startProjectMonth'])

# === Target for Module 2 ===
# Already exists: totalEstimatedCost

# === Derived Features ===
df['booking_rate']    = df['bookedUnits'] / df['totalUnits']
df['cost_per_unit']   = df['totalEstimatedCost'] / df['totalUnits']
df['land_cost_ratio'] = df['totalLandCost'] / df['totalEstimatedCost']
df['sell_dev_ratio']  = df['totalSellingAmount'] / df['totalDevelopCost']
df['is_redevelop']    = (df['underRedevelopment'] == 'YES').astype(int)
df['log_cost']        = np.log1p(df['totalEstimatedCost'])
df['log_units']       = np.log1p(df['totalUnits'])
df['start_quarter']   = df['start_month'].apply(lambda m: (m-1)//3 + 1)
```

### Step 3.2 — Encode Categorical Variables

```python
from sklearn.preprocessing import LabelEncoder
import joblib

encoders = {}

for col in ['projectType', 'distName', 'promoterType']:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

joblib.dump(encoders, "models/encoders.pkl")
```

### Step 3.3 — Define Final Feature Sets

```python
# Module 1 — Duration Prediction features
FEATURES_DURATION = [
    'projectType_enc', 'distName_enc', 'promoterType_enc',
    'totalUnits', 'log_cost', 'totalLandCost',
    'is_redevelop', 'start_year', 'start_quarter',
    'land_cost_ratio', 'avgCostPerUnit'
]

# Module 2 — Cost Prediction features
FEATURES_COST = [
    'projectType_enc', 'distName_enc', 'promoterType_enc',
    'totalUnits', 'duration_months', 'totalLandCost',
    'totalCarpetArea_form3A', 'avgCostPerSqFt',
    'is_redevelop', 'start_year'
]

# Module 4 — Clustering features
FEATURES_CLUSTER = [
    'log_cost', 'log_units', 'duration_months',
    'avgCostPerSqFt', 'booking_rate',
    'land_cost_ratio', 'projectType_enc', 'distName_enc'
]

df.to_csv("data/processed/features.csv", index=False)
```

---

## Phase 4 — Module 1: Project Duration Prediction (Days 5–6)

**Notebook:** `notebooks/04_module1_duration.ipynb`  
**Script:** `src/train_duration.py`

**Goal:** Predict how long (in months) a project will take to complete.

### Step 4.1 — Prepare Data

```python
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/features.csv")
df_clean = df.dropna(subset=FEATURES_DURATION + ['duration_months'])

X = df_clean[FEATURES_DURATION]
y = df_clean['duration_months']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Step 4.2 — Train Models

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

models = {
    'Ridge':         Ridge(),
    'RandomForest':  RandomForestRegressor(n_estimators=200, random_state=42),
    'XGBoost':       XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)
    results[name] = {'MAE': round(mae, 2), 'R2': round(r2, 4)}
    print(f"{name}: MAE={mae:.2f} months, R²={r2:.4f}")
```

### Step 4.3 — SHAP Explainability

```python
import shap

best_model = models['XGBoost']     # use whichever wins
explainer   = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=FEATURES_DURATION)
plt.savefig("reports/shap_plots/duration_shap.png", bbox_inches='tight')
```

### Step 4.4 — Save Best Model

```python
import joblib
joblib.dump(best_model, "models/duration_model.pkl")
```

---

## Phase 5 — Module 2: Project Cost Estimation (Days 6–7)

**Notebook:** `notebooks/05_module2_cost.ipynb`  
**Script:** `src/train_cost.py`

**Goal:** Predict `totalEstimatedCost` from project features.

### Step 5.1 — Prepare Data

```python
df_clean = df.dropna(subset=FEATURES_COST + ['totalEstimatedCost'])

X = df_clean[FEATURES_COST]
y = np.log1p(df_clean['totalEstimatedCost'])   # log-transform for normality

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Step 5.2 — Train Models

```python
from lightgbm import LGBMRegressor

models = {
    'Ridge':        Ridge(),
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42),
    'XGBoost':      XGBRegressor(n_estimators=300, learning_rate=0.05),
    'LightGBM':     LGBMRegressor(n_estimators=300, learning_rate=0.05)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    # Convert back from log scale
    preds_actual  = np.expm1(preds)
    y_test_actual = np.expm1(y_test)
    mae = mean_absolute_error(y_test_actual, preds_actual)
    r2  = r2_score(y_test_actual, preds_actual)
    print(f"{name}: MAE=₹{mae:,.0f}, R²={r2:.4f}")
```

### Step 5.3 — SHAP + Save

```python
# SHAP for best model (LightGBM or XGBoost usually wins)
best_model = models['LightGBM']
explainer   = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
plt.savefig("reports/shap_plots/cost_shap.png", bbox_inches='tight')

joblib.dump(best_model, "models/cost_model.pkl")
```

---

## Phase 6 — Module 3: Market Growth Forecasting (Days 8–9)

**Notebook:** `notebooks/06_module3_forecasting.ipynb`  
**Script:** `src/forecast.py`

**Goal:** Forecast future project registrations and investment per district.

### Step 6.1 — Prepare Time Series Data

```python
# Aggregate by year + district
ts = df.groupby(['start_year', 'distName']).agg(
    project_count=('projectRegId', 'count'),
    total_investment=('totalEstimatedCost', 'sum')
).reset_index()

# Focus on top 6 districts (enough data density)
top_districts = ['Ahmedabad', 'Vadodara', 'Surat', 'Rajkot', 'Gandhinagar', 'Bhavnagar']
ts = ts[ts['distName'].isin(top_districts)]
```

### Step 6.2 — Prophet Forecast per District

```python
from prophet import Prophet

forecasts = {}

for district in top_districts:
    dist_data = ts[ts['distName'] == district][['start_year', 'project_count']].copy()
    dist_data.columns = ['ds', 'y']
    dist_data['ds'] = pd.to_datetime(dist_data['ds'], format='%Y')

    model = Prophet(yearly_seasonality=True, uncertainty_samples=200)
    model.fit(dist_data)

    future = model.make_future_dataframe(periods=3, freq='Y')
    forecast = model.predict(future)

    forecasts[district] = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    fig = model.plot(forecast)
    fig.savefig(f"reports/forecast_{district}.png")

print("Forecasts done for:", list(forecasts.keys()))
```

### Step 6.3 — Save Forecasts

```python
all_forecasts = pd.concat(
    [v.assign(district=k) for k, v in forecasts.items()]
)
all_forecasts.to_csv("data/predictions/forecasts.csv", index=False)
```

---

## Phase 7 — Module 4: Project Clustering + Investment Scoring (Days 9–10)

**Notebook:** `notebooks/07_module4_clustering.ipynb`  
**Script:** `src/train_clustering.py`

**Goal:** Group projects into natural clusters AND score districts for investment.

### Step 7.1 — Prepare & Scale

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df_clust = df.dropna(subset=FEATURES_CLUSTER).copy()
X = df_clust[FEATURES_CLUSTER]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Step 7.2 — Find Optimal K (Elbow Method)

```python
inertias = []
K_range = range(2, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.plot(K_range, inertias, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.savefig("reports/elbow_plot.png")
```

### Step 7.3 — Train Final Clustering Model

```python
# Based on elbow plot — typically K=4 or 5 for this dataset
best_k = 5
km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_clust['cluster'] = km.fit_predict(X_scaled)

# Name clusters based on centroid characteristics
cluster_summary = df_clust.groupby('cluster')[FEATURES_CLUSTER].mean()
print(cluster_summary)

# Suggested names after inspecting centroids:
cluster_names = {
    0: 'Affordable Residential',
    1: 'Luxury High-Rise',
    2: 'Commercial Zones',
    3: 'Mixed Development Mid-Range',
    4: 'Large Plotted Schemes'
}
df_clust['cluster_label'] = df_clust['cluster'].map(cluster_names)
```

### Step 7.4 — Investment Hotspot Scoring

```python
# Score each district on 3 signals (0-10 scale each)
district_scores = df.groupby('distName').agg(
    project_count   = ('projectRegId', 'count'),
    avg_booking     = ('booking_rate', 'mean'),
    total_invest    = ('totalEstimatedCost', 'sum'),
    avg_cost_sqft   = ('avgCostPerSqFt', 'mean')
).reset_index()

# Normalize 0–10
def norm(s): return (s - s.min()) / (s.max() - s.min()) * 10

district_scores['growth_score']   = norm(district_scores['project_count'])
district_scores['demand_score']   = norm(district_scores['avg_booking'])
district_scores['invest_score']   = norm(district_scores['total_invest'])

district_scores['final_score'] = (
    district_scores['growth_score']  * 0.4 +
    district_scores['demand_score']  * 0.35 +
    district_scores['invest_score']  * 0.25
).round(2)

district_scores.sort_values('final_score', ascending=False).to_csv(
    "data/predictions/investment_scores.csv", index=False
)

joblib.dump((km, scaler), "models/clustering_model.pkl")
```

---

## Phase 8 — Module 5: Risk & Anomaly Detection (Day 11)

**Notebook:** `notebooks/08_module5_anomaly.ipynb`  
**Script:** `src/train_anomaly.py`

**Goal:** Flag projects with abnormal cost, booking, or duration patterns.

### Step 8.1 — Prepare Features

```python
from sklearn.ensemble import IsolationForest

ANOMALY_FEATURES = [
    'totalEstimatedCost', 'avgCostPerSqFt', 'duration_months',
    'booking_rate', 'totalLandCost', 'land_cost_ratio',
    'totalUnits', 'cost_per_unit'
]

df_anom = df.dropna(subset=ANOMALY_FEATURES).copy()
X_anom = df_anom[ANOMALY_FEATURES]

scaler_anom = StandardScaler()
X_anom_scaled = scaler_anom.fit_transform(X_anom)
```

### Step 8.2 — Train Isolation Forest

```python
iso = IsolationForest(
    n_estimators=200,
    contamination=0.03,    # assume 3% anomalies
    random_state=42
)
df_anom['anomaly_score'] = iso.fit_predict(X_anom_scaled)
df_anom['risk_flag']     = (df_anom['anomaly_score'] == -1).astype(int)

print("Flagged as risky:", df_anom['risk_flag'].sum(), "projects")
print("Sample risky projects:")
print(df_anom[df_anom['risk_flag'] == 1][
    ['projectName', 'distName', 'totalEstimatedCost', 'avgCostPerSqFt', 'duration_months']
].head(10))
```

### Step 8.3 — Risk Score (Continuous)

```python
# Raw anomaly scores (more negative = more anomalous)
df_anom['raw_score'] = iso.score_samples(X_anom_scaled)

# Normalize to 0–100 risk scale (higher = riskier)
df_anom['risk_score'] = (
    (df_anom['raw_score'] - df_anom['raw_score'].max()) /
    (df_anom['raw_score'].min() - df_anom['raw_score'].max()) * 100
).round(1)

df_anom[['projectRegId', 'risk_flag', 'risk_score']].to_csv(
    "data/predictions/risk_scores.csv", index=False
)
joblib.dump((iso, scaler_anom), "models/anomaly_model.pkl")
```

---

## Phase 9 — Unified Inference Pipeline (Day 12)

**Script:** `src/predict.py`

**Goal:** One function that takes a new project and returns a full intelligence report.

```python
import joblib
import pandas as pd
import numpy as np

# Load everything
encoders       = joblib.load("models/encoders.pkl")
duration_model = joblib.load("models/duration_model.pkl")
cost_model     = joblib.load("models/cost_model.pkl")
km, scaler_cl  = joblib.load("models/clustering_model.pkl")
iso, scaler_an = joblib.load("models/anomaly_model.pkl")

def predict_project(project: dict) -> dict:
    """
    Input: dict with keys matching raw feature names
    Output: full intelligence report
    """
    row = pd.DataFrame([project])

    # Encode categoricals
    for col in ['projectType', 'distName', 'promoterType']:
        row[col + '_enc'] = encoders[col].transform(row[col].astype(str))

    # Predict duration
    dur_pred = duration_model.predict(row[FEATURES_DURATION])[0]

    # Predict cost
    cost_pred = np.expm1(cost_model.predict(row[FEATURES_COST])[0])

    # Predict cluster
    cluster_scaled = scaler_cl.transform(row[FEATURES_CLUSTER])
    cluster_id     = km.predict(cluster_scaled)[0]
    cluster_label  = cluster_names[cluster_id]

    # Anomaly score
    anom_scaled  = scaler_an.transform(row[ANOMALY_FEATURES])
    risk_raw     = iso.score_samples(anom_scaled)[0]
    risk_flag    = iso.predict(anom_scaled)[0] == -1

    return {
        "predicted_duration_months": round(dur_pred, 1),
        "estimated_cost_INR":        round(cost_pred, 0),
        "project_cluster":           cluster_label,
        "risk_flag":                 bool(risk_flag),
        "risk_score_raw":            round(float(risk_raw), 4)
    }

# Example usage
sample_project = {
    "projectType": "Residential/Group Housing",
    "distName": "Ahmedabad",
    "promoterType": "PARTNERSHIP FIRM",
    "totalUnits": 120,
    "totalLandCost": 15000000,
    "avgCostPerSqFt": 32000,
    ...
}
report = predict_project(sample_project)
print(report)
```

---

## Phase 10 — Evaluation & Documentation (Day 13)

### Step 10.1 — Model Results Table

Create this table in your README:

| Module | Model | Key Metric | Score |
|--------|-------|-----------|-------|
| Duration Prediction | XGBoost | MAE | X months |
| Cost Estimation | LightGBM | R² | X.XX |
| Market Forecast | Prophet | MAPE | X% |
| Clustering | K-Means | Silhouette | X.XX |
| Anomaly Detection | Isolation Forest | Flagged | XXX projects |

### Step 10.2 — SHAP Summary

For Duration and Cost models, save:
- `shap_summary_plot` — which features matter most
- `shap_waterfall_plot` — explanation for one prediction

### Step 10.3 — Write README.md

Sections to include:
1. Project Overview
2. Dataset Description
3. Module Architecture (diagram)
4. How to Run
5. Model Results Table
6. Key Findings
7. Sample Predictions

---

## Phase 11 — Streamlit Dashboard (Days 14–16)

**File:** `dashboard/app.py`

Build 5 pages in Streamlit:

```python
import streamlit as st

pages = {
    "🏠 Overview":              page_overview,
    "⏱️ Duration Predictor":    page_duration,
    "💰 Cost Estimator":        page_cost,
    "📈 Market Forecast":       page_forecast,
    "🗺️ Investment Hotspots":   page_investment,
    "⚠️ Risk Detector":         page_risk,
}

selected = st.sidebar.selectbox("Navigate", list(pages.keys()))
pages[selected]()
```

Each page has:
- Input form (sidebar or main panel)
- Model prediction output
- Supporting chart from EDA

Run with: `streamlit run dashboard/app.py`

---

## Full Timeline Summary

| Phase | Task | Days |
|-------|------|------|
| 0 | Setup & Environment | Day 1 |
| 1 | EDA | Days 2–3 |
| 2 | Cleaning | Days 3–4 |
| 3 | Feature Engineering | Days 4–5 |
| 4 | Module 1: Duration | Days 5–6 |
| 5 | Module 2: Cost | Days 6–7 |
| 6 | Module 3: Forecasting | Days 8–9 |
| 7 | Module 4: Clustering | Days 9–10 |
| 8 | Module 5: Anomaly | Day 11 |
| 9 | Unified Pipeline | Day 12 |
| 10 | Eval + Docs | Day 13 |
| 11 | Streamlit Dashboard | Days 14–16 |

**Total: ~16 days for full project**

---

## What to Put on Resume / GitHub

```
GujEstateAI — Gujarat Real Estate Intelligence System

• Built end-to-end ML pipeline on 14,500+ government-registered construction 
  projects (2010–2023) using Python, XGBoost, LightGBM, Prophet, and scikit-learn

• Developed 5 ML modules: project duration regression (MAE: X months), 
  cost estimation (R²: X.XX), district-level growth forecasting (Prophet), 
  project clustering (K-Means, K=5), and risk/anomaly detection (Isolation Forest)

• Applied SHAP explainability to cost and duration models for interpretable predictions

• Built unified inference pipeline: single input → duration + cost + cluster + 
  risk report in one call

• Deployed interactive Streamlit dashboard with district investment scoring 
  and live predictions
```

---

*GujEstateAI Architecture Guide — built for Gujarat Real Estate Dataset (2010–2023)*
