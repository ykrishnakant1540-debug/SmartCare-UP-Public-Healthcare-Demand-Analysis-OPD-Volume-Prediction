"""
SMARTCARE-UP | Healthcare Analytics & Predictive Analysis
========================================================
Run:
    pip install -r requirements_smartcare_up.txt
    streamlit run smartcare_up_app.py

Keep the cleaned CSV beside this script:
    SmartCare_UP_HMIS_MASTER_CLEANED_2012_2020.csv

Model target:
    allopathic_opd_total

Five algorithms:
    Linear Regression, Decision Tree, Random Forest,
    Gradient Boosting, Extra Trees

Method note:
The supplied HMIS files are sparse "upto month" snapshots and may be
cumulative. This application estimates reported OPD volume within a
snapshot using other indicators from that same snapshot. It is NOT a
validated next-month forecasting system.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)

# ---------------------------------------------------------------------
# PAGE CONFIG + THEME
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="SmartCare-UP | Healthcare Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px;}
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(16, 42, 67, .95), rgba(20, 75, 92, .88));
        border: 1px solid rgba(77, 208, 190, .25);
        padding: 16px 18px;
        border-radius: 14px;
    }
    [data-testid="stMetricLabel"] {color: #b8d7dc;}
    [data-testid="stMetricValue"] {color: #ffffff;}
    .hero {
        padding: 28px 30px;
        border-radius: 18px;
        background: linear-gradient(115deg, #102a43 0%, #155e75 58%, #0f766e 100%);
        border: 1px solid rgba(94, 234, 212, .25);
        margin-bottom: 18px;
    }
    .hero h1 {color: white; font-size: 2.45rem; margin: 0;}
    .hero p {color: #d8f3f0; font-size: 1.04rem; margin: 8px 0 0 0;}
    .section-note {
        padding: 12px 16px;
        border-left: 4px solid #14b8a6;
        background: rgba(20, 184, 166, .08);
        border-radius: 7px;
        margin: 8px 0 18px 0;
    }
    div[data-testid="stTabs"] button {font-weight: 650;}
</style>
""", unsafe_allow_html=True)

TARGET = "allopathic_opd_total"
DEFAULT_FILE = "SmartCare_UP_HMIS_MASTER_CLEANED_2012_2020.csv"

st.markdown("""
<div class="hero">
  <h1>🏥 SmartCare-UP</h1>
  <p>Public Healthcare Analytics • District-level OPD Volume Estimation • Uttar Pradesh HMIS</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# DATA LOADING + VALIDATION
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_dataset(file_bytes):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes), low_memory=False)

def prepare_dataset(raw):
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required = {"district", "financial_year", "report_upto", TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(sorted(missing))
        )

    text_cols = [
        "district", "financial_year", "report_upto",
        "schema_version", "source_file", "source_group", "quality_flag"
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    df["district"] = df["district"].replace(
        {"": pd.NA, "nan": pd.NA, "None": pd.NA}
    )
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    # Older schema rows are shown in the overview but excluded from modeling.
    if "schema_version" in df.columns:
        detailed_mask = df["schema_version"].astype("string").str.contains(
            "detailed_2017_2020", case=False, na=False
        )
        model_df = df.loc[detailed_mask].copy()
    else:
        model_df = df.copy()

    model_df = model_df.loc[
        model_df["district"].notna()
        & model_df[TARGET].notna()
        & (model_df[TARGET] >= 0)
    ].copy()

    identifier_cols = {
        "district", "financial_year", "report_upto", "schema_version",
        "source_file", "source_group", "quality_flag"
    }
    for col in model_df.columns:
        if col not in identifier_cols:
            model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

    model_df = model_df.drop_duplicates().reset_index(drop=True)
    return df, model_df

def build_preprocessor(num_cols, cat_cols):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, num_cols),
        ("categorical", categorical_pipe, cat_cols)
    ], remainder="drop")

@st.cache_data(show_spinner=True)
def train_models(data):
    # Exclude target and target component fields to prevent direct leakage.
    excluded = {
        TARGET, "allopathic_opd_public", "allopathic_opd_rural",
        "source_file", "source_group", "schema_version", "quality_flag"
    }

    cat_candidates = [
        c for c in ["district", "financial_year", "report_upto"]
        if c in data.columns and data[c].nunique(dropna=True) > 1
    ]

    num_candidates = [
        c for c in data.select_dtypes(include=[np.number]).columns
        if c not in excluded
        and not c.startswith("allopathic_opd_")
        and data[c].notna().mean() >= 0.35
        and data[c].nunique(dropna=True) > 1
    ]

    num_cols = [
        c for c in num_candidates
        if data[c].notna().sum() > 0 and data[c].nunique(dropna=True) > 1
    ]
    cat_cols = cat_candidates
    features = num_cols + cat_cols

    if not features:
        raise ValueError("No usable predictor features remain after validation.")

    X = data[features].copy()
    y = pd.to_numeric(data[TARGET], errors="coerce")
    groups = data["district"].astype(str)

    valid = y.notna() & np.isfinite(y) & (y >= 0)
    X, y, groups = X.loc[valid], y.loc[valid], groups.loc[valid]

    if len(X) < 30 or groups.nunique() < 5:
        raise ValueError("Insufficient valid records or districts for model evaluation.")

    # Group split: unseen districts are used for the test set.
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=42
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    test_meta = data.loc[X_test.index, [
        "district", "financial_year", "report_upto"
    ]].copy()

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=8, min_samples_leaf=3, random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05,
            max_depth=2, random_state=42
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
    }

    fitted, rows, pred_rows = {}, [], []
    preprocessor = build_preprocessor(num_cols, cat_cols)

    for name, estimator in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])
        pipe.fit(X_train, y_train)
        pred = np.maximum(pipe.predict(X_test), 0)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)

        rows.append({
            "Algorithm": name,
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R²": float(r2)
        })
        fitted[name] = pipe

        frame = test_meta.copy()
        frame["Actual_OPD"] = y_test.to_numpy()
        frame["Predicted_OPD"] = pred
        frame["Residual"] = frame["Actual_OPD"] - frame["Predicted_OPD"]
        frame["Absolute_Error"] = np.abs(frame["Residual"])
        frame["Algorithm"] = name
        pred_rows.append(frame.reset_index(drop=True))

    metrics = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    all_predictions = pd.concat(pred_rows, ignore_index=True)

    return (
        metrics, fitted, all_predictions, features,
        num_cols, cat_cols, X_train, X_test, y_train, y_test
    )

def plotly_layout(fig, height=420):
    fig.update_layout(
        height=height,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=65, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text=""
    )
    return fig

# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
st.sidebar.title("⚙️ Project Controls")
st.sidebar.caption("SmartCare-UP • HMIS Analytics")
uploaded = st.sidebar.file_uploader(
    "Upload cleaned HMIS master CSV", type=["csv"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Prediction target**")
st.sidebar.code(TARGET)
st.sidebar.markdown("**Algorithms**")
st.sidebar.markdown(
    "1. Linear Regression\n"
    "2. Decision Tree\n"
    "3. Random Forest\n"
    "4. Gradient Boosting\n"
    "5. Extra Trees"
)
st.sidebar.info(
    "Test split is grouped by district. The model is evaluated on "
    "districts not used during training."
)

# ---------------------------------------------------------------------
# LOAD SOURCE FILE
# ---------------------------------------------------------------------
try:
    if uploaded is not None:
        raw_data = read_dataset(uploaded.getvalue())
        source_name = uploaded.name
    else:
        file_path = Path(__file__).resolve().parent / DEFAULT_FILE
        if not file_path.exists():
            st.warning(
                f"{DEFAULT_FILE} was not found beside this Python file. "
                "Upload the cleaned CSV in the sidebar."
            )
            st.stop()
        raw_data = read_dataset(file_path.read_bytes())
        source_name = DEFAULT_FILE

    all_data, model_data = prepare_dataset(raw_data)
except Exception as exc:
    st.error(f"Dataset loading/validation error: {exc}")
    st.stop()

if model_data.empty:
    st.error("No valid target records were found in the detailed schema.")
    st.stop()

# ---------------------------------------------------------------------
# TOP-LEVEL SUMMARY
# ---------------------------------------------------------------------
st.success(f"Dataset loaded: {source_name}")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Cleaned master records", f"{len(all_data):,}")
k2.metric("Model-ready records", f"{len(model_data):,}")
k3.metric("Districts in model data", f"{model_data['district'].nunique():,}")
k4.metric(
    "Numeric healthcare fields",
    f"{len(model_data.select_dtypes(include=[np.number]).columns):,}"
)

st.markdown("""
<div class="section-note">
<b>Project objective:</b> Analyze district-level HMIS service indicators and
compare five regression algorithms for estimating reported Allopathic OPD volume.
This version estimates OPD values within report snapshots; it does not claim
to forecast the next calendar month's demand.
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# NAVIGATION TABS
# ---------------------------------------------------------------------
overview_tab, explore_tab, models_tab, conclusion_tab = st.tabs([
    "📊 Data Overview",
    "🔎 Exploratory Analysis",
    "🤖 Predictive Model Lab",
    "📌 Insights & Methodology"
])

# =====================================================================
# TAB 1: DATA OVERVIEW - MIXED CHART TYPES
# =====================================================================
with overview_tab:
    st.subheader("Dataset overview")
    c1, c2 = st.columns([1.25, 1])

    with c1:
        year_counts = (
            all_data["financial_year"].astype(str)
            .value_counts().sort_index()
            .rename_axis("Financial Year")
            .reset_index(name="Records")
        )
        fig = px.area(
            year_counts, x="Financial Year", y="Records",
            markers=True, title="HMIS records across financial years"
        )
        st.plotly_chart(plotly_layout(fig, 360), use_container_width=True)

    with c2:
        schema_counts = (
            all_data["schema_version"].fillna("Unknown").astype(str)
            .value_counts().rename_axis("Schema").reset_index(name="Records")
            if "schema_version" in all_data.columns
            else pd.DataFrame({"Schema": ["Available"], "Records": [len(all_data)]})
        )
        fig = px.pie(
            schema_counts, names="Schema", values="Records",
            hole=0.55, title="Records by schema version"
        )
        st.plotly_chart(plotly_layout(fig, 360), use_container_width=True)

    st.subheader("Target variable distribution")
    target_values = model_data[TARGET].dropna()
    if not target_values.empty:
        fig = px.histogram(
            x=target_values, nbins=35, marginal="box",
            title="Distribution of reported Allopathic OPD values",
            labels={"x": "Allopathic OPD", "y": "Number of records"}
        )
        st.plotly_chart(plotly_layout(fig, 390), use_container_width=True)

    st.subheader("Cleaned data preview")
    st.dataframe(all_data.head(30), use_container_width=True)
    st.download_button(
        "⬇️ Download loaded dataset",
        all_data.to_csv(index=False).encode("utf-8"),
        "SmartCare_UP_loaded_dataset.csv",
        "text/csv"
    )

    missing = (
        all_data.isna().mean().mul(100)
        .sort_values(ascending=False).head(15)
        .rename_axis("Column").reset_index(name="Missing (%)")
    )
    fig = px.bar(
        missing.sort_values("Missing (%)"),
        x="Missing (%)", y="Column", orientation="h",
        title="Columns with the highest missing-value percentage",
        text_auto=".1f"
    )
    st.plotly_chart(plotly_layout(fig, 470), use_container_width=True)

# =====================================================================
# TAB 2: EXPLORATORY ANALYSIS - BAR, LINE, SCATTER, HEATMAP
# =====================================================================
with explore_tab:
    st.subheader("District-level healthcare exploration")
    district_list = sorted(model_data["district"].dropna().astype(str).unique())
    chosen_district = st.selectbox(
        "Select district", ["All districts"] + district_list
    )
    view = model_data.copy()
    if chosen_district != "All districts":
        view = view[view["district"].astype(str) == chosen_district]

    # Snapshot sequence: do not label it as monthly values.
    view = view.copy()
    view["Report Snapshot"] = (
        view["financial_year"].astype(str)
        + " | "
        + view["report_upto"].astype(str)
    )

    st.metric("Records in selected view", f"{len(view):,}")

    # Line chart
    trend = (
        view.groupby("Report Snapshot", as_index=False)[TARGET]
        .mean()
    )
    fig = px.line(
        trend, x="Report Snapshot", y=TARGET, markers=True,
        title="Average reported Allopathic OPD by available report snapshot"
    )
    st.plotly_chart(plotly_layout(fig, 390), use_container_width=True)

    # Horizontal bar chart: top districts
    if chosen_district == "All districts":
        district_opd = (
            model_data.groupby("district", as_index=False)[TARGET]
            .mean().nlargest(15, TARGET)
        )
        fig = px.bar(
            district_opd.sort_values(TARGET),
            x=TARGET, y="district", orientation="h",
            title="Top 15 districts by average reported Allopathic OPD",
            text_auto=".3s"
        )
        st.plotly_chart(plotly_layout(fig, 520), use_container_width=True)

    # Indicator scatter plot
    possible_indicators = [
        c for c in model_data.select_dtypes(include=[np.number]).columns
        if c != TARGET and not c.startswith("allopathic_opd_")
        and model_data[c].notna().sum() > 10
        and model_data[c].nunique(dropna=True) > 5
    ]

    if possible_indicators:
        selected_indicator = st.selectbox(
            "Select an indicator to compare with Allopathic OPD",
            sorted(possible_indicators)
        )
        scatter_data = model_data[
            [selected_indicator, TARGET, "district"]
        ].dropna()
        if len(scatter_data) > 1500:
            scatter_data = scatter_data.sample(1500, random_state=42)
        fig = px.scatter(
            scatter_data, x=selected_indicator, y=TARGET,
            color="district", opacity=0.65,
            title=f"{selected_indicator} vs reported Allopathic OPD",
            hover_data=["district"]
        )
        st.plotly_chart(plotly_layout(fig, 470), use_container_width=True)

    # Correlation heatmap for selected high-coverage numeric indicators.
    st.subheader("Healthcare indicator correlation")
    numeric_data = model_data.select_dtypes(include=[np.number])
    corr_candidates = [
        c for c in numeric_data.columns
        if numeric_data[c].notna().mean() >= 0.5
        and numeric_data[c].nunique(dropna=True) > 1
    ]
    corr_candidates = corr_candidates[:18]
    if TARGET in numeric_data.columns and TARGET not in corr_candidates:
        corr_candidates = ([TARGET] + corr_candidates)[:18]
    if len(corr_candidates) >= 2:
        corr = numeric_data[corr_candidates].corr()
        fig = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Correlation heatmap of selected numeric indicators"
        )
        st.plotly_chart(plotly_layout(fig, 600), use_container_width=True)
    else:
        st.info("Not enough numeric indicators are available for a correlation heatmap.")

# =====================================================================
# TAB 3: PREDICTIVE ANALYSIS - FIVE MODELS + VARIED CHARTS
# =====================================================================
with models_tab:
    st.subheader("Five-algorithm model comparison")
    st.markdown("""
    <div class="section-note">
    All five models use the same train/test split and the same predictor set.
    The test set contains complete districts not included in training.
    Lower MAE/RMSE is better; higher R² is better.
    </div>
    """, unsafe_allow_html=True)

    try:
        with st.spinner("Training all five regression algorithms..."):
            (
                metrics, fitted, predictions, features, num_cols, cat_cols,
                X_train, X_test, y_train, y_test
            ) = train_models(model_data)
    except Exception as exc:
        st.error(f"Model training error: {exc}")
        st.stop()

    best = metrics.iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Lowest RMSE", best["Algorithm"])
    m2.metric("Best model MAE", f"{best['MAE']:,.0f}")
    m3.metric("Best model R²", f"{best['R²']:.3f}")

    st.subheader("Performance scorecard")
    st.dataframe(
        metrics.style.format({
            "MAE": "{:,.2f}",
            "RMSE": "{:,.2f}",
            "R²": "{:.3f}"
        }).highlight_min(subset=["MAE", "RMSE"], color="#14532d")
          .highlight_max(subset=["R²"], color="#14532d"),
        use_container_width=True,
        hide_index=True
    )

    # Three separate charts, because metrics have different scales.
    st.subheader("Compare the five models")
    a, b = st.columns(2)
    with a:
        fig = px.bar(
            metrics.sort_values("MAE"),
            x="Algorithm", y="MAE", color="Algorithm",
            title="MAE by algorithm (lower is better)",
            text_auto=".3s"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(plotly_layout(fig, 390), use_container_width=True)
    with b:
        fig = px.bar(
            metrics.sort_values("RMSE"),
            x="Algorithm", y="RMSE", color="Algorithm",
            title="RMSE by algorithm (lower is better)",
            text_auto=".3s"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(plotly_layout(fig, 390), use_container_width=True)

    fig = px.bar(
        metrics.sort_values("R²"),
        x="Algorithm", y="R²", color="Algorithm",
        title="R² by algorithm (higher is better)",
        text_auto=".3f"
    )
    fig.add_hline(y=0, line_dash="dash", line_color="white")
    fig.update_layout(showlegend=False)
    st.plotly_chart(plotly_layout(fig, 390), use_container_width=True)

    # Select a model to inspect predictions and residuals.
    model_names = metrics["Algorithm"].tolist()
    chosen_model = st.selectbox(
        "Select model for detailed diagnostic charts",
        model_names,
        index=model_names.index(best["Algorithm"])
    )
    chosen = predictions[predictions["Algorithm"] == chosen_model].copy()

    st.subheader(f"Diagnostic analysis: {chosen_model}")
    left, right = st.columns(2)

    # Scatter: actual vs predicted.
    with left:
        fig = px.scatter(
            chosen, x="Actual_OPD", y="Predicted_OPD",
            color="district", opacity=0.7,
            title="Actual vs predicted OPD",
            hover_data=["district", "financial_year", "report_upto"]
        )
        max_val = float(max(
            chosen["Actual_OPD"].max(),
            chosen["Predicted_OPD"].max()
        ))
        min_val = float(min(
            chosen["Actual_OPD"].min(),
            chosen["Predicted_OPD"].min()
        ))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines", name="Ideal prediction",
            line=dict(dash="dash", color="white")
        ))
        st.plotly_chart(plotly_layout(fig, 440), use_container_width=True)

    # Histogram: residual distribution.
    with right:
        fig = px.histogram(
            chosen, x="Residual", nbins=35, marginal="box",
            title="Prediction residual distribution",
            labels={"Residual": "Actual − predicted OPD"}
        )
        fig.add_vline(x=0, line_dash="dash", line_color="white")
        st.plotly_chart(plotly_layout(fig, 440), use_container_width=True)

    # Box plot: absolute errors across all five algorithms.
    st.subheader("Error distribution across all five algorithms")
    fig = px.box(
        predictions, x="Algorithm", y="Absolute_Error",
        color="Algorithm", points="outliers",
        title="Absolute prediction error by algorithm"
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(plotly_layout(fig, 450), use_container_width=True)

    # Residual scatter: identify patterns in errors.
    fig = px.scatter(
        chosen, x="Predicted_OPD", y="Residual",
        color="district", opacity=0.65,
        title="Residuals vs predicted OPD",
        labels={"Residual": "Actual − predicted OPD"}
    )
    fig.add_hline(y=0, line_dash="dash", line_color="white")
    st.plotly_chart(plotly_layout(fig, 430), use_container_width=True)

    st.subheader("Test-set prediction sample")
    st.dataframe(
        chosen[
            [
                "district", "financial_year", "report_upto",
                "Actual_OPD", "Predicted_OPD", "Absolute_Error"
            ]
        ].head(100).round(2),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download all five models' test predictions",
        predictions.to_csv(index=False).encode("utf-8"),
        "SmartCare_UP_all_model_predictions.csv",
        "text/csv"
    )
    st.download_button(
        "⬇️ Download model comparison metrics",
        metrics.to_csv(index=False).encode("utf-8"),
        "SmartCare_UP_model_comparison.csv",
        "text/csv"
    )

    st.caption(
        f"Predictor fields: {len(features)} "
        f"({len(num_cols)} numeric, {len(cat_cols)} categorical). "
        f"Training records: {len(X_train):,}; test records: {len(X_test):,}; "
        f"held-out districts: {model_data.loc[X_test.index, 'district'].nunique():,}."
    )

# =====================================================================
# TAB 4: FINDINGS + METHODOLOGY
# =====================================================================
with conclusion_tab:
    st.subheader("Project methodology")
    st.markdown("""
    **Problem statement:** District-level healthcare planning benefits from
    evidence about reported healthcare service volumes. This project studies
    Uttar Pradesh HMIS indicators and estimates reported Allopathic OPD volume.

    **Data preparation**
    1. Load the cleaned HMIS master CSV.
    2. Validate district, financial year, report month and OPD target fields.
    3. Restrict model training to the detailed 2017–2020 schema.
    4. Remove rows without a valid non-negative OPD target.
    5. Convert measurement fields to numeric.
    6. Impute missing predictor values inside the model pipeline.

    **Predictive analysis**
    1. Use the same predictor fields for all five algorithms.
    2. Exclude direct Allopathic OPD component fields from predictors.
    3. Hold out entire districts using GroupShuffleSplit.
    4. Train Linear Regression, Decision Tree, Random Forest,
       Gradient Boosting and Extra Trees.
    5. Evaluate with MAE, RMSE and R².
    6. Compare error metrics and inspect residual charts.
    """)

    st.subheader("How to interpret the metrics")
    st.markdown("""
    - **MAE:** Average absolute error in OPD units. Lower is better.
    - **RMSE:** Penalizes large errors more strongly. Lower is better.
    - **R²:** Measures fit relative to a baseline. It can be negative
      when predictions on unseen districts perform poorly.
    """)

    st.warning("""
    **Data limitation:** HMIS source files are sparse "upto month" snapshots.
    Some fields may be cumulative. Do not interpret the snapshot values as
    individual monthly totals. This project currently estimates OPD volume
    recorded in a snapshot; it does not validate future-month forecasting.
    """)

    st.info("""
    **For future-month forecasting:** obtain consecutive monthly HMIS files,
    verify cumulative versus monthly reporting for each indicator, derive
    monthly increments only from comparable consecutive snapshots, and use
    a chronological time-based train/test split.
    """)

st.markdown("---")
st.caption(
    "SmartCare-UP | Academic Predictive Analysis Project | "
    "Source: user-provided Uttar Pradesh HMIS files"
)
