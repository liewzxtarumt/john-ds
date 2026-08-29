"""
Telcom Customer Churn — Interactive Dashboard
==============================================

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scipy.stats import chi2_contingency

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ==========================================================================
# PAGE CONFIG
# ==========================================================================

st.set_page_config(
    page_title="Telcom Customer Churn Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================================
# CUSTOM CSS
# ==========================================================================

CUSTOM_CSS = """
<style>

/* ----------------------------------------------------------------------
   Main page
---------------------------------------------------------------------- */

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2rem;
}


/* ----------------------------------------------------------------------
   Metrics
---------------------------------------------------------------------- */

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
    border: 1px solid #2d3748;
    border-radius: 14px;
    padding: 14px 16px 8px 16px;
}

[data-testid="stMetricLabel"] {
    color: #9CA3AF;
}

[data-testid="stMetricValue"] {
    color: #F9FAFB;
}

h1, h2, h3 {
    font-weight: 700;
}


/* ----------------------------------------------------------------------
   Top navigation tabs
   ---------------------------------------------------------------------- */

.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
    border-bottom: 1px solid #dddddd;
}

.stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    border-radius: 0 !important;
    padding: 10px 0 12px 0 !important;
    color: #333333 !important;
    font-size: 16px;
    white-space: nowrap;
}

.stTabs [data-baseweb="tab"] p {
    color: inherit !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #e45757 !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: #e45757 !important;
    height: 2px !important;
}

/* ----------------------------------------------------------------------
   Sidebar
---------------------------------------------------------------------- */

[data-testid="stSidebar"] {
    background-color: #f4f6f9;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.35rem;
    padding-left: 1.05rem;
    padding-right: 1.05rem;
}

.sidebar-brand {
    padding: 0 4px 8px 4px;
    margin-bottom: 14px;
}

.sidebar-company {
    font-size: 16px;
    font-weight: 700;
    color: #374151;
    line-height: 1.35;
    text-align: left;
    margin-bottom: 7px;
}

.sidebar-course {
    font-size: 14px;
    font-weight: 500;
    color: #6B7280;
    line-height: 1.35;
    text-align: left;
}

.sidebar-best-model {
    background: #e2f5eb;
    border-radius: 10px;
    padding: 12px 13px;
    margin: 0 0 12px 0;
    color: #087443;
    text-align: left;
    box-sizing: border-box;
}

.sidebar-best-model-title {
    font-size: 14px;
    font-weight: 600;
    line-height: 1.4;
    margin: 0 0 8px 0;
    text-align: left;
}

.sidebar-best-model-metrics {
    display: flex;
    flex-direction: column;
    gap: 3px;
    padding-left: 22px;
}

.sidebar-metric-link {
    font-size: 13px;
    font-weight: 500;
    line-height: 1.4;
    margin: 0;
    padding: 0;
    text-align: left;
    color: #087443 !important;
    text-decoration: none !important;
}

.sidebar-metric-link:hover {
    text-decoration: underline !important;
}

.sidebar-auto-loaded {
    background: #e2f5eb;
    border-radius: 10px;
    padding: 12px 13px;
    margin: 14px 0 0 0;
    color: #087443;
    text-align: left;
    box-sizing: border-box;
}

.sidebar-auto-loaded-title {
    font-size: 14px;
    font-weight: 600;
    line-height: 1.4;
    margin: 0 0 7px 0;
    text-align: left;
}

.sidebar-auto-loaded-file {
    font-size: 12px;
    font-weight: 500;
    line-height: 1.4;
    padding-left: 22px;
    margin: 0;
    text-align: left;
    color: #087443;
}

/* ----------------------------------------------------------------------
   Section selector
---------------------------------------------------------------------- */

.analysis-selector {
    margin-top: 8px;
    margin-bottom: 22px;
}


/* ----------------------------------------------------------------------
   Headings
---------------------------------------------------------------------- */

.major-heading {
    font-size: 30px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 12px;
}

.sub-heading {
    font-size: 22px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 10px;
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================================================
# SIDEBAR LAYOUT PLACEHOLDERS
# ==========================================================================

sidebar_brand = st.sidebar.empty()
sidebar_best_model = st.sidebar.empty()
sidebar_file = st.sidebar.empty()

sidebar_brand.markdown(
    """
    <div class="sidebar-brand">
        <div class="sidebar-company">
            LT Telecommunications Service Sdn. Bhd. (LTTS)
        </div>
        <div class="sidebar-course">
            BMDS2003 Data Science
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# CONSTANTS
# ==========================================================================

LOCAL_DATA_CANDIDATES = [
    "Telco_Customer_Churn.csv",
    "Telco_Cusomer_Churn.csv",
    "Telco-Customer-Churn.csv",
    "telco_customer_churn.csv",
]

PALETTE = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#937860",
]

CONTINUOUS_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

BINARY_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn",
]

MULTI_CLASS_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]


# ==========================================================================
# BEST PARAMETERS
# ==========================================================================

BEST_PARAMS = {
    "Logistic Regression": {
        "C": 500,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42,
    },
    "Random Forest": {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 4,
        "random_state": 42,
        "n_jobs": -1,
    },
    "SVM": {
        "C": 10,
        "kernel": "rbf",
        "gamma": "auto",
        "probability": True,
        "random_state": 42,
    },
}


# ==========================================================================
# HELPER FUNCTIONS
# ==========================================================================

def is_text_col(s: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )


# --------------------------------------------------------------------------
# Missing Value
# --------------------------------------------------------------------------

def make_missing_values_figure(raw_df: pd.DataFrame):
    missing_counts = raw_df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    plot_df = missing_counts.rename("Missing Values").reset_index()
    plot_df.columns = ["Variable", "Missing Values"]

    fig = px.bar(
        plot_df,
        x="Variable",
        y="Missing Values",
        text="Missing Values",
        title="Missing Values Identified After Data Type Conversion",
        labels={
            "Variable": "Variable",
            "Missing Values": "Number of Missing Values",
        },
    )

    fig.update_traces(
        textposition="outside",
    )

    fig.update_layout(
        height=450,
        showlegend=False,
        margin=dict(
            l=70,
            r=30,
            t=60,
            b=70,
        ),
    )

    return fig


# --------------------------------------------------------------------------
# Tenure vs TotalCharges
# --------------------------------------------------------------------------

def make_tenure_totalcharges_figure(raw_df: pd.DataFrame):

    plot_df = raw_df[
        ["tenure", "TotalCharges"]
    ].copy()

    plot_df["TotalCharges"] = pd.to_numeric(
        plot_df["TotalCharges"],
        errors="coerce",
    )

    missing_mask = plot_df["TotalCharges"].isna()

    normal_df = plot_df.loc[
        ~missing_mask
    ].copy()

    missing_df = plot_df.loc[
        missing_mask
    ].copy()

    missing_df["TotalCharges"] = 0.0

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=normal_df["tenure"],
            y=normal_df["TotalCharges"],
            mode="markers",
            name="Observed TotalCharges",
            marker=dict(
                size=5,
                opacity=0.45,
            ),
            hovertemplate=(
                "Tenure: %{x} months<br>"
                "TotalCharges: %{y:.2f}"
                "<extra></extra>"
            ),
        )
    )

    if len(missing_df) > 0:

        fig.add_trace(
            go.Scatter(
                x=missing_df["tenure"],
                y=missing_df["TotalCharges"],
                mode="markers",
                name="Missing TotalCharges",
                marker=dict(
                    size=12,
                    symbol="x",
                ),
                hovertemplate=(
                    "Tenure: %{x} months<br>"
                    "TotalCharges: Missing"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Relationship Between Tenure and TotalCharges",
        xaxis_title="Tenure (months)",
        yaxis_title="Total Charges",
        height=450,
    )

    return fig


# --------------------------------------------------------------------------
# Missing before / after
# --------------------------------------------------------------------------

def make_missing_replacement_figure(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
):

    before = int(
        pd.to_numeric(
            raw_df["TotalCharges"],
            errors="coerce",
        ).isna().sum()
    )

    after = int(
        clean_df["TotalCharges"].isna().sum()
    )

    plot_df = pd.DataFrame({
        "Stage": [
            "Before Replacement",
            "After Replacement",
        ],
        "Missing Values": [
            before,
            after,
        ],
    })

    fig = px.bar(
        plot_df,
        x="Stage",
        y="Missing Values",
        text="Missing Values",
        title="Missing Values Before and After Replacement",
        labels={
            "Missing Values":
                "Number of Missing Values",
        },
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        showlegend=False,
        height=400,
    )

    return fig


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------

def make_encoding_figure(
    encoded_df: pd.DataFrame,
    clean_df: pd.DataFrame,
):

    original_predictors = (
        clean_df.shape[1] - 1
    )

    encoded_predictors = (
        encoded_df.shape[1] - 1
    )

    plot_df = pd.DataFrame({
        "Stage": [
            "Before Encoding",
            "After Encoding",
        ],
        "Number of Predictors": [
            original_predictors,
            encoded_predictors,
        ],
    })

    fig = px.bar(
        plot_df,
        x="Stage",
        y="Number of Predictors",
        text="Number of Predictors",
        title="Feature Expansion Through Encoding",
        labels={
            "Number of Predictors":
                "Number of Numerical Predictors",
        },
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        showlegend=False,
        height=400,
    )

    return fig


# --------------------------------------------------------------------------
# Standardisation
# --------------------------------------------------------------------------

def make_standardisation_figure(
    encoded_df: pd.DataFrame,
    X_train: pd.DataFrame,
):

    stats_df = pd.DataFrame({
        "Feature": CONTINUOUS_COLS,

        "Before Standardisation": [
            encoded_df.loc[
                X_train.index,
                c,
            ].mean()
            for c in CONTINUOUS_COLS
        ],

        "After Standardisation": [
            X_train[c].mean()
            for c in CONTINUOUS_COLS
        ],
    })

    plot_df = stats_df.melt(
        id_vars="Feature",
        var_name="Stage",
        value_name="Mean",
    )

    fig = px.bar(
        plot_df,
        x="Feature",
        y="Mean",
        color="Stage",
        barmode="group",
        text_auto=".2f",
        title="Mean Comparison Before and After Standardisation",
        labels={
            "Mean": "Mean Value",
        },
    )

    fig.update_layout(
        height=450,
    )

    return fig


# ==========================================================================
# DATA LOADING
# ==========================================================================

@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:

    df = pd.read_csv(file)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    return df


# ==========================================================================
# DATA CLEANING
# ==========================================================================

@st.cache_data(show_spinner=False)
def clean_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df_clean = df.drop(
        columns=["customerID"]
    ).copy()

    df_clean["TotalCharges"] = (
        df_clean["TotalCharges"]
        .fillna(0.0)
    )

    return df_clean


# ==========================================================================
# ENCODING
# ==========================================================================

@st.cache_data(show_spinner=False)
def encode_data(
    df_clean: pd.DataFrame,
) -> pd.DataFrame:

    enc = df_clean.copy()

    enc["gender"] = enc["gender"].map({
        "Female": 1,
        "Male": 0,
    })

    enc["Partner"] = enc["Partner"].map({
        "Yes": 1,
        "No": 0,
    })

    enc["Dependents"] = enc["Dependents"].map({
        "Yes": 1,
        "No": 0,
    })

    enc["PhoneService"] = enc["PhoneService"].map({
        "Yes": 1,
        "No": 0,
    })

    enc["PaperlessBilling"] = (
        enc["PaperlessBilling"].map({
            "Yes": 1,
            "No": 0,
        })
    )

    enc["Churn"] = enc["Churn"].map({
        "Yes": 1,
        "No": 0,
    })

    enc = pd.get_dummies(
        enc,
        columns=MULTI_CLASS_COLS,
        drop_first=True,
        dtype=int,
    )

    return enc


# ==========================================================================
# TRAIN / TEST SPLIT + SCALING
# ==========================================================================

@st.cache_resource(show_spinner=False)
def split_and_scale(
    encoded_df: pd.DataFrame,
):

    X = encoded_df.drop(
        columns=["Churn"]
    )

    y = encoded_df["Churn"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[CONTINUOUS_COLS] = (
        scaler.fit_transform(
            X_train[CONTINUOUS_COLS]
        )
    )

    X_test[CONTINUOUS_COLS] = (
        scaler.transform(
            X_test[CONTINUOUS_COLS]
        )
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
    )


# ==========================================================================
# TRAIN MODELS
# ==========================================================================

@st.cache_resource(show_spinner=False)
def train_models(
    X_train,
    y_train,
):

    models = {

        "Logistic Regression":
            LogisticRegression(
                **BEST_PARAMS[
                    "Logistic Regression"
                ]
            ),

        "Random Forest":
            RandomForestClassifier(
                **BEST_PARAMS[
                    "Random Forest"
                ]
            ),

        "SVM":
            SVC(
                **BEST_PARAMS["SVM"]
            ),
    }

    for model in models.values():
        model.fit(
            X_train,
            y_train,
        )

    return models


# ==========================================================================
# EVALUATE MODELS
# ==========================================================================

@st.cache_data(show_spinner=False)
def evaluate_models(
    _models,
    X_train,
    y_train,
    X_test,
    y_test,
):

    rows = []
    curves = {}
    cms = {}

    for name, model in _models.items():

        train_pred = model.predict(
            X_train
        )

        test_pred = model.predict(
            X_test
        )

        test_proba = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        train_acc = round(accuracy_score(y_train, train_pred) * 100, 2)
        test_acc = round(accuracy_score(y_test, test_pred) * 100, 2)
        prec = round(precision_score(y_test, test_pred, zero_division=0) * 100, 2)
        rec = round(recall_score(y_test, test_pred, zero_division=0) * 100, 2)
        f1 = round(f1_score(y_test, test_pred, zero_division=0) * 100, 2)
        auc = round(roc_auc_score(y_test, test_proba) * 100, 2)

        avg_score = round(np.mean([
            test_acc,
            prec,
            rec,
            f1,
            auc,
        ]), 2)

        rows.append({

            "Model":
                name,

            "Training Accuracy (%)":
                train_acc,

            "Test Accuracy (%)":
                test_acc,

            "Precision (%)":
                prec,

            "Recall (%)":
                rec,

            "F1-score (%)":
                f1,

            "AUC (%)":
                auc,

            "Average Score (%)":
                avg_score,
        })

        fpr, tpr, _ = roc_curve(
            y_test,
            test_proba,
        )

        curves[name] = (
            fpr,
            tpr,
        )

        cms[name] = confusion_matrix(
            y_test,
            test_pred,
        )

    return (
        pd.DataFrame(rows),
        curves,
        cms,
    )


# ==========================================================================
# CHI-SQUARE TEST
# ==========================================================================

@st.cache_data(show_spinner=False)
def chi_square_table(
    df_clean: pd.DataFrame,
) -> pd.DataFrame:

    results = []

    columns_to_test = [

        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
        "SeniorCitizen",
    ]

    for col in columns_to_test:

        if col not in df_clean.columns:
            continue

        table = pd.crosstab(
            df_clean[col],
            df_clean["Churn"],
        )

        if (
            table.shape[0] < 2
            or table.shape[1] < 2
        ):
            continue

        chi2, p, _, _ = (
            chi2_contingency(table)
        )

        results.append({

            "Variable":
                col,

            "Chi-Square":
                round(chi2, 4),

            "p-value":
                p,

            "Significant (p < 0.05)":
                (
                    "Yes"
                    if p < 0.05
                    else "No"
                ),
        })

    result_df = (
        pd.DataFrame(results)
        .sort_values(
            "Chi-Square",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return result_df


# ==========================================================================
# TOP HEADER
# ==========================================================================

st.title(
    "📉 Telcom Customer Churn Dashboard"
)

st.caption(
    "Telcom Customer Churn analysis and prediction dashboard."
)


# ==========================================================================
# AUTO-DETECT CSV
# ==========================================================================

local_path = next(
    (
        p
        for p in LOCAL_DATA_CANDIDATES
        if os.path.exists(p)
    ),
    None,
)

data_source = None


if local_path is not None:

    data_source = local_path

    with sidebar_file.container():

        st.markdown(
            """
            <div class="sidebar-auto-loaded">
                <div class="sidebar-auto-loaded-title">
                    📁 Auto-loaded
                </div>
                <div class="sidebar-auto-loaded-file">
                    Telco_Cusomer_Churn.csv
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


else:

    with sidebar_file.container():

        st.warning(
            "No local CSV found next to app.py."
        )

        data_source = st.file_uploader(
            "Upload Telco_Customer_Churn.csv",
            type=["csv"],
        )


# ==========================================================================
# NO DATA
# ==========================================================================

if data_source is None:

    st.info(
        "Place **Telco_Customer_Churn.csv** "
        "in the same folder as **app.py**, "
        "or upload the CSV using the sidebar."
    )

    st.stop()


# ==========================================================================
# LOAD RAW DATA
# ==========================================================================

raw_df = load_data(
    data_source
)


# ==========================================================================
# SCHEMA VALIDATION
# ==========================================================================

EXPECTED_COLUMNS = [

    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

missing_cols = [
    c
    for c in EXPECTED_COLUMNS
    if c not in raw_df.columns
]

if missing_cols:

    st.error(
        "This CSV does not match the expected "
        "Telco Customer Churn schema."
    )

    st.write(
        "**Missing column(s):**",
        missing_cols,
    )

    st.write(
        "**Columns found:**",
        list(raw_df.columns),
    )

    st.stop()


# ==========================================================================
# PREPARE DATA
# ==========================================================================

clean_df = clean_data(
    raw_df
)

encoded_df = encode_data(
    clean_df
)

(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
) = split_and_scale(
    encoded_df
)


# ==========================================================================
# TRAIN MODELS
# ==========================================================================

with st.spinner(
    "Training Logistic Regression, Random Forest and SVM..."
):

    models = train_models(
        X_train,
        y_train,
    )

    (
        results_df,
        roc_curves,
        conf_matrices,
    ) = evaluate_models(
        models,
        X_train,
        y_train,
        X_test,
        y_test,
    )


# ==========================================================================
# BEST MODEL
# ==========================================================================

best_row = results_df.loc[
    results_df[
        "Test Accuracy (%)"
    ].idxmax()
]


# ==========================================================================
# SIDEBAR BEST MODEL
# ==========================================================================

metric_link_map = {
    "Training Accuracy (%)": "training_accuracy",
    "Test Accuracy (%)": "test_accuracy",
    "Precision (%)": "precision",
    "Recall (%)": "recall",
    "F1-score (%)": "f1_score",
    "AUC (%)": "auc",
}

metric_query_map = {
    query_value: metric_label
    for metric_label, query_value in metric_link_map.items()
}

sidebar_metrics_html = ""

for metric_label, query_value in metric_link_map.items():
    sidebar_metrics_html += (
        f'<a class="sidebar-metric-link" '
        f'href="?metric={query_value}" '
        f'>{metric_label.replace(" (%)", "")}: '
        f'{best_row[metric_label]:.2f}%</a>'
    )

sidebar_best_model.markdown(
    f"""
    <div class="sidebar-best-model">
        <div class="sidebar-best-model-title">
            🏆 Best Model: {best_row['Model']}
        </div>
        <div class="sidebar-best-model-metrics">
            {sidebar_metrics_html}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# TOP NAVIGATION — UNDERLINE TABS
# ==========================================================================

requested_metric = st.query_params.get("metric", "")
valid_metric = metric_query_map.get(
    requested_metric
)

metric_options = [
    "Training Accuracy (%)",
    "Test Accuracy (%)",
    "Precision (%)",
    "Recall (%)",
    "F1-score (%)",
    "AUC (%)",
]

tab_overview, tab_eda, tab_model, tab_predict, tab_about = st.tabs(
    [
        "Overview",
        "Data Exploratory",
        "Model Performance",
        "Prediction",
        "About",
    ],
    default=(
        "Model Performance"
        if valid_metric
        else "Overview"
    ),
)


# ==========================================================================
# OVERVIEW
# ==========================================================================

with tab_overview:

    st.subheader(
        "Overview Summary"
    )

    churn_rate = (
        (
            raw_df["Churn"] == "Yes"
        ).mean()
        * 100
    )

    avg_tenure = (
        raw_df["tenure"].mean()
    )

    avg_monthly = (
        raw_df["MonthlyCharges"].mean()
    )

    avg_total = (
        raw_df["TotalCharges"].mean()
    )

    c1, c2, c3, c4, c5 = (
        st.columns(5)
    )

    c1.metric(
        "Total Customers",
        f"{raw_df.shape[0]:,}",
    )

    c2.metric(
        "Churn Rate",
        f"{churn_rate:.1f}%",
    )

    c3.metric(
        "Avg. Tenure",
        f"{avg_tenure:.1f} months",
    )

    c4.metric(
        "Avg. Monthly Charges",
        f"${avg_monthly:,.2f}",
    )

    c5.metric(
        "Avg. Total Charges",
        f"${avg_total:,.2f}",
    )

    st.subheader(
        "Raw Data Sample"
    )

    st.dataframe(
        raw_df.head(10),
        use_container_width=True,
    )

    with st.expander(
        "Dataset structure (`df.info()` equivalent)"
    ):

        info_df = pd.DataFrame({

            "Column":
                raw_df.columns,

            "Non-Null Count":
                raw_df.notnull()
                .sum()
                .values,

            "Dtype":
                raw_df.dtypes
                .astype(str)
                .values,
        })

        st.dataframe(
            info_df,
            use_container_width=True,
            height=420,
        )


# ==========================================================================
# EXPLORATORY DATA ANALYSIS
# ==========================================================================

with tab_eda:

    st.markdown(
        '<div class="major-heading">'
        'Exploratory Data Analysis'
        '</div>',
        unsafe_allow_html=True,
    )


    # ======================================================================
    # DATA UNDERSTANDING
    # ======================================================================

    st.markdown(
        '<div class="major-heading">'
        'Data Understanding'
        '</div>',
        unsafe_allow_html=True,
    )

    understanding_tabs = st.tabs([
        "Distribution of Customer Churn",
        "Customer Churn Rate by Contract Type",
        "Correlation Matrix",
    ])


    # ----------------------------------------------------------------------
    # Distribution of Customer Churn
    # ----------------------------------------------------------------------

    with understanding_tabs[0]:

        st.subheader(
            "Distribution of Customer Churn"
        )

        churn_counts = (
            raw_df["Churn"]
            .value_counts()
            .rename({
                "No": "No Churn",
                "Yes": "Churn",
            })
        )

        fig = px.pie(
            names=churn_counts.index,
            values=churn_counts.values,
            color=churn_counts.index,
            color_discrete_sequence=PALETTE,
            hole=0.55,
            title="Distribution of Customer Churn",
        )

        fig.update_traces(
            textinfo="label+percent+value"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    # ----------------------------------------------------------------------
    # Customer Churn Rate by Contract Type
    # ----------------------------------------------------------------------

    with understanding_tabs[1]:

        st.subheader(
            "Customer Churn Rate by Contract Type"
        )

        contract_order = [
            "Month-to-month",
            "One year",
            "Two year",
        ]

        contract_churn = (
            raw_df.groupby(
                "Contract"
            )["Churn"]
            .apply(
                lambda x:
                (x == "Yes").mean()
                * 100
            )
            .reindex(
                contract_order
            )
            .reset_index(
                name="ChurnRate"
            )
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=contract_churn[
                    "Contract"
                ],
                y=contract_churn[
                    "ChurnRate"
                ],
                mode="markers+text",
                text=[
                    f"{rate:.1f}%"
                    for rate
                    in contract_churn[
                        "ChurnRate"
                    ]
                ],
                textposition="top center",
                textfont=dict(
                    size=14,
                    color="#222222",
                ),
                marker=dict(
                    size=16,
                    color="#2E86C1",
                    line=dict(
                        color="black",
                        width=1.2,
                    ),
                ),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Churn Rate: %{y:.1f}%"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

        fig.update_layout(

            title=dict(
                text=(
                    "Customer Churn Rate "
                    "by Contract Type"
                ),
                font=dict(
                    size=18,
                    color="#222222",
                ),
                x=0.5,
                xanchor="center",
            ),

            xaxis=dict(
                title="Contract Type",
                categoryorder="array",
                categoryarray=contract_order,
            ),

            yaxis=dict(
                title="Churn Rate",
                range=[0, 50],
                tickmode="linear",
                tick0=0,
                dtick=10,
                ticksuffix="%",
            ),

            height=430,

            plot_bgcolor="white",
            paper_bgcolor="white",

            margin=dict(
                l=60,
                r=30,
                t=70,
                b=70,
            ),
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor=(
                "rgba(0,0,0,0.12)"
            ),
            zeroline=True,
            zerolinecolor=(
                "rgba(0,0,0,0.25)"
            ),
        )

        fig.update_xaxes(
            showgrid=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    # ----------------------------------------------------------------------
    # Correlation Matrix
    # ----------------------------------------------------------------------

    with understanding_tabs[2]:

        st.subheader(
            "Correlation Matrix of Numerical Variables"
        )

        corr = (
            raw_df.assign(
                TotalCharges=
                raw_df[
                    "TotalCharges"
                ].fillna(0.0)
            )[
                [
                    "SeniorCitizen",
                    "tenure",
                    "MonthlyCharges",
                    "TotalCharges",
                ]
            ]
            .corr()
        )

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="Blues",
            aspect="auto",
            title=(
                "Correlation Matrix "
                "of Numerical Variables"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    st.divider()


    # ======================================================================
    # DATA PREPARATION
    # ======================================================================

    st.markdown(
        '<div class="major-heading">'
        'Data Preparation'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Select a data preparation step below."
    )

    preparation_tabs = st.tabs([
        "Missing Value",
        "Chi-Square Test",
        "Encoding",
        "IQR",
        "Boxplot",
        "Stratified Train/Test Split",
        "Distribution",
        "Feature Scaling",
    ])


    # ======================================================================
    # MISSING VALUE
    # ======================================================================

    with preparation_tabs[0]:

        st.subheader(
            "Missing Value"
        )

        missing_before = int(
            pd.to_numeric(
                raw_df[
                    "TotalCharges"
                ],
                errors="coerce",
            )
            .isnull()
            .sum()
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Missing TotalCharges (raw)",
            missing_before,
        )

        c2.metric(
            "Duplicate rows",
            int(
                raw_df.duplicated()
                .sum()
            ),
        )

        c3.metric(
            "Missing after imputation",
            int(
                clean_df[
                    "TotalCharges"
                ].isnull()
                .sum()
            ),
        )

        st.caption(
            "TotalCharges is converted to numeric "
            "format. The hidden missing values are "
            "then replaced with 0.00 because the "
            "affected records correspond to "
            "customers with zero-month tenure."
        )

        st.plotly_chart(
            make_missing_values_figure(
                raw_df
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            make_tenure_totalcharges_figure(
                raw_df
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            make_missing_replacement_figure(
                raw_df,
                clean_df,
            ),
            use_container_width=True,
        )


    # ======================================================================
    # CHI-SQUARE TEST
    # ======================================================================

    with preparation_tabs[1]:

        st.subheader(
            "Chi-Square Test"
        )

        st.caption(
            "A low p-value (< 0.05) indicates "
            "that the variable has a statistically "
            "significant association with Churn."
        )

        chi_df = chi_square_table(
            clean_df
        )

        heatmap_df = (
            chi_df
            .set_index(
                "Variable"
            )[["p-value"]]
            .T
        )

        fig = px.imshow(
            heatmap_df,
            text_auto=".4f",
            color_continuous_scale="Blues_r",
            aspect="auto",
            height=300,
            title=(
                "Chi-Square Test: "
                "p-values by Variable"
            ),
        )

        fig.update_xaxes(
            tickangle=35
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        display_chi = (
            chi_df.copy()
        )

        display_chi[
            "p-value"
        ] = display_chi[
            "p-value"
        ].map(
            lambda x:
            f"{x:.6f}"
        )

        display_chi[
            "Chi-Square"
        ] = display_chi[
            "Chi-Square"
        ].map(
            lambda x:
            f"{x:.4f}"
        )

        st.dataframe(
            display_chi,
            use_container_width=True,
            hide_index=True,
        )

        significant = chi_df[
            chi_df["p-value"] < 0.05
        ]

        if len(significant) > 0:

            st.success(
                f"{len(significant)} variable(s) "
                "show statistically significant "
                "association with Churn at p < 0.05."
            )

        else:

            st.info(
                "No variables have p < 0.05."
            )


    # ======================================================================
    # ENCODING
    # ======================================================================

    with preparation_tabs[2]:

        st.subheader(
            "Encoding"
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Columns before encoding",
            raw_df.shape[1] - 1,
        )

        c2.metric(
            "Columns after encoding",
            encoded_df.shape[1] - 1,
        )

        st.caption(
            "Binary categorical variables are "
            "converted into 0/1 values, while "
            "multi-category variables are transformed "
            "using one-hot encoding with "
            "drop_first=True."
        )

        st.plotly_chart(
            make_encoding_figure(
                encoded_df,
                clean_df,
            ),
            use_container_width=True,
        )

        with st.expander(
            "Preview encoded feature matrix"
        ):

            st.dataframe(
                encoded_df.head(10),
                use_container_width=True,
            )


    # ======================================================================
    # IQR
    # ======================================================================

    with preparation_tabs[3]:

        st.subheader(
            "IQR"
        )

        Q1 = (
            encoded_df[
                CONTINUOUS_COLS
            ].quantile(0.25)
        )

        Q3 = (
            encoded_df[
                CONTINUOUS_COLS
            ].quantile(0.75)
        )

        IQR = Q3 - Q1

        lower = (
            Q1 - 1.5 * IQR
        )

        upper = (
            Q3 + 1.5 * IQR
        )

        outlier_rows = []

        for col in CONTINUOUS_COLS:

            n_out = int(
                (
                    (
                        encoded_df[col]
                        < lower[col]
                    )
                    |
                    (
                        encoded_df[col]
                        > upper[col]
                    )
                ).sum()
            )

            outlier_rows.append({

                "Variable":
                    col,

                "Q1":
                    Q1[col],

                "Q3":
                    Q3[col],

                "IQR":
                    IQR[col],

                "Lower Fence":
                    lower[col],

                "Upper Fence":
                    upper[col],

                "Potential Outliers":
                    n_out,

                "Percentage (%)":
                    (
                        n_out
                        / len(encoded_df)
                        * 100
                    ),
            })

        st.dataframe(
            pd.DataFrame(
                outlier_rows
            ).round(2),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Potential outliers are identified using "
            "the IQR rule: values below Q1 − 1.5×IQR "
            "or above Q3 + 1.5×IQR."
        )


    # ======================================================================
    # BOXPLOT
    # ======================================================================

    with preparation_tabs[4]:

        st.subheader(
            "Boxplot"
        )

        box_df = raw_df[
            CONTINUOUS_COLS
        ].copy()

        box_df[
            "TotalCharges"
        ] = (
            box_df[
                "TotalCharges"
            ].fillna(0.0)
        )

        fig = px.box(
            box_df,
            y=CONTINUOUS_COLS,
            points=False,
            title=(
                "Boxplot for Numerical Variables"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    # ======================================================================
    # STRATIFIED TRAIN / TEST SPLIT
    # ======================================================================

    with preparation_tabs[5]:

        st.subheader(
            "Stratified Train/Test Split (80% / 20%)"
        )

        dist = pd.DataFrame({

            "No Churn (0)": [

                (
                    y_train == 0
                ).mean() * 100,

                (
                    y_test == 0
                ).mean() * 100,
            ],

            "Churn (1)": [

                (
                    y_train == 1
                ).mean() * 100,

                (
                    y_test == 1
                ).mean() * 100,
            ],

        }, index=[

            "Training Set (80%)",

            "Testing Set (20%)",
        ])

        c1, c2 = st.columns(2)

        with c1:

            fig = px.bar(
                dist,
                barmode="stack",
                color_discrete_sequence=[
                    "#2b5c8f",
                    "#d95f02",
                ],
                title=(
                    "Stratified Class "
                    "Distribution Preservation"
                ),
                labels={
                    "value":
                        "Percentage (%)",
                    "index":
                        "",
                },
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with c2:

            st.metric(
                "Training rows",
                f"{X_train.shape[0]:,}",
            )

            st.metric(
                "Testing rows",
                f"{X_test.shape[0]:,}",
            )

            st.metric(
                "Train churn rate",
                f"{y_train.mean() * 100:.2f}%",
            )

            st.metric(
                "Test churn rate",
                f"{y_test.mean() * 100:.2f}%",
            )


    # ======================================================================
    # DISTRIBUTION
    # ======================================================================

    with preparation_tabs[6]:

        st.subheader(
            "Distribution of Numerical Variables"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            fig = px.histogram(
                raw_df,
                x="MonthlyCharges",
                nbins=40,
                marginal="box",
                color_discrete_sequence=[
                    "#2E8B57"
                ],
                title=(
                    "Distribution of "
                    "Monthly Charges"
                ),
            )

            fig.update_layout(
                height=430
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with c2:

            fig = px.histogram(
                raw_df,
                x="tenure",
                nbins=40,
                marginal="box",
                color_discrete_sequence=[
                    "#2E86C1"
                ],
                title=(
                    "Distribution of "
                    "Tenure (months)"
                ),
            )

            fig.update_layout(
                height=430
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with c3:

            total_charges_plot = (
                raw_df[
                    "TotalCharges"
                ].fillna(0.0)
            )

            fig = px.histogram(
                total_charges_plot,
                x=total_charges_plot,
                nbins=40,
                marginal="box",
                color_discrete_sequence=[
                    "#8172B2"
                ],
                title=(
                    "Distribution of "
                    "Total Charges"
                ),
            )

            fig.update_layout(
                height=430
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


    # ======================================================================
    # FEATURE SCALING
    # ======================================================================

    with preparation_tabs[7]:

        st.subheader(
            "Feature Scaling — StandardScaler"
        )

        st.caption(
            "StandardScaler applies Z-score "
            "standardisation to the continuous "
            "variables using the training-set "
            "mean and standard deviation."
        )

        scaled_stats = pd.DataFrame({

            "Feature":
                CONTINUOUS_COLS,

            "Mean Before": [

                encoded_df.loc[
                    X_train.index,
                    c,
                ].mean()

                for c in CONTINUOUS_COLS
            ],

            "Std Before": [

                encoded_df.loc[
                    X_train.index,
                    c,
                ].std()

                for c in CONTINUOUS_COLS
            ],

            "Mean After": [

                X_train[c].mean()

                for c in CONTINUOUS_COLS
            ],

            "Std After": [

                X_train[c].std()

                for c in CONTINUOUS_COLS
            ],
        })

        st.dataframe(
            scaled_stats.round(4),
            use_container_width=True,
            hide_index=True,
        )

        st.plotly_chart(
            make_standardisation_figure(
                encoded_df,
                X_train,
            ),
            use_container_width=True,
        )


# ==========================================================================
# MODELLING & EVALUATION
# ==========================================================================

with tab_model:

    st.subheader(
        "🤖 Modelling & Evaluation"
    )

    st.caption(
        "Logistic Regression, Random Forest and SVM "
        "are trained using the selected notebook "
        "hyperparameters."
    )

    st.subheader(
        "Comparative Results of LR, RF and SVM"
    )

    display_df = results_df.copy()

    for c in display_df.columns[1:]:

        display_df[c] = (
            display_df[c]
            .map(
                lambda x:
                f"{x:.2f}%"
            )
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    best = (
        results_df.assign(

            Average=
                results_df[
                    [
                        "Test Accuracy (%)",
                        "Precision (%)",
                        "Recall (%)",
                        "F1-score (%)",
                        "AUC (%)",
                    ]
                ].mean(axis=1)

        )
        .sort_values(
            "Average",
            ascending=False,
        )
    )

    st.success(
        f"🏆 Recommended model: "
        f"**{best.iloc[0]['Model']}** "
        f"(highest average score)"
    )

    st.subheader(
        "Metric Comparison"
    )

    metric_tabs = st.tabs(
        metric_options,
        default=(
            valid_metric
            if valid_metric
            else metric_options[0]
        ),
    )

    for metric_tab, metric in zip(
        metric_tabs,
        metric_options,
    ):
        with metric_tab:
            fig = px.bar(
                results_df,
                x="Model",
                y=metric,
                color="Model",
                text_auto=".2f",
                color_discrete_sequence=PALETTE,
                title=f"{metric} by Model",
                range_y=[0, 100],
            )

            fig.update_layout(
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    st.subheader(
        "Training vs Test Accuracy"
    )

    fig = go.Figure()

    fig.add_bar(
        name="Training Accuracy",
        x=results_df["Model"],
        y=results_df[
            "Training Accuracy (%)"
        ],
        marker_color="#4C72B0",
    )

    fig.add_bar(
        name="Test Accuracy",
        x=results_df["Model"],
        y=results_df[
            "Test Accuracy (%)"
        ],
        marker_color="#DD8452",
    )

    fig.update_layout(
        barmode="group",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 100],
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader(
        "All Metrics — Grouped View"
    )

    melted = results_df.melt(
        id_vars="Model",
        value_vars=[
            "Test Accuracy (%)",
            "Precision (%)",
            "Recall (%)",
            "F1-score (%)",
            "AUC (%)",
        ],
        var_name="Metric",
        value_name="Value",
    )

    fig = px.bar(
        melted,
        x="Metric",
        y="Value",
        color="Model",
        barmode="group",
        color_discrete_sequence=PALETTE,
        range_y=[0, 100],
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.divider()

    st.subheader(
        "Model Inspection"
    )

    chosen_model = st.selectbox(
        "Inspect a model",
        list(models.keys()),
    )

    model = models[
        chosen_model
    ]

    c1, c2 = st.columns(2)

    with c1:

        cm = conf_matrices[
            chosen_model
        ]

        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            x=[
                "Predicted No Churn",
                "Predicted Churn",
            ],
            y=[
                "Actual No Churn",
                "Actual Churn",
            ],
            title=(
                f"Confusion Matrix — "
                f"{chosen_model}"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with c2:

        fpr, tpr = (
            roc_curves[
                chosen_model
            ]
        )

        auc_val = (
            results_df.loc[
                results_df["Model"]
                == chosen_model,
                "AUC (%)",
            ].values[0]
            / 100
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=(
                    f"{chosen_model} "
                    f"(AUC={auc_val:.3f})"
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(
                    dash="dash",
                    color="grey",
                ),
                name="Random Guessing",
            )
        )

        fig.update_layout(
            title=(
                f"ROC Curve — "
                f"{chosen_model}"
            ),
            xaxis_title=(
                "False Positive Rate"
            ),
            yaxis_title=(
                "True Positive Rate"
            ),
            xaxis=dict(
                range=[0, 1]
            ),
            yaxis=dict(
                range=[0, 1]
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.subheader(
        f"Feature Importance — "
        f"{chosen_model}"
    )

    feature_names = X_train.columns

    if chosen_model == "Logistic Regression":

        importance = pd.Series(
            model.coef_[0],
            index=feature_names,
        )

        top15 = (
            importance.reindex(
                importance.abs()
                .sort_values(
                    ascending=False
                )
                .index
            )
            .head(15)
            .sort_values()
        )

        direction = np.where(
            top15.values > 0,
            "Increases churn risk",
            "Decreases churn risk",
        )

        importance_plot_df = (
            pd.DataFrame({
                "Feature":
                    top15.index,

                "Coefficient":
                    top15.values,

                "Direction":
                    direction,
            })
        )

        fig = px.bar(
            importance_plot_df,
            x="Coefficient",
            y="Feature",
            orientation="h",
            color="Direction",
            color_discrete_map={
                "Increases churn risk":
                    "#DD8452",

                "Decreases churn risk":
                    "#4C72B0",
            },
            title=(
                "Top 15 Logistic "
                "Regression Coefficients"
            ),
            labels={
                "Coefficient":
                    "Coefficient",

                "Feature":
                    "Feature",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    elif chosen_model == "Random Forest":

        importance = pd.Series(
            model.feature_importances_,
            index=feature_names,
        )

        top15 = (
            importance
            .sort_values(
                ascending=False
            )
            .head(15)
            .sort_values()
        )

        importance_plot_df = (
            pd.DataFrame({
                "Feature":
                    top15.index,

                "Importance":
                    top15.values,
            })
        )

        fig = px.bar(
            importance_plot_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title=(
                "Top 15 Feature Importances "
                "— Random Forest"
            ),
            labels={
                "Importance":
                    "Importance",

                "Feature":
                    "Feature",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    else:

        st.info(
            "SVM with an RBF kernel does not have "
            "a built-in coefficient/feature-importance "
            "attribute. Permutation importance is "
            "therefore used."
        )

        with st.spinner(
            "Calculating SVM permutation feature importance..."
        ):

            perm = permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=10,
                random_state=42,
                scoring="accuracy",
                n_jobs=-1,
            )

        importance = pd.Series(
            perm.importances_mean,
            index=feature_names,
        )

        top15 = (
            importance
            .sort_values(
                ascending=False
            )
            .head(15)
            .sort_values()
        )

        importance_plot_df = (
            pd.DataFrame({
                "Feature":
                    top15.index,

                "Importance":
                    top15.values,
            })
        )

        fig = px.bar(
            importance_plot_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title=(
                "Top 15 Permutation "
                "Feature Importances — SVM"
            ),
            labels={
                "Importance":
                    "Importance",

                "Feature":
                    "Feature",
            },
        )

        fig.update_layout(
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with st.expander(
        "Full classification report"
    ):

        preds = model.predict(
            X_test
        )

        report = classification_report(
            y_test,
            preds,
            target_names=[
                "No Churn",
                "Churn",
            ],
            output_dict=True,
        )

        st.dataframe(
            pd.DataFrame(
                report
            ).T.round(3),
            use_container_width=True,
        )


# ==========================================================================
# PREDICTION
# ==========================================================================

with tab_predict:

    st.subheader(
        "🔮 Predict Churn for a Customer"
    )

    st.caption(
        "Fill in a customer's profile and score "
        "it with any of the three trained models."
    )

    model_choice = st.selectbox(
        "Model to use",
        list(models.keys()),
        index=1,
    )

    model = models[
        model_choice
    ]

    raw_cat_cols = [

        c
        for c in clean_df.columns

        if (
            is_text_col(
                clean_df[c]
            )
            and c != "Churn"
        )
    ]

    raw_num_cols = [

        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    with st.form(
        "predict_form"
    ):

        cols = st.columns(3)

        user_input = {}

        for i, col in enumerate(
            raw_cat_cols
        ):

            options = sorted(
                clean_df[col]
                .dropna()
                .unique()
                .tolist()
            )

            with cols[
                i % 3
            ]:

                user_input[col] = (
                    st.selectbox(
                        col,
                        options,
                    )
                )

        for i, col in enumerate(
            raw_num_cols
        ):

            with cols[
                (
                    i
                    + len(raw_cat_cols)
                ) % 3
            ]:

                if col == "SeniorCitizen":

                    user_input[col] = (
                        st.selectbox(
                            col,
                            [0, 1],
                        )
                    )

                else:

                    lo = float(
                        clean_df[
                            col
                        ].min()
                    )

                    hi = float(
                        clean_df[
                            col
                        ].max()
                    )

                    mean = float(
                        clean_df[
                            col
                        ].mean()
                    )

                    user_input[col] = (
                        st.slider(
                            col,
                            min_value=lo,
                            max_value=hi,
                            value=mean,
                        )
                    )

        submitted = (
            st.form_submit_button(
                "Predict"
            )
        )

    if submitted:

        input_df = pd.DataFrame(
            [user_input]
        )

        input_df["gender"] = (
            input_df[
                "gender"
            ].map({
                "Female": 1,
                "Male": 0,
            })
        )

        for c in [
            "Partner",
            "Dependents",
            "PhoneService",
            "PaperlessBilling",
        ]:

            input_df[c] = (
                input_df[c].map({
                    "Yes": 1,
                    "No": 0,
                })
            )

        input_encoded = (
            pd.get_dummies(
                input_df,
                columns=MULTI_CLASS_COLS,
                drop_first=True,
                dtype=int,
            )
        )

        input_encoded = (
            input_encoded.reindex(
                columns=X_train.columns,
                fill_value=0,
            )
        )

        input_encoded[
            CONTINUOUS_COLS
        ] = scaler.transform(
            input_encoded[
                CONTINUOUS_COLS
            ]
        )

        pred = model.predict(
            input_encoded
        )[0]

        proba = (
            model.predict_proba(
                input_encoded
            )[0, 1]
        )

        c1, c2 = st.columns(2)

        with c1:

            if pred == 1:

                st.error(
                    "⚠️ Predicted: "
                    "**Will Churn**"
                )

            else:

                st.success(
                    "✅ Predicted: "
                    "**Will Stay**"
                )

        with c2:

            st.metric(
                "Churn Probability",
                f"{proba * 100:.1f}%",
            )

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={
                    "text":
                        "Churn Risk"
                },
                gauge={

                    "axis": {
                        "range": [
                            0,
                            100,
                        ]
                    },

                    "bar": {
                        "color": (
                            "#DD8452"
                            if proba > 0.5
                            else "#55A868"
                        )
                    },

                    "steps": [

                        {
                            "range": [
                                0,
                                33,
                            ],
                            "color":
                                "#1f4023",
                        },

                        {
                            "range": [
                                33,
                                66,
                            ],
                            "color":
                                "#4a3a1f",
                        },

                        {
                            "range": [
                                66,
                                100,
                            ],
                            "color":
                                "#4a1f1f",
                        },
                    ],
                },
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.subheader(
            "Customer Profile Used for Prediction"
        )

        profile_df = pd.DataFrame({

            "Feature":
                list(
                    user_input.keys()
                ),

            "Value":
                list(
                    user_input.values()
                ),
        })

        st.dataframe(
            profile_df,
            use_container_width=True,
            hide_index=True,
        )


# ==========================================================================
# ABOUT
# ==========================================================================

with tab_about:

    st.subheader("ℹ️ About the Project")
    st.caption("Telcom Customer Churn Prediction & Analysis System")

    # 1. High-Level Summary Cards
    st.markdown("### 📌 Executive Overview")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Project Name", "Telcom Customer Churn")
    m2.metric("Dataset Size", "7,043 Records")
    m3.metric("Total Features", "21 Attributes")

    st.divider()

    # 2. Project Scope & Purpose
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### 🎯 Core Purpose & Business Goal")
        st.info(
            "**Primary Goal:** Analyse customer churn patterns and identify customers who are highly likely to leave.\n\n"
            "**Business Value:** Helps the telecommunications company identify high-risk customers early and take proactive retention actions to minimize revenue loss."
        )

    with c2:
        st.markdown("#### 💡 Key Business Finding")
        st.warning(
            "**Primary Churn Driver:**\n\n"
            "Customer churn is particularly associated with **shorter tenure** and **month-to-month contracts**. "
            "Focusing retention offers on new users and incentivizing long-term contracts will yield the highest ROI."
        )

    st.divider()

    # 3. Best Model Performance Card
    st.markdown("### 🏆 Best Model Performance (Random Forest)")
    
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Training Acc.", "84.35%")
    k2.metric("Test Acc.", "80.55%")
    k3.metric("Precision", "67.12%")
    k4.metric("Recall", "52.41%")
    k5.metric("F1-score", "58.86%")
    k6.metric("AUC", "84.38%")

    st.divider()

  # 4. Pipeline & Technical Details
    st.markdown("### 🛠️ Technical Pipeline & Methodology")

    col_top_left, col_top_right = st.columns([1, 1])

    with col_top_left:
        st.markdown("#### Data Preparation")
        st.markdown(
            """
            * **Data Cleaning:** Handling raw dataset noise
            * **Missing Values:** Imputing missing `TotalCharges`
            * **Encoding:** Mapping binary & one-hot encoding
            * **Outlier Detection:** IQR fence inspection
            * **Partitioning:** Stratified 80/20 train/test split
            * **Scaling:** `StandardScaler` for continuous features
            """
        )

    with col_top_right:
        st.markdown("#### Evaluation Metrics")
        st.markdown(
            """
            * **Accuracy:** Overall correct predictions ratio
            * **Precision:** True churners out of predicted churns
            * **Recall:** True churners successfully captured
            * **F1-score:** Balance between Precision & Recall
            * **AUC:** Multi-threshold separation power
            """
        )

    st.divider()

    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #f0f7ff 0%, #e6f0fa 100%);
            border: 1px solid #b3d4fc;
            border-left: 5px solid #1e88e5;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            margin-top: 10px;
        ">
            <h4 style="color: #0d47a1; margin-top: 0; margin-bottom: 12px;">🤖 Machine Learning Models</h4>
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 15px;">
                <div style="flex: 1; min-width: 200px;">
                    <strong style="color: #1565c0;">1. Logistic Regression</strong><br>
                    <span style="font-size: 0.88em; color: #5c6bc0;">Linear baseline model</span>
                </div>
                <div style="flex: 1; min-width: 200px;">
                    <strong style="color: #2e7d32;">2. Random Forest ⭐</strong><br>
                    <span style="font-size: 0.88em; color: #2e7d32; font-weight: 500;">Best performing ensemble model</span>
                </div>
                <div style="flex: 1; min-width: 200px;">
                    <strong style="color: #1565c0;">3. Support Vector Machine</strong><br>
                    <span style="font-size: 0.88em; color: #5c6bc0;">Non-linear kernel classification</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
