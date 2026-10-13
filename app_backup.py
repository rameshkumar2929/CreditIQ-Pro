from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
)

from creditiq.data import load_uci_credit_data, prepare_dataset
from creditiq.pipeline import (
    train_and_compare,
    predict_default_probability,
    assess_borrower,
    classify_risk_band,
)
from creditiq.ecl import calculate_ecl


# ============================================================
# CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="CreditIQ Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL DARK THEME
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --ink: #e8eef9;
        --muted: #9eacc5;
        --panel: #111c30;
        --line: #273650;
    }

    .stApp {
        background: radial-gradient(
            ellipse at top left,
            #172b49 0%,
            #0a1220 48%,
            #08101b 100%
        );
        color: var(--ink);
    }

    [data-testid="stSidebar"] {
        background: #0d1728;
        border-right: 1px solid #273650;
    }

    h1, h2, h3 {
        letter-spacing: -0.025em;
    }

    .hero {
        padding: 1.4rem 1.6rem;
        border: 1px solid #2c4264;
        border-radius: 20px;
        background: linear-gradient(
            120deg,
            rgba(35,68,112,.7),
            rgba(14,26,44,.9)
        );
        margin-bottom: 1rem;
    }

    .hero p {
        color: #b9c8df;
        margin-bottom: 0;
    }

    .small-note {
        color: #9eacc5;
        font-size: .88rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(20,34,55,.78);
        border: 1px solid #273650;
        padding: 14px 16px;
        border-radius: 14px;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
        <h1>CreditIQ Pro</h1>
        <p>
            Credit risk modeling • Probability of default •
            Explainability • IFRS 9-style ECL
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Portfolio-grade educational prototype. "
    "Not a validated lending or regulatory decision system."
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("## CreditIQ Pro")

    page = st.radio(
        "Workspace",
        [
            "Overview",
            "Data & Quality",
            "Model Lab",
            "Borrower Risk",
            "ECL Scenario Lab",
            "Model Card",
        ],
    )

    st.divider()

    st.markdown("**Model governance**")
    st.caption(
        "Holdout evaluation · Pipeline-based preprocessing · "
        "Reproducible random seed"
    )
    st.caption(
        "Data is processed in this app session. "
        "Do not upload personal or confidential customer data."
    )


# ============================================================
# SESSION STATE
# ============================================================

if "dataset" not in st.session_state:
    st.session_state.dataset = None

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = "Not loaded"

if "training_result" not in st.session_state:
    st.session_state.training_result = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_dataset(df: pd.DataFrame, name: str):
    """Prepare and store a dataset in the current Streamlit session."""
    cleaned, target = prepare_dataset(df)

    st.session_state.dataset = cleaned
    st.session_state.target = target
    st.session_state.dataset_name = name
    st.session_state.training_result = None


def get_data():
    """Return the currently loaded dataset."""
    return st.session_state.dataset


def fmt_pct(value):
    """Format a decimal as a percentage."""
    return f"{100 * value:.1f}%" if pd.notna(value) else "—"


# ============================================================
# PAGE 1: OVERVIEW
# ============================================================

if page == "Overview":

    data = get_data()

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Dataset",
        "Loaded" if data is not None else "Not loaded",
    )

    m2.metric(
        "Rows",
        f"{len(data):,}" if data is not None else "—",
    )

    m3.metric(
        "Input features",
        f"{data.shape[1] - 1:,}" if data is not None else "—",
    )

    m4.metric(
        "Model status",
        "Trained"
        if st.session_state.training_result
        else "Awaiting training",
    )

    st.markdown("### Start here")

    st.markdown(
        """
        1. Load the public UCI dataset or upload a CSV with a binary target.
        2. Review missing values, class balance, and data quality.
        3. Train and compare models in **Model Lab**.
        4. Use **Borrower Risk** for an illustrative probability estimate.
        5. Explore discounted, scenario-weighted ECL assumptions.
        """
    )

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("#### Load data")

        if st.button(
            "Load UCI Default of Credit Card Clients dataset",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Downloading public UCI dataset..."):
                try:
                    load_dataset(
                        load_uci_credit_data(),
                        "UCI Default of Credit Card Clients",
                    )

                    st.success(
                        "Dataset loaded. Review it under Data & Quality."
                    )
                    st.rerun()

                except Exception as exc:
                    st.error(f"Could not load the UCI dataset: {exc}")

        uploaded = st.file_uploader(
            "Or upload CSV",
            type=["csv"],
            key="overview_csv",
        )

        if uploaded is not None and st.button(
            "Use uploaded CSV",
            use_container_width=True,
        ):
            try:
                load_dataset(
                    pd.read_csv(uploaded),
                    uploaded.name,
                )

                st.success("CSV loaded.")
                st.rerun()

            except Exception as exc:
                st.error(f"Could not prepare CSV: {exc}")

    with col2:
        st.markdown("#### Current dataset")

        st.write(f"**Source:** {st.session_state.dataset_name}")

        if data is not None:
            target = st.session_state.target

            st.write(f"**Target:** `{target}`")

            st.write(
                "**Positive class rate:** "
                f"{fmt_pct(pd.to_numeric(data[target], errors='coerce').mean())}"
            )

        else:
            st.info(
                "Load a dataset to begin. "
                "The UCI download requires internet access."
            )


# ============================================================
# PAGE 2: DATA & QUALITY
# ============================================================

elif page == "Data & Quality":

    data = get_data()

    st.header("Data & Quality")

    if data is None:
        st.info(
            "Load the UCI dataset or upload a CSV from Overview first."
        )

    else:
        target = st.session_state.target

        a, b, c, d = st.columns(4)

        a.metric("Rows", f"{len(data):,}")
        b.metric("Features", f"{data.shape[1] - 1:,}")
        c.metric(
            "Missing cells",
            f"{int(data.isna().sum().sum()):,}",
        )
        d.metric("Default rate", fmt_pct(data[target].mean()))

        st.markdown("#### Column profile")

        profile = pd.DataFrame(
            {
                "dtype": data.dtypes.astype(str),
                "missing": data.isna().sum(),
                "missing_%": (data.isna().mean() * 100).round(2),
                "unique_values": data.nunique(dropna=True),
            }
        ).sort_values("missing", ascending=False)

        st.dataframe(profile, use_container_width=True)

        st.markdown("#### Target balance")

        counts = (
            data[target]
            .value_counts()
            .rename_axis("class")
            .reset_index(name="count")
        )

        st.plotly_chart(
            px.bar(
                counts,
                x="class",
                y="count",
                title="Target class counts",
                template="plotly_dark",
            ),
            use_container_width=True,
        )

        st.markdown("#### Data preview")

        st.dataframe(data.head(30), use_container_width=True)

        st.download_button(
            "Download prepared dataset",
            data.to_csv(index=False).encode("utf-8"),
            "creditiq_prepared_dataset.csv",
            "text/csv",
        )


# ============================================================
# PAGE 3: MODEL LAB
# ============================================================

elif page == "Model Lab":

    data = get_data()

    st.header("Model Lab")

    if data is None:
        st.info("Load a dataset first.")

    else:
        target = st.session_state.target

        st.write(
            f"Target column: `{target}`. "
            "All imputers, encoders, and scalers are fitted inside "
            "each training fold to reduce leakage."
        )

        with st.form("train_form"):

            candidates = st.multiselect(
                "Models to compare",
                [
                    "Logistic Regression",
                    "Random Forest",
                    "HistGradientBoosting",
                ],
                default=[
                    "Logistic Regression",
                    "Random Forest",
                    "HistGradientBoosting",
                ],
            )

            test_size = st.slider(
                "Holdout test fraction",
                min_value=0.15,
                max_value=0.35,
                value=0.20,
                step=0.05,
            )

            cv_folds = st.select_slider(
                "Cross-validation folds",
                options=[3, 4, 5],
                value=5,
            )

            seed = st.number_input(
                "Random seed",
                min_value=1,
                max_value=99999,
                value=42,
                step=1,
            )

            threshold = st.slider(
                "Decision threshold for confusion matrix",
                0.05,
                0.95,
                0.50,
                0.05,
            )

            submitted = st.form_submit_button(
                "Train and compare models",
                type="primary",
                use_container_width=True,
            )

        if submitted:

            if not candidates:
                st.error("Select at least one model.")

            else:
                with st.spinner(
                    "Training models and cross-validating. "
                    "This may take a few minutes..."
                ):
                    try:
                        result = train_and_compare(
                            data,
                            target,
                            candidates,
                            test_size=float(test_size),
                            cv_folds=int(cv_folds),
                            random_state=int(seed),
                            threshold=float(threshold),
                        )

                        st.session_state.training_result = result

                        st.success(
                            "Training completed. The winner is selected "
                            "by cross-validated average precision (PR-AUC), "
                            "not holdout performance."
                        )

                    except Exception as exc:
                        st.exception(exc)

        result = st.session_state.training_result

        if result:

            summary = result["summary"].copy()

            st.markdown("### Cross-validation comparison")

            st.dataframe(
                summary.style.format(
                    {
                        col: "{:.4f}"
                        for col in summary.select_dtypes(
                            include="number"
                        ).columns
                    }
                ),
                use_container_width=True,
            )

            winner = result["winner_name"]

            st.success(
                f"Selected candidate: {winner} · "
                "selection metric: mean CV PR-AUC"
            )

            metrics = result["holdout_metrics"]

            cols = st.columns(5)

            metric_labels = [
                "ROC-AUC",
                "PR-AUC",
                "Brier score",
                "Recall",
                "F1",
            ]

            metric_keys = [
                "roc_auc",
                "pr_auc",
                "brier",
                "recall",
                "f1",
            ]

            for col, label, key in zip(
                cols,
                metric_labels,
                metric_keys,
            ):
                col.metric(
                    label,
                    f"{metrics.get(key, float('nan')):.4f}",
                )

            st.caption(
                "These are holdout estimates from this split, not proof "
                "of future performance. Brier score is better when lower."
            )

            # Confusion matrix
            cm = confusion_matrix(
                result["y_test"],
                result["holdout_pred"],
                labels=[0, 1],
            )

            st.plotly_chart(
                px.imshow(
                    cm,
                    text_auto=True,
                    x=[
                        "Predicted non-default",
                        "Predicted default",
                    ],
                    y=[
                        "Actual non-default",
                        "Actual default",
                    ],
                    color_continuous_scale="Blues",
                    title="Holdout confusion matrix",
                    template="plotly_dark",
                ),
                use_container_width=True,
            )

            # ROC curve
            fpr, tpr, _ = roc_curve(
                result["y_test"],
                result["holdout_prob"],
            )

            st.plotly_chart(
                px.area(
                    x=fpr,
                    y=tpr,
                    title="Holdout ROC curve",
                    labels={
                        "x": "False positive rate",
                        "y": "True positive rate",
                    },
                    template="plotly_dark",
                ),
                use_container_width=True,
            )

            # Probability calibration
            st.markdown("#### Probability calibration")

            calibration = result.get("calibration_data")

            if calibration is not None and not calibration.empty:

                cols_lower = {
                    str(col).lower(): col
                    for col in calibration.columns
                }

                predicted_col = next(
                    (
                        cols_lower[key]
                        for key in cols_lower
                        if "predicted" in key
                        and (
                            "prob" in key
                            or "mean" in key
                        )
                    ),
                    None,
                )

                observed_col = next(
                    (
                        cols_lower[key]
                        for key in cols_lower
                        if (
                            "fraction" in key
                            and "positive" in key
                        )
                        or "observed" in key
                        or "actual_rate" in key
                    ),
                    None,
                )

                if predicted_col is None or observed_col is None:
                    numeric_cols = calibration.select_dtypes(
                        include="number"
                    ).columns.tolist()

                    if len(numeric_cols) >= 2:
                        predicted_col = numeric_cols[0]
                        observed_col = numeric_cols[1]

                if predicted_col is not None and observed_col is not None:

                    chart_data = calibration[
                        [predicted_col, observed_col]
                    ].rename(
                        columns={
                            predicted_col: "Mean predicted probability",
                            observed_col: "Observed default rate",
                        }
                    )

                    fig_cal = px.line(
                        chart_data,
                        x="Mean predicted probability",
                        y="Observed default rate",
                        markers=True,
                        title="Calibration curve",
                        template="plotly_dark",
                    )

                    fig_cal.add_shape(
                        type="line",
                        x0=0,
                        y0=0,
                        x1=1,
                        y1=1,
                        line=dict(dash="dash"),
                    )

                    st.plotly_chart(
                        fig_cal,
                        use_container_width=True,
                    )

                    st.caption(
                        "Closer to the dashed diagonal generally indicates "
                        "better calibration. Calibration estimates can be "
                        "noisy, especially in small samples."
                    )

                else:
                    st.dataframe(
                        calibration,
                        use_container_width=True,
                    )

            else:
                st.info(
                    "Calibration data is not available for this run."
                )

            # Classification report
            st.markdown("#### Classification report")

            report = classification_report(
                result["y_test"],
                result["holdout_pred"],
                labels=[0, 1],
                target_names=["Non-default", "Default"],
                output_dict=True,
                zero_division=0,
            )

            st.dataframe(
                pd.DataFrame(report).T,
                use_container_width=True,
            )

            # Feature importance
            st.markdown("#### Permutation importance")

            importance = result.get("importance")

            if importance is not None and not importance.empty:

                st.plotly_chart(
                    px.bar(
                        importance.head(20).sort_values(
                            "importance_mean"
                        ),
                        x="importance_mean",
                        y="feature",
                        error_x="importance_std",
                        orientation="h",
                        title=(
                            "Holdout permutation importance "
                            "(ROC-AUC scoring)"
                        ),
                        template="plotly_dark",
                    ),
                    use_container_width=True,
                )

            # Download trained model
            bundle = {
                "model": result["winner_model"],
                "features": result["features"],
                "target": result["target"],
                "model_name": winner,
                "threshold": result["threshold"],
                "metrics": result["holdout_metrics"],
                "summary": result["summary"].to_dict(
                    orient="records"
                ),
                "random_state": result["random_state"],
            }

            import joblib

            model_bytes = io.BytesIO()

            joblib.dump(bundle, model_bytes)

            st.download_button(
                "Download trained model bundle (.joblib)",
                model_bytes.getvalue(),
                "creditiq_model_bundle.joblib",
                "application/octet-stream",
            )

            st.download_button(
                "Download CV comparison (.csv)",
                summary.to_csv(index=False).encode(),
                "creditiq_model_comparison.csv",
                "text/csv",
            )


# ============================================================
# PAGE 4: BORROWER RISK ANALYZER
# ============================================================

elif page == "Borrower Risk":

    data = get_data()
    result = st.session_state.training_result

    st.header("Borrower Risk Analyzer")

    if data is None:
        st.info("Load a dataset first.")

    elif not result:
        st.info("Train at least one model in Model Lab first.")

    else:
        st.caption(
            "This is an illustrative model score. "
            "Do not use it as the sole basis for credit decisions."
        )

        features = result["features"]
        input_values = {}

        numeric_cols = [
            col
            for col in features
            if pd.api.types.is_numeric_dtype(data[col])
        ]

        categorical_cols = [
            col
            for col in features
            if col not in numeric_cols
        ]

        st.markdown("#### Enter feature values")

        with st.form("borrower_form"):

            left, right = st.columns(2)

            for i, col in enumerate(numeric_cols):

                series = pd.to_numeric(
                    data[col],
                    errors="coerce",
                ).dropna()

                if series.empty:
                    default = 0.0
                    lo, hi = -1e6, 1e6

                else:
                    default = float(series.median())
                    lo = float(series.min())
                    hi = float(series.max())

                    if lo == hi:
                        lo -= 1.0
                        hi += 1.0

                with (left if i % 2 == 0 else right):

                    input_values[col] = st.number_input(
                        col,
                        min_value=float(lo),
                        max_value=float(hi),
                        value=float(np.clip(default, lo, hi)),
                        key=f"risk_{col}",
                    )

            for col in categorical_cols:

                values = (
                    data[col]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if values:
                    input_values[col] = st.selectbox(
                        col,
                        sorted(values),
                        key=f"cat_{col}",
                    )

                else:
                    input_values[col] = ""

            st.markdown("#### Illustrative risk-band cutoffs")

            low_risk_max = st.slider(
                "Low Risk maximum PD",
                min_value=0.01,
                max_value=0.30,
                value=0.05,
                step=0.01,
                format="%.2f",
            )

            medium_risk_max = st.slider(
                "Medium Risk maximum PD",
                min_value=0.02,
                max_value=0.60,
                value=0.15,
                step=0.01,
                format="%.2f",
            )

            submitted = st.form_submit_button(
                "Estimate default probability",
                type="primary",
                use_container_width=True,
            )

        if submitted:

            row = pd.DataFrame(
                [
                    {
                        col: input_values.get(col, np.nan)
                        for col in features
                    }
                ]
            )

            try:

                if low_risk_max >= medium_risk_max:

                    st.error(
                        "The Medium Risk cutoff must be greater "
                        "than the Low Risk cutoff."
                    )

                else:

                    assessment = assess_borrower(
                        result["winner_model"],
                        row,
                        low_max=float(low_risk_max),
                        medium_max=float(medium_risk_max),
                        importance=result.get("importance"),
                    )

                    pd_default = float(
                        assessment["pd_probability"]
                    )

                    threshold = float(result["threshold"])

                    a, b, c = st.columns(3)

                    a.metric(
                        "Estimated probability of default",
                        f"{pd_default:.1%}",
                    )

                    b.metric(
                        "Risk band",
                        assessment["risk_band"],
                    )

                    c.metric(
                        "Model decision threshold",
                        f"{threshold:.2f}",
                    )

                    st.progress(
                        min(max(pd_default, 0.0), 1.0)
                    )

                    st.write(
                        "At the model's decision threshold, "
                        "classification: **"
                        + (
                            "