from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    accuracy_score,
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
# APPLICATION CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / "artifacts"

st.set_page_config(
    page_title="CreditIQ Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GOOGLE AUTHENTICATION AND AUTHORIZATION
# Requires Streamlit authentication configuration in secrets.toml
# ============================================================

# Never allow dashboard access without an authenticated session.
try:
    is_logged_in = st.user.is_logged_in
except Exception:
    is_logged_in = False

if not is_logged_in:
    st.title("CreditIQ Pro")
    st.subheader("Secure sign in")
    st.write(
        "Sign in with your approved Google account to access "
        "CreditIQ Pro."
    )

    st.info("Use the Google account authorized by your administrator.")

    if st.button(
        "Sign in with Google",
        type="primary",
        use_container_width=True,
    ):
        st.login("google")

    st.stop()


# Get the email from the authenticated identity.
user_email = str(st.user.get("email", "")).strip().lower()

if not user_email:
    st.error(
        "Google authentication did not provide an email address. "
        "Sign out and try again."
    )

    if st.button("Sign out"):
        st.logout()

    st.stop()


# Only emails explicitly listed in approved_users may access the app.
try:
    approved_users = dict(st.secrets["approved_users"])
except Exception:
    st.error(
        "The approved_users configuration is missing or invalid. "
        "Configure it in Streamlit secrets before continuing."
    )

    if st.button("Sign out"):
        st.logout()

    st.stop()


bank_id = approved_users.get(user_email)

if not bank_id:
    st.error(
        "Access denied. Your Google account is not approved "
        "for CreditIQ Pro."
    )
    st.caption(
        "Contact the application administrator if you believe "
        "you should have access."
    )

    if st.button("Sign out"):
        st.logout()

    st.stop()


# Keep identity and tenant information scoped to this session.
st.session_state["authenticated_email"] = user_email
st.session_state["bank_id"] = str(bank_id)


# ============================================================
# GLOBAL DESIGN
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
        background:
            radial-gradient(
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


# ============================================================
# HEADER
# ============================================================

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

    st.caption(f"Signed in: {user_email}")

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

    if st.button("Sign out", use_container_width=True):
        st.logout()


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
    cleaned, target = prepare_dataset(df)

    st.session_state.dataset = cleaned
    st.session_state.target = target
    st.session_state.dataset_name = name
    st.session_state.training_result = None


def get_data():
    return st.session_state.dataset


def fmt_pct(x):
    return f"{100 * x:.1f}%" if pd.notna(x) else "—"


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
                    st.success("Dataset loaded.")
                    st.rerun()

                except Exception as exc:
                    st.error(f"Could not load the UCI dataset: {exc}")

        uploaded = st.file_uploader(
            "Or upload CSV",
            type=["csv"],
            key="overview_csv",
        )

        if uploaded is not None:
            if st.button(
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
# PAGE 2: DATA AND QUALITY
# ============================================================

elif page == "Data & Quality":

    data = get_data()

    st.header("Data & Quality")

    if data is None:
        st.info("Load the UCI dataset or upload a CSV from Overview.")

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
            f"Target column: `{target}`. All imputers, encoders, "
            "and scalers are fitted inside each training fold "
            "to reduce leakage."
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

            numeric_columns = summary.select_dtypes(
                include="number"
            ).columns

            st.dataframe(
                summary.style.format(
                    {column: "{:.4f}" for column in numeric_columns}
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

            for col, label, key in zip(
                cols,
                ["ROC-AUC", "PR-AUC", "Brier score", "Recall", "F1"],
                ["roc_auc", "pr_auc", "brier", "recall", "f1"],
            ):
                col.metric(
                    label,
                    f"{metrics.get(key, float('nan')):.4f}",
                )

            st.caption(
                "These are holdout estimates from this split, not proof "
                "of future performance. Brier score is better when lower."
            )

            cm = confusion_matrix(
                result["y_test"],
                result["holdout_pred"],
                labels=[0, 1],
            )

            st.plotly_chart(
                px.imshow(
                    cm,
                    text_auto=True,
                    x=["Predicted non-default", "Predicted default"],
                    y=["Actual non-default", "Actual default"],
                    color_continuous_scale="Blues",
                    title="Holdout confusion matrix",
                    template="plotly_dark",
                ),
                use_container_width=True,
            )

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

            st.markdown("#### Probability calibration")

            calibration = result.get("calibration_data")

            if calibration is not None and not calibration.empty:

                cols_lower = {
                    str(column).lower(): column
                    for column in calibration.columns
                }

                predicted_col = next(
                    (
                        cols_lower[key]
                        for key in cols_lower
                        if "predicted" in key
                        and ("prob" in key or "mean" in key)
                    ),
                    None,
                )

                observed_col = next(
                    (
                        cols_lower[key]
                        for key in cols_lower
                        if (
                            "fraction" in key and "positive" in key
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
                        predicted_col, observed_col = numeric_cols[:2]

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
                        "better calibration. Estimates can be noisy, "
                        "especially in small samples."
                    )

                else:
                    st.dataframe(
                        calibration,
                        use_container_width=True,
                    )

            else:
                st.info("Calibration data is not available for this run.")

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

            st.markdown("#### Permutation importance")

            importance = result["importance"]

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
# PAGE 4: BORROWER RISK
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
            column
            for column in features
            if pd.api.types.is_numeric_dtype(data[column])
        ]

        categorical_cols = [
            column
            for column in features
            if column not in numeric_cols
        ]

        st.markdown("#### Enter feature values")

        with st.form("borrower_form"):

            left, right = st.columns(2)

            for i, column in enumerate(numeric_cols):

                series = pd.to_numeric(
                    data[column],
                    errors="coerce",
                ).dropna()

                if series.empty:
                    default, lo, hi = 0.0, -1e6, 1e6

                else:
                    default = float(series.median())
                    lo = float(series.min())
                    hi = float(series.max())

                    if lo == hi:
                        lo, hi = lo - 1.0, hi + 1.0

                with (left if i % 2 == 0 else right):
                    input_values[column] = st.number_input(
                        column,
                        min_value=float(lo),
                        max_value=float(hi),
                        value=float(np.clip(default, lo, hi)),
                        key=f"risk_{column}",
                    )

            for column in categorical_cols:

                values = (
                    data[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if values:
                    input_values[column] = st.selectbox(
                        column,
                        sorted(values),
                        key=f"cat_{column}",
                    )
                else:
                    input_values[column] = ""

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
                        column: input_values.get(column, np.nan)
                        for column in features
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
                        "At the model's decision threshold, classification: "
                        f"**{'Default-risk class' if pd_default >= threshold else 'Non-default-risk class'}**."
                    )

                    st.caption(
                        f"Illustrative bands: Low Risk below "
                        f"{low_risk_max:.0%}; Medium Risk from "
                        f"{low_risk_max:.0%} to below "
                        f"{medium_risk_max:.0%}; High Risk at or above "
                        f"{medium_risk_max:.0%}."
                    )

                    st.warning(
                        "This is an educational model estimate, not a "
                        "lending decision. Validate probability calibration, "
                        "fairness, data quality, and policy thresholds "
                        "before any real-world use."
                    )

                    important = assessment.get(
                        "global_important_features",
                        [],
                    )

                    if important:
                        st.markdown(
                            "#### Global feature-importance context"
                        )
                        st.dataframe(
                            pd.DataFrame(important),
                            use_container_width=True,
                        )

                    st.caption(
                        assessment.get(
                            "explanation_note",
                            "Global feature importance does not establish "
                            "causation or explain an individual prediction.",
                        )
                    )

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")


# ============================================================
# PAGE 5: ECL SCENARIO LAB
# ============================================================

elif page == "ECL Scenario Lab":

    st.header("ECL Scenario Lab")

    st.caption(
        "Illustrative calculator only. A complete IFRS 9 estimate "
        "requires governed staging, contractual cash flows, "
        "forward-looking scenarios, appropriate discounting, "
        "and institution-specific policy."
    )

    col1, col2 = st.columns(2)

    with col1:

        stage = st.selectbox(
            "Illustrative stage",
            ["Stage 1", "Stage 2", "Stage 3"],
        )

        pd_base = st.number_input(
            "12-month PD / lifetime PD as applicable (%)",
            0.0,
            100.0,
            3.0,
            0.1,
        ) / 100

        lgd = st.number_input(
            "LGD (%)",
            0.0,
            100.0,
            45.0,
            0.5,
        ) / 100

        ead = st.number_input(
            "Exposure at default (EAD)",
            min_value=0.0,
            value=100000.0,
            step=5000.0,
        )

        eir = st.number_input(
            "Effective interest rate for discounting (%)",
            0.0,
            100.0,
            8.0,
            0.25,
        ) / 100

    with col2:

        st.markdown("#### Forward-looking scenarios")

        w_base = st.number_input(
            "Base scenario weight (%)",
            0.0,
            100.0,
            60.0,
            5.0,
        ) / 100

        w_up = st.number_input(
            "Upside scenario weight (%)",
            0.0,
            100.0,
            15.0,
            5.0,
        ) / 100

        w_down = st.number_input(
            "Downside scenario weight (%)",
            0.0,
            100.0,
            25.0,
            5.0,
        ) / 100

        mult_up = st.number_input(
            "Upside PD multiplier",
            0.0,
            5.0,
            0.8,
            0.1,
        )

        mult_down = st.number_input(
            "Downside PD multiplier",
            0.0,
            5.0,
            1.8,
            0.1,
        )

        horizon = st.number_input(
            "Illustrative discount horizon (years)",
            0.0,
            30.0,
            1.0,
            0.5,
        )

    weights_sum = w_base + w_up + w_down

    st.metric("Scenario weights sum", f"{weights_sum:.0%}")

    if st.button(
        "Calculate illustrative ECL",
        type="primary",
    ):

        if not np.isclose(weights_sum, 1.0, atol=1e-6):

            st.error("Scenario weights must sum to 100%.")

        else:

            scenarios = [
                {
                    "name": "Base",
                    "weight": w_base,
                    "pd_multiplier": 1.0,
                },
                {
                    "name": "Upside",
                    "weight": w_up,
                    "pd_multiplier": mult_up,
                },
                {
                    "name": "Downside",
                    "weight": w_down,
                    "pd_multiplier": mult_down,
                },
            ]

            output = calculate_ecl(
                pd_base,
                lgd,
                ead,
                eir,
                horizon,
                scenarios,
            )

            st.metric(
                "Scenario-weighted discounted ECL",
                f"₹{output['weighted_discounted_ecl']:,.2f}",
            )

            ecl_df = pd.DataFrame(output["scenarios"])

            st.dataframe(
                ecl_df,
                use_container_width=True,
            )

            st.write(
                f"Illustrative treatment selected: **{stage}**. "
                "Stage selection does not automatically determine "
                "the correct PD horizon in this demo; ensure the PD "
                "input matches your policy assumption."
            )

            st.download_button(
                "Download ECL scenario results",
                ecl_df.to_csv(index=False).encode(),
                "creditiq_ecl_scenarios.csv",
                "text/csv",
            )


# ============================================================
# PAGE 6: MODEL CARD
# ============================================================

elif page == "Model Card":

    st.header("Model Card & Governance")

    result = st.session_state.training_result

    st.markdown(
        """
        ### Intended use

        Educational demonstration of binary default-risk modeling
        and credit-risk concepts using public or properly authorized data.

        ### Not intended for

        Automated lending decisions, adverse action, regulatory reporting,
        or production IFRS 9 measurement without independent validation
        and governance.

        ### Data

        The bundled loader uses the UCI Default of Credit Card Clients
        dataset. It is historical and may not represent current borrowers,
        products, geographies, or macroeconomic conditions.

        ### Methodology safeguards

        - Preprocessing is encapsulated in scikit-learn pipelines and fitted
          within cross-validation folds.
        - A stratified holdout set is kept separate from model selection.
        - Candidate selection uses mean cross-validated PR-AUC.
        - Holdout metrics are reported separately and should not be
          repeatedly optimized against.
        - Thresholds are decision-policy parameters, not universal truths.
        - Missingness, representativeness, class imbalance, drift, and
          subgroup performance require review.

        ### Limitations

        This demo does not implement a complete IFRS 9 staging engine,
        significant increase in credit risk (SICR) rules, contractual
        cash-flow discounting, or approved economic forecasts.
        ECL outputs are illustrative.
        """
    )

    if result:

        st.markdown("#### Current session run")

        st.json(
            {
                "winner": result["winner_name"],
                "target": result["target"],
                "features": len(result["features"]),
                "random_state": result["random_state"],
                "threshold": result["threshold"],
                "calibration_method": result.get(
                    "calibration_method",
                    "Not recorded",
                ),
                "illustrative_default_risk_bands": {
                    "low_risk_max_pd": result.get(
                        "low_risk_max",
                        0.05,
                    ),
                    "medium_risk_max_pd": result.get(
                        "medium_risk_max",
                        0.15,
                    ),
                },
                "holdout_metrics": result["holdout_metrics"],
            }
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <p class="small-note">
        CreditIQ Pro · Educational portfolio project ·
        Validate models and assumptions before any real-world use.
    </p>
    """,
    unsafe_allow_html=True,
)