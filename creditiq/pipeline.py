
from __future__ import annotations

import time
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SUPPORTED_MODELS = (
    "Logistic Regression",
    "Random Forest",
    "HistGradientBoosting",
)


def _build_pipeline(
    X: pd.DataFrame,
    model_name: str,
    random_state: int,
) -> Pipeline:
    """Build preprocessing and classifier without preprocessing leakage."""

    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    transformers = []

    if numeric:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            )),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipeline, numeric))

    if categorical:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(
                strategy="most_frequent",
                keep_empty_features=True,
            )),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            )),
        ])
        transformers.append(("cat", categorical_pipeline, categorical))

    prep = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    if model_name == "Logistic Regression":
        estimator = LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        )

    elif model_name == "Random Forest":
        estimator = RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )

    elif model_name == "HistGradientBoosting":
        if categorical:
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(
                    strategy="most_frequent",
                    keep_empty_features=True,
                )),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                )),
            ])

            transformers = []

            if numeric:
                transformers.append((
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(
                            strategy="median",
                            keep_empty_features=True,
                        )),
                    ]),
                    numeric,
                ))

            transformers.append(("cat", cat_pipeline, categorical))

            prep = ColumnTransformer(
                transformers=transformers,
                remainder="drop",
                sparse_threshold=0,
            )

        estimator = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.08,
            l2_regularization=1.0,
            random_state=random_state,
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline([
        ("preprocess", prep),
        ("model", estimator),
    ])


def _safe_metric(function, y_true, predictions) -> float:
    """Return NaN instead of crashing when a metric is undefined."""
    try:
        return float(function(y_true, predictions))
    except (ValueError, TypeError, ZeroDivisionError):
        return float("nan")


def classify_risk_band(
    probability: float,
    low_max: float = 0.05,
    medium_max: float = 0.15,
) -> str:
    """
    Classify predicted PD using configurable cutoffs.

    Defaults are illustrative project settings, not bank-approved
    lending thresholds.
    """
    probability = float(probability)
    low_max = float(low_max)
    medium_max = float(medium_max)

    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")

    if not 0 <= low_max < medium_max <= 1:
        raise ValueError(
            "Thresholds must satisfy 0 <= low_max < medium_max <= 1."
        )

    if probability < low_max:
        return "Low Risk"
    if probability < medium_max:
        return "Medium Risk"
    return "High Risk"


def assess_borrower(
    model,
    borrower: pd.DataFrame,
    low_max: float = 0.05,
    medium_max: float = 0.15,
    importance: pd.DataFrame | None = None,
) -> dict:
    """
    Return a borrower PD estimate, configurable risk band, and
    clearly labelled global feature-importance context.
    """
    if len(borrower) != 1:
        raise ValueError("Provide exactly one borrower row.")

    probability = float(model.predict_proba(borrower)[:, 1][0])

    top_features = []
    if (
        importance is not None
        and not importance.empty
        and "feature" in importance.columns
        and "importance_mean" in importance.columns
    ):
        top_features = (
            importance.loc[
                importance["importance_mean"] > 0,
                ["feature", "importance_mean"],
            ]
            .head(5)
            .to_dict(orient="records")
        )

    return {
        "pd_probability": probability,
        "pd_percent": probability * 100,
        "risk_band": classify_risk_band(
            probability,
            low_max=low_max,
            medium_max=medium_max,
        ),
        "low_risk_cutoff": float(low_max),
        "medium_risk_cutoff": float(medium_max),
        "global_important_features": top_features,
        "explanation_note": (
            "Feature importance is a global model-level signal. "
            "It does not prove causation or establish why this "
            "individual borrower received this prediction."
        ),
    }


def train_and_compare(
    df: pd.DataFrame,
    target: str,
    model_names: list[str],
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
    threshold: float = 0.5,
    low_risk_max: float = 0.05,
    medium_risk_max: float = 0.15,
) -> dict:
    """Compare models, calibrate the winner, and evaluate on holdout."""

    started = time.perf_counter()

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Provide a non-empty pandas DataFrame.")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    if not 0.1 <= float(test_size) <= 0.5:
        raise ValueError("Test size must be between 0.10 and 0.50.")

    if not 0 < float(threshold) < 1:
        raise ValueError("Classification threshold must be between 0 and 1.")

    # Validate risk cutoffs before training.
    classify_risk_band(
        0.5,
        low_max=low_risk_max,
        medium_max=medium_risk_max,
    )

    if not model_names:
        raise ValueError("Select at least one model.")

    invalid_models = [m for m in model_names if m not in SUPPORTED_MODELS]
    if invalid_models:
        raise ValueError(f"Unsupported model selection: {invalid_models}")

    if len(set(model_names)) != len(model_names):
        raise ValueError("Duplicate model selections are not allowed.")

    X = df.drop(columns=[target]).copy()
    if X.shape[1] == 0:
        raise ValueError("At least one feature column is required.")

    if X.columns.duplicated().any():
        raise ValueError("Duplicate feature column names are not supported.")

    if X.empty:
        raise ValueError("No rows are available for training.")

    y_numeric = pd.to_numeric(df[target], errors="raise")

    if y_numeric.isna().any():
        raise ValueError("Target contains missing values.")

    if not set(y_numeric.unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0/1 values.")

    if set(y_numeric.unique()) != {0, 1}:
        raise ValueError("Target must contain both classes: 0 and 1.")

    y = y_numeric.astype(int)
    class_counts = y.value_counts()

    if int(class_counts.min()) < 2:
        raise ValueError(
            "At least two observations per class are needed for a "
            "stratified train/test split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y,
    )

    train_minority = int(y_train.value_counts().min())
    n_splits = min(int(cv_folds), train_minority)

    if n_splits < 2:
        raise ValueError(
            "Not enough minority-class training examples for cross-validation."
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=int(random_state),
    )

    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "balanced_accuracy": "balanced_accuracy",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
    }

    raw_models = {}
    rows = []

    for name in model_names:
        print(f"[CreditIQ] Cross-validating {name}...", flush=True)

        pipeline = _build_pipeline(X_train, name, int(random_state))

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )

        row = {"model": name}

        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"cv_{metric}_mean"] = float(np.mean(values))
            row[f"cv_{metric}_std"] = (
                float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0
            )

        row["cv_fit_seconds_mean"] = float(np.mean(scores["fit_time"]))

        print(f"[CreditIQ] Fitting {name} on training data...", flush=True)
        raw_models[name] = clone(pipeline).fit(X_train, y_train)
        rows.append(row)

    summary = (
        pd.DataFrame(rows)
        .sort_values(
            ["cv_pr_auc_mean", "cv_roc_auc_mean"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    winner_name = str(summary.iloc[0]["model"])
    raw_winner_model = raw_models[winner_name]

    # Calibration uses training data only; the final holdout is untouched.
    calibration_splits = min(3, int(y_train.value_counts().min()))

    if calibration_splits < 2:
        raise ValueError(
            "Not enough training observations per class for calibration."
        )

    print(
        f"[CreditIQ] Calibrating {winner_name} using training folds...",
        flush=True,
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=_build_pipeline(
            X_train,
            winner_name,
            int(random_state),
        ),
        method="sigmoid",
        cv=StratifiedKFold(
            n_splits=calibration_splits,
            shuffle=True,
            random_state=int(random_state),
        ),
    )
    calibrated_model.fit(X_train, y_train)

    winner_model = calibrated_model

    print(
        f"[CreditIQ] Evaluating calibrated winner on holdout: {winner_name}",
        flush=True,
    )

    probabilities = winner_model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= float(threshold)).astype(int)

    holdout_metrics = {
        "roc_auc": _safe_metric(roc_auc_score, y_test, probabilities),
        "pr_auc": _safe_metric(average_precision_score, y_test, probabilities),
        "brier": _safe_metric(brier_score_loss, y_test, probabilities),
        "log_loss": _safe_metric(
            log_loss,
            y_test,
            np.clip(probabilities, 1e-7, 1 - 1e-7),
        ),
        "accuracy": _safe_metric(accuracy_score, y_test, predictions),
        "balanced_accuracy": _safe_metric(
            balanced_accuracy_score, y_test, predictions
        ),
        "precision": _safe_metric(precision_score, y_test, predictions),
        "recall": _safe_metric(recall_score, y_test, predictions),
        "f1": _safe_metric(f1_score, y_test, predictions),
    }

    # Holdout calibration curve is diagnostic only.
    observed_rate, predicted_rate = calibration_curve(
        y_test,
        probabilities,
        n_bins=min(10, max(2, len(y_test) // 20)),
        strategy="quantile",
    )

    calibration_data = pd.DataFrame({
        "mean_predicted_probability": predicted_rate,
        "observed_default_rate": observed_rate,
    })

    importance = pd.DataFrame(
        columns=["feature", "importance_mean", "importance_std"]
    )

    try:
        print("[CreditIQ] Calculating permutation importance...", flush=True)

        permutation = permutation_importance(
            winner_model,
            X_test,
            y_test,
            scoring="roc_auc",
            n_repeats=3,
            random_state=int(random_state),
            n_jobs=1,
        )

        importance = pd.DataFrame({
            "feature": X_test.columns.astype(str),
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }).sort_values(
            "importance_mean",
            ascending=False,
        ).reset_index(drop=True)

    except Exception as error:
        print(
            f"[CreditIQ] Importance skipped: {error}",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    print(
        f"[CreditIQ] Training and evaluation finished in {elapsed:.1f} seconds.",
        flush=True,
    )

    return {
        "winner_name": winner_name,
        "winner_model": winner_model,
        "raw_winner_model": raw_winner_model,
        "models": raw_models,
        "summary": summary,
        "holdout_metrics": holdout_metrics,
        "y_test": y_test.reset_index(drop=True),
        "holdout_prob": np.asarray(probabilities),
        "holdout_pred": np.asarray(predictions),
        "calibration_data": calibration_data,
        "importance": importance,
        "features": list(X.columns),
        "target": target,
        "threshold": float(threshold),
        "low_risk_max": float(low_risk_max),
        "medium_risk_max": float(medium_risk_max),
        "random_state": int(random_state),
        "test_size": float(test_size),
        "cv_folds": int(n_splits),
        "calibration_method": "sigmoid",
        "training_seconds": float(elapsed),
    }


def predict_default_probability(
    model,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return probability estimates for the positive/default class."""
    return model.predict_proba(X)[:, 1]
