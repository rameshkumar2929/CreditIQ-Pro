from __future__ import annotations

import pandas as pd
from ucimlrepo import fetch_ucirepo

TARGET_CANDIDATES = (
    "default payment next month", "default.payment.next.month", "default",
    "target", "label", "y", "is_default", "defaulted",
)

def load_uci_credit_data() -> pd.DataFrame:
    """Fetch UCI Default of Credit Card Clients (dataset ID 350)."""
    dataset = fetch_ucirepo(id=350)
    x = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    if y.shape[1] == 0:
        raise ValueError("UCI dataset did not return a target column.")
    target = y.iloc[:, 0].rename("default")
    frame = pd.concat([x.reset_index(drop=True), target.reset_index(drop=True)], axis=1)
    frame.columns = [str(c).strip().replace("\n", " ") for c in frame.columns]
    frame = frame.loc[:, ~frame.columns.astype(str).str.match(r"^Unnamed")]
    return frame

def _find_target(columns) -> str | None:
    lowered = {str(c).strip().lower(): c for c in columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    # Fallback only when one clearly binary column is named target-like.
    for c in columns:
        name = str(c).lower()
        if any(token in name for token in ("default", "target", "label")):
            return c
    return None

def prepare_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Clean names and normalize a binary target to integer 0/1."""
    if frame is None or frame.empty:
        raise ValueError("The dataset is empty.")
    df = frame.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if len(set(df.columns)) != len(df.columns):
        raise ValueError("Column names must be unique after trimming whitespace.")
    target = _find_target(df.columns)
    if target is None:
        raise ValueError(
            "Could not identify a binary target column. Name it 'default', 'target', "
            "'label', or 'default payment next month' and upload again."
        )
    y = pd.to_numeric(df[target], errors="coerce")
    observed = sorted(y.dropna().unique().tolist())
    if len(observed) != 2:
        raise ValueError(f"Target column '{target}' must contain exactly two classes; found {observed[:10]}.")
    # Conventional positive-class handling: {0,1} stays as-is; other two labels map larger to 1.
    if set(observed) == {0, 1}:
        mapped = y
    else:
        mapped = y.map({observed[0]: 0, observed[1]: 1})
    df[target] = mapped
    df = df.loc[df[target].notna()].reset_index(drop=True)
    if df[target].nunique() != 2:
        raise ValueError("Both target classes must remain after removing missing target values.")
    if df[target].value_counts().min() < 5:
        raise ValueError("Each target class needs at least 5 examples for a meaningful train/test split.")
    return df, str(target)
