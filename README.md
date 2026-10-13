# CreditIQ Pro — Credit Risk ML & IFRS 9-style ECL

CreditIQ Pro is an educational portfolio project demonstrating a reproducible binary default-risk modeling workflow, model comparison, probability evaluation, permutation importance, and an illustrative scenario-weighted ECL calculator.

## Features
- UCI Default of Credit Card Clients dataset loader and CSV upload
- Target checks and basic data-quality profiling
- Leakage-aware preprocessing in scikit-learn pipelines
- Logistic Regression, Random Forest, and HistGradientBoosting candidates
- Stratified holdout test split and cross-validation on training data
- Candidate selection by mean cross-validated PR-AUC
- Holdout ROC-AUC, PR-AUC, Brier score, log loss, recall, precision, F1 and confusion matrix
- Holdout permutation importance
- Individual estimated default probability
- Scenario-weighted PD × LGD × EAD illustration with simple discounting
- Model card and limitations

## Run locally (Windows PowerShell)

```powershell
cd "C:\Users\1105\Desktop\Credit-Risk-Analytics\Credit-Risk-Analytics"
# Extract/copy the CreditIQ-Pro files into this folder, replacing app.py only after you back up the current version.
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed in the terminal. If port 8501 is occupied, Streamlit may choose another port.

## Tests

```powershell
python -m pytest -q
```

## Deploy
Commit `app.py`, `creditiq/`, `requirements.txt`, `README.md`, `.gitignore`, and `tests/` to GitHub. On Streamlit Community Cloud select the repository, `main` branch, and `app.py` as the main file.

## Important methodology notes
- All imputers, encoders and scalers are inside scikit-learn pipelines. During cross-validation they are fit separately inside each training fold.
- A stratified holdout is kept separate from model selection. Candidate selection uses mean cross-validated PR-AUC. Holdout metrics should be treated as an estimate and not repeatedly optimized against.
- The default threshold of 0.50 is only a starting point. A production decision threshold requires documented costs, calibration, and policy review.
- Brier score and log loss assess probability quality, but this implementation does **not** yet calibrate the models. Calibration should be evaluated and, if needed, fitted using training-only cross-validation.
- The UCI dataset is historical and does not establish performance on current portfolios.
- The ECL calculator is a simplified illustration, not a full IFRS 9 implementation. It does not automatically apply SICR rules, staging policy, contractual cash-flow schedules, multiple future periods, or institution-approved macroeconomic forecasts.
- Do not use these outputs as the sole basis for real lending decisions or regulatory reporting.
- Only use data you are authorized to process. This app is not designed for confidential customer data.

## Suggested next enhancements
1. Add calibration curves and calibration using cross-validated training predictions.
2. Add temporal validation and out-of-time performance where a valid time field is available.
3. Add subgroup performance/fairness review where legally and statistically appropriate.
4. Add drift monitoring and model registry/versioned artifacts.
5. Add typed tests for preprocessing, target validation, and model training.
