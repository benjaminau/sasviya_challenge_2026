# sasviya_challenge_2026
SAS Viya Challenge 2026 - Case Study on Hospital Length of Stay

## Overview
This project addresses the **SAS Viya for Learners Challenge 2026** (Workbench track), a Kaggle competition hosted by SAS. The task is to predict `ADMIT_LOS` — hospital length of stay in days — for inpatient encounters involving cardiac and respiratory conditions, using clinical, demographic, and operational data. Accurate LOS prediction supports hospitals in care planning and bed capacity management. Submissions are scored on RMSE against the true length of stay, with MAE reported for reference. Link: https://www.kaggle.com/competitions/sasviya-for-learners-challenge-2026/

## Methodology

**Data cleaning (SAS)**: Raw train/test data (100,000 / 15,000 encounters) was cleaned using traditional SAS `DATA`/`PROC` steps against the `WORK` library. This included resolving a corrupted `DEPARTMENT` category (`'Hosp 39'`, ~2.6% of rows, recoded to `Unknown`), stripping embedded hospital-ID text from several description fields, correcting inconsistent `STATECODE` values via ZIP-level majority voting, and recoding missing procedure codes (`'.'`) as an explicit "no procedure performed" category rather than imputing — confirmed via 100% alignment between missing procedure descriptions and placeholder procedure codes.

**Feature engineering (Python)**: Building on the cleaned data, features were engineered in Python, including:
- K-fold target encoding for high-cardinality categorical fields (diagnosis codes, procedure codes, DRG codes, hospital)
- Geospatial features (KMeans regional clustering) from patient coordinates
- Domain-informed interaction terms (e.g. ICU days × severity, comorbidity × age) capturing compounding clinical effects
- One-hot encoding for low-cardinality categoricals (department, gender, discharge destination, etc.)

Feature selection used correlation analysis and variance inflation factor (VIF) to identify and remove redundant features (e.g. an engineered feature that was an exact linear combination of two existing columns, and overlapping diagnosis-hierarchy target encodings), reducing the feature set from over 80 columns to the final set used in modeling.

**Modeling (Python)**: Five candidate models were trained and compared on an 80/20 train-validation split: Ridge regression, Lasso regression, Random Forest, XGBoost (log-transformed target), and XGBoost with a Poisson objective (raw count target). `ADMIT_LOS` was log-transformed (`log1p`) for the linear/default tree models to address strong right-skew (skewness ≈ 3.26), while the Poisson variant was trained directly on the raw count target, which better suits the distribution's shape. Error analysis by length-of-stay bucket showed all models underperformed disproportionately on long stays (15+ days); the Poisson objective measurably reduced this gap compared to the log-transform approach, motivating its use as the final model family. All predictions were converted back to raw-day scale before evaluation, since the competition scores on actual days. The best-performing model (XGBoost, Poisson objective, hyperparameter-tuned via `RandomizedSearchCV`) was retrained on the full training set before generating final predictions on the held-out test set.

## Notebooks
Run in the following order to reproduce the full pipeline:

1. **`notebooks/hospital_los_eda_cleaning_wrangling.ipynb`** (EDA section) — initial exploration: dataset structure, missingness, distinct-value checks, data quality scans, train/test consistency checks.
2. **SAS cleaning script** (`hospital_los_cleaning_wrangling.sasnb`, run via SAS Viya Workbench) — applies the cleaning steps identified in EDA (department/description corruption fixes, ZIP/state correction, procedure code recoding), exports cleaned data to CSV.
3. **`notebooks/hospital_los_eda_cleaning_wrangling.ipynb`** (feature engineering section, Python) — loads the SAS-cleaned CSVs, applies target encoding, geospatial features, interaction terms, one-hot encoding, and feature selection (correlation/VIF trimming).
4. **`notebooks/hospital_los_modeling.ipynb`** — train/validation split, model comparison (Ridge, Lasso, Random Forest, XGBoost variants, Poisson), hyperparameter tuning, diagnostics, final model selection, and `submission.csv` generation.