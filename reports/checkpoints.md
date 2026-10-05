# Project Execution Checkpoints

Running progress log tracking milestone delivery, commands executed, verification results, and key metrics.

## Milestone Status Overview

- [x] **M0: Scaffold**
- [x] **M1: Data audit and cohort**
- [x] **M2: Leakage audit and splits**
- [x] **M3: Features and baselines**
- [x] **M4: LR and XGBoost, imbalance ablation**
- [ ] **M5: Calibration and thresholds**
- [ ] **M6: Explainability (SHAP & odds ratios)**
- [ ] **M7: Fairness audit and mitigation**
- [ ] **M8: Final test evaluation (locked)**
- [ ] **M9: Engineering layer (API, Dashboard, Docker)**
- [ ] **M10: Write-up and executive report**

---

## Checkpoint 0: M0 Scaffold Complete
- **Status:** Complete.

---

## Checkpoint 1: M1 Data Audit & M2 Leakage Audit & Splits Complete
- **What was built:**
  - src/readmit/data.py: Automated download from UCI repository (ID 296), raw ingestion, cohort filtering logic, and stratified patient-grouped splitting.
  - src/readmit/audit.py: Automated markdown report generators for data quality (
eports/data_quality.md) and prediction-point clinical leakage audit (
eports/leakage_audit.md).
  - 
eports/cohort_flow.md: Full attrition flow tracking raw encounters down to primary cohort with exact counts.
  - data/splits/: Committed patient-ID JSON files (	rain_patients.json, al_patients.json, 	est_patients.json) guaranteeing 100% split determinism and zero patient overlap.
  - data/processed/: Processed parquet splits (cohort.parquet, 	rain.parquet, al.parquet, 	est.parquet).
  - Unit tests in 	ests/test_data.py: verified label mapping, hospice/expired exclusion, and zero patient overlap across splits.
- **Key measured numbers:**
  - Raw encounters: **101,766** across **71,518** unique patients (50 columns).
  - Excluded expired/hospice (disposition IDs 11, 13, 14, 19, 20, 21): **2,423** encounters.
  - Excluded invalid gender: **3** encounters.
  - Excluded repeat encounters (kept index/first encounter by lowest encounter_id): **29,353** encounters.
  - Final Primary Cohort: **69,987** patients/encounters.
  - 30-day readmissions (
eadmitted == '<30'): **6,285** cases.
  - Cohort prevalence: **8.9802%** (~8.98%).
  - Patient-grouped splits (70 / 15 / 15):
    - Train: **48,990** patients (prevalence **8.98%**, 4,399 positives)
    - Val: **10,497** patients (prevalence **8.97%**, 942 positives)
    - Test: **10,500** patients (prevalence **8.99%**, 944 positives)
    - Zero patient overlap across splits: **VERIFIED**.
- **Commands run & results:**
  - python -m readmit.cli data: Successfully generated cohort, audits, and splits.
  - 
uff check src tests: Passed (0 errors).
  - pytest -v: 5 passed in 0.83s.
- **Assumptions made:**
  - Discharge disposition IDs 11, 13, 14, 19, 20, 21 represent mortality/hospice per data/raw/IDS_mapping.csv.
  - Lowest encounter_id represents the patient index encounter (chronological proxy).
- **Open issues:** None.

---

## Checkpoint 2: M3 Features & Baselines Complete
- **What was built:**
  - `src/readmit/evaluation.py`: PR-AUC, ROC-AUC, Brier score, Expected Calibration Error (`compute_ece`), capacity metrics (`compute_capacity_metrics` at K=5, 10, 20), cost-ratio threshold optimization, and bootstrap confidence intervals (`bootstrap_metric_ci`).
  - `src/readmit/features.py`: ICD-9 clinical grouping (`group_icd9`), demographics midpoints, prior visits summation, medication dynamics, `ClinicalFeatureEngineer`, and `build_preprocessor_pipeline` (StandardScaler + OneHotEncoder).
  - Baselines: `PrevalenceBaseline` (uninformative constant prevalence) and `PriorInpatientRuleBaseline` (clinical heuristic on prior inpatient admissions).
  - Artifacts generated: `artifacts/feature_pipeline.joblib`, `reports/feature_dictionary.md`.
  - Unit tests in `tests/test_features.py`: verified ICD-9 clinical mapping, verified forbidden features exclusion (race, gender, identifiers, target), verified preprocessor fit strictly on train only, capacity threshold calculations on toy data, metric definitions, and bootstrap CI reproducibility.
- **Key measured numbers:**
  - Extracted feature dimension: **87 engineered features** (14 numeric, 5 binary, 68 one-hot encoded categories).
  - Forbidden features audit: 0 race, 0 gender, 0 target, 0 identifier columns in preprocessed feature matrix.
  - Baseline validation metrics:
    - **Prevalence Baseline:** PR-AUC = **0.0897**, ROC-AUC = **0.5000**, Brier = **0.0817**, ECE = **0.0000**.
    - **Prior Inpatient Rule Baseline:** PR-AUC = **0.1201**, ROC-AUC = **0.5452**, Brier = **0.1691**, ECE = **0.1118**.
    - **Rule Baseline Capacity (K=20%):** Flagged = 2,100, Captured Positives = 263, Recall = **27.92%**, Precision = **12.52%**, Lift = **1.40x**.
- **Commands run & results:**
  - `python -m readmit.cli features`: Feature pipeline fitted on train, transformed val, baselines evaluated.
  - `ruff check src tests`: Passed (0 errors).
  - `pytest -v`: All 11 tests passed in 1.93s.
- **Assumptions made:**
  - Unknown ICD-9 or missing values categorized as "Unknown" or "Other".
  - Age converted to midpoint for numeric modeling while preserving original age group for fairness audit.
- **Open issues:** None.

---

## Checkpoint 3: M4 Models & Imbalance Strategy Ablation Complete
- **What was built:**
  - `src/readmit/models.py`: Model training for L2 Logistic Regression and XGBoost (`tree_method='hist'`) with early stopping on an inner-train holdout split (Invariant 3 & 4: zero validation or test peeking).
  - Class imbalance ablation across 3 distinct paradigms on the training split:
    1. Natural clinical prevalence (unweighted).
    2. Cost-sensitive reweighting (`balanced` weights for LR; `scale_pos_weight = 10.14` for XGBoost).
    3. Resampling: SMOTE synthetic oversampling on training split only.
  - SQLite-backed MLflow experiment tracking (`sqlite:///mlflow.db`) logging run parameters, hyperparameters, and validation metrics without file store deprecation warnings.
  - Generated report: `reports/model_comparison.md`.
  - Artifacts: `artifacts/logistic_regression.joblib`, `artifacts/xgboost.joblib`.
- **Key measured numbers (Validation Set: N=10,497, Prevalence=8.97%):**
  - **Logistic Regression (Unweighted):** PR-AUC = **0.1647**, ROC-AUC = **0.6454**, Brier = **0.0794**, ECE = **0.0014**, Recall@K=20% = **36.94%**, Lift = **1.85x**.
  - **XGBoost (Unweighted - Best Primary):** PR-AUC = **0.1716**, ROC-AUC = **0.6489**, Brier = **0.0792**, ECE = **0.0014**, Recall@K=20% = **37.69%**, Lift = **1.88x**.
  - **LR (Class Weighted balanced):** PR-AUC = **0.1634**, ROC-AUC = **0.6459**, Brier = **0.2309**, ECE = **0.3800** (calibration severely distorted without post-hoc scaling).
  - **XGBoost (scale_pos_weight 10.14):** PR-AUC = **0.1645**, ROC-AUC = **0.6385**, Brier = **0.2141**, ECE = **0.3532**.
  - **LR (SMOTE):** PR-AUC = **0.1576**, ROC-AUC = **0.6345**, Brier = **0.2315**, ECE = **0.3756**.
  - **XGBoost (SMOTE):** PR-AUC = **0.1599**, ROC-AUC = **0.6365**, Brier = **0.0807**, ECE = **0.0295**.
  - **Leakage check:** Validation ROC-AUC (0.6489) is well below the 0.80 leakage red flag threshold; results align with published clinical benchmarks on this dataset.
- **Commands run & results:**
  - `python -m readmit.cli train`: Completed in ~10s. All 6 candidate runs trained and evaluated.
  - `ruff check src tests`: Passed (0 errors).
  - `pytest -v`: 11 passed.
- **Assumptions made:**
  - Unweighted XGBoost selected as primary base model for M5 calibration given superior PR-AUC and natural probability calibration.
- **Open issues:** None. Proceeding autonomously to M5 (Calibration & Thresholds).


