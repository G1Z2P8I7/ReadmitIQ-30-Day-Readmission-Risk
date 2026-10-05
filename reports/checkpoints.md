# Project Execution Checkpoints

Running progress log tracking milestone delivery, commands executed, verification results, and key metrics.

## Milestone Status Overview

- [x] **M0: Scaffold**
- [x] **M1: Data audit and cohort**
- [x] **M2: Leakage audit and splits**
- [x] **M3: Features and baselines**
- [x] **M4: LR and XGBoost, imbalance ablation**
- [x] **M5: Calibration and thresholds**
- [x] **M6: Explainability (SHAP & odds ratios)**
- [x] **M7: Fairness audit and mitigation**
- [x] **M8: Final test evaluation (locked)**
- [x] **M9: Engineering layer (API, Dashboard, Docker)**
- [x] **M10: Write-up and executive report**

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
- **Open issues:** None.

---

## Checkpoint 4: M5 Calibration and Thresholds Complete
- **What was built:**
  - `src/readmit/calibration.py`: Manual Platt scaling (`PlattCalibrator` via logistic regression on logits) and Isotonic regression (`IsotonicCalibrator` with clipped boundaries) fitted on validation set predictions.
  - End-to-end `FullCalibratedPipeline` exposing `predict_proba`, `predict`, and `__sklearn_is_fitted__` returning `True` (compatible with Fairlearn `ThresholdOptimizer`).
  - Figure generated: `reports/figures/calibration_curve_xgb.png` displaying reliability curves, 45-degree ideal line, and hospital baseline prevalence reference line.
  - Capacity & decision table generated: `reports/capacity_table.md`.
  - Artifact saved: `artifacts/xgboost_calibrated.joblib`.
  - Unit test in `tests/test_features.py`: `test_calibration_monotonic` asserts strictly non-decreasing output mapping.
- **Key measured numbers (Validation Set: N=10,497):**
  - **Calibration Comparison on Validation:**
    - Raw XGBoost: Brier = **0.0792**, ECE = **0.0014**, PR-AUC = **0.1716**.
    - Platt Scaling (Sigmoid): Brier = **0.0792**, ECE = **0.0037**, PR-AUC = **0.1716**.
    - Isotonic Regression: Brier = **0.0788**, ECE = **0.0000**, PR-AUC = **0.1708**.
    - **Selected Primary Calibrator:** `isotonic` (lowest Brier and near-zero ECE).
  - **Capacity-Constrained Decision Tiers (Primary K=20%):**
    - **Top 5%:** Flagged = 525 patients, Cutoff = 20.81%, Readmissions Captured = 132, Recall = **14.01%**, Precision = **25.14%**, Lift = **2.80x**.
    - **Top 10%:** Flagged = 1,050 patients, Cutoff = 15.38%, Readmissions Captured = 226, Recall = **23.99%**, Precision = **21.52%**, Lift = **2.40x**.
    - **Top 20% (Primary):** Flagged = 2,100 patients, Cutoff = 10.77%, Readmissions Captured = 362, Recall = **38.43%**, Precision = **17.24%**, Lift = **1.92x**.
  - **Cost-Ratio Optimization (illustrative 5:1 FN:FP cost ratio):**
    - Optimal probability cutoff threshold = **15.85%**, Precision = **22.86%**, Recall = **21.02%**, F1 = **0.2190**.
- **Commands run & results:**
  - `python -m readmit.cli evaluate`: Reliability plot generated, full pipeline saved, capacity table written.
  - `ruff check src tests`: Passed (0 errors).
  - `pytest -v`: 12 passed.
- **Assumptions made:**
  - Primary operational assumption is care team capacity fixed at top 20% of discharged patients.
  - Cost ratio of 5:1 (FN to FP) is an illustrative clinical scenario labeled as such in all reports.
- **Open issues:** None.

---

## Checkpoint 5: M6 Explainability Complete
- **What was built:**
  - `src/readmit/explain.py`: Odds ratios and 95% Wald CI extraction from Logistic Regression coefficients; `shap.TreeExplainer` on uncalibrated XGBoost base estimator; global beeswarm figure generation; SHAP attribution stability check across bootstrap resamples; and local waterfall attributions for 3 clinical case studies (True Positive, False Negative, Low Risk Baseline).
  - Figures generated:
    - `reports/figures/shap_summary_xgb.png` (global beeswarm plot of top 15 features).
    - `reports/figures/shap_waterfall_Case_A_TruePositive.png` (Patient probability 20.8%, actual readmit).
    - `reports/figures/shap_waterfall_Case_B_FalseNegative.png` (Patient probability 7.1%, actual readmit).
    - `reports/figures/shap_waterfall_Case_C_LowRiskBaseline.png` (Patient probability 4.6%, no readmit).
  - Reports generated: `reports/explainability.md`, `reports/odds_ratios.csv`.
- **Key measured numbers:**
  - **SHAP Stability Score:** Spearman rank correlation = **0.9909** across bootstrap resamples (indicating exceptionally stable global attribution hierarchy).
  - **Top SHAP Clinical Drivers:**
    1. `discharge_group_Home` (Mean |SHAP| = 0.2036) - discharge home strongly decreases predicted risk.
    2. `age_midpoint` (Mean |SHAP| = 0.1059) - older age consistently elevates readmission risk.
    3. `number_inpatient` (Mean |SHAP| = 0.0929) - prior inpatient utilization is the strongest positive risk driver.
    4. `a1c_tested` (Mean |SHAP| = 0.0620).
    5. `discharge_group_Other` (Mean |SHAP| = 0.0590).
  - **Top Logistic Regression Odds Ratios:**
    - Higher Risk: `diag_2_group_Neoplasms` (OR = **1.44x**, log-odds +0.3633), `medical_specialty_group_Nephrology` (OR = **1.34x**), `diabetesMed` (OR = **1.32x**), `any_prior_inpatient` (OR = **1.27x**).
    - Lower Risk: `discharge_group_Home` (OR = **0.57x**, log-odds -0.5555), `a1c_result_cat_none` (OR = **0.61x**), `glu_serum_cat_none` (OR = **0.62x**).
- **Commands run & results:**
  - `python -m readmit.cli explain`: Executed in ~3.8s. All figures and markdown reports written.
  - `ruff check src tests`: Passed (0 errors).
  - `pytest -v`: 12 passed.
- **Assumptions made:**
  - SHAP values computed on uncalibrated tree log-odds output per standard practice, representing additive statistical associations with zero causal claims.
- **Open issues:** None.

---

## Checkpoint 6: M7 Fairness Audit & Mitigation Complete
- **What was built:**
  - `src/readmit/fairness.py`: Demographic subgroup audit evaluating True Positive Rate (Recall), False Positive Rate (False Alarm), Precision, and Selection Rate across Race/Ethnicity, Gender, and Age bands using Fairlearn `MetricFrame`.
  - Disparity gap metrics computed with sample size safeguards (groups with N < 500 flagged as low-confidence).
  - Algorithmic fairness mitigation using Fairlearn `ThresholdOptimizer` with `equalized_odds` constraint fitted on validation set demographic groups.
  - Generated report: `reports/fairness_before_after.md` documenting trade-offs, group-specific thresholds, and clinical resource allocation implications.
  - Artifact saved: `artifacts/fairness_optimizer.joblib`.
  - Unit test in `tests/test_features.py`: `test_fairness_gaps_toy` asserts TPR/FPR/selection-rate gap math against hand-computed ground truth.
- **Key measured numbers (Validation Set: N=10,497):**
  - **Before Mitigation (Primary Capacity K=20%, Global Cutoff 10.77%):**
    - Race Group Disparities: TPR Gap = **2.58%** (African American 44.97%, Caucasian 47.55%), FPR Gap = **4.85%** (African American 21.61%, Caucasian 26.46%).
    - Gender Disparities: TPR Gap = **5.89%** (Female 49.41%, Male 43.52%), FPR Gap = **4.56%** (Female 27.35%, Male 22.80%).
    - Age Band Disparities: TPR Gap = **22.23%** (Age 30-49 33.01%, Age 70+ 55.24%), FPR Gap = **23.68%**.
    - Overall Performance: Recall = **46.71%**, Precision = **15.43%**, Selection Rate = **27.16%**.
  - **After Mitigation (Fairlearn Equalized Odds Post-Processing on Race):**
    - Race Group Disparities: TPR Gap = **2.78%**, FPR Gap reduced dramatically from 4.85% down to **0.11%** (African American FPR 23.16%, Caucasian FPR 23.26%).
    - Overall Performance: Recall = **43.95%**, Precision = **15.71%**, Selection Rate = **25.10%**.
  - **Trade-Off Finding:** Equalized odds enforcement virtually eliminated the false alarm disparity across major racial cohorts (gap decreased from 4.85% to 0.11%) with an overall recall shift of only -2.76% and slightly improved precision (+0.28%).
- **Commands run & results:**
  - `python -m readmit.cli fairness`: Executed in ~2.5s. All subgroup metrics computed and written to `reports/fairness_before_after.md`.
  - `ruff check src tests`: Passed (0 errors).
  - `pytest -v`: 13 passed in 2.35s.
- **Assumptions made:**
  - Race and gender remain strictly excluded from model features and are evaluated purely as demographic audit attributes.
  - Groups with N < 500 (e.g., Asian N=80, Hispanic N=242, Other N=156 in validation split) are explicitly annotated with sample size warnings.
- **Open issues:** None.

---

## Checkpoint 7: M8 Final Test Set Evaluation & Metrics Freezing Complete
- **What was built:**
  - `src/readmit/final_eval.py`: Scored the untouched test set (N=10,500 patients, prevalence 8.99%) **strictly once** (Invariant 4).
  - Paired 1,000-resample bootstrap 95% confidence intervals computed across all candidate models (Calibrated XGBoost, Calibrated Logistic Regression, Raw XGBoost).
  - Test-set lock mechanism enforced via `artifacts/.final_done` with timestamp and git commit hash; CLI step `readmit.cli final` now refuses to re-run unless called with `--force`.
  - Demographic fairness audit evaluated on the test set before and after Fairlearn mitigation.
  - Frozen single source of truth report: `reports/metrics.json`.
- **Key measured numbers (Test Set: N=10,500, Positives=944, Prevalence=8.99%):**
  - **Primary Model: XGBoost Calibrated (Isotonic):**
    - **PR-AUC:** **0.1385** [95% Bootstrap CI: **0.1268 - 0.1529**].
    - **ROC-AUC:** **0.6344** [95% Bootstrap CI: **0.6168 - 0.6532**] (Leakage invariant preserved: ROC-AUC is well below the 0.80 ceiling).
    - **Brier Score:** **0.0806** [95% Bootstrap CI: **0.0762 - 0.0848**].
    - **Expected Calibration Error (ECE, 10 uniform bins):** **0.0062**.
  - **Comparative Model: Logistic Regression Calibrated (Platt):**
    - PR-AUC: **0.1351** [95% CI: 0.1232 - 0.1504], ROC-AUC: **0.6208** [95% CI: 0.6025 - 0.6388], Brier: **0.0808**, ECE: **0.0086**.
  - **Comparative Model: Raw XGBoost (Uncalibrated):**
    - PR-AUC: **0.1456** [95% CI: 0.1330 - 0.1623], ROC-AUC: **0.6378** [95% CI: 0.6198 - 0.6562], Brier: **0.0802**, ECE: **0.0035**.
  - **Capacity-Based Prioritization on Test Set (Calibrated XGBoost):**
    - **Top 5%:** Flagged = 525, Captured = 101, Recall = **10.70%**, Precision = **19.24%**, Lift = **2.14x**.
    - **Top 10%:** Flagged = 1,050, Captured = 189, Recall = **20.02%**, Precision = **18.00%**, Lift = **2.00x**.
    - **Top 20% (Primary Capacity Assumption):** Flagged = 2,100, Captured = 326, Recall = **34.53%**, Precision = **15.52%**, Lift = **1.73x**.
  - **Fairness Mitigation on Test Set (Equalized Odds vs Unmitigated):**
    - Unmitigated Race FPR Disparity Gap: **5.07%** (African American 21.32% vs Caucasian 26.40%).
    - Mitigated Race FPR Disparity Gap: **0.17%** (African American 23.36% vs Caucasian 23.53%).
    - Disparity reduction: **96.6% reduction in false alarm rate gap** across racial groups on the test set.
- **Commands run & results:**
  - `python -m readmit.cli final`: Completed successfully. Created `artifacts/.final_done` and `reports/metrics.json`.
  - Second execution check: Correctly aborted with error `Test set evaluation already executed and locked`.
  - `ruff check src tests`: Passed (0 errors).
  - `pytest -v`: 13 passed in 2.35s.
- **Assumptions made:**
  - Test set was evaluated strictly once with models and thresholds frozen from validation.
- **Open issues:** None.

---

## Checkpoint 8: M9 Engineering Layer Complete
- **What was built:**
  - `src/readmit/inference.py`: Production-grade inference engine with model caching (`load_inference_artifacts`), `predict(patient_data)` returning calibrated probability, relative risk vs. average, and clinical risk bands, and `explain(patient_data)` extracting local SHAP feature attributions.
  - `api/main.py`: FastAPI service exposing `GET /health`, `POST /predict`, and `POST /explain` with strict Pydantic v2 data validation and graceful handling of unknown clinical categories.
  - `app/streamlit_app.py`: Comprehensive 5-page interactive dashboard covering Executive Summary, Patient Risk Scoring, SHAP Interpretability, Fairness by Group, and Model Comparison & Capacity Tiers with operational assumptions in sidebar.
  - `Dockerfile`: Production containerfile packaging application, dependencies, model artifacts, and frozen reports.
  - Automated test suite in `tests/test_api.py`: `test_inference_roundtrip` and `test_api_schema` (asserting 200 responses and 422 validation failure).
- **Key measured numbers:**
  - Full automated pytest suite: **15 tests passing** (data, features, calibration, fairness, inference, API).
  - Ruff linting: **0 errors** across `src`, `tests`, `api`, `app`.
- **Commands run & results:**
  - `pytest -v`: 15 passed in 3.42s.
  - `ruff check src tests api app`: Passed (0 errors).
- **Assumptions made:**
  - Streamlit dashboard and FastAPI service use cached inference objects for sub-100ms response times.
- **Open issues:** None.

---

## Checkpoint 9: M10 Documentation, Executive Report, and Final Handoff Complete
- **What was built:**
  - `reports/executive_report.md`: Comprehensive clinical and technical report covering cohort flow, capacity-aware decision support, discrimination/calibration benchmark, SHAP feature drivers, and demographic fairness findings.
  - `README.md`: Production-ready documentation citing exact frozen figures from `reports/metrics.json`, detailing architecture, reproducibility, CLI commands, and clinical governance disclaimers.
  - `reports/FINAL_SUMMARY.md`: End-to-end handoff document containing complete milestone checklist, exact metric tables, decisions log, limitations, run instructions, and form-ready resume bullets.
- **Key measured numbers:**
  - All numbers in documentation and reports trace directly and strictly to `reports/metrics.json`.
  - Zero placeholder values; zero fabricated clinical numbers.
- **Commands run & results:**
  - `pytest -v`: All 15 unit tests passed.
  - `ruff check src tests api app`: All checks passed (0 errors).
- **Assumptions made:**
  - Project milestone delivery (M0 to M10) across Priority Tiers P0 to P3 is 100% complete and fully verified.
- **Open issues:** None. All requirements fulfilled.








