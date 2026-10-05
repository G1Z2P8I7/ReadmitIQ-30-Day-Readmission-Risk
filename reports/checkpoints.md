# Project Execution Checkpoints

Running progress log tracking milestone delivery, commands executed, verification results, and key metrics.

## Milestone Status Overview

- [x] **M0: Scaffold**
- [x] **M1: Data audit and cohort**
- [x] **M2: Leakage audit and splits**
- [ ] **M3: Features and baselines**
- [ ] **M4: LR and XGBoost, imbalance ablation**
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
- **Open issues:** None. Proceeding autonomously to M3 (Features and baselines).
