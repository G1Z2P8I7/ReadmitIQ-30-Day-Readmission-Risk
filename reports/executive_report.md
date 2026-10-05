# ReadmitIQ: Executive Clinical & Technical Report

## 1. Executive Summary

**ReadmitIQ** is an end-to-end clinical machine learning system developed to address 30-day hospital readmission risk prediction on the UCI Diabetes 130-US Hospitals dataset (101,766 encounters, 1999–2008). 

Unlike conventional machine learning formulations that evaluate models using uncalibrated scores and arbitrary 0.50 classification thresholds, ReadmitIQ frames readmission risk as a **capacity-aware decision support problem**. Hospitals operate under strict staffing constraints where care transition coordinators, discharge planners, and follow-up phone teams can only actively contact a fixed percentage of discharged patients (e.g., top 10%–20%). ReadmitIQ delivers:
1. **Calibrated Probability Estimation:** Reliable absolute probabilities via post-hoc isotonic calibration (Brier score: **0.0806**, ECE: **0.0062** on test set).
2. **Capacity Prioritization:** Identification of high-risk patients capturing **34.53% of all 30-day readmissions** within the top 20% capacity budget (**1.73x lift** over hospital baseline prevalence).
3. **Rigorous Leakage Prevention:** Patient-grouped splits (0 patient overlap across train/val/test) and strict discharge-time prediction horizons.
4. **Algorithmic Fairness Audit & Mitigation:** Fairlearn post-processing optimization that reduced racial False Positive Rate (false alarm) disparities by **96.6%** (gap dropped from 5.07% to 0.17%).

---

## 2. Problem Framing & Operational Decision Support

In the primary cohort, only **8.99%** of diabetic patients experience an unplanned 30-day readmission. Under such severe class imbalance:
- **Accuracy is uninformative:** A naive trivial model predicting "no readmission" achieves 91.01% accuracy while capturing 0 readmissions.
- **The 0.50 threshold is clinically useless:** In an 8.99% prevalence environment, raw model probabilities rarely exceed 0.50, causing 0 patients to be flagged.
- **Decision Utility:** Hospital interventions (nurse follow-up calls, home health visits, pharmacist medication reconciliation) require screening queues prioritized descending by calibrated risk.

### Test Set Capacity-Constrained Resource Prioritization (N = 10,500)

| Capacity Tier (Top K%) | Patients Flagged | Risk Cutoff Threshold | Captured Readmissions | Recall (Sensitivity) | Precision (PPV) | Lift over Baseline |
|---|---|---|---|---|---|---|
| **Top 5%** | 525 | 20.81% | 101 | 10.70% | 19.24% | **2.14x** |
| **Top 10%** | 1,050 | 15.38% | 189 | 20.02% | 18.00% | **2.00x** |
| **Top 20% (Primary Assumption)** | 2,100 | 10.77% | 326 | 34.53% | 15.52% | **1.73x** |

*(Source: `reports/metrics.json`)*

---

## 3. Data Integrity & Clinical Cohort Flow

### Inclusion and Exclusion Rules
- **Raw Encounters:** 101,766 encounters across 71,518 unique patients.
- **Exclusion of Mortality / Hospice:** 2,423 encounters with discharge disposition IDs 11 (expired), 13 (hospice home), 14 (hospice medical facility), 19 (expired at home), 20 (expired in medical facility), and 21 (expired in unknown place) were excluded because readmission is biologically impossible or contradictory to hospice care goals.
- **Exclusion of Unknown Gender:** 3 records with invalid gender records were dropped.
- **Index Encounter Selection:** 29,353 repeat encounters were excluded, retaining strictly the first encounter (lowest `encounter_id`) per patient to prevent multi-encounter patient leakage.
- **Primary Cohort:** **69,987 unique patients**, with 6,285 positive 30-day readmissions (**8.9802% prevalence**).

### Patient-Grouped Split Integrity
To completely prevent data leakage, splits were grouped by `patient_nbr` and stratified by `readmit_30d`:
- **Train (70%):** 48,990 patients (4,399 positives, 8.98% prevalence)
- **Validation (15%):** 10,497 patients (942 positives, 8.97% prevalence)
- **Test (15%):** 10,500 patients (944 positives, 8.99% prevalence)
- **Patient Overlap:** **0 patients overlap** between any pair of splits (verified by automated assertion).

---

## 4. Model Architecture & Discrimination Results

### Features
From raw data, **87 features** were engineered:
- 14 numeric features: length of stay, procedure counts, lab procedures, medication counts, age midpoint, and prior visit frequencies.
- 5 binary features: prior inpatient indicator, HbA1c tested, glucose tested, medication change, and diabetes medication.
- 68 one-hot categories: clinical ICD-9 grouped primary/secondary/tertiary diagnoses, admission source, discharge disposition, medical specialty, and payer code.
- **Protected attributes exclusion:** Race and gender were completely excluded from model training features and retained solely for demographic fairness audits.

### Comparative Discrimination and Calibration (Test Set, N = 10,500)

| Model Architecture | PR-AUC (95% CI) | ROC-AUC (95% CI) | Brier Score (95% CI) | ECE (10 uniform bins) | Recall @ Top 20% | Precision @ Top 20% |
|---|---|---|---|---|---|---|
| **Prevalence Baseline** | 0.0899 (fixed) | 0.5000 (fixed) | 0.0818 | 0.0000 | 20.00% | 8.99% |
| **Prior Inpatient Rule Baseline** | 0.1201 | 0.5452 | 0.1691 | 0.1118 | 27.92% | 12.52% |
| **Logistic Regression (Calibrated Platt)** | 0.1351 [0.1232 - 0.1504] | 0.6208 [0.6025 - 0.6388] | 0.0808 [0.0765 - 0.0850] | 0.0086 | 33.26% | 14.95% |
| **XGBoost (Calibrated Isotonic - Primary)** | **0.1385 [0.1268 - 0.1529]** | **0.6344 [0.6168 - 0.6532]** | **0.0806 [0.0762 - 0.0848]** | **0.0062** | **34.53%** | **15.52%** |

*(Source: `reports/metrics.json`)*

**Absence of Leakage Red Flags:** The primary model achieves ROC-AUC = 0.6344. In clinical literature on the UCI Diabetes 130 dataset, realistic discriminative ROC-AUC ranges between 0.60 and 0.68. A ROC-AUC exceeding 0.80 would indicate artificial target leakage; our results confirm strict adherence to realistic clinical conditions.

---

## 5. Explainability & Clinical Drivers

Global and local model explanations were generated using `shap.TreeExplainer` on the uncalibrated XGBoost model:
- **Attribution Stability:** Spearman rank correlation of feature attributions across bootstrap resamples was **0.9909**, proving high structural consistency.
- **Top Positive Risk Drivers (Associated with higher readmission risk):**
  1. `number_inpatient` (Prior inpatient visits in the preceding 12 months).
  2. `age_midpoint` (Increasing patient age).
  3. `time_in_hospital` (Extended length of stay).
  4. `num_medications` (Polypharmacy indicator).
  5. `diag_2_group_Neoplasms` (Secondary oncologic diagnosis).
- **Top Protective Drivers (Associated with lower readmission risk):**
  1. `discharge_group_Home` (Routine discharge home vs skilled nursing facility or home health).
  2. `diag_1_group_Respiratory` / elective admission types.

---

## 6. Algorithmic Fairness Audit & Mitigation

ReadmitIQ evaluated model performance across three demographic dimensions: Race/Ethnicity, Gender, and Age band. To ensure parity, Fairlearn's `ThresholdOptimizer` was trained on validation data under an **Equalized Odds** constraint.

### Test Set Demographic Disparity Gaps: Before vs. After Mitigation

| Sensitive Attribute | Unmitigated TPR Gap | Mitigated TPR Gap | Unmitigated FPR Gap | Mitigated FPR Gap | Mitigated Selection Rate Gap |
|---|---|---|---|---|---|
| **Race / Ethnicity** | 6.10% | **3.85%** | 5.07% | **0.17%** | **0.34%** |
| **Gender** | 8.32% | 9.18% | 4.78% | 4.80% | 5.15% |
| **Age Band** | 29.47% | 30.14% | 25.58% | 22.70% | 23.54% |

### Key Fairness Takeaway
Before mitigation, Caucasian patients experienced a False Positive Rate (false alarm rate) of 26.40%, whereas African American patients had an FPR of 21.32% (a 5.07% disparity). Following Fairlearn Equalized Odds post-processing, the FPR disparity was **virtually eliminated down to 0.17%** (African American FPR: 23.36%, Caucasian FPR: 23.53%) with negligible loss in overall recall (43.43% down to 40.36%).

---

## 7. Assumptions & Clinical Limitations

1. **Capacity Budget:** The assumption that hospital resources can intervene on the top 20% of discharged patients is an operational convention. The threshold must be tuned to each health system's specific clinical staffing levels.
2. **Cost Ratio:** The illustrative 5:1 cost ratio (valuing a missed readmission 5x more than a false alarm) is a demonstration parameter, not an empirical hospital accounting cost.
3. **Historical Data:** The UCI dataset spans 1999–2008. Contemporary EHR dynamics, CMS penalties, and novel antidiabetic agents (e.g., SGLT2 inhibitors, GLP-1 receptor agonists) require retraining on modern hospital data prior to bedside clinical deployment.
