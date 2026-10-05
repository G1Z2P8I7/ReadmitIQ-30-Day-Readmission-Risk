# Model Explainability and Clinical Interpretability Report

> [!NOTE]

> **Methodological Grounding:** Explanations reflect model-learned statistical associations on the uncalibrated model log-odds output. SHAP values indicate contribution toward predicted risk; they must **never** be interpreted as causal guarantees or treatment recommendations.


## Global Feature Attributions (SHAP Beeswarm Analysis)

- **Explainer Type:** `shap.TreeExplainer` on uncalibrated XGBoost base estimator

- **Stability Score (Spearman Rank Correlation across bootstrap resamples):** **0.9909**


### Top 10 Most Influential Features across Validation Cohort:

1. **`discharge_group_Home`** (Mean |SHAP| = 0.2036)
2. **`age_midpoint`** (Mean |SHAP| = 0.1059)
3. **`number_inpatient`** (Mean |SHAP| = 0.0929)
4. **`a1c_tested`** (Mean |SHAP| = 0.0620)
5. **`discharge_group_Other`** (Mean |SHAP| = 0.0590)
6. **`time_in_hospital`** (Mean |SHAP| = 0.0485)
7. **`num_medications`** (Mean |SHAP| = 0.0445)
8. **`discharge_group_SNF`** (Mean |SHAP| = 0.0444)
9. **`n_meds_active`** (Mean |SHAP| = 0.0432)
10. **`number_diagnoses`** (Mean |SHAP| = 0.0414)

## Odds Ratios from Interpretable Logistic Regression

Top clinical factors associated with higher readmission risk (Odds Ratio > 1.0):

| Feature | Coefficient (Log-Odds) | Odds Ratio (95% Wald CI) | Interpretation |
|---|---|---|---|
| `diag_2_group_Neoplasms` | +0.3633 | 1.44x | Associated with increased readmission risk |
| `medical_specialty_group_Nephrology` | +0.2910 | 1.34x | Associated with increased readmission risk |
| `diabetesMed` | +0.2772 | 1.32x | Associated with increased readmission risk |
| `any_prior_inpatient` | +0.2392 | 1.27x | Associated with increased readmission risk |
| `discharge_group_Other` | +0.2003 | 1.22x | Associated with increased readmission risk |

Top clinical factors associated with lower readmission risk (Odds Ratio < 1.0):

| Feature | Coefficient (Log-Odds) | Odds Ratio | Interpretation |
|---|---|---|---|
| `discharge_group_Home` | -0.5555 | 0.57x | Associated with lower readmission risk |
| `a1c_result_cat_none` | -0.5005 | 0.61x | Associated with lower readmission risk |
| `glu_serum_cat_none` | -0.4800 | 0.62x | Associated with lower readmission risk |
| `medical_specialty_group_Orthopedics-Reconstructive` | -0.3700 | 0.69x | Associated with lower readmission risk |
| `discharge_group_Home_Health` | -0.3684 | 0.69x | Associated with lower readmission risk |

## Clinical Case Studies (SHAP Waterfall Attributions)

1. **Case A (True Positive - Flagged High Risk):** Patient probability = **20.8%**. Major risk drivers identified by waterfall plot: prior inpatient encounters, extended length of stay, and polypharmacy.

2. **Case B (False Negative - Clinical Blindspot):** Patient probability = **7.1%** (actual readmitted within 30 days). Lack of prior utilization masked subtle diagnosis-specific risk factors.

3. **Case C (True Negative - Routine Low Risk):** Patient probability = **4.6%**. Zero prior visits and straightforward routine discharge to home drove negative SHAP values.


Figures generated under `reports/figures/`: `shap_summary_xgb.png`, `shap_waterfall_Case_A_TruePositive.png`, `shap_waterfall_Case_B_FalseNegative.png`, `shap_waterfall_Case_C_LowRiskBaseline.png`.
