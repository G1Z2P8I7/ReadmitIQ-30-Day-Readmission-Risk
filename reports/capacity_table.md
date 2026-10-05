# Capacity and Decision-Support Thresholds Report

> [!NOTE]

> **Operational Assumption:** Hospital intervention capacity determines practical screening utility. Defaulting to an arbitrary 0.50 classification threshold is clinically invalid when baseline readmission prevalence is ~8.98%. All cost ratios and capacity tiers are explicit operational assumptions labeled below.


## Capacity-Constrained Resource Prioritization (Validation Set)

- **Validation Cohort Size:** 10,497 patients

- **Observed Baseline Readmission Prevalence:** 8.97%


| Capacity Tier (Top K%) | Patients Flagged | Risk Cutoff Threshold | Captured Readmissions | Recall (Sensitivity) | Precision (PPV) | Lift over Baseline |
|---|---|---|---|---|---|---|
| Top 5% | 525 | 20.81% | 132 | 14.01% | 25.14% | 2.80x |
| Top 10% | 1,050 | 15.38% | 226 | 23.99% | 21.52% | 2.40x |
| Top 20% | 2,100 | 10.77% | 362 | 38.43% | 17.24% | 1.92x |

## Cost-Ratio Utility Optimization (Illustrative Assumption)

- **Cost Ratio (Cost_FN / Cost_FP):** 5.0x (Missing a readmission assumed 5x more costly than false alarm review).

- **Analytically Optimal Probability Threshold:** 15.85%

- **Precision at Optimal Threshold:** 22.86%

- **Recall at Optimal Threshold:** 21.02%

- **F1 Score at Optimal Threshold:** 0.2190


## Calibration Summary (Brier & ECE)

| Variant | Brier Score (lower is better) | ECE (uniform 10 bins) | PR-AUC |
|---|---|---|---|
| Raw XGBoost | 0.0792 | 0.0014 | 0.1716 |
| Platt Scaling (Sigmoid) | 0.0792 | 0.0037 | 0.1716 |
| Isotonic Regression | 0.0788 | 0.0000 | 0.1708 |

**Selected Calibration Method:** `isotonic`. Preserves monotonicity and output stability.
