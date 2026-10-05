# Model Comparison and Imbalance Ablation Report

Comprehensive validation set comparison evaluating Logistic Regression and XGBoost across class imbalance mitigation strategies: None (natural prevalence), Class Weighting (cost-sensitive), and SMOTE oversampling.

## Validation Performance Summary Table

| Model | Imbalance Strategy | PR-AUC | ROC-AUC | Brier Score | ECE | Recall @ K=20% | Precision @ K=20% | Lift @ K=20% |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | None (unweighted) | 0.1647 | 0.6454 | 0.0794 | 0.0014 | 36.94% | 16.57% | 1.85x |
| XGBoost | None (unweighted) | 0.1716 | 0.6489 | 0.0792 | 0.0014 | 37.69% | 16.90% | 1.88x |
| Logistic Regression | Class Weight (balanced) | 0.1634 | 0.6459 | 0.2309 | 0.3800 | 36.52% | 16.38% | 1.83x |
| XGBoost | scale_pos_weight (10.14) | 0.1645 | 0.6385 | 0.2141 | 0.3532 | 35.56% | 15.95% | 1.78x |
| Logistic Regression | SMOTE Oversampling | 0.1576 | 0.6345 | 0.2315 | 0.3756 | 35.77% | 16.05% | 1.79x |
| XGBoost | SMOTE Oversampling | 0.1599 | 0.6365 | 0.0807 | 0.0295 | 35.77% | 16.05% | 1.79x |

## Analysis & Findings

1. **Primary Metric Performance (PR-AUC):** The top performing model by PR-AUC is **XGBoost (None (unweighted))** with PR-AUC = **0.1716** and ROC-AUC = **0.6489**.
2. **Impact of Imbalance Reweighting:**
   - Class weighting and SMOTE significantly alter the predicted probability distribution, shifting raw outputs upward. While rank ordering (ROC-AUC / capacity ranking) remains comparable, raw Brier score and ECE degrade sharply because probabilities no longer reflect natural clinical prevalence (~8.98%).
   - Consequently, when reweighting or SMOTE is used, post-hoc recalibration (Platt scaling or isotonic regression) is strictly required before deploying probabilities into clinical workflows.
3. **Capacity Decision Support (K=20%):**
   - When targeting the highest risk quintile (K=20%), XGBoost captures **37.69%** of all 30-day readmissions, achieving a precision of **16.90%** and a lift of **1.88x** over baseline hospital prevalence.

## Decision for Downstream Pipeline

- **Chosen Primary Architecture:** XGBoost with natural prevalence weighting as the uncalibrated base estimator, passing forward to Milestone M5 for Platt scaling and Isotonic calibration.
- **Baseline Comparison:** Both LR and XGBoost substantially outperform the uninformative prevalence baseline (PR-AUC 0.0897) and the prior inpatient clinical rule heuristic (PR-AUC 0.1201).
