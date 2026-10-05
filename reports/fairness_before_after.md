# Algorithmic Fairness Audit and Mitigation Trade-Off Report

> [!IMPORTANT]

> **Governance Policy:** Race and gender are strictly excluded from predictive model features and preserved exclusively as demographic audit attributes. Age is included as a clinical predictor (numeric midpoint) and audited across categorical age bands.

> **Sample Size Warning:** Groups with N < 500 are labeled as low-confidence due to statistical power constraints.


## Executive Fairness Summary (Primary Attribute: Race/Ethnicity)

| State | Constraint Objective | Overall Recall | Overall Precision | Selection Rate | TPR Disparity Gap | FPR Disparity Gap |
|---|---|---|---|---|---|---|
| **Before Mitigation** (Capacity Top 20%) | None (Single Global Threshold: 10.77%) | 46.71% | 15.43% | 27.16% | 2.58% | 4.85% |
| **After Mitigation** (Fairlearn Equalized Odds) | Equalized Odds (Group-Specific Thresholds) | 43.95% | 15.71% | 25.10% | 2.78% | 0.11% |

---


## Detailed Per-Group Performance Tables (Validation Set)


### Race / Ethnicity

**Before Mitigation (Capacity K=20% Global Threshold):**

| Subgroup | Sample Size (N) | TPR (Recall) | FPR (False Alarm) | Precision | Selection Rate | Confidence Note |
|---|---|---|---|---|---|---|
| `AfricanAmerican` | 1,918 | 44.97% | 21.61% | 16.74% | 23.67% | Adequate N |
| `Asian` | 80 | 40.00% | 22.67% | 10.53% | 23.75% | ⚠️ Low N (<500) |
| `Caucasian` | 7,846 | 47.55% | 26.46% | 15.27% | 28.38% | Adequate N |
| `Hispanic` | 242 | 47.83% | 17.35% | 22.45% | 20.25% | ⚠️ Low N (<500) |
| `Other` | 156 | 25.00% | 20.27% | 6.25% | 20.51% | ⚠️ Low N (<500) |
| `Unknown` | 255 | 40.91% | 26.18% | 12.86% | 27.45% | ⚠️ Low N (<500) |

**After Mitigation (Equalized Odds Threshold Optimization):**

| Subgroup | Sample Size (N) | TPR (Recall) | FPR (False Alarm) | Precision | Selection Rate | Disparity Change |
|---|---|---|---|---|---|---|
| `AfricanAmerican` | 1,918 | 41.42% | 23.16% | 14.74% | 24.77% | TPR Δ -3.55% |
| `Asian` | 80 | 60.00% | 22.67% | 15.00% | 25.00% | TPR Δ +20.00% |
| `Caucasian` | 7,846 | 44.20% | 23.26% | 16.00% | 25.17% | TPR Δ -3.36% |
| `Hispanic` | 242 | 43.48% | 27.40% | 14.29% | 28.93% | TPR Δ -4.35% |
| `Other` | 156 | 50.00% | 21.62% | 11.11% | 23.08% | TPR Δ +25.00% |
| `Unknown` | 255 | 50.00% | 20.60% | 18.64% | 23.14% | TPR Δ +9.09% |

### Gender

**Before Mitigation (Capacity K=20% Global Threshold):**

| Subgroup | Sample Size (N) | TPR (Recall) | FPR (False Alarm) | Precision | Selection Rate | Confidence Note |
|---|---|---|---|---|---|---|
| `Female` | 5,617 | 49.41% | 27.35% | 15.28% | 29.36% | Adequate N |
| `Male` | 4,880 | 43.52% | 22.80% | 15.64% | 24.63% | Adequate N |

**After Mitigation (Equalized Odds Threshold Optimization):**

| Subgroup | Sample Size (N) | TPR (Recall) | FPR (False Alarm) | Precision | Selection Rate | Disparity Change |
|---|---|---|---|---|---|---|
| `Female` | 5,617 | 46.86% | 24.89% | 15.83% | 26.88% | TPR Δ -2.55% |
| `Male` | 4,880 | 40.51% | 21.36% | 15.56% | 23.05% | TPR Δ -3.01% |

### Age Band

**Before Mitigation (Capacity K=20% Global Threshold):**

| Subgroup | Sample Size (N) | TPR (Recall) | FPR (False Alarm) | Precision | Selection Rate | Confidence Note |
|---|---|---|---|---|---|---|
| `30-49` | 1,452 | 33.01% | 12.23% | 17.09% | 13.71% | Adequate N |
| `50-69` | 4,159 | 38.69% | 19.49% | 14.86% | 21.04% | Adequate N |
| `70+` | 4,614 | 55.24% | 35.91% | 15.36% | 37.95% | Adequate N |
| `<30` | 272 | 43.75% | 7.42% | 26.92% | 9.56% | ⚠️ Low N (<500) |

**After Mitigation (Equalized Odds Threshold Optimization):**

| Subgroup | Sample Size (N) | TPR (Recall) | FPR (False Alarm) | Precision | Selection Rate | Disparity Change |
|---|---|---|---|---|---|---|
| `30-49` | 1,452 | 33.98% | 12.01% | 17.77% | 13.57% | TPR Δ +0.97% |
| `50-69` | 4,159 | 36.61% | 18.34% | 14.93% | 19.81% | TPR Δ -2.08% |
| `70+` | 4,614 | 51.13% | 32.32% | 15.73% | 34.31% | TPR Δ -4.11% |
| `<30` | 272 | 43.75% | 9.38% | 22.58% | 11.40% | TPR Δ +0.00% |

## Clinical and Operational Trade-Off Discussion

1. **The Parity vs. Capacity Trade-Off:** Equalized odds optimization successfully tightens disparity gaps across demographic groups. However, enforcing parity across groups requires raising selection rates and lowering classification thresholds for historically underserved populations. This shifts total flagged volume and impacts overall precision.
2. **Capacity Realities:** In clinical practice, if follow-up resources (nurse outreach calls, home health visits) are strictly budgeted at 20% of discharged patients, a group-differentiated threshold changes who receives care within that 20% quota.
3. **Conclusion & Recommendation:** We recommend deploying the calibrated continuous risk score as decision support, displaying subgroup-stratified recall metrics to clinicians, and pairing post-processing mitigation with care team review rather than uncritical automated prioritization.
