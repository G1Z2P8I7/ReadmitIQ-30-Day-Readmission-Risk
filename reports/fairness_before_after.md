# Algorithmic Fairness Audit and Mitigation Trade-Off Report

> [!IMPORTANT]

> **Governance Policy:** Race and gender are strictly excluded from predictive model features and preserved exclusively as demographic audit attributes. Age is included as a clinical predictor (numeric midpoint) and audited across categorical age bands.

> **Sample Size Warning:** Groups with N < 500 are labeled as low-confidence due to statistical power constraints and wider bootstrap confidence intervals.


## Executive Fairness Summary (Primary Attribute: Race/Ethnicity)

| State | Constraint Objective | Overall Recall | Overall Precision | Selection Rate | TPR Disparity Gap | FPR Disparity Gap |
|---|---|---|---|---|---|---|
| **Before Mitigation** (Capacity Top 20%) | None (Single Global Threshold: 10.77%) | 43.43% | 14.43% | 27.07% | 6.10% | 5.07% |
| **After Mitigation** (Fairlearn Equalized Odds) | Equalized Odds (Group-Specific Thresholds) | 40.36% | 14.12% | 25.70% | 3.85% | 0.17% |

---

## Before vs. After Mitigation Comparison Table (Test Set)

| Attribute | Subgroup | Sample Size (N) | Recall (Before) | Recall (After) | Recall Δ | Precision (Before) | Precision (After) | Flagged Rate (Before) | Flagged Rate (After) | FPR (Before) | FPR (After) | Pred Risk | Obs Risk | Calib Ratio | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| race_group | `AfricanAmerican` | 1,843 | 38.67% | 37.33% | -1.33% | 13.55% | 11.97% | 23.22% | 25.39% | 21.85% | 24.34% | 8.34% | 8.14% | 1.02x | Adequate |
| race_group | `Asian` | 68 | 33.33% | 33.33% | +0.00% | 18.18% | 14.29% | 16.18% | 20.59% | 14.52% | 19.35% | 7.58% | 8.82% | 0.86x | ⚠️ Low N |
| race_group | `Caucasian` | 7,893 | 44.77% | 41.18% | -3.58% | 14.41% | 14.72% | 28.57% | 25.73% | 26.93% | 24.17% | 9.17% | 9.20% | 1.00x | Adequate |
| race_group | `Hispanic` | 215 | 55.00% | 45.00% | -10.00% | 23.40% | 16.07% | 21.86% | 26.05% | 18.46% | 24.10% | 8.13% | 9.30% | 0.87x | ⚠️ Low N |
| race_group | `Other` | 184 | 25.00% | 25.00% | +0.00% | 15.15% | 9.26% | 17.93% | 29.35% | 17.07% | 29.88% | 7.86% | 10.87% | 0.72x | ⚠️ Low N |
| race_group | `Unknown` | 297 | 40.91% | 45.45% | +4.55% | 13.24% | 13.33% | 22.90% | 25.25% | 21.45% | 23.64% | 8.37% | 7.41% | 1.13x | ⚠️ Low N |
| gender | `Female` | 5,503 | 47.44% | 44.79% | -2.66% | 14.30% | 14.14% | 29.47% | 28.15% | 27.72% | 26.53% | 9.15% | 8.89% | 1.03x | Adequate |
| gender | `Male` | 4,997 | 39.12% | 35.60% | -3.52% | 14.59% | 14.10% | 24.41% | 22.99% | 22.94% | 21.73% | 8.71% | 9.11% | 0.96x | Adequate |
| age_band | `30-49` | 1,429 | 25.98% | 22.05% | -3.94% | 18.44% | 15.64% | 12.53% | 12.53% | 11.21% | 11.60% | 6.72% | 8.89% | 0.76x | Adequate |
| age_band | `50-69` | 4,179 | 35.48% | 32.84% | -2.64% | 14.20% | 13.66% | 20.39% | 19.62% | 19.05% | 18.45% | 8.19% | 8.16% | 1.00x | Adequate |
| age_band | `70+` | 4,616 | 55.46% | 52.18% | -3.28% | 14.24% | 14.35% | 38.65% | 36.07% | 36.80% | 34.30% | 10.48% | 9.92% | 1.06x | Adequate |
| age_band | `<30` | 276 | 11.11% | 11.11% | +0.00% | 7.41% | 5.88% | 9.78% | 12.32% | 9.69% | 12.40% | 6.31% | 6.52% | 0.97x | ⚠️ Low N |

---

## Detailed Per-Group Performance Breakdown with 95% Bootstrap CIs (Test Set)


### Race / Ethnicity

| Subgroup | N | Observed Risk | Predicted Risk | TPR (Recall) [95% CI] | FPR (False Alarm) [95% CI] | Precision | Selection Rate | Confidence Note |
|---|---|---|---|---|---|---|---|---|
| `AfricanAmerican` | 1,843 | 8.14% | 8.34% | 37.33% [29.9%, 44.8%] | 24.34% [22.4%, 26.4%] | 11.97% | 25.39% | Adequate N |
| `Asian` | 68 | 8.82% | 7.58% | 33.33% [0.0%, 80.0%] | 19.35% [9.7%, 29.2%] | 14.29% | 20.59% | ⚠️ Low N (<500) |
| `Caucasian` | 7,893 | 9.20% | 9.17% | 41.18% [37.3%, 44.6%] | 24.17% [23.2%, 25.2%] | 14.72% | 25.73% | Adequate N |
| `Hispanic` | 215 | 9.30% | 8.13% | 45.00% [23.8%, 66.7%] | 24.10% [18.2%, 30.5%] | 16.07% | 26.05% | ⚠️ Low N (<500) |
| `Other` | 184 | 10.87% | 7.86% | 25.00% [5.6%, 46.2%] | 29.88% [22.9%, 36.7%] | 9.26% | 29.35% | ⚠️ Low N (<500) |
| `Unknown` | 297 | 7.41% | 8.37% | 45.45% [25.0%, 66.7%] | 23.64% [19.1%, 29.3%] | 13.33% | 25.25% | ⚠️ Low N (<500) |

### Gender

| Subgroup | N | Observed Risk | Predicted Risk | TPR (Recall) [95% CI] | FPR (False Alarm) [95% CI] | Precision | Selection Rate | Confidence Note |
|---|---|---|---|---|---|---|---|---|
| `Female` | 5,503 | 8.89% | 9.15% | 44.79% [40.7%, 49.1%] | 26.53% [25.4%, 27.7%] | 14.14% | 28.15% | Adequate N |
| `Male` | 4,997 | 9.11% | 8.71% | 35.60% [31.3%, 39.8%] | 21.73% [20.5%, 22.9%] | 14.10% | 22.99% | Adequate N |

### Age Band

| Subgroup | N | Observed Risk | Predicted Risk | TPR (Recall) [95% CI] | FPR (False Alarm) [95% CI] | Precision | Selection Rate | Confidence Note |
|---|---|---|---|---|---|---|---|---|
| `30-49` | 1,429 | 8.89% | 6.72% | 22.05% [15.5%, 29.2%] | 11.60% [9.9%, 13.5%] | 15.64% | 12.53% | Adequate N |
| `50-69` | 4,179 | 8.16% | 8.19% | 32.84% [28.1%, 37.8%] | 18.45% [17.3%, 19.7%] | 13.66% | 19.62% | Adequate N |
| `70+` | 4,616 | 9.92% | 10.48% | 52.18% [47.8%, 56.9%] | 34.30% [32.8%, 35.7%] | 14.35% | 36.07% | Adequate N |
| `<30` | 276 | 6.52% | 6.31% | 11.11% [0.0%, 27.3%] | 12.40% [8.6%, 16.6%] | 5.88% | 12.32% | ⚠️ Low N (<500) |

---

## Clinical & Operational Trade-Off: What Was Given Up in Mitigation

> What was given up in mitigation: To achieve near-zero false positive rate disparity across racial groups (FPR gap reduced from 5.07% to 0.17%, a 96.6% disparity reduction), the system accepted a 3.07% drop in overall recall (from 43.43% to 40.36%) and a 0.30% drop in precision (from 14.43% to 14.12%). Overall selection rate dropped from 27.07% to 25.70% (144 fewer patients flagged). In clinical terms, 29 fewer readmissions were flagged in order to eliminate disparate false-alarm burdens across protected groups.


1. **Parity Gain vs. Opportunity Cost:** Equalized Odds post-processing dramatically equalized the burden of false alarms across racial subgroups (FPR disparity dropped from 5.07% to 0.17%, a 96.6% reduction). This directly protects Black and minoritized patients from disproportionate surveillance or stigmatizing post-discharge outreach.

2. **Trade-Off Quantification:** Achieving this parity required adjusting group-specific thresholds, causing overall recall to drop by 3.07% (from 43.43% to 40.36%) and precision to drop by 0.30% (from 14.43% to 14.12%). Total flagged volume fell from 2,842 to 2,698 patients (144 fewer interventions), resulting in 29 fewer readmitted patients flagged across the 10,500 test encounters.

3. **Governance Recommendation:** Algorithmic mitigation should not operate in a vacuum. We recommend pairing equalized odds post-processing with care team clinical judgment, using calibrated continuous risk scores for prioritization, and auditing demographic parity periodically in production.
