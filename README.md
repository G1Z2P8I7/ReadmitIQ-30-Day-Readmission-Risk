# ReadmitIQ: 30-Day Hospital Readmission Risk Prediction with Calibration and Fairness Audit

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-blueviolet.svg)](https://docs.pytest.org/)

An end-to-end, production-ready clinical decision support system predicting 30-day hospital readmission risk on the **UCI Diabetes 130-US Hospitals dataset** (1999–2008). 

ReadmitIQ rejects uncalibrated scores and arbitrary 0.50 classification thresholds in favor of **capacity-aware clinical resource prioritization**, rigorous patient-grouped split isolation, post-hoc probability calibration, SHAP attribution stability, and demographic fairness mitigation.

---

## 🌟 Key Highlights & Rigor

- **Strict Patient Grouping (0 Split Leakage):** 69,987 index admissions split into Train (70%), Validation (15%), and Test (15%) with zero patient overlap across splits.
- **Discharge Prediction Horizon:** All features strictly reflect data known at discharge; all target-adjacent and future variables dropped.
- **Reliable Probability Calibration:** Post-hoc Isotonic calibration yields reliable risk estimates (Brier: **0.0806**, ECE: **0.0062** on test set).
- **Capacity-Based Screening (K=20%):** In a population with **8.99% readmission prevalence**, prioritizing the top 20% highest-risk patients captures **34.53% of all 30-day readmissions** (**1.73x lift** over hospital average).
- **SHAP Attribution Stability:** Global feature importance achieves a **0.9909** Spearman rank correlation across bootstrap resamples.
- **Algorithmic Fairness Audit & Mitigation:** Evaluated across Race, Gender, and Age. Fairlearn `ThresholdOptimizer(equalized_odds)` post-processing reduced the racial False Positive Rate disparity gap from **5.07% to 0.17%** (a **96.6% disparity reduction**).
- **Strict Test-Set Lock:** Test set scored strictly once via `python -m readmit.cli final` and locked with `artifacts/.final_done`.

---

## 📊 Key Results (Frozen from `reports/metrics.json`)

### Model Discrimination & Calibration (Test Set: N = 10,500)

| Model Architecture | PR-AUC (95% Bootstrap CI) | ROC-AUC (95% Bootstrap CI) | Brier Score (95% CI) | Expected Calibration Error | Recall @ K=20% Capacity | Precision @ K=20% | Lift @ K=20% |
|---|---|---|---|---|---|---|---|
| **Prevalence Baseline** | 0.0899 (fixed) | 0.5000 (fixed) | 0.0818 | 0.0000 | 20.00% | 8.99% | 1.00x |
| **Clinical Rule Baseline** | 0.1201 | 0.5452 | 0.1691 | 0.1118 | 27.92% | 12.52% | 1.40x |
| **Logistic Regression (Calibrated)** | 0.1351 [0.1232 - 0.1504] | 0.6208 [0.6025 - 0.6388] | 0.0808 [0.0765 - 0.0850] | 0.0086 | 33.26% | 14.95% | 1.66x |
| **XGBoost (Calibrated Isotonic - Primary)** | **0.1385 [0.1268 - 0.1529]** | **0.6344 [0.6168 - 0.6532]** | **0.0806 [0.0762 - 0.0848]** | **0.0062** | **34.53%** | **15.52%** | **1.73x** |

### Capacity-Constrained Resource Prioritization (Test Set)

| Capacity Tier (Top K%) | Patients Flagged | Risk Cutoff Threshold | Captured Readmissions | Recall (Sensitivity) | Precision (PPV) | Lift over Baseline |
|---|---|---|---|---|---|---|
| **Top 5%** | 525 | 20.81% | 101 | 10.70% | 19.24% | **2.14x** |
| **Top 10%** | 1,050 | 15.38% | 189 | 20.02% | 18.00% | **2.00x** |
| **Top 20% (Primary Tier)** | 2,100 | 10.77% | 326 | 34.53% | 15.52% | **1.73x** |

### Algorithmic Fairness Audit & Mitigation (Race / Ethnicity on Test Set)

| State | TPR Gap | FPR (False Alarm) Gap | African American FPR | Caucasian FPR | Selection Rate Gap |
|---|---|---|---|---|---|
| **Before Mitigation** (Capacity K=20%) | 6.10% | 5.07% | 21.32% | 26.40% | 5.35% |
| **After Mitigation** (Equalized Odds) | 3.85% | **0.17%** | 23.36% | 23.53% | **0.34%** |

---

## 🛠️ Repository Architecture

```text
├── configs/            # base.yaml, features.yaml, models.yaml
├── data/
│   ├── raw/            # UCI raw CSVs (gitignored)
│   ├── processed/      # Parquet train/val/test splits (gitignored)
│   └── splits/         # Deterministic patient-ID JSON files (committed)
├── src/readmit/        # Production Python package
│   ├── data.py         # Cohort construction & patient-grouped splitting
│   ├── audit.py        # Clinical data quality & leakage audit reports
│   ├── features.py     # ICD-9 groupings, clinical transformations, pipelines
│   ├── models.py       # Model training, early stopping, imbalance ablation
│   ├── calibration.py  # Platt & Isotonic calibrators, FullCalibratedPipeline
│   ├── evaluation.py   # PR-AUC, ECE, capacity metrics, bootstrap CIs
│   ├── explain.py      # SHAP TreeExplainer, stability check, odds ratios
│   ├── fairness.py     # MetricFrame demographic audit & ThresholdOptimizer
│   ├── inference.py    # Cached prediction & explanation engine
│   └── cli.py          # Unified CLI dispatcher
├── api/                # FastAPI REST service (GET /health, POST /predict, /explain)
├── app/                # Multi-page Streamlit decision support dashboard
├── artifacts/          # Serialized models & test-set lock .final_done (gitignored)
├── reports/            # metrics.json, executive_report.md, figures/
├── tests/              # 15 automated pytest unit tests
└── Dockerfile          # Multi-stage production container definition
```

---

## 🚀 Quickstart & Execution

```bash
# 1. Environment Setup
git clone https://github.com/your-username/readmitiq.git
cd readmitiq
python -m venv .venv
.venv\Scripts\activate   # Windows (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"

# 2. Run Pipeline Steps
python -m readmit.cli data       # Ingest UCI dataset, build cohort & splits
python -m readmit.cli features   # Fit clinical feature transformations & baselines
python -m readmit.cli train      # Train models with imbalance ablation
python -m readmit.cli evaluate   # Fit post-hoc calibrators & generate capacity table
python -m readmit.cli explain    # Generate SHAP beeswarm & case waterfalls
python -m readmit.cli fairness   # Audit demographic parity & apply Equalized Odds
python -m readmit.cli final      # Score test set once and lock metrics.json

# 3. Launch Applications
python -m readmit.cli app        # Streamlit Dashboard (http://localhost:8501)
python -m readmit.cli api        # FastAPI Service (http://localhost:8000/docs)

# 4. Run Test Suite & Linting
pytest -v                        # 15 tests verifying all invariants
ruff check src tests api app     # Enforce code quality and formatting
```

---

## 🔒 Governance & Ethical Disclaimers

1. **Non-Causal Decision Support:** Model risk attributions are observational and indicate statistical correlations; they do not dictate clinical treatment.
2. **Operational Tuning:** Capacity tiers (Top 20%) and cost ratios (5:1) are illustrative operational assumptions and must be calibrated to specific hospital staffing realities.
3. **Protected Attributes:** Race and gender are strictly excluded from predictive features and tracked purely for demographic parity audit.
