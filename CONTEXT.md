# CONTEXT.md: Fair & Explainable 30-Day Hospital Readmission Risk

> Single source of truth for **what** we are building and **why**.
> Humans and AI agents: read this first. Read `AGENTS.md` for **how** to work in the repo.
> Status: plan locked (v1.0). Owner: Sumit Gupta. Build time-box: about 3 hours (see Sec. 18).

---

## 1. TL;DR

A patient-level machine learning project on the public UCI "Diabetes 130-US Hospitals" dataset that predicts **30-day hospital readmission at the point of discharge** and treats it as a **decision-support problem**, not a leaderboard problem.

The finished project must show, with measured numbers:

1. A leakage-aware, patient-grouped, reproducible modeling pipeline
2. An interpretable baseline (logistic regression) vs a stronger model (XGBoost), plus calibration
3. A **capacity-based operating threshold** ("the care team can follow up on the top K% of discharges")
4. Explanations (SHAP global and local, optional counterfactuals)
5. A **fairness audit with mitigation and a quantified trade-off**
6. A thin engineering layer (inference module, FastAPI endpoint, Streamlit dashboard, Docker, MLflow) so it reads as AI Engineering, not a notebook

Final title: **ReadmitIQ: 30-Day Hospital Readmission Risk Prediction with Calibration and Fairness Audit**
Repo name: `hospital-readmission-risk`

---

## 2. Why this project exists

- **Target:** the ZS Associates AI Engineering campus drive. The application form has a section "Project Details relevant to Data Science".
- **Resume slot:** this is project #3 of exactly 4 on the resume (the others are AegisVision and the speculative-decoding engine; the 4th slot is still undecided). It is the **DS-heavy** project and fills the gap left by the systems-heavy ones.
- **What screeners look for:** a real decision problem, messy-data handling, correct methodology (baseline, validation, metric fit), explainability, responsible AI, deployment, measurable results, and a clear personal contribution.

How the project maps to that:

| What a ZS-style screener wants | Where it shows up here |
|---|---|
| Real decision / business framing | Capacity-based intervention prioritization (Sec. 3, 8) |
| Messy data, leakage, imbalance | Data audit, leakage audit, cohort rules, imbalance ablation (Sec. 6, 7) |
| Sound methodology | Patient-grouped splits, baselines, calibration, bootstrap CIs (Sec. 8) |
| Explainability | SHAP, odds ratios, stability check, optional counterfactuals (Sec. 9) |
| Responsible AI | Fairness audit, mitigation, trade-off table, limitations (Sec. 10, 16) |
| AI Engineering | Package, API, dashboard, Docker, MLflow, tests (Sec. 11) |
| Communication | Executive summary page, README, one-page report (Sec. 14, 17) |

---

## 3. Problem statement and decision framing

**Question the project answers:**
> Among patients being discharged, who is most likely to be readmitted within 30 days, how reliable are those risk estimates, why does the model rate a patient as high risk, does it perform comparably across demographic groups, and how many patients can a care team realistically follow up on at a given capacity?

**Framing rules (important for the form and interviews):**

- Say "decision-support / intervention prioritization", never "the model decides treatment".
- Output is a **risk score plus an explanation**, used to prioritize care-team review.
- Never claim causal effects. Counterfactuals are *model-based what-ifs*, not interventions.
- Prediction point is **at or near discharge**. Every feature must be known to the care team at that moment.

**Label:** `readmitted == "<30"` becomes 1; `">30"` and `"NO"` become 0.

---

## 4. Scope

**In scope (Core):** data audit, leakage audit, cohort definition, grouped splits, feature engineering, baselines, logistic regression, XGBoost, imbalance ablation, calibration, metrics with CIs, capacity/threshold analysis, SHAP, fairness audit, one mitigation with before/after, model comparison, executive report, README with limitations.

**In scope (Engineering layer):** installable `readmit` package, config files, tests, MLflow tracking, inference module, FastAPI `/predict` and `/explain`, Streamlit dashboard, Dockerfile.

**Stretch (only after Core is done and verified):** counterfactuals (DiCE), reweighing as a second mitigation, decision-curve analysis, sensitivity run on all encounters, race-in/race-out ablation, GitHub Actions CI.

**Out of scope (deliberately):** deep learning, transformers, LLM/GenAI features, LIME on top of SHAP, external validation datasets, real clinical claims, hyperparameter-tuning marathons. Depth beats buzzword count.

---

## 5. Decision log (what changed vs. the earlier ChatGPT spec)

The ChatGPT spec was a good skeleton (leakage audit, calibration, capacity threshold, fairness trade-off, limitations section). These are the changes made and why.

| # | Topic | ChatGPT spec | Final decision | Reason |
|---|---|---|---|---|
| 1 | Split | Stratified random 70/15/15 | **Patient-grouped** split; primary cohort = one index encounter per patient | The dataset has repeat patients. A random encounter split leaks the same patient across train and test and inflates metrics |
| 2 | Cohort | Not specified | **Exclude encounters ending in death or hospice** (discharge_disposition_id 11, 13, 14, 19, 20, 21) | Those patients cannot be readmitted; keeping them corrupts the label |
| 3 | Temporal validation | "If there is temporal structure, add it" | **Drop it** | The dataset has no usable date field; `encounter_id` order is only an unverified proxy |
| 4 | Race and gender as features | Race used as feature and audit attribute | **Race and gender are audit-only attributes, not model features**; age is a feature and is audited | Avoids using protected attributes for prediction while still auditing outcomes. Optional ablation (stretch) |
| 5 | Mitigation | "Pick reweighing or thresholds" | **Primary: Fairlearn `ThresholdOptimizer`** on the calibrated model; reweighing is stretch | Cleanest before/after story with no retraining. Group-specific thresholds are a policy decision and must be stated as such |
| 6 | Counterfactuals | Core | **Stretch**, restricted to mutable features, labeled associational | Most features here (age, prior visits) are not actionable, so naive counterfactuals are misleading |
| 7 | Example numbers | "78% risk", "Recall 78%" | **No target numbers**; report only measured values | Prevalence is roughly 9 to 11%. Calibrated risks of 78% would be unrealistic. See Sec. 15 |
| 8 | Engineering | Streamlit only, "no extra tech" | **Add** a package, FastAPI, Docker, MLflow, tests | This is for an AI *Engineering* role, so reproducibility and serving are the differentiator. Modeling stays lean |
| 9 | Ratings ("9.2/10") | Present | **Ignored** | Not evidence. Execution quality is what counts |
| 10 | Extra baselines | Logistic only | **Add** prevalence baseline and a simple rule baseline (rank by prior inpatient visits) | Shows the model beats a trivial rule, which is what a client would ask |

---

## 6. Data

**Source:** UCI Machine Learning Repository, "Diabetes 130-US Hospitals for Years 1999-2008" (dataset id 296, CC BY 4.0). De-identified. About 101,766 encounters, about 50 columns, 130 hospitals.
**Location in repo:** `data/raw/diabetic_data.csv` (and `IDS_mapping.csv`). **Never commit data.** `data/README.md` explains how to download it.

**Key columns:** `encounter_id`, `patient_nbr`, `race`, `gender`, `age` (10-year bins), `weight`, `admission_type_id`, `discharge_disposition_id`, `admission_source_id`, `time_in_hospital`, `payer_code`, `medical_specialty`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient` (prior-year visits), `diag_1/2/3`, `number_diagnoses`, `max_glu_serum`, `A1Cresult`, about 23 medication columns, `change`, `diabetesMed`, `readmitted`.

**Known quirks to handle (verify each in the data audit, do not assume):**

- Missing values are coded `?`. `weight` is almost entirely missing and is dropped. `payer_code` and `medical_specialty` are heavily missing and become an explicit `Unknown` category.
- `race` has a small `?` share. `gender` has a handful of `Unknown/Invalid` rows.
- Repeat patients: the same `patient_nbr` appears many times.
- `diag_1/2/3` are ICD-9 codes: group into about 9 clinical categories (circulatory, respiratory, digestive, diabetes, injury, musculoskeletal, genitourinary, neoplasms, other).
- Medication columns are mostly `No`; summarize as counts (number prescribed, number changed up or down, number steady).
- Class imbalance: positive class is roughly 9 to 11% depending on cohort rules.

**Cohort definition (primary analysis):**

1. Remove encounters with discharge to death or hospice (ids 11, 13, 14, 19, 20, 21).
2. Remove gender `Unknown/Invalid` rows from the audit only if they cannot be assigned; log counts.
3. Keep **one index encounter per patient** (lowest `encounter_id`, assumed approximately chronological; state this assumption).
4. Log row counts after every step in `reports/cohort_flow.md`.

Sensitivity run (stretch): all eligible encounters with a patient-grouped split.

---

## 7. Pipeline

```
Raw CSV
  -> Data-quality audit            (reports/data_quality.md)
  -> Leakage audit                 (reports/leakage_audit.md)
  -> Cohort definition + counts    (reports/cohort_flow.md)
  -> Patient-grouped split 70/15/15, saved ID lists (data/splits/)
  -> Feature engineering           (sklearn Pipeline, fit on TRAIN only)
  -> Baselines: prevalence, rule-based (prior inpatient visits)
  -> Logistic regression           (class_weight, L2)
  -> Imbalance ablation: none vs class weights vs SMOTE (train only)
  -> XGBoost                       (early stopping via inner split of TRAIN)
  -> Calibration                   (sigmoid vs isotonic, fit on VALIDATION)
  -> Model comparison              (VALIDATION) -> choose final model
  -> Capacity / threshold analysis (VALIDATION chooses, TEST reports once)
  -> SHAP + odds ratios (+ counterfactuals, stretch)
  -> Fairness audit (race, gender, age band) -> mitigation -> before/after
  -> Decision framework + executive report
  -> Inference module -> FastAPI + Streamlit -> Docker
```

**Leakage audit categories** (document every column in one of these):

- **A. Known at discharge, safe:** demographics (age), admission type and source, time in hospital, counts of labs, procedures, medications, prior-year visits, diagnoses, medication changes, discharge disposition (discharge disposition is known at discharge).
- **B. Investigate:** anything that could encode later utilization or the outcome. Decide keep/drop with a written reason.
- **C. Not available at prediction time:** drop. Includes the target, `encounter_id`, `patient_nbr` (identifiers are never features).

Test for every feature: *"Would the care team know this at discharge?"*

---

## 8. Modeling and evaluation spec

**Splits.** 70% train, 15% validation, 15% test, stratified on the label and **grouped by patient**. Save patient ID lists so splits are reproducible. Fit all preprocessing on train only.

**Role of each split.**

- Train: fit models, inner holdout for XGBoost early stopping.
- Validation: fit the calibrator, choose the model, choose the threshold or capacity.
- Test: touched **once**, at the end, to report final numbers.

**Models.**

| Model | Purpose |
|---|---|
| Prevalence baseline | Floor for every metric |
| Rule baseline (rank by prior inpatient visits) | "Does ML beat a simple rule?" |
| Logistic regression (L2, class weights) | Interpretable statistical baseline; report odds ratios |
| XGBoost | Nonlinear model; modest tuning (depth, learning rate, trees, subsample, colsample, min child weight) |
| Calibrated best model | Reliable probabilities for risk-based decisions |

**Imbalance ablation.** Compare no handling, class weights / `scale_pos_weight`, and SMOTE (train only, inside a pipeline). Judge by PR-AUC, recall at capacity, **and calibration**. Any reweighting distorts probabilities, so recalibrate afterwards.

**Metrics (primary first).**

1. PR-AUC (average precision), always shown against prevalence
2. Recall and precision at capacity K% (K = 5, 10, 20), and **lift** over prevalence
3. Brier score, calibration curve, expected calibration error
4. ROC-AUC (secondary, never the headline)
5. F1 only as a secondary number at the chosen threshold

All headline test metrics get **95% bootstrap confidence intervals** (1,000 resamples, seeded).

**Threshold and capacity (the decision layer).**

- Primary: **capacity-based**. If the care team can follow up on K% of discharges, flag the top K% by risk. Report recall, precision, lift, and number flagged.
- Secondary: **cost-ratio based**. Pick the threshold minimizing expected cost with an explicit, configurable false-negative to false-positive cost ratio. This ratio is an **assumption** and must be labeled as such in every table and in the README.
- Never default to 0.5.

**Model selection rule.** Choose by validation PR-AUC and calibration jointly. The winner does not have to be XGBoost; if calibrated logistic regression is within noise of XGBoost, prefer it and say why.

---

## 9. Explainability

- **Logistic regression:** odds ratios with confidence intervals.
- **XGBoost:** SHAP `TreeExplainer`. Global bar and beeswarm plots, 3 dependence plots, and 3 local waterfalls (one true positive, one false negative, one low-risk patient). Note that calibration is monotonic, so it does not change driver ranking.
- **Stability check:** Spearman rank correlation of mean |SHAP| across bootstrap resamples.
- **Sanity check:** do LR odds ratios and SHAP drivers broadly agree? Discuss disagreements.
- **Counterfactuals (stretch):** DiCE or a custom search restricted to **mutable** features (for example medication-change flag, A1C testing, discharge disposition). Wording must be: "the model associates this change with lower predicted risk", never "this change will prevent readmission".

---

## 10. Fairness

**Attributes audited:** race (Caucasian, AfricanAmerican, Hispanic, Asian, Other; `Unknown` reported but excluded from gaps), gender (Male, Female), age band (<30, 30-49, 50-69, 70+). Race and gender are **not** model inputs.

**Per-group metrics (Fairlearn `MetricFrame`):** recall/TPR, false positive rate, precision, selection rate, mean predicted vs observed risk (calibration), PR-AUC.

**Gap reporting:** TPR gap, FPR gap, precision gap, selection-rate ratio (max vs min across groups), each with bootstrap CIs. Flag any group with n < 500 as low-confidence; do not over-interpret it.

**Mitigation (primary):** Fairlearn `ThresholdOptimizer` on the calibrated model, constraint chosen and justified (equalized odds or TPR parity). Document that it needs the sensitive attribute at decision time and that group-specific thresholds are a policy and legal question, not just a technical one.
**Stretch:** reweighing (training-time) as a comparison.

**Mandatory before/after table:**

| Metric | Before | After |
|---|---|---|
| PR-AUC / recall at capacity | | |
| Flagged rate | | |
| TPR gap | | |
| FPR gap | | |
| Calibration by group | | |

**Discussion must cover:** what performance was given up, why calibration parity and error-rate parity can conflict when base rates differ, and why a fairness metric depends on the chosen groups and thresholds. Never write "the model is fair"; write what was measured.

---

## 11. Architecture

```
                 OFFLINE (reproducible)                              ONLINE (demo)
 +-----------+   +-----------------+   +--------------+     +---------------------+
 | raw CSV   |-->| readmit.data    |-->| readmit.     |     | Streamlit dashboard |
 | (not in   |   | audit, cohort,  |   | features     |     |  - exec summary     |
 |  git)     |   | splits          |   | (sklearn     |     |  - patient risk     |
 +-----------+   +-----------------+   |  Pipeline)   |     |  - SHAP             |
                                       +------+-------+     |  - fairness         |
                                              v             |  - model comparison |
                      +-----------------------+---------+   +----------+----------+
                      | readmit.models / calibration /  |              |
                      | evaluation / explain / fairness |              v
                      +-----------------------+---------+     +--------+--------+
                                              |               | FastAPI service |
                          +-------------------+------+        | /health         |
                          v                          v        | /predict        |
                  +---------------+          +---------------+| /explain        |
                  | MLflow runs   |          | artifacts/    |+--------+--------+
                  | params,metrics|          | model, calib, |         ^
                  +---------------+          | metrics.json, |---------+
                                             | figures       |  readmit.inference
                                             +---------------+  (single loader)
```

Design rules:

- All logic lives in `src/readmit/`. Notebooks only explore and plot.
- The dashboard and API both call `readmit.inference`, so there is **one** prediction code path.
- Every number shown anywhere comes from `reports/metrics.json` written by the pipeline.
- The API ships in a Docker image; training runs from `make` targets with fixed seeds and config files.

---

## 12. Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| Data | pandas, numpy, pyarrow |
| Modeling | scikit-learn, xgboost, imbalanced-learn (SMOTE ablation only) |
| Calibration and metrics | scikit-learn calibration, custom ECE and bootstrap helpers |
| Explainability | shap; dice-ml (stretch) |
| Fairness | fairlearn |
| Tuning | small randomized search or Optuna (keep it modest) |
| Visualization | matplotlib, seaborn, plotly |
| Tracking | MLflow (local file store) |
| Serving | FastAPI, uvicorn, pydantic |
| Dashboard | Streamlit |
| Quality | pytest, ruff, pre-commit, nbstripout |
| Packaging | pyproject.toml, `pip install -e ".[dev]"`, Makefile |
| Containers | Docker (API image) |
| VCS | Git + GitHub |

---

## 13. Repository structure

```
hospital-readmission-risk/
├── CONTEXT.md
├── AGENTS.md
├── README.md                     # public-facing summary, built from metrics.json
├── pyproject.toml
├── Makefile
├── Dockerfile
├── configs/
│   ├── base.yaml                 # seeds, paths, split ratios, capacity K, cost ratio
│   ├── features.yaml             # column roles: drop / numeric / categorical / audit-only
│   └── models.yaml
├── data/
│   ├── README.md                 # download instructions only
│   ├── raw/                      # gitignored
│   ├── processed/                # gitignored
│   └── splits/                   # patient-ID lists (small JSON, committed)
├── notebooks/                    # exploration only; outputs stripped
│   ├── 01_data_audit.ipynb
│   ├── 02_eda.ipynb
│   └── 03_results_walkthrough.ipynb
├── src/readmit/
│   ├── data.py                   # load, clean, cohort, splits
│   ├── audit.py                  # data-quality and leakage report generators
│   ├── features.py               # ICD grouping, counts, ColumnTransformer
│   ├── models.py                 # baselines, LR, XGBoost, training entrypoints
│   ├── calibration.py
│   ├── evaluation.py             # metrics, bootstrap CIs, capacity tables
│   ├── explain.py                # SHAP, odds ratios, stability
│   ├── fairness.py               # MetricFrame wrappers, gaps, mitigation
│   ├── inference.py              # load artifacts, predict, explain one patient
│   └── cli.py                    # make-target entrypoints
├── api/main.py                   # FastAPI app
├── app/streamlit_app.py
├── tests/
├── artifacts/                    # trained models (gitignored, small ones optional)
├── reports/                      # metrics.json, figures, audits, executive_report.md
└── docs/decisions.md             # running log of assumptions and deviations
```

---

## 14. Deliverables by tier

**Minimum (must exist for the form):** audits, grouped splits, LR, XGBoost, calibration, PR-AUC, recall at capacity, SHAP, fairness audit, README with limitations, GitHub repo.

**Strong (target):** the above plus imbalance ablation, bootstrap CIs, capacity and cost-ratio thresholds, mitigation with before/after, model comparison table, executive report, Streamlit dashboard, FastAPI, Docker, MLflow, tests.

**Exceptional (only if time remains):** counterfactuals, reweighing comparison, decision-curve analysis, all-encounters sensitivity run, race-in/race-out ablation, CI workflow.

Rule: a rigorous Strong-tier project beats a half-working Exceptional one.

**Dashboard pages:** (1) Executive summary, (2) Patient risk with risk shown relative to average prevalence, (3) SHAP explanation, (4) Fairness by group, (5) Model comparison and capacity table.

---

## 15. Realistic expectations on results

From general knowledge of this dataset (verify against your own run):

- Prevalence of 30-day readmission is roughly 9 to 11%.
- Published models on this data usually reach **ROC-AUC around 0.63 to 0.70** and PR-AUC well below 0.5. This is a hard, noisy problem.
- A calibrated top-20% flag typically gives a modest lift over prevalence, not dramatic recall.
- Calibrated individual risks will rarely be very high. In the UI, show "X times the average risk" next to the percentage.
- **If you see ROC-AUC above about 0.80, suspect leakage first** (a patient in two splits, a post-discharge feature, a target-derived column).

An honest, well-calibrated, well-explained result with correct caveats is the goal. Do not tune toward impressive numbers, and do not quote any number that is not in `reports/metrics.json`.

---

## 16. Limitations and ethics (must appear in the README)

- Historical US data (1999-2008) from diabetic inpatients; may not reflect current practice or other populations.
- Prediction is not causation; counterfactuals are associational.
- Fairness results depend on the groups, metric, and threshold chosen; small groups have wide uncertainty.
- Group-specific thresholds raise policy and legal questions that a technical project cannot settle.
- The model supports prioritization and must not independently determine care.
- No external validation; real deployment would need local validation, monitoring, and clinical governance.
- Data is de-identified and public; never attempt re-identification; never commit data.

---

## 17. How to present it

**Form entry template (fill only with measured values, trim to the character limit):**

- **Title:** ReadmitIQ: 30-Day Hospital Readmission Risk Prediction with Calibration and Fairness Audit
- **Problem:** Prioritize discharged patients for follow-up under limited care-team capacity, with reliable, explainable, and fairness-audited risk estimates.
- **Approach:** Leakage-audited, patient-grouped pipeline on ~[N] encounters; logistic regression vs XGBoost with calibration; capacity-based threshold; SHAP; fairness audit with threshold-based mitigation.
- **Results:** PR-AUC [x] (prevalence [p]); flagging top [K]% captures [r]% of readmissions ([lift]x lift); TPR gap [a] to [b] after mitigation with [c] change in recall.
- **Deployment:** FastAPI + Streamlit dashboard, Dockerized, MLflow-tracked. GitHub: [link].
- **Role and stack:** Sole developer. Python, pandas, scikit-learn, XGBoost, SHAP, Fairlearn, FastAPI, Streamlit, Docker, MLflow.

**Resume bullets (templates):**

- Built a leakage-audited, patient-grouped 30-day readmission pipeline on [N] diabetic-encounter records; calibrated XGBoost reached PR-AUC [x] vs [y] for logistic baseline.
- Designed a capacity-based risk-prioritization layer: flagging the top [K]% identifies [r]% of readmissions ([lift]x lift).
- Audited performance across race, gender, and age; threshold-based mitigation cut TPR gap from [a] to [b] at a [c] recall cost.
- Shipped as a Dockerized FastAPI service and Streamlit dashboard with SHAP explanations and MLflow tracking.

**60-second interview story skeleton:** problem and decision, data and leakage choices (grouped split, hospice exclusion), baseline to model to calibration, why capacity threshold instead of 0.5, what SHAP showed, fairness findings and the trade-off, limitations, what you would do next.

---

## 18. Milestones (ordered, with exit criteria)

| # | Milestone | Exit criteria |
|---|---|---|
| M0 | Scaffold | Repo, `pyproject.toml`, Makefile, configs, pre-commit, empty tests run green |
| M1 | Data audit and cohort | `data_quality.md` and `cohort_flow.md` generated; counts logged at every step |
| M2 | Leakage audit and splits | Every column classified A/B/C with reasons; saved splits; test proves zero patient overlap |
| M3 | Features and baselines | Feature pipeline fit on train only; prevalence and rule baselines reported |
| M4 | LR and XGBoost, imbalance ablation | Comparison table on validation; ablation conclusion written |
| M5 | Calibration and thresholds | Calibration curves, Brier/ECE, capacity and cost-ratio tables |
| M6 | Explainability | SHAP global/local, stability score, odds ratios |
| M7 | Fairness | Audit tables with CIs; mitigation; before/after trade-off table |
| M8 | Final test evaluation | Test set scored once; `metrics.json` frozen with bootstrap CIs |
| M9 | Engineering layer | Inference module, API, dashboard, Docker image builds and runs |
| M10 | Write-up | README, executive report, form entry, resume bullets, all numbers traced to `metrics.json` |

### Build plan (3-hour time-box; the agent builds everything in priority order)

The agent builds the whole project autonomously (see `AGENTS.md` Sec. 0). Work is ordered so the project is **usable if stopped at any point**: finish each tier before starting the next.

| Tier | Contents | Milestones |
|---|---|---|
| **P0 Core** | Scaffold, data audit, cohort, leakage audit, grouped splits, features, baselines, LR, XGBoost, calibration, capacity table, SHAP, fairness audit and mitigation, final test scoring, README with limitations | M0 to M8, M10 |
| **P1 Demo** | Inference module, Streamlit dashboard, executive report | M9 (part 1) |
| **P2 Engineering** | FastAPI, Dockerfile, MLflow logging polish, extra tests, ruff and pre-commit | M9 (part 2) |
| **P3 Stretch** | Counterfactuals, reweighing comparison, decision-curve analysis, all-encounters sensitivity run, race-in/race-out ablation | Only if P0 to P2 are done and verified |

Rough time guide for P0 (keep pace, but never skip an invariant):

| Time | Work |
|---|---|
| 0:00 to 0:15 | M0: package, configs, CLI, tracking wrapper (MLflow if installed, otherwise a no-op), test skeleton |
| 0:15 to 0:45 | M1 and M2: data download, audits, cohort, leakage audit, grouped splits (checkpoint 1) |
| 0:45 to 1:30 | M3 to M5: features, baselines, LR, XGBoost, imbalance ablation, calibration, capacity table (checkpoint 2) |
| 1:30 to 2:00 | M6: SHAP global and 3 local examples, odds ratios |
| 2:00 to 2:30 | M7: fairness audit, `ThresholdOptimizer`, before/after table (checkpoint 3) |
| 2:30 to 2:45 | M8: score the test set **once**, freeze `metrics.json` |
| 2:45 to 3:00 | M10: README, executive report, `FINAL_SUMMARY.md` |

P1 and P2 (M9) run after M8 and may extend past the 3 hours if the session continues. **P0 alone is enough for the application form.** On the form, describe only what actually exists.

**Checkpoints** (after M2, M5, M7): the agent writes a summary to `reports/checkpoints.md` and **continues**, unless a red flag from `AGENTS.md` Sec. 0 fires. The owner should read the leakage audit, the split code, and the fairness output personally before making any claim on the form, because they must be defensible in an interview.

**Never cut:** leakage audit, grouped split, calibration, capacity table, SHAP, fairness before/after, limitations section.

**If time truly runs out, drop in this order:** P3, then P2 items (Docker, FastAPI, MLflow polish), then SMOTE and the stability check, then Streamlit polish.

**Data access:** the machine has internet, so the agent downloads the dataset itself (UCI dataset id 296) into `data/raw/` as part of `data` step M1, then verifies that `diabetic_data.csv` and `IDS_mapping.csv` exist and that the row and column counts look right (about 101,766 rows and about 50 columns; record the actual numbers). Install dependencies in M0 (`pandas`, `scikit-learn`, `xgboost`, `shap`, `fairlearn`, `matplotlib`, `seaborn`, `pytest`, `pyyaml`).

**On the form, only describe what exists.** If the API, Docker, or dashboard are not built, do not list them.

---

## 19. Assumptions and items to verify

- `encounter_id` is only an approximate chronological proxy; the "first encounter" choice is an assumption.
- Exact cohort counts, prevalence, missingness rates, and group sizes are **not** hardcoded anywhere; read them from the data audit.
- ICD-9 grouping follows the standard clinical categories; document any deviation.
- Cost ratio and capacity K are assumptions; keep them in `configs/base.yaml` and label them everywhere they appear.
- Confirm the dataset license and citation text from the UCI page before publishing the repo.

---

## 20. Glossary

- **PR-AUC:** area under the precision-recall curve; informative when the positive class is rare.
- **Lift:** precision among flagged patients divided by overall prevalence.
- **Capacity (K%):** share of discharges the care team can follow up on.
- **Calibration:** whether predicted probabilities match observed frequencies.
- **ECE:** expected calibration error, the average gap between predicted and observed risk across probability bins.
- **Leakage:** information in training features that would not be available at prediction time, or that overlaps train and test.
- **TPR / FPR:** true and false positive rates; the basis of equalized-odds fairness.
- **Equalized odds:** equal TPR and FPR across groups.
- **Selection rate:** fraction of a group flagged as positive.
- **SHAP:** additive feature-attribution method; here applied to the tree model.
- **Counterfactual:** a minimally changed input that flips the model's prediction; associational, not causal.

---

## Appendix A: ICD-9 diagnosis grouping

Apply to `diag_1`, `diag_2`, `diag_3`. Values are strings: strip whitespace, treat `?` or empty as `Unknown`, treat codes starting with `V` or `E` as `Other`, otherwise parse as a float. Grouping follows the widely used scheme from the original study on this dataset (Strack et al., 2014); verify and record any deviation.

| Group | ICD-9 codes |
|---|---|
| Circulatory | 390 to 459, and 785 |
| Respiratory | 460 to 519, and 786 |
| Digestive | 520 to 579, and 787 |
| Diabetes | 250.xx |
| Injury | 800 to 999 |
| Musculoskeletal | 710 to 739 |
| Genitourinary | 580 to 629, and 788 |
| Neoplasms | 140 to 239 |
| Other | Everything else (V and E codes; 001 to 139; 240 to 279 except 250; 680 to 709; 780 to 784; 790 to 799) |
| Unknown | Missing |

---

## Appendix B: Feature specification

Record the final decisions in `configs/features.yaml` and `reports/leakage_audit.md`. Verify each assumption against the data.

| Role | Columns |
|---|---|
| Identifier, split only (never a feature) | `encounter_id`, `patient_nbr` |
| Target | `readmitted` -> `readmit_30d` |
| Audit-only (not model inputs) | `race` (-> `race_group`), `gender`; `age_band` is derived from `age` for auditing |
| Drop | `weight` (almost entirely missing); constant columns such as `examide` and `citoglipton` (verify they are constant) |
| Numeric | `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses`, `age_midpoint` (from the 10-year `age` bin) |
| Engineered | `prior_visits_total` (outpatient + emergency + inpatient); `any_prior_inpatient`; `n_meds_active` (medication columns not `No`); `n_meds_changed` (`Up` or `Down`); `n_meds_steady`; `a1c_tested` and `a1c_result_cat`; `glu_tested` and `glu_serum_cat`; `diag_1_group`, `diag_2_group`, `diag_3_group`; `n_distinct_diag_groups` |
| Categorical (grouped) | `admission_type_group`, `admission_source_group`, `discharge_group` (home, home health, SNF/other facility, other; built from `IDS_mapping.csv`); `medical_specialty_group` (top 10 by train frequency, `Other`, `Unknown`); `payer_code_group` (top categories, `Other`, `Unknown`) |
| Binary | `change`, `diabetesMed` |

Rules: fit category vocabularies and "top N" lists on **train only**; handle unseen categories safely at inference; keep a feature dictionary (`reports/feature_dictionary.md`) that explains every final feature in one line. `payer_code` can proxy socioeconomic status; keep it but note this in the limitations and audit its effect if time allows.

---

## Appendix C: Dependencies and packaging

Python 3.11. Declare in `pyproject.toml` (loosen pins if there are conflicts and log the change).

- **Runtime:** pandas, numpy, scipy, pyarrow, scikit-learn (1.4 or newer), xgboost (2.0 or newer), shap, fairlearn (0.10 or newer), matplotlib, seaborn, plotly, pyyaml, joblib, ucimlrepo, streamlit, fastapi, uvicorn[standard], pydantic (2.x), mlflow
- **Optional extras:** imbalanced-learn (SMOTE ablation), optuna (tuning), dice-ml (counterfactuals, P3)
- **Dev:** pytest, httpx, ruff, pre-commit, nbstripout, jupyter

Package name `readmit`, `src` layout, installed with `pip install -e ".[dev]"`. CLI entrypoint: `python -m readmit.cli <data|features|train|evaluate|explain|fairness|final|app|api>`. Each `make` target calls the matching CLI command.

---

## Appendix D: Repo hygiene, container, and serving

**`.gitignore` must include:** `data/raw/`, `data/processed/`, `artifacts/`, `mlruns/`, `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `.env`, `*.joblib`, `.pytest_cache/`, `.ruff_cache/`. Commit `data/splits/` (small ID lists) and `reports/`.

**Dockerfile (API image):** base `python:3.11-slim`; install system build essentials only if needed; copy `pyproject.toml` and `src/` first for layer caching; `pip install .`; copy `api/` and the final model artifact; expose 8000; `CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]`. Add `.dockerignore` for data, notebooks, `mlruns/`, and tests. Provide a health check against `/health`. Build and run it once to verify before claiming it works.

**API behavior:** validated input (pydantic), graceful handling of unseen categories, no patient data written to logs, `risk_vs_average` computed against train prevalence. **Dashboard:** five pages (executive summary, patient risk, SHAP, fairness, model comparison and capacity), with assumptions and limitations in the sidebar, and every number loaded from `reports/metrics.json`.

---

## Appendix E: Report and README outlines

**README.md sections:** Title, one-paragraph summary, Business problem and decision framing, Dataset and cohort (with counts from `cohort_flow.md`), Prediction point and leakage strategy, Methods (splits, features, models, imbalance, calibration), Results (model comparison table, calibration plot, capacity table, SHAP highlights), Fairness audit and mitigation (before/after table), Limitations and ethics, How to run, Repository structure, Reproducibility notes, Citation and license for the dataset.

**`reports/executive_report.md` (one to two pages):** the question, headline result with CIs, recommended model and operating capacity, what a care team would do with it, top risk drivers, fairness findings and the trade-off, key caveats, and recommended next steps. Plain language, no jargon without a gloss.

**`reports/FINAL_SUMMARY.md`:** see `AGENTS.md` Sec. 14.
