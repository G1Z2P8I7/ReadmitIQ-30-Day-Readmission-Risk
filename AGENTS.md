# AGENTS.md

Instructions for AI coding agents (Claude Code, Codex, Cursor, Copilot, and similar) and for human contributors working in this repository.

**Read `CONTEXT.md` first** for the what and why (scope, decisions, data, pipeline). This file covers **how to work here**. If the two ever conflict, `CONTEXT.md` wins on scope and decisions; this file wins on process and conventions. If something is unclear, write your assumption in `docs/decisions.md` and continue; do not silently improvise.

---

## 0. Kickoff and execution protocol

**This repository is built end-to-end by the agent.** The owner expects you to build everything described in `CONTEXT.md` (milestones M0 to M10) autonomously, in priority order, within a session of roughly 3 hours. Be efficient, but never at the cost of the non-negotiable rules in Section 4.

**On start**
1. Read `CONTEXT.md` and this file completely.
2. Create `docs/decisions.md` and `reports/checkpoints.md`.
3. Post a short plan (milestone order and the priority tiers P0 to P3 from `CONTEXT.md` Sec. 18). Do not wait for approval; start immediately.
4. Execute M0 to M10 in order. Finish P0 first, then P1, then P2. Start P3 (stretch) only when everything else is done and verified.
5. After every milestone: run lint and tests, run that milestone's CLI step, and append a block to `reports/checkpoints.md` containing what was built, the commands run with a short result, key numbers read from generated files, assumptions made, and open issues.
6. Keep going between milestones without waiting for the owner.
7. When done, write `reports/FINAL_SUMMARY.md` (Sec. 14) and print its path.

**Stop and ask the owner only on a red flag**
- The patient-overlap test or the forbidden-feature test fails and the fix is not obvious.
- Validation or test ROC-AUC is above 0.80 (assume leakage until proven otherwise).
- The dataset download or verification fails.
- A step would require using the test set for any decision, or re-running the final scoring.
- You would need to change the cohort rules, split design, or fairness approach defined in `CONTEXT.md`.

**Verification discipline**
- Never claim a step is done, a test passes, or a number is correct without having run it. Report real command results.
- If something fails, fix it or log it honestly in `reports/checkpoints.md`; do not hide or paper over failures.
- For anything touching the cohort, splits, features, or evaluation, write a brief plan in `reports/checkpoints.md` before editing.
- Adding a dependency is allowed if it is listed in `CONTEXT.md` Appendix C or genuinely necessary; log why in `docs/decisions.md`.

---

## 1. Project snapshot

- **What:** 30-day hospital readmission risk prediction on the UCI Diabetes 130-US Hospitals dataset, framed as capacity-aware decision support with calibration, SHAP explainability, and a fairness audit with mitigation.
- **Why:** portfolio project for a data science / AI engineering application (ZS Associates). Rigor and honesty matter more than flashy numbers.
- **Package:** `src/readmit/`. **Python:** 3.11. **Layout:** see Section 3.
- **Primary metric:** PR-AUC, plus recall/precision/lift at capacity K%. Accuracy is never a headline metric.

---

## 2. Setup and commands

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate     Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

Expected `make` targets (create them while scaffolding if missing, then keep them working):

| Command | What it does |
|---|---|
| `make data` | Load raw CSV, run audits, build cohort, write splits |
| `make features` | Fit feature pipeline on train, write processed data |
| `make train` | Train baselines, LR, XGBoost, calibration; log to MLflow |
| `make evaluate` | Validation comparison, capacity/threshold tables, bootstrap CIs |
| `make explain` | SHAP, odds ratios, stability check |
| `make fairness` | Fairness audit, mitigation, before/after table |
| `make final` | Score the **test set once** and freeze `reports/metrics.json` |
| `make app` | Run the Streamlit dashboard |
| `make api` | Run the FastAPI service |
| `make test` | Run pytest |
| `make lint` | Run ruff (lint and format check) |
| `make all` | Everything above in order |

**Windows note:** `make` may not be installed. Implement `python -m readmit.cli <target>` first (for example `python -m readmit.cli data`) and treat the CLI as the source of truth; each `make` target is only a thin alias for it. Use `pathlib` and avoid shell-specific syntax so everything runs on Windows, Linux, and macOS.

The dataset is **not** in git. If `data/raw/diabetic_data.csv` and `data/raw/IDS_mapping.csv` are missing, download the UCI "Diabetes 130-US Hospitals for Years 1999-2008" dataset (id 296) yourself, using the `ucimlrepo` package or the zip from the dataset page, and document the method in `data/README.md`. Verify both files exist and log the row and column counts. If the download fails, stop and tell the user; never fabricate or simulate the data.

---

## 3. Repository map

```
configs/        base.yaml (seeds, paths, splits, K, cost ratio), features.yaml, models.yaml
data/           raw/ and processed/ are gitignored; splits/ holds patient-ID JSON (committed)
notebooks/      exploration only; outputs stripped
src/readmit/    data, audit, features, models, calibration, evaluation, explain, fairness, inference, cli
api/main.py     FastAPI app (thin wrapper over readmit.inference)
app/            streamlit_app.py (thin wrapper over readmit.inference)
tests/          pytest suite
artifacts/      trained models (gitignored)
reports/        metrics.json, figures, audit markdown, executive_report.md
docs/decisions.md   running log of assumptions and deviations
```

Logic goes in `src/readmit/`. Notebooks, the API, and the dashboard only call into it.

---

## 4. Non-negotiable rules

These are invariants. Violating one is a bug, even if metrics look better.

1. **Prediction point is discharge.** A feature is allowed only if the care team would know it at discharge. Record every column's decision (A safe, B investigate, C drop) with a reason in `reports/leakage_audit.md`.
2. **Split by patient.** No `patient_nbr` may appear in more than one of train, validation, test. Splits are created once, seeded, and saved to `data/splits/`. Never re-split ad hoc.
3. **Fit on train only.** Imputers, encoders, scalers, SMOTE, feature selection, and anything else learned from data are fit on the training split only, inside a pipeline.
4. **Test set is touched once.** Model selection, calibration, and threshold choice use train and validation. Only `make final` scores the test set. If the test set is used for any decision, stop and flag it.
5. **Cohort rules are fixed:** exclude discharge dispositions 11, 13, 14, 19, 20, 21 (death/hospice); one index encounter per patient (lowest `encounter_id`) for the primary analysis. Log row counts at each step. **Verify these disposition IDs against `IDS_mapping.csv` before hardcoding them**; if the mapping disagrees, use the file and record the difference in `docs/decisions.md`.
6. **Race and gender are audit-only attributes.** They must not be model features. Age is a feature and is also audited.
7. **Never default to a 0.5 threshold.** Use the capacity-based or cost-ratio threshold from `configs/base.yaml` and label assumptions as assumptions.
8. **Recalibrate after any reweighting** (class weights, SMOTE). Report calibration alongside discrimination.
9. **Report uncertainty.** Headline test metrics and fairness gaps carry 95% bootstrap CIs (1,000 resamples, seeded). Flag groups with n < 500 as low-confidence.
10. **No fabricated numbers.** Every number in the README, report, dashboard, or resume text must trace to `reports/metrics.json` (or another generated file). If it was not measured, write "not measured". Do not write placeholder results as if real.
11. **No causal language.** Say "associated with a lower predicted risk", never "will prevent readmission". The model supports prioritization; it does not decide treatment.
12. **Suspect leakage when results look too good.** ROC-AUC above about 0.80 on this dataset is a red flag; investigate before celebrating.
13. **Never commit data, secrets, or large model files.** The data is de-identified; never try to re-identify anyone.
14. **Reproducibility.** Global seed from config, pinned dependency ranges, deterministic splits, and MLflow-logged params for every training run.

---

## 5. Workflow for any task

1. **Read** `CONTEXT.md`, the relevant module, and existing tests.
2. **Plan** briefly: what files change, which invariants are touched. For anything touching splits, features, or evaluation, state how invariants 1 to 4 are preserved.
3. **Implement** in `src/readmit/` with type hints and docstrings; keep functions small and pure where possible.
4. **Test:** add or update tests (Section 7). Run `make lint` and `make test`.
5. **Run the relevant `make` target** and confirm artifacts and `metrics.json` updated.
6. **Report:** summarize what changed, what was measured (with numbers from files), what assumptions you made, and anything unverified. Log notable decisions in `docs/decisions.md`.

Prefer small, reviewable changes. Do not refactor unrelated code. Do not add dependencies without a reason; if you add one, add it to `pyproject.toml` and mention why.

---

## 6. Conventions

**Code**
- Python 3.11, type hints on public functions, docstrings (what, inputs, outputs, assumptions).
- Format and lint with ruff. Line length 100.
- No hardcoded paths, seeds, thresholds, K, or cost ratio: read from `configs/*.yaml`.
- Use `pathlib`, `logging` (not `print`) in library code, and explicit `random_state`.
- Data transforms are sklearn-compatible (`fit` / `transform`) so they live inside one `Pipeline` / `ColumnTransformer`.

**Naming**
- Label column: `readmit_30d` (0/1). Group columns: `race_group`, `gender`, `age_band`.
- Artifacts: `artifacts/{model_name}.joblib`, `artifacts/{model_name}_calibrated.joblib`.
- Figures: `reports/figures/{topic}_{model}.png`.

**Notebooks**
- Exploration and plotting only. Strip outputs (`nbstripout`). Any logic worth keeping moves into `src/readmit/` and gets a test.

**Git**
- Branches: `feat/...`, `fix/...`, `docs/...`. Commit style: conventional commits (`feat: add capacity table`).
- Never commit `data/raw`, `data/processed`, `artifacts/`, `.env`, MLflow stores, or notebook outputs.

---

## 7. Required tests

Add and keep these passing (`tests/`):

| Test | Asserts |
|---|---|
| `test_splits_no_patient_overlap` | No patient in two splits; ratios within tolerance; label rate similar across splits |
| `test_cohort_excludes_expired_hospice` | No rows with disposition ids 11, 13, 14, 19, 20, 21; one row per patient |
| `test_no_forbidden_features` | Final feature list contains no target, identifiers, `race`, or `gender` |
| `test_preprocessing_fit_on_train_only` | Fitted statistics (means, category sets) are identical whether or not validation/test rows are present; `fit` only ever receives train rows |
| `test_label_encoding` | `"<30"` maps to 1; `">30"` and `"NO"` map to 0 |
| `test_icd_grouping` | Known ICD-9 codes map to expected categories; unknown codes map to `other` |
| `test_capacity_threshold` | Flagging top K% flags that share of rows (within rounding) on toy scores |
| `test_metrics_on_toy_data` | PR-AUC, recall, precision, lift match hand-computed values |
| `test_bootstrap_reproducible` | Same seed gives same CI |
| `test_fairness_gaps_toy` | TPR/FPR/selection-rate gaps match hand-computed values on a toy frame |
| `test_calibration_monotonic` | Calibrator preserves score ordering |
| `test_inference_roundtrip` | `readmit.inference.predict` on a sample row returns a probability in [0, 1] and an explanation |
| `test_api_schema` | `/health`, `/predict`, `/explain` return the documented schema; invalid input returns 422 |

---

## 8. Config contract (`configs/base.yaml`)

```yaml
seed: 42
paths: {raw: data/raw, processed: data/processed, splits: data/splits, artifacts: artifacts, reports: reports}
cohort:
  exclude_disposition_ids: [11, 13, 14, 19, 20, 21]
  one_encounter_per_patient: true
split: {train: 0.70, val: 0.15, test: 0.15, stratify_on: readmit_30d, group_by: patient_nbr}
decision:
  capacity_k_percent: [5, 10, 20]
  primary_capacity_k: 20          # assumption: care team follows up on top 20%
  cost_ratio_fn_to_fp: 5          # assumption, illustrative only; label everywhere it is shown
bootstrap: {n_resamples: 1000, ci: 0.95}
fairness:
  audit_attributes: [race_group, gender, age_band]
  min_group_n: 500
  mitigation: threshold_optimizer
  constraint: equalized_odds
```

The values marked as assumptions are not facts about any hospital. Say so wherever they appear.

---

## 9. Output contracts

**`reports/metrics.json`** is the single source of truth for numbers. Suggested shape:

```json
{
  "run_id": "…",
  "cohort": {"n_patients": 0, "prevalence": 0.0, "split_sizes": {"train": 0, "val": 0, "test": 0}},
  "models": {
    "xgboost_calibrated": {
      "split": "test",
      "pr_auc": {"value": 0.0, "ci95": [0.0, 0.0]},
      "roc_auc": {"value": 0.0, "ci95": [0.0, 0.0]},
      "brier": {"value": 0.0, "ci95": [0.0, 0.0]},
      "ece": 0.0,
      "capacity": [{"k_percent": 20, "recall": 0.0, "precision": 0.0, "lift": 0.0, "n_flagged": 0}]
    }
  },
  "fairness": {"attribute": {"group": {"n": 0, "tpr": 0.0, "fpr": 0.0, "precision": 0.0, "selection_rate": 0.0}}},
  "fairness_gaps": {"before": {}, "after": {}},
  "assumptions": {"primary_capacity_k": 20, "cost_ratio_fn_to_fp": 5}
}
```

(Zeros above are schema placeholders, not results.)

**Reports** written by the pipeline: `data_quality.md`, `leakage_audit.md`, `cohort_flow.md`, `model_comparison.md`, `capacity_table.md`, `fairness_before_after.md`, `executive_report.md`.

**API (`api/main.py`)**
- `GET /health` returns status and model version.
- `POST /predict` takes validated patient features and returns `risk_probability`, `risk_vs_average` (probability divided by prevalence), and `risk_band`.
- `POST /explain` returns the top SHAP contributions for that patient.
- Pydantic models validate input; unknown categories are handled gracefully; no patient data is logged.

**Dashboard pages:** Executive summary, Patient risk (show risk relative to average, not just a bare percentage), SHAP explanation, Fairness by group, Model comparison and capacity table. Show assumptions and limitations in the sidebar.

---

## 10. Definition of done per stage

A stage is done only when its outputs exist, tests pass, and `docs/decisions.md` records any deviation.

| Stage | Done when |
|---|---|
| Data audit and cohort | `data_quality.md` and `cohort_flow.md` exist; counts logged at each step |
| Leakage audit and splits | Every column classified with a reason; splits saved; overlap test passes |
| Features and baselines | Pipeline fits on train only; baselines reported; forbidden-feature test passes |
| Models and ablation | LR and XGBoost compared on validation; imbalance ablation conclusion written |
| Calibration and thresholds | Calibration plots, Brier/ECE, capacity and cost-ratio tables with assumptions labeled |
| Explainability | SHAP global/local, stability score, odds ratios, short interpretation |
| Fairness | Per-group tables with CIs, mitigation applied, before/after trade-off table, written discussion |
| Final evaluation | Test scored once via `make final`; `metrics.json` frozen |
| Engineering layer | Inference, API, dashboard run locally; Docker image builds; API tests pass |
| Write-up | README and executive report cite only numbers from generated files; limitations section present |

---

## 11. Do and don't

**Do**
- Prefer simple, explainable choices and justify each in one or two sentences.
- Compare against baselines every time you add complexity.
- State what a result does **not** show.
- Keep the README, dashboard, and report in sync with `metrics.json`.

**Don't**
- Don't add deep learning, LLM features, or extra explainers just to add buzzwords.
- Don't tune on the test set, peek at it, or re-split to get a nicer number.
- Don't drop hard-to-handle groups from the fairness audit to make gaps look smaller; report low-n groups with a warning.
- Don't write "the model is fair/unbiased"; write what was measured and the trade-off.
- Don't hardcode dataset-specific counts or percentages in code or docs; read them from generated reports.
- Don't invent clinical claims, hospital names, or deployment results.

---

## 12. When you are unsure

1. Check `CONTEXT.md` (Sections 5 and 19 list decisions and open assumptions).
2. If it is still ambiguous and low-risk, choose the more conservative, more reproducible option, record it in `docs/decisions.md`, and proceed.
3. If it affects scope, the cohort, the split, or the fairness approach, take the most conservative option consistent with `CONTEXT.md`, log it in `docs/decisions.md` and `reports/checkpoints.md`, and continue, unless it is a red flag (Sec. 0).

Report back with: what you did, what you measured (with file references), what you assumed, and what remains unverified.

---

## 13. Implementation notes and gotchas

Library APIs change between versions. Check installed versions and adapt; where an API looks unstable, prefer the small manual implementation described here.

**Calibration**
- Implement calibration manually rather than relying on `cv="prefit"` (deprecated in recent scikit-learn). Platt scaling: fit a `LogisticRegression` on the logit of the clipped validation scores. Isotonic: `IsotonicRegression(out_of_bounds="clip")` on validation scores.
- Wrap the full pipeline (preprocessing + model + calibrator) in one class exposing `predict_proba`, `fit` (no-op or refit of the calibrator only), and `__sklearn_is_fitted__` returning `True`, so downstream tools such as Fairlearn accept it as a prefit estimator.
- Choose Platt vs isotonic by validation Brier score and ECE; report both.

**XGBoost**
- Use `tree_method="hist"`, set `early_stopping_rounds` in the constructor (xgboost 2.x) and pass an `eval_set` taken from an inner split of the **training** data only.
- For the class-weighted variant, set `scale_pos_weight = n_negative / n_positive` computed on train. Recalibrate afterwards (rule 8).

**Metrics**
- PR-AUC is `average_precision_score`. ECE uses 10 bins (state whether uniform or quantile) and the same choice everywhere.
- Capacity: sort by predicted risk descending (stable sort for ties), flag the top `ceil(K% * n)`, and compute recall, precision, and lift (precision divided by prevalence).
- Bootstrap: resample test rows with replacement, recompute metrics, take 2.5th and 97.5th percentiles. Reuse the same resample indices across models so comparisons are paired.

**SHAP**
- Use `shap.TreeExplainer` on the fitted `XGBClassifier` and the transformed feature matrix; keep feature names from `get_feature_names_out`. Subsample 2,000 to 5,000 rows for beeswarm plots. Build `shap.Explanation` objects for waterfall plots. SHAP values are in log-odds of the uncalibrated model; say so in captions.

**Fairlearn mitigation**
- `ThresholdOptimizer(estimator=calibrated_model, constraints="equalized_odds", objective="balanced_accuracy_score", predict_method="predict_proba", prefit=True)`.
- Fit with `fit(X_val, y_val, sensitive_features=s_val)`; predict on test with `predict(X_test, sensitive_features=s_test, random_state=seed)`. The estimator must accept the same input as `X` (use the full pipeline wrapper).
- The output is hard labels (possibly randomized), so PR-AUC is only reported for the **before** state. For the **after** state report recall, precision, selection rate, and the gaps. Compute per-group metrics with `fairlearn.metrics.MetricFrame`.
- If `equalized_odds` is infeasible or unstable for small groups, fall back to `true_positive_rate_parity`, log the reason, and flag small groups.

**Test-set lock**
- `python -m readmit.cli final` must write `artifacts/.final_done` (timestamp and git commit if available) and **refuse to run again** unless called with `--force`, in which case it logs the forced re-run in `docs/decisions.md`. This enforces rule 4.

**Reproducibility and environment**
- Set `random_state` everywhere, including XGBoost, bootstrap, SMOTE, `ThresholdOptimizer`, and SHAP subsampling.
- Use `pathlib` and guard script entrypoints with `if __name__ == "__main__":` (Windows multiprocessing).
- Save plots as PNG at 150 dpi with titles, axis labels, units, and a prevalence reference line where relevant. Keep a consistent style across figures.
- In Streamlit, load the model once with `st.cache_resource`.
- The whole pipeline on this dataset should run in minutes; if something is slow, check for accidental per-row loops.

**Git hygiene**
- Create `.gitignore` early (Appendix D in `CONTEXT.md`) so data, artifacts, MLflow stores, and notebook outputs are never committed.

---

## 14. Final handoff (`reports/FINAL_SUMMARY.md`)

Write this file at the end. It lets the owner see the real state quickly. It must contain:

1. **What was built**, as a checklist by milestone and tier (P0 to P3), marking each item done, partial, or not done, with the command that proves it.
2. **Key results**, copied from `reports/metrics.json`: cohort size, prevalence, split sizes, PR-AUC with CI for each model, ROC-AUC, Brier, ECE, the capacity table, the chosen model and why, and the fairness before/after table.
3. **Decisions and deviations** from `CONTEXT.md`, with reasons (mirrors `docs/decisions.md`).
4. **Assumptions** that affect interpretation (K, cost ratio, first-encounter rule).
5. **Known issues and limitations**, including anything unverified and any red flags encountered.
6. **How to run**: exact commands for setup, training, evaluation, dashboard, API, and tests.
7. **Form-ready material**: a draft project entry and three resume bullets in which every number is filled from `metrics.json`; anything not measured is written as "not measured".
