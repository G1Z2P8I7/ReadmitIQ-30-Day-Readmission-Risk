"""One-time final test set evaluation, test-set locking, and metrics.json freezing."""

import json
import logging
import pathlib
import subprocess
import sys
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
import yaml
from fairlearn.metrics import selection_rate, true_positive_rate
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    roc_auc_score,
)

from readmit.calibration import PlattCalibrator
from readmit.evaluation import compute_capacity_metrics, compute_ece
from readmit.fairness import compute_group_fairness_table, compute_parity_gaps, map_age_band

logger = logging.getLogger(__name__)


def get_git_commit() -> str:
    """Returns the current git commit hash if available, or 'unknown'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return res.stdout.strip() if res.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def run_final_step(args=None, config_dir: str = "configs"):
    """Scores the test set once, computes 1,000 bootstrap 95% CIs, and freezes reports/metrics.json."""
    logger.info("Executing M8 Final Evaluation step on the untouched TEST set...")

    with open(f"{config_dir}/base.yaml") as f:
        base_cfg = yaml.safe_load(f)

    seed = base_cfg.get("seed", 42)
    min_group_n = base_cfg["fairness"].get("min_group_n", 500)
    primary_k = base_cfg["decision"].get("primary_capacity_k", 20)
    cost_ratio = base_cfg["decision"].get("cost_ratio_fn_to_fp", 5)
    bootstrap_resamples = base_cfg.get("bootstrap", {}).get("n_resamples", 1000)

    artifacts_dir = pathlib.Path(base_cfg["paths"]["artifacts"])
    reports_dir = pathlib.Path(base_cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    lock_file = artifacts_dir / ".final_done"
    force = getattr(args, "force", False) if args is not None else False

    # Enforce Invariant 4: Test set lock
    if lock_file.exists() and not force:
        logger.error(
            f"Test set evaluation already executed and locked at {lock_file}. "
            "To re-run, you must pass --force."
        )
        print(f"ERROR: Final evaluation already completed and locked ({lock_file}). Use --force to override.")
        sys.exit(1)

    if lock_file.exists() and force:
        logger.warning("FORCED RE-RUN of final evaluation step! Logging deviation to docs/decisions.md...")
        decisions_file = pathlib.Path("docs/decisions.md")
        timestamp = datetime.now(UTC).isoformat()
        with open(decisions_file, "a", encoding="utf-8") as f:
            f.write(
                f"| {timestamp} | M8 Final Re-Run | User/Agent ran `final` with `--force` flag | "
                "Re-evaluated test set upon explicit request. |\n"
            )

    # 1. Load test set and models
    test_df = pd.read_parquet("data/processed/test.parquet").reset_index(drop=True)
    val_df = pd.read_parquet("data/processed/val.parquet").reset_index(drop=True)
    train_df = pd.read_parquet("data/processed/train.parquet").reset_index(drop=True)

    y_test = test_df["readmit_30d"].to_numpy().astype(int)
    n_test = len(y_test)
    prevalence = float(np.mean(y_test))

    feature_pipeline = joblib.load(artifacts_dir / "feature_pipeline.joblib")
    calibrated_xgb = joblib.load(artifacts_dir / "xgboost_calibrated.joblib")
    raw_xgb = joblib.load(artifacts_dir / "xgboost.joblib")
    raw_lr = joblib.load(artifacts_dir / "logistic_regression.joblib")
    fairness_optimizer = joblib.load(artifacts_dir / "fairness_optimizer.joblib")

    # Fit Platt calibrated LR using validation set strictly
    val_trans = feature_pipeline.transform(val_df)
    test_trans = feature_pipeline.transform(test_df)
    p_val_lr = raw_lr.predict_proba(val_trans)[:, 1]
    platt_lr = PlattCalibrator()
    platt_lr.fit(p_val_lr, val_df["readmit_30d"].to_numpy().astype(int))

    # Evaluate models on test set
    p_test_xgb_cal = calibrated_xgb.predict_proba(test_df)[:, 1]
    p_test_xgb_raw = raw_xgb.predict_proba(test_trans)[:, 1]
    p_test_lr_raw = raw_lr.predict_proba(test_trans)[:, 1]
    p_test_lr_cal = platt_lr.predict_proba(p_test_lr_raw)[:, 1]

    # Pre-generate paired bootstrap resample indices across all models
    rng = np.random.RandomState(seed)
    bootstrap_indices = [
        rng.randint(0, n_test, size=n_test) for _ in range(bootstrap_resamples)
    ]

    def paired_bootstrap(y_true, y_prob, metric_fn):
        point = float(metric_fn(y_true, y_prob))
        boot_vals = []
        for idx in bootstrap_indices:
            sy = y_true[idx]
            sp = y_prob[idx]
            if len(np.unique(sy)) < 2:
                continue
            boot_vals.append(float(metric_fn(sy, sp)))
        lower = float(np.percentile(boot_vals, 2.5))
        upper = float(np.percentile(boot_vals, 97.5))
        return {"value": point, "ci95": [lower, upper]}

    logger.info("Computing test set bootstrap confidence intervals for primary model...")
    pr_xgb = paired_bootstrap(y_test, p_test_xgb_cal, average_precision_score)
    roc_xgb = paired_bootstrap(y_test, p_test_xgb_cal, roc_auc_score)
    brier_xgb = paired_bootstrap(y_test, p_test_xgb_cal, brier_score_loss)
    ece_xgb = compute_ece(y_test, p_test_xgb_cal, n_bins=10, strategy="uniform")

    # Check for leakage red flag
    if roc_xgb["value"] > 0.80:
        logger.error(
            f"LEAKAGE RED FLAG: Test ROC-AUC is {roc_xgb['value']:.4f} (> 0.80). "
            "Suspect target leakage in feature engineering."
        )

    # Capacity metrics on test set
    cap_tiers = [5, 10, 20]
    cap_metrics = compute_capacity_metrics(y_test, p_test_xgb_cal, k_percents=cap_tiers)

    # Demographic sensitive features
    s_race = test_df["race"].replace("?", "Unknown")
    s_gender = test_df["gender"]
    s_age = test_df["age"].map(map_age_band)

    # Capacity Top 20% Unmitigated Decision on Test
    n_flagged_20 = int(np.ceil(n_test * (primary_k / 100.0)))
    sort_idx = np.argsort(-p_test_xgb_cal, kind="mergesort")
    cutoff_20 = float(p_test_xgb_cal[sort_idx[n_flagged_20 - 1]])
    y_pred_unmitigated = (p_test_xgb_cal >= cutoff_20).astype(int)

    # Fairness Audit on Test - Before Mitigation
    fairness_test_before = {}
    fairness_test_before_gaps = {}
    for attr_name, series in [("race_group", s_race), ("gender", s_gender), ("age_band", s_age)]:
        df_grp = compute_group_fairness_table(y_test, y_pred_unmitigated, series, min_group_n=min_group_n)
        gaps = compute_parity_gaps(df_grp)
        fairness_test_before[attr_name] = df_grp.to_dict(orient="index")
        fairness_test_before_gaps[attr_name] = gaps

    # Mitigated Predictions on Test using fitted ThresholdOptimizer
    logger.info("Applying Fairlearn ThresholdOptimizer to test set...")
    y_pred_mitigated = fairness_optimizer.predict(test_df, sensitive_features=s_race, random_state=seed)

    fairness_test_after = {}
    fairness_test_after_gaps = {}
    for attr_name, series in [("race_group", s_race), ("gender", s_gender), ("age_band", s_age)]:
        df_grp = compute_group_fairness_table(y_test, y_pred_mitigated, series, min_group_n=min_group_n)
        gaps = compute_parity_gaps(df_grp)
        fairness_test_after[attr_name] = df_grp.to_dict(orient="index")
        fairness_test_after_gaps[attr_name] = gaps

    # Build frozen reports/metrics.json
    now_utc = datetime.now(UTC)
    metrics_payload = {
        "run_id": f"run_{now_utc.strftime('%Y%m%d_%H%M%S')}",
        "git_commit": get_git_commit(),
        "timestamp": now_utc.isoformat(),
        "cohort": {
            "n_patients": len(train_df) + len(val_df) + len(test_df),
            "prevalence": prevalence,
            "split_sizes": {
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df),
            },
        },
        "models": {
            "xgboost_calibrated": {
                "split": "test",
                "pr_auc": pr_xgb,
                "roc_auc": roc_xgb,
                "brier": brier_xgb,
                "ece": ece_xgb,
                "capacity": cap_metrics,
            },
            "logistic_regression_calibrated": {
                "split": "test",
                "pr_auc": paired_bootstrap(y_test, p_test_lr_cal, average_precision_score),
                "roc_auc": paired_bootstrap(y_test, p_test_lr_cal, roc_auc_score),
                "brier": paired_bootstrap(y_test, p_test_lr_cal, brier_score_loss),
                "ece": compute_ece(y_test, p_test_lr_cal, n_bins=10, strategy="uniform"),
                "capacity": compute_capacity_metrics(y_test, p_test_lr_cal, k_percents=cap_tiers),
            },
            "xgboost_raw": {
                "split": "test",
                "pr_auc": paired_bootstrap(y_test, p_test_xgb_raw, average_precision_score),
                "roc_auc": paired_bootstrap(y_test, p_test_xgb_raw, roc_auc_score),
                "brier": paired_bootstrap(y_test, p_test_xgb_raw, brier_score_loss),
                "ece": compute_ece(y_test, p_test_xgb_raw, n_bins=10, strategy="uniform"),
            },
        },
        "fairness_audit": {
            "before_mitigation": fairness_test_before,
            "after_mitigation": fairness_test_after,
        },
        "fairness_gaps": {
            "before": fairness_test_before_gaps,
            "after": fairness_test_after_gaps,
        },
        "mitigation_summary": {
            "overall_before": {
                "recall": float(true_positive_rate(y_test, y_pred_unmitigated)),
                "precision": float(precision_score(y_test, y_pred_unmitigated, zero_division=0)),
                "selection_rate": float(selection_rate(y_test, y_pred_unmitigated)),
            },
            "overall_after": {
                "recall": float(true_positive_rate(y_test, y_pred_mitigated)),
                "precision": float(precision_score(y_test, y_pred_mitigated, zero_division=0)),
                "selection_rate": float(selection_rate(y_test, y_pred_mitigated)),
            },
        },
        "assumptions": {
            "primary_capacity_k": primary_k,
            "cost_ratio_fn_to_fp": cost_ratio,
            "first_encounter_rule": "Index encounter defined as lowest encounter_id per patient",
        },
    }

    metrics_file = reports_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info(f"FROZEN final metrics written to {metrics_file}")

    # Write lock file
    git_hash = get_git_commit()
    with open(lock_file, "w", encoding="utf-8") as f:
        f.write(f"timestamp: {now_utc.isoformat()}\ncommit: {git_hash}\n")
    logger.info(f"Test-set evaluation locked at {lock_file}")

    print("\n=======================================================")
    print("M8 FINAL TEST EVALUATION COMPLETE AND FROZEN")
    print(f"Metrics written to: {metrics_file}")
    print(f"Primary XGBoost Calibrated Test PR-AUC: {pr_xgb['value']:.4f} [95% CI: {pr_xgb['ci95'][0]:.4f} - {pr_xgb['ci95'][1]:.4f}]")
    print(f"Primary XGBoost Calibrated Test ROC-AUC: {roc_xgb['value']:.4f} [95% CI: {roc_xgb['ci95'][0]:.4f} - {roc_xgb['ci95'][1]:.4f}]")
    print(f"Test Recall @ K=20%: {cap_metrics[2]['recall'] * 100:.2f}%, Precision: {cap_metrics[2]['precision'] * 100:.2f}%, Lift: {cap_metrics[2]['lift']:.2f}x")
    print(f"Race FPR Disparity: {fairness_test_before_gaps['race_group']['fpr_gap'] * 100:.2f}% -> {fairness_test_after_gaps['race_group']['fpr_gap'] * 100:.2f}%")
    print("=======================================================\n")
    return 0
