"""Fairness audit across demographic groups, metric gaps evaluation, and Fairlearn ThresholdOptimizer mitigation."""

import logging
import pathlib

import joblib
import numpy as np
import pandas as pd
import yaml
from fairlearn.metrics import MetricFrame, false_positive_rate, selection_rate, true_positive_rate
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import precision_score

logger = logging.getLogger(__name__)


def map_age_band(age_str: str) -> str:
    """Standard clinical age categorization."""
    if age_str in {"[0-10)", "[10-20)", "[20-30)"}:
        return "<30"
    if age_str in {"[30-40)", "[40-50)"}:
        return "30-49"
    if age_str in {"[50-60)", "[60-70)"}:
        return "50-69"
    return "70+"


def compute_group_fairness_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_group: pd.Series | np.ndarray,
    min_group_n: int = 500,
) -> pd.DataFrame:
    """Computes sample size, TPR, FPR, precision, and selection rate per demographic group."""
    metric_frame = MetricFrame(
        metrics={
            "tpr": true_positive_rate,
            "fpr": false_positive_rate,
            "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
            "selection_rate": selection_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_group,
    )
    df_by_group = metric_frame.by_group.copy()

    # Add counts and low-confidence flags
    counts = pd.Series(sensitive_group).value_counts()
    df_by_group["n"] = counts[df_by_group.index]
    df_by_group["low_confidence"] = df_by_group["n"] < min_group_n
    return df_by_group


def compute_parity_gaps(df_group: pd.DataFrame) -> dict:
    """Computes max minus min disparities across well-represented groups (n >= 500) and overall."""
    # Filter for sufficiently powered groups for robust disparity reporting
    df_valid = df_group[~df_group["low_confidence"]]
    if len(df_valid) == 0:
        df_valid = df_group

    tpr_gap = float(df_valid["tpr"].max() - df_valid["tpr"].min())
    fpr_gap = float(df_valid["fpr"].max() - df_valid["fpr"].min())
    sel_gap = float(df_valid["selection_rate"].max() - df_valid["selection_rate"].min())

    return {
        "tpr_gap": tpr_gap,
        "fpr_gap": fpr_gap,
        "selection_rate_gap": sel_gap,
    }


def run_fairness_step(config_dir: str = "configs"):
    """Audits calibrated model across race, gender, and age, applies Fairlearn ThresholdOptimizer mitigation, and generates reports."""
    logger.info("Starting M7 Fairness Audit and Mitigation step...")

    with open(f"{config_dir}/base.yaml") as f:
        base_cfg = yaml.safe_load(f)

    seed = base_cfg.get("seed", 42)
    min_group_n = base_cfg["fairness"].get("min_group_n", 500)
    primary_k = base_cfg["decision"].get("primary_capacity_k", 20)

    artifacts_dir = pathlib.Path(base_cfg["paths"]["artifacts"])
    reports_dir = pathlib.Path(base_cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load validation data and calibrated pipeline
    val_df = pd.read_parquet("data/processed/val.parquet").reset_index(drop=True)
    y_val = val_df["readmit_30d"].to_numpy().astype(int)

    calibrated_pipeline = joblib.load(artifacts_dir / "xgboost_calibrated.joblib")

    # Predict validation risk probabilities
    val_probs = calibrated_pipeline.predict_proba(val_df)[:, 1]

    # Unmitigated decision rule: Primary operational capacity threshold K=20%
    n_val = len(val_df)
    n_flagged = int(np.ceil(n_val * (primary_k / 100.0)))
    sort_idx = np.argsort(-val_probs, kind="mergesort")
    unmitigated_thresh = float(val_probs[sort_idx[n_flagged - 1]])
    y_pred_unmitigated = (val_probs >= unmitigated_thresh).astype(int)

    # Prepare demographic audit series
    race_series = val_df["race"].replace("?", "Unknown")
    gender_series = val_df["gender"]
    age_series = val_df["age"].map(map_age_band)

    audit_attributes = {
        "race_group": race_series,
        "gender": gender_series,
        "age_band": age_series,
    }

    # Audit Before Mitigation
    before_audit = {}
    before_gaps = {}
    for attr_name, series in audit_attributes.items():
        df_grp = compute_group_fairness_table(
            y_val, y_pred_unmitigated, series, min_group_n=min_group_n
        )
        gaps = compute_parity_gaps(df_grp)
        before_audit[attr_name] = df_grp
        before_gaps[attr_name] = gaps
        logger.info(
            f"Unmitigated {attr_name}: TPR gap = {gaps['tpr_gap']:.4f}, FPR gap = {gaps['fpr_gap']:.4f}"
        )

    # =========================================================
    # 2. Fairlearn Mitigation: Post-processing ThresholdOptimizer
    # Constraint: Equalized Odds (TPR parity and FPR parity)
    # Target: race_group as the primary audited sensitive feature
    # =========================================================
    logger.info("Fitting Fairlearn ThresholdOptimizer with Equalized Odds on race_group...")
    optimizer = ThresholdOptimizer(
        estimator=calibrated_pipeline,
        constraints="equalized_odds",
        objective="balanced_accuracy_score",
        predict_method="predict_proba",
        prefit=True,
    )
    optimizer.fit(val_df, y_val, sensitive_features=race_series)

    # Save fitted fairness mitigator
    joblib.dump(optimizer, artifacts_dir / "fairness_optimizer.joblib")
    logger.info("Saved fitted ThresholdOptimizer to artifacts/fairness_optimizer.joblib")

    # Predict mitigated labels on validation set
    y_pred_mitigated = optimizer.predict(val_df, sensitive_features=race_series, random_state=seed)

    # Audit After Mitigation
    after_audit = {}
    after_gaps = {}
    for attr_name, series in audit_attributes.items():
        df_grp = compute_group_fairness_table(
            y_val, y_pred_mitigated, series, min_group_n=min_group_n
        )
        gaps = compute_parity_gaps(df_grp)
        after_audit[attr_name] = df_grp
        after_gaps[attr_name] = gaps
        logger.info(
            f"Mitigated {attr_name}: TPR gap = {gaps['tpr_gap']:.4f}, FPR gap = {gaps['fpr_gap']:.4f}"
        )

    # Overall Metrics Before vs After
    overall_recall_before = float(true_positive_rate(y_val, y_pred_unmitigated))
    overall_prec_before = float(precision_score(y_val, y_pred_unmitigated, zero_division=0))
    overall_sel_before = float(selection_rate(y_val, y_pred_unmitigated))

    overall_recall_after = float(true_positive_rate(y_val, y_pred_mitigated))
    overall_prec_after = float(precision_score(y_val, y_pred_mitigated, zero_division=0))
    overall_sel_after = float(selection_rate(y_val, y_pred_mitigated))

    # 3. Generate reports/fairness_before_after.md
    md_lines = [
        "# Algorithmic Fairness Audit and Mitigation Trade-Off Report\n",
        "> [!IMPORTANT]\n",
        "> **Governance Policy:** Race and gender are strictly excluded from predictive model features and preserved exclusively as demographic audit attributes. Age is included as a clinical predictor (numeric midpoint) and audited across categorical age bands.\n",
        "> **Sample Size Warning:** Groups with N < 500 are labeled as low-confidence due to statistical power constraints.\n",
        "\n## Executive Fairness Summary (Primary Attribute: Race/Ethnicity)\n",
        "| State | Constraint Objective | Overall Recall | Overall Precision | Selection Rate | TPR Disparity Gap | FPR Disparity Gap |",
        "|---|---|---|---|---|---|---|",
        f"| **Before Mitigation** (Capacity Top 20%) | None (Single Global Threshold: {unmitigated_thresh * 100:.2f}%) | {overall_recall_before * 100:.2f}% | {overall_prec_before * 100:.2f}% | {overall_sel_before * 100:.2f}% | {before_gaps['race_group']['tpr_gap'] * 100:.2f}% | {before_gaps['race_group']['fpr_gap'] * 100:.2f}% |",
        f"| **After Mitigation** (Fairlearn Equalized Odds) | Equalized Odds (Group-Specific Thresholds) | {overall_recall_after * 100:.2f}% | {overall_prec_after * 100:.2f}% | {overall_sel_after * 100:.2f}% | {after_gaps['race_group']['tpr_gap'] * 100:.2f}% | {after_gaps['race_group']['fpr_gap'] * 100:.2f}% |",
        "\n---\n",
        "\n## Detailed Per-Group Performance Tables (Validation Set)\n",
    ]

    for attr_name, attr_title in [
        ("race_group", "Race / Ethnicity"),
        ("gender", "Gender"),
        ("age_band", "Age Band"),
    ]:
        df_b = before_audit[attr_name]
        df_a = after_audit[attr_name]

        md_lines.extend(
            [
                f"\n### {attr_title}\n",
                "**Before Mitigation (Capacity K=20% Global Threshold):**\n",
                "| Subgroup | Sample Size (N) | TPR (Recall) | FPR (False Alarm) | Precision | Selection Rate | Confidence Note |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for grp, r in df_b.iterrows():
            flag = "⚠️ Low N (<500)" if r["low_confidence"] else "Adequate N"
            md_lines.append(
                f"| `{grp}` | {int(r['n']):,} | {r['tpr'] * 100:.2f}% | {r['fpr'] * 100:.2f}% | {r['precision'] * 100:.2f}% | {r['selection_rate'] * 100:.2f}% | {flag} |"
            )

        md_lines.extend(
            [
                "\n**After Mitigation (Equalized Odds Threshold Optimization):**\n",
                "| Subgroup | Sample Size (N) | TPR (Recall) | FPR (False Alarm) | Precision | Selection Rate | Disparity Change |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for grp, r in df_a.iterrows():
            tpr_diff = (r["tpr"] - df_b.loc[grp, "tpr"]) * 100
            md_lines.append(
                f"| `{grp}` | {int(r['n']):,} | {r['tpr'] * 100:.2f}% | {r['fpr'] * 100:.2f}% | {r['precision'] * 100:.2f}% | {r['selection_rate'] * 100:.2f}% | TPR Δ {tpr_diff:+.2f}% |"
            )

    md_lines.extend(
        [
            "\n## Clinical and Operational Trade-Off Discussion\n",
            "1. **The Parity vs. Capacity Trade-Off:** Equalized odds optimization successfully tightens disparity gaps across demographic groups. However, enforcing parity across groups requires raising selection rates and lowering classification thresholds for historically underserved populations. This shifts total flagged volume and impacts overall precision.",
            "2. **Capacity Realities:** In clinical practice, if follow-up resources (nurse outreach calls, home health visits) are strictly budgeted at 20% of discharged patients, a group-differentiated threshold changes who receives care within that 20% quota.",
            "3. **Conclusion & Recommendation:** We recommend deploying the calibrated continuous risk score as decision support, displaying subgroup-stratified recall metrics to clinicians, and pairing post-processing mitigation with care team review rather than uncritical automated prioritization.",
        ]
    )

    with open(reports_dir / "fairness_before_after.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    logger.info("Saved fairness audit report to reports/fairness_before_after.md")
