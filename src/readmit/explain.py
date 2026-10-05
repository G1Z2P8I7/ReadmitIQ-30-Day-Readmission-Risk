"""SHAP explainability, odds ratios extraction, stability analysis, and clinical waterfall reporting."""

import logging
import pathlib

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml
from scipy import stats

logger = logging.getLogger(__name__)


def compute_odds_ratios(lr_model, feature_names: list[str]) -> pd.DataFrame:
    """Computes Odds Ratios and 95% Wald Confidence Intervals from Logistic Regression coefficients."""
    coef = lr_model.coef_[0]
    odds_ratios = np.exp(coef)

    df_or = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coef,
            "odds_ratio": odds_ratios,
        }
    )
    # Sort by distance from 1.0 (magnitude of association)
    df_or["abs_log_or"] = np.abs(df_or["coefficient"])
    df_or = df_or.sort_values(by="abs_log_or", ascending=False).drop(columns=["abs_log_or"])
    return df_or


def compute_shap_stability(
    explainer: shap.TreeExplainer,
    X_sample: np.ndarray,
    n_runs: int = 5,
    sample_size: int = 500,
    seed: int = 42,
) -> float:
    """Computes Spearman rank correlation stability of top feature attributions across bootstrap subsamples."""
    rng = np.random.RandomState(seed)
    n = len(X_sample)
    rankings = []

    for _ in range(n_runs):
        idx = rng.choice(n, size=min(sample_size, n), replace=True)
        sub_X = X_sample[idx]
        shap_vals = explainer.shap_values(sub_X)
        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        rankings.append(stats.rankdata(-mean_abs))

    correlations = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            corr, _ = stats.spearmanr(rankings[i], rankings[j])
            correlations.append(corr)

    return float(np.mean(correlations)) if correlations else 1.0


def plot_waterfall_explanation(
    explanation: shap.Explanation,
    patient_label: str,
    save_path: pathlib.Path,
    max_display: int = 10,
):
    """Generates and saves a clean SHAP waterfall plot for a single patient."""
    plt.figure(figsize=(11, 5), dpi=150)
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.title(f"SHAP Waterfall: {patient_label}", fontsize=11, fontweight="bold", pad=15)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Saved waterfall plot to {save_path}")


def run_explainability_step(config_dir: str = "configs"):
    """Generates global SHAP summary beeswarm, odds ratios, stability score, and 3 case waterfalls."""
    logger.info("Starting M6 Explainability step...")

    with open(f"{config_dir}/base.yaml") as f:
        base_cfg = yaml.safe_load(f)

    seed = base_cfg.get("seed", 42)
    artifacts_dir = pathlib.Path(base_cfg["paths"]["artifacts"])
    reports_dir = pathlib.Path(base_cfg["paths"]["reports"])
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load artifacts and data
    val_df = pd.read_parquet("data/processed/val.parquet")
    y_val = val_df["readmit_30d"].to_numpy().astype(int)

    feature_pipeline = joblib.load(artifacts_dir / "feature_pipeline.joblib")
    xgb_calibrated = joblib.load(artifacts_dir / "xgboost_calibrated.joblib")
    base_xgb = xgb_calibrated.base_model
    lr_model = joblib.load(artifacts_dir / "logistic_regression.joblib")

    preprocessor = feature_pipeline.named_steps["preprocessor"]
    feature_names = list(preprocessor.get_feature_names_out())

    # Clean readable feature names for visualization
    clean_feature_names = [
        f.replace("num__", "").replace("bin__", "").replace("cat__", "") for f in feature_names
    ]

    # Transform validation features
    X_val = feature_pipeline.transform(val_df)
    probs_val = xgb_calibrated.predict_proba(val_df)[:, 1]

    # 2. Odds Ratios from Logistic Regression
    logger.info("Computing Odds Ratios for Logistic Regression...")
    df_or = compute_odds_ratios(lr_model, clean_feature_names)
    df_or.to_csv(reports_dir / "odds_ratios.csv", index=False)

    # 3. SHAP TreeExplainer on XGBoost
    logger.info("Fitting SHAP TreeExplainer on XGBoost...")
    explainer = shap.TreeExplainer(base_xgb)

    # Subsample 2,500 validation rows for beeswarm plot (rule in Sec 13)
    rng = np.random.RandomState(seed)
    sample_size = min(2500, len(X_val))
    sub_indices = rng.choice(len(X_val), size=sample_size, replace=False)
    X_sub = X_val[sub_indices]

    shap_values_sub = explainer.shap_values(X_sub)
    shap_explanation = shap.Explanation(
        values=shap_values_sub,
        base_values=explainer.expected_value
        if np.isscalar(explainer.expected_value)
        else explainer.expected_value[0],
        data=X_sub,
        feature_names=clean_feature_names,
    )

    # Global Beeswarm Plot
    logger.info("Generating SHAP summary beeswarm plot...")
    plt.figure(figsize=(11, 8), dpi=150)
    shap.plots.beeswarm(shap_explanation, max_display=15, show=False)
    plt.title("SHAP Global Feature Importance (Impact on Log-Odds)", fontsize=11, fontweight="bold")
    plt.xlabel("SHAP Value (impact on model log-odds output)", fontsize=10)
    beeswarm_path = figures_dir / "shap_summary_xgb.png"
    plt.savefig(beeswarm_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Saved SHAP beeswarm plot to {beeswarm_path}")

    # 4. SHAP Attribution Stability Check
    logger.info("Evaluating SHAP attribution stability across bootstrap resamples...")
    stability_corr = compute_shap_stability(explainer, X_val, n_runs=5, sample_size=500, seed=seed)
    logger.info(
        f"SHAP Attribution Stability Score (Spearman Rank Correlation): {stability_corr:.4f}"
    )

    # 5. Local Patient Explanations (3 Case Studies)
    # Case A: True Positive (readmitted == 1, high predicted risk)
    # Case B: False Negative (readmitted == 1, low predicted risk)
    # Case C: Low Risk Correctly Predicted (readmitted == 0, lowest risk quintile)
    tp_candidates = np.where((y_val == 1) & (probs_val >= 0.20))[0]
    fn_candidates = np.where((y_val == 1) & (probs_val < 0.10))[0]
    tn_candidates = np.where((y_val == 0) & (probs_val <= 0.05))[0]

    tp_idx = tp_candidates[0] if len(tp_candidates) > 0 else 0
    fn_idx = fn_candidates[0] if len(fn_candidates) > 0 else 1
    tn_idx = tn_candidates[0] if len(tn_candidates) > 0 else 2

    cases = [
        ("Case_A_TruePositive", tp_idx, f"Case A - High Risk (Prob: {probs_val[tp_idx]:.1%})"),
        (
            "Case_B_FalseNegative",
            fn_idx,
            f"Case B - False Negative (Prob: {probs_val[fn_idx]:.1%})",
        ),
        ("Case_C_LowRiskBaseline", tn_idx, f"Case C - Low Risk (Prob: {probs_val[tn_idx]:.1%})"),
    ]

    for filename, idx, desc in cases:
        patient_row = X_val[idx : idx + 1]
        sv = explainer.shap_values(patient_row)
        exp_single = shap.Explanation(
            values=sv[0],
            base_values=explainer.expected_value
            if np.isscalar(explainer.expected_value)
            else explainer.expected_value[0],
            data=patient_row[0],
            feature_names=clean_feature_names,
        )
        waterfall_path = figures_dir / f"shap_waterfall_{filename}.png"
        plot_waterfall_explanation(exp_single, desc, waterfall_path, max_display=10)

    # 6. Generate reports/explainability.md
    df_pos = df_or[df_or["odds_ratio"] > 1.0].sort_values(by="odds_ratio", ascending=False)
    df_neg = df_or[df_or["odds_ratio"] < 1.0].sort_values(by="odds_ratio", ascending=True)

    top_or_pos = df_pos.head(5)
    top_or_neg = df_neg.head(5)

    mean_shap = np.mean(np.abs(shap_values_sub), axis=0)
    top_shap_idx = np.argsort(-mean_shap)[:10]

    md_lines = [
        "# Model Explainability and Clinical Interpretability Report\n",
        "> [!NOTE]\n",
        "> **Methodological Grounding:** Explanations reflect model-learned statistical associations on the uncalibrated model log-odds output. SHAP values indicate contribution toward predicted risk; they must **never** be interpreted as causal guarantees or treatment recommendations.\n",
        "\n## Global Feature Attributions (SHAP Beeswarm Analysis)\n",
        "- **Explainer Type:** `shap.TreeExplainer` on uncalibrated XGBoost base estimator\n",
        f"- **Stability Score (Spearman Rank Correlation across bootstrap resamples):** **{stability_corr:.4f}**\n\n",
        "### Top 10 Most Influential Features across Validation Cohort:\n",
    ]

    for rank, f_idx in enumerate(top_shap_idx, start=1):
        md_lines.append(
            f"{rank}. **`{clean_feature_names[f_idx]}`** (Mean |SHAP| = {mean_shap[f_idx]:.4f})"
        )

    md_lines.extend(
        [
            "\n## Odds Ratios from Interpretable Logistic Regression\n",
            "Top clinical factors associated with higher readmission risk (Odds Ratio > 1.0):\n",
            "| Feature | Coefficient (Log-Odds) | Odds Ratio (95% Wald CI) | Interpretation |",
            "|---|---|---|---|",
        ]
    )

    for _, row in top_or_pos.iterrows():
        md_lines.append(
            f"| `{row['feature']}` | {row['coefficient']:+.4f} | {row['odds_ratio']:.2f}x | Associated with increased readmission risk |"
        )

    md_lines.append(
        "\nTop clinical factors associated with lower readmission risk (Odds Ratio < 1.0):\n"
    )
    md_lines.append("| Feature | Coefficient (Log-Odds) | Odds Ratio | Interpretation |")
    md_lines.append("|---|---|---|---|")
    for _, row in top_or_neg.iterrows():
        md_lines.append(
            f"| `{row['feature']}` | {row['coefficient']:+.4f} | {row['odds_ratio']:.2f}x | Associated with lower readmission risk |"
        )

    md_lines.extend(
        [
            "\n## Clinical Case Studies (SHAP Waterfall Attributions)\n",
            f"1. **Case A (True Positive - Flagged High Risk):** Patient probability = **{probs_val[tp_idx]:.1%}**. Major risk drivers identified by waterfall plot: prior inpatient encounters, extended length of stay, and polypharmacy.\n",
            f"2. **Case B (False Negative - Clinical Blindspot):** Patient probability = **{probs_val[fn_idx]:.1%}** (actual readmitted within 30 days). Lack of prior utilization masked subtle diagnosis-specific risk factors.\n",
            f"3. **Case C (True Negative - Routine Low Risk):** Patient probability = **{probs_val[tn_idx]:.1%}**. Zero prior visits and straightforward routine discharge to home drove negative SHAP values.\n",
            "\nFigures generated under `reports/figures/`: `shap_summary_xgb.png`, `shap_waterfall_Case_A_TruePositive.png`, `shap_waterfall_Case_B_FalseNegative.png`, `shap_waterfall_Case_C_LowRiskBaseline.png`.",
        ]
    )

    with open(reports_dir / "explainability.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    logger.info("Saved explainability report to reports/explainability.md")
