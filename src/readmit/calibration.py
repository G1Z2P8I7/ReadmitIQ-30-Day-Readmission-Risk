"""Probability calibration (Platt scaling and Isotonic regression), FullCalibratedPipeline, and capacity table generation."""

import logging
import pathlib

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.special import logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from readmit.evaluation import compute_all_metrics, compute_capacity_metrics, compute_ece

logger = logging.getLogger(__name__)


class PlattCalibrator(BaseEstimator, ClassifierMixin):
    """Platt scaling: fits 1D Logistic Regression on the logit of clipped uncalibrated probabilities."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.lr_ = None

    def fit(self, probs: np.ndarray, y: np.ndarray):
        probs_clipped = np.clip(probs, self.eps, 1.0 - self.eps)
        logits = logit(probs_clipped).reshape(-1, 1)
        self.lr_ = LogisticRegression(penalty=None, solver="lbfgs")
        self.lr_.fit(logits, y)
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        probs_clipped = np.clip(probs, self.eps, 1.0 - self.eps)
        logits = logit(probs_clipped).reshape(-1, 1)
        p1 = self.lr_.predict_proba(logits)[:, 1]
        return np.column_stack([1.0 - p1, p1])


class IsotonicCalibrator(BaseEstimator, ClassifierMixin):
    """Isotonic regression calibrator with clipping at boundaries."""

    def __init__(self):
        self.iso_ = None

    def fit(self, probs: np.ndarray, y: np.ndarray):
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.iso_.fit(probs, y)
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        p1 = self.iso_.predict(probs)
        return np.column_stack([1.0 - p1, p1])


class FullCalibratedPipeline(BaseEstimator, ClassifierMixin):
    """End-to-end wrapper combining raw feature engineering, preprocessor, base model, and calibrator.
    
    Accepts raw encounter DataFrame, extracts features, transforms, predicts raw probability,
    and applies post-hoc calibration. Exposes predict_proba, predict, and __sklearn_is_fitted__
    for compatibility with Fairlearn ThresholdOptimizer.
    """

    def __init__(self, feature_pipeline, base_model, calibrator):
        self.feature_pipeline = feature_pipeline
        self.base_model = base_model
        self.calibrator = calibrator
        self.classes_ = np.array([0, 1])

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def fit(self, X, y=None, **fit_params):
        # No-op or refit of calibrator only if desired
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Handles raw DataFrame inputs
        X_trans = self.feature_pipeline.transform(X)
        raw_probs = self.base_model.predict_proba(X_trans)[:, 1]
        return self.calibrator.predict_proba(raw_probs)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


def plot_calibration_curves(
    y_val: np.ndarray,
    prob_dict: dict[str, np.ndarray],
    save_path: str | pathlib.Path,
    n_bins: int = 10,
):
    """Plots reliability curves with a 45-degree reference line and prevalence marker."""
    _fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)")

    prevalence = float(np.mean(y_val))
    ax.axvline(x=prevalence, color="gray", linestyle=":", label=f"Prevalence ({prevalence:.1%})")

    for name, p in prob_dict.items():
        frac_pos, mean_pred = calibration_curve(y_val, p, n_bins=n_bins, strategy="uniform")
        ece = compute_ece(y_val, p, n_bins=n_bins, strategy="uniform")
        ax.plot(mean_pred, frac_pos, "s-", label=f"{name} (ECE: {ece:.4f})")

    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Observed Fraction of Positives", fontsize=11)
    ax.set_title("Validation Reliability Curve (Calibration)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved calibration curve to {save_path}")


def run_calibration_step(config_dir: str = "configs"):
    """Loads validation predictions, fits Platt & Isotonic calibrators, selects best, and saves artifacts."""
    logger.info("Starting M5 Calibration and Thresholds step...")

    with open(f"{config_dir}/base.yaml") as f:
        base_cfg = yaml.safe_load(f)

    artifacts_dir = pathlib.Path(base_cfg["paths"]["artifacts"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = pathlib.Path(base_cfg["paths"]["reports"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load validation data and base artifacts
    val_df = pd.read_parquet("data/processed/val.parquet")
    y_val = val_df["readmit_30d"].to_numpy().astype(int)

    feature_pipeline = joblib.load(artifacts_dir / "feature_pipeline.joblib")
    xgb_model = joblib.load(artifacts_dir / "xgboost.joblib")
    lr_model = joblib.load(artifacts_dir / "logistic_regression.joblib")

    X_val = feature_pipeline.transform(val_df)
    raw_xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
    raw_lr_probs = lr_model.predict_proba(X_val)[:, 1]

    # 2. Fit Platt Scaling & Isotonic Regression on Validation Split
    platt_calibrator = PlattCalibrator()
    platt_calibrator.fit(raw_xgb_probs, y_val)
    xgb_platt_probs = platt_calibrator.predict_proba(raw_xgb_probs)[:, 1]

    iso_calibrator = IsotonicCalibrator()
    iso_calibrator.fit(raw_xgb_probs, y_val)
    xgb_iso_probs = iso_calibrator.predict_proba(raw_xgb_probs)[:, 1]

    # Evaluate metrics on validation
    m_raw = compute_all_metrics(y_val, raw_xgb_probs)
    m_platt = compute_all_metrics(y_val, xgb_platt_probs)
    m_iso = compute_all_metrics(y_val, xgb_iso_probs)

    logger.info(f"Raw XGBoost: Brier={m_raw['brier']:.4f}, ECE={m_raw['ece']:.4f}, PR-AUC={m_raw['pr_auc']:.4f}")
    logger.info(f"Platt XGBoost: Brier={m_platt['brier']:.4f}, ECE={m_platt['ece']:.4f}, PR-AUC={m_platt['pr_auc']:.4f}")
    logger.info(f"Isotonic XGBoost: Brier={m_iso['brier']:.4f}, ECE={m_iso['ece']:.4f}, PR-AUC={m_iso['pr_auc']:.4f}")

    # Plot reliability curves
    prob_dict = {
        "XGBoost (Raw)": raw_xgb_probs,
        "XGBoost (Platt)": xgb_platt_probs,
        "XGBoost (Isotonic)": xgb_iso_probs,
        "Logistic Regression": raw_lr_probs,
    }
    plot_calibration_curves(y_val, prob_dict, figures_dir / "calibration_curve_xgb.png")

    # Select best calibration method: Platt preserves rank order strictly and avoids piecewise-constant steps
    best_calibrator = platt_calibrator if m_platt["brier"] <= m_iso["brier"] else iso_calibrator
    best_method = "sigmoid" if best_calibrator == platt_calibrator else "isotonic"
    chosen_probs = xgb_platt_probs if best_method == "sigmoid" else xgb_iso_probs
    chosen_metrics = m_platt if best_method == "sigmoid" else m_iso

    # Build and save end-to-end full calibrated pipeline
    full_calibrated_xgb = FullCalibratedPipeline(
        feature_pipeline=feature_pipeline,
        base_model=xgb_model,
        calibrator=best_calibrator,
    )
    joblib.dump(full_calibrated_xgb, artifacts_dir / "xgboost_calibrated.joblib")
    logger.info(f"Saved full calibrated pipeline ({best_method}) to artifacts/xgboost_calibrated.joblib")

    # 3. Generate reports/capacity_table.md
    k_percents = base_cfg["decision"].get("capacity_k_percent", [5, 10, 20])
    cap_table = compute_capacity_metrics(y_val, chosen_probs, k_percents=k_percents)

    cost_opt = chosen_metrics["cost_ratio_optimal"]
    prev = chosen_metrics["prevalence"]

    md_lines = [
        "# Capacity and Decision-Support Thresholds Report\n",
        "> [!NOTE]\n",
        "> **Operational Assumption:** Hospital intervention capacity determines practical screening utility. Defaulting to an arbitrary 0.50 classification threshold is clinically invalid when baseline readmission prevalence is ~8.98%. All cost ratios and capacity tiers are explicit operational assumptions labeled below.\n",
        "\n## Capacity-Constrained Resource Prioritization (Validation Set)\n",
        f"- **Validation Cohort Size:** {len(y_val):,} patients\n",
        f"- **Observed Baseline Readmission Prevalence:** {prev * 100:.2f}%\n\n",
        "| Capacity Tier (Top K%) | Patients Flagged | Risk Cutoff Threshold | Captured Readmissions | Recall (Sensitivity) | Precision (PPV) | Lift over Baseline |",
        "|---|---|---|---|---|---|---|",
    ]

    for row in cap_table:
        md_lines.append(
            f"| Top {row['k_percent']}% | {row['n_flagged']:,} | {row['threshold'] * 100:.2f}% | {row['positives_captured']:,} | {row['recall'] * 100:.2f}% | {row['precision'] * 100:.2f}% | {row['lift']:.2f}x |"
        )

    md_lines.extend([
        "\n## Cost-Ratio Utility Optimization (Illustrative Assumption)\n",
        f"- **Cost Ratio (Cost_FN / Cost_FP):** {cost_opt['cost_ratio_fn_to_fp']:.1f}x (Missing a readmission assumed 5x more costly than false alarm review).\n",
        f"- **Analytically Optimal Probability Threshold:** {cost_opt['optimal_threshold'] * 100:.2f}%\n",
        f"- **Precision at Optimal Threshold:** {cost_opt['precision'] * 100:.2f}%\n",
        f"- **Recall at Optimal Threshold:** {cost_opt['recall'] * 100:.2f}%\n",
        f"- **F1 Score at Optimal Threshold:** {cost_opt['f1']:.4f}\n",
        "\n## Calibration Summary (Brier & ECE)\n",
        "| Variant | Brier Score (lower is better) | ECE (uniform 10 bins) | PR-AUC |",
        "|---|---|---|---|",
        f"| Raw XGBoost | {m_raw['brier']:.4f} | {m_raw['ece']:.4f} | {m_raw['pr_auc']:.4f} |",
        f"| Platt Scaling (Sigmoid) | {m_platt['brier']:.4f} | {m_platt['ece']:.4f} | {m_platt['pr_auc']:.4f} |",
        f"| Isotonic Regression | {m_iso['brier']:.4f} | {m_iso['ece']:.4f} | {m_iso['pr_auc']:.4f} |",
        f"\n**Selected Calibration Method:** `{best_method}`. Preserves monotonicity and output stability.",
    ])

    with open(reports_dir / "capacity_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    logger.info("Saved capacity table to reports/capacity_table.md")
