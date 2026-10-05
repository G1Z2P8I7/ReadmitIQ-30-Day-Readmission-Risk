"""Metrics, calibration evaluation, capacity analysis, and bootstrap confidence intervals."""

import logging
import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, strategy: str = "uniform") -> float:
    """Computes Expected Calibration Error (ECE) across probability bins.
    
    strategy='uniform' partitions [0, 1] into equal-width bins.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        bins = np.percentile(y_prob, np.linspace(0.0, 100.0, n_bins + 1))
        bins[0] = 0.0
        bins[-1] = 1.0
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")

    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = bin_indices == i
        bin_size = np.sum(mask)
        if bin_size > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_capacity_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    k_percents: list[int] | None = None,
) -> list[dict]:
    """Computes capacity-based intervention metrics.
    
    Sorts patients descending by predicted risk (stable sort for ties),
    flags the top ceil(K% * n), and computes Recall, Precision, Lift, and Flagged Count.
    """
    if k_percents is None:
        k_percents = [5, 10, 20]

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    total_positives = int(np.sum(y_true))
    prevalence = total_positives / n if n > 0 else 0.0

    # Stable sort descending by predicted probability
    # Using kind='mergesort' for stability
    sort_order = np.argsort(-y_prob, kind="mergesort")
    y_true_sorted = y_true[sort_order]

    results = []
    for k in k_percents:
        n_flagged = max(1, math.ceil(n * (k / 100.0)))
        flagged_true = y_true_sorted[:n_flagged]
        positives_flagged = int(np.sum(flagged_true))

        recall = positives_flagged / total_positives if total_positives > 0 else 0.0
        precision = positives_flagged / n_flagged if n_flagged > 0 else 0.0
        lift = precision / prevalence if prevalence > 0 else 0.0

        # Cutoff threshold (minimum probability in flagged group)
        threshold_at_k = float(y_prob[sort_order[n_flagged - 1]])

        results.append({
            "k_percent": k,
            "n_flagged": n_flagged,
            "threshold": threshold_at_k,
            "positives_captured": positives_flagged,
            "recall": float(recall),
            "precision": float(precision),
            "lift": float(lift),
        })

    return results


def compute_all_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    primary_k: int = 20,
    cost_ratio: float = 5.0,
) -> dict:
    """Computes comprehensive discrimination, calibration, and decision metrics."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    ece = compute_ece(y_true, y_prob, n_bins=10, strategy="uniform")

    cap_metrics = compute_capacity_metrics(y_true, y_prob, k_percents=[5, 10, 20])
    primary_cap = next((m for m in cap_metrics if m["k_percent"] == primary_k), cap_metrics[-1])

    # Cost-ratio optimal threshold: Cost = Cost_FN * FN + Cost_FP * FP = cost_ratio * FN + FP
    thresholds = np.linspace(0.01, 0.99, 100)
    best_cost = float("inf")
    best_thresh = 0.5
    for t in thresholds:
        pred_pos = y_prob >= t
        fn = np.sum((y_true == 1) & (~pred_pos))
        fp = np.sum((y_true == 0) & pred_pos)
        cost = cost_ratio * fn + fp
        if cost < best_cost:
            best_cost = cost
            best_thresh = float(t)

    pred_at_best = (y_prob >= best_thresh).astype(int)
    cost_opt_precision = float(precision_score(y_true, pred_at_best, zero_division=0))
    cost_opt_recall = float(recall_score(y_true, pred_at_best, zero_division=0))
    cost_opt_f1 = float(f1_score(y_true, pred_at_best, zero_division=0))

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier": brier,
        "ece": ece,
        "prevalence": float(np.mean(y_true)),
        "capacity_metrics": cap_metrics,
        "primary_capacity": primary_cap,
        "cost_ratio_optimal": {
            "cost_ratio_fn_to_fp": cost_ratio,
            "optimal_threshold": best_thresh,
            "precision": cost_opt_precision,
            "recall": cost_opt_recall,
            "f1": cost_opt_f1,
        },
    }


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, tuple[float, float]]:
    """Computes empirical 95% bootstrap confidence interval using seeded resampling."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point_val = float(metric_fn(y_true, y_prob))

    boot_vals = []
    for _ in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        sample_y = y_true[indices]
        sample_p = y_prob[indices]
        # Skip if sample has only one class
        if len(np.unique(sample_y)) < 2:
            continue
        boot_vals.append(float(metric_fn(sample_y, sample_p)))

    lower_pct = ((1.0 - ci) / 2.0) * 100
    upper_pct = (1.0 - (1.0 - ci) / 2.0) * 100
    ci_lower = float(np.percentile(boot_vals, lower_pct))
    ci_upper = float(np.percentile(boot_vals, upper_pct))

    return point_val, (ci_lower, ci_upper)
