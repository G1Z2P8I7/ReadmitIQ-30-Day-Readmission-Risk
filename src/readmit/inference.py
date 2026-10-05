"""Inference engine for ReadmitIQ: loads calibrated pipeline and explainer for patient risk prediction and local SHAP explanations."""

import json
import logging
import pathlib

import joblib
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

# Global cache for inference assets
_MODEL_PIPELINE = None
_EXPLAINER = None
_FEATURE_NAMES = None
_PREVALENCE = 0.0898


def load_inference_artifacts(
    artifacts_dir: str | pathlib.Path = "artifacts", reports_dir: str | pathlib.Path = "reports"
):
    """Loads calibrated pipeline, explainer, feature names, and baseline prevalence."""
    global _MODEL_PIPELINE, _EXPLAINER, _FEATURE_NAMES, _PREVALENCE
    if _MODEL_PIPELINE is not None:
        return _MODEL_PIPELINE, _EXPLAINER, _FEATURE_NAMES, _PREVALENCE

    art_dir = pathlib.Path(artifacts_dir)
    rep_dir = pathlib.Path(reports_dir)

    model_path = art_dir / "xgboost_calibrated.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")

    _MODEL_PIPELINE = joblib.load(model_path)
    base_xgb = _MODEL_PIPELINE.base_model
    _EXPLAINER = shap.TreeExplainer(base_xgb)

    preprocessor = _MODEL_PIPELINE.feature_pipeline.named_steps["preprocessor"]
    raw_names = list(preprocessor.get_feature_names_out())
    _FEATURE_NAMES = [
        f.replace("num__", "").replace("bin__", "").replace("cat__", "") for f in raw_names
    ]

    # Load prevalence from metrics.json if available
    metrics_path = rep_dir / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, encoding="utf-8") as f:
                data = json.load(f)
                _PREVALENCE = float(data.get("cohort", {}).get("prevalence", 0.0898))
        except Exception:
            _PREVALENCE = 0.0898

    return _MODEL_PIPELINE, _EXPLAINER, _FEATURE_NAMES, _PREVALENCE


def get_risk_band(probability: float) -> str:
    """Categorizes risk probability into actionable clinical bands."""
    if probability < 0.06:
        return "Low Risk (<6%)"
    if probability < 0.12:
        return "Average Risk (6-12%)"
    if probability < 0.20:
        return "Moderate Risk (12-20%)"
    return "High Risk (>20%)"


def predict(patient_data: dict | pd.DataFrame) -> dict:
    """Predicts 30-day readmission risk probability, relative risk vs average, and risk band."""
    pipeline, _, _, prevalence = load_inference_artifacts()

    if isinstance(patient_data, dict):
        df = pd.DataFrame([patient_data])
    else:
        df = patient_data.copy()

    probs = pipeline.predict_proba(df)[:, 1]
    p = float(probs[0])
    risk_vs_avg = float(p / prevalence) if prevalence > 0 else 1.0
    band = get_risk_band(p)

    return {
        "risk_probability": round(p, 4),
        "risk_vs_average": round(risk_vs_avg, 2),
        "risk_band": band,
        "cohort_prevalence": round(prevalence, 4),
    }


def explain(patient_data: dict | pd.DataFrame, top_k: int = 5) -> dict:
    """Returns the top K positive and negative SHAP feature contributions for a single patient."""
    pipeline, explainer, feature_names, _ = load_inference_artifacts()

    if isinstance(patient_data, dict):
        df = pd.DataFrame([patient_data])
    else:
        df = patient_data.copy()

    # Transform features through feature pipeline
    X_trans = pipeline.feature_pipeline.transform(df)
    shap_vals = explainer.shap_values(X_trans)[0]

    # Sort contributions by absolute value
    idx_sorted = np.argsort(-np.abs(shap_vals))
    top_indices = idx_sorted[:top_k]

    contributions = []
    for i in top_indices:
        contributions.append(
            {
                "feature": feature_names[i],
                "shap_value": round(float(shap_vals[i]), 4),
                "direction": "increases_risk" if shap_vals[i] > 0 else "decreases_risk",
            }
        )

    base_val = explainer.expected_value
    base_val_float = float(base_val if np.isscalar(base_val) else base_val[0])

    return {
        "base_value_log_odds": round(base_val_float, 4),
        "top_contributions": contributions,
    }
