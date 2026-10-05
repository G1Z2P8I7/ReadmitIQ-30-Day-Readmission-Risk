"""Unit tests for feature engineering, baseline models, and evaluation metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from readmit.evaluation import (
    bootstrap_metric_ci,
    compute_all_metrics,
    compute_capacity_metrics,
)
from readmit.features import (
    ClinicalFeatureEngineer,
    build_preprocessor_pipeline,
    group_icd9,
)


def test_icd_grouping():
    """Known ICD-9 codes map to expected categories; unknown codes map to other/unknown."""
    # Circulatory: 390-459, 785
    assert group_icd9("410") == "Circulatory"
    assert group_icd9("428.0") == "Circulatory"
    assert group_icd9("785") == "Circulatory"

    # Diabetes: 250.xx
    assert group_icd9("250.02") == "Diabetes"
    assert group_icd9("250") == "Diabetes"

    # Respiratory: 460-519, 786
    assert group_icd9("493") == "Respiratory"
    assert group_icd9("786") == "Respiratory"

    # Digestive: 520-579, 787
    assert group_icd9("540") == "Digestive"
    assert group_icd9("787") == "Digestive"

    # Genitourinary: 580-629, 788
    assert group_icd9("599.0") == "Genitourinary"
    assert group_icd9("788") == "Genitourinary"

    # Neoplasms: 140-239
    assert group_icd9("174.9") == "Neoplasms"

    # Musculoskeletal: 710-739
    assert group_icd9("715") == "Musculoskeletal"

    # Injury: 800-999
    assert group_icd9("820") == "Injury"

    # Other / Missing
    assert group_icd9("E876") == "Other"
    assert group_icd9("V45") == "Other"
    assert group_icd9("?") == "Unknown"
    assert group_icd9("") == "Unknown"


def test_no_forbidden_features():
    """Verify feature pipeline outputs contain no target, identifiers, race, or gender."""
    sample_df = pd.DataFrame(
        {
            "encounter_id": [1, 2],
            "patient_nbr": [101, 102],
            "race": ["Caucasian", "AfricanAmerican"],
            "gender": ["Female", "Male"],
            "age": ["[50-60)", "[70-80)"],
            "admission_type_id": [1, 2],
            "discharge_disposition_id": [1, 3],
            "admission_source_id": [7, 1],
            "time_in_hospital": [3, 5],
            "payer_code": ["MC", "MD"],
            "medical_specialty": ["Cardiology", "InternalMedicine"],
            "num_lab_procedures": [40, 50],
            "num_procedures": [1, 2],
            "num_medications": [10, 15],
            "number_outpatient": [0, 1],
            "number_emergency": [0, 0],
            "number_inpatient": [1, 0],
            "diag_1": ["414", "250.0"],
            "diag_2": ["250", "401"],
            "diag_3": ["401", "428"],
            "number_diagnoses": [5, 7],
            "max_glu_serum": ["None", ">200"],
            "A1Cresult": ["None", ">8"],
            "metformin": ["No", "Steady"],
            "glipizide": ["No", "No"],
            "glyburide": ["No", "No"],
            "pioglitazone": ["No", "No"],
            "rosiglitazone": ["No", "No"],
            "insulin": ["Steady", "Up"],
            "change": ["No", "Ch"],
            "diabetesMed": ["Yes", "Yes"],
            "readmitted": ["<30", "NO"],
            "readmit_30d": [1, 0],
        }
    )

    engineer = ClinicalFeatureEngineer(top_n_specialties=5, top_n_payers=5)
    engineer.fit(sample_df)
    features_df = engineer.transform(sample_df)

    numeric_cols = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "age_midpoint",
        "prior_visits_total",
        "n_meds_active",
        "n_meds_changed",
        "n_meds_steady",
        "n_distinct_diag_groups",
    ]
    binary_cols = [
        "any_prior_inpatient",
        "a1c_tested",
        "glu_tested",
        "change",
        "diabetesMed",
    ]
    categorical_cols = [
        "admission_type_group",
        "admission_source_group",
        "discharge_group",
        "medical_specialty_group",
        "payer_code_group",
        "a1c_result_cat",
        "glu_serum_cat",
        "diag_1_group",
        "diag_2_group",
        "diag_3_group",
    ]

    preprocessor = build_preprocessor_pipeline(numeric_cols, binary_cols, categorical_cols)
    preprocessor.fit(features_df)
    feat_names = list(preprocessor.get_feature_names_out())

    forbidden = ["race", "gender", "readmitted", "readmit_30d", "patient_nbr", "encounter_id"]
    for f in feat_names:
        for forbid in forbidden:
            assert forbid not in f.lower(), f"Forbidden substring '{forbid}' found in feature '{f}'"


def test_preprocessing_fit_on_train_only():
    """Fitted statistics (means, categories) are identical whether val/test rows exist or not."""
    train_df = pd.DataFrame(
        {
            "time_in_hospital": [2, 4, 6],
            "number_inpatient": [0, 1, 2],
            "admission_type_group": ["Emergency", "Elective", "Emergency"],
        }
    )
    val_df = pd.DataFrame(
        {
            "time_in_hospital": [10, 15, 20],  # Out-of-distribution values
            "number_inpatient": [5, 10, 15],
            "admission_type_group": ["Urgent", "Other", "Emergency"],
        }
    )

    prep = build_preprocessor_pipeline(
        numeric_cols=["time_in_hospital", "number_inpatient"],
        binary_cols=[],
        categorical_cols=["admission_type_group"],
    )

    # Fit strictly on train
    prep.fit(train_df)
    train_mean = prep.named_transformers_["num"].named_steps["scaler"].mean_

    # Verify that transforming val does not mutate fitted means
    prep.transform(val_df)
    assert np.allclose(prep.named_transformers_["num"].named_steps["scaler"].mean_, train_mean)


def test_capacity_threshold():
    """Flagging top K% flags that share of rows (within rounding) on toy scores."""
    y_true = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])  # 2 positives / 10 rows
    y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])

    # K=20% -> 2 patients flagged. Highest scores are index 0 (y=1) and index 1 (y=0).
    cap = compute_capacity_metrics(y_true, y_prob, k_percents=[20])
    assert len(cap) == 1
    assert cap[0]["n_flagged"] == 2
    assert cap[0]["positives_captured"] == 1
    assert cap[0]["recall"] == 0.5  # 1 out of 2 captured
    assert cap[0]["precision"] == 0.5  # 1 out of 2 flagged is positive
    assert cap[0]["lift"] == 2.5  # 0.50 / 0.20 prevalence = 2.5x


def test_metrics_on_toy_data():
    """PR-AUC, ROC-AUC, ECE match expected bounds on toy predictions."""
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.8, 0.2, 0.7, 0.3])

    metrics = compute_all_metrics(y_true, y_prob, primary_k=20, cost_ratio=5.0)
    assert metrics["roc_auc"] == 1.0  # Perfect separation
    assert metrics["pr_auc"] == 1.0
    assert 0.0 <= metrics["ece"] <= 1.0
    assert 0.0 <= metrics["brier"] <= 1.0


def test_bootstrap_reproducible():
    """Same seed gives exact same confidence intervals."""
    rng = np.random.RandomState(42)
    y_true = rng.binomial(1, 0.1, size=100)
    y_prob = rng.uniform(0, 1, size=100)

    val1, ci1 = bootstrap_metric_ci(y_true, y_prob, roc_auc_score, n_resamples=50, seed=123)
    val2, ci2 = bootstrap_metric_ci(y_true, y_prob, roc_auc_score, n_resamples=50, seed=123)

    assert val1 == val2
    assert ci1 == ci2


def test_calibration_monotonic():
    """Calibrators preserve score ordering monotonically."""
    from readmit.calibration import IsotonicCalibrator, PlattCalibrator

    scores = np.linspace(0.05, 0.95, 20)
    # Binary targets generally correlated with scores
    y = (scores > 0.5).astype(int)

    # 1. Platt scaling test
    platt = PlattCalibrator()
    platt.fit(scores, y)
    cal_platt = platt.predict_proba(scores)[:, 1]
    assert np.all(np.diff(cal_platt) >= -1e-6), (
        "Platt calibrator did not produce monotonically increasing probabilities"
    )

    # 2. Isotonic regression test
    iso = IsotonicCalibrator()
    iso.fit(scores, y)
    cal_iso = iso.predict_proba(scores)[:, 1]
    assert np.all(np.diff(cal_iso) >= 0.0), (
        "Isotonic calibrator did not produce monotonically non-decreasing probabilities"
    )


def test_fairness_gaps_toy():
    """TPR/FPR/selection-rate gaps match hand-computed values on a toy frame."""
    from readmit.fairness import compute_group_fairness_table, compute_parity_gaps

    # Group A: 2 patients, y_true=[1, 0], y_pred=[1, 0] -> TPR=1.0, FPR=0.0, Sel=0.5
    # Group B: 2 patients, y_true=[1, 0], y_pred=[0, 1] -> TPR=0.0, FPR=1.0, Sel=0.5
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1])
    groups = pd.Series(["A", "A", "B", "B"])

    df_grp = compute_group_fairness_table(y_true, y_pred, groups, min_group_n=1)
    gaps = compute_parity_gaps(df_grp)

    # TPR gap: |1.0 - 0.0| = 1.0
    assert abs(gaps["tpr_gap"] - 1.0) < 1e-6
    # FPR gap: |0.0 - 1.0| = 1.0
    assert abs(gaps["fpr_gap"] - 1.0) < 1e-6
    # Selection rate gap: |0.5 - 0.5| = 0.0
    assert abs(gaps["selection_rate_gap"] - 0.0) < 1e-6
