"""Feature engineering pipeline, ICD-9 clinical grouping, and baseline models."""

import logging
import pathlib

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from readmit.evaluation import compute_all_metrics

logger = logging.getLogger(__name__)


# Standard ICD-9 clinical grouping per Appendix A
def group_icd9(code: str | float | None) -> str:
    """Groups ICD-9 diagnosis codes into standard clinical categories."""
    if code is None or pd.isna(code) or str(code).strip() in {"?", "", "nan", "None"}:
        return "Unknown"
    c = str(code).strip().upper()
    if c.startswith(("V", "E")):
        return "Other"
    try:
        val = float(c)
    except ValueError:
        return "Other"

    if (390 <= val <= 459) or (val == 785):
        return "Circulatory"
    if (460 <= val <= 519) or (val == 786):
        return "Respiratory"
    if (520 <= val <= 579) or (val == 787):
        return "Digestive"
    if 250 <= val < 251:
        return "Diabetes"
    if 800 <= val <= 999:
        return "Injury"
    if 710 <= val <= 739:
        return "Musculoskeletal"
    if (580 <= val <= 629) or (val == 788):
        return "Genitourinary"
    if 140 <= val <= 239:
        return "Neoplasms"
    return "Other"


AGE_MIDPOINTS = {
    "[0-10)": 5.0,
    "[10-20)": 15.0,
    "[20-30)": 25.0,
    "[30-40)": 35.0,
    "[40-50)": 45.0,
    "[50-60)": 55.0,
    "[60-70)": 65.0,
    "[70-80)": 75.0,
    "[80-90)": 85.0,
    "[90-100)": 95.0,
}

MEDICATION_COLS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transforms raw clinical encounter frame into engineered feature matrix.

    Fits categorical vocabularies (e.g. top specialties) on training data only.
    """

    def __init__(self, top_n_specialties: int = 10, top_n_payers: int = 5):
        self.top_n_specialties = top_n_specialties
        self.top_n_payers = top_n_payers
        self.top_specialties_ = []
        self.top_payers_ = []

    def fit(self, X: pd.DataFrame, y=None):
        # Learn top medical specialties from train
        spec_series = X["medical_specialty"].replace("?", "Unknown").fillna("Unknown")
        spec_counts = spec_series[spec_series != "Unknown"].value_counts()
        self.top_specialties_ = list(spec_counts.head(self.top_n_specialties).index)

        # Learn top payer codes from train
        payer_series = X["payer_code"].replace("?", "Unknown").fillna("Unknown")
        payer_counts = payer_series[payer_series != "Unknown"].value_counts()
        self.top_payers_ = list(payer_counts.head(self.top_n_payers).index)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Demographics & Auditing Attributes
        df["age_midpoint"] = df["age"].map(AGE_MIDPOINTS).fillna(55.0)

        def map_age_band(age_str):
            if age_str in {"[0-10)", "[10-20)", "[20-30)"}:
                return "<30"
            if age_str in {"[30-40)", "[40-50)"}:
                return "30-49"
            if age_str in {"[50-60)", "[60-70)"}:
                return "50-69"
            return "70+"

        df["age_band"] = df["age"].map(map_age_band)
        df["race_group"] = df["race"].replace("?", "Unknown").fillna("Unknown")

        # 2. Prior Utilization
        df["prior_visits_total"] = (
            df["number_outpatient"].fillna(0)
            + df["number_emergency"].fillna(0)
            + df["number_inpatient"].fillna(0)
        )
        df["any_prior_inpatient"] = (df["number_inpatient"].fillna(0) > 0).astype(int)

        # 3. ICD-9 Clinical Groups
        df["diag_1_group"] = df["diag_1"].apply(group_icd9)
        df["diag_2_group"] = df["diag_2"].apply(group_icd9)
        df["diag_3_group"] = df["diag_3"].apply(group_icd9)

        # Count of distinct non-unknown diagnosis categories
        def count_distinct_groups(row):
            groups = {row["diag_1_group"], row["diag_2_group"], row["diag_3_group"]}
            groups.discard("Unknown")
            return len(groups)

        df["n_distinct_diag_groups"] = df.apply(count_distinct_groups, axis=1)

        # 4. Medication Dynamics
        active_med_cols = [c for c in MEDICATION_COLS if c in df.columns]
        med_sub = df[active_med_cols].fillna("No")

        df["n_meds_active"] = (med_sub != "No").sum(axis=1)
        df["n_meds_changed"] = med_sub.isin(["Up", "Down"]).sum(axis=1)
        df["n_meds_steady"] = (med_sub == "Steady").sum(axis=1)

        df["change"] = (df["change"] == "Ch").astype(int)
        df["diabetesMed"] = (df["diabetesMed"] == "Yes").astype(int)

        # 5. Glycemic Testing (A1C & Glucose)
        df["a1c_tested"] = (df["A1Cresult"].replace("None", np.nan).notna()).astype(int)
        df["a1c_result_cat"] = df["A1Cresult"].fillna("none").replace({"None": "none"})

        df["glu_tested"] = (df["max_glu_serum"].replace("None", np.nan).notna()).astype(int)
        df["glu_serum_cat"] = df["max_glu_serum"].fillna("none").replace({"None": "none"})

        # 6. Admission and Discharge Grouping
        def map_admission_type(val):
            try:
                v = int(val)
                if v == 1:
                    return "Emergency"
                if v == 2:
                    return "Urgent"
                if v == 3:
                    return "Elective"
                return "Other"
            except (ValueError, TypeError):
                return "Other"

        def map_admission_source(val):
            try:
                v = int(val)
                if v == 7:
                    return "Emergency_Room"
                if v == 1:
                    return "Physician_Referral"
                return "Other"
            except (ValueError, TypeError):
                return "Other"

        def map_discharge(val):
            try:
                v = int(val)
                if v == 1:
                    return "Home"
                if v == 6:
                    return "Home_Health"
                if v == 3:
                    return "SNF"
                return "Other"
            except (ValueError, TypeError):
                return "Other"

        df["admission_type_group"] = df["admission_type_id"].apply(map_admission_type)
        df["admission_source_group"] = df["admission_source_id"].apply(map_admission_source)
        df["discharge_group"] = df["discharge_disposition_id"].apply(map_discharge)

        # 7. Medical Specialty & Payer Code
        spec_clean = df["medical_specialty"].replace("?", "Unknown").fillna("Unknown")
        df["medical_specialty_group"] = spec_clean.apply(
            lambda x: (
                x if x in self.top_specialties_ else ("Unknown" if x == "Unknown" else "Other")
            )
        )

        payer_clean = df["payer_code"].replace("?", "Unknown").fillna("Unknown")
        df["payer_code_group"] = payer_clean.apply(
            lambda x: x if x in self.top_payers_ else ("Unknown" if x == "Unknown" else "Other")
        )

        return df


def build_preprocessor_pipeline(
    numeric_cols: list[str],
    binary_cols: list[str],
    categorical_cols: list[str],
) -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer that scales numerics and one-hot encodes categoricals."""
    num_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        [
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("bin", "passthrough", binary_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor


class PrevalenceBaseline(BaseEstimator, ClassifierMixin):
    """Prevalence baseline: outputs constant train positive rate as risk estimate."""

    def __init__(self):
        self.prevalence_ = 0.0

    def fit(self, X, y):
        self.prevalence_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        n = len(X)
        p = np.full(n, self.prevalence_)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class PriorInpatientRuleBaseline(BaseEstimator, ClassifierMixin):
    """Rule baseline: ranks patients by prior inpatient visits (with total prior visits as tie-break)."""

    def __init__(self):
        self.is_fitted_ = False

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame):
        # Score proportional to number_inpatient and prior_visits_total
        inp = X["number_inpatient"].fillna(0).values.astype(float)
        tot = (
            X["number_outpatient"].fillna(0) + X["number_emergency"].fillna(0) + inp
        ).values.astype(float)

        raw_score = inp * 10.0 + tot
        min_s = np.min(raw_score)
        max_s = np.max(raw_score)
        if max_s > min_s:
            prob = (raw_score - min_s) / (max_s - min_s)
        else:
            prob = np.zeros(len(X))

        # Scale to calibrated range around baseline prevalence
        prob = np.clip(prob * 0.40, 0.01, 0.99)
        return np.column_stack([1.0 - prob, prob])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.20).astype(int)


def save_feature_dictionary(output_path: str = "reports/feature_dictionary.md") -> None:
    p = pathlib.Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    md = """# Feature Dictionary & Specification

Every feature engineered for the ReadmitIQ modeling pipeline, its role, clinical rationale, and transformation logic.

| Feature Name | Type | Source Columns | Transformation & Description |
|---|---|---|---|
| `time_in_hospital` | Numeric (Scaled) | `time_in_hospital` | Inpatient duration in days. Standardized (mean 0, std 1). |
| `num_lab_procedures` | Numeric (Scaled) | `num_lab_procedures` | Number of laboratory tests administered during encounter. Standardized. |
| `num_procedures` | Numeric (Scaled) | `num_procedures` | Number of non-lab procedures performed. Standardized. |
| `num_medications` | Numeric (Scaled) | `num_medications` | Total distinct medications administered. Standardized. |
| `number_outpatient` | Numeric (Scaled) | `number_outpatient` | Outpatient visits in preceding 12 months. Standardized. |
| `number_emergency` | Numeric (Scaled) | `number_emergency` | Emergency department encounters in preceding 12 months. Standardized. |
| `number_inpatient` | Numeric (Scaled) | `number_inpatient` | Prior inpatient admissions in preceding 12 months. Standardized. |
| `number_diagnoses` | Numeric (Scaled) | `number_diagnoses` | Number of ICD-9 diagnosis entries on encounter record. Standardized. |
| `age_midpoint` | Numeric (Scaled) | `age` | Midpoint of 10-year age bracket (e.g. `[60-70)` mapped to 65.0). Standardized. |
| `prior_visits_total` | Numeric (Scaled) | `number_outpatient`, `number_emergency`, `number_inpatient` | Sum of all outpatient, emergency, and inpatient encounters in preceding 12 months. |
| `n_meds_active` | Numeric (Scaled) | 23 medication columns | Number of distinct diabetes medications currently active (not 'No'). |
| `n_meds_changed` | Numeric (Scaled) | 23 medication columns | Number of diabetes medications adjusted during stay ('Up' or 'Down'). |
| `n_meds_steady` | Numeric (Scaled) | 23 medication columns | Number of diabetes medications kept steady. |
| `n_distinct_diag_groups` | Numeric (Scaled) | `diag_1`, `diag_2`, `diag_3` | Number of distinct ICD-9 clinical organ systems represented across diagnoses. |
| `any_prior_inpatient` | Binary | `number_inpatient` | Binary flag (1 if `number_inpatient > 0`, 0 otherwise). |
| `a1c_tested` | Binary | `A1Cresult` | Binary flag (1 if HbA1c test result recorded, 0 if 'None'). |
| `glu_tested` | Binary | `max_glu_serum` | Binary flag (1 if serum glucose test performed, 0 if 'None'). |
| `change` | Binary | `change` | Binary flag (1 if diabetic medication dosage or regimen changed, 0 otherwise). |
| `diabetesMed` | Binary | `diabetesMed` | Binary flag (1 if any diabetes medication was prescribed). |
| `admission_type_group` | Categorical (OHE) | `admission_type_id` | Grouped into 'Emergency', 'Urgent', 'Elective', 'Other'. One-hot encoded. |
| `admission_source_group` | Categorical (OHE) | `admission_source_id` | Grouped into 'Emergency_Room', 'Physician_Referral', 'Other'. One-hot encoded. |
| `discharge_group` | Categorical (OHE) | `discharge_disposition_id` | Grouped into 'Home', 'Home_Health', 'SNF', 'Other'. One-hot encoded. |
| `medical_specialty_group` | Categorical (OHE) | `medical_specialty` | Top 10 specialties on training set; others grouped to 'Other', missing to 'Unknown'. |
| `payer_code_group` | Categorical (OHE) | `payer_code` | Top 5 payment codes on training set; others grouped to 'Other', missing to 'Unknown'. |
| `a1c_result_cat` | Categorical (OHE) | `A1Cresult` | Categories: `>8`, `>7`, `norm`, `none`. One-hot encoded. |
| `glu_serum_cat` | Categorical (OHE) | `max_glu_serum` | Categories: `>300`, `>200`, `norm`, `none`. One-hot encoded. |
| `diag_1_group` | Categorical (OHE) | `diag_1` | Primary diagnosis mapped to 9 clinical categories (Circulatory, Respiratory, etc.). |
| `diag_2_group` | Categorical (OHE) | `diag_2` | Secondary diagnosis mapped to clinical categories. |
| `diag_3_group` | Categorical (OHE) | `diag_3` | Tertiary diagnosis mapped to clinical categories. |
| `race_group` | Audit-Only | `race` | Protected demographic attribute; withheld from model features, audited for parity. |
| `gender` | Audit-Only | `gender` | Protected demographic attribute; withheld from model features, audited for parity. |
| `age_band` | Audit-Only | `age` | Categorized into `<30`, `30-49`, `50-69`, `70+` for demographic parity audits. |
"""
    p.write_text(md, encoding="utf-8")
    logger.info(f"Saved feature dictionary to {output_path}")


def run_features_step(args=None) -> int:
    """CLI handler for make features / python -m readmit.cli features."""
    logger.info("Starting Feature Engineering pipeline step...")
    save_feature_dictionary()

    # Load splits from data/processed/
    proc_dir = pathlib.Path("data/processed")
    train_df = pd.read_parquet(proc_dir / "train.parquet")
    val_df = pd.read_parquet(proc_dir / "val.parquet")

    # Step 1: Fit feature engineer on TRAIN ONLY
    engineer = ClinicalFeatureEngineer(top_n_specialties=10, top_n_payers=5)
    engineer.fit(train_df)

    train_eng = engineer.transform(train_df)
    val_eng = engineer.transform(val_df)

    import yaml

    with open("configs/features.yaml", "r", encoding="utf-8") as f:
        feat_cfg = yaml.safe_load(f)

    num_cols = feat_cfg.get("numeric_cols", [])
    bin_cols = feat_cfg.get("binary_cols", [])
    cat_cols = feat_cfg.get("categorical_cols", [])

    # Step 2: Fit preprocessor on TRAIN ONLY
    preprocessor = build_preprocessor_pipeline(num_cols, bin_cols, cat_cols)
    preprocessor.fit(train_eng)

    X_train = preprocessor.transform(train_eng)
    _ = preprocessor.transform(val_eng)
    feature_names = list(preprocessor.get_feature_names_out())

    logger.info(f"Feature transformation complete: {X_train.shape[1]} features extracted.")
    logger.info(f"Sample feature names: {feature_names[:8]}")

    # Save feature pipeline
    art_dir = pathlib.Path("artifacts")
    art_dir.mkdir(parents=True, exist_ok=True)
    full_feature_pipeline = Pipeline(
        [
            ("engineer", engineer),
            ("preprocessor", preprocessor),
        ]
    )
    joblib.dump(full_feature_pipeline, art_dir / "feature_pipeline.joblib")
    logger.info(f"Saved feature pipeline to {art_dir / 'feature_pipeline.joblib'}")

    # Step 3: Evaluate Baselines on Validation Set
    y_train = train_df["readmit_30d"].values
    y_val = val_df["readmit_30d"].values

    # Prevalence baseline
    prev_baseline = PrevalenceBaseline().fit(train_df, y_train)
    val_prob_prev = prev_baseline.predict_proba(val_df)[:, 1]
    prev_metrics = compute_all_metrics(y_val, val_prob_prev)

    # Rule-based baseline (prior inpatient rank)
    rule_baseline = PriorInpatientRuleBaseline().fit(train_df, y_train)
    val_prob_rule = rule_baseline.predict_proba(val_df)[:, 1]
    rule_metrics = compute_all_metrics(y_val, val_prob_rule)

    logger.info("=== Baseline Evaluation on Validation Set ===")
    logger.info(
        f"Prevalence Baseline PR-AUC: {prev_metrics['pr_auc']:.4f}, ROC-AUC: {prev_metrics['roc_auc']:.4f}"
    )
    logger.info(
        f"Rule Baseline (Prior Inpatient) PR-AUC: {rule_metrics['pr_auc']:.4f}, ROC-AUC: {rule_metrics['roc_auc']:.4f}"
    )
    logger.info(
        f"Rule Baseline Top-20% Recall: {rule_metrics['primary_capacity']['recall']:.2%}, Lift: {rule_metrics['primary_capacity']['lift']:.2f}x"
    )

    return 0
