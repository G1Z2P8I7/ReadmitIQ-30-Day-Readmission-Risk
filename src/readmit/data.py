"""Data ingestion, cohort construction, and data splitting."""

import io
import json
import logging
import pathlib
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

UCI_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"


def download_raw(raw_dir: str = "data/raw") -> None:
    target_dir = pathlib.Path(raw_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    data_file = target_dir / "diabetic_data.csv"
    mapping_file = target_dir / "IDS_mapping.csv"

    if data_file.exists() and mapping_file.exists():
        logger.info("Raw files diabetic_data.csv and IDS_mapping.csv exist.")
        return

    logger.info("Downloading UCI dataset...")
    req = urllib.request.Request(UCI_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        content = resp.read()

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        zf.extractall(target_dir)
    logger.info(f"Successfully extracted dataset to {target_dir}")


def load_raw_data(raw_dir: str = "data/raw") -> tuple[pd.DataFrame, pathlib.Path]:
    target_dir = pathlib.Path(raw_dir)
    data_file = target_dir / "diabetic_data.csv"
    mapping_file = target_dir / "IDS_mapping.csv"

    if not data_file.exists() or not mapping_file.exists():
        download_raw(raw_dir)

    df = pd.read_csv(data_file, low_memory=False)
    return df, mapping_file


def build_cohort(
    df: pd.DataFrame,
    exclude_disposition_ids: list[int] | None = None,
    one_encounter_per_patient: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Applies cohort filtering rules per CONTEXT.md Section 6."""
    if exclude_disposition_ids is None:
        exclude_disposition_ids = [11, 13, 14, 19, 20, 21]

    flow = {
        "raw_encounters": len(df),
        "raw_unique_patients": int(df["patient_nbr"].nunique()),
    }

    mask_hospice = df["discharge_disposition_id"].isin(exclude_disposition_ids)
    flow["excluded_expired_hospice"] = int(mask_hospice.sum())
    df_step1 = df[~mask_hospice].copy()
    flow["remaining_step1"] = len(df_step1)

    mask_invalid_gender = (
        df_step1["sender"] == "Unknown/Invalid"
        if "sender" in df_step1.columns
        else df_step1["gender"] == "Unknown/Invalid"
    )
    flow["excluded_invalid_gender"] = int(mask_invalid_gender.sum())
    df_step2 = df_step1[~mask_invalid_gender].copy()
    flow["remaining_step2"] = len(df_step2)

    if one_encounter_per_patient:
        df_cohort = (
            df_step2.sort_values("encounter_id", ascending=True)
            .groupby("patient_nbr", as_index=False)
            .first()
        )
        flow["excluded_repeat_encounters"] = int(len(df_step2) - len(df_cohort))
    else:
        df_cohort = df_step2
        flow["excluded_repeat_encounters"] = 0

    flow["final_cohort_encounters"] = len(df_cohort)
    flow["final_cohort_patients"] = int(df_cohort["patient_nbr"].nunique())

    target_mapping = {"<30": 1, ">30": 0, "NO": 0}
    df_cohort["readmit_30d"] = df_cohort["readmitted"].map(target_mapping).astype(int)

    flow["readmit_30d_positives"] = int(df_cohort["readmit_30d"].sum())
    flow["readmit_30d_prevalence"] = float(df_cohort["readmit_30d"].mean())

    return df_cohort, flow


def save_cohort_flow_report(flow: dict, output_path: str = "reports/cohort_flow.md") -> None:
    p = pathlib.Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    prev_pct = flow["readmit_30d_prevalence"] * 100

    md = f"""# Cohort Definition & Flow Diagram

Single source of truth for cohort filtering rules and attrition counts.

## Cohort Invariants & Decision Logic
1. **Target Population**: Discharged diabetic inpatients eligible for 30-day intervention.
2. **Exclusion of Expired & Hospice Encounters**: Encounters with `discharge_disposition_id` in `[11, 13, 14, 19, 20, 21]` (expired, hospice home, hospice medical facility) are excluded because readmission is clinically impossible.
3. **Gender Quality Filter**: Exclude non-assigned / invalid gender records (`Unknown/Invalid`).
4. **Primary Cohort (Unit of Analysis)**: One index encounter per patient (selected by lowest `encounter_id`, serving as an approximate chronological proxy). This prevents patient leakage across temporal repeats.
5. **Prediction Horizon (Label)**: 30-day unplanned readmission (`readmitted == '<30'` mapped to `1`; `'>30'` and `'NO'` mapped to `0`).

---

## Attrition Flow Table

| Step | Description | Rows Excluded | Encounters Remaining | Unique Patients |
|---|---|---|---|---|
| **0. Raw Ingestion** | UCI Diabetes 130-US Hospitals (1999–2008) | 0 | {flow["raw_encounters"]:,} | {flow["raw_unique_patients"]:,} |
| **1. Hospice/Mortality** | Excluded discharge disposition IDs 11, 13, 14, 19, 20, 21 | {flow["excluded_expired_hospice"]:,} | {flow["remaining_step1"]:,} | {flow["remaining_step1"]:,} |
| **2. Gender Filter** | Excluded `gender == 'Unknown/Invalid'` | {flow["excluded_invalid_gender"]:,} | {flow["remaining_step2"]:,} | {flow["remaining_step2"]:,} |
| **3. Index Encounter** | Primary analysis: first encounter per patient (lowest `encounter_id`) | {flow["excluded_repeat_encounters"]:,} | **{flow["final_cohort_encounters"]:,}** | **{flow["final_cohort_patients"]:,}** |

---

## Final Cohort Target Summary

- **Total Cohort Size**: **{flow["final_cohort_encounters"]:,}** patients
- **Positive Readmission Count (<30 days)**: **{flow["readmit_30d_positives"]:,}**
- **Cohort Prevalence**: **{prev_pct:.4f}%** (~{prev_pct:.2f}%)
- **Negative Count (>30 days or None)**: **{flow["final_cohort_encounters"] - flow["readmit_30d_positives"]:,}** ({100 - prev_pct:.4f}%)

All numbers are measured directly from the verified raw dataset.
"""
    p.write_text(md, encoding="utf-8")
    logger.info(f"Saved cohort flow report to {output_path}")


def create_patient_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    splits_dir: str = "data/splits",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Creates patient-grouped splits (70/15/15) with zero patient overlap."""
    splits_path = pathlib.Path(splits_dir)
    splits_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    patients = df[["patient_nbr", "readmit_30d"]].drop_duplicates().copy()
    pos_patients = rng.permutation(patients[patients["readmit_30d"] == 1]["patient_nbr"].values)
    neg_patients = rng.permutation(patients[patients["readmit_30d"] == 0]["patient_nbr"].values)

    def split_array(arr):
        n = len(arr)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return arr[:n_train], arr[n_train : n_train + n_val], arr[n_train + n_val :]

    tr_pos, val_pos, te_pos = split_array(pos_patients)
    tr_neg, val_neg, te_neg = split_array(neg_patients)

    train_patients = set(tr_pos).union(set(tr_neg))
    val_patients = set(val_pos).union(set(val_neg))
    test_patients = set(te_pos).union(set(te_neg))

    assert len(train_patients.intersection(val_patients)) == 0, "Patient overlap train-val!"
    assert len(train_patients.intersection(test_patients)) == 0, "Patient overlap train-test!"
    assert len(val_patients.intersection(test_patients)) == 0, "Patient overlap val-test!"

    split_ids = {
        "train": sorted(map(int, train_patients)),
        "val": sorted(map(int, val_patients)),
        "test": sorted(map(int, test_patients)),
    }
    with open(splits_path / "train_patients.json", "w", encoding="utf-8") as f:
        json.dump(split_ids["train"], f)
    with open(splits_path / "val_patients.json", "w", encoding="utf-8") as f:
        json.dump(split_ids["val"], f)
    with open(splits_path / "test_patients.json", "w", encoding="utf-8") as f:
        json.dump(split_ids["test"], f)

    train_df = df[df["patient_nbr"].isin(train_patients)].copy()
    val_df = df[df["patient_nbr"].isin(val_patients)].copy()
    test_df = df[df["patient_nbr"].isin(test_patients)].copy()

    split_summary = {
        "train_size": len(train_df),
        "train_prevalence": float(train_df["readmit_30d"].mean()),
        "val_size": len(val_df),
        "val_prevalence": float(val_df["readmit_30d"].mean()),
        "test_size": len(test_df),
        "test_prevalence": float(test_df["readmit_30d"].mean()),
    }

    logger.info(
        f"Splits: Train={len(train_df):,} ({split_summary['train_prevalence']:.2%}), "
        f"Val={len(val_df):,} ({split_summary['val_prevalence']:.2%}), "
        f"Test={len(test_df):,} ({split_summary['test_prevalence']:.2%})"
    )

    return train_df, val_df, test_df, split_summary


def run_data_step(args=None) -> int:
    logger.info("Starting Data pipeline step...")
    download_raw()
    df, _mapping_file = load_raw_data()
    logger.info(f"Loaded raw dataset with {len(df):,} rows and {len(df.columns)} columns.")

    with open("configs/base.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cohort_cfg = cfg.get("cohort", {})
    exclude_ids = cohort_cfg.get("exclude_disposition_ids", [11, 13, 14, 19, 20, 21])
    one_enc = cohort_cfg.get("one_encounter_per_patient", True)

    cohort_df, flow = build_cohort(
        df, exclude_disposition_ids=exclude_ids, one_encounter_per_patient=one_enc
    )
    logger.info(
        f"Built primary cohort: {len(cohort_df):,} encounters, prevalence: {flow['readmit_30d_prevalence']:.4%}"
    )

    save_cohort_flow_report(flow)

    from readmit.audit import generate_data_quality_report, generate_leakage_audit_report

    generate_data_quality_report(df, cohort_df, flow)
    generate_leakage_audit_report()

    split_cfg = cfg.get("split", {})
    seed = cfg.get("seed", 42)
    train_df, val_df, test_df, _split_summary = create_patient_splits(
        cohort_df,
        train_ratio=split_cfg.get("train", 0.70),
        val_ratio=split_cfg.get("val", 0.15),
        test_ratio=split_cfg.get("test", 0.15),
        seed=seed,
    )

    proc_dir = pathlib.Path("data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)
    cohort_df.to_parquet(proc_dir / "cohort.parquet", index=False)
    train_df.to_parquet(proc_dir / "train.parquet", index=False)
    val_df.to_parquet(proc_dir / "val.parquet", index=False)
    test_df.to_parquet(proc_dir / "test.parquet", index=False)

    logger.info("Data step completed successfully.")
    return 0
