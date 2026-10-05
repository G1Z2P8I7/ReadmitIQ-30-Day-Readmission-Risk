"""Tests for cohort definition, data ingestion, and patient splitting."""

import json
import pathlib

import pandas as pd
import pytest

from readmit.data import build_cohort


@pytest.fixture
def raw_sample_data():
    """Generates a representative sample of encounters."""
    data = {
        "encounter_id": [101, 102, 103, 104, 105, 106, 107, 108],
        "patient_nbr": [1, 1, 2, 3, 4, 5, 6, 7],
        "race": [
            "Caucasian",
            "Caucasian",
            "AfricanAmerican",
            "Hispanic",
            "Asian",
            "Other",
            "?",
            "Caucasian",
        ],
        "gender": [
            "Female",
            "Female",
            "Male",
            "Female",
            "Unknown/Invalid",
            "Male",
            "Female",
            "Male",
        ],
        "discharge_disposition_id": [1, 1, 11, 6, 1, 13, 1, 3],  # 11 and 13 are expired/hospice
        "readmitted": ["<30", ">30", "<30", "NO", "<30", "<30", ">30", "<30"],
        "time_in_hospital": [3, 4, 2, 5, 1, 7, 2, 3],
    }
    return pd.DataFrame(data)


def test_label_encoding(raw_sample_data):
    cohort_df, _ = build_cohort(
        raw_sample_data, exclude_disposition_ids=[11, 13], one_encounter_per_patient=False
    )
    # Check label mapping: <30 -> 1, >30 -> 0, NO -> 0
    assert set(cohort_df["readmit_30d"].unique()).issubset({0, 1})
    for _, row in cohort_df.iterrows():
        if row["readmitted"] == "<30":
            assert row["readmit_30d"] == 1
        else:
            assert row["readmit_30d"] == 0


def test_cohort_excludes_expired_hospice(raw_sample_data):
    cohort_df, _flow = build_cohort(
        raw_sample_data, exclude_disposition_ids=[11, 13], one_encounter_per_patient=True
    )
    # Disposition ids 11 and 13 must not be present
    assert not cohort_df["discharge_disposition_id"].isin([11, 13]).any()
    # Invalid gender must not be present
    assert not (cohort_df["gender"] == "Unknown/Invalid").any()
    # Exactly one encounter per patient
    assert cohort_df["patient_nbr"].nunique() == len(cohort_df)
    # Encounter 101 should be chosen over 102 for patient 1 (lowest encounter_id)
    p1 = cohort_df[cohort_df["patient_nbr"] == 1].iloc[0]
    assert p1["encounter_id"] == 101


def test_splits_no_patient_overlap():
    """Verify generated production splits have zero patient overlap."""
    splits_dir = pathlib.Path("data/splits")
    if not (splits_dir / "train_patients.json").exists():
        pytest.skip("Splits not generated yet.")

    with open(splits_dir / "train_patients.json", "r", encoding="utf-8") as f:
        train_p = set(json.load(f))
    with open(splits_dir / "val_patients.json", "r", encoding="utf-8") as f:
        val_p = set(json.load(f))
    with open(splits_dir / "test_patients.json", "r", encoding="utf-8") as f:
        test_p = set(json.load(f))

    # Invariants
    assert len(train_p.intersection(val_p)) == 0, "Train and Val share patients!"
    assert len(train_p.intersection(test_p)) == 0, "Train and Test share patients!"
    assert len(val_p.intersection(test_p)) == 0, "Val and Test share patients!"

    # Check split proportions roughly 70/15/15
    tot = len(train_p) + len(val_p) + len(test_p)
    assert pytest.approx(len(train_p) / tot, abs=0.02) == 0.70
    assert pytest.approx(len(val_p) / tot, abs=0.02) == 0.15
    assert pytest.approx(len(test_p) / tot, abs=0.02) == 0.15
