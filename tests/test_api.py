"""Unit tests for inference roundtrip and FastAPI schema validation."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from readmit.inference import explain, predict


@pytest.fixture
def sample_patient_dict():
    return {
        "admission_type_id": 1,
        "admission_source_id": 7,
        "discharge_disposition_id": 1,
        "time_in_hospital": 4,
        "payer_code": "MC",
        "medical_specialty": "InternalMedicine",
        "age": "[60-70)",
        "race": "Caucasian",
        "gender": "Female",
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 1,
        "num_lab_procedures": 45,
        "num_procedures": 1,
        "num_medications": 14,
        "number_diagnoses": 7,
        "diag_1": "414",
        "diag_2": "250.0",
        "diag_3": "401",
        "max_glu_serum": "None",
        "A1Cresult": "None",
        "change": "No",
        "diabetesMed": "Yes",
        "metformin": "No",
        "glipizide": "No",
        "glyburide": "No",
        "pioglitazone": "No",
        "rosiglitazone": "No",
        "insulin": "Steady",
    }


def test_inference_roundtrip(sample_patient_dict):
    """readmit.inference.predict on a sample row returns probability in [0, 1] and an explanation."""
    pred_res = predict(sample_patient_dict)
    assert "risk_probability" in pred_res
    assert 0.0 <= pred_res["risk_probability"] <= 1.0
    assert "risk_vs_average" in pred_res
    assert "risk_band" in pred_res

    exp_res = explain(sample_patient_dict, top_k=5)
    assert "base_value_log_odds" in exp_res
    assert "top_contributions" in exp_res
    assert len(exp_res["top_contributions"]) == 5
    for c in exp_res["top_contributions"]:
        assert "feature" in c
        assert "shap_value" in c
        assert "direction" in c


def test_api_schema(sample_patient_dict):
    """/health, /predict, /explain return documented schemas; invalid input returns 422."""
    client = TestClient(app)

    # 1. /health
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"
    assert "model_version" in r_health.json()

    # 2. /predict
    r_pred = client.post("/predict", json=sample_patient_dict)
    assert r_pred.status_code == 200
    p_data = r_pred.json()
    assert "risk_probability" in p_data
    assert 0.0 <= p_data["risk_probability"] <= 1.0
    assert "risk_vs_average" in p_data
    assert "risk_band" in p_data

    # 3. /explain
    r_exp = client.post("/explain?top_k=3", json=sample_patient_dict)
    assert r_exp.status_code == 200
    e_data = r_exp.json()
    assert "base_value_log_odds" in e_data
    assert len(e_data["top_contributions"]) == 3

    # 4. Invalid input validation (422 Unprocessable Entity)
    invalid_payload = sample_patient_dict.copy()
    invalid_payload["time_in_hospital"] = -5  # Violates ge=1 validation
    r_inv = client.post("/predict", json=invalid_payload)
    assert r_inv.status_code == 422
