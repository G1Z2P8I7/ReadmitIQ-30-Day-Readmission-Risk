"""FastAPI REST Service for ReadmitIQ 30-Day Hospital Readmission Risk Prediction."""

import logging

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from readmit import __version__
from readmit.inference import explain, predict

logger = logging.getLogger("api")

app = FastAPI(
    title="ReadmitIQ Decision Support API",
    description="Capacity-aware 30-day hospital readmission risk prediction and SHAP explainability service.",
    version=__version__,
)


class PatientFeatures(BaseModel):
    # Admission characteristics
    admission_type_id: int = Field(default=1, description="Admission type ID (1=Emergency, 2=Urgent, 3=Elective, etc.)")
    admission_source_id: int = Field(default=7, description="Admission source ID (7=Emergency Room, 1=Physician Referral, etc.)")
    discharge_disposition_id: int = Field(default=1, description="Discharge disposition ID (1=Home, 3=SNF, 6=Home Health, etc.)")
    time_in_hospital: int = Field(ge=1, le=14, default=3, description="Days spent in hospital (1-14)")
    payer_code: str | None = Field(default="MC", description="Payer code or health plan insurer")
    medical_specialty: str | None = Field(default="InternalMedicine", description="Admitting or attending physician specialty")

    # Demographics
    age: str = Field(default="[60-70)", description="Age decade string, e.g. [60-70)")
    race: str | None = Field(default="Caucasian", description="Audit-only race attribute (excluded from predictive features)")
    gender: str | None = Field(default="Female", description="Audit-only gender attribute (excluded from predictive features)")

    # Utilization
    number_outpatient: int = Field(ge=0, default=0, description="Number of outpatient visits in prior year")
    number_emergency: int = Field(ge=0, default=0, description="Number of emergency room visits in prior year")
    number_inpatient: int = Field(ge=0, default=0, description="Number of inpatient hospitalizations in prior year")

    # Clinical & Inpatient Labs
    num_lab_procedures: int = Field(ge=0, default=40, description="Number of laboratory procedures during encounter")
    num_procedures: int = Field(ge=0, default=1, description="Number of surgical or bedside procedures")
    num_medications: int = Field(ge=0, default=12, description="Number of medications administered")
    number_diagnoses: int = Field(ge=1, default=6, description="Number of diagnoses recorded")

    # Diagnoses (ICD-9 codes)
    diag_1: str | None = Field(default="414", description="Primary ICD-9 diagnosis code")
    diag_2: str | None = Field(default="250.0", description="Secondary ICD-9 diagnosis code")
    diag_3: str | None = Field(default="401", description="Tertiary ICD-9 diagnosis code")

    # Glycemic Testing & Medications
    max_glu_serum: str | None = Field(default="None", description="Blood glucose test result (None, Norm, >200, >300)")
    A1Cresult: str | None = Field(default="None", description="HbA1c test result (None, Norm, >7, >8)")
    change: str | None = Field(default="No", description="Medication changed during encounter (Ch or No)")
    diabetesMed: str | None = Field(default="Yes", description="Any diabetes medication prescribed (Yes or No)")

    # Key Diabetic Medications
    metformin: str | None = Field(default="No", description="Metformin dosage change")
    glipizide: str | None = Field(default="No", description="Glipizide dosage change")
    glyburide: str | None = Field(default="No", description="Glyburide dosage change")
    pioglitazone: str | None = Field(default="No", description="Pioglitazone dosage change")
    rosiglitazone: str | None = Field(default="No", description="Rosiglitazone dosage change")
    insulin: str | None = Field(default="Steady", description="Insulin dosage change (No, Steady, Up, Down)")


class PredictionResponse(BaseModel):
    risk_probability: float = Field(..., description="Calibrated predicted probability of 30-day readmission")
    risk_vs_average: float = Field(..., description="Risk relative to hospital baseline prevalence")
    risk_band: str = Field(..., description="Clinical risk tier (Low, Average, Moderate, High)")
    cohort_prevalence: float = Field(..., description="Baseline cohort readmission prevalence")


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float
    direction: str


class ExplanationResponse(BaseModel):
    base_value_log_odds: float
    top_contributions: list[FeatureContribution]


@app.get("/health", status_code=status.HTTP_200_OK)
def health():
    """Health check endpoint returning system status and package version."""
    return {
        "status": "healthy",
        "model_version": __version__,
        "service": "ReadmitIQ Decision Support API",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(patient: PatientFeatures):
    """Predicts calibrated 30-day hospital readmission risk probability and relative risk."""
    try:
        data_dict = patient.model_dump()
        result = predict(data_dict)
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {e!s}",
        )


@app.post("/explain", response_model=ExplanationResponse)
def explain_endpoint(patient: PatientFeatures, top_k: int = 5):
    """Returns top SHAP contributions explaining the individual risk factors for a patient."""
    try:
        data_dict = patient.model_dump()
        result = explain(data_dict, top_k=top_k)
        return result
    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation failed: {e!s}",
        )
