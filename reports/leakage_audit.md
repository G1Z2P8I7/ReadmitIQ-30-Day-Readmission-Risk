# Clinical Leakage Audit & Feature Eligibility

**Prediction Point Invariant**: Point of discharge.
**Core Test**: *"Would the care team know this feature at discharge?"*

## Column-by-Column Classification

| Column Name | Category | Role | Justification / Prediction Point Status |
|---|---|---|---|
| `encounter_id` | **C** | Identifier | Unique encounter identifier; dropped from features. |
| `patient_nbr` | **C** | Identifier | Unique patient identifier; strictly used for grouped splitting. |
| `readmitted` | **C** | Target | The outcome being predicted (`<30` vs `>30`/`NO`). Dropped from features. |
| `race` | **C / Audit** | Protected Attribute | Audited for algorithmic fairness across demographic groups. Excluded from features. |
| `gender` | **C / Audit** | Protected Attribute | Audited for parity across male/female subgroups. Excluded from features. |
| `weight` | **C** | Unreliable / Missing | 96.86% missing; unreliably documented. Dropped. |
| `examide` | **C** | Constant | 100% 'No' across all encounters. Dropped (zero variance). |
| `citoglipton` | **C** | Constant | 100% 'No' across all encounters. Dropped (zero variance). |
| `age` | **A** | Feature (Numeric/Band) | Known at admission/discharge. Midpoint used for modeling; 4-band group used for fairness audit. |
| `admission_type_id` | **A** | Categorical | Emergency vs Urgent vs Elective. Known at admission. Grouped. |
| `discharge_disposition_id` | **A** | Categorical | Destination at discharge (Home, SNF, Home Health, etc.). Finalized at discharge. |
| `admission_source_id` | **A** | Categorical | Referral source (Physician, Emergency Room, Transfer). Known at admission. |
| `time_in_hospital` | **A** | Numeric | Length of stay in days. Known at discharge. |
| `payer_code` | **B** | Categorical | Financial coverage / payer. Known at admission; proxy for socioeconomic status. Explicit 'Unknown'. |
| `medical_specialty` | **B** | Categorical | Specialty of admitting physician. Known during stay; grouped + 'Unknown'. |
| `num_lab_procedures` | **A** | Numeric | Total lab tests performed during stay. Cumulative count finalized at discharge. |
| `num_procedures` | **A** | Numeric | Non-lab procedures during stay. Finalized at discharge. |
| `num_medications` | **A** | Numeric | Distinct medications administered during stay. Finalized at discharge. |
| `number_outpatient` | **A** | Numeric | Outpatient visits in preceding 12 months. Prior utilization known from EHR history. |
| `number_emergency` | **A** | Numeric | Emergency room visits in preceding 12 months. Prior utilization known from EHR history. |
| `number_inpatient` | **A** | Numeric | Inpatient admissions in preceding 12 months. Strong historical utilization marker. |
| `diag_1` | **A** | Categorical | Primary diagnosis ICD-9 code. Coded at discharge; grouped into 9 clinical categories. |
| `diag_2` | **A** | Categorical | Secondary diagnosis ICD-9 code. Grouped into clinical taxonomy. |
| `diag_3` | **A** | Categorical | Additional secondary diagnosis ICD-9 code. Grouped into clinical taxonomy. |
| `number_diagnoses` | **A** | Numeric | Total diagnoses entered into encounter billing record. Finalized at discharge. |
| `max_glu_serum` | **A** | Categorical/Binary | Maximum glucose serum test result during stay (`>200`, `>300`, `norm`, `none`). |
| `A1Cresult` | **A** | Categorical/Binary | Hemoglobin A1c test result (`>8`, `>7`, `norm`, `none`). Measured during inpatient episode. |
| 23 Medication Columns | **A** | Categorical/Counts | Specific diabetes medications. Summarized into active meds, changes (Up/Down), steady meds. |
| `change` | **A** | Binary | Change in diabetic medications during encounter. Known at discharge. |
| `diabetesMed` | **A** | Binary | Whether any diabetic medication was prescribed. Known at discharge. |

---

## Invariant Adherence
- **Zero Patient Overlap**: Ensured by patient-level splitting in `data.py`.
- **Target Independence**: Target (`readmit_30d`) is completely isolated and never enters the transformation pipeline.
- **Audit Attributes Isolated**: `race_group` and `gender` are stripped before feature transformation and only attached during fairness evaluation.
