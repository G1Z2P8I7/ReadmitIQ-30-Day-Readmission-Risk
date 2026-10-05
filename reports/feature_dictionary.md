# Feature Dictionary & Specification

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
