# Cohort Definition & Flow Diagram

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
| **0. Raw Ingestion** | UCI Diabetes 130-US Hospitals (1999–2008) | 0 | 101,766 | 71,518 |
| **1. Hospice/Mortality** | Excluded discharge disposition IDs 11, 13, 14, 19, 20, 21 | 2,423 | 99,343 | 99,343 |
| **2. Gender Filter** | Excluded `gender == 'Unknown/Invalid'` | 3 | 99,340 | 99,340 |
| **3. Index Encounter** | Primary analysis: first encounter per patient (lowest `encounter_id`) | 29,353 | **69,987** | **69,987** |

---

## Final Cohort Target Summary

- **Total Cohort Size**: **69,987** patients
- **Positive Readmission Count (<30 days)**: **6,285**
- **Cohort Prevalence**: **8.9802%** (~8.98%)
- **Negative Count (>30 days or None)**: **63,702** (91.0198%)

All numbers are measured directly from the verified raw dataset.
