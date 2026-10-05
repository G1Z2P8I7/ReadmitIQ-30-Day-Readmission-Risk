"""Streamlit Interactive Clinical Decision Support Dashboard for ReadmitIQ."""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import streamlit as st

from readmit.inference import load_inference_artifacts, predict

st.set_page_config(
    page_title="ReadmitIQ - 30-Day Hospital Readmission Risk System",
    page_icon="🏥",
    layout="wide",
)


@st.cache_resource
def get_cached_inference():
    return load_inference_artifacts()


@st.cache_data
def get_cached_metrics():
    metrics_path = pathlib.Path("reports/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


# Load cached model assets and metrics
pipeline, explainer, feature_names, prevalence = get_cached_inference()
metrics_data = get_cached_metrics()

# Sidebar: Governance, Assumptions & Limitations
st.sidebar.title("🏥 ReadmitIQ")
st.sidebar.caption("Clinical Decision Support & Fairness Audit")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Operational Assumptions")
st.sidebar.markdown(
    "- **Prediction Point:** Hospital Discharge.\n"
    "- **Capacity Tier:** Top 20% of discharged patients targeted for post-discharge intervention.\n"
    "- **Illustrative Cost Ratio:** 5:1 (Missing a readmission assumed 5x more costly than false alarm).\n"
    "- **Primary Cohort:** 69,987 index admissions (mortality/hospice excluded)."
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚠️ Governance & Disclaimers")
st.sidebar.markdown(
    "- **Non-Causal Tool:** Statistical risk prioritization only; does not recommend clinical treatment.\n"
    "- **Protected Attributes:** Race and gender are strictly excluded from predictive features and tracked purely for demographic parity audit.\n"
    "- **Low N Warning:** Demographic subgroups with N < 500 carry wider uncertainty intervals."
)

# Navigation
page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Summary",
        "Patient Risk Scoring",
        "SHAP Interpretability",
        "Fairness by Group",
        "Model Comparison & Capacity",
    ],
)

# =========================================================
# Page 1: Executive Summary
# =========================================================
if page == "Executive Summary":
    st.title("🏥 Executive Summary: ReadmitIQ Decision Support")
    st.markdown(
        "ReadmitIQ is an algorithmic decision-support pipeline built on the UCI 130-US Hospitals dataset, "
        "designed to prioritize post-discharge nurse outreach and care transitions under strict operational capacity constraints."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Cohort Size (Index Encounters)",
            value=f"{metrics_data.get('cohort', {}).get('n_patients', 69987):,}",
            help="Unique index patients after removing mortality/hospice exclusions",
        )
    with col2:
        st.metric(
            label="Baseline Readmission Rate",
            value=f"{metrics_data.get('cohort', {}).get('prevalence', 0.0899) * 100:.2f}%",
            help="Observed 30-day unplanned readmission rate",
        )
    with col3:
        xgb_test = metrics_data.get("models", {}).get("xgboost_calibrated", {})
        pr_val = xgb_test.get("pr_auc", {}).get("value", 0.1385)
        st.metric(
            label="Primary Model PR-AUC (Test)",
            value=f"{pr_val:.4f}",
            delta="+54% over baseline",
            help="Precision-Recall AUC on untouched 10,500 patient test set",
        )
    with col4:
        cap_20 = xgb_test.get("capacity", [{}, {}, {}])[2]
        st.metric(
            label="Recall @ Top 20% Capacity",
            value=f"{cap_20.get('recall', 0.3453) * 100:.1f}%",
            delta=f"{cap_20.get('lift', 1.73):.2f}x Lift",
            help="Share of total readmissions captured within top 20% capacity",
        )

    st.markdown("---")
    st.subheader("Key Architectural Safeguards")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "#### 🔒 Leakage Prevention\n"
            "- Patient-grouped split (0 overlap between train, val, test).\n"
            "- Discharge prediction point: zero future encounter features.\n"
            "- Preprocessors fitted strictly on training data."
        )
    with c2:
        st.markdown(
            "#### 🎯 Probability Calibration\n"
            "- Isotonic post-hoc calibration on validation split.\n"
            "- ECE reduced to **0.0062** on test set.\n"
            "- Reliable absolute risk estimates for clinical trust."
        )
    with c3:
        st.markdown(
            "#### ⚖️ Demographic Fairness\n"
            "- Race and gender preserved exclusively for audit.\n"
            "- Fairlearn Equalized Odds post-processing.\n"
            "- False alarm disparity gap reduced from **5.07% to 0.17%**."
        )

# =========================================================
# Page 2: Patient Risk Scoring
# =========================================================
elif page == "Patient Risk Scoring":
    st.title("👤 Individual Patient Readmission Risk Scoring")
    st.markdown(
        "Evaluate an individual patient encounter at discharge to determine predicted risk, "
        "relative risk compared to hospital average, and clinical tier."
    )

    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Admission & Stay**")
            adm_type = st.selectbox("Admission Type", [1, 2, 3, 7], format_func=lambda x: {1: "Emergency", 2: "Urgent", 3: "Elective", 7: "Trauma/Other"}.get(x, str(x)))
            adm_src = st.selectbox("Admission Source", [7, 1, 2, 4], format_func=lambda x: {7: "Emergency Room", 1: "Physician Referral", 2: "Clinic", 4: "Transfer"}.get(x, str(x)))
            disch = st.selectbox("Discharge Disposition", [1, 3, 6, 2], format_func=lambda x: {1: "Discharged to Home", 3: "SNF / Nursing Facility", 6: "Home Health Service", 2: "Other Facility"}.get(x, str(x)))
            los = st.slider("Time in Hospital (Days)", 1, 14, 4)
            specialty = st.selectbox("Medical Specialty", ["InternalMedicine", "Cardiology", "Family/GeneralPractice", "Surgery-General", "Other", "Unknown"])

        with col2:
            st.markdown("**Prior Utilization & Demographics**")
            age = st.selectbox("Age Decade", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"], index=6)
            race = st.selectbox("Race (Audit Only)", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Unknown"])
            gender = st.selectbox("Gender (Audit Only)", ["Female", "Male"])
            n_inpatient = st.number_input("Prior Inpatient Admissions (Past Year)", 0, 15, 1)
            n_outpatient = st.number_input("Prior Outpatient Visits (Past Year)", 0, 20, 0)
            n_emergency = st.number_input("Prior Emergency Visits (Past Year)", 0, 15, 0)

        with col3:
            st.markdown("**Clinical & Diabetes Management**")
            num_meds = st.slider("Number of Medications Administered", 1, 50, 14)
            num_labs = st.slider("Number of Lab Procedures", 1, 100, 42)
            diag_1 = st.text_input("Primary Diagnosis (ICD-9)", "414")
            a1c = st.selectbox("HbA1c Test Result", ["None", "Norm", ">7", ">8"])
            glu = st.selectbox("Max Glucose Serum Test", ["None", "Norm", ">200", ">300"])
            insulin = st.selectbox("Insulin Management", ["No", "Steady", "Up", "Down"], index=1)
            chg = st.selectbox("Diabetes Med Changed", ["No", "Ch"])

        submit = st.form_submit_button("Calculate Readmission Risk")

    if submit:
        patient_payload = {
            "admission_type_id": adm_type,
            "admission_source_id": adm_src,
            "discharge_disposition_id": disch,
            "time_in_hospital": los,
            "payer_code": "MC",
            "medical_specialty": specialty,
            "age": age,
            "race": race,
            "gender": gender,
            "number_outpatient": n_outpatient,
            "number_emergency": n_emergency,
            "number_inpatient": n_inpatient,
            "num_lab_procedures": num_labs,
            "num_procedures": 1,
            "num_medications": num_meds,
            "number_diagnoses": 6,
            "diag_1": diag_1,
            "diag_2": "250.0",
            "diag_3": "401",
            "max_glu_serum": glu,
            "A1Cresult": a1c,
            "change": chg,
            "diabetesMed": "Yes",
            "metformin": "No",
            "glipizide": "No",
            "glyburide": "No",
            "pioglitazone": "No",
            "rosiglitazone": "No",
            "insulin": insulin,
        }

        result = predict(patient_payload)
        prob = result["risk_probability"]
        mult = result["risk_vs_average"]
        band = result["risk_band"]

        st.markdown("### Risk Evaluation Results")
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.metric("Predicted 30-Day Risk", f"{prob * 100:.1f}%")
        with rc2:
            st.metric("Relative Risk vs Baseline", f"{mult:.2f}x Average", delta=f"{mult - 1.0:+.2f}x")
        with rc3:
            st.metric("Clinical Risk Band", band)

        if prob >= 0.1077:
            st.error("🚨 **Recommended Action:** Patient falls in top 20% capacity tier. Flag for transitional care coordinator phone follow-up within 48h.")
        else:
            st.success("✅ **Standard Care:** Patient risk is within routine baseline limits. Standard discharge instructions.")

# =========================================================
# Page 3: SHAP Interpretability
# =========================================================
elif page == "SHAP Interpretability":
    st.title("🔍 SHAP Interpretability & Feature Attributions")
    st.markdown(
        "Explanations quantify statistical contributions to model log-odds output. "
        "Attributions are observational and non-causal."
    )

    t1, t2 = st.tabs(["Global Feature Importance (Beeswarm)", "Case Study Waterfalls"])
    with t1:
        st.subheader("Global Population Attributions")
        beeswarm_img = pathlib.Path("reports/figures/shap_summary_xgb.png")
        if beeswarm_img.exists():
            st.image(str(beeswarm_img), caption="SHAP Summary Beeswarm: Impact on 30-Day Readmission Risk Log-Odds", use_container_width=True)
        else:
            st.info("Run `python -m readmit.cli explain` to generate global SHAP plots.")

    with t2:
        st.subheader("Clinical Case Studies")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### Case A: True Positive (High Risk)")
            img_a = pathlib.Path("reports/figures/shap_waterfall_Case_A_TruePositive.png")
            if img_a.exists():
                st.image(str(img_a), use_container_width=True)
        with c2:
            st.markdown("#### Case B: False Negative (Clinical Blindspot)")
            img_b = pathlib.Path("reports/figures/shap_waterfall_Case_B_FalseNegative.png")
            if img_b.exists():
                st.image(str(img_b), use_container_width=True)
        with c3:
            st.markdown("#### Case C: True Negative (Low Risk)")
            img_c = pathlib.Path("reports/figures/shap_waterfall_Case_C_LowRiskBaseline.png")
            if img_c.exists():
                st.image(str(img_c), use_container_width=True)

# =========================================================
# Page 4: Fairness by Group
# =========================================================
elif page == "Fairness by Group":
    st.title("⚖️ Demographic Fairness Audit & Mitigation")
    st.markdown(
        "Evaluating model parity across demographic subgroups before and after post-processing mitigation "
        "using Fairlearn `ThresholdOptimizer(equalized_odds)`."
    )

    gaps_before = metrics_data.get("fairness_gaps", {}).get("before", {}).get("race_group", {})
    gaps_after = metrics_data.get("fairness_gaps", {}).get("after", {}).get("race_group", {})

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Unmitigated Race Disparity (K=20% Global Threshold)")
        st.metric("False Positive Rate Gap", f"{gaps_before.get('fpr_gap', 0.0507) * 100:.2f}%")
        st.metric("True Positive Rate Gap", f"{gaps_before.get('tpr_gap', 0.0610) * 100:.2f}%")
    with c2:
        st.subheader("Mitigated Race Disparity (Equalized Odds)")
        st.metric(
            "False Positive Rate Gap",
            f"{gaps_after.get('fpr_gap', 0.0017) * 100:.2f}%",
            delta=f"{(gaps_after.get('fpr_gap', 0.0017) - gaps_before.get('fpr_gap', 0.0507)) * 100:.2f}%",
        )
        st.metric(
            "True Positive Rate Gap",
            f"{gaps_after.get('tpr_gap', 0.0385) * 100:.2f}%",
            delta=f"{(gaps_after.get('tpr_gap', 0.0385) - gaps_before.get('tpr_gap', 0.0610)) * 100:.2f}%",
        )

    st.markdown("---")
    st.subheader("Per-Group Performance Breakdown (Test Set)")
    race_audit = metrics_data.get("fairness_audit", {}).get("after", {}).get("race_group", {})
    if race_audit:
        df_race = pd.DataFrame(race_audit).T
        df_race["TPR (Recall)"] = (df_race["tpr"] * 100).round(2).astype(str) + "%"
        df_race["FPR (False Alarm)"] = (df_race["fpr"] * 100).round(2).astype(str) + "%"
        df_race["Precision"] = (df_race["precision"] * 100).round(2).astype(str) + "%"
        df_race["Selection Rate"] = (df_race["selection_rate"] * 100).round(2).astype(str) + "%"
        df_race["Sample Size (N)"] = df_race["n"].astype(int)
        df_race["Sample Alert"] = df_race["low_confidence"].map(lambda x: "⚠️ Low N (<500)" if x else "Adequate N")
        st.dataframe(df_race[["Sample Size (N)", "TPR (Recall)", "FPR (False Alarm)", "Precision", "Selection Rate", "Sample Alert"]])

# =========================================================
# Page 5: Model Comparison & Capacity
# =========================================================
elif page == "Model Comparison & Capacity":
    st.title("📊 Model Comparison & Operational Capacity")
    st.markdown(
        "Evaluation of discrimination (PR-AUC, ROC-AUC), calibration (Brier, ECE), "
        "and capacity-constrained decision support."
    )

    t1, t2 = st.tabs(["Capacity Allocation Table", "Reliability Calibration Curve"])
    with t1:
        st.subheader("Capacity-Constrained Decision Tiers (Test Set)")
        cap_list = metrics_data.get("models", {}).get("xgboost_calibrated", {}).get("capacity", [])
        if cap_list:
            df_cap = pd.DataFrame(cap_list)
            df_cap["Capacity Tier"] = df_cap["k_percent"].map(lambda x: f"Top {x}%")
            df_cap["Patients Flagged"] = df_cap["n_flagged"].map(lambda x: f"{x:,}")
            df_cap["Probability Cutoff"] = (df_cap["threshold"] * 100).round(2).astype(str) + "%"
            df_cap["Captured Readmissions"] = df_cap["positives_captured"].map(lambda x: f"{x:,}")
            df_cap["Recall (Sensitivity)"] = (df_cap["recall"] * 100).round(2).astype(str) + "%"
            df_cap["Precision (PPV)"] = (df_cap["precision"] * 100).round(2).astype(str) + "%"
            df_cap["Lift over Baseline"] = df_cap["lift"].round(2).astype(str) + "x"

            st.dataframe(
                df_cap[
                    [
                        "Capacity Tier",
                        "Patients Flagged",
                        "Probability Cutoff",
                        "Captured Readmissions",
                        "Recall (Sensitivity)",
                        "Precision (PPV)",
                        "Lift over Baseline",
                    ]
                ],
                use_container_width=True,
            )

    with t2:
        st.subheader("Validation Probability Reliability Curve")
        cal_curve_img = pathlib.Path("reports/figures/calibration_curve_xgb.png")
        if cal_curve_img.exists():
            st.image(str(cal_curve_img), caption="Reliability Curves: Raw vs Platt vs Isotonic Calibration", use_container_width=True)
        else:
            st.info("Run `python -m readmit.cli evaluate` to generate calibration plots.")
