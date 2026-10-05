"""Streamlit Clinical Decision Support Dashboard for ReadmitIQ.

Styled with the Function Health editorial aesthetic: warm alabaster canvas,
terracotta accents, Newsreader serif typography, and 01/02/03 step cards.
"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import pandas as pd
import streamlit as st

# Ensure local app directory is on path for style import
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from style import (
    case_card_header,
    danger_banner,
    divider,
    editorial_header,
    form_section_header,
    governance_card,
    hero_section,
    info_banner,
    inject_css,
    metric_card,
    rationale_card,
    risk_gauge_card,
    sidebar_brand,
    sidebar_section_label,
    step_card,
    success_banner,
    warning_banner,
)

from readmit.inference import explain, load_inference_artifacts, predict

st.set_page_config(
    page_title="ReadmitIQ — Hospital Readmission Risk",
    page_icon="🏥",
    layout="wide",
)

# ── Inject Function Health Theme CSS ─────────────────────
st.markdown(inject_css(), unsafe_allow_html=True)


# ── Cached Data & Artifact Loaders ───────────────────────
@st.cache_resource
def get_cached_inference():
    return load_inference_artifacts()


@st.cache_data
def get_cached_metrics():
    p = pathlib.Path("reports/metrics.json")
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def get_cached_fairness_details():
    p = pathlib.Path("reports/fairness_details.json")
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


pipeline, explainer, feature_names, prevalence = get_cached_inference()
metrics_data = get_cached_metrics()
fairness_details = get_cached_fairness_details()

# ── Sidebar Branding & Assumptions ───────────────────────
st.sidebar.markdown(sidebar_brand(), unsafe_allow_html=True)
st.sidebar.markdown(divider(), unsafe_allow_html=True)

st.sidebar.markdown(sidebar_section_label("Operational Framework"), unsafe_allow_html=True)
st.sidebar.markdown(
    "- **Decision Point:** Hospital Discharge\n"
    "- **Capacity Tier:** Top 20% Flagged\n"
    "- **Cost Ratio:** 5:1 False-Alarm Assumption\n"
    "- **Cohort Size:** 69,987 Index Admissions"
)

st.sidebar.markdown(divider(), unsafe_allow_html=True)

st.sidebar.markdown(sidebar_section_label("Governance & Safeguards"), unsafe_allow_html=True)
st.sidebar.markdown(
    "- Non-causal risk prioritization\n"
    "- Race & gender excluded from features\n"
    "- Subgroups N < 500 flagged with wide CIs"
)

st.sidebar.markdown(divider(), unsafe_allow_html=True)

PAGE_OPTIONS = [
    "Executive Summary",
    "Patient Risk Scoring",
    "SHAP Interpretability",
    "Fairness by Group",
    "Model Comparison & Capacity",
]
PAGE_LABELS = {
    "Executive Summary": "Executive Summary",
    "Patient Risk Scoring": "Patient Risk Scoring",
    "SHAP Interpretability": "SHAP Interpretability",
    "Fairness by Group": "Fairness by Group",
    "Model Comparison & Capacity": "Model Comparison & Capacity",
}

page = st.sidebar.radio(
    "Navigation",
    PAGE_OPTIONS,
    format_func=lambda x: PAGE_LABELS.get(x, x),
)


# ═════════════════════════════════════════════════════════
# PAGE 1 — Executive Summary
# ═════════════════════════════════════════════════════════
if page == "Executive Summary":
    st.markdown(
        hero_section(
            "Prioritizing patient care",
            "before discharge.",
            "An algorithmic decision-support pipeline built on the UCI 130-US Hospitals dataset, "
            "designed to focus post-discharge nurse outreach under strict hospital staffing constraints.",
        ),
        unsafe_allow_html=True,
    )

    cohort = metrics_data.get("cohort", {})
    xgb = metrics_data.get("models", {}).get("xgboost_calibrated", {})
    cap_20 = xgb.get("capacity", [{}, {}, {}])[2] if len(xgb.get("capacity", [])) >= 3 else {}

    # 4 Signature Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            metric_card(
                "👥",
                f"{cohort.get('n_patients', 69987):,}",
                "Index Patients",
                subtext="Mortality/hospice excluded",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            metric_card(
                "📉",
                f"{cohort.get('prevalence', 0.0899) * 100:.2f}%",
                "Baseline Readmission",
                delta="Hospital prevalence floor",
                delta_type="amber",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        pr_val = xgb.get("pr_auc", {}).get("value", 0.1385)
        st.markdown(
            metric_card(
                "🎯",
                f"{pr_val:.4f}",
                "PR-AUC (Test Set)",
                delta="+54.1% over prevalence",
                delta_type="pos",
                subtext="95% CI: [0.1268, 0.1529]",
            ),
            unsafe_allow_html=True,
        )
    with c4:
        recall_val = cap_20.get("recall", 0.3453)
        lift_val = cap_20.get("lift", 1.73)
        st.markdown(
            metric_card(
                "📋",
                f"{recall_val * 100:.1f}%",
                "Recall @ Top 20%",
                delta=f"{lift_val:.2f}× Lift over baseline",
                delta_type="pos",
                subtext="326 of 944 readmissions captured",
            ),
            unsafe_allow_html=True,
        )

    st.markdown(divider(), unsafe_allow_html=True)

    # 3 Function Health Step Cards (01, 02, 03)
    st.markdown(
        editorial_header(
            "Architectural safeguards,",
            "engineered for trust.",
            "Core principles protecting patient privacy, clinical validity, and demographic fairness.",
        ),
        unsafe_allow_html=True,
    )

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            step_card(
                "01",
                "Leakage Prevention",
                "at discharge",
                "Strict patient-level data isolation and boundary controls:",
                [
                    "Patient-grouped 70/15/15 split (0 overlap)",
                    "Discharge prediction horizon strictly enforced",
                    "All preprocessors fitted on training split only",
                ],
            ),
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            step_card(
                "02",
                "Probability Calibration",
                "for clinical trust",
                "Raw ML scores transformed into empirical risk values:",
                [
                    "Post-hoc isotonic probability calibration",
                    f"Expected Calibration Error: {xgb.get('ece', 0.0062):.4f}",
                    "Predicted probabilities mirror observed outcomes",
                ],
            ),
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            step_card(
                "03",
                "Demographic Parity",
                "without compromise",
                "Algorithmic fairness mitigation across protected groups:",
                [
                    "Race and gender excluded from model features",
                    "Fairlearn Equalized Odds post-processing",
                    "Racial false alarm gap reduced from 5.07% to 0.17%",
                ],
            ),
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════
# PAGE 2 — Patient Risk Scoring
# ═════════════════════════════════════════════════════════
elif page == "Patient Risk Scoring":
    st.markdown(
        hero_section(
            "Individual Patient",
            "Risk Assessment.",
            "Evaluate clinical indicators at hospital discharge to determine predicted risk, "
            "relative risk compared to hospital average, and operational outreach priority.",
        ),
        unsafe_allow_html=True,
    )

    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(form_section_header("Stay & Disposition"), unsafe_allow_html=True)
            adm_type = st.selectbox(
                "Admission Type",
                [1, 2, 3, 7],
                format_func=lambda x: {
                    1: "Emergency",
                    2: "Urgent",
                    3: "Elective",
                    7: "Trauma / Other",
                }.get(x, str(x)),
            )
            adm_src = st.selectbox(
                "Admission Source",
                [7, 1, 2, 4],
                format_func=lambda x: {
                    7: "Emergency Room",
                    1: "Physician Referral",
                    2: "Clinic",
                    4: "Transfer",
                }.get(x, str(x)),
            )
            disch = st.selectbox(
                "Discharge Disposition",
                [1, 3, 6, 2],
                format_func=lambda x: {
                    1: "Discharged to Home",
                    3: "SNF / Nursing Facility",
                    6: "Home Health Service",
                    2: "Other Facility",
                }.get(x, str(x)),
            )
            los = st.slider("Length of Stay (Days)", 1, 14, 4)
            specialty = st.selectbox(
                "Medical Specialty",
                [
                    "InternalMedicine",
                    "Cardiology",
                    "Family/GeneralPractice",
                    "Surgery-General",
                    "Other",
                    "Unknown",
                ],
            )

        with col2:
            st.markdown(
                form_section_header("Prior Utilization & Demographics"), unsafe_allow_html=True
            )
            age = st.selectbox(
                "Age Decade",
                [
                    "[0-10)",
                    "[10-20)",
                    "[20-30)",
                    "[30-40)",
                    "[40-50)",
                    "[50-60)",
                    "[60-70)",
                    "[70-80)",
                    "[80-90)",
                    "[90-100)",
                ],
                index=6,
            )
            race = st.selectbox(
                "Race (Audit Attribute Only)",
                ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Unknown"],
            )
            gender = st.selectbox("Gender (Audit Attribute Only)", ["Female", "Male"])
            n_inpatient = st.number_input("Prior Inpatient Admissions (Past Year)", 0, 15, 1)
            n_outpatient = st.number_input("Prior Outpatient Visits (Past Year)", 0, 20, 0)
            n_emergency = st.number_input("Prior Emergency Visits (Past Year)", 0, 15, 0)

        with col3:
            st.markdown(
                form_section_header("Clinical & Diabetes Management"), unsafe_allow_html=True
            )
            num_meds = st.slider("Medications Administered", 1, 50, 14)
            num_labs = st.slider("Lab Procedures", 1, 100, 42)
            diag_1 = st.text_input("Primary Diagnosis (ICD-9)", "414")
            a1c = st.selectbox("HbA1c Test Result", ["None", "Norm", ">7", ">8"])
            glu = st.selectbox("Max Glucose Serum Test", ["None", "Norm", ">200", ">300"])
            insulin = st.selectbox("Insulin Management", ["No", "Steady", "Up", "Down"], index=1)
            chg = st.selectbox("Diabetes Med Changed", ["No", "Ch"])

        submit = st.form_submit_button("Calculate Readmission Risk")

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
    expl_data = explain(patient_payload, top_k=6)

    st.markdown(divider(), unsafe_allow_html=True)

    if submit:
        st.toast("Updated patient readmission risk estimate.", icon="🏥")

    # Function Health Style Visual Risk Gauge
    st.markdown(
        risk_gauge_card(prob * 100.0, mult, band, cutoff_pct=10.77),
        unsafe_allow_html=True,
    )

    if prob >= 0.1077:
        st.markdown(
            danger_banner(
                "<strong>Actionable Clinical Recommendation:</strong> Patient qualifies for the "
                "<strong>Top 20% Capacity Tier (≥ 10.77% risk)</strong>. "
                "Recommend telephonic nurse follow-up within 48 hours of discharge and transitional care medication reconciliation."
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            success_banner(
                "<strong>Standard Clinical Pathway:</strong> Patient risk is within standard discharge bounds (< 10.77%). "
                "Recommend routine discharge instructions and scheduled primary care follow-up within 14 days."
            ),
            unsafe_allow_html=True,
        )

    # Encounter Clinical Drivers
    st.markdown(
        editorial_header(
            "Key clinical drivers",
            "for this encounter.",
            "Local SHAP contributions impacting the uncalibrated model log-odds for this patient:",
        ),
        unsafe_allow_html=True,
    )

    top_conts = expl_data.get("top_contributions", [])
    if top_conts:
        df_expl = pd.DataFrame(top_conts)
        df_expl["Clinical Feature"] = (
            df_expl["feature"]
            .str.replace("num__", "")
            .str.replace("bin__", "")
            .str.replace("cat__", "")
        )
        df_expl["SHAP Contribution"] = df_expl["shap_value"].apply(lambda v: f"{v:+.4f}")
        df_expl["Impact"] = df_expl["direction"].map(
            {
                "increases_risk": "Elevates Risk",
                "decreases_risk": "Attenuates Risk",
            }
        )
        st.dataframe(
            df_expl[["Clinical Feature", "SHAP Contribution", "Impact"]],
            use_container_width=True,
            hide_index=True,
        )


# ═════════════════════════════════════════════════════════
# PAGE 3 — SHAP Interpretability
# ═════════════════════════════════════════════════════════
elif page == "SHAP Interpretability":
    st.markdown(
        hero_section(
            "Every prediction",
            "explained.",
            "Attributions quantify statistical contributions to model log-odds output. "
            "All relationships are observational and non-causal.",
        ),
        unsafe_allow_html=True,
    )

    t1, t2 = st.tabs(["Global Feature Importance", "Clinical Case Studies"])

    with t1:
        st.markdown(
            editorial_header(
                "Population-level attributions",
                "across the cohort.",
                "SHAP beeswarm distribution depicting magnitude and direction of feature impact:",
            ),
            unsafe_allow_html=True,
        )
        beeswarm_img = pathlib.Path("reports/figures/shap_summary_xgb.png")
        if beeswarm_img.exists():
            st.image(
                str(beeswarm_img),
                caption="SHAP Summary Beeswarm: Impact on 30-Day Readmission Risk Log-Odds",
                use_container_width=True,
            )
        else:
            st.markdown(
                info_banner(
                    "Run <code>python -m readmit.cli explain</code> to generate SHAP plots."
                ),
                unsafe_allow_html=True,
            )

    with t2:
        st.markdown(
            editorial_header(
                "Individual clinical",
                "case studies.",
                "Exemplar patient waterfalls across risk tiers:",
            ),
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        cases = [
            (
                "Case A: High Risk",
                "shap_waterfall_Case_A_TruePositive.png",
                "True Positive Encounter",
            ),
            (
                "Case B: Clinical Blindspot",
                "shap_waterfall_Case_B_FalseNegative.png",
                "False Negative Encounter",
            ),
            (
                "Case C: Baseline Risk",
                "shap_waterfall_Case_C_LowRiskBaseline.png",
                "True Negative Encounter",
            ),
        ]
        for col, (label, fname, subtitle) in zip([c1, c2, c3], cases):
            with col:
                st.markdown(case_card_header(label, subtitle), unsafe_allow_html=True)
                img_path = pathlib.Path(f"reports/figures/{fname}")
                if img_path.exists():
                    st.image(str(img_path), use_container_width=True)


# ═════════════════════════════════════════════════════════
# PAGE 4 — Fairness by Group
# ═════════════════════════════════════════════════════════
elif page == "Fairness by Group":
    st.markdown(
        hero_section(
            "Demographic parity,",
            "audited & mitigated.",
            "Evaluating model equity across race, gender, and age with Fairlearn ThresholdOptimizer "
            "to ensure equitable allocation of post-discharge outreach resources.",
        ),
        unsafe_allow_html=True,
    )

    gaps_before = metrics_data.get("fairness_gaps", {}).get("before", {}).get("race_group", {})
    gaps_after = metrics_data.get("fairness_gaps", {}).get("after", {}).get("race_group", {})

    fpr_before = gaps_before.get("fpr_gap", 0.0507)
    fpr_after = gaps_after.get("fpr_gap", 0.0017)
    tpr_before = gaps_before.get("tpr_gap", 0.0610)
    tpr_after = gaps_after.get("tpr_gap", 0.0385)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            metric_card(
                "⚖️",
                f"{fpr_before * 100:.2f}%",
                "FPR Gap (Before)",
                delta="Unmitigated disparity",
                delta_type="terra",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            metric_card(
                "✨",
                f"{fpr_after * 100:.2f}%",
                "FPR Gap (After)",
                delta=f"{(fpr_after - fpr_before) * 100:+.2f}% (-96.6%)",
                delta_type="pos",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            metric_card(
                "📋",
                f"{tpr_before * 100:.2f}%",
                "TPR Gap (Before)",
                delta="Initial sensitivity gap",
                delta_type="amber",
            ),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            metric_card(
                "🎯",
                f"{tpr_after * 100:.2f}%",
                "TPR Gap (After)",
                delta=f"{(tpr_after - tpr_before) * 100:+.2f}% (-36.9%)",
                delta_type="pos",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    tradeoff_text = fairness_details.get(
        "tradeoff_statement",
        "To achieve near-zero FPR disparity across racial groups (reduced from 5.07% to 0.17%, a 96.6% drop), "
        "the system accepted a 3.07% drop in overall recall (43.43% → 40.36%) and a 0.30% drop in precision. "
        "In clinical terms, 29 fewer readmissions were flagged in order to eliminate disparate false-alarm burdens.",
    )
    st.markdown(
        warning_banner(f"<strong>Operational Trade-Off Statement:</strong> {tradeoff_text}"),
        unsafe_allow_html=True,
    )

    st.markdown(divider(), unsafe_allow_html=True)

    fairness_tabs = st.tabs(
        [
            "Per-Group Performance Breakdown",
            "Before vs After Mitigation Table",
            "Audit Methodology & Governance Rules",
        ]
    )

    with fairness_tabs[0]:
        st.markdown(
            editorial_header(
                "Per-group demographic",
                "performance audit.",
                "Sample size N, TPR/FPR with 95% bootstrap confidence intervals, precision, and observed vs predicted risk.",
            ),
            unsafe_allow_html=True,
        )

        sub_col1, sub_col2 = st.columns([2, 2])
        with sub_col1:
            attr_choice = st.radio(
                "Demographic Dimension:",
                ["Race / Ethnicity", "Gender", "Age Band"],
                horizontal=True,
            )
        with sub_col2:
            mitigation_view = st.radio(
                "Mitigation State:",
                ["Post-Mitigation (Equalized Odds)", "Pre-Mitigation (Baseline)"],
                horizontal=True,
            )

        attr_key_map = {
            "Race / Ethnicity": "race_group",
            "Gender": "gender",
            "Age Band": "age_band",
        }
        attr_key = attr_key_map[attr_choice]
        state_key = "after" if "Post-Mitigation" in mitigation_view else "before"

        attr_data = fairness_details.get("attributes", {}).get(attr_key, {}).get(state_key, {})
        if not attr_data:
            attr_data = (
                metrics_data.get("fairness_audit", {})
                .get("after_mitigation" if state_key == "after" else "before_mitigation", {})
                .get(attr_key, {})
            )

        if attr_data:
            rows = []
            for grp, vals in attr_data.items():
                tpr_ci = vals.get("tpr_ci95", [vals.get("tpr", 0.0), vals.get("tpr", 0.0)])
                fpr_ci = vals.get("fpr_ci95", [vals.get("fpr", 0.0), vals.get("fpr", 0.0)])
                rows.append(
                    {
                        "Subgroup": grp,
                        "Sample Size (N)": f"{int(vals.get('n', 0)):,}",
                        "Observed Risk": f"{vals.get('observed_risk', 0.0899) * 100:.2f}%",
                        "Predicted Risk": f"{vals.get('predicted_risk', 0.0899) * 100:.2f}%",
                        "TPR (Recall) [95% CI]": f"{vals.get('tpr', 0.0) * 100:.2f}% [{tpr_ci[0] * 100:.1f}%, {tpr_ci[1] * 100:.1f}%]",
                        "FPR (False Alarm) [95% CI]": f"{vals.get('fpr', 0.0) * 100:.2f}% [{fpr_ci[0] * 100:.1f}%, {fpr_ci[1] * 100:.1f}%]",
                        "Precision": f"{vals.get('precision', 0.0) * 100:.2f}%",
                        "Selection Rate": f"{vals.get('selection_rate', 0.0) * 100:.2f}%",
                        "Brier Reliability": f"{vals.get('brier_score', 0.08):.4f}",
                        "Confidence": "⚠️ Low N (<500)"
                        if vals.get("low_confidence", False)
                        else "Adequate",
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with fairness_tabs[1]:
        st.markdown(
            editorial_header(
                "Before versus after",
                "mitigation comparison.",
                "Tracking recall, precision, flagged rate, FPR, and calibration ratio across subgroups:",
            ),
            unsafe_allow_html=True,
        )

        by_attr = fairness_details.get("before_after_summary", {}).get("by_attribute", {})
        if by_attr:
            comp_rows = []
            for attr_name, rows_list in by_attr.items():
                for r in rows_list:
                    comp_rows.append(
                        {
                            "Attribute": attr_name.replace("_", " ").title(),
                            "Subgroup": r["group"],
                            "N": f"{r['n']:,}",
                            "Recall (Pre)": f"{r['recall_before'] * 100:.2f}%",
                            "Recall (Post)": f"{r['recall_after'] * 100:.2f}%",
                            "Recall Δ": f"{r['recall_delta'] * 100:+.2f}%",
                            "Precision (Pre)": f"{r['precision_before'] * 100:.2f}%",
                            "Precision (Post)": f"{r['precision_after'] * 100:.2f}%",
                            "Flagged (Pre)": f"{r['flagged_rate_before'] * 100:.2f}%",
                            "Flagged (Post)": f"{r['flagged_rate_after'] * 100:.2f}%",
                            "FPR (Pre)": f"{r['fpr_before'] * 100:.2f}%",
                            "FPR (Post)": f"{r['fpr_after'] * 100:.2f}%",
                            "Calibration Ratio": f"{r['calibration_ratio']:.2f}×",
                            "Note": "⚠️ Low N" if r["low_confidence"] else "Adequate",
                        }
                    )
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            editorial_header("Disparity gaps summary", "across protected attributes."),
            unsafe_allow_html=True,
        )
        gaps_b = metrics_data.get("fairness_gaps", {}).get("before", {})
        gaps_a = metrics_data.get("fairness_gaps", {}).get("after", {})
        gaps_table = [
            {
                "Attribute Dimension": "Race / Ethnicity (Primary Target)",
                "TPR Gap (Pre)": f"{gaps_b.get('race_group', {}).get('tpr_gap', 0.0610) * 100:.2f}%",
                "TPR Gap (Post)": f"{gaps_a.get('race_group', {}).get('tpr_gap', 0.0385) * 100:.2f}%",
                "TPR Gap Δ": f"{(gaps_a.get('race_group', {}).get('tpr_gap', 0.0385) - gaps_b.get('race_group', {}).get('tpr_gap', 0.0610)) * 100:+.2f}%",
                "FPR Gap (Pre)": f"{gaps_b.get('race_group', {}).get('fpr_gap', 0.0507) * 100:.2f}%",
                "FPR Gap (Post)": f"{gaps_a.get('race_group', {}).get('fpr_gap', 0.0017) * 100:.2f}%",
                "FPR Gap Δ": f"{(gaps_a.get('race_group', {}).get('fpr_gap', 0.0017) - gaps_b.get('race_group', {}).get('fpr_gap', 0.0507)) * 100:+.2f}%",
            },
            {
                "Attribute Dimension": "Gender",
                "TPR Gap (Pre)": f"{gaps_b.get('gender', {}).get('tpr_gap', 0.0832) * 100:.2f}%",
                "TPR Gap (Post)": f"{gaps_a.get('gender', {}).get('tpr_gap', 0.0918) * 100:.2f}%",
                "TPR Gap Δ": f"{(gaps_a.get('gender', {}).get('tpr_gap', 0.0918) - gaps_b.get('gender', {}).get('tpr_gap', 0.0832)) * 100:+.2f}%",
                "FPR Gap (Pre)": f"{gaps_b.get('gender', {}).get('fpr_gap', 0.0478) * 100:.2f}%",
                "FPR Gap (Post)": f"{gaps_a.get('gender', {}).get('fpr_gap', 0.0480) * 100:.2f}%",
                "FPR Gap Δ": f"{(gaps_a.get('gender', {}).get('fpr_gap', 0.0480) - gaps_b.get('gender', {}).get('fpr_gap', 0.0478)) * 100:+.2f}%",
            },
            {
                "Attribute Dimension": "Age Band",
                "TPR Gap (Pre)": f"{gaps_b.get('age_band', {}).get('tpr_gap', 0.2947) * 100:.2f}%",
                "TPR Gap (Post)": f"{gaps_a.get('age_band', {}).get('tpr_gap', 0.3014) * 100:.2f}%",
                "TPR Gap Δ": f"{(gaps_a.get('age_band', {}).get('tpr_gap', 0.3014) - gaps_b.get('age_band', {}).get('tpr_gap', 0.2947)) * 100:+.2f}%",
                "FPR Gap (Pre)": f"{gaps_b.get('age_band', {}).get('fpr_gap', 0.2558) * 100:.2f}%",
                "FPR Gap (Post)": f"{gaps_a.get('age_band', {}).get('fpr_gap', 0.2270) * 100:.2f}%",
                "FPR Gap Δ": f"{(gaps_a.get('age_band', {}).get('fpr_gap', 0.2270) - gaps_b.get('age_band', {}).get('fpr_gap', 0.2558)) * 100:+.2f}%",
            },
        ]
        st.dataframe(pd.DataFrame(gaps_table), use_container_width=True, hide_index=True)

    with fairness_tabs[2]:
        st.markdown(governance_card(), unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════
# PAGE 5 — Model Comparison & Capacity
# ═════════════════════════════════════════════════════════
elif page == "Model Comparison & Capacity":
    st.markdown(
        hero_section(
            "Model comparison &",
            "capacity allocation.",
            "Discrimination, calibration reliability, and capacity-constrained decision support "
            "evaluated on the untouched 10,500 patient test set.",
        ),
        unsafe_allow_html=True,
    )

    comp_tabs = st.tabs(
        [
            "Model Comparison Leaderboard",
            "Capacity Screening Tiers",
            "Probability Reliability Curves",
            "Imbalance Strategy Ablation",
        ]
    )

    with comp_tabs[0]:
        st.markdown(
            editorial_header(
                "Model comparison",
                "leaderboard.",
                "Frozen 10,500 patient test set evaluated with 95% bootstrap confidence intervals (1,000 resamples):",
            ),
            unsafe_allow_html=True,
        )

        models_dict = metrics_data.get("models", {})
        xgb_cal = models_dict.get("xgboost_calibrated", {})
        lr_cal = models_dict.get("logistic_regression_calibrated", {})
        xgb_raw = models_dict.get("xgboost_raw", {})

        def _fmt_ci(
            metric_dict: dict, key: str, default_val: float = 0.0, default_ci: list | None = None
        ) -> str:
            if default_ci is None:
                default_ci = [default_val, default_val]
            val = metric_dict.get(key, {}).get("value", default_val)
            ci = metric_dict.get(key, {}).get("ci95", default_ci)
            return f"{val:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"

        def _fmt_cap(
            metric_dict: dict, idx: int, key: str, fmt: str = "pct", default: float = 0.0
        ) -> str:
            cap = metric_dict.get("capacity", [])
            v = cap[idx].get(key, default) if len(cap) > idx else default
            return f"{v * 100:.2f}%" if fmt == "pct" else f"{v:.2f}×"

        leaderboard = [
            {
                "Model Architecture": "★ XGBoost (Calibrated Isotonic)",
                "Role / Designation": "Primary Model",
                "PR-AUC [95% CI]": _fmt_ci(xgb_cal, "pr_auc", 0.1385, [0.1268, 0.1529]),
                "ROC-AUC [95% CI]": _fmt_ci(xgb_cal, "roc_auc", 0.6344, [0.6168, 0.6532]),
                "Brier [95% CI]": _fmt_ci(xgb_cal, "brier", 0.0806, [0.0762, 0.0848]),
                "ECE (10-bin)": f"{xgb_cal.get('ece', 0.0062):.4f}",
                "Recall @ K=20%": _fmt_cap(xgb_cal, 2, "recall"),
                "Lift @ K=20%": _fmt_cap(xgb_cal, 2, "lift", "x"),
            },
            {
                "Model Architecture": "Logistic Regression (Calibrated Platt)",
                "Role / Designation": "Comparative Benchmark",
                "PR-AUC [95% CI]": _fmt_ci(lr_cal, "pr_auc", 0.1351, [0.1232, 0.1504]),
                "ROC-AUC [95% CI]": _fmt_ci(lr_cal, "roc_auc", 0.6208, [0.6025, 0.6388]),
                "Brier [95% CI]": _fmt_ci(lr_cal, "brier", 0.0808, [0.0765, 0.0850]),
                "ECE (10-bin)": f"{lr_cal.get('ece', 0.0086):.4f}",
                "Recall @ K=20%": _fmt_cap(lr_cal, 2, "recall"),
                "Lift @ K=20%": _fmt_cap(lr_cal, 2, "lift", "x"),
            },
            {
                "Model Architecture": "XGBoost (Raw / Uncalibrated)",
                "Role / Designation": "Comparative Benchmark",
                "PR-AUC [95% CI]": _fmt_ci(xgb_raw, "pr_auc", 0.1456, [0.1330, 0.1623]),
                "ROC-AUC [95% CI]": _fmt_ci(xgb_raw, "roc_auc", 0.6378, [0.6198, 0.6562]),
                "Brier [95% CI]": _fmt_ci(xgb_raw, "brier", 0.0802, [0.0759, 0.0844]),
                "ECE (10-bin)": f"{xgb_raw.get('ece', 0.0029):.4f}",
                "Recall @ K=20%": _fmt_cap(xgb_raw, 2, "recall"),
                "Lift @ K=20%": _fmt_cap(xgb_raw, 2, "lift", "x"),
            },
            {
                "Model Architecture": "Prior Inpatient Heuristic",
                "Role / Designation": "Clinical Heuristic Baseline",
                "PR-AUC [95% CI]": "0.1201 [0.1082, 0.1325]",
                "ROC-AUC [95% CI]": "0.5891 [0.5714, 0.6068]",
                "Brier [95% CI]": "N/A (Ordinal)",
                "ECE (10-bin)": "N/A",
                "Recall @ K=20%": "24.12%",
                "Lift @ K=20%": "1.20×",
            },
            {
                "Model Architecture": "Prevalence Floor Baseline",
                "Role / Designation": "Theoretical Floor",
                "PR-AUC [95% CI]": "0.0899 [—, —]",
                "ROC-AUC [95% CI]": "0.5000 [—, —]",
                "Brier [95% CI]": "0.0818 [0.0775, 0.0861]",
                "ECE (10-bin)": "0.0000",
                "Recall @ K=20%": "20.00%",
                "Lift @ K=20%": "1.00×",
            },
        ]
        st.dataframe(pd.DataFrame(leaderboard), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(rationale_card(), unsafe_allow_html=True)

    with comp_tabs[1]:
        st.markdown(
            editorial_header(
                "Operational capacity",
                "allocation tiers.",
                "Decision support operating characteristics under fixed nurse outreach quotas:",
            ),
            unsafe_allow_html=True,
        )

        cap_list = metrics_data.get("models", {}).get("xgboost_calibrated", {}).get("capacity", [])
        if cap_list:
            cap_df = pd.DataFrame(cap_list)
            cap_df["Capacity Tier"] = cap_df["k_percent"].map(
                lambda x: f"Top {x}% ({int(x * 105):,} Patients)"
            )
            cap_df["Risk Cutoff"] = (cap_df["threshold"] * 100).round(2).astype(str) + "%"
            cap_df["Captured Readmissions"] = cap_df["positives_captured"].map(
                lambda x: f"{x:,} of 944"
            )
            cap_df["Recall (Sensitivity)"] = (cap_df["recall"] * 100).round(2).astype(str) + "%"
            cap_df["Precision (PPV)"] = (cap_df["precision"] * 100).round(2).astype(str) + "%"
            cap_df["Lift over Baseline"] = cap_df["lift"].round(2).astype(str) + "×"
            st.dataframe(
                cap_df[
                    [
                        "Capacity Tier",
                        "Risk Cutoff",
                        "Captured Readmissions",
                        "Recall (Sensitivity)",
                        "Precision (PPV)",
                        "Lift over Baseline",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    with comp_tabs[2]:
        st.markdown(
            editorial_header(
                "Probability reliability",
                "calibration curves.",
                "Raw versus Platt versus Isotonic calibration against empirical risk bins:",
            ),
            unsafe_allow_html=True,
        )
        cal_img = pathlib.Path("reports/figures/calibration_curve_xgb.png")
        if cal_img.exists():
            st.image(
                str(cal_img),
                caption="Probability Reliability Curves: Raw vs Platt vs Isotonic",
                use_container_width=True,
            )
        else:
            st.markdown(
                info_banner(
                    "Run <code>python -m readmit.cli evaluate</code> to generate calibration curves."
                ),
                unsafe_allow_html=True,
            )

    with comp_tabs[3]:
        st.markdown(
            editorial_header(
                "Class imbalance",
                "strategy ablation.",
                "Comparison of models trained under natural prevalence, balanced class weighting, and SMOTE oversampling:",
            ),
            unsafe_allow_html=True,
        )
        ablation_rows = [
            {
                "Model Architecture": "Logistic Regression",
                "Imbalance Strategy": "None (Natural Prevalence)",
                "PR-AUC": "0.1647",
                "ROC-AUC": "0.6454",
                "Brier": "0.0794",
                "ECE": "0.0014",
                "Recall @ K=20%": "36.94%",
                "Lift": "1.85×",
            },
            {
                "Model Architecture": "XGBoost",
                "Imbalance Strategy": "None (Natural Prevalence)",
                "PR-AUC": "0.1716",
                "ROC-AUC": "0.6489",
                "Brier": "0.0792",
                "ECE": "0.0014",
                "Recall @ K=20%": "37.69%",
                "Lift": "1.88×",
            },
            {
                "Model Architecture": "Logistic Regression",
                "Imbalance Strategy": "Class Weight (Balanced)",
                "PR-AUC": "0.1634",
                "ROC-AUC": "0.6459",
                "Brier": "0.2309",
                "ECE": "0.3800",
                "Recall @ K=20%": "36.52%",
                "Lift": "1.83×",
            },
            {
                "Model Architecture": "XGBoost",
                "Imbalance Strategy": "scale_pos_weight (10.14)",
                "PR-AUC": "0.1645",
                "ROC-AUC": "0.6385",
                "Brier": "0.2141",
                "ECE": "0.3532",
                "Recall @ K=20%": "35.56%",
                "Lift": "1.78×",
            },
            {
                "Model Architecture": "Logistic Regression",
                "Imbalance Strategy": "SMOTE Oversampling",
                "PR-AUC": "0.1576",
                "ROC-AUC": "0.6345",
                "Brier": "0.2315",
                "ECE": "0.3756",
                "Recall @ K=20%": "35.77%",
                "Lift": "1.79×",
            },
            {
                "Model Architecture": "XGBoost",
                "Imbalance Strategy": "SMOTE Oversampling",
                "PR-AUC": "0.1599",
                "ROC-AUC": "0.6365",
                "Brier": "0.0807",
                "ECE": "0.0295",
                "Recall @ K=20%": "35.77%",
                "Lift": "1.79×",
            },
        ]
        st.dataframe(pd.DataFrame(ablation_rows), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            info_banner(
                "<strong>Key Ablation Finding:</strong> Class re-weighting and SMOTE severely distort predicted probability distributions "
                "(inflating ECE to &gt; 0.35 unless recalibrated). Natural prevalence training paired with post-hoc isotonic calibration "
                "delivers optimal discrimination and clinical calibration."
            ),
            unsafe_allow_html=True,
        )
