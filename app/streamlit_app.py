"""Streamlit Interactive Clinical Decision Support Dashboard for ReadmitIQ.

Dark-mode glassmorphism design with teal/coral/gold accent palette.
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
    ACCENT_BLUE,
    ACCENT_CORAL,
    ACCENT_GOLD,
    ACCENT_TEAL,
    danger_banner,
    glass_card,
    gradient_divider,
    hero_section,
    info_banner,
    inject_css,
    metric_card,
    risk_badge,
    safeguard_card,
    section_header,
    success_banner,
    warning_banner,
)

from readmit.inference import explain, load_inference_artifacts, predict

st.set_page_config(
    page_title="ReadmitIQ — 30-Day Readmission Risk",
    page_icon="🏥",
    layout="wide",
)

# ── Inject global CSS theme ──────────────────────────────
st.markdown(inject_css(), unsafe_allow_html=True)


# ── Cached loaders ───────────────────────────────────────
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

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <span style="font-size:2.2rem;">🏥</span>
        <div style="font-size:1.4rem; font-weight:800; color:#F0F2F6;
                     letter-spacing:1px; margin-top:0.3rem;">
            Readmit<span style="color:#00D4AA;">IQ</span>
        </div>
        <div style="font-size:0.75rem; color:#8892B0; margin-top:0.15rem;">
            Clinical Decision Support &amp; Fairness Audit
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(gradient_divider(), unsafe_allow_html=True)
st.sidebar.markdown("##### ⚙️ Operational Assumptions")
st.sidebar.markdown(
    "- **Prediction Point:** Hospital Discharge\n"
    "- **Capacity Tier:** Top 20% targeted\n"
    "- **Cost Ratio:** 5:1 (illustrative)\n"
    "- **Cohort:** 69,987 index encounters"
)

st.sidebar.markdown(gradient_divider(), unsafe_allow_html=True)
st.sidebar.markdown("##### ⚠️ Governance")
st.sidebar.markdown(
    "- Non-causal risk prioritization only\n"
    "- Race & gender excluded from features\n"
    "- Low-N groups (< 500) flagged"
)

st.sidebar.markdown(gradient_divider(), unsafe_allow_html=True)

PAGE_OPTIONS = [
    "Executive Summary",
    "Patient Risk Scoring",
    "SHAP Interpretability",
    "Fairness by Group",
    "Model Comparison & Capacity",
]
PAGE_LABELS = {
    "Executive Summary": "📊 Executive Summary",
    "Patient Risk Scoring": "👤 Patient Risk Scoring",
    "SHAP Interpretability": "🔍 SHAP Interpretability",
    "Fairness by Group": "⚖️ Fairness by Group",
    "Model Comparison & Capacity": "📈 Model Comparison & Capacity",
}

page = st.sidebar.radio(
    "Navigate",
    PAGE_OPTIONS,
    format_func=lambda x: PAGE_LABELS.get(x, x),
)


# ═════════════════════════════════════════════════════════
# PAGE 1 — Executive Summary
# ═════════════════════════════════════════════════════════
if page == "Executive Summary":
    st.markdown(
        hero_section(
            "30-Day Readmission Risk Prediction",
            "AI-driven decision support for post-discharge nurse outreach, "
            "built on the UCI 130-US Hospitals dataset with calibration, "
            "SHAP explainability, and fairness audit.",
        ),
        unsafe_allow_html=True,
    )

    # ── KPI Metric Cards ─────────────────────────────────
    cohort = metrics_data.get("cohort", {})
    xgb = metrics_data.get("models", {}).get("xgboost_calibrated", {})
    cap_20 = xgb.get("capacity", [{}, {}, {}])[2] if len(xgb.get("capacity", [])) >= 3 else {}

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            metric_card(
                "👥",
                f"{cohort.get('n_patients', 69987):,}",
                "Index Patients",
                ACCENT_TEAL,
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            metric_card(
                "📉",
                f"{cohort.get('prevalence', 0.0899) * 100:.2f}%",
                "Baseline Readmission",
                ACCENT_CORAL,
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
                ACCENT_GOLD,
                delta="+54% over prevalence floor",
                delta_type="positive",
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
                ACCENT_BLUE,
                delta=f"{lift_val:.2f}x Lift",
                delta_type="positive",
            ),
            unsafe_allow_html=True,
        )

    st.markdown(gradient_divider(), unsafe_allow_html=True)

    # ── Key Safeguards ───────────────────────────────────
    st.markdown(
        section_header("🛡️", "Key Architectural Safeguards"),
        unsafe_allow_html=True,
    )

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            safeguard_card(
                "🔒",
                "Leakage Prevention",
                [
                    "Patient-grouped split (0 overlap)",
                    "Discharge prediction point only",
                    "Preprocessors fitted on train only",
                ],
            ),
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            safeguard_card(
                "🎯",
                "Probability Calibration",
                [
                    "Isotonic post-hoc calibration",
                    f"ECE: <strong>{xgb.get('ece', 0.0062):.4f}</strong> on test",
                    "Reliable absolute risk estimates",
                ],
            ),
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            safeguard_card(
                "⚖️",
                "Demographic Fairness",
                [
                    "Race & gender excluded from model",
                    "Fairlearn Equalized Odds mitigation",
                    "FPR gap: <strong>5.07% → 0.17%</strong>",
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
            "Individual Patient Risk Scoring",
            "Evaluate a patient encounter at discharge to determine predicted risk, "
            "relative risk vs hospital average, and clinical risk tier.",
        ),
        unsafe_allow_html=True,
    )

    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                section_header("🏨", "Admission & Stay"),
                unsafe_allow_html=True,
            )
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
            los = st.slider("Time in Hospital (Days)", 1, 14, 4)
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
                section_header("📋", "Prior Utilization & Demographics"),
                unsafe_allow_html=True,
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
                "Race (Audit Only — not a model feature)",
                ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Unknown"],
            )
            gender = st.selectbox(
                "Gender (Audit Only — not a model feature)",
                ["Female", "Male"],
            )
            n_inpatient = st.number_input("Prior Inpatient Admissions", 0, 15, 1)
            n_outpatient = st.number_input("Prior Outpatient Visits", 0, 20, 0)
            n_emergency = st.number_input("Prior Emergency Visits", 0, 15, 0)

        with col3:
            st.markdown(
                section_header("💊", "Clinical & Diabetes Management"),
                unsafe_allow_html=True,
            )
            num_meds = st.slider("Medications Administered", 1, 50, 14)
            num_labs = st.slider("Lab Procedures", 1, 100, 42)
            diag_1 = st.text_input("Primary Diagnosis (ICD-9)", "414")
            a1c = st.selectbox("HbA1c Result", ["None", "Norm", ">7", ">8"])
            glu = st.selectbox("Max Glucose Serum", ["None", "Norm", ">200", ">300"])
            insulin = st.selectbox("Insulin Management", ["No", "Steady", "Up", "Down"], index=1)
            chg = st.selectbox("Diabetes Med Changed", ["No", "Ch"])

        submit = st.form_submit_button("🔬  Calculate Readmission Risk")

    # ── Build payload ─────────────────────────────────────
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

    st.markdown(gradient_divider(), unsafe_allow_html=True)

    if submit:
        st.toast("Calculated real-time readmission risk!", icon="🏥")

    # ── Risk Results Cards ────────────────────────────────
    st.markdown(
        section_header("🎯", "Patient Risk Evaluation"),
        unsafe_allow_html=True,
    )

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.markdown(
            metric_card(
                "📊",
                f"{prob * 100:.1f}%",
                "Predicted 30-Day Risk",
                ACCENT_CORAL if prob >= 0.1077 else ACCENT_TEAL,
            ),
            unsafe_allow_html=True,
        )
    with rc2:
        delta_type = "negative" if mult > 1.5 else ("neutral" if mult > 1.0 else "positive")
        st.markdown(
            metric_card(
                "📈",
                f"{mult:.2f}x",
                "Relative Risk vs Average",
                ACCENT_GOLD,
                delta=f"{mult - 1.0:+.2f}x baseline",
                delta_type=delta_type,
            ),
            unsafe_allow_html=True,
        )
    with rc3:
        st.markdown(
            f"""
            <div class="metric-card" style="padding-top:1.75rem;">
                <div class="metric-label" style="margin-bottom:0.75rem;">CLINICAL RISK BAND</div>
                {risk_badge(band)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Actionable recommendation ─────────────────────────
    if prob >= 0.1077:
        st.markdown(
            danger_banner(
                "<strong>🚨 Recommended Action:</strong> Patient falls within the "
                "<strong>Top 20% Capacity Tier</strong> (Risk ≥ 10.77%). "
                "Flag for nurse outreach within 48 hours and transitional care "
                "medication reconciliation."
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            success_banner(
                "<strong>✅ Standard Pathway:</strong> Patient risk within standard "
                "discharge limits (< 10.77%). Standard discharge instructions and "
                "scheduled outpatient follow-up recommended."
            ),
            unsafe_allow_html=True,
        )

    # ── SHAP local explainability ─────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        section_header(
            "🧬",
            "Clinical Feature Drivers",
            "Top SHAP contributions (uncalibrated log-odds) for this encounter:",
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
        df_expl["Clinical Impact"] = df_expl["direction"].map(
            {
                "increases_risk": "🔺 Increases Risk",
                "decreases_risk": "🔻 Decreases Risk",
            }
        )
        st.dataframe(
            df_expl[["Clinical Feature", "SHAP Contribution", "Clinical Impact"]],
            use_container_width=True,
            hide_index=True,
        )


# ═════════════════════════════════════════════════════════
# PAGE 3 — SHAP Interpretability
# ═════════════════════════════════════════════════════════
elif page == "SHAP Interpretability":
    st.markdown(
        hero_section(
            "SHAP Interpretability & Feature Attributions",
            "Explanations quantify statistical contributions to model log-odds. "
            "All attributions are observational and non-causal.",
        ),
        unsafe_allow_html=True,
    )

    t1, t2 = st.tabs(["🌐 Global Feature Importance", "🔬 Case Study Waterfalls"])

    with t1:
        st.markdown(
            section_header(
                "📊",
                "Global Population Attributions",
                "SHAP beeswarm showing feature impact distribution across the validation cohort:",
            ),
            unsafe_allow_html=True,
        )
        beeswarm_img = pathlib.Path("reports/figures/shap_summary_xgb.png")
        if beeswarm_img.exists():
            st.image(
                str(beeswarm_img),
                caption="SHAP Beeswarm: Impact on 30-Day Readmission Risk (Log-Odds)",
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
            section_header("🔬", "Clinical Case Studies"),
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        cases = [
            ("Case A: True Positive", "shap_waterfall_Case_A_TruePositive.png", "High Risk"),
            (
                "Case B: False Negative",
                "shap_waterfall_Case_B_FalseNegative.png",
                "Clinical Blindspot",
            ),
            ("Case C: True Negative", "shap_waterfall_Case_C_LowRiskBaseline.png", "Low Risk"),
        ]
        for col, (label, fname, subtitle) in zip([c1, c2, c3], cases):
            with col:
                st.markdown(
                    glass_card(
                        f"<h4 style='color:#F0F2F6; margin:0 0 0.5rem 0;'>{label}</h4>"
                        f"<p style='color:#8892B0; font-size:0.8rem; margin:0;'>{subtitle}</p>"
                    ),
                    unsafe_allow_html=True,
                )
                img_path = pathlib.Path(f"reports/figures/{fname}")
                if img_path.exists():
                    st.image(str(img_path), use_container_width=True)


# ═════════════════════════════════════════════════════════
# PAGE 4 — Fairness by Group
# ═════════════════════════════════════════════════════════
elif page == "Fairness by Group":
    st.markdown(
        hero_section(
            "Demographic Fairness Audit & Mitigation",
            "Evaluating model parity across demographic subgroups before and after "
            "Fairlearn ThresholdOptimizer (Equalized Odds) mitigation.",
        ),
        unsafe_allow_html=True,
    )

    gaps_before = metrics_data.get("fairness_gaps", {}).get("before", {}).get("race_group", {})
    gaps_after = metrics_data.get("fairness_gaps", {}).get("after", {}).get("race_group", {})

    # ── Disparity Gap Cards ──────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    fpr_before = gaps_before.get("fpr_gap", 0.0507)
    fpr_after = gaps_after.get("fpr_gap", 0.0017)
    tpr_before = gaps_before.get("tpr_gap", 0.0610)
    tpr_after = gaps_after.get("tpr_gap", 0.0385)

    with c1:
        st.markdown(
            metric_card(
                "🔴",
                f"{fpr_before * 100:.2f}%",
                "FPR Gap (Before)",
                ACCENT_CORAL,
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            metric_card(
                "🟢",
                f"{fpr_after * 100:.2f}%",
                "FPR Gap (After)",
                ACCENT_TEAL,
                delta=f"{(fpr_after - fpr_before) * 100:+.2f}%",
                delta_type="positive",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            metric_card(
                "🔴",
                f"{tpr_before * 100:.2f}%",
                "TPR Gap (Before)",
                ACCENT_CORAL,
            ),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            metric_card(
                "🟡",
                f"{tpr_after * 100:.2f}%",
                "TPR Gap (After)",
                ACCENT_GOLD,
                delta=f"{(tpr_after - tpr_before) * 100:+.2f}%",
                delta_type="positive",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Trade-off statement ──────────────────────────────
    tradeoff_text = fairness_details.get(
        "tradeoff_statement",
        "To achieve near-zero FPR disparity (5.07% → 0.17%, 96.6% reduction), "
        "the system accepted a 3.07% recall drop (43.43% → 40.36%) and 0.30% "
        "precision drop. 29 fewer readmissions flagged to eliminate disparate "
        "false-alarm burdens across protected groups.",
    )
    st.markdown(
        warning_banner(f"<strong>⚖️ Trade-Off:</strong> {tradeoff_text}"),
        unsafe_allow_html=True,
    )

    st.markdown(gradient_divider(), unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────
    fairness_tabs = st.tabs(
        [
            "📋 Per-Group Performance",
            "🔄 Before vs After Mitigation",
            "📖 Audit Methodology",
        ]
    )

    with fairness_tabs[0]:
        st.markdown(
            section_header(
                "📋",
                "Per-Group Performance Audit",
                "N, TPR/FPR with 95% bootstrap CIs, precision, selection rate, "
                "and predicted vs observed risk.",
            ),
            unsafe_allow_html=True,
        )

        fc1, fc2 = st.columns([2, 2])
        with fc1:
            attr_choice = st.radio(
                "Demographic Attribute:",
                ["Race / Ethnicity", "Gender", "Age Band"],
                horizontal=True,
            )
        with fc2:
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
                        "N": f"{int(vals.get('n', 0)):,}",
                        "Observed Risk": f"{vals.get('observed_risk', 0.0899) * 100:.2f}%",
                        "Predicted Risk": f"{vals.get('predicted_risk', 0.0899) * 100:.2f}%",
                        "TPR [95% CI]": (
                            f"{vals.get('tpr', 0.0) * 100:.2f}% "
                            f"[{tpr_ci[0] * 100:.1f}%, {tpr_ci[1] * 100:.1f}%]"
                        ),
                        "FPR [95% CI]": (
                            f"{vals.get('fpr', 0.0) * 100:.2f}% "
                            f"[{fpr_ci[0] * 100:.1f}%, {fpr_ci[1] * 100:.1f}%]"
                        ),
                        "Precision": f"{vals.get('precision', 0.0) * 100:.2f}%",
                        "Selection Rate": f"{vals.get('selection_rate', 0.0) * 100:.2f}%",
                        "Brier": f"{vals.get('brier_score', 0.08):.4f}",
                        "Flag": "⚠️ Low N" if vals.get("low_confidence", False) else "✅",
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.markdown(
                info_banner(
                    "Per-group fairness metrics not found. Run "
                    "<code>python scripts/update_fairness_report.py</code>."
                ),
                unsafe_allow_html=True,
            )

    with fairness_tabs[1]:
        st.markdown(
            section_header(
                "🔄",
                "Before vs After Mitigation",
                "Recall, precision, flagged rate, FPR, and calibration ratio by group.",
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
                            "Recall ⬅": f"{r['recall_before'] * 100:.2f}%",
                            "Recall ➡": f"{r['recall_after'] * 100:.2f}%",
                            "Δ Recall": f"{r['recall_delta'] * 100:+.2f}%",
                            "Precision ⬅": f"{r['precision_before'] * 100:.2f}%",
                            "Precision ➡": f"{r['precision_after'] * 100:.2f}%",
                            "Flagged ⬅": f"{r['flagged_rate_before'] * 100:.2f}%",
                            "Flagged ➡": f"{r['flagged_rate_after'] * 100:.2f}%",
                            "FPR ⬅": f"{r['fpr_before'] * 100:.2f}%",
                            "FPR ➡": f"{r['fpr_after'] * 100:.2f}%",
                            "Cal Ratio": f"{r['calibration_ratio']:.2f}x",
                            "Flag": "⚠️" if r["low_confidence"] else "✅",
                        }
                    )
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
        else:
            st.markdown(
                info_banner(
                    "Run <code>python scripts/update_fairness_report.py</code> "
                    "to populate comparison tables."
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            section_header("📊", "Disparity Gaps Summary"),
            unsafe_allow_html=True,
        )
        gaps_b = metrics_data.get("fairness_gaps", {}).get("before", {})
        gaps_a = metrics_data.get("fairness_gaps", {}).get("after", {})
        gaps_table = [
            {
                "Attribute": "Race / Ethnicity",
                "TPR Gap ⬅": f"{gaps_b.get('race_group', {}).get('tpr_gap', 0.0610) * 100:.2f}%",
                "TPR Gap ➡": f"{gaps_a.get('race_group', {}).get('tpr_gap', 0.0385) * 100:.2f}%",
                "Δ TPR": f"{(gaps_a.get('race_group', {}).get('tpr_gap', 0.0385) - gaps_b.get('race_group', {}).get('tpr_gap', 0.0610)) * 100:+.2f}%",
                "FPR Gap ⬅": f"{gaps_b.get('race_group', {}).get('fpr_gap', 0.0507) * 100:.2f}%",
                "FPR Gap ➡": f"{gaps_a.get('race_group', {}).get('fpr_gap', 0.0017) * 100:.2f}%",
                "Δ FPR": f"{(gaps_a.get('race_group', {}).get('fpr_gap', 0.0017) - gaps_b.get('race_group', {}).get('fpr_gap', 0.0507)) * 100:+.2f}%",
            },
            {
                "Attribute": "Gender",
                "TPR Gap ⬅": f"{gaps_b.get('gender', {}).get('tpr_gap', 0.0832) * 100:.2f}%",
                "TPR Gap ➡": f"{gaps_a.get('gender', {}).get('tpr_gap', 0.0918) * 100:.2f}%",
                "Δ TPR": f"{(gaps_a.get('gender', {}).get('tpr_gap', 0.0918) - gaps_b.get('gender', {}).get('tpr_gap', 0.0832)) * 100:+.2f}%",
                "FPR Gap ⬅": f"{gaps_b.get('gender', {}).get('fpr_gap', 0.0478) * 100:.2f}%",
                "FPR Gap ➡": f"{gaps_a.get('gender', {}).get('fpr_gap', 0.0480) * 100:.2f}%",
                "Δ FPR": f"{(gaps_a.get('gender', {}).get('fpr_gap', 0.0480) - gaps_b.get('gender', {}).get('fpr_gap', 0.0478)) * 100:+.2f}%",
            },
            {
                "Attribute": "Age Band",
                "TPR Gap ⬅": f"{gaps_b.get('age_band', {}).get('tpr_gap', 0.2947) * 100:.2f}%",
                "TPR Gap ➡": f"{gaps_a.get('age_band', {}).get('tpr_gap', 0.3014) * 100:.2f}%",
                "Δ TPR": f"{(gaps_a.get('age_band', {}).get('tpr_gap', 0.3014) - gaps_b.get('age_band', {}).get('tpr_gap', 0.2947)) * 100:+.2f}%",
                "FPR Gap ⬅": f"{gaps_b.get('age_band', {}).get('fpr_gap', 0.2558) * 100:.2f}%",
                "FPR Gap ➡": f"{gaps_a.get('age_band', {}).get('fpr_gap', 0.2270) * 100:.2f}%",
                "Δ FPR": f"{(gaps_a.get('age_band', {}).get('fpr_gap', 0.2270) - gaps_b.get('age_band', {}).get('fpr_gap', 0.2558)) * 100:+.2f}%",
            },
        ]
        st.dataframe(pd.DataFrame(gaps_table), use_container_width=True, hide_index=True)

    with fairness_tabs[2]:
        st.markdown(
            section_header("📖", "Audit Methodology & Governance"),
            unsafe_allow_html=True,
        )
        st.markdown(
            glass_card(
                "<h4 style='color:#00D4AA; margin-top:0;'>Clinical AI Governance Rules</h4>"
                "<ul style='color:#8892B0; line-height:1.8;'>"
                "<li><strong>Fairness Invariant:</strong> Race and gender strictly excluded from "
                "predictive features, tracked only for demographic parity audit.</li>"
                "<li><strong>Mitigation:</strong> Fairlearn <code>ThresholdOptimizer"
                "(constraints='equalized_odds', objective='balanced_accuracy_score', "
                "prefit=True)</code></li>"
                "<li><strong>Low-N Policy:</strong> Subgroups with N < 500 flagged as "
                "low-confidence due to bootstrap variance.</li>"
                "<li><strong>Non-Causal:</strong> The model supports risk prioritization; "
                "it does not recommend clinical treatment.</li>"
                "</ul>"
            ),
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════
# PAGE 5 — Model Comparison & Capacity
# ═════════════════════════════════════════════════════════
elif page == "Model Comparison & Capacity":
    st.markdown(
        hero_section(
            "Model Comparison & Operational Capacity",
            "Discrimination (PR-AUC, ROC-AUC), calibration reliability (Brier, ECE), "
            "and capacity-constrained decision support evaluation.",
        ),
        unsafe_allow_html=True,
    )

    comp_tabs = st.tabs(
        [
            "🏆 Leaderboard",
            "📋 Capacity Tiers",
            "📈 Calibration Curves",
            "🔬 Imbalance Ablation",
        ]
    )

    with comp_tabs[0]:
        st.markdown(
            section_header(
                "🏆",
                "Model Leaderboard (Test Set)",
                "Frozen 10,500 patient test set · 95% bootstrap CIs (1,000 resamples)",
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
            if len(cap) > idx:
                v = cap[idx].get(key, default)
            else:
                v = default
            if fmt == "pct":
                return f"{v * 100:.2f}%"
            return f"{v:.2f}x"

        leaderboard = [
            {
                "Model": "⭐ XGBoost (Calibrated Isotonic)",
                "PR-AUC [95% CI]": _fmt_ci(xgb_cal, "pr_auc", 0.1385, [0.1268, 0.1529]),
                "ROC-AUC [95% CI]": _fmt_ci(xgb_cal, "roc_auc", 0.6344, [0.6168, 0.6532]),
                "Brier [95% CI]": _fmt_ci(xgb_cal, "brier", 0.0806, [0.0762, 0.0848]),
                "ECE": f"{xgb_cal.get('ece', 0.0062):.4f}",
                "Recall@20%": _fmt_cap(xgb_cal, 2, "recall"),
                "Lift@20%": _fmt_cap(xgb_cal, 2, "lift", "x"),
            },
            {
                "Model": "LR (Calibrated Platt)",
                "PR-AUC [95% CI]": _fmt_ci(lr_cal, "pr_auc", 0.1351, [0.1232, 0.1504]),
                "ROC-AUC [95% CI]": _fmt_ci(lr_cal, "roc_auc", 0.6208, [0.6025, 0.6388]),
                "Brier [95% CI]": _fmt_ci(lr_cal, "brier", 0.0808, [0.0765, 0.0850]),
                "ECE": f"{lr_cal.get('ece', 0.0086):.4f}",
                "Recall@20%": _fmt_cap(lr_cal, 2, "recall"),
                "Lift@20%": _fmt_cap(lr_cal, 2, "lift", "x"),
            },
            {
                "Model": "XGBoost (Raw / Uncalibrated)",
                "PR-AUC [95% CI]": _fmt_ci(xgb_raw, "pr_auc", 0.1456, [0.1330, 0.1623]),
                "ROC-AUC [95% CI]": _fmt_ci(xgb_raw, "roc_auc", 0.6378, [0.6198, 0.6562]),
                "Brier [95% CI]": _fmt_ci(xgb_raw, "brier", 0.0802, [0.0759, 0.0844]),
                "ECE": f"{xgb_raw.get('ece', 0.0029):.4f}",
                "Recall@20%": _fmt_cap(xgb_raw, 2, "recall"),
                "Lift@20%": _fmt_cap(xgb_raw, 2, "lift", "x"),
            },
            {
                "Model": "Prior Inpatient Heuristic",
                "PR-AUC [95% CI]": "0.1201 [0.1082, 0.1325]",
                "ROC-AUC [95% CI]": "0.5891 [0.5714, 0.6068]",
                "Brier [95% CI]": "N/A (Ordinal)",
                "ECE": "N/A",
                "Recall@20%": "24.12%",
                "Lift@20%": "1.20x",
            },
            {
                "Model": "Prevalence Floor",
                "PR-AUC [95% CI]": "0.0899 [—, —]",
                "ROC-AUC [95% CI]": "0.5000 [—, —]",
                "Brier [95% CI]": "0.0818 [0.0775, 0.0861]",
                "ECE": "0.0000",
                "Recall@20%": "20.00%",
                "Lift@20%": "1.00x",
            },
        ]
        st.dataframe(pd.DataFrame(leaderboard), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            glass_card(
                "<h4 style='color:#00D4AA; margin-top:0;'>⭐ Primary Model Selection</h4>"
                "<ul style='color:#8892B0; line-height:1.8;'>"
                "<li><strong>Discrimination:</strong> 0.1385 PR-AUC (+54% over prevalence), "
                "0.6344 ROC-AUC</li>"
                "<li><strong>Calibration:</strong> ECE 0.0062 · Brier 0.0806 — predicted risks "
                "match observed incidence</li>"
                "<li><strong>Capacity:</strong> Top 20% captures 34.53% of readmissions "
                "(1.73× lift)</li>"
                "</ul>"
            ),
            unsafe_allow_html=True,
        )

    with comp_tabs[1]:
        st.markdown(
            section_header(
                "📋",
                "Capacity-Constrained Decision Tiers",
                "Operating characteristics under fixed nurse outreach quotas:",
            ),
            unsafe_allow_html=True,
        )

        cap_list = metrics_data.get("models", {}).get("xgboost_calibrated", {}).get("capacity", [])
        if cap_list:
            cap_df = pd.DataFrame(cap_list)
            cap_df["Tier"] = cap_df["k_percent"].map(
                lambda x: f"Top {x}% ({int(x * 105):,} patients)"
            )
            cap_df["Cutoff"] = (cap_df["threshold"] * 100).round(2).astype(str) + "%"
            cap_df["Captured"] = cap_df["positives_captured"].map(lambda x: f"{x:,} / 944")
            cap_df["Recall"] = (cap_df["recall"] * 100).round(2).astype(str) + "%"
            cap_df["Precision"] = (cap_df["precision"] * 100).round(2).astype(str) + "%"
            cap_df["Lift"] = cap_df["lift"].round(2).astype(str) + "×"
            st.dataframe(
                cap_df[["Tier", "Cutoff", "Captured", "Recall", "Precision", "Lift"]],
                use_container_width=True,
                hide_index=True,
            )

    with comp_tabs[2]:
        st.markdown(
            section_header(
                "📈",
                "Probability Reliability Curves",
                "Raw vs Platt vs Isotonic calibration against empirical risk bins:",
            ),
            unsafe_allow_html=True,
        )
        cal_img = pathlib.Path("reports/figures/calibration_curve_xgb.png")
        if cal_img.exists():
            st.image(
                str(cal_img),
                caption="Reliability Curves: Raw vs Platt vs Isotonic",
                use_container_width=True,
            )
        else:
            st.markdown(
                info_banner(
                    "Run <code>python -m readmit.cli evaluate</code> to generate calibration plots."
                ),
                unsafe_allow_html=True,
            )

    with comp_tabs[3]:
        st.markdown(
            section_header(
                "🔬",
                "Imbalance Strategy Ablation (Validation)",
                "Natural prevalence vs balanced weighting vs SMOTE oversampling:",
            ),
            unsafe_allow_html=True,
        )
        ablation_rows = [
            {
                "Model": "LR",
                "Strategy": "None (unweighted)",
                "PR-AUC": "0.1647",
                "ROC-AUC": "0.6454",
                "Brier": "0.0794",
                "ECE": "0.0014",
                "Recall@20%": "36.94%",
                "Lift": "1.85×",
            },
            {
                "Model": "XGBoost",
                "Strategy": "None (unweighted)",
                "PR-AUC": "0.1716",
                "ROC-AUC": "0.6489",
                "Brier": "0.0792",
                "ECE": "0.0014",
                "Recall@20%": "37.69%",
                "Lift": "1.88×",
            },
            {
                "Model": "LR",
                "Strategy": "Class Weight (balanced)",
                "PR-AUC": "0.1634",
                "ROC-AUC": "0.6459",
                "Brier": "0.2309",
                "ECE": "0.3800",
                "Recall@20%": "36.52%",
                "Lift": "1.83×",
            },
            {
                "Model": "XGBoost",
                "Strategy": "scale_pos_weight (10.14)",
                "PR-AUC": "0.1645",
                "ROC-AUC": "0.6385",
                "Brier": "0.2141",
                "ECE": "0.3532",
                "Recall@20%": "35.56%",
                "Lift": "1.78×",
            },
            {
                "Model": "LR",
                "Strategy": "SMOTE",
                "PR-AUC": "0.1576",
                "ROC-AUC": "0.6345",
                "Brier": "0.2315",
                "ECE": "0.3756",
                "Recall@20%": "35.77%",
                "Lift": "1.79×",
            },
            {
                "Model": "XGBoost",
                "Strategy": "SMOTE",
                "PR-AUC": "0.1599",
                "ROC-AUC": "0.6365",
                "Brier": "0.0807",
                "ECE": "0.0295",
                "Recall@20%": "35.77%",
                "Lift": "1.79×",
            },
        ]
        st.dataframe(pd.DataFrame(ablation_rows), use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            info_banner(
                "<strong>Key Finding:</strong> Class reweighting and SMOTE severely distort "
                "the predicted risk distribution (ECE > 0.35 without recalibration). "
                "Natural prevalence training + isotonic post-hoc calibration delivers "
                "optimal discrimination and clinical reliability."
            ),
            unsafe_allow_html=True,
        )
