"""Script to update reports/fairness_before_after.md with test set metrics and bootstrap CIs."""

import json
import pathlib

def main():
    fairness_details_path = pathlib.Path("reports/fairness_details.json")
    with open(fairness_details_path, encoding="utf-8") as f:
        d = json.load(f)

    md = []
    md.append("# Algorithmic Fairness Audit and Mitigation Trade-Off Report\n")
    md.append("> [!IMPORTANT]\n")
    md.append("> **Governance Policy:** Race and gender are strictly excluded from predictive model features and preserved exclusively as demographic audit attributes. Age is included as a clinical predictor (numeric midpoint) and audited across categorical age bands.\n")
    md.append("> **Sample Size Warning:** Groups with N < 500 are labeled as low-confidence due to statistical power constraints and wider bootstrap confidence intervals.\n\n")

    md.append("## Executive Fairness Summary (Primary Attribute: Race/Ethnicity)\n")
    md.append("| State | Constraint Objective | Overall Recall | Overall Precision | Selection Rate | TPR Disparity Gap | FPR Disparity Gap |")
    md.append("|---|---|---|---|---|---|---|")
    md.append("| **Before Mitigation** (Capacity Top 20%) | None (Single Global Threshold: 10.77%) | 43.43% | 14.43% | 27.07% | 6.10% | 5.07% |")
    md.append("| **After Mitigation** (Fairlearn Equalized Odds) | Equalized Odds (Group-Specific Thresholds) | 40.36% | 14.12% | 25.70% | 3.85% | 0.17% |\n")
    md.append("---\n")

    md.append("## Before vs. After Mitigation Comparison Table (Test Set)\n")
    md.append("| Attribute | Subgroup | Sample Size (N) | Recall (Before) | Recall (After) | Recall Δ | Precision (Before) | Precision (After) | Flagged Rate (Before) | Flagged Rate (After) | FPR (Before) | FPR (After) | Pred Risk | Obs Risk | Calib Ratio | Note |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    for attr, rows in d["before_after_summary"]["by_attribute"].items():
        for r in rows:
            flag = "⚠️ Low N" if r["low_confidence"] else "Adequate"
            rec_b = f"{r['recall_before']*100:.2f}%"
            rec_a = f"{r['recall_after']*100:.2f}%"
            rec_d = f"{r['recall_delta']*100:+.2f}%"
            prec_b = f"{r['precision_before']*100:.2f}%"
            prec_a = f"{r['precision_after']*100:.2f}%"
            sel_b = f"{r['flagged_rate_before']*100:.2f}%"
            sel_a = f"{r['flagged_rate_after']*100:.2f}%"
            fpr_b = f"{r['fpr_before']*100:.2f}%"
            fpr_a = f"{r['fpr_after']*100:.2f}%"
            pred_r = f"{r['predicted_risk']*100:.2f}%"
            obs_r = f"{r['observed_risk']*100:.2f}%"
            cal_rat = f"{r['calibration_ratio']:.2f}x"
            grp_name = r['group']
            n_str = f"{r['n']:,}"
            md.append(f"| {attr} | `{grp_name}` | {n_str} | {rec_b} | {rec_a} | {rec_d} | {prec_b} | {prec_a} | {sel_b} | {sel_a} | {fpr_b} | {fpr_a} | {pred_r} | {obs_r} | {cal_rat} | {flag} |")

    md.append("\n---\n")
    md.append("## Detailed Per-Group Performance Breakdown with 95% Bootstrap CIs (Test Set)\n")

    for attr, attr_title in [("race_group", "Race / Ethnicity"), ("gender", "Gender"), ("age_band", "Age Band")]:
        md.append(f"\n### {attr_title}\n")
        md.append("| Subgroup | N | Observed Risk | Predicted Risk | TPR (Recall) [95% CI] | FPR (False Alarm) [95% CI] | Precision | Selection Rate | Confidence Note |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        groups = d["attributes"][attr]["after"]
        for grp, r in groups.items():
            flag = "⚠️ Low N (<500)" if r["low_confidence"] else "Adequate N"
            tpr_str = f"{r['tpr']*100:.2f}% [{r['tpr_ci95'][0]*100:.1f}%, {r['tpr_ci95'][1]*100:.1f}%]"
            fpr_str = f"{r['fpr']*100:.2f}% [{r['fpr_ci95'][0]*100:.1f}%, {r['fpr_ci95'][1]*100:.1f}%]"
            obs_str = f"{r['observed_risk']*100:.2f}%"
            pred_str = f"{r['predicted_risk']*100:.2f}%"
            prec_str = f"{r['precision']*100:.2f}%"
            sel_str = f"{r['selection_rate']*100:.2f}%"
            n_val = f"{r['n']:,}"
            md.append(f"| `{grp}` | {n_val} | {obs_str} | {pred_str} | {tpr_str} | {fpr_str} | {prec_str} | {sel_str} | {flag} |")

    md.append("\n---\n")
    md.append("## Clinical & Operational Trade-Off: What Was Given Up in Mitigation\n")
    md.append(f"> {d['tradeoff_statement']}\n\n")
    md.append("1. **Parity Gain vs. Opportunity Cost:** Equalized Odds post-processing dramatically equalized the burden of false alarms across racial subgroups (FPR disparity dropped from 5.07% to 0.17%, a 96.6% reduction). This directly protects Black and minoritized patients from disproportionate surveillance or stigmatizing post-discharge outreach.\n")
    md.append("2. **Trade-Off Quantification:** Achieving this parity required adjusting group-specific thresholds, causing overall recall to drop by 3.07% (from 43.43% to 40.36%) and precision to drop by 0.30% (from 14.43% to 14.12%). Total flagged volume fell from 2,842 to 2,698 patients (144 fewer interventions), resulting in 29 fewer readmitted patients flagged across the 10,500 test encounters.\n")
    md.append("3. **Governance Recommendation:** Algorithmic mitigation should not operate in a vacuum. We recommend pairing equalized odds post-processing with care team clinical judgment, using calibrated continuous risk scores for prioritization, and auditing demographic parity periodically in production.")

    output_path = pathlib.Path("reports/fairness_before_after.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"Updated {output_path} successfully!")

if __name__ == "__main__":
    main()
