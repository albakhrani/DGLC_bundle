"""
Idea 1 - Stage 7: compile the diagnostic + remedy into one paper-ready table.

Merges the Merit-Reward Reversal diagnostic (gap05) and the DGLC remedy (gap06)
into a single master summary (CSV + LaTeX) for the methods/results section.
"""
from __future__ import annotations
import json
import pandas as pd
import gap_common as G

def jload(p):
    with open(p) as fh: return json.load(fh)

mrr  = jload(f"{G.GAP_MET}/mrr_diagnostic.json")
dglc = jload(f"{G.GAP_MET}/dglc.json")
reg  = {r["regime"]: r for r in dglc["regimes"]}

rows = [
    ("Diagnosis: realized reward gap (F-M)", f"{mrr['delta_label_F_minus_M']:+.3f}",
     f"[{mrr['delta_label_95ci'][0]:+.3f}, {mrr['delta_label_95ci'][1]:+.3f}]",
     "women penalised by the label"),
    ("Diagnosis: merit selection gap (F-M)", f"{mrr['delta_merit_F_minus_M']:+.3f}",
     f"[{mrr['delta_merit_95ci'][0]:+.3f}, {mrr['delta_merit_95ci'][1]:+.3f}]",
     "women favoured on merit"),
    ("Diagnosis: reversal magnitude", f"{mrr['reversal_magnitude']:+.3f}",
     f"[{mrr['reversal_magnitude_95ci'][0]:+.3f}, {mrr['reversal_magnitude_95ci'][1]:+.3f}]",
     "label-resident (reversal=%s)" % mrr["merit_reward_reversal"]),
    ("Oaxaca unexplained (triangulation)",
     f"{mrr['triangulation']['oaxaca_unexplained_logpts']:+.3f}", "log pts", "agrees"),
    ("Counterfactual gender effect (triangulation)",
     f"{mrr['triangulation']['counterfactual_female_minus_male_logpts']:+.3f}", "log pts", "agrees"),
    ("Remedy A baseline: women selected", f"{reg['A_baseline_realized']['sel_F']:.3f}", "-", "reference"),
    ("Remedy B post-hoc parity: women selected", f"{reg['B_posthoc_equal_opp']['sel_F']:.3f}", "-",
     f"levels DOWN ({dglc['levelling_down_womens_selection_change']['post_hoc_equal_opp']:+.3f})"),
    ("Remedy C DGLC: women selected", f"{reg['C_dglc_alpha1']['sel_F']:.3f}", "-",
     f"rewards merit ({dglc['levelling_down_womens_selection_change']['dglc']:+.3f})"),
    ("DGLC realized AUROC vs baseline",
     f"{dglc['auroc_preserved_dglc_vs_baseline']['dglc_realized']:.3f}",
     f"(baseline {dglc['auroc_preserved_dglc_vs_baseline']['baseline_realized']:.3f})", "preserved"),
    ("Self-validation: Oaxaca unexplained after DGLC",
     f"{dglc['self_validation_oaxaca_unexplained']['after_logpts']:+.4f}",
     f"(before {dglc['self_validation_oaxaca_unexplained']['before_logpts']:+.4f})", "collapses to 0"),
]
master = pd.DataFrame(rows, columns=["quantity", "value", "95% CI / note", "interpretation"])
master.to_csv(f"{G.GAP_TAB}/MASTER_method_summary.csv", index=False)
with open(f"{G.GAP_TAB}/MASTER_method_summary.tex", "w", encoding="utf-8") as fh:
    fh.write(master.to_latex(index=False, escape=True,
             caption="Merit-Reward Reversal diagnosis and Decomposition-Guided "
                     "Label Correction remedy (AMEO 2015).",
             label="tab:dglc_master", column_format="llll"))
print(master.to_string(index=False))
print(f"\nsaved -> {G.GAP_TAB}/MASTER_method_summary.csv / .tex")
