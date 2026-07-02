"""
Idea 1 - Stage 8: cross-dataset validation of the MRR diagnostic + DGLC remedy.

Runs the IDENTICAL gap_core pipeline on every available dataset and compiles one
"it generalizes" summary table + figure. Datasets that are not present (missing
Campus CSV, or no internet for ACS) are skipped with a message, so the script
always produces a result for whatever is available.

Config via environment (optional):
  GAP_DATASETS     comma list, default "ameo,campus,acs"
                   (use "ameo,campus,acs_openml" if census.gov is blocked)
  GAP_CAMPUS_CSV   path to Campus Recruitment CSV (default <ROOT>/campus_recruitment.csv)
  GAP_ACS_STATE    US state for ACSIncome via folktables (default "CA")
  GAP_ACS_MAXROWS  optional subsample cap for ACS (e.g. 60000) to iterate fast

Outputs:
  outputs/gap/cross/cross_dataset_summary.csv / .tex / .json
  outputs/gap/cross/cross_dataset.png
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as C
from utils import set_plot_style, save_json
import gap_common as G
import gap_core as core
import gap_datasets as D

CROSS = os.path.join(G.GAP_DIR, "cross"); os.makedirs(CROSS, exist_ok=True)


def _load_spec(name):
    if name == "campus":
        return D.load_campus(os.environ.get("GAP_CAMPUS_CSV"))
    if name == "acs_openml":
        mr = os.environ.get("GAP_ACS_MAXROWS")
        return D.load_acs_openml(max_rows=int(mr) if mr else None)
    if name == "acs":
        kw = {"state": os.environ.get("GAP_ACS_STATE", "CA")}
        mr = os.environ.get("GAP_ACS_MAXROWS")
        if mr: kw["max_rows"] = int(mr)
        return D.load_acs(**kw)
    return D.load_ameo()


def main():
    set_plot_style()
    names = os.environ.get("GAP_DATASETS", "ameo,campus,acs").split(",")
    rows = []
    for name in [n.strip() for n in names]:
        try:
            s = _load_spec(name)
        except Exception as e:
            print(f"[skip] {name}: {type(e).__name__}: {e}")
            continue
        print(f"[run ] {s['name']}  (n={len(s['df'])})")
        mrr = core.run_mrr(s["df"], s["outcome"], s["group"], s["ref"], s["oth"],
                           s["num"], s["cat"])
        dg = core.run_dglc(s["df"], s["outcome"], s["group"], s["ref"], s["oth"],
                           s["num"], s["cat"])
        rows.append(dict(
            dataset=s["name"], n=len(s["df"]),
            n_female=mrr["n_oth"], n_male=mrr["n_ref"],
            delta_label=mrr["delta_label"],
            delta_merit=mrr["delta_merit"],
            reversal_mag=mrr["reversal_magnitude"],
            reversal_ci_lo=mrr["reversal_magnitude_ci"][0],
            reversal_ci_hi=mrr["reversal_magnitude_ci"][1],
            reversal=mrr["merit_reward_reversal"],
            oaxaca_unexplained=mrr["oaxaca_unexplained"],
            sel_women_baseline=dg["sel_oth"]["A_baseline"],
            sel_women_posthoc=dg["sel_oth"]["B_posthoc"],
            sel_women_dglc=dg["sel_oth"]["C_dglc"],
            levelling_down_posthoc=dg["levelling_down_posthoc"],
            dglc_change=dg["dglc_change"],
            auroc_baseline=dg["auroc_baseline"],
            auroc_dglc=dg["auroc_dglc"],
            selfval_before=dg["self_validation"]["unexplained_before"],
            selfval_after=dg["self_validation"]["unexplained_after"]))

    if not rows:
        print("No datasets available. Add campus_recruitment.csv and/or enable "
              "internet for ACS (try GAP_DATASETS=acs_openml), then re-run.")
        return
    df = pd.DataFrame(rows)
    df.round(4).to_csv(f"{CROSS}/cross_dataset_summary.csv", index=False)
    save_json(rows, f"{CROSS}/cross_dataset_summary.json")

    show = df[["dataset", "n", "reversal_mag", "reversal_ci_lo", "reversal_ci_hi",
               "reversal", "sel_women_baseline", "sel_women_posthoc",
               "sel_women_dglc", "auroc_baseline", "auroc_dglc",
               "selfval_before", "selfval_after"]]
    with open(f"{CROSS}/cross_dataset_summary.tex", "w", encoding="utf-8") as fh:
        fh.write(show.round(3).to_latex(index=False, escape=True,
            caption="Cross-dataset validation of Merit-Reward Reversal and DGLC.",
            label="tab:cross_dataset"))

    # ---- figure: reversal-with-CI  +  selection direction (post-hoc vs DGLC)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    yy = np.arange(len(df))
    err = np.vstack([df["reversal_mag"] - df["reversal_ci_lo"],
                     df["reversal_ci_hi"] - df["reversal_mag"]])
    ax1.errorbar(df["reversal_mag"], yy, xerr=err, fmt="o", color="#2E6F40",
                 capsize=4, lw=2)
    ax1.axvline(0, color="grey", lw=1, ls="--")
    ax1.set_yticks(yy); ax1.set_yticklabels(df["dataset"], fontsize=8)
    ax1.set_xlabel("Merit-Reward Reversal magnitude (95% CI)")
    ax1.set_title("Reversal is positive across datasets\n(label penalises whom merit favours)")

    w = 0.36
    ax2.barh(yy - w/2, df["levelling_down_posthoc"], w, color="#D1495B",
             label="post-hoc parity")
    ax2.barh(yy + w/2, df["dglc_change"], w, color="#2E6F40", label="DGLC (ours)")
    ax2.axvline(0, color="grey", lw=1)
    ax2.set_yticks(yy); ax2.set_yticklabels(df["dataset"], fontsize=8)
    ax2.set_xlabel("change in women's selection vs baseline")
    ax2.set_title("Post-hoc levels women DOWN; DGLC rewards merit")
    ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{CROSS}/cross_dataset.png", dpi=C.PLOT_DPI)
    plt.close(fig)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n==== CROSS-DATASET SUMMARY ====")
    print(show.round(3).to_string(index=False))
    print(f"\nsaved -> {CROSS}/  (cross_dataset_summary.csv/.tex/.json, cross_dataset.png)")


if __name__ == "__main__":
    main()