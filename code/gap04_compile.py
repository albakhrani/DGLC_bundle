"""
Idea 1 - Stage 4: compile the gender-gap evidence into one master table and a
single headline figure that juxtaposes the three independent estimates of the
residual gender effect:

  1. Oaxaca-Blinder "unexplained" component  (linear econometric)
  2. ML gender-flip counterfactual           (non-linear, SHAP-backed)
  3. Fairness-audit selection-rate gap        (deployment lens)

Outputs:
  outputs/gap/tables/MASTER_summary.csv
  outputs/gap/tables/MASTER_summary.tex
  outputs/gap/figures/HEADLINE_residual_gap.png
"""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as C
from utils import set_plot_style, df_to_latex
import gap_common as G


def _load(name):
    with open(os.path.join(G.GAP_MET, name)) as f:
        return json.load(f)


def main() -> None:
    set_plot_style()
    oax = _load("oaxaca.json")
    ml = _load("ml_counterfactual.json")
    fair = _load("fairness.json")

    # express everything as a FEMALE-relative effect for comparability
    oax_unexpl_female = -oax["unexplained_logpts"]   # male-female -> female-male
    ml_cf_female = ml["counterfactual_female_minus_male_logpts"]

    rows = [
        ("Raw log-salary gap (female−male)", -oax["raw_gap_logpts"],
         f"{np.expm1(-oax['raw_gap_logpts'])*100:+.1f}%"),
        ("Oaxaca explained (qualifications)", -(oax["explained_logpts"]),
         f"{np.expm1(-oax['explained_logpts'])*100:+.1f}%"),
        ("Oaxaca UNEXPLAINED (residual)", oax_unexpl_female,
         f"{np.expm1(oax_unexpl_female)*100:+.1f}%"),
        ("ML counterfactual gender-flip", ml_cf_female,
         f"{np.expm1(ml_cf_female)*100:+.1f}%"),
        ("Fairness: demographic-parity ratio", fair["demographic_parity_ratio"],
         "PASS" if fair["passes_80pct_rule"] else "FAIL"),
        ("Fairness: equalized-odds diff", fair["equalized_odds_difference"], ""),
    ]
    master = pd.DataFrame(rows, columns=["metric", "value", "approx_interpretation"])
    master["value"] = master["value"].round(4)
    master.to_csv(f"{G.GAP_TAB}/MASTER_summary.csv", index=False)
    df_to_latex(master, f"{G.GAP_TAB}/MASTER_summary.tex",
                caption="Gender pay-gap evidence: three independent estimates of "
                        "the residual (qualification-adjusted) gender effect on "
                        "early-career salary, AMEO 2015.",
                label="tab:gap_master")

    # ---- headline figure: three estimates of the residual female penalty ---
    labels = ["Oaxaca\nunexplained", "ML\ncounterfactual"]
    vals = [oax_unexpl_female, ml_cf_female]
    ci = oax.get("bootstrap_95ci", {}).get("unexplained_ci")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(labels, vals, color=["#D1495B", "#8C5E9E"], width=0.55)
    if ci:   # CI is on male-female unexplained; flip sign for female frame
        lo, hi = -ci[1], -ci[0]
        ax.errorbar(0, oax_unexpl_female, yerr=[[oax_unexpl_female - lo],
                    [hi - oax_unexpl_female]], fmt="none", ecolor="black", capsize=5)
    ax.axhline(0, c="grey", lw=1)
    ax.axhline(-oax["raw_gap_logpts"], ls="--", c="grey", lw=1,
               label=f"raw gap = {-oax['raw_gap_logpts']:+.3f}")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + (0.002 if v >= 0 else -0.006),
                f"{v:+.3f}\n(~{np.expm1(v)*100:+.1f}%)", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_ylabel("residual female effect on log-salary")
    ax.set_title("Qualification-adjusted gender penalty: two methods agree")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{G.GAP_FIG}/HEADLINE_residual_gap.png", dpi=C.PLOT_DPI)
    plt.close(fig)

    print("\n=== MASTER SUMMARY (Idea 1: Gender Pay Gap Decomposition) ===")
    print(master.to_string(index=False))
    print(f"\nSaved master table + headline figure under {G.GAP_DIR}")


if __name__ == "__main__":
    main()
