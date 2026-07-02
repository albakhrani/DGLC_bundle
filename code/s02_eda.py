"""
Stage 2 - Exploratory data analysis (descriptive, no modelling).

Outputs:
  outputs/figures/fig_eda_salary.png        salary distribution + threshold
  outputs/figures/fig_eda_corr.png          correlation heatmap (numeric)
  outputs/figures/fig_eda_gender_scores.png score & outcome gaps by gender
  outputs/tables/eda_group_summary.csv      group means by target & gender
  outputs/metrics/eda_gender_gaps.json      raw gender gaps + Mann-Whitney tests
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import config as C
from utils import set_plot_style, save_json
import data_module as D

C.set_seed()
set_plot_style()

KEY_SCORES = ["English", "Logical", "Quant", "Domain", "ComputerProgramming",
              "collegeGPA", "10percentage", "12percentage"]


def main():
    df = D.load_frame()
    aux = pd.read_parquet(C.PROC_PARQUET.replace("processed", "processed_aux"))

    # ---- salary distribution ----------------------------------------------
    plt.figure(figsize=(6.5, 4))
    sal = aux["Salary"].clip(upper=aux["Salary"].quantile(0.99))
    plt.hist(sal, bins=50, color="#4c78a8", alpha=0.85)
    plt.axvline(aux["Salary"].median(), color="red", ls="--",
                label=f"median = {aux['Salary'].median():,.0f}")
    plt.xlabel("Annual salary (INR, 99th-pct capped)"); plt.ylabel("Count")
    plt.title("Salary distribution and class threshold"); plt.legend()
    plt.savefig(f"{C.FIG_DIR}/fig_eda_salary.png"); plt.close()

    # ---- correlation heatmap ----------------------------------------------
    num = [c for c in KEY_SCORES + C.BIG_FIVE if c in df.columns]
    corr = df[num].corr()
    plt.figure(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, cbar_kws={"shrink": .8}, annot_kws={"size": 7})
    plt.title("Correlation among assessment & personality features")
    plt.savefig(f"{C.FIG_DIR}/fig_eda_corr.png"); plt.close()

    # ---- gender gaps -------------------------------------------------------
    gaps = {}
    g = df["Gender"]
    for c in KEY_SCORES:
        if c not in df.columns:
            continue
        m = df.loc[g == "male", c].dropna()
        f = df.loc[g == "female", c].dropna()
        u, p = stats.mannwhitneyu(m, f, alternative="two-sided")
        gaps[c] = {"male_mean": float(m.mean()), "female_mean": float(f.mean()),
                   "diff_m_minus_f": float(m.mean() - f.mean()),
                   "mannwhitney_p": float(p)}
    # outcome gap
    hs_m = df.loc[g == "male", C.TARGET].mean()
    hs_f = df.loc[g == "female", C.TARGET].mean()
    gaps["_high_salary_rate"] = {"male": float(hs_m), "female": float(hs_f),
                                 "ratio_f_over_m": float(hs_f / hs_m)}
    gaps["_female_share"] = float((g == "female").mean())
    save_json(gaps, f"{C.MET_DIR}/eda_gender_gaps.json")

    # plot gender score gaps + outcome
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    sub = df.melt(id_vars="Gender", value_vars=[c for c in KEY_SCORES[:5]],
                  var_name="feature", value_name="score")
    sns.boxplot(data=sub, x="feature", y="score", hue="Gender", ax=ax[0],
                showfliers=False)
    ax[0].set_title("Assessment scores by gender"); ax[0].tick_params(axis="x", rotation=20)
    ax[1].bar(["male", "female"], [hs_m, hs_f], color=["#4c78a8", "#e45756"])
    ax[1].set_ylabel("P(high salary)"); ax[1].set_title("High-salary rate by gender")
    for i, v in enumerate([hs_m, hs_f]):
        ax[1].text(i, v + 0.01, f"{v:.2f}", ha="center")
    fig.savefig(f"{C.FIG_DIR}/fig_eda_gender_scores.png"); plt.close(fig)

    # ---- group summary table ----------------------------------------------
    summary = df.groupby([C.TARGET])[num].mean().T
    summary.columns = ["low_salary", "high_salary"]
    summary.to_csv(f"{C.TAB_DIR}/eda_group_summary.csv")
    print("Female share:", round(gaps['_female_share'], 3))
    print("High-salary rate  male/female:", round(hs_m, 3), round(hs_f, 3),
          "ratio:", round(hs_f / hs_m, 3))
    print("EDA stage complete.")


if __name__ == "__main__":
    main()
