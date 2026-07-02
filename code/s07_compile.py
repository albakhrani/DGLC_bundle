"""
Stage 7 - Compile all results into master tables and a headline figure.

Merges the System-1 panel with the DeBERTa transformer, produces the final
test-performance comparison (CSV + LaTeX), an AUROC-with-CI bar chart over all
models, and a single results_summary.json consumed by the manuscript.
"""
from __future__ import annotations
import json, glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as C
from utils import set_plot_style, classification_metrics, bootstrap_ci, df_to_latex

set_plot_style()


def main():
    test_df = pd.read_csv(f"{C.MET_DIR}/system1_test.csv", index_col=0)
    yte = np.load(f"{C.MET_DIR}/test_y_true.npy")

    # ---- fold in the deep-learning models (DeBERTa, FT-Transformer, TabNet) -
    DEEP = ["DeBERTa", "FT-Transformer", "TabNet", "TabTransformer"]
    for name in DEEP:
        p = f"{C.MET_DIR}/test_probs_{name}.npy"
        if os.path.exists(p):
            prob = np.load(p)
            met = classification_metrics(yte, prob)
            _, lo, hi = bootstrap_ci(yte, prob, "AUROC", seed=C.SEED)
            met.update({"AUROC_lo": lo, "AUROC_hi": hi})
            test_df = pd.concat([test_df, pd.DataFrame([met], index=[name])[test_df.columns]])

    test_df = test_df.sort_values("AUROC", ascending=False)
    test_df.to_csv(f"{C.MET_DIR}/master_test_comparison.csv")
    df_to_latex(test_df.round(3).reset_index().rename(columns={"index": "Model"}),
                f"{C.TAB_DIR}/master_test_comparison.tex",
                "Held-out test performance of all seventeen models: thirteen "
                "classical, ensemble, and glass-box classifiers (including the "
                "interpretable Explainable Boosting Machine, EBM) and four "
                "deep-learning baselines (DeBERTa table-to-text, FT-Transformer, "
                "TabNet, TabTransformer). Intervals are 95\\% bootstrap CIs.",
                "tab:master")

    # ---- headline figure: AUROC with bootstrap CI -------------------------
    DEEP = ["DeBERTa", "FT-Transformer", "TabNet", "TabTransformer"]
    d = test_df.sort_values("AUROC")
    y = np.arange(len(d))
    err = np.vstack([d["AUROC"] - d["AUROC_lo"], d["AUROC_hi"] - d["AUROC"]])
    colors = ["#d62728" if m in DEEP else "#1f77b4" for m in d.index]
    plt.figure(figsize=(7.5, 5))
    plt.barh(y, d["AUROC"], xerr=err, color=colors, alpha=0.85,
             error_kw={"elinewidth": 1.1, "capsize": 3})
    plt.yticks(y, d.index)
    plt.xlim(0.45, max(0.83, d["AUROC"].max() + 0.03))
    plt.axvline(0.5, color="k", ls=":", lw=0.8)
    plt.xlabel("Test AUROC (95% bootstrap CI)")
    plt.title("Model comparison - high-salary prediction (AMEO 2015)")
    plt.savefig(f"{C.FIG_DIR}/fig_master_auroc.png", bbox_inches="tight"); plt.close()

    # ---- results summary for the manuscript -------------------------------
    cv = pd.read_csv(f"{C.MET_DIR}/system1_cv.csv", index_col=0)
    with open(f"{C.MET_DIR}/system1_significance.json") as f:
        sig = json.load(f)
    with open(f"{C.MET_DIR}/system2_fairness.json") as f:
        fair = json.load(f)
    with open(f"{C.MET_DIR}/data_prep_report.json") as f:
        prep = json.load(f)

    summary = {
        "n_samples": prep["n_model_rows"],
        "n_features_model": prep["n_model_cols"],
        "salary_median_INR": prep["salary_median_INR"],
        "target_balance": prep["target_balance"],
        "best_model_cv": sig["best_model"],
        "best_model_test": test_df.index[0],
        "best_test_AUROC": float(test_df["AUROC"].iloc[0]),
        "best_test_AUROC_CI": [float(test_df["AUROC_lo"].iloc[0]),
                               float(test_df["AUROC_hi"].iloc[0])],
        "deberta_test_AUROC": (float(test_df.loc["DeBERTa", "AUROC"])
                               if "DeBERTa" in test_df.index else None),
        "friedman_p": sig["friedman"]["p"],
        "nemenyi_CD": sig["nemenyi_CD"],
        "fairness_unmitigated": fair["unmitigated"],
        "fairness_mitigated": fair["mitigated_equalized_odds"],
    }
    with open(f"{C.MET_DIR}/results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Master comparison:\n", test_df[["AUROC", "AUROC_lo", "AUROC_hi",
                                           "F1", "MCC", "Brier"]].round(3))
    print("\nresults_summary.json written.")


if __name__ == "__main__":
    main()
