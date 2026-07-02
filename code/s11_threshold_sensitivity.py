"""
Stage 11 - Threshold-sensitivity and fairness-metric confidence intervals.

(1) Sweeps the decision threshold (0.3, 0.4, 0.5, 0.6 and the Youden-J, F1, and
    balanced-accuracy optima) and reports accuracy together with the gender
    selection-rate gap, TPR gap, FPR gap, demographic-parity difference, and
    equalized-odds difference, demonstrating the fairness conclusion is not an
    artefact of the default 0.5 threshold.
(2) Bootstrap 95% CIs for the demographic-parity and equalized-odds differences
    at the default threshold, since the female test subgroup is small.

Outputs:
  outputs/tables/threshold_sensitivity.csv / .tex
  outputs/metrics/fairness_cis.json
  outputs/figures/fig_threshold.png
"""
from __future__ import annotations
import glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, f1_score, balanced_accuracy_score
from fairlearn.metrics import (demographic_parity_difference,
                               demographic_parity_ratio,
                               equalized_odds_difference)

import config as C
from utils import set_plot_style, save_json, df_to_latex
import data_module as D

C.set_seed()
set_plot_style()


def group_rates(y, yhat, g):
    out = {}
    for grp in ["male", "female"]:
        m = g == grp
        yt, yp = y[m], yhat[m]
        tpr = yp[yt == 1].mean() if (yt == 1).any() else np.nan
        fpr = yp[yt == 0].mean() if (yt == 0).any() else np.nan
        out[grp] = {"sel": yp.mean(), "tpr": tpr, "fpr": fpr}
    return out


def row_for(thr, y, prob, g, name):
    yhat = (prob >= thr).astype(int)
    r = group_rates(y, yhat, g)
    return {
        "threshold": name,
        "accuracy": round(accuracy_score(y, yhat), 3),
        "sel_rate": round(yhat.mean(), 3),
        "TPR_gap": round(r["male"]["tpr"] - r["female"]["tpr"], 3),
        "FPR_gap": round(r["male"]["fpr"] - r["female"]["fpr"], 3),
        "EO_diff": round(equalized_odds_difference(y, yhat, sensitive_features=g), 3),
        "DP_diff": round(demographic_parity_difference(y, yhat, sensitive_features=g), 3),
        "DI_ratio": round(demographic_parity_ratio(y, yhat, sensitive_features=g), 3),
    }


def bootstrap_fairness(y, prob, g, thr=0.5, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    dp, eo = [], []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.integers(0, len(y), len(y))
        yb, pb, gb = y[b], prob[b], g[b]
        if len(np.unique(yb)) < 2 or len(np.unique(gb)) < 2:
            continue
        yhat = (pb >= thr).astype(int)
        try:
            dp.append(demographic_parity_difference(yb, yhat, sensitive_features=gb))
            eo.append(equalized_odds_difference(yb, yhat, sensitive_features=gb))
        except Exception:
            continue
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"DP_diff_CI": ci(dp), "EO_diff_CI": ci(eo),
            "DP_diff_mean": float(np.mean(dp)), "EO_diff_mean": float(np.mean(eo))}


def main():
    df = D.load_frame()
    X, y, num, cat = D.get_Xy(df)
    g = df[C.SENSITIVE].values
    sp = D.get_splits(df); te = np.array(sp["test"])
    pipe = joblib.load(glob.glob(f"{C.MODEL_DIR}/system1_best_*.joblib")[0])
    prob = pipe.predict_proba(X.iloc[te])[:, 1]
    yte, gte = y[te], g[te]

    # data-driven thresholds (computed on the test ROC; reported as sensitivity)
    fpr, tpr, thr = roc_curve(yte, prob)
    youden = thr[np.argmax(tpr - fpr)]
    grid = np.linspace(0.05, 0.95, 91)
    f1opt = grid[np.argmax([f1_score(yte, prob >= t, zero_division=0) for t in grid])]
    baopt = grid[np.argmax([balanced_accuracy_score(yte, prob >= t) for t in grid])]

    rows = [row_for(t, yte, prob, gte, f"{t:.1f}") for t in (0.3, 0.4, 0.5, 0.6)]
    rows += [row_for(youden, yte, prob, gte, f"Youden ({youden:.2f})"),
             row_for(f1opt, yte, prob, gte, f"F1-opt ({f1opt:.2f})"),
             row_for(baopt, yte, prob, gte, f"BalAcc ({baopt:.2f})")]
    tbl = pd.DataFrame(rows)
    tbl.to_csv(f"{C.MET_DIR}/threshold_sensitivity.csv", index=False)
    df_to_latex(tbl, f"{C.TAB_DIR}/threshold_sensitivity.tex",
                "Fairness and accuracy across decision thresholds (held-out test). "
                "Gaps are male minus female. The gender gap persists at every "
                "threshold.", "tab:threshold")
    print(tbl.to_string(index=False))

    cis = bootstrap_fairness(yte, prob, gte)
    save_json(cis, f"{C.MET_DIR}/fairness_cis.json")
    print("\nFairness 95% CIs (threshold 0.5):", cis)

    # figure: metrics vs threshold
    gridt = np.linspace(0.25, 0.7, 19)
    acc = [accuracy_score(yte, prob >= t) for t in gridt]
    eo = [equalized_odds_difference(yte, (prob >= t).astype(int), sensitive_features=gte) for t in gridt]
    dp = [demographic_parity_difference(yte, (prob >= t).astype(int), sensitive_features=gte) for t in gridt]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(gridt, acc, "o-", label="accuracy", color="#4c78a8")
    ax.plot(gridt, eo, "s-", label="equalized-odds diff", color="#e45756")
    ax.plot(gridt, dp, "^-", label="demographic-parity diff", color="#54a24b")
    ax.axvline(0.5, ls=":", c="k", lw=0.8)
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Value")
    ax.set_title("Accuracy and fairness gaps vs.\\ threshold")
    ax.legend(fontsize=8)
    fig.savefig(f"{C.FIG_DIR}/fig_threshold.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("\nStage 11 complete.")


if __name__ == "__main__":
    main()
