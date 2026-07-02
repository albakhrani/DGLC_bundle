"""
Stage 17 - Decision-curve analysis (net benefit).

A model can have an acceptable AUROC yet add little practical value at the
thresholds a decision-maker would use. Decision-curve analysis plots the net
benefit of acting on the model against the two default strategies, treat-all
(select every candidate) and treat-none (select no one), across a range of
threshold probabilities.

Net benefit at threshold p_t:  NB = TP/n - (FP/n) * p_t/(1-p_t).

Outputs:
  outputs/figures/fig_decision_curve.png
  outputs/tables/decision_curve.csv
"""
from __future__ import annotations
import glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

import config as C
from utils import set_plot_style
import data_module as D

C.set_seed()
set_plot_style()


def net_benefit(y, p, pt):
    yhat = (p >= pt).astype(int)
    n = len(y)
    tp = np.sum((yhat == 1) & (y == 1))
    fp = np.sum((yhat == 1) & (y == 0))
    return tp / n - (fp / n) * (pt / (1 - pt))


def main():
    df = D.load_frame()
    X, y, num, cat = D.get_Xy(df)
    sp = D.get_splits(df); te = np.array(sp["test"])
    pipe = joblib.load(glob.glob(f"{C.MODEL_DIR}/system1_best_*.joblib")[0])
    prob = pipe.predict_proba(X.iloc[te])[:, 1]
    yte = y[te]
    prev = yte.mean()

    pts = np.arange(0.20, 0.71, 0.02)
    nb_model = [net_benefit(yte, prob, t) for t in pts]
    nb_all = [prev - (1 - prev) * (t / (1 - t)) for t in pts]   # treat all
    nb_none = [0.0 for _ in pts]                                # treat none

    tbl = pd.DataFrame({"threshold": np.round(pts, 2),
                        "NB_model": np.round(nb_model, 4),
                        "NB_treat_all": np.round(nb_all, 4),
                        "NB_treat_none": nb_none})
    tbl.to_csv(f"{C.TAB_DIR}/decision_curve.csv", index=False)

    plt.figure(figsize=(6.2, 4.4))
    plt.plot(pts, nb_model, "o-", color="#1f77b4", lw=1.8, ms=4, label="deployed model")
    plt.plot(pts, nb_all, "--", color="#888", lw=1.2, label="treat all")
    plt.plot(pts, nb_none, ":", color="k", lw=1.0, label="treat none")
    plt.xlabel("Threshold probability"); plt.ylabel("Net benefit")
    plt.title("Decision-curve analysis (held-out test)")
    plt.legend(fontsize=8); plt.ylim(min(0, min(nb_all)) - 0.02, max(nb_model) + 0.05)
    plt.savefig(f"{C.FIG_DIR}/fig_decision_curve.png", bbox_inches="tight", dpi=300)
    plt.close()

    # range where the model beats both defaults
    beats = [round(t, 2) for t, m, a in zip(pts, nb_model, nb_all) if m > a and m > 0]
    print("Net benefit (model) at 0.3/0.4/0.5:",
          [round(net_benefit(yte, prob, t), 3) for t in (0.3, 0.4, 0.5)])
    print("treat-all at 0.3/0.4/0.5:",
          [round(prev - (1 - prev) * (t / (1 - t)), 3) for t in (0.3, 0.4, 0.5)])
    print(f"Model beats both defaults across thresholds ~{min(beats) if beats else 'n/a'}"
          f"-{max(beats) if beats else 'n/a'}")
    print("Stage 17 complete.")


if __name__ == "__main__":
    main()
