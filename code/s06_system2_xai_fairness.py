"""
Stage 6 - System 2 (slow, deliberative reasoning layer).

Where System 1 returns a fast prediction, System 2 *deliberates* about it:
  (a) Attribution      - SHAP global + local feature attributions.
  (b) Counterfactuals  - minimal, actionable changes that would flip an
                         unfavourable decision (recourse).
  (c) Fairness audit   - group metrics + demographic-parity / equalised-odds
                         gaps across gender (fairlearn).
  (d) Mitigation       - post-hoc ThresholdOptimizer; reports the
                         fairness-accuracy trade-off.

Outputs:
  outputs/figures/fig_shap_global.png, fig_shap_local.png, fig_fairness.png
  outputs/tables/counterfactuals.csv, fairness_by_group.csv
  outputs/metrics/system2_fairness.json
"""
from __future__ import annotations
import glob, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.base import clone
from fairlearn.metrics import (MetricFrame, selection_rate, true_positive_rate,
                               false_positive_rate, count,
                               demographic_parity_difference,
                               demographic_parity_ratio,
                               equalized_odds_difference)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score

import config as C
from utils import set_plot_style, save_json
import data_module as D

C.set_seed()
set_plot_style()

ACTIONABLE = ["English", "Logical", "Quant", "ComputerProgramming",
              "Domain", "collegeGPA"]


def load_best_pipeline():
    paths = glob.glob(f"{C.MODEL_DIR}/system1_best_*.joblib")
    if not paths:
        raise FileNotFoundError("run s04_system1_models.py first")
    name = os.path.basename(paths[0]).replace("system1_best_", "").replace(".joblib", "")
    return joblib.load(paths[0]), name


# --------------------------------------------------------------------------- #
# (a) SHAP attribution
# --------------------------------------------------------------------------- #
def shap_analysis(pipe, Xte):
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    Xt = prep.transform(Xte)
    feats = list(prep.get_feature_names_out())
    Xs = Xt[:400]
    expl = shap.TreeExplainer(clf)
    sv = expl.shap_values(Xs)
    sv = np.asarray(sv)
    # normalise to (n, n_feat) for the positive class across shap versions
    if sv.ndim == 3:
        sv_pos = sv[:, :, 1] if sv.shape[2] == 2 else sv[:, :, -1]
    elif isinstance(expl.shap_values(Xs[:1]), list):
        sv_pos = sv[1]
    else:
        sv_pos = sv
    shap.summary_plot(sv_pos, Xs, feature_names=feats, show=False,
                      max_display=15, plot_size=(8, 6))
    plt.title("System-2 attribution: SHAP (positive class = high salary)")
    plt.savefig(f"{C.FIG_DIR}/fig_shap_global.png", bbox_inches="tight"); plt.close()

    # local: one favourable, one unfavourable case
    prob = clf.predict_proba(Xt)[:, 1]
    hi, lo = int(prob.argmax()), int(prob.argmin())
    base = expl.expected_value
    base = base[1] if np.ndim(base) else base
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, idx, ttl in [(axes[0], hi, "high-salary prediction"),
                         (axes[1], lo, "low-salary prediction")]:
        order = np.argsort(np.abs(sv_pos[idx % len(sv_pos)]))[::-1][:8]
        vals = sv_pos[idx % len(sv_pos)][order]
        ax.barh([feats[j] for j in order][::-1], vals[::-1],
                color=["#2b8a3e" if v > 0 else "#c92a2a" for v in vals[::-1]])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"Local SHAP - {ttl}", fontsize=10)
    fig.suptitle("System-2 local attributions", y=1.02)
    fig.savefig(f"{C.FIG_DIR}/fig_shap_local.png", bbox_inches="tight"); plt.close(fig)
    return feats


# --------------------------------------------------------------------------- #
# (b) Counterfactual recourse
# --------------------------------------------------------------------------- #
def counterfactuals(pipe, Xte, n_cases=6, step_grid=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0)):
    """
    Minimal-change actionable recourse to flip an unfavourable (0) decision.
    Searches single-feature then pairwise-feature increases in actionable
    assessment scores, returning the change with the smallest total effort
    (sum of standardised shifts) that reaches P>=0.5.
    """
    import itertools
    prob = pipe.predict_proba(Xte)[:, 1]
    denied = np.where(prob < 0.5)[0][:n_cases]
    act = [c for c in ACTIONABLE if c in Xte.columns]
    stds = Xte[act].std()
    rows = []
    for i in denied:
        x0 = Xte.iloc[[i]].copy()
        p0 = float(pipe.predict_proba(x0)[:, 1][0])
        best = None
        # single-feature
        for f in act:
            for k in step_grid:
                xc = x0.copy(); xc[f] = xc[f] + k * stds[f]
                pc = float(pipe.predict_proba(xc)[:, 1][0])
                if pc >= 0.5 and (best is None or k < best["effort"]):
                    best = {"desc": f"+{k:.1f}sd {f}", "effort": k, "p_new": pc}
        # pairwise (only if no cheap single-feature flip found)
        if best is None or best["effort"] > 1.5:
            for f1, f2 in itertools.combinations(act, 2):
                for k in (0.5, 1.0, 1.5):
                    xc = x0.copy()
                    xc[f1] = xc[f1] + k * stds[f1]
                    xc[f2] = xc[f2] + k * stds[f2]
                    pc = float(pipe.predict_proba(xc)[:, 1][0])
                    eff = 2 * k
                    if pc >= 0.5 and (best is None or eff < best["effort"]):
                        best = {"desc": f"+{k:.1f}sd {f1} & {f2}",
                                "effort": eff, "p_new": pc}
        rows.append({"case": int(i), "p_orig": round(p0, 3),
                     "recourse": best["desc"] if best else "no recourse <=3sd",
                     "total_effort_sd": round(best["effort"], 2) if best else np.nan,
                     "p_after": round(best["p_new"], 3) if best else np.nan})
    cf = pd.DataFrame(rows)
    cf.to_csv(f"{C.TAB_DIR}/counterfactuals.csv", index=False)
    return cf


# --------------------------------------------------------------------------- #
# (c) + (d) Fairness audit and mitigation
# --------------------------------------------------------------------------- #
def fairness(pipe, Xtr, ytr, Xte, yte, gtr, gte):
    prob = pipe.predict_proba(Xte)[:, 1]
    yhat = (prob >= 0.5).astype(int)

    mf = MetricFrame(
        metrics={"selection_rate": selection_rate, "TPR": true_positive_rate,
                 "FPR": false_positive_rate, "accuracy": accuracy_score,
                 "count": count},
        y_true=yte, y_pred=yhat, sensitive_features=gte)
    by_group = mf.by_group.copy()
    by_group.to_csv(f"{C.TAB_DIR}/fairness_by_group.csv")

    res = {
        "unmitigated": {
            "demographic_parity_diff": float(demographic_parity_difference(
                yte, yhat, sensitive_features=gte)),
            "disparate_impact_ratio": float(demographic_parity_ratio(
                yte, yhat, sensitive_features=gte)),
            "equalized_odds_diff": float(equalized_odds_difference(
                yte, yhat, sensitive_features=gte)),
            "accuracy": float(accuracy_score(yte, yhat)),
            "balanced_accuracy": float(balanced_accuracy_score(yte, yhat)),
        }
    }

    # ---- mitigation: ThresholdOptimizer (equalised odds) ------------------
    to = ThresholdOptimizer(estimator=pipe, constraints="equalized_odds",
                            objective="balanced_accuracy_score",
                            predict_method="predict_proba", prefit=True)
    to.fit(Xtr, ytr, sensitive_features=gtr)
    yhat_m = to.predict(Xte, sensitive_features=gte, random_state=C.SEED)
    res["mitigated_equalized_odds"] = {
        "demographic_parity_diff": float(demographic_parity_difference(
            yte, yhat_m, sensitive_features=gte)),
        "disparate_impact_ratio": float(demographic_parity_ratio(
            yte, yhat_m, sensitive_features=gte)),
        "equalized_odds_diff": float(equalized_odds_difference(
            yte, yhat_m, sensitive_features=gte)),
        "accuracy": float(accuracy_score(yte, yhat_m)),
        "balanced_accuracy": float(balanced_accuracy_score(yte, yhat_m)),
    }
    save_json(res, f"{C.MET_DIR}/system2_fairness.json")

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    by_group[["selection_rate", "TPR", "FPR"]].plot.bar(ax=ax[0], rot=0)
    ax[0].set_title("Group metrics by gender (unmitigated)")
    ax[0].set_ylabel("rate")
    labels = ["DP diff", "EO diff", "1 - DI ratio"]
    un = [res["unmitigated"]["demographic_parity_diff"],
          res["unmitigated"]["equalized_odds_diff"],
          1 - res["unmitigated"]["disparate_impact_ratio"]]
    mi = [res["mitigated_equalized_odds"]["demographic_parity_diff"],
          res["mitigated_equalized_odds"]["equalized_odds_diff"],
          1 - res["mitigated_equalized_odds"]["disparate_impact_ratio"]]
    x = np.arange(len(labels)); w = 0.36
    ax[1].bar(x - w/2, un, w, label="unmitigated", color="#e45756")
    ax[1].bar(x + w/2, mi, w, label="mitigated (EO)", color="#54a24b")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels)
    ax[1].set_title("Fairness gaps before/after mitigation (lower=fairer)")
    ax[1].legend(fontsize=8)
    fig.savefig(f"{C.FIG_DIR}/fig_fairness.png", bbox_inches="tight"); plt.close(fig)
    return res, by_group


def main():
    df = D.load_frame()
    X, y, numeric, categorical = D.get_Xy(df)
    g = df[C.SENSITIVE].values
    sp = D.get_splits(df)
    tr, te = np.array(sp["train"]), np.array(sp["test"])
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    ytr, yte = y[tr], y[te]
    gtr, gte = g[tr], g[te]

    pipe, name = load_best_pipeline()
    print("Deployed model (System-1 best):", name)

    print("SHAP attribution ...")
    shap_analysis(pipe, Xte)
    print("Counterfactual recourse ...")
    cf = counterfactuals(pipe, Xte)
    print(cf.to_string(index=False))
    print("Fairness audit + mitigation ...")
    res, by_group = fairness(pipe, Xtr, ytr, Xte, yte, gtr, gte)
    print("\nGroup metrics:\n", by_group.round(3))
    print("\nFairness summary:")
    for k, v in res.items():
        print(" ", k, {kk: round(vv, 3) for kk, vv in v.items()})
    print("\nSystem-2 stage complete.")


if __name__ == "__main__":
    main()
