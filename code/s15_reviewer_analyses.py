"""
Stage 15 - Reviewer-pre-empting analyses 2-4 (no model refit).

A2  Group-wise performance and calibration parity: per-gender AUROC (bootstrap CI),
    ECE, and Brier for the deployed model, to test whether it is equally skilled and
    equally calibrated across groups, not only equal in selection/TPR.
A3  Second fairness-mitigation method (in-processing ExponentiatedGradient with an
    EqualizedOdds constraint), to show the gap resists more than ThresholdOptimizer.
A4  Input-side group differences with effect sizes (Mann-Whitney U + Benjamini-
    Hochberg correction + rank-biserial effect size with bootstrap CI), to quantify
    the label-bias argument: are measured aptitudes comparable across genders while
    the salary label differs?

Outputs:
  outputs/tables/group_performance_calibration.csv/.tex
  outputs/tables/mitigation_comparison.csv/.tex
  outputs/tables/input_group_differences.csv/.tex
"""
from __future__ import annotations
import glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference

import config as C
from utils import save_json, df_to_latex
import data_module as D

C.set_seed()
RNG = np.random.default_rng(C.SEED)


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            e += abs(p[m].mean() - y[m].mean()) * m.sum() / len(y)
    return float(e)


def boot_metric(y, p, fn, n=2000):
    out = []
    for _ in range(n):
        b = RNG.integers(0, len(y), len(y))
        if len(np.unique(y[b])) < 2:
            continue
        out.append(fn(y[b], p[b]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    df = D.load_frame()
    X, y, numeric, categorical = D.get_Xy(df)
    g = df[C.SENSITIVE].values
    sp = D.get_splits(df); tr, te = np.array(sp["train"]), np.array(sp["test"])
    pipe = joblib.load(glob.glob(f"{C.MODEL_DIR}/system1_best_*.joblib")[0])
    prob = pipe.predict_proba(X.iloc[te])[:, 1]
    yte, gte = y[te], g[te]

    # ---- A2: group-wise AUROC / ECE / Brier ------------------------------
    rows = []
    for grp in ["male", "female"]:
        m = gte == grp
        auc = roc_auc_score(yte[m], prob[m])
        lo, hi = boot_metric(yte[m], prob[m], roc_auc_score)
        rows.append({"Group": grp, "n": int(m.sum()),
                     "AUROC": round(auc, 3), "AUROC_CI": f"[{lo:.3f}, {hi:.3f}]",
                     "Brier": round(brier_score_loss(yte[m], prob[m]), 3),
                     "ECE": round(ece(yte[m], prob[m]), 3)})
    a2 = pd.DataFrame(rows)
    a2.to_csv(f"{C.MET_DIR}/group_performance_calibration.csv".replace("metrics", "tables"), index=False)
    a2.to_csv(f"{C.TAB_DIR}/group_performance_calibration.csv", index=False)
    df_to_latex(a2, f"{C.TAB_DIR}/group_performance_calibration.tex",
                "Per-gender discrimination and calibration of the deployed model on "
                "the held-out test set (bootstrap 95\\% CIs).", "tab:groupperf")
    auc_gap = a2.loc[a2.Group == "male", "AUROC"].values[0] - a2.loc[a2.Group == "female", "AUROC"].values[0]
    print("A2 group performance:\n", a2.to_string(index=False), f"\n  AUROC gap (M-F) = {auc_gap:+.3f}")

    # ---- A3: second mitigation (in-processing ExponentiatedGradient) -----
    prep = pipe.named_steps["prep"]
    Xt_tr = prep.transform(X.iloc[tr]); Xt_te = prep.transform(X.iloc[te])
    base = LogisticRegression(max_iter=2000, class_weight="balanced")
    base.fit(Xt_tr, y[tr])
    p_base = base.predict_proba(Xt_te)[:, 1]; yhat_base = (p_base >= 0.5).astype(int)
    eg = ExponentiatedGradient(LogisticRegression(max_iter=2000), EqualizedOdds(), max_iter=50)
    eg.fit(Xt_tr, y[tr], sensitive_features=g[tr])
    yhat_eg = eg.predict(Xt_te, random_state=C.SEED)
    a3 = pd.DataFrame([
        {"Method": "baseline (LogReg)",
         "EO_diff": round(equalized_odds_difference(yte, yhat_base, sensitive_features=gte), 3),
         "DP_diff": round(demographic_parity_difference(yte, yhat_base, sensitive_features=gte), 3),
         "accuracy": round(accuracy_score(yte, yhat_base), 3)},
        {"Method": "ExponentiatedGradient (EO)",
         "EO_diff": round(equalized_odds_difference(yte, yhat_eg, sensitive_features=gte), 3),
         "DP_diff": round(demographic_parity_difference(yte, yhat_eg, sensitive_features=gte), 3),
         "accuracy": round(accuracy_score(yte, yhat_eg), 3)},
    ])
    a3.to_csv(f"{C.TAB_DIR}/mitigation_comparison.csv", index=False)
    df_to_latex(a3, f"{C.TAB_DIR}/mitigation_comparison.tex",
                "Second mitigation method: in-processing ExponentiatedGradient with an "
                "equalized-odds constraint, on the held-out test set, against an "
                "unmitigated baseline of the same model family.", "tab:mitig2")
    print("\nA3 second mitigation:\n", a3.to_string(index=False))

    # ---- A4: input group differences with effect sizes -------------------
    feats = ["English", "Logical", "Quant", "Domain", "ComputerProgramming", "collegeGPA"]
    rows, pvals = [], []
    for f in feats:
        a = df.loc[df[C.SENSITIVE] == "male", f].dropna()
        b = df.loc[df[C.SENSITIVE] == "female", f].dropna()
        U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rbc = 2 * U / (len(a) * len(b)) - 1            # rank-biserial effect size
        # bootstrap CI for rank-biserial
        bs = []
        for _ in range(2000):
            aa = a.values[RNG.integers(0, len(a), len(a))]
            bb = b.values[RNG.integers(0, len(b), len(b))]
            Ub, _ = stats.mannwhitneyu(aa, bb, alternative="two-sided")
            bs.append(2 * Ub / (len(aa) * len(bb)) - 1)
        rows.append({"feature": f, "median_M": round(float(a.median()), 1),
                     "median_F": round(float(b.median()), 1),
                     "rank_biserial": round(rbc, 3),
                     "effect_CI": f"[{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]",
                     "p_raw": p})
        pvals.append(p)
    # Benjamini-Hochberg correction
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        k = m - rank
        prev = min(prev, pvals[idx] * m / k); adj[idx] = prev
    a4 = pd.DataFrame(rows); a4["p_BH"] = np.round(adj, 4); a4 = a4.drop(columns="p_raw")
    a4["sig_BH"] = np.where(adj < 0.05, "yes", "no")
    a4.to_csv(f"{C.TAB_DIR}/input_group_differences.csv", index=False)
    df_to_latex(a4, f"{C.TAB_DIR}/input_group_differences.tex",
                "Input-side gender differences in assessment scores: medians, "
                "rank-biserial effect size (bootstrap 95\\% CI), and Benjamini-Hochberg "
                "corrected Mann-Whitney $p$-values.", "tab:inputdiff")
    print("\nA4 input group differences:\n", a4.to_string(index=False))

    save_json({"auc_gap_M_minus_F": auc_gap,
               "eo_baseline": float(a3.loc[0, "EO_diff"]),
               "eo_expgrad": float(a3.loc[1, "EO_diff"]),
               "max_abs_rank_biserial": float(a4["rank_biserial"].abs().max())},
              f"{C.MET_DIR}/reviewer_analyses.json")
    print("\nStage 15 complete.")


if __name__ == "__main__":
    main()
