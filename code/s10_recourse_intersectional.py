"""
Stage 10 - Enhanced counterfactual recourse + intersectional fairness.

(1) Recourse: a formal actionability specification (mutable vs immutable features,
    allowed range, cost, feasibility rule) and worked easy / moderate / infeasible
    cases for the deployed model.
(2) Intersectional fairness: subgroup metrics for gender x college-tier,
    gender x degree, and gender x specialization, computed on out-of-fold
    predictions (larger subgroups than a single test split allows).

Outputs:
  outputs/tables/recourse_actionability.csv
  outputs/tables/recourse_examples.csv
  outputs/tables/intersectional_fairness.csv
  outputs/metrics/intersectional_fairness.json
  outputs/figures/fig_intersectional.png
"""
from __future__ import annotations
import glob, os, json, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, StratifiedKFold

import config as C
from utils import set_plot_style, save_json
import data_module as D

C.set_seed()
set_plot_style()

# Actionability taxonomy (documented in the manuscript)
MUTABLE   = ["English", "Logical", "Quant", "Domain", "ComputerProgramming"]
SEMI      = ["collegeGPA"]
IMMUTABLE = ["Gender", "GraduationYear", "CollegeTier", "CollegeState",
             "Specialization", "Degree", "10percentage", "12percentage",
             "age_at_grad"]


def actionability_table(df):
    rows = []
    stds = df[MUTABLE + SEMI].std()
    for f in MUTABLE:
        rows.append({"feature": f, "class": "mutable",
                     "rationale": "AMCAT section re-assessable through study/practice",
                     "allowed_change": "increase only, <= +2.5 SD",
                     "unit_SD": round(float(stds[f]), 2)})
    for f in SEMI:
        rows.append({"feature": f, "class": "semi-mutable",
                     "rationale": "historical GPA fixed; proxies trainable skill",
                     "allowed_change": "increase only, <= +1.5 SD",
                     "unit_SD": round(float(stds[f]), 2)})
    for f in IMMUTABLE:
        rows.append({"feature": f, "class": "immutable",
                     "rationale": "protected, historical, or institutional",
                     "allowed_change": "no change", "unit_SD": np.nan})
    t = pd.DataFrame(rows)
    t.to_csv(f"{C.TAB_DIR}/recourse_actionability.csv", index=False)
    return t


def recourse_cost(pipe, x0, df, budget=3.0):
    """Minimal-cost increase over mutable features (cost = sum standardized shift).
    Returns the change description, cost, and resulting probability, or None."""
    act = [c for c in MUTABLE + SEMI if c in x0.columns]
    stds = df[act].std()
    caps = {f: (2.5 if f in MUTABLE else 1.5) for f in act}
    best = None
    for f in act:
        for k in np.arange(0.25, caps[f] + 0.01, 0.25):
            xc = x0.copy(); xc[f] = xc[f] + k * stds[f]
            p = float(pipe.predict_proba(xc)[:, 1][0])
            if p >= 0.5 and (best is None or k < best["cost"]):
                best = {"cost": float(k), "desc": f"+{k:.2f} SD {f}", "p_new": p}
    if best is None or best["cost"] > 1.5:
        for f1, f2 in itertools.combinations(act, 2):
            for k in np.arange(0.25, 1.51, 0.25):
                if k > caps[f1] or k > caps[f2]:
                    continue
                xc = x0.copy()
                xc[f1] = xc[f1] + k * stds[f1]; xc[f2] = xc[f2] + k * stds[f2]
                p = float(pipe.predict_proba(xc)[:, 1][0])
                if p >= 0.5 and (best is None or 2 * k < best["cost"]):
                    best = {"cost": float(2 * k),
                            "desc": f"+{k:.2f} SD {f1} & {f2}", "p_new": p}
    return best


def recourse_examples(pipe, df, Xte):
    prob = pipe.predict_proba(Xte)[:, 1]
    rows = []
    # pick a spread of denied cases across probability bands
    for label, lo, hi in [("near-boundary", 0.40, 0.49),
                          ("moderate", 0.25, 0.40),
                          ("far", 0.0, 0.20)]:
        cand = np.where((prob >= lo) & (prob < hi))[0]
        if len(cand) == 0:
            continue
        i = int(cand[0])
        rec = recourse_cost(pipe, Xte.iloc[[i]], df)
        if rec is None:
            cat = "easy" if False else "infeasible"
            rows.append({"band": label, "p_orig": round(float(prob[i]), 3),
                         "recommended_change": "no change within budget",
                         "cost_SD": np.nan, "p_after": np.nan,
                         "feasible": "no (within 3 SD)", "difficulty": "infeasible",
                         "interpretation": "no feasible recourse found within the "
                         "predefined 3-SD budget"})
        else:
            cat = "easy" if rec["cost"] <= 1.0 else "moderate"
            interp = ("small, single-feature improvement (about a coaching cycle)"
                      if cat == "easy" else
                      "moderate, multi-feature improvement over a term")
            rows.append({"band": label, "p_orig": round(float(prob[i]), 3),
                         "recommended_change": rec["desc"],
                         "cost_SD": round(rec["cost"], 2),
                         "p_after": round(rec["p_new"], 3),
                         "feasible": "yes", "difficulty": cat,
                         "interpretation": interp})
    t = pd.DataFrame(rows)[["band", "p_orig", "recommended_change", "cost_SD",
                            "p_after", "feasible", "difficulty", "interpretation"]]
    t.to_csv(f"{C.TAB_DIR}/recourse_examples.csv", index=False)
    return t


def intersectional(df, oof):
    """Subgroup TPR / selection-rate gaps on out-of-fold predictions."""
    d = df.copy()
    d["yhat"] = (oof >= 0.5).astype(int)
    d["y"] = d[C.TARGET].values
    # broad specialization buckets to keep groups large enough
    top_spec = d["Specialization"].value_counts().head(4).index
    d["spec_grp"] = np.where(d["Specialization"].isin(top_spec), d["Specialization"], "other")
    axes = {"college tier": "CollegeTier", "degree": "Degree", "specialization": "spec_grp"}
    rows, summary = [], {}
    for axis_name, col in axes.items():
        gaps = []
        for lvl in d[col].dropna().unique():
            for g in ["male", "female"]:
                sub = d[(d[col] == lvl) & (d["Gender"] == g)]
                if len(sub) < 25:
                    continue
                tpr = (sub.loc[sub.y == 1, "yhat"]).mean() if (sub.y == 1).any() else np.nan
                rows.append({"axis": axis_name, "level": str(lvl), "gender": g,
                             "n": int(len(sub)),
                             "selection_rate": round(float(sub.yhat.mean()), 3),
                             "TPR": round(float(tpr), 3) if tpr == tpr else np.nan,
                             "base_rate_high": round(float(sub.y.mean()), 3)})
        # per-axis female-vs-male selection gap range
        sub = pd.DataFrame([r for r in rows if r["axis"] == axis_name])
        if not sub.empty:
            piv = sub.pivot_table(index="level", columns="gender", values="selection_rate")
            if {"male", "female"}.issubset(piv.columns):
                diff = (piv["male"] - piv["female"]).dropna()
                summary[axis_name] = {"max_sel_gap_m_minus_f": float(diff.max()),
                                      "min_sel_gap": float(diff.min()),
                                      "levels_assessed": int(len(diff))}
    t = pd.DataFrame(rows)
    t.to_csv(f"{C.TAB_DIR}/intersectional_fairness.csv", index=False)
    save_json(summary, f"{C.MET_DIR}/intersectional_fairness.json")
    # figure: selection rate by gender across tiers and degrees
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, axis_name in zip(ax, ["college tier", "degree"]):
        sub = t[t.axis == axis_name]
        piv = sub.pivot_table(index="level", columns="gender", values="selection_rate")
        piv.plot.bar(ax=a, rot=0, color={"male": "#4c78a8", "female": "#e45756"})
        a.set_title(f"Selection rate by gender x {axis_name}")
        a.set_ylabel("P(predicted high salary)")
    fig.savefig(f"{C.FIG_DIR}/fig_intersectional.png", bbox_inches="tight"); plt.close(fig)
    return t, summary


def main():
    df = D.load_frame()
    X, y, numeric, categorical = D.get_Xy(df)
    sp = D.get_splits(df); te = np.array(sp["test"])
    pipe = joblib.load(glob.glob(f"{C.MODEL_DIR}/system1_best_*.joblib")[0])

    print("Actionability specification ...")
    at = actionability_table(df); print(at.to_string(index=False))
    print("\nRecourse examples ...")
    ex = recourse_examples(pipe, df, X.iloc[te]); print(ex.to_string(index=False))

    print("\nOut-of-fold predictions for intersectional analysis ...")
    cv = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
    oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    t, summary = intersectional(df, oof)
    print("\nIntersectional subgroup metrics:\n", t.to_string(index=False))
    print("\nSelection-gap summary:", json.dumps(summary, indent=2))
    print("\nStage 10 complete.")


if __name__ == "__main__":
    main()
