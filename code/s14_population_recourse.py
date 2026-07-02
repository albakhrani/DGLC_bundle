"""
Stage 14 - Population-level recourse feasibility and equity of recourse.

Runs the EXISTING counterfactual search (s10.recourse_cost, unchanged) over every
rejected candidate and reports the distribution of feasibility and cost, broken
down by gender. This converts the three worked cases into a distributional,
threshold-independent fairness signal: near the boundary, do women obtain a
feasible, affordable recourse path as often as men?

No model is refit, retuned, or re-engineered. The deployed cross-validation-best
tree (RandomForest) and the cached split are reused. The decision threshold (0.5),
actionability taxonomy, cost definition, and 3-SD budget are exactly those of
Tables V-VI. "Not found within budget" is not a proof of impossibility.

Outputs (outputs/tables, outputs/figures, outputs/metrics):
  recourse_per_candidate.csv, recourse_summary_overall.csv,
  recourse_summary_by_gender.csv, recourse_summary_intersectional.csv
  fig_recourse_feasibility_by_gender.png, fig_recourse_cost_distribution.png
"""
from __future__ import annotations
import glob, re, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

import itertools
import config as C
from utils import set_plot_style, save_json
import data_module as D
from s10_recourse_intersectional import recourse_cost, MUTABLE, SEMI

C.set_seed()
set_plot_style()
RNG = np.random.default_rng(C.SEED)
THRESHOLD = 0.5


def recourse_cost_fast(pipe, x0, stds, caps, act):
    """Batched re-implementation of s10.recourse_cost: identical grid, costs, and
    min-cost tie-breaking, but all perturbations for one candidate are scored in a
    single predict_proba call. Verified to reproduce the worked cases exactly."""
    # single-feature candidates
    cand = []      # (cost, desc, row_dict)
    for f in act:
        for k in np.arange(0.25, caps[f] + 0.01, 0.25):
            cand.append((float(k), f"+{k:.2f} SD {f}", {f: k * stds[f]}))
    # pairwise candidates
    pair = []
    for f1, f2 in itertools.combinations(act, 2):
        for k in np.arange(0.25, 1.51, 0.25):
            if k > caps[f1] or k > caps[f2]:
                continue
            pair.append((float(2 * k), f"+{k:.2f} SD {f1} & {f2}",
                         {f1: k * stds[f1], f2: k * stds[f2]}))

    def score(batch):
        if not batch:
            return np.array([])
        X = pd.concat([x0] * len(batch), ignore_index=True)
        cols = set().union(*[d.keys() for _, _, d in batch])
        for c in cols:                              # avoid int/float dtype clash
            X[c] = X[c].astype(float)
        for j, (_, _, delta) in enumerate(batch):
            for col, add in delta.items():
                X.loc[j, col] = X.loc[j, col] + add
        return pipe.predict_proba(X)[:, 1]

    best = None
    ps = score(cand)
    for (cost, desc, _), p in zip(cand, ps):
        if p >= 0.5 and (best is None or cost < best["cost"]):
            best = {"cost": cost, "desc": desc, "p_new": float(p)}
    if best is None or best["cost"] > 1.5:
        pp = score(pair)
        for (cost, desc, _), p in zip(pair, pp):
            if p >= 0.5 and (best is None or cost < best["cost"]):
                best = {"cost": cost, "desc": desc, "p_new": float(p)}
    return best


def boot_ci(mask, n=2000):
    """Bootstrap 95% CI for a proportion (mask = booleans)."""
    m = np.asarray(mask, float)
    if len(m) == 0:
        return (np.nan, np.nan)
    stats = [m[RNG.integers(0, len(m), len(m))].mean() for _ in range(n)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def n_changed(desc):
    return 0 if desc is None else len(re.findall(r"SD\s+([A-Za-z]+)", desc.replace(" & ", " SD ")))


def run_population(pipe, X, df, idx, label):
    prob = pipe.predict_proba(X.iloc[idx])[:, 1]
    rejected = idx[prob < THRESHOLD]
    act = [c for c in MUTABLE + SEMI if c in X.columns]
    stds = df[act].std()
    caps = {f: (2.5 if f in MUTABLE else 1.5) for f in act}
    rows = []
    p_all = pipe.predict_proba(X.loc[rejected])[:, 1]
    for i, p0 in zip(rejected, p_all):
        x0 = X.loc[[i]].reset_index(drop=True)
        best = recourse_cost_fast(pipe, x0, stds, caps, act)
        feats = []
        if best is not None:
            feats = re.findall(r"SD\s+([A-Za-z]+)", best["desc"].replace(" & ", " SD "))
        rows.append({
            "idx": int(i), "population": label,
            "gender": df.loc[i, "Gender"],
            "degree": df.loc[i, "Degree"],
            "college_tier": df.loc[i, "CollegeTier"],
            "p_orig": round(p0, 4),
            "feasible": best is not None,
            "cost": best["cost"] if best else np.nan,
            "p_final": round(best["p_new"], 4) if best else np.nan,
            "n_features_changed": len(feats),
            "features_changed": "|".join(feats),
        })
    return pd.DataFrame(rows)


def summarize_gender(pc):
    out = []
    for g in ["male", "female"]:
        sub = pc[pc.gender == g]
        feas = sub.feasible.values
        lo, hi = boot_ci(feas)
        cost = sub.loc[sub.feasible, "cost"]
        out.append({"gender": g, "n_rejected": len(sub),
                    "feasibility_rate": round(feas.mean(), 3) if len(sub) else np.nan,
                    "feas_CI_lo": round(lo, 3), "feas_CI_hi": round(hi, 3),
                    "median_cost_SD": round(float(cost.median()), 2) if len(cost) else np.nan,
                    "IQR_cost_SD": round(float(cost.quantile(.75) - cost.quantile(.25)), 2)
                    if len(cost) else np.nan})
    return pd.DataFrame(out)


def main():
    df = D.load_frame()
    X, y, num, cat = D.get_Xy(df)
    sp = D.get_splits(df); te = np.array(sp["test"])
    pipe = joblib.load(glob.glob(f"{C.MODEL_DIR}/system1_best_*.joblib")[0])

    # ---- sanity: fast search reproduces the original (Table VI) exactly ---
    act = [c for c in MUTABLE + SEMI if c in X.columns]
    stds = df[act].std(); caps = {f: (2.5 if f in MUTABLE else 1.5) for f in act}
    prob_te = pipe.predict_proba(X.iloc[te])[:, 1]
    rej = te[prob_te < THRESHOLD]
    mism = 0
    for i in rej[:30]:
        a = recourse_cost(pipe, X.loc[[i]], df)
        b = recourse_cost_fast(pipe, X.loc[[i]].reset_index(drop=True), stds, caps, act)
        if (a is None) != (b is None) or (a and abs(a["cost"] - b["cost"]) > 1e-9):
            mism += 1
    print(f"Sanity check: {mism}/30 mismatches between original and fast search "
          f"(0 = exact reproduction).")

    # ---- primary: held-out test set --------------------------------------
    pc = run_population(pipe, X, df, te, "test")
    pc.to_csv(f"{C.TAB_DIR}/recourse_per_candidate.csv", index=False)

    feas = pc.feasible.values
    lo, hi = boot_ci(feas)
    cost = pc.loc[pc.feasible, "cost"]
    overall = {"population": "test", "n_rejected": int(len(pc)),
               "feasibility_rate": round(feas.mean(), 3),
               "feas_CI": [round(lo, 3), round(hi, 3)],
               "median_cost_SD": round(float(cost.median()), 2),
               "cost_IQR_SD": [round(float(cost.quantile(.25)), 2), round(float(cost.quantile(.75)), 2)]}
    pd.DataFrame([overall]).to_csv(f"{C.TAB_DIR}/recourse_summary_overall.csv", index=False)
    save_json(overall, f"{C.MET_DIR}/recourse_overall.json")

    by_g = summarize_gender(pc)
    by_g.to_csv(f"{C.TAB_DIR}/recourse_summary_by_gender.csv", index=False)
    from utils import df_to_latex
    pretty = by_g.rename(columns={"gender": "Gender", "n_rejected": "n rejected",
                                  "feasibility_rate": "feasible rate",
                                  "feas_CI_lo": "CI low", "feas_CI_hi": "CI high",
                                  "median_cost_SD": "median cost (SD)",
                                  "IQR_cost_SD": "IQR cost (SD)"})
    df_to_latex(pretty, f"{C.TAB_DIR}/recourse_by_gender.tex",
                f"Equity of recourse on the held-out test set: among {len(pc)} "
                "rejected candidates (recalibrated $p<0.5$), the rate of feasible "
                "recourse within the 3-SD budget (with a bootstrap 95\\% CI) and the "
                "median cost, by gender. ``Not found'' is not a proof of "
                "impossibility.", "tab:recourse_gender")

    # ---- indicative intersectional ---------------------------------------
    inter = []
    for axis in ["degree", "college_tier"]:
        for lvl in pc[axis].dropna().unique():
            for g in ["male", "female"]:
                sub = pc[(pc[axis] == lvl) & (pc.gender == g)]
                if len(sub) == 0:
                    continue
                c = sub.loc[sub.feasible, "cost"]
                inter.append({"axis": axis, "level": str(lvl), "gender": g,
                              "n": len(sub), "feasibility_rate": round(sub.feasible.mean(), 3),
                              "median_cost_SD": round(float(c.median()), 2) if len(c) else np.nan,
                              "indicative": len(sub) < 20})
    pd.DataFrame(inter).to_csv(f"{C.TAB_DIR}/recourse_summary_intersectional.csv", index=False)

    # ---- feature-change profile among feasible ---------------------------
    feat_counts = (pc.loc[pc.feasible, "features_changed"].str.split("|").explode()
                   .value_counts())

    # ---- robustness: out-of-fold rejected across full data ---------------
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    cv = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
    oofp = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    oof_idx = np.where(oofp < THRESHOLD)[0]
    pc_oof = run_population(pipe, X, df, oof_idx, "out-of-fold")
    by_g_oof = summarize_gender(pc_oof)

    # ---- figures ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.4, 4))
    colors = {"male": "#4c78a8", "female": "#e45756"}
    x = np.arange(2)
    rates = [by_g.loc[by_g.gender == g, "feasibility_rate"].values[0] for g in ["male", "female"]]
    err = [[rates[k] - by_g.loc[by_g.gender == g, "feas_CI_lo"].values[0] for k, g in enumerate(["male", "female"])],
           [by_g.loc[by_g.gender == g, "feas_CI_hi"].values[0] - rates[k] for k, g in enumerate(["male", "female"])]]
    ax.bar(x, rates, yerr=err, color=[colors["male"], colors["female"]], capsize=5, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(["male", "female"])
    ax.set_ylabel("Feasible-recourse rate among rejected"); ax.set_ylim(0, 1)
    for k, g in enumerate(["male", "female"]):
        n = by_g.loc[by_g.gender == g, "n_rejected"].values[0]
        ax.text(k, rates[k] + 0.03, f"{rates[k]:.2f}\n(n={n})", ha="center", fontsize=8)
    ax.set_title("Equity of recourse (held-out test)")
    fig.savefig(f"{C.FIG_DIR}/fig_recourse_feasibility_by_gender.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4))
    for g in ["male", "female"]:
        c = pc.loc[(pc.gender == g) & pc.feasible, "cost"].sort_values()
        if len(c):
            ax.plot(c.values, np.linspace(0, 1, len(c)), label=f"{g} (n={len(c)})", color=colors[g])
    ax.set_xlabel("Recourse cost (total standardized shift, SD)")
    ax.set_ylabel("Cumulative fraction of feasible cases")
    ax.set_title("Recourse-cost distribution among feasible cases"); ax.legend(fontsize=8)
    fig.savefig(f"{C.FIG_DIR}/fig_recourse_cost_distribution.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # ---- headline stdout --------------------------------------------------
    print(f"Threshold = {THRESHOLD} on deployed RandomForest probability.")
    print(f"\nOf {overall['n_rejected']} rejected candidates (test), "
          f"{overall['feasibility_rate']*100:.0f}% had feasible recourse within the "
          f"3-SD budget (95% CI {overall['feas_CI']}); median cost "
          f"{overall['median_cost_SD']} SD.")
    print("\nBy gender (test):\n", by_g.to_string(index=False))
    print("\nBy gender (out-of-fold robustness):\n", by_g_oof.to_string(index=False))
    print("\nMost-used mutable features among feasible cases:\n", feat_counts.head(6).to_string())
    print("\nStage 14 complete.")


if __name__ == "__main__":
    main()
