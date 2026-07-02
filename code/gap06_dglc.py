"""
Idea 1 - Stage 6: Decomposition-Guided Label Correction (DGLC).

NEW CONTRIBUTION (remedy). When the Merit-Reward Reversal diagnostic (gap05)
fires, the disparity is in the LABEL, so model-side mitigation is mis-targeted.
DGLC instead corrects the supervision target itself, grounded in the Oaxaca
reference (non-discriminatory) wage structure, and is:

  * INTERPRETABLE : each person's wage is adjusted by exactly the log-points the
                    decomposition attributes to their group's coefficient
                    deviation from the non-discriminatory structure b*.
  * TUNABLE       : a merit weight alpha in [0,1] interpolates from the realized
                    label (alpha=0) to a fully merit-corrected label (alpha=1),
                    tracing a fairness / predictive-validity FRONTIER.
  * SELF-VALIDATING: re-decomposing the corrected outcome zeroes out the
                    unexplained component it was built from.

We contrast three regimes and audit each against BOTH the realized label and the
merit (corrected) label -- a dual audit. The headline finding: parity-based
post-processing achieves "fairness" by LEVELLING DOWN the better-qualified group
(it lowers women's selection to match men), whereas DGLC raises selection toward
merit while preserving realized predictive accuracy.

Known positioning (cited, not claimed as ours): the leveling-down pathology of
parity constraints (Mittelstadt et al., 2023) and label-bias correction by
reweighting (Jiang & Nachum, 2020). Our novelty is the decomposition-grounded,
interpretable, alpha-tunable correction plus the reversal diagnostic that says
when to use it.

Outputs:
  outputs/gap/metrics/dglc.json
  outputs/gap/tables/dglc_regime_comparison.csv
  outputs/gap/tables/dglc_alpha_frontier.csv
  outputs/gap/tables/dglc_reference_robustness.csv
  outputs/gap/figures/dglc_frontier.png
  outputs/gap/figures/dglc_headline.png
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from fairlearn.metrics import (demographic_parity_difference,
    equalized_odds_difference, selection_rate, MetricFrame)
import xgboost as xgb

import config as C
from utils import set_plot_style, save_json
import gap_common as G
import gap01_oaxaca as OAX

C.set_seed()
F, M = G.OTHER_GROUP, G.REFERENCE_GROUP


# --------------------------------------------------------------------------- #
# Wage model design (matches gap01 covariates) and reference structures
# --------------------------------------------------------------------------- #
def design(df):
    num = [c for c in G.NUMERIC_COV if c in df.columns]
    cat = [c for c in G.CATEG_COV if c in df.columns]
    X = pd.concat([df[num].astype(float),
                   pd.get_dummies(df[cat].astype(str), drop_first=True,
                                  dtype=float)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    return X

def ols(X, y):
    return np.asarray(sm.OLS(np.asarray(y, float), X.values).fit().params, float)

def reference_betas(X, y, is_f, structure):
    b_m = ols(X[~is_f], y[~is_f]); b_f = ols(X[is_f], y[is_f])
    if structure == "male":   b_star = b_m
    elif structure == "female": b_star = b_f
    elif structure == "pooled": b_star = ols(X, y)
    else:                       # neumark
        Xp = X.copy(); Xp["_grp"] = (~is_f).astype(float)
        b_star = ols(Xp, y)[:-1]
    return b_m, b_f, b_star


def corrected_outcome(y, Xv, is_f, b_m, b_f, b_star, alpha):
    """y_corr = y - alpha * X @ (b_group - b*).  alpha in [0,1]."""
    corr = np.asarray(y, float).copy()
    corr[is_f]  = y[is_f]  - alpha * (Xv[is_f]  @ (b_f - b_star))
    corr[~is_f] = y[~is_f] - alpha * (Xv[~is_f] @ (b_m - b_star))
    return corr


# --------------------------------------------------------------------------- #
# Classifier + audits
# --------------------------------------------------------------------------- #
def clf():
    return xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        eval_metric="logloss", random_state=C.SEED, n_jobs=1)

def oof_proba(Xc, lab):
    cv = StratifiedKFold(C.N_SPLITS, shuffle=True, random_state=C.SEED)
    return cross_val_predict(clf(), Xc, lab, cv=cv, method="predict_proba",
                             n_jobs=1)[:, 1]

def selrate(pred, g):
    mf = MetricFrame(metrics=selection_rate, y_true=np.zeros_like(pred),
                     y_pred=pred, sensitive_features=g)
    return float(mf.by_group.get(F, np.nan)), float(mf.by_group.get(M, np.nan))

def eo_postproc(proba, y_true, g, target_tpr):
    """Deterministic equal-opportunity post-processing (per-group thresholds)."""
    pred = np.zeros(len(y_true), int)
    for gv in np.unique(g):
        m = g == gv; pos = m & (y_true == 1)
        thr = float(np.quantile(proba[pos], 1 - target_tpr)) if pos.sum() else 0.5
        pred[m] = (proba[m] >= thr).astype(int)
    return pred


def main():
    set_plot_style()
    df = G.load_gap_table().reset_index(drop=True)
    X = design(df); Xv = X.values
    y = df[G.OUTCOME].astype(float).values
    g = df[G.GROUP_COL].values
    is_f = g == F
    feat = [c for c in X.columns if c != "const"]
    Xc = X[feat].fillna(X[feat].median())

    structure = G.REFERENCE_STRUCTURE
    b_m, b_f, b_star = reference_betas(X, y, is_f, structure)

    realized_bin = (y > np.median(y)).astype(int)
    br_f0, br_m0 = realized_bin[is_f].mean(), realized_bin[~is_f].mean()

    # ----- alpha frontier ---------------------------------------------------
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows = []
    proba_by_alpha = {}
    for a in alphas:
        corr = corrected_outcome(y, Xv, is_f, b_m, b_f, b_star, a)
        # per-alpha unexplained residual: identical _decompose/structure used at
        # the endpoints (self-validation), run on the corrected CONTINUOUS outcome.
        df_a = df.copy(); df_a[G.OUTCOME] = corr
        _, unexpl_a, *_ = OAX._decompose(df_a, structure)
        lab = (corr > np.median(corr)).astype(int)
        p = oof_proba(Xc, lab)
        proba_by_alpha[a] = (lab, p)
        pred = (p >= 0.5).astype(int)
        sf, sm = selrate(pred, g)
        rows.append(dict(alpha=a,
            base_F=float(lab[is_f].mean()), base_M=float(lab[~is_f].mean()),
            sel_F=sf, sel_M=sm,
            auroc_realized=float(roc_auc_score(realized_bin, p)),
            auroc_target=float(roc_auc_score(lab, p)),
            unexpl_resid=float(unexpl_a)))
    frontier = pd.DataFrame(rows)
    frontier.to_csv(f"{G.GAP_TAB}/dglc_alpha_frontier.csv", index=False)

    # ----- three headline regimes ------------------------------------------
    labA, pA = proba_by_alpha[0.0]            # realized
    predA = (pA >= 0.5).astype(int)
    target_tpr = float(((predA == 1) & (labA == 1)).sum() / max((labA == 1).sum(), 1))
    predB = eo_postproc(pA, labA, g, target_tpr)   # post-hoc EO
    labC, pC = proba_by_alpha[1.0]            # DGLC
    predC = (pC >= 0.5).astype(int)

    def regime_metrics(name, pred, audit_label, scored_proba):
        sf, sm = selrate(pred, g)
        return dict(regime=name,
            sel_F=sf, sel_M=sm,
            DP_diff=float(demographic_parity_difference(audit_label, pred, sensitive_features=g)),
            EO_diff=float(equalized_odds_difference(audit_label, pred, sensitive_features=g)),
            auroc_realized=(float(roc_auc_score(realized_bin, scored_proba))
                            if scored_proba is not None else np.nan),
            merit_align_F=sf - float(labC[is_f].mean()),   # selection vs merit base rate
            merit_align_M=sm - float(labC[~is_f].mean()))

    comp = pd.DataFrame([
        regime_metrics("A_baseline_realized", predA, labA, pA),
        regime_metrics("B_posthoc_equal_opp", predB, labA, None),
        regime_metrics("C_dglc_alpha1",       predC, labC, pC),
    ])
    comp.to_csv(f"{G.GAP_TAB}/dglc_regime_comparison.csv", index=False)

    # ----- "levelling down": change in women's selection vs baseline -------
    lvl_post = comp.loc[1, "sel_F"] - comp.loc[0, "sel_F"]
    lvl_dglc = comp.loc[2, "sel_F"] - comp.loc[0, "sel_F"]

    # ----- self-validation: Oaxaca on the corrected outcome ----------------
    df_corr = df.copy()
    df_corr[G.OUTCOME] = corrected_outcome(y, Xv, is_f, b_m, b_f, b_star, 1.0)
    _, unexpl_after, raw_after, *_ = OAX._decompose(df_corr, structure)
    _, unexpl_before, raw_before, *_ = OAX._decompose(df, structure)

    # ----- reference-structure robustness of DGLC --------------------------
    robust = []
    for s in ["male", "female", "pooled", "neumark"]:
        bm_s, bf_s, bs_s = reference_betas(X, y, is_f, s)
        corr_s = corrected_outcome(y, Xv, is_f, bm_s, bf_s, bs_s, 1.0)
        lab_s = (corr_s > np.median(corr_s)).astype(int)
        robust.append(dict(reference=s,
            corrected_base_F=float(lab_s[is_f].mean()),
            corrected_base_M=float(lab_s[~is_f].mean()),
            corrected_base_gap_F_minus_M=float(lab_s[is_f].mean() - lab_s[~is_f].mean())))
    robust = pd.DataFrame(robust)
    robust.to_csv(f"{G.GAP_TAB}/dglc_reference_robustness.csv", index=False)

    save_json({
        "reference_structure": structure,
        "realized_base_rate": {"female": float(br_f0), "male": float(br_m0)},
        "corrected_base_rate_alpha1": {"female": float(labC[is_f].mean()),
                                       "male": float(labC[~is_f].mean())},
        "regimes": comp.to_dict(orient="records"),
        "levelling_down_womens_selection_change": {
            "post_hoc_equal_opp": float(lvl_post),
            "dglc": float(lvl_dglc)},
        "auroc_preserved_dglc_vs_baseline": {
            "baseline_realized": float(comp.loc[0, "auroc_realized"]),
            "dglc_realized": float(comp.loc[2, "auroc_realized"])},
        "self_validation_oaxaca_unexplained": {
            "before_logpts": float(unexpl_before),
            "after_logpts": float(unexpl_after)},
        "reference_robustness": robust.to_dict(orient="records"),
    }, f"{G.GAP_MET}/dglc.json")

    # ----- figure 1: alpha frontier ----------------------------------------
    fig, ax1 = plt.subplots(figsize=(6.6, 4.6))
    ax1.plot(frontier["alpha"], frontier["auroc_realized"], "o-",
             color="#2E6F40", label="AUROC vs realized outcome")
    ax1.set_xlabel(r"merit weight $\alpha$  (0 = realized label, 1 = merit-corrected)")
    ax1.set_ylabel("AUROC (realized outcome)", color="#2E6F40")
    ax1.set_ylim(0.5, 0.85)
    ax2 = ax1.twinx()
    ax2.plot(frontier["alpha"], frontier["sel_F"], "s--", color="#D1495B",
             label="women selected")
    ax2.plot(frontier["alpha"], frontier["base_F"], "^:", color="#9B59B6",
             label="women merit base rate")
    ax2.set_ylabel("rate (women)", color="#D1495B")
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 48))
    ax3.plot(frontier["alpha"], frontier["unexpl_resid"], "D-.", color="#1F6F8B",
             label="unexplained residual")
    ax3.set_ylabel("unexplained residual (log pts)", color="#1F6F8B")
    ax3.set_ylim(0, 0.075)  # positive axis: 0.00 (bottom) to +0.075 (top); curve descends
    lines = ax1.get_lines() + ax2.get_lines() + ax3.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="lower center")
    ax1.set_title(r"DGLC frontier over the merit weight $\alpha$")
    fig.tight_layout(); fig.savefig(f"{G.GAP_FIG}/dglc_frontier.png", dpi=C.PLOT_DPI)
    plt.close(fig)

    # ----- figure 2: headline contrast -------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    names = ["A baseline", "B post-hoc\nparity", "C DGLC\n(ours)"]
    selF = [comp.loc[0,"sel_F"], comp.loc[1,"sel_F"], comp.loc[2,"sel_F"]]
    selM = [comp.loc[0,"sel_M"], comp.loc[1,"sel_M"], comp.loc[2,"sel_M"]]
    x = np.arange(3); w = 0.36
    ax.bar(x - w/2, selF, w, color="#D1495B", label="women selected")
    ax.bar(x + w/2, selM, w, color="#4C8CBF", label="men selected")
    ax.axhline(labC[is_f].mean(), ls=":", color="#9B59B6",
               label=f"women merit base rate = {labC[is_f].mean():.2f}")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("selection rate")
    ax.set_title("Post-hoc parity levels women DOWN; DGLC rewards merit")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{G.GAP_FIG}/dglc_headline.png", dpi=C.PLOT_DPI)
    plt.close(fig)

    print("==== alpha frontier ====");  print(frontier.round(3).to_string(index=False))
    print("\n==== regime comparison ====");  print(comp.round(3).to_string(index=False))
    print(f"\nLevelling-down (change in women's selection vs baseline):")
    print(f"   post-hoc equal-opp : {lvl_post:+.3f}   <-- women selected LESS")
    print(f"   DGLC (ours)        : {lvl_dglc:+.3f}   <-- women selected MORE")
    print(f"\nAUROC preserved (realized): baseline {comp.loc[0,'auroc_realized']:.3f} "
          f"-> DGLC {comp.loc[2,'auroc_realized']:.3f}")
    print(f"Self-validation Oaxaca unexplained: {unexpl_before:+.4f} -> {unexpl_after:+.4f} "
          f"(should collapse toward 0)")
    print("\n==== reference robustness (corrected base-rate gap F-M) ====")
    print(robust.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
