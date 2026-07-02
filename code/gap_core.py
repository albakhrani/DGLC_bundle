"""
Idea 1 - dataset-agnostic core for the Merit-Reward Reversal diagnostic and
Decomposition-Guided Label Correction (DGLC).

All functions take an explicit (df, outcome, group, numeric_cov, categ_cov)
so the SAME math runs on AMEO, Campus Recruitment, and ACSIncome. This module
is the single source of truth; gap08_cross_dataset.py calls it per dataset.

Conventions
  * outcome   : continuous log-wage column already present in df.
  * group     : column with two string values; `ref` (advantaged) vs `oth`.
  * binary target for the diagnostic = above-median(outcome) earner.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from fairlearn.metrics import (demographic_parity_difference,
    equalized_odds_difference, selection_rate, MetricFrame)
import xgboost as xgb

SEED = 42


# --------------------------------------------------------------------------- #
# Design / OLS / reference structure
# --------------------------------------------------------------------------- #
def design(df, num, cat):
    num = [c for c in num if c in df.columns]
    cat = [c for c in cat if c in df.columns]
    X = df[num].astype(float).copy()
    if cat:
        X = pd.concat([X, pd.get_dummies(df[cat].astype(str), drop_first=True,
                                         dtype=float)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    return X

def _ols(X, y):
    return np.asarray(sm.OLS(np.asarray(y, float), X.values).fit().params, float)

def reference_betas(X, y, is_ref, structure):
    b_ref = _ols(X[is_ref], y[is_ref])
    b_oth = _ols(X[~is_ref], y[~is_ref])
    if structure == "ref":      b_star = b_ref
    elif structure == "oth":    b_star = b_oth
    elif structure == "pooled": b_star = _ols(X, y)
    else:                       # neumark: pooled incl. group dummy
        Xp = X.copy(); Xp["_grp"] = is_ref.astype(float)
        b_star = _ols(Xp, y)[:-1]
    return b_ref, b_oth, b_star


# --------------------------------------------------------------------------- #
# Oaxaca twofold (for triangulation + DGLC self-validation)
# --------------------------------------------------------------------------- #
def oaxaca_twofold(df, outcome, group, ref, oth, num, cat, structure="neumark"):
    X = design(df, num, cat); y = df[outcome].astype(float).values
    is_ref = (df[group].values == ref)
    b_ref, b_oth, b_star = reference_betas(X, y, is_ref, structure)
    mean_ref = X[is_ref].mean(axis=0).values
    mean_oth = X[~is_ref].mean(axis=0).values
    explained   = float((mean_ref - mean_oth) @ b_star)
    unexplained = float(mean_ref @ (b_ref - b_star) + mean_oth @ (b_star - b_oth))
    raw = float(y[is_ref].mean() - y[~is_ref].mean())
    return raw, explained, unexplained


# --------------------------------------------------------------------------- #
# DGLC: merit-counterfactual corrected outcome
# --------------------------------------------------------------------------- #
def corrected_outcome(df, outcome, group, ref, oth, num, cat, structure, alpha):
    X = design(df, num, cat); Xv = X.values
    y = df[outcome].astype(float).values
    is_ref = (df[group].values == ref)
    b_ref, b_oth, b_star = reference_betas(X, y, is_ref, structure)
    corr = y.copy()
    corr[is_ref]  = y[is_ref]  - alpha * (Xv[is_ref]  @ (b_ref - b_star))
    corr[~is_ref] = y[~is_ref] - alpha * (Xv[~is_ref] @ (b_oth - b_star))
    return corr


# --------------------------------------------------------------------------- #
# Merit classifier (gender withheld) + honest OOF predictions
# --------------------------------------------------------------------------- #
def _clf():
    return xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        eval_metric="logloss", random_state=SEED, n_jobs=1)

def merit_features(df, num, cat):
    return design(df, num, cat).drop(columns=["const"]).fillna(
        design(df, num, cat).drop(columns=["const"]).median())

def oof_proba(Xc, label, n_splits=5):
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
    return cross_val_predict(_clf(), Xc, label, cv=cv,
                             method="predict_proba", n_jobs=1)[:, 1]

def _selrate(pred, g, ref, oth):
    mf = MetricFrame(metrics=selection_rate, y_true=np.zeros_like(pred),
                     y_pred=pred, sensitive_features=g)
    return float(mf.by_group.get(oth, np.nan)), float(mf.by_group.get(ref, np.nan))


# --------------------------------------------------------------------------- #
# Merit-Reward Reversal diagnostic
# --------------------------------------------------------------------------- #
def run_mrr(df, outcome, group, ref, oth, num, cat, n_boot=2000):
    y_cont = df[outcome].astype(float).values
    target = (y_cont > np.median(y_cont)).astype(int)        # above-median earner
    Xc = merit_features(df, num, cat)
    g = df[group].values
    proba = oof_proba(Xc, target)
    pred = (proba >= 0.5).astype(int)

    is_oth = g == oth; is_ref = g == ref
    delta_label = target[is_oth].mean() - target[is_ref].mean()    # reward gap
    delta_merit = pred[is_oth].mean()  - pred[is_ref].mean()       # selection gap
    reversal = bool(delta_merit > 0 > delta_label)
    magnitude = delta_merit - delta_label

    rng = np.random.default_rng(SEED)
    io, ir = np.where(is_oth)[0], np.where(is_ref)[0]
    bl, bm, mag = [], [], []
    for _ in range(n_boot):
        so = rng.choice(io, len(io), True); sr = rng.choice(ir, len(ir), True)
        dl = target[so].mean() - target[sr].mean()
        dm = pred[so].mean()   - pred[sr].mean()
        bl.append(dl); bm.append(dm); mag.append(dm - dl)
    ci = lambda a: [float(np.quantile(a, .025)), float(np.quantile(a, .975))]
    raw, expl, unexpl = oaxaca_twofold(df, outcome, group, ref, oth, num, cat)
    return dict(n_oth=int(len(io)), n_ref=int(len(ir)),
        delta_label=float(delta_label), delta_label_ci=ci(bl),
        delta_merit=float(delta_merit), delta_merit_ci=ci(bm),
        reversal_magnitude=float(magnitude), reversal_magnitude_ci=ci(mag),
        merit_reward_reversal=reversal,
        oaxaca_unexplained=float(unexpl),
        triangulation_agree=bool(reversal and unexpl > 0))


# --------------------------------------------------------------------------- #
# DGLC: frontier + dual-audit regimes + self-validation + robustness
# --------------------------------------------------------------------------- #
def _eo_postproc(proba, y_true, g, target_tpr):
    pred = np.zeros(len(y_true), int)
    for gv in np.unique(g):
        m = g == gv; pos = m & (y_true == 1)
        thr = float(np.quantile(proba[pos], 1 - target_tpr)) if pos.sum() else 0.5
        pred[m] = (proba[m] >= thr).astype(int)
    return pred

def run_dglc(df, outcome, group, ref, oth, num, cat,
             structure="neumark", alphas=(0.0, 0.25, 0.5, 0.75, 1.0)):
    Xc = merit_features(df, num, cat)
    g = df[group].values
    is_oth = g == oth; is_ref = g == ref
    y = df[outcome].astype(float).values
    realized = (y > np.median(y)).astype(int)

    frontier, proba_by_alpha = [], {}
    for a in alphas:
        corr = corrected_outcome(df, outcome, group, ref, oth, num, cat, structure, a)
        lab = (corr > np.median(corr)).astype(int)
        p = oof_proba(Xc, lab); proba_by_alpha[a] = (lab, p)
        pred = (p >= 0.5).astype(int)
        sf, sm = _selrate(pred, g, ref, oth)
        frontier.append(dict(alpha=a, base_oth=float(lab[is_oth].mean()),
            sel_oth=sf, sel_ref=sm,
            auroc_realized=float(roc_auc_score(realized, p))))

    labA, pA = proba_by_alpha[alphas[0]]; predA = (pA >= 0.5).astype(int)
    ttpr = float(((predA == 1) & (labA == 1)).sum() / max((labA == 1).sum(), 1))
    predB = _eo_postproc(pA, labA, g, ttpr)
    labC, pC = proba_by_alpha[alphas[-1]]; predC = (pC >= 0.5).astype(int)

    def sel(pred): return _selrate(pred, g, ref, oth)
    selA, selB, selC = sel(predA), sel(predB), sel(predC)

    df_corr = df.copy()
    df_corr[outcome] = corrected_outcome(df, outcome, group, ref, oth, num, cat, structure, 1.0)
    _, _, unexpl_after  = oaxaca_twofold(df_corr, outcome, group, ref, oth, num, cat, structure)
    _, _, unexpl_before = oaxaca_twofold(df,      outcome, group, ref, oth, num, cat, structure)

    robust = {}
    for s in ["ref", "oth", "pooled", "neumark"]:
        cs = corrected_outcome(df, outcome, group, ref, oth, num, cat, s, 1.0)
        ls = (cs > np.median(cs)).astype(int)
        robust[s] = float(ls[is_oth].mean() - ls[is_ref].mean())

    return dict(
        frontier=frontier,
        sel_oth={"A_baseline": selA[0], "B_posthoc": selB[0], "C_dglc": selC[0]},
        sel_ref={"A_baseline": selA[1], "B_posthoc": selB[1], "C_dglc": selC[1]},
        levelling_down_posthoc=float(selB[0] - selA[0]),
        dglc_change=float(selC[0] - selA[0]),
        auroc_baseline=float(roc_auc_score(realized, pA)),
        auroc_dglc=float(roc_auc_score(realized, pC)),
        DP_diff={"A": float(demographic_parity_difference(labA, predA, sensitive_features=g)),
                 "C": float(demographic_parity_difference(labC, predC, sensitive_features=g))},
        self_validation={"unexplained_before": float(unexpl_before),
                         "unexplained_after": float(unexpl_after)},
        reference_robustness=robust)