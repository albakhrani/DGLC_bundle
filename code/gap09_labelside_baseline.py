"""
Idea 1 - Stage 9: label-side / data-centric baselines for DGLC (AMEO).

DGLC is a label-side remedy; this stage benchmarks it against other members of
its own family (data-centric / label-modifying preprocessing) on an IDENTICAL
footing: the same merit covariates, the same gap_core out-of-fold protocol, the
same StratifiedKFold(5, shuffle, random_state=SEED), the same XGBoost classifier
(n_jobs=1), evaluated on AMEO. We also include the two model-side mitigations
(post-hoc equalized-odds and in-processing ExponentiatedGradient) so the table is
complete.

Baselines added (no new heavy dependencies; built on numpy + xgboost + fairlearn):
  * Reweighing (Kamiran & Calders, 2012): instance weights
        w(s,y) = P(s) P(y) / P(s,y)
    passed as sample_weight; data-centric, targets demographic parity.
  * Massaging (Kamiran & Calders, 2012): relabels borderline cases - promotes the
    highest-scored disadvantaged negatives and demotes the lowest-scored
    advantaged positives until group base rates equalize; label-side.

Every method is audited against the REALIZED label on common ground (selection
rates, realized-outcome AUROC where scores exist, equalized-odds gap vs realized,
and the levelling direction relative to the no-correction baseline). DGLC's
self-validation (Oaxaca unexplained -> 0) is reported alongside; the other
methods never touch the continuous wage structure, so their Oaxaca residual is
unchanged by construction.

Outputs:
  outputs/gap/metrics/labelside_baseline.json
  outputs/gap/tables/labelside_headtohead.csv
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from fairlearn.metrics import (demographic_parity_difference,
    equalized_odds_difference, selection_rate, MetricFrame)

import config as C
from utils import save_json
import gap_core as core
import gap_datasets as D

C.set_seed()
SEED = core.SEED


# --------------------------------------------------------------------------- #
# OOF helpers that reuse gap_core's exact fold protocol
# --------------------------------------------------------------------------- #
def _folds(y):
    return StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y)

def oof_proba_weighted(Xv, y, w):
    """Out-of-fold predict_proba with per-sample weights, gap_core fold protocol."""
    proba = np.zeros(len(y))
    for tr, va in _folds(y):
        clf = core._clf()
        clf.fit(Xv[tr], y[tr], sample_weight=w[tr])
        proba[va] = clf.predict_proba(Xv[va])[:, 1]
    return proba

def oof_eg(Xv, y, g):
    """Out-of-fold ExponentiatedGradient (EqualizedOdds) labels + ensemble scores."""
    from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
    pred = np.zeros(len(y), int)
    score = np.full(len(y), np.nan)
    for tr, va in _folds(y):
        eg = ExponentiatedGradient(estimator=core._clf(),
                                   constraints=EqualizedOdds())
        eg.fit(Xv[tr], y[tr], sensitive_features=g[tr])
        try:
            p = np.asarray(eg._pmf_predict(Xv[va]))[:, 1]
            score[va] = p
            pred[va] = (p >= 0.5).astype(int)
        except Exception:
            pred[va] = np.asarray(eg.predict(Xv[va])).astype(int)
    return pred, score


# --------------------------------------------------------------------------- #
# Metric block for one method (audited against the realized label)
# --------------------------------------------------------------------------- #
def metrics(name, pred, score, realized, g, oth, ref, base_sel_oth):
    sf, sr = core._selrate(pred, g, ref, oth)            # (women, men)
    auroc = float(roc_auc_score(realized, score)) if score is not None and not np.all(np.isnan(score)) else None
    eo = float(equalized_odds_difference(realized, pred, sensitive_features=g))
    dp = float(demographic_parity_difference(realized, pred, sensitive_features=g))
    lift = sf - base_sel_oth
    return dict(method=name, sel_women=round(sf, 5), sel_men=round(sr, 5),
                sel_gap_M_minus_F=round(sr - sf, 5),
                women_lift_vs_baseline=round(lift, 5),
                levelling=("up" if lift > 1e-9 else "down" if lift < -1e-9 else "flat"),
                auroc_realized=(round(auroc, 5) if auroc is not None else None),
                eo_diff_vs_realized=round(eo, 5),
                dp_diff_vs_realized=round(dp, 5))


def main():
    spec = D.load_ameo()
    df, outcome, group, ref, oth, num, cat = (spec["df"], spec["outcome"],
        spec["group"], spec["ref"], spec["oth"], spec["num"], spec["cat"])
    Xc = core.merit_features(df, num, cat)
    Xv = Xc.values
    g = df[group].values
    is_oth = (g == oth); is_ref = (g == ref)
    y_cont = df[outcome].astype(float).values
    realized = (y_cont > np.median(y_cont)).astype(int)

    rows = []

    # ---- (0) no correction (== DGLC regime A baseline) --------------------
    p_base = core.oof_proba(Xc, realized)
    pred_base = (p_base >= 0.5).astype(int)
    base_sel_oth, _ = core._selrate(pred_base, g, ref, oth)
    rows.append(metrics("no_correction", pred_base, p_base, realized, g, oth, ref, base_sel_oth))

    # ---- (1) post-hoc equalized-odds (model-side; == DGLC regime B) -------
    ttpr = float(((pred_base == 1) & (realized == 1)).sum() / max((realized == 1).sum(), 1))
    pred_post = core._eo_postproc(p_base, realized, g, ttpr)
    rows.append(metrics("posthoc_eq_odds", pred_post, None, realized, g, oth, ref, base_sel_oth))

    # ---- (2) ExponentiatedGradient (model-side, in-processing) ------------
    try:
        pred_eg, score_eg = oof_eg(Xv, realized, g)
        rows.append(metrics("exp_gradient", pred_eg, score_eg, realized, g, oth, ref, base_sel_oth))
    except Exception as e:
        rows.append(dict(method="exp_gradient", error=f"{type(e).__name__}: {e}"))

    # ---- (3) Reweighing (Kamiran-Calders; data-centric) -------------------
    N = len(realized)
    w = np.ones(N, float)
    for s_val in (oth, ref):
        for y_val in (0, 1):
            m = (g == s_val) & (realized == y_val)
            n_sy = m.sum()
            if n_sy:
                n_s = (g == s_val).sum(); n_y = (realized == y_val).sum()
                w[m] = (n_s * n_y) / (N * n_sy)
    p_rw = oof_proba_weighted(Xv, realized, w)
    pred_rw = (p_rw >= 0.5).astype(int)
    rows.append(metrics("reweighing", pred_rw, p_rw, realized, g, oth, ref, base_sel_oth))

    # ---- (4) Massaging (Kamiran-Calders; label-side relabelling) ----------
    P_oth = int(realized[is_oth].sum()); P_ref = int(realized[is_ref].sum())
    N_oth = int(is_oth.sum()); N_ref = int(is_ref.sum())
    M = int(round((P_ref * N_oth - P_oth * N_ref) / (N_oth + N_ref)))
    M = max(M, 0)
    massaged = realized.copy()
    # promote highest-scored disadvantaged (women) negatives 0 -> 1
    cand_promote = np.where(is_oth & (realized == 0))[0]
    cand_promote = cand_promote[np.argsort(-p_base[cand_promote])]
    promote = cand_promote[:min(M, len(cand_promote))]
    # demote lowest-scored advantaged (men) positives 1 -> 0
    cand_demote = np.where(is_ref & (realized == 1))[0]
    cand_demote = cand_demote[np.argsort(p_base[cand_demote])]
    demote = cand_demote[:min(M, len(cand_demote))]
    massaged[promote] = 1; massaged[demote] = 0
    p_ms = core.oof_proba(Xc, massaged)
    pred_ms = (p_ms >= 0.5).astype(int)
    rows.append(metrics("massaging", pred_ms, p_ms, realized, g, oth, ref, base_sel_oth))

    # ---- (5) DGLC (label-side, decomposition-grounded) --------------------
    dg = core.run_dglc(df, outcome, group, ref, oth, num, cat)
    selC = dg["sel_oth"]["C_dglc"]; selC_m = dg["sel_ref"]["C_dglc"]
    # audit DGLC predictions against the realized label for apples-to-apples:
    corr = core.corrected_outcome(df, outcome, group, ref, oth, num, cat, "neumark", 1.0)
    labC = (corr > np.median(corr)).astype(int)
    pC = core.oof_proba(Xc, labC)
    predC = (pC >= 0.5).astype(int)
    rows.append(metrics("dglc", predC, pC, realized, g, oth, ref, base_sel_oth))

    # ---- self-validation: Oaxaca unexplained residual after each method ----
    _, _, unexpl_before = core.oaxaca_twofold(df, outcome, group, ref, oth, num, cat)
    selfval = {
        "unexplained_before": round(float(unexpl_before), 5),
        "dglc_after": round(float(dg["self_validation"]["unexplained_after"]), 5),
        "reweighing_after": round(float(unexpl_before), 5),   # wage structure untouched
        "massaging_after": round(float(unexpl_before), 5),    # wage structure untouched
        "note": ("Reweighing and Massaging never modify the continuous wage outcome, "
                 "so re-decomposing it returns the original residual; only DGLC builds "
                 "a corrected continuous target whose residual collapses to ~0."),
    }

    massaging_meta = {"M_relabelled_each_side": M,
                      "promoted_women_0to1": int(len(promote)),
                      "demoted_men_1to0": int(len(demote))}

    out = {"dataset": spec["name"], "seed": SEED,
           "merit_covariates": {"numeric": num, "categorical": cat},
           "audit_label": "realized (above-median salary)",
           "methods": rows, "self_validation": selfval,
           "massaging_meta": massaging_meta,
           "dglc_merit_implied_rate": {"women": round(float(selC), 5),
                                       "men": round(float(selC_m), 5)}}
    save_json(out, f"{D.G.GAP_MET}/labelside_baseline.json")

    tab = pd.DataFrame([r for r in rows if "error" not in r])
    tab.to_csv(f"{D.G.GAP_TAB}/labelside_headtohead.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("==== LABEL-SIDE HEAD-TO-HEAD (AMEO, audited vs realized label) ====")
    print(tab.to_string(index=False))
    print("\nDGLC merit-implied target rate: women "
          f"{selC:.3f} vs men {selC_m:.3f}")
    print("\n==== self-validation (Oaxaca unexplained residual) ====")
    print(f"  before                 : {unexpl_before:+.5f}")
    print(f"  DGLC after             : {dg['self_validation']['unexplained_after']:+.2e}")
    print(f"  Reweighing/Massaging   : {unexpl_before:+.5f} (wage structure untouched)")
    print(f"\nMassaging relabelled M={M} per side "
          f"(promoted {len(promote)} women, demoted {len(demote)} men).")


if __name__ == "__main__":
    main()
