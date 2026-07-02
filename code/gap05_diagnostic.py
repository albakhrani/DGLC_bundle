"""
Idea 1 - Stage 5: the Merit-Reward Reversal (MRR) diagnostic.

NEW CONTRIBUTION (diagnostic). A formal, bootstrapped test that classifies a
group fairness gap as LABEL-RESIDENT rather than MODEL-RESIDENT. The signature:

    a merit-only model SELECTS the protected group at a HIGHER rate than the
    realized label REWARDS them.

Operationally, with female (F) as the disadvantaged-by-label group:
    delta_label  = baserate(F) - baserate(M)      [realized high-salary label]
    delta_merit  = sel_rate(F) - sel_rate(M)       [merit-only classifier, gender withheld]
    REVERSAL holds  <=>  sign(delta_merit) != sign(delta_label)
    reversal magnitude  = delta_merit - delta_label   (how far the model and the
                                                        label disagree)

The test is reported with stratified bootstrap 95% CIs and triangulated against
the Oaxaca unexplained component (Stage 1) and the ceteris-paribus gender-flip
counterfactual (Stage 2). When all three agree, model-side mitigation is
mis-targeted and the appropriate remedy is on the LABEL (see gap06_dglc.py).

Outputs:
  outputs/gap/metrics/mrr_diagnostic.json
  outputs/gap/tables/mrr_diagnostic.csv
  outputs/gap/figures/mrr_diagnostic.png
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import xgboost as xgb

import config as C
from utils import set_plot_style, save_json
import gap_common as G

C.set_seed()

F, M = G.OTHER_GROUP, G.REFERENCE_GROUP   # "female", "male"


def merit_classifier():
    return xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        eval_metric="logloss", random_state=C.SEED, n_jobs=1)


def merit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merit covariates only -- gender is audited, never an input."""
    num = [c for c in G.NUMERIC_COV if c in df.columns]
    X = df[num].astype(float).copy()
    cat = [c for c in G.CATEG_COV if c in df.columns]
    if cat:
        X = pd.concat([X, pd.get_dummies(df[cat].astype(str),
                       drop_first=True, dtype=float)], axis=1)
    return X.fillna(X.median(numeric_only=True))


def _rates(label, pred, group):
    """(base rate F, base rate M, selection rate F, selection rate M)."""
    f, m = group == F, group == M
    return (float(label[f].mean()), float(label[m].mean()),
            float(pred[f].mean()),  float(pred[m].mean()))


def main() -> None:
    set_plot_style()

    # --- load the SAME cleaned rows + merit features used everywhere --------
    df = pd.read_parquet(C.PROC_PARQUET).reset_index(drop=True)
    df[G.GROUP_COL] = df[G.GROUP_COL].astype(str).str.lower().str.strip()
    df = df[df[G.GROUP_COL].isin([F, M])].reset_index(drop=True)
    if "Specialization" in df.columns:
        spec = df["Specialization"].astype(str).str.lower().str.strip()
        top = spec.value_counts().head(G.TOP_K_SPEC).index
        df["spec_group"] = np.where(spec.isin(top), spec, "other")

    X = merit_features(df)
    y = df[C.TARGET].astype(int).values
    g = df[G.GROUP_COL].values

    # --- merit-only out-of-fold predictions (honest) -----------------------
    cv = StratifiedKFold(C.N_SPLITS, shuffle=True, random_state=C.SEED)
    proba = cross_val_predict(merit_classifier(), X, y, cv=cv,
                              method="predict_proba", n_jobs=1)[:, 1]
    pred = (proba >= 0.5).astype(int)

    br_f, br_m, sr_f, sr_m = _rates(y, pred, g)
    delta_label = br_f - br_m       # realized reward gap (expect < 0: women penalised)
    delta_merit = sr_f - sr_m       # merit selection gap (expect > 0: women favoured)
    reversal = bool(np.sign(delta_merit) != np.sign(delta_label)
                    and delta_merit > 0 > delta_label)
    magnitude = delta_merit - delta_label

    # --- stratified bootstrap CIs (resample within each group) -------------
    rng = np.random.default_rng(C.SEED)
    idx_f = np.where(g == F)[0]; idx_m = np.where(g == M)[0]
    bl, bm, mag = [], [], []
    for _ in range(2000):
        sf = rng.choice(idx_f, len(idx_f), replace=True)
        sm = rng.choice(idx_m, len(idx_m), replace=True)
        dl = y[sf].mean() - y[sm].mean()
        dm = pred[sf].mean() - pred[sm].mean()
        bl.append(dl); bm.append(dm); mag.append(dm - dl)
    ci = lambda a: [float(np.quantile(a, .025)), float(np.quantile(a, .975))]

    # --- triangulation: pull Oaxaca + counterfactual if present ------------
    def _load(path, key, default=None):
        try:
            with open(path) as fh: return json.load(fh).get(key, default)
        except Exception: return default
    oaxaca_unexpl = _load(f"{G.GAP_MET}/oaxaca.json", "unexplained_logpts")
    oaxaca_ci     = _load(f"{G.GAP_MET}/oaxaca.json", "bootstrap_95ci")
    cf_effect     = _load(f"{G.GAP_MET}/ml_counterfactual.json",
                          "counterfactual_female_minus_male_logpts")

    out = {
        "n_female": int(len(idx_f)), "n_male": int(len(idx_m)),
        "baserate_female": br_f, "baserate_male": br_m,
        "selrate_female": sr_f, "selrate_male": sr_m,
        "delta_label_F_minus_M": delta_label, "delta_label_95ci": ci(bl),
        "delta_merit_F_minus_M": delta_merit, "delta_merit_95ci": ci(bm),
        "reversal_magnitude": magnitude, "reversal_magnitude_95ci": ci(mag),
        "merit_reward_reversal": reversal,
        "triangulation": {
            "oaxaca_unexplained_logpts": oaxaca_unexpl,
            "oaxaca_unexplained_95ci": oaxaca_ci,
            "counterfactual_female_minus_male_logpts": cf_effect,
            "all_three_agree": bool(
                reversal and (oaxaca_unexpl or 0) > 0 and (cf_effect or 0) < 0)},
        "decision": ("LABEL-RESIDENT: model-side mitigation mis-targeted; "
                     "apply label correction (gap06_dglc.py)") if reversal else
                    ("MODEL-RESIDENT: standard in-/post-processing appropriate"),
    }
    save_json(out, f"{G.GAP_MET}/mrr_diagnostic.json")

    tab = pd.DataFrame([
        ("realized base rate", br_f, br_m, delta_label, *ci(bl)),
        ("merit selection rate", sr_f, sr_m, delta_merit, *ci(bm)),
    ], columns=["quantity", "female", "male", "F_minus_M", "ci_lo", "ci_hi"])
    tab.to_csv(f"{G.GAP_TAB}/mrr_diagnostic.csv", index=False)

    # --- figure ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    x = np.arange(2); w = 0.36
    ax.bar(x - w/2, [br_f, sr_f], w, color="#D1495B", label="female")
    ax.bar(x + w/2, [br_m, sr_m], w, color="#4C8CBF", label="male")
    ax.set_xticks(x)
    ax.set_xticklabels(["Realized reward\n(high-salary label)",
                        "Merit selection\n(model, gender withheld)"])
    ax.set_ylabel("rate")
    ax.set_title("Merit-Reward Reversal: the label penalises the group\n"
                 "a merit model favours" if reversal else "No reversal detected")
    for xi, (vf, vm) in zip(x, [(br_f, br_m), (sr_f, sr_m)]):
        ax.text(xi - w/2, vf + .01, f"{vf:.3f}", ha="center", fontsize=8)
        ax.text(xi + w/2, vm + .01, f"{vm:.3f}", ha="center", fontsize=8)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{G.GAP_FIG}/mrr_diagnostic.png", dpi=C.PLOT_DPI)
    plt.close(fig)

    print(f"[mrr] delta_label (F-M realized) = {delta_label:+.3f}  CI {ci(bl)}")
    print(f"[mrr] delta_merit (F-M selection)= {delta_merit:+.3f}  CI {ci(bm)}")
    print(f"[mrr] reversal magnitude         = {magnitude:+.3f}  CI {ci(mag)}")
    print(f"[mrr] MERIT-REWARD REVERSAL      = {reversal}")
    print(f"[mrr] triangulation all-agree    = {out['triangulation']['all_three_agree']}")
    print(f"[mrr] decision                   : {out['decision']}")


if __name__ == "__main__":
    main()
