"""
Idea 1 - Stage 2: ML salary model + SHAP + gender-flip counterfactual (ML arm).

A gradient-boosted regressor predicts winsorized log-salary from the same merit
covariates plus a binary gender flag. We then:
  * report cross-validated fit (R^2, RMSE);
  * attribute the gender signal with SHAP (mean |SHAP| ranking + the gender
    feature's mean SHAP);
  * run a CETERIS-PARIBUS counterfactual: flip every record's gender, hold all
    else fixed, and measure the mean change in predicted log-salary -- a
    model-based analogue of the Oaxaca "unexplained" term.

This non-parametric arm captures interactions/non-linearities the linear
Oaxaca model cannot, and the two estimates of the residual gender effect should
be reported side by side.

Outputs:
  outputs/gap/metrics/ml_counterfactual.json
  outputs/gap/tables/shap_importance.csv
  outputs/gap/figures/shap_summary.png
  outputs/gap/figures/counterfactual_gender_flip.png
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import shap

import config as C
from utils import set_plot_style, save_json
import gap_common as G

C.set_seed()

IS_FEMALE = "is_female"


def _encode(df: pd.DataFrame):
    """Numeric design matrix with a single flippable gender flag."""
    num = [c for c in G.NUMERIC_COV if c in df.columns]
    X = df[num].astype(float).copy()
    # binary gender flag (the counterfactual lever)
    X[IS_FEMALE] = (df[G.GROUP_COL] == G.OTHER_GROUP).astype(int).values
    # categorical endowments -> one-hot (stable, low cardinality)
    cat = [c for c in G.CATEG_COV if c in df.columns]
    if cat:
        X = pd.concat([X, pd.get_dummies(df[cat].astype(str),
                                         drop_first=True, dtype=float)], axis=1)
    y = df[G.OUTCOME].astype(float).values
    return X, y


def _model():
    return xgb.XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=C.SEED, n_jobs=1,
    )


def main() -> None:
    set_plot_style()
    df = G.load_gap_table()
    X, y = _encode(df)

    # ---- cross-validated fit ----------------------------------------------
    cv = KFold(n_splits=C.N_SPLITS, shuffle=True, random_state=C.SEED)
    oof = cross_val_predict(_model(), X, y, cv=cv, n_jobs=1)
    r2 = float(r2_score(y, oof))
    rmse = float(np.sqrt(mean_squared_error(y, oof)))

    # ---- fit on full data for attribution ---------------------------------
    model = _model().fit(X, y)

    # ---- counterfactual gender flip (ceteris paribus) ---------------------
    X_flip = X.copy()
    X_flip[IS_FEMALE] = 1 - X_flip[IS_FEMALE]
    pred_obs = model.predict(X)
    pred_cf = model.predict(X_flip)
    # effect of being female vs male, per record, holding all else fixed:
    # for observed males  -> (flip->female) - observed
    # for observed females-> observed - (flip->male)
    female_mask = X[IS_FEMALE].values == 1
    delta_female_minus_male = np.where(
        female_mask, pred_obs - pred_cf, pred_cf - pred_obs)
    cf_gap = float(np.mean(delta_female_minus_male))   # mean female-male effect

    # ---- SHAP attribution -------------------------------------------------
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    mean_abs = np.abs(sv).mean(axis=0)
    imp = (pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
           .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    imp.to_csv(f"{G.GAP_TAB}/shap_importance.csv", index=False)
    gender_idx = list(X.columns).index(IS_FEMALE)
    gender_mean_shap = float(sv[:, gender_idx].mean())
    gender_rank = int(imp.index[imp["feature"] == IS_FEMALE][0]) + 1

    save_json({
        "cv_r2": r2, "cv_rmse": rmse,
        "counterfactual_female_minus_male_logpts": cf_gap,
        "counterfactual_approx_pct_INR": float(np.expm1(cf_gap) * 100),
        "gender_mean_shap_logpts": gender_mean_shap,
        "gender_importance_rank": gender_rank,
        "n_features": int(X.shape[1]),
    }, f"{G.GAP_MET}/ml_counterfactual.json")

    # ---- figures ----------------------------------------------------------
    shap.summary_plot(sv, X, show=False, max_display=12)
    plt.title("SHAP summary - drivers of log-salary")
    plt.tight_layout()
    plt.savefig(f"{G.GAP_FIG}/shap_summary.png", dpi=C.PLOT_DPI)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(delta_female_minus_male, bins=40, color="#D1495B", alpha=.85)
    ax.axvline(cf_gap, c="black", ls="--",
               label=f"mean = {cf_gap:+.4f} log pts\n(~{np.expm1(cf_gap)*100:+.1f}% INR)")
    ax.axvline(0, c="grey", lw=1)
    ax.set_xlabel("counterfactual (female − male) effect on predicted log-salary")
    ax.set_ylabel("graduates")
    ax.set_title("Ceteris-paribus gender-flip counterfactual")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{G.GAP_FIG}/counterfactual_gender_flip.png", dpi=C.PLOT_DPI)
    plt.close(fig)

    print(f"[ml] CV R^2={r2:.3f}  RMSE={rmse:.3f}")
    print(f"[ml] counterfactual female-male effect: {cf_gap:+.4f} log pts "
          f"(~{np.expm1(cf_gap)*100:+.1f}% INR)")
    print(f"[ml] gender mean SHAP: {gender_mean_shap:+.4f}  "
          f"(importance rank {gender_rank}/{X.shape[1]})")
    print(f"[ml] saved metrics/figures under {G.GAP_DIR}")


if __name__ == "__main__":
    main()
