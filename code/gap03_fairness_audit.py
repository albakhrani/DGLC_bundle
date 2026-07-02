"""
Idea 1 - Stage 3: group-fairness audit of an above-median-salary classifier.

Where stages 1-2 quantify the *wage* gap, this stage asks whether a deployed
predictor of a strong earnings outcome would itself treat genders unequally.
We train a gradient-boosted classifier of `high_salary` on merit covariates
ONLY (gender withheld from the model, as in a fairness-aware deployment), take
honest out-of-fold predictions, and measure standard group-fairness criteria
with Fairlearn:

  * selection rate by group          (who gets the positive prediction)
  * demographic-parity difference/ratio  (ratio < 0.8 -> "80% rule" flag)
  * equalized-odds difference        (TPR & FPR balance)
  * per-group TPR / FPR / accuracy

Outputs:
  outputs/gap/tables/fairness_by_group.csv
  outputs/gap/metrics/fairness.json
  outputs/gap/figures/fairness_rates.png
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import xgboost as xgb
from fairlearn.metrics import (
    MetricFrame, selection_rate, demographic_parity_difference,
    demographic_parity_ratio, equalized_odds_difference,
    true_positive_rate, false_positive_rate,
)
from sklearn.metrics import accuracy_score

import config as C
from utils import set_plot_style, save_json
import gap_common as G

C.set_seed()


def _features(df: pd.DataFrame):
    """Merit covariates only -- gender is the audited attribute, not an input."""
    num = [c for c in G.NUMERIC_COV if c in df.columns]
    X = df[num].astype(float).copy()
    cat = [c for c in G.CATEG_COV if c in df.columns]
    if cat:
        X = pd.concat([X, pd.get_dummies(df[cat].astype(str),
                                         drop_first=True, dtype=float)], axis=1)
    return X


def main() -> None:
    set_plot_style()
    # processed.parquet carries the binary target + gender + merit features
    df = pd.read_parquet(C.PROC_PARQUET).reset_index(drop=True)
    df[G.GROUP_COL] = df[G.GROUP_COL].astype(str).str.lower().str.strip()
    df = df[df[G.GROUP_COL].isin([G.REFERENCE_GROUP, G.OTHER_GROUP])].reset_index(drop=True)

    # collapse specializations the same way as gap_common for parity
    if "Specialization" in df.columns:
        spec = df["Specialization"].astype(str).str.lower().str.strip()
        top = spec.value_counts().head(G.TOP_K_SPEC).index
        df["spec_group"] = np.where(spec.isin(top), spec, "other")

    X = _features(df)
    y = df[C.TARGET].astype(int).values
    group = df[G.GROUP_COL].values

    # impute any residual gaps (median) so the classifier is well-defined
    X = X.fillna(X.median(numeric_only=True))

    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        eval_metric="logloss", random_state=C.SEED, n_jobs=1,
    )
    cv = StratifiedKFold(n_splits=C.N_SPLITS, shuffle=True, random_state=C.SEED)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba",
                              n_jobs=1)[:, 1]
    pred = (proba >= 0.5).astype(int)

    mf = MetricFrame(
        metrics={"accuracy": accuracy_score,
                 "selection_rate": selection_rate,
                 "TPR": true_positive_rate,
                 "FPR": false_positive_rate},
        y_true=y, y_pred=pred, sensitive_features=group)
    by_group = mf.by_group.copy()
    by_group.to_csv(f"{G.GAP_TAB}/fairness_by_group.csv")

    dp_diff  = float(demographic_parity_difference(y, pred, sensitive_features=group))
    dp_ratio = float(demographic_parity_ratio(y, pred, sensitive_features=group))
    eo_diff  = float(equalized_odds_difference(y, pred, sensitive_features=group))

    save_json({
        "selection_rate_by_group": by_group["selection_rate"].to_dict(),
        "tpr_by_group": by_group["TPR"].to_dict(),
        "fpr_by_group": by_group["FPR"].to_dict(),
        "accuracy_by_group": by_group["accuracy"].to_dict(),
        "demographic_parity_difference": dp_diff,
        "demographic_parity_ratio": dp_ratio,
        "passes_80pct_rule": bool(dp_ratio >= 0.8),
        "equalized_odds_difference": eo_diff,
        "note": "Gender withheld from the model; disparities reflect proxy "
                "signals in merit covariates, not direct use of gender.",
    }, f"{G.GAP_MET}/fairness.json")

    # ---- figure: selection rate & TPR by group ----------------------------
    groups = list(by_group.index)
    x = np.arange(len(groups)); w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x - w/2, by_group["selection_rate"], w, label="selection rate",
           color="#4C8CBF")
    ax.bar(x + w/2, by_group["TPR"], w, label="true-positive rate",
           color="#E3A857")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("rate")
    ax.set_title(f"Fairness audit (DI ratio = {dp_ratio:.2f}, "
                 f"{'PASS' if dp_ratio>=0.8 else 'FAIL'} 80% rule)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{G.GAP_FIG}/fairness_rates.png", dpi=C.PLOT_DPI)
    plt.close(fig)

    print(f"[fairness] selection rate by group: "
          f"{by_group['selection_rate'].round(3).to_dict()}")
    print(f"[fairness] demographic-parity ratio (DI): {dp_ratio:.3f}  "
          f"({'PASS' if dp_ratio>=0.8 else 'FAIL'} 80% rule)")
    print(f"[fairness] demographic-parity diff       : {dp_diff:+.3f}")
    print(f"[fairness] equalized-odds difference     : {eo_diff:+.3f}")
    print(f"[fairness] saved under {G.GAP_DIR}")


if __name__ == "__main__":
    main()
