"""
Robustness check for Reviewer 3, item 1 (lives in code/ next to gap06_dglc.py).
gap06_dglc.py estimates b_M, b_F, b* once on the full cohort and thresholds at the cohort
median; only the classifier is out-of-fold. This script repeats the DGLC frontier with
everything estimated on the TRAINING folds only.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import config as C
import gap_common as G
from gap06_dglc import design, reference_betas, corrected_outcome, clf, F

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def main() -> None:
    C.set_seed()
    df = G.load_gap_table().reset_index(drop=True)
    X = design(df)
    Xv = X.values
    y = df[G.OUTCOME].astype(float).values
    is_f = (df[G.GROUP_COL].values == F)
    feat = [c for c in X.columns if c != "const"]
    Xc = X[feat].fillna(X[feat].median()).values
    realized_bin = (y > np.median(y)).astype(int)
    structure = G.REFERENCE_STRUCTURE

    cv = StratifiedKFold(C.N_SPLITS, shuffle=True, random_state=C.SEED)
    splits = list(cv.split(Xc, realized_bin))

    rows = []
    for a in ALPHAS:
        p = np.zeros(len(y))
        lab_oof = np.zeros(len(y), dtype=int)
        for tr, te in splits:
            b_m, b_f, b_star = reference_betas(X.iloc[tr], y[tr], is_f[tr], structure)
            corr_tr = corrected_outcome(y[tr], Xv[tr], is_f[tr], b_m, b_f, b_star, a)
            corr_te = corrected_outcome(y[te], Xv[te], is_f[te], b_m, b_f, b_star, a)
            thr = float(np.median(corr_tr))
            lab_tr = (corr_tr > thr).astype(int)
            lab_oof[te] = (corr_te > thr).astype(int)
            model = clf().fit(Xc[tr], lab_tr)
            p[te] = model.predict_proba(Xc[te])[:, 1]
        pred = (p >= 0.5).astype(int)
        rows.append(dict(
            alpha=a,
            sel_F=float(pred[is_f].mean()),
            sel_M=float(pred[~is_f].mean()),
            base_F=float(lab_oof[is_f].mean()),
            base_M=float(lab_oof[~is_f].mean()),
            auroc_realized=float(roc_auc_score(realized_bin, p)),
        ))
    fold = pd.DataFrame(rows)

    os.makedirs(G.GAP_TAB, exist_ok=True)
    fold.to_csv(f"{G.GAP_TAB}/dglc_foldwise_check.csv", index=False)

    cohort_path = f"{G.GAP_TAB}/dglc_alpha_frontier.csv"
    print("\n=== Fold-wise estimation (coefficients and threshold from training folds only) ===")
    print(fold.round(4).to_string(index=False))
    if os.path.exists(cohort_path):
        cohort = pd.read_csv(cohort_path)
        keep = [c for c in ["alpha", "sel_F", "sel_M", "base_F", "base_M", "auroc_realized"]
                if c in cohort.columns]
        print("\n=== Cohort-level estimation (gap06_dglc.py, as reported in the paper) ===")
        print(cohort[keep].round(4).to_string(index=False))
        m = fold.merge(cohort[keep], on="alpha", suffixes=("_fold", "_cohort"))
        print("\n=== Differences (fold-wise minus cohort-level) ===")
        for c in ["sel_F", "sel_M", "base_F", "base_M", "auroc_realized"]:
            if f"{c}_cohort" in m.columns:
                d = m[f"{c}_fold"] - m[f"{c}_cohort"]
                print(f"  {c:15s} max |diff| = {d.abs().max():.4f}")
    print(f"\nWritten: {G.GAP_TAB}/dglc_foldwise_check.csv")


if __name__ == "__main__":
    main()
