"""
Stage 16 - Analysis 1: label-threshold sensitivity (highest reviewer value).

The above-median (INR 300,000) label is discretionary. We re-derive the target under
alternative definitions and re-run the two headline results, using the identical
protocol (same split, same in-fold preprocessing, same leakage exclusions); only the
label definition changes:
  * median  (original; sanity check that the protocol reproduces the paper)
  * 60th-percentile cutoff
  * top-tercile (top 33%) cutoff
  * continuous-salary regression (Spearman rank performance + gender rank gap)

For each definition we report the best tree (CatBoost) and best deep (FT-Transformer)
test AUROC with CIs, whether the tree beats the deep model, and the gender
equalized-odds gap of the deployed-style tree at the default threshold.

Outputs: outputs/tables/label_sensitivity.csv/.tex
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from xgboost import XGBRegressor
from fairlearn.metrics import equalized_odds_difference

import config as C
from utils import classification_metrics, bootstrap_ci, df_to_latex
import data_module as D
from s09_tabular_dl import prepare as dl_prepare, train_ft

C.set_seed()


def labels_for(aux, kind):
    s = aux["Salary"].values
    if kind == "median":
        return (s > np.median(s)).astype(int)
    if kind == "p60":
        return (s > np.percentile(s, 60)).astype(int)
    if kind == "tercile":
        return (s >= np.percentile(s, 67)).astype(int)
    raise ValueError(kind)


def main():
    df = D.load_frame()
    X, _, numeric, categorical = D.get_Xy(df)
    aux = pd.read_parquet(C.PROC_PARQUET.replace("processed", "processed_aux"))
    g = df[C.SENSITIVE].values
    sp = D.get_splits(df); tr_all, te = np.array(sp["train"]), np.array(sp["test"])

    rows = []
    for kind in ["median", "p60", "tercile"]:
        y = labels_for(aux, kind)
        # --- best tree: CatBoost, identical pipeline/preprocessing ---
        cat = Pipeline([("prep", D.build_preprocessor(numeric, categorical)),
                        ("clf", CatBoostClassifier(iterations=500, learning_rate=0.05,
                                                   depth=5, l2_leaf_reg=3.0, verbose=0,
                                                   random_seed=C.SEED))])
        cat.fit(X.iloc[tr_all], y[tr_all])
        p_tree = cat.predict_proba(X.iloc[te])[:, 1]
        auc_t = classification_metrics(y[te], p_tree)["AUROC"]
        _, tlo, thi = bootstrap_ci(y[te], p_tree, "AUROC", seed=C.SEED)
        eo = equalized_odds_difference(y[te], (p_tree >= 0.5).astype(int),
                                       sensitive_features=g[te])
        # --- best deep: FT-Transformer, identical split + val selection ---
        tr, va = train_test_split(tr_all, test_size=0.15, random_state=C.SEED,
                                  stratify=y[tr_all])
        (Xtr, Xctr), (Xva, Xcva), cards, _, _ = dl_prepare(df, tr, va)
        (_, _), (Xte, Xcte), _, _, _ = dl_prepare(df, tr, te)
        p_deep = train_ft(Xtr, Xctr, y[tr], Xva, Xcva, y[va], Xte, Xcte, cards, epochs=60)
        auc_d = classification_metrics(y[te], p_deep)["AUROC"]
        _, dlo, dhi = bootstrap_ci(y[te], p_deep, "AUROC", seed=C.SEED)
        rows.append({"label": kind, "pos_rate": round(float(y.mean()), 3),
                     "best_tree": "CatBoost", "tree_AUROC": round(auc_t, 3),
                     "best_deep": "FT-Transformer", "deep_AUROC": round(auc_d, 3),
                     "AUROC_diff": round(auc_t - auc_d, 3),
                     "EO_gap": round(float(eo), 3)})
        print(f"[{kind}] tree={auc_t:.3f} deep={auc_d:.3f} "
              f"tree>=deep={auc_t>=auc_d} EO={eo:.3f}")

    # --- continuous-salary regression robustness ---
    reg = Pipeline([("prep", D.build_preprocessor(numeric, categorical)),
                    ("clf", XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4,
                                         subsample=0.85, colsample_bytree=0.85,
                                         random_state=C.SEED))])
    reg.fit(X.iloc[tr_all], aux["log_salary"].values[tr_all])
    pred = reg.predict(X.iloc[te])
    rho = stats.spearmanr(pred, aux["log_salary"].values[te]).statistic
    rank = pd.Series(pred).rank(pct=True).values
    gap = rank[g[te] == "male"].mean() - rank[g[te] == "female"].mean()
    print(f"[continuous] Spearman rho={rho:.3f} | gender mean-rank gap (M-F)={gap:+.3f}")

    out = pd.DataFrame(rows)
    out.to_csv(f"{C.TAB_DIR}/label_sensitivity.csv", index=False)
    df_to_latex(out, f"{C.TAB_DIR}/label_sensitivity.tex",
                "Label-threshold sensitivity: best tree (CatBoost) and best deep model "
                "(FT-Transformer) test AUROC, their difference, and the gender equalized-odds gap under "
                "alternative target definitions. The continuous-salary regression gives "
                f"Spearman $\\rho={rho:.2f}$ with a gender mean-rank gap of {gap:+.2f}. "
                "The equalized-odds gap here is computed for the best tree (CatBoost) "
                "under the label-sensitivity re-run protocol and differs slightly from "
                "the deployed random-forest audit in Section~V-G.",
                "tab:labelsens")
    print("\n", out.to_string(index=False))
    print("\nStage 16 complete.")


if __name__ == "__main__":
    main()
