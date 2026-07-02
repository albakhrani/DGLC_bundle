"""
Stage 4 - System 1 (fast, intuitive predictors).

Trains a panel of eleven classifiers under an identical
RepeatedStratifiedKFold protocol on the training partition, then evaluates the
refit models on the held-out test set.

Rigour features:
  * Identical folds across all models  -> paired comparison is valid.
  * Nadeau-Bengio corrected resampled paired t-test (the correct test for
    repeated k-fold CV on a single dataset) between the top model and each rival.
  * Friedman omnibus test + Nemenyi critical-difference ranking (descriptive).
  * Held-out test metrics with percentile-bootstrap 95% CIs.
  * Out-of-sample test probabilities saved for the System-2 stage.

Outputs:
  outputs/metrics/system1_cv.csv, system1_test.csv, system1_cv_folds.csv
  outputs/metrics/system1_significance.json
  outputs/metrics/test_probs_<model>.npy
  outputs/figures/fig_roc.png, fig_pr.png, fig_calibration.png,
                  fig_model_ranking.png
  outputs/models/<best>.joblib
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import joblib
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from interpret.glassbox import ExplainableBoostingClassifier

import config as C
from utils import (set_plot_style, classification_metrics, bootstrap_ci,
                   save_json, df_to_latex)
import data_module as D

C.set_seed()
set_plot_style()


def make_models():
    """Eleven classifiers with sensible, documented default hyper-parameters."""
    return {
        "LogReg":      LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
        "GaussianNB":  GaussianNB(),
        "kNN":         KNeighborsClassifier(n_neighbors=25, weights="distance"),
        "SVM-RBF":     SVC(C=2.0, gamma="scale", probability=True,
                           class_weight="balanced", random_state=C.SEED),
        "DecisionTree":DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                              random_state=C.SEED),
        "RandomForest":RandomForestClassifier(n_estimators=400, max_depth=None,
                                              min_samples_leaf=2, n_jobs=1,
                                              class_weight="balanced_subsample",
                                              random_state=C.SEED),
        "ExtraTrees":  ExtraTreesClassifier(n_estimators=400, n_jobs=1,
                                            class_weight="balanced",
                                            random_state=C.SEED),
        "HistGB":      HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                                      max_depth=None, l2_regularization=1.0,
                                                      random_state=C.SEED),
        "XGBoost":     XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4,
                                     subsample=0.85, colsample_bytree=0.85,
                                     reg_lambda=1.5, eval_metric="auc",
                                     n_jobs=1, random_state=C.SEED),
        "CatBoost":    CatBoostClassifier(iterations=500, learning_rate=0.05, depth=5,
                                          l2_leaf_reg=3.0, verbose=0,
                                          random_seed=C.SEED),
        "LightGBM":    LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                      num_leaves=31, subsample=0.85,
                                      colsample_bytree=0.85, reg_lambda=1.5,
                                      class_weight="balanced", n_jobs=1,
                                      random_state=C.SEED, verbose=-1),
        "MLP":         MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                                     max_iter=400, early_stopping=True,
                                     random_state=C.SEED),
        "EBM":         ExplainableBoostingClassifier(interactions=0,
                                                     random_state=C.SEED),
    }


def pipe(model, numeric, categorical):
    return Pipeline([("prep", D.build_preprocessor(numeric, categorical)),
                     ("clf", model)])


def nadeau_bengio_t(diffs, n_train, n_test):
    """Corrected resampled paired t-test (Nadeau & Bengio, 2003)."""
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    mean = diffs.mean()
    var = diffs.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    corr = (1.0 / n) + (n_test / n_train)        # correction factor
    t = mean / np.sqrt(corr * var)
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return float(t), float(p)


def nemenyi_cd(k, n, alpha="0.05"):
    """Critical difference for the Nemenyi post-hoc test."""
    q = {  # studentized range / sqrt(2) at alpha=0.05
        2:1.960,3:2.343,4:2.569,5:2.728,6:2.850,7:2.949,8:3.031,
        9:3.102,10:3.164,11:3.219,12:3.268,13:3.313,14:3.354,15:3.391}
    return q.get(k, 3.4) * np.sqrt(k * (k + 1) / (6.0 * n))


def main():
    df = D.load_frame()
    X, y, numeric, categorical = D.get_Xy(df)
    sp = D.get_splits(df)
    tr, te = np.array(sp["train"]), np.array(sp["test"])
    Xtr, ytr = X.iloc[tr], y[tr]
    Xte, yte = X.iloc[te], y[te]

    models = make_models()
    cv = RepeatedStratifiedKFold(n_splits=C.N_SPLITS, n_repeats=C.N_REPEATS,
                                 random_state=C.SEED)
    folds = list(cv.split(Xtr, ytr))
    n_fold = len(folds)
    n_train_fold = len(folds[0][0])
    n_test_fold = len(folds[0][1])

    # ---- aligned per-fold AUROC matrix (folds x models) -------------------
    fold_auc = {m: np.zeros(n_fold) for m in models}
    print(f"Repeated CV: {n_fold} folds  x  {len(models)} models")
    for fi, (itr, iva) in enumerate(folds):
        Xf_tr, yf_tr = Xtr.iloc[itr], ytr[itr]
        Xf_va, yf_va = Xtr.iloc[iva], ytr[iva]
        for name, mdl in models.items():
            p = pipe(mdl, numeric, categorical)
            p.fit(Xf_tr, yf_tr)
            prob = p.predict_proba(Xf_va)[:, 1]
            fold_auc[name][fi] = roc_auc_score(yf_va, prob)
        print(f"  fold {fi+1}/{n_fold} done")

    cv_df = pd.DataFrame(fold_auc)
    cv_df.to_csv(f"{C.MET_DIR}/system1_cv_folds.csv", index=False)
    cv_summary = (cv_df.agg(["mean", "std"]).T
                  .rename(columns={"mean": "CV_AUROC_mean", "std": "CV_AUROC_std"})
                  .sort_values("CV_AUROC_mean", ascending=False))
    cv_summary.to_csv(f"{C.MET_DIR}/system1_cv.csv")
    print("\nCV AUROC ranking:\n", cv_summary.round(4))

    # ---- significance: corrected t-test vs best, + Friedman/Nemenyi -------
    best = cv_summary.index[0]
    sig = {"best_model": best, "n_folds": n_fold, "vs_best": {}}
    for m in models:
        if m == best:
            continue
        t, p = nadeau_bengio_t(cv_df[best] - cv_df[m], n_train_fold, n_test_fold)
        sig["vs_best"][m] = {"delta_AUROC": float(cv_df[best].mean() - cv_df[m].mean()),
                             "t": t, "p_corrected": p}
    fr_stat, fr_p = stats.friedmanchisquare(*[cv_df[m].values for m in models])
    ranks = cv_df.rank(axis=1, ascending=False).mean().sort_values()
    sig["friedman"] = {"chi2": float(fr_stat), "p": float(fr_p)}
    sig["mean_ranks"] = ranks.round(3).to_dict()
    sig["nemenyi_CD"] = float(nemenyi_cd(len(models), n_fold))
    save_json(sig, f"{C.MET_DIR}/system1_significance.json")
    print(f"\nFriedman chi2={fr_stat:.2f} p={fr_p:.2e} | Nemenyi CD={sig['nemenyi_CD']:.3f}")

    # ---- deployed System-2 model: CV-best *tree* --------------------------
    # System 2 (TreeExplainer SHAP + counterfactual recourse) is defined for
    # tree ensembles, so the deployed model is the best-performing tree, even
    # if a glass-box GAM (EBM) edges it on overall CV AUROC. EBM is reported
    # as an interpretable baseline only.
    TREE_MODELS = {"RandomForest", "ExtraTrees", "HistGB", "XGBoost",
                   "CatBoost", "LightGBM", "DecisionTree"}
    deploy = next(m for m in cv_summary.index if m in TREE_MODELS)
    print(f"Overall CV-best: {best} | deployed tree for System 2: {deploy}")

    # ---- refit on full train, evaluate on held-out test -------------------
    test_rows, roc_data, pr_data, cal_data = [], {}, {}, {}
    for name, mdl in models.items():
        p = pipe(mdl, numeric, categorical)
        p.fit(Xtr, ytr)
        prob = p.predict_proba(Xte)[:, 1]
        np.save(f"{C.MET_DIR}/test_probs_{name}.npy", prob)
        met = classification_metrics(yte, prob)
        _, lo, hi = bootstrap_ci(yte, prob, "AUROC", seed=C.SEED)
        met.update({"Model": name, "AUROC_lo": lo, "AUROC_hi": hi})
        test_rows.append(met)
        roc_data[name] = roc_curve(yte, prob)
        pr_data[name] = precision_recall_curve(yte, prob)
        cal_data[name] = calibration_curve(yte, prob, n_bins=10, strategy="quantile")
        if name == deploy:
            joblib.dump(p, f"{C.MODEL_DIR}/system1_best_{name}.joblib")

    test_df = (pd.DataFrame(test_rows)
               .set_index("Model")
               .sort_values("AUROC", ascending=False))
    cols = ["AUROC", "AUROC_lo", "AUROC_hi", "AUPRC", "Accuracy", "BalAcc",
            "F1", "Precision", "Recall", "MCC", "Brier"]
    test_df = test_df[cols]
    test_df.to_csv(f"{C.MET_DIR}/system1_test.csv")
    df_to_latex(test_df.round(3).reset_index(), f"{C.TAB_DIR}/system1_test.tex",
                "Held-out test performance of System-1 models (AMEO 2015).",
                "tab:system1")
    np.save(f"{C.MET_DIR}/test_y_true.npy", yte)
    print("\nTest ranking:\n", test_df.round(3))

    _plot_curves(roc_data, pr_data, cal_data, test_df)
    _plot_ranking(ranks, sig["nemenyi_CD"])
    print("\nSystem-1 stage complete.")


def _plot_curves(roc_data, pr_data, cal_data, test_df):
    order = test_df.index.tolist()
    top = order[:6]
    # ROC
    plt.figure(figsize=(6, 5))
    for n in top:
        fpr, tpr, _ = roc_data[n]
        plt.plot(fpr, tpr, lw=1.6, label=f"{n} ({test_df.loc[n,'AUROC']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC - held-out test (top 6)"); plt.legend(fontsize=8)
    plt.savefig(f"{C.FIG_DIR}/fig_roc.png"); plt.close()
    # PR
    plt.figure(figsize=(6, 5))
    for n in top:
        prec, rec, _ = pr_data[n]
        plt.plot(rec, prec, lw=1.6, label=f"{n} ({test_df.loc[n,'AUPRC']:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall - held-out test (top 6)"); plt.legend(fontsize=8)
    plt.savefig(f"{C.FIG_DIR}/fig_pr.png"); plt.close()
    # Calibration
    plt.figure(figsize=(6, 5))
    for n in top[:4]:
        frac, mean_pred = cal_data[n]
        plt.plot(mean_pred, frac, "o-", lw=1.4, ms=4, label=n)
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
    plt.title("Calibration - held-out test"); plt.legend(fontsize=8)
    plt.savefig(f"{C.FIG_DIR}/fig_calibration.png"); plt.close()


def _plot_ranking(ranks, cd):
    plt.figure(figsize=(7, 3.2))
    y = np.arange(len(ranks))
    plt.errorbar(ranks.values, y, xerr=cd / 2, fmt="o", color="#2b6cb0",
                 capsize=3, lw=1.4)
    plt.yticks(y, ranks.index)
    plt.gca().invert_yaxis()
    plt.xlabel("Mean rank (lower = better)  -  bars = +/- CD/2")
    plt.title(f"Friedman-Nemenyi ranking (CD={cd:.2f})")
    plt.savefig(f"{C.FIG_DIR}/fig_model_ranking.png"); plt.close()


if __name__ == "__main__":
    main()
