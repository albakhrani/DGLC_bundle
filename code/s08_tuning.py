"""
Stage 8 - Bayesian hyperparameter optimization (Optuna).

Tunes the three strongest System-1 families (Random Forest, XGBoost, CatBoost)
with the TPE sampler, optimizing mean stratified 5-fold AUROC on the TRAINING
partition only, then refits the best configuration and evaluates once on the
held-out test set. This addresses the common reviewer question of whether the
default-hyperparameter ranking survives proper tuning.

Outputs:
  outputs/metrics/tuning_best_params.json
  outputs/metrics/tuning_comparison.csv
  outputs/tables/tuning_comparison.tex
  outputs/metrics/test_probs_<model>_tuned.npy
  outputs/figures/fig_tuning.png
"""
from __future__ import annotations
import warnings, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import config as C
from utils import (set_plot_style, classification_metrics, bootstrap_ci,
                   save_json, df_to_latex)
import data_module as D

optuna.logging.set_verbosity(optuna.logging.WARNING)
C.set_seed()
set_plot_style()

N_TRIALS = 30
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.SEED)


def _pipe(model, numeric, categorical):
    return Pipeline([("prep", D.build_preprocessor(numeric, categorical)),
                     ("clf", model)])


def _spaces(trial, name):
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
            max_depth=trial.suggest_int("max_depth", 4, 24),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            max_features=trial.suggest_float("max_features", 0.3, 1.0),
            class_weight="balanced_subsample", n_jobs=1, random_state=C.SEED)
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 900, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 5.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 8),
            eval_metric="auc", n_jobs=1, random_state=C.SEED)
    if name == "CatBoost":
        return CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 200, 800, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            depth=trial.suggest_int("depth", 3, 8),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            verbose=0, random_seed=C.SEED)
    raise ValueError(name)


def main():
    df = D.load_frame()
    X, y, numeric, categorical = D.get_Xy(df)
    sp = D.get_splits(df)
    tr, te = np.array(sp["train"]), np.array(sp["test"])
    Xtr, ytr, Xte, yte = X.iloc[tr], y[tr], X.iloc[te], y[te]

    # default-model test AUROC (from Stage 4) for the before/after comparison
    default_test = pd.read_csv(f"{C.MET_DIR}/system1_test.csv", index_col=0)["AUROC"]

    best_params, rows = {}, []
    for name in ["RandomForest", "XGBoost", "CatBoost"]:
        def objective(trial):
            mdl = _spaces(trial, name)
            sc = cross_val_score(_pipe(mdl, numeric, categorical), Xtr, ytr,
                                 cv=CV, scoring="roc_auc", n_jobs=1)
            return float(np.mean(sc))
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=C.SEED))
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
        best_params[name] = {"cv_auroc": study.best_value, "params": study.best_params}
        print(f"{name}: best CV AUROC={study.best_value:.4f}")

        # refit best on full train, evaluate on test
        mdl = _spaces(optuna.trial.FixedTrial(study.best_params), name)
        pipe = _pipe(mdl, numeric, categorical).fit(Xtr, ytr)
        prob = pipe.predict_proba(Xte)[:, 1]
        np.save(f"{C.MET_DIR}/test_probs_{name}_tuned.npy", prob)
        met = classification_metrics(yte, prob)
        _, lo, hi = bootstrap_ci(yte, prob, "AUROC", seed=C.SEED)
        rows.append({"Model": name,
                     "AUROC_default": float(default_test.get(name, np.nan)),
                     "AUROC_tuned": met["AUROC"], "tuned_lo": lo, "tuned_hi": hi,
                     "F1_tuned": met["F1"], "MCC_tuned": met["MCC"],
                     "delta": met["AUROC"] - float(default_test.get(name, np.nan)),
                     "best_iter_history": [t.value for t in study.trials]})

    save_json(best_params, f"{C.MET_DIR}/tuning_best_params.json")
    comp = pd.DataFrame(rows)
    comp_tbl = comp.drop(columns=["best_iter_history"]).round(4)
    comp_tbl.to_csv(f"{C.MET_DIR}/tuning_comparison.csv", index=False)
    df_to_latex(comp_tbl, f"{C.TAB_DIR}/tuning_comparison.tex",
                "Default vs.\\ Optuna-tuned held-out test AUROC (30 TPE trials, "
                "5-fold CV objective).", "tab:tuning")
    print("\nDefault vs tuned:\n", comp_tbl.to_string(index=False))

    # ---- figure: optimization history + before/after ----------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for r in rows:
        hist = np.maximum.accumulate(r["best_iter_history"])
        ax[0].plot(range(1, len(hist) + 1), hist, marker="o", ms=3, label=r["Model"])
    ax[0].set_xlabel("Trial"); ax[0].set_ylabel("Best CV AUROC so far")
    ax[0].set_title("Optuna optimization history"); ax[0].legend(fontsize=8)
    xpos = np.arange(len(comp)); w = 0.38
    ax[1].bar(xpos - w/2, comp["AUROC_default"], w, label="default", color="#9abcd6")
    ax[1].bar(xpos + w/2, comp["AUROC_tuned"], w, label="tuned", color="#1f77b4")
    ax[1].set_xticks(xpos); ax[1].set_xticklabels(comp["Model"])
    ax[1].set_ylim(0.74, 0.80); ax[1].set_ylabel("Test AUROC")
    ax[1].set_title("Default vs.\\ tuned (held-out test)"); ax[1].legend(fontsize=8)
    fig.savefig(f"{C.FIG_DIR}/fig_tuning.png", bbox_inches="tight"); plt.close(fig)
    print("\nTuning stage complete.")


if __name__ == "__main__":
    main()
