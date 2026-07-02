"""
Stage 12 - Calibration analysis.

The framework presents System 1 probabilities to a human, so the probabilities
must be calibrated, not merely discriminative. We report the Brier score and
Expected Calibration Error (ECE) for representative models, before and after
Platt (sigmoid) and isotonic recalibration fitted on the training partition.

Outputs:
  outputs/tables/calibration.csv / .tex
  outputs/metrics/calibration.json
"""
from __future__ import annotations
import glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier

import config as C
from utils import save_json, df_to_latex
import data_module as D

C.set_seed()


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum() > 0:
            e += abs(p[m].mean() - y[m].mean()) * m.sum() / len(y)
    return float(e)


def main():
    df = D.load_frame()
    X, y, numeric, categorical = D.get_Xy(df)
    sp = D.get_splits(df)
    tr, te = np.array(sp["train"]), np.array(sp["test"])
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]

    def pipe(clf):
        return Pipeline([("prep", D.build_preprocessor(numeric, categorical)),
                         ("clf", clf)])
    models = {
        "Deployed (RandomForest)": joblib.load(
            glob.glob(f"{C.MODEL_DIR}/system1_best_*.joblib")[0]),
        "CatBoost": pipe(CatBoostClassifier(iterations=500, learning_rate=0.05,
                                            depth=5, verbose=0, random_seed=C.SEED)),
        "LogReg": pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
    }
    rows = []
    for name, base in models.items():
        if not name.startswith("Deployed"):     # deployed model is already fitted
            base.fit(Xtr, ytr)
        praw = base.predict_proba(Xte)[:, 1]
        row = {"model": name,
               "Brier_raw": round(brier_score_loss(yte, praw), 4),
               "ECE_raw": round(ece(yte, praw), 4)}
        for method in ("sigmoid", "isotonic"):
            cal = CalibratedClassifierCV(base, method=method, cv=5)
            cal.fit(Xtr, ytr)
            pc = cal.predict_proba(Xte)[:, 1]
            tag = "Platt" if method == "sigmoid" else "Isotonic"
            row[f"Brier_{tag}"] = round(brier_score_loss(yte, pc), 4)
            row[f"ECE_{tag}"] = round(ece(yte, pc), 4)
        rows.append(row)
        print(name, "->", row)

    wide = pd.DataFrame(rows)
    wide.to_csv(f"{C.MET_DIR}/calibration.csv", index=False)
    save_json(rows, f"{C.MET_DIR}/calibration.json")
    # long format: one row per (model, method) with Brier and ECE
    long = []
    for r in rows:
        for tag in ["raw", "Platt", "Isotonic"]:
            long.append({"Model": r["model"], "Method": tag,
                         "Brier": r[f"Brier_{tag}"], "ECE": r[f"ECE_{tag}"]})
    long_df = pd.DataFrame(long)
    long_df.to_csv(f"{C.TAB_DIR}/calibration_long.csv", index=False)
    df_to_latex(long_df, f"{C.TAB_DIR}/calibration.tex",
                "Calibration on the held-out test set: Brier score and Expected "
                "Calibration Error (ECE) for each model, raw and after Platt or "
                "isotonic recalibration (lower is better).",
                "tab:calibration")
    print(long_df.to_string(index=False))

    # ---- calibration-curve (reliability) figure --------------------------
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from utils import set_plot_style
    set_plot_style()
    rf = models["Deployed (RandomForest)"]
    rf_iso = CalibratedClassifierCV(rf, method="isotonic", cv=5).fit(Xtr, ytr)
    curves = {
        "RF raw":        rf.predict_proba(Xte)[:, 1],
        "RF isotonic":   rf_iso.predict_proba(Xte)[:, 1],
        "CatBoost raw":  models["CatBoost"].predict_proba(Xte)[:, 1],
        "LogReg raw":    models["LogReg"].predict_proba(Xte)[:, 1],
    }
    plt.figure(figsize=(5.6, 5))
    plt.plot([0, 1], [0, 1], "k--", lw=0.9, label="perfectly calibrated")
    styles = {"RF raw": ("o-", "#4c78a8"), "RF isotonic": ("s-", "#1b5e9b"),
              "CatBoost raw": ("^-", "#e45756"), "LogReg raw": ("d-", "#54a24b")}
    for name, p in curves.items():
        frac, mean_pred = calibration_curve(yte, p, n_bins=10, strategy="quantile")
        st, col = styles[name]
        plt.plot(mean_pred, frac, st, color=col, lw=1.5, ms=5, label=name)
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
    plt.title("Calibration (reliability) curves — held-out test")
    plt.legend(fontsize=8); plt.xlim(0, 1); plt.ylim(0, 1)
    plt.savefig(f"{C.FIG_DIR}/fig_calibration_curve.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("\nStage 12 complete (with calibration-curve figure).")


if __name__ == "__main__":
    main()
