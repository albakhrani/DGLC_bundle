"""
Stage 18 - Error analysis, subgroup sample sizes, and deployed-model sensitivity.

Three reviewer-oriented analyses on the held-out test set:

  E1  Error analysis. Confusion-level breakdown of the deployed model at the
      0.5 threshold (TP/TN/FP/FN) with group composition (gender), aptitude
      means, and mean predicted probability; plus a SHAP comparison of which
      features the model leans on in false positives versus false negatives.
      The point is not to "fix" individual errors but to show where automated
      scores are least reliable and human review matters most.

  E2  Subgroup sample sizes. Gender x specialization counts and positive-label
      rates, so that the fairness estimates for thin cells are read with the
      appropriate caution.

  E3  Deployed-model sensitivity. The audit conclusions (discrimination,
      calibration, equalized-odds/demographic-parity gaps, feasible-recourse
      rate) recomputed for RandomForest (deployed) versus CatBoost, to check
      they are not an artefact of one model choice.

Outputs:
  outputs/tables/error_analysis.csv/.tex
  outputs/tables/subgroup_sizes.csv/.tex
  outputs/tables/deployed_sensitivity.csv/.tex
  outputs/figures/fig_error_shap.png
"""
from __future__ import annotations
import glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from fairlearn.metrics import (equalized_odds_difference,
                               demographic_parity_difference)

import config as C
from utils import save_json, df_to_latex, set_plot_style
import data_module as D
from s12_calibration import ece
from s14_population_recourse import run_population

C.set_seed()
set_plot_style()
THRESH = 0.5
APT = ["Quant", "English", "Logical", "ComputerProgramming", "collegeGPA"]


def shap_pos(pipe, Xsub):
    """Positive-class SHAP values for the rows in Xsub, with feature names."""
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    Xt = prep.transform(Xsub)
    feats = list(prep.get_feature_names_out())
    sv = shap.TreeExplainer(clf).shap_values(Xt)
    sv = np.asarray(sv)
    if sv.ndim == 3:                                  # (n, feat, class)
        sv = sv[:, :, 1] if sv.shape[2] == 2 else sv[:, :, -1]
    return sv, feats


def main():
    df = D.load_frame()
    X, y, numeric, categorical = D.get_Xy(df)
    g = df[C.SENSITIVE].values
    sp = D.get_splits(df); tr, te = np.array(sp["train"]), np.array(sp["test"])
    pipe = joblib.load(glob.glob(f"{C.MODEL_DIR}/system1_best_*.joblib")[0])
    Xte, yte, gte = X.iloc[te], y[te], g[te]
    prob = pipe.predict_proba(Xte)[:, 1]
    yhat = (prob >= THRESH).astype(int)

    # ---- E1: error analysis ----------------------------------------------
    kind = np.where((yhat == 1) & (yte == 1), "TP",
            np.where((yhat == 0) & (yte == 0), "TN",
             np.where((yhat == 1) & (yte == 0), "FP", "FN")))
    rows = []
    for k in ["TP", "TN", "FP", "FN"]:
        m = kind == k
        rows.append({
            "Outcome": k, "n": int(m.sum()),
            "% of test": round(100 * m.mean(), 1),
            "% female": round(100 * (gte[m] == "female").mean(), 1),
            "Mean p": round(float(prob[m].mean()), 3),
            "Quant": round(float(Xte["Quant"][m].mean()), 0),
            "English": round(float(Xte["English"][m].mean()), 0),
            "GPA": round(float(Xte["collegeGPA"][m].mean()), 1),
        })
    err = pd.DataFrame(rows)
    err.to_csv(f"{C.TAB_DIR}/error_analysis.csv", index=False)
    df_to_latex(err, f"{C.TAB_DIR}/error_analysis.tex",
                "Error analysis of the deployed model on the held-out test set "
                "at the 0.5 threshold: group composition, aptitude means, and "
                "mean predicted probability for correct (TP, TN) and incorrect "
                "(FP, FN) decisions. False decisions cluster near the threshold, "
                "where automated scores are least reliable and human review is "
                "most warranted.", "tab:error")
    print("E1 error analysis:\n", err.to_string(index=False))

    # SHAP: mean |SHAP| over FP vs FN
    fp_m, fn_m = kind == "FP", kind == "FN"
    sv_fp, feats = shap_pos(pipe, Xte[fp_m])
    sv_fn, _ = shap_pos(pipe, Xte[fn_m])
    imp = (pd.DataFrame({"feature": feats,
                         "FP": np.abs(sv_fp).mean(0),
                         "FN": np.abs(sv_fn).mean(0)})
           .assign(tot=lambda d: d.FP + d.FN)
           .sort_values("tot", ascending=False).head(8).iloc[::-1])
    yy = np.arange(len(imp))
    plt.figure(figsize=(6.2, 4.2))
    plt.barh(yy - 0.2, imp.FP, height=0.4, color="#e45756", label="false positives")
    plt.barh(yy + 0.2, imp.FN, height=0.4, color="#4c78a8", label="false negatives")
    plt.yticks(yy, [f.replace("num__", "").replace("cat__", "") for f in imp.feature],
               fontsize=8)
    plt.xlabel("Mean |SHAP| (positive class)"); plt.legend(fontsize=8)
    plt.title("Feature attribution in model errors (held-out test)")
    plt.savefig(f"{C.FIG_DIR}/fig_error_shap.png", bbox_inches="tight", dpi=300)
    plt.close()

    # ---- E2: subgroup sample sizes ---------------------------------------
    spec = df.loc[te, "Specialization"].values
    top = pd.Series(spec).value_counts().head(6).index.tolist()
    srows = []
    for s in top:
        for grp in ["male", "female"]:
            m = (spec == s) & (gte == grp)
            srows.append({"Specialization": s[:34], "Gender": grp,
                          "n": int(m.sum()),
                          "Positive rate": round(float(yte[m].mean()), 3)
                          if m.sum() else np.nan,
                          "small cell (<30)": "yes" if 0 < m.sum() < 30 else ""})
    sub = pd.DataFrame(srows)
    sub.to_csv(f"{C.TAB_DIR}/subgroup_sizes.csv", index=False)
    df_to_latex(sub, f"{C.TAB_DIR}/subgroup_sizes.tex",
                "Held-out test sample sizes and positive-label rates by gender "
                "and specialization (six largest specializations). Cells with "
                "fewer than 30 candidates are flagged; subgroup fairness "
                "estimates for these cells should be read with caution.",
                "tab:subgroup")
    print("\nE2 subgroup sizes:\n", sub.to_string(index=False))

    # ---- E3: deployed-model sensitivity (RF vs CatBoost) -----------------
    def metrics(name, p, model_pipe):
        yh = (p >= THRESH).astype(int)
        pc = run_population(model_pipe, X, df, te, name)   # recourse over rejects
        return {"Model": name,
                "AUROC": round(roc_auc_score(yte, p), 3),
                "Brier": round(brier_score_loss(yte, p), 3),
                "ECE": round(ece(yte, p), 3),
                "EO diff": round(equalized_odds_difference(
                    yte, yh, sensitive_features=gte), 3),
                "DP diff": round(demographic_parity_difference(
                    yte, yh, sensitive_features=gte), 3),
                "Feasible recourse": round(float(pc.feasible.mean()), 3)}

    cat = Pipeline([("prep", D.build_preprocessor(numeric, categorical)),
                    ("clf", CatBoostClassifier(iterations=500, learning_rate=0.05,
                                               depth=5, verbose=0,
                                               random_seed=C.SEED))])
    cat.fit(X.iloc[tr], y[tr])
    sens = pd.DataFrame([
        metrics("RandomForest (deployed)", prob, pipe),
        metrics("CatBoost", cat.predict_proba(Xte)[:, 1], cat),
    ])
    sens.to_csv(f"{C.TAB_DIR}/deployed_sensitivity.csv", index=False)
    df_to_latex(sens, f"{C.TAB_DIR}/deployed_sensitivity.tex",
                "Sensitivity of the audit conclusions to the deployed model: "
                "discrimination, calibration, fairness gaps, and feasible-recourse "
                "rate on the held-out test set for RandomForest (deployed) versus "
                "CatBoost. The two models give the same qualitative picture.",
                "tab:sensitivity")
    print("\nE3 deployed-model sensitivity:\n", sens.to_string(index=False))
    save_json({"error": err.to_dict("records"),
               "sensitivity": sens.to_dict("records")},
              f"{C.MET_DIR}/error_subgroup_sensitivity.json")
    print("\nStage 18 complete.")


if __name__ == "__main__":
    main()
