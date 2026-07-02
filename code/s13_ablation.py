"""
Stage 13 - Feature-group ablation (System 1) + System-2 capability ablation.

Quantifies the predictive contribution of each feature family by cumulatively
adding groups and refitting a gradient-boosted model on the same split:
  cognitive -> + technical -> + academic -> + personality -> + demographic (full).
The qualitative System-2 ablation (what each deliberative module adds) is reported
in the manuscript text, since SHAP, recourse, and the fairness audit add
capability rather than predictive accuracy.

Outputs:
  outputs/tables/ablation.csv / .tex
  outputs/figures/fig_ablation.png
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

import config as C
from utils import set_plot_style, df_to_latex, bootstrap_ci
import data_module as D

C.set_seed()
set_plot_style()

GROUPS = {
    "cognitive": ["English", "Logical", "Quant"],
    "technical": ["Domain", "ComputerProgramming", "ElectronicsAndSemicon",
                  "ComputerScience", "MechanicalEngg", "ElectricalEngg",
                  "TelecomEngg", "CivilEngg"],
    "academic": ["10percentage", "12percentage", "12graduation", "collegeGPA",
                 "GraduationYear", "CollegeTier", "CollegeCityTier", "Degree",
                 "Specialization", "10board", "12board"],
    "personality": C.BIG_FIVE,
    "demographic": ["Gender", "CollegeState", "age_at_grad"],
}
ORDER = ["cognitive", "technical", "academic", "personality", "demographic"]


def build_prep(cols, df):
    num = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
           and c not in ("Degree", "Specialization", "10board", "12board",
                         "CollegeState", "CollegeTier", "CollegeCityTier", "Gender")]
    cat = [c for c in cols if c in df.columns and c not in num]
    tf = []
    if num:
        tf.append(("num", Pipeline([("i", SimpleImputer(strategy="median")),
                                    ("s", StandardScaler())]), num))
    if cat:
        tf.append(("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                                    ("o", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                          min_frequency=10, max_categories=15,
                                          sparse_output=False))]), cat))
    return ColumnTransformer(tf, remainder="drop")


def main():
    df = D.load_frame()
    y = df[C.TARGET].astype(int).values
    sp = D.get_splits(df); tr, te = np.array(sp["train"]), np.array(sp["test"])

    rows, cols = [], []
    for i, gname in enumerate(ORDER):
        # include missingness indicators alongside the technical modules
        cols = cols + [c for c in GROUPS[gname] if c in df.columns]
        if gname == "technical":
            cols += [c for c in df.columns if c.endswith("_missing")]
        prep = build_prep(cols, df)
        mdl = Pipeline([("prep", prep),
                        ("clf", XGBClassifier(n_estimators=500, learning_rate=0.05,
                                              max_depth=4, subsample=0.85,
                                              colsample_bytree=0.85, reg_lambda=1.5,
                                              eval_metric="auc", n_jobs=1,
                                              random_state=C.SEED))])
        mdl.fit(df.iloc[tr][cols], y[tr])
        prob = mdl.predict_proba(df.iloc[te][cols])[:, 1]
        auc = roc_auc_score(y[te], prob)
        _, lo, hi = bootstrap_ci(y[te], prob, "AUROC", seed=C.SEED)
        rows.append({"features_added": f"+ {gname}",
                     "cumulative_groups": " + ".join(ORDER[:i + 1]),
                     "n_features": len(cols),
                     "test_AUROC": round(auc, 3),
                     "CI95": f"[{lo:.3f}, {hi:.3f}]"})
    tbl = pd.DataFrame(rows)
    tbl["delta"] = tbl["test_AUROC"].diff().round(3).fillna(0.0)
    tbl.to_csv(f"{C.MET_DIR}/ablation.csv".replace("metrics", "tables"), index=False)
    tbl.to_csv(f"{C.TAB_DIR}/ablation.csv", index=False)
    df_to_latex(tbl[["features_added", "n_features", "test_AUROC", "CI95", "delta"]],
                f"{C.TAB_DIR}/ablation.tex",
                "Cumulative feature-group ablation (XGBoost, held-out test AUROC). "
                "Cognitive aptitude alone is already strongly predictive; later "
                "groups add little.", "tab:ablation")
    print(tbl.to_string(index=False))

    plt.figure(figsize=(6.2, 4))
    plt.plot(range(len(tbl)), tbl["test_AUROC"], "o-", color="#1f77b4")
    plt.xticks(range(len(tbl)), tbl["features_added"], rotation=20, ha="right")
    plt.ylabel("Test AUROC"); plt.title("Cumulative feature-group ablation")
    plt.ylim(0.6, 0.8)
    plt.savefig(f"{C.FIG_DIR}/fig_ablation.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("\nStage 13 complete.")


if __name__ == "__main__":
    main()
