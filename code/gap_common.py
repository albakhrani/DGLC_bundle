"""
Idea 1 - Gender Pay Gap Decomposition: shared data layer & configuration.

This sub-pipeline answers: *among graduates with comparable test scores,
grades, college tier and specialization, how large is the residual gender
salary gap that qualifications cannot explain, and which features drive it?*

It reuses the SAME cleaned rows as the main study (outputs/processed*.parquet)
so the two papers never silently diverge.

Outputs land under outputs/gap/ to keep them separate from the main study.

Design decisions (documented for the paper's reproducibility section):
  * Outcome     : log1p(Salary)  (multiplicative wage effects; standard in the
                  wage-gap literature).
  * Group       : Gender, with MALE as the reference (advantaged) group, so the
                  decomposition reads as "female shortfall".
  * Outliers    : Salary winsorized to the [P1, P99] range BEFORE the log to
                  tame the Rs.35k / Rs.4M extremes flagged in the EDA.
  * Endowments  : interpretable, pre-labour-market "merit" covariates only
                  (academics, AMCAT cognitive scores, college tier, broad
                  specialization). Post-hiring fields stay quarantined.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

import config as C

# --------------------------------------------------------------------------- #
# Paths (isolated from the main study's outputs)
# --------------------------------------------------------------------------- #
GAP_DIR     = os.path.join(C.OUT, "gap")
GAP_FIG     = os.path.join(GAP_DIR, "figures")
GAP_TAB     = os.path.join(GAP_DIR, "tables")
GAP_MET     = os.path.join(GAP_DIR, "metrics")
GAP_TABLE   = os.path.join(GAP_DIR, "gap_table.parquet")
for _d in (GAP_DIR, GAP_FIG, GAP_TAB, GAP_MET):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------- #
# Modelling constants
# --------------------------------------------------------------------------- #
GROUP_COL      = "Gender"
REFERENCE_GROUP = "male"        # advantaged reference; gap = ref - other
OTHER_GROUP     = "female"
OUTCOME        = "log_salary_w" # winsorized log salary
WINSOR_P       = 0.01           # trim 1% each tail before logging
TOP_K_SPEC     = 8             # collapse rare specializations to "other"

# Continuous "merit" endowments (pre-labour-market, interpretable).
NUMERIC_COV = [
    "10percentage", "12percentage", "collegeGPA",       # academics
    "English", "Logical", "Quant",                       # AMCAT cognitive
    "age_at_grad",                                        # experience proxy
]
# Categorical endowments.
CATEG_COV = ["CollegeTier", "spec_group"]

# ===========================================================================
#  DECISION POINT (see gap01_oaxaca.py): the reference coefficient structure
#  used to define the "no-discrimination" counterfactual wage in the TWOFOLD
#  Oaxaca-Blinder decomposition. Valid choices: "male", "female", "pooled",
#  "neumark" (pooled regression incl. a group dummy, Neumark 1988).
#  This single choice can move the "unexplained" share materially, which is
#  why it is surfaced as a named constant rather than buried in code.
# ===========================================================================
REFERENCE_STRUCTURE = "neumark"


def _winsorize(s: pd.Series, p: float) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def build_gap_table(verbose: bool = True) -> pd.DataFrame:
    """Assemble the wage-gap modelling table from the cleaned study outputs."""
    model = pd.read_parquet(C.PROC_PARQUET).reset_index(drop=True)
    aux   = pd.read_parquet(
        C.PROC_PARQUET.replace("processed", "processed_aux")
    ).reset_index(drop=True)

    if len(model) != len(aux):
        raise RuntimeError("processed and processed_aux are not row-aligned; "
                           "re-run s01_data_prep.py")

    df = model.copy()
    df["Salary"] = aux["Salary"].values

    # ---- outcome: winsorized log salary -----------------------------------
    sal_w = _winsorize(df["Salary"].astype(float), WINSOR_P)
    df[OUTCOME] = np.log1p(sal_w)

    # ---- group ------------------------------------------------------------
    df[GROUP_COL] = df[GROUP_COL].astype(str).str.lower().str.strip()
    df = df[df[GROUP_COL].isin([REFERENCE_GROUP, OTHER_GROUP])].copy()

    # ---- collapse rare specializations to a stable, low-dimension factor ---
    if "Specialization" in df.columns:
        spec = df["Specialization"].astype(str).str.lower().str.strip()
    else:
        spec = pd.Series("unknown", index=df.index)
    top = spec.value_counts().head(TOP_K_SPEC).index
    df["spec_group"] = np.where(spec.isin(top), spec, "other")

    # ---- keep only the columns the decomposition needs --------------------
    cols = [OUTCOME, GROUP_COL] + NUMERIC_COV + CATEG_COV
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()

    # impute numeric covariates by group-median (keeps groups comparable),
    # categorical by global mode.
    for c in NUMERIC_COV:
        if c in out.columns:
            out[c] = out.groupby(GROUP_COL)[c].transform(
                lambda s: s.fillna(s.median()))
            out[c] = out[c].fillna(out[c].median())
    for c in CATEG_COV:
        if c in out.columns:
            out[c] = out[c].astype(str).fillna(out[c].mode().iloc[0])

    out = out.dropna(subset=[OUTCOME]).reset_index(drop=True)
    out.to_parquet(GAP_TABLE, index=False)

    if verbose:
        n = len(out)
        n_ref = int((out[GROUP_COL] == REFERENCE_GROUP).sum())
        n_oth = int((out[GROUP_COL] == OTHER_GROUP).sum())
        raw_gap = (out.loc[out[GROUP_COL] == REFERENCE_GROUP, OUTCOME].mean()
                   - out.loc[out[GROUP_COL] == OTHER_GROUP, OUTCOME].mean())
        print(f"[gap_common] n={n}  {REFERENCE_GROUP}={n_ref}  {OTHER_GROUP}={n_oth}")
        print(f"[gap_common] raw log-salary gap ({REFERENCE_GROUP}-{OTHER_GROUP}) "
              f"= {raw_gap:+.4f}  (~{(np.expm1(raw_gap))*100:+.1f}% in INR)")
        print(f"[gap_common] covariates: {NUMERIC_COV + CATEG_COV}")
        print(f"[gap_common] saved -> {GAP_TABLE}")
    return out


def load_gap_table() -> pd.DataFrame:
    if not os.path.exists(GAP_TABLE):
        return build_gap_table(verbose=False)
    return pd.read_parquet(GAP_TABLE)


if __name__ == "__main__":
    C.set_seed()
    build_gap_table()
