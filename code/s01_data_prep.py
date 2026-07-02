"""
Stage 1 - Data preparation and cleaning.

Produces:
  outputs/processed.parquet / .csv   cleaned analysis table
  outputs/tables/data_dictionary.csv variable dictionary
  outputs/metrics/data_prep_report.json  cleaning log + target balance

Design decisions (documented for the paper's reproducibility section):
  * Target: high_salary = 1[ Salary > median(Salary) ]  (a balanced,
    threshold-justified proxy for a strong early-career earnings outcome).
  * Leakage control: Designation, JobCity, DOJ, DOL and Salary are removed
    from the predictor set (assigned at/after hiring). Designation is retained
    in a separate text column ONLY for the descriptive LDA analysis.
  * The AMCAT domain modules use -1 as a 'module-not-attempted' sentinel; we
    convert -1 -> NaN and add explicit missingness indicators rather than
    treating non-attempts as zero scores.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config as C
from utils import save_json

C.set_seed()


def main() -> None:
    df = pd.read_excel(C.RAW_XLSX)
    report = {"n_raw_rows": int(df.shape[0]), "n_raw_cols": int(df.shape[1])}

    # ---- drop pure identifier / index columns -----------------------------
    drop_ids = [c for c in C.ID_COLS if c in df.columns]
    df = df.drop(columns=drop_ids)
    report["dropped_id_cols"] = drop_ids

    # ---- target ------------------------------------------------------------
    df = df[df["Salary"] > 0].copy()
    sal_median = float(df["Salary"].median())
    df[C.TARGET] = (df["Salary"] > sal_median).astype(int)
    df["log_salary"] = np.log1p(df["Salary"])      # for the regression robustness task
    report["salary_median_INR"] = sal_median
    report["target_balance"] = df[C.TARGET].value_counts(normalize=True).round(4).to_dict()

    # ---- keep Designation text for LDA (cleaned), then quarantine leakage ---
    df["designation_text"] = (
        df["Designation"].astype(str).str.lower().str.strip()
        .str.replace(r"[^a-z\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True)
    )

    # ---- derive graduate age at graduation --------------------------------
    if "DOB" in df.columns and "GraduationYear" in df.columns:
        dob_year = pd.to_datetime(df["DOB"], errors="coerce").dt.year
        gy = pd.to_numeric(df["GraduationYear"], errors="coerce")
        age = gy - dob_year
        df["age_at_grad"] = age.where((age > 16) & (age < 40))

    # ---- AMCAT domain-module sentinels (-1) -> NaN + missingness flags ------
    miss_flags = []
    for col in C.DOMAIN_MODULES:
        if col in df.columns:
            flag = f"{col}_missing"
            df[flag] = (df[col] == -1).astype(int)
            df[col] = df[col].replace(-1, np.nan)
            miss_flags.append(flag)
    report["domain_module_missing_rate"] = {
        c: round(float((df[f"{c}_missing"]).mean()), 4)
        for c in C.DOMAIN_MODULES if f"{c}_missing" in df.columns
    }

    # ---- tidy categoricals -------------------------------------------------
    df["Gender"] = df["Gender"].astype(str).str.lower().str.strip().map(
        {"m": "male", "f": "female"}).fillna(df["Gender"])
    for c in ["Degree", "Specialization", "CollegeState", "10board", "12board"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().str.strip()

    # GraduationYear sometimes 0 -> NaN
    df["GraduationYear"] = pd.to_numeric(df["GraduationYear"], errors="coerce")
    df.loc[df["GraduationYear"] < 2000, "GraduationYear"] = np.nan

    # ---- quarantine leakage columns from the modelling frame --------------
    leak_present = [c for c in C.LEAKAGE_COLS if c in df.columns]
    df_model = df.drop(columns=leak_present)
    report["leakage_cols_removed_from_predictors"] = leak_present

    # ---- persist -----------------------------------------------------------
    # keep designation_text + Salary/log_salary as auxiliary (non-predictor)
    df_model.to_parquet(C.PROC_PARQUET, index=False)
    df_model.to_csv(C.PROC_CSV, index=False)

    # also persist a salary-carrying copy for descriptive / regression use
    aux = df[["Salary", "log_salary", C.TARGET, "Gender", "designation_text"]].copy()
    aux.to_parquet(C.PROC_PARQUET.replace("processed", "processed_aux"), index=False)

    # ---- data dictionary ---------------------------------------------------
    dd = pd.DataFrame({
        "column": df_model.columns,
        "dtype": [str(t) for t in df_model.dtypes],
        "n_missing": df_model.isna().sum().values,
        "pct_missing": (df_model.isna().mean().values * 100).round(2),
        "n_unique": [df_model[c].nunique() for c in df_model.columns],
    })
    dd.to_csv(f"{C.TAB_DIR}/data_dictionary.csv", index=False)

    report["n_model_rows"] = int(df_model.shape[0])
    report["n_model_cols"] = int(df_model.shape[1])
    save_json(report, f"{C.MET_DIR}/data_prep_report.json")

    print("Rows (modelling):", df_model.shape[0], "| Cols:", df_model.shape[1])
    print("Salary median (INR):", sal_median)
    print("Target balance:", report["target_balance"])
    print("Saved:", C.PROC_PARQUET)


if __name__ == "__main__":
    main()
