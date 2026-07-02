"""
Global configuration for the AMEO-2015 Dual-Process (System 1 / System 2)
explainable-ML study.

All paths are resolved relative to the project root (the folder that also
contains AMEO_2015_train.xlsx), so the pipeline is portable.
"""
from __future__ import annotations
import os

# Deterministic threading: pin BLAS / OpenMP to a single thread so results are
# bit-reproducible run-to-run (must be set BEFORE numpy / XGBoost initialise
# their backends). When running the full pipeline use run_gap.sh, which also
# exports these before the interpreter starts so even pre-config imports see them.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import random
import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_XLSX     = os.path.join(ROOT, "AMEO_2015_train.xlsx")

OUT          = os.path.join(ROOT, "outputs")
FIG_DIR      = os.path.join(OUT, "figures")
TAB_DIR      = os.path.join(OUT, "tables")
MET_DIR      = os.path.join(OUT, "metrics")
MODEL_DIR    = os.path.join(OUT, "models")
PROC_PARQUET = os.path.join(OUT, "processed.parquet")
PROC_CSV     = os.path.join(OUT, "processed.csv")

for _d in (OUT, FIG_DIR, TAB_DIR, MET_DIR, MODEL_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

# --------------------------------------------------------------------------- #
# Modelling constants
# --------------------------------------------------------------------------- #
TARGET           = "high_salary"          # binary: salary above sample median
SENSITIVE        = "Gender"               # protected attribute for fairness audit
TEST_SIZE        = 0.20
N_SPLITS         = 5                       # stratified CV folds
N_REPEATS        = 3                       # repeated CV for variance estimates
POS_LABEL        = 1

# AMCAT "section" scores that use -1 as a "module-not-attempted" sentinel.
DOMAIN_MODULES = [
    "Domain", "ComputerProgramming", "ElectronicsAndSemicon", "ComputerScience",
    "MechanicalEngg", "ElectricalEngg", "TelecomEngg", "CivilEngg",
]

# Columns that are determined *at / after* hiring -> excluded from predictors
# to avoid target leakage. (Designation is reused for the LDA analysis only.)
LEAKAGE_COLS = ["Salary", "DOJ", "DOL", "Designation", "JobCity"]

# Identifier / index columns to drop entirely.
ID_COLS = ["Unnamed: 0", "ID", "CollegeID", "CollegeCityID"]

# Big-Five personality dimensions (kept as-is, already standardised).
BIG_FIVE = ["conscientiousness", "agreeableness", "extraversion",
            "nueroticism", "openess_to_experience"]

PLOT_DPI = 300
