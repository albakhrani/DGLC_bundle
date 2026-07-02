#!/usr/bin/env bash
# Idea 1 - Gender Pay Gap Decomposition + MRR diagnostic + DGLC remedy pipeline.
# Run from the code/ directory.
# Depends on s01_data_prep.py having produced ../outputs/processed*.parquet.
set -e
cd "$(dirname "$0")"

# --- Determinism: pin every threading backend to a single thread BEFORE the
#     interpreter starts, so BLAS (statsmodels OLS) and XGBoost OpenMP give
#     bit-reproducible results run-to-run. Combined with n_jobs=1 in the code. ---
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=42
# Windows consoles default to cp1252 and crash on Unicode (e.g. the U+2212 minus
# in some labels); force UTF-8 stdio so progress prints never abort the run.
export PYTHONIOENCODING=utf-8

# --- Cross-dataset stage (gap08) configuration. acs_openml avoids census.gov
#     and deterministically subsamples 120k rows via random_state=SEED. ---
export GAP_DATASETS="${GAP_DATASETS:-ameo,campus,acs_openml}"
export GAP_ACS_MAXROWS="${GAP_ACS_MAXROWS:-120000}"

# Interpreter: this environment ships a broken `python3` stub; default to `python`.
PY="${PYTHON:-python}"

echo "[0/10] Assemble gender-gap table";          "$PY" gap_common.py
echo "[1/10] Oaxaca-Blinder decomposition";       "$PY" gap01_oaxaca.py
echo "[2/10] ML model + SHAP + counterfactual";   "$PY" gap02_ml_counterfactual.py
echo "[3/10] Group fairness audit";               "$PY" gap03_fairness_audit.py
echo "[4/10] Compile master table + headline";    "$PY" gap04_compile.py
echo "[5/10] Merit-Reward Reversal diagnostic";   "$PY" gap05_diagnostic.py
echo "[6/10] DGLC label correction";              "$PY" gap06_dglc.py
echo "[7/10] Compile diagnostic + remedy table";  "$PY" gap07_compile.py
echo "[8/10] Cross-dataset validation";           "$PY" gap08_cross_dataset.py
echo "[9/10] Label-side baseline head-to-head";   "$PY" gap09_labelside_baseline.py
echo "[10/10] MRR partition sensitivity";         "$PY" gap10_partition_sensitivity.py
echo "Done. See ../outputs/gap (figures, tables, metrics, cross)."
