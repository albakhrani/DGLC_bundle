#!/usr/bin/env bash
# Reproduce the full study end-to-end. Run from the code/ directory.
# LightGBM needs OpenMP from torch's bundled libgomp:
export LD_LIBRARY_PATH="$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__)+"/lib")'):$LD_LIBRARY_PATH"
set -e
cd "$(dirname "$0")"

echo "[1/19] Data preparation";            python3 s01_data_prep.py
echo "[2/19] EDA";                         python3 s02_eda.py
echo "[3/19] LDA (designations)";          python3 s03_lda_designation.py
echo "[4/19] System 1 panel (13 models, incl EBM)"; python3 s04_system1_models.py
echo "[5/19] DeBERTa table-to-text";       python3 s05_deberta.py
echo "[6/19] Tabular DL (FT/TabNet/TabTr)"; python3 s09_tabular_dl.py
echo "[7/19] System 2 XAI + fairness";     python3 s06_system2_xai_fairness.py
echo "[8/19] Recourse + intersectional";   python3 s10_recourse_intersectional.py
echo "[9/19] Threshold sensitivity";       python3 s11_threshold_sensitivity.py
echo "[10/19] Calibration + reliability";  python3 s12_calibration.py
echo "[11/19] Feature-group ablation";     python3 s13_ablation.py
echo "[12/19] Hyperparameter tuning";      python3 s08_tuning.py
echo "[13/19] Architecture diagram";       python3 make_architecture.py
echo "[14/19] Population recourse";       python3 s14_population_recourse.py
echo "[15/19] Reviewer analyses";         python3 s15_reviewer_analyses.py
echo "[16/19] Label sensitivity";         python3 s16_label_sensitivity.py
echo "[17/19] Decision-curve analysis";   python3 s17_decision_curve.py
echo "[18/19] Error/subgroup/sensitivity"; python3 s18_error_subgroup_sensitivity.py
echo "[19/19] Compile results";            python3 s07_compile.py
echo "Done. Build the Word file with: python3 build_docx.py"
