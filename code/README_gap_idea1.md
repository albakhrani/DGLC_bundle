# Idea 1 — Gender Pay Gap Decomposition (AMEO 2015)

> *Among graduates with comparable test scores, grades, college tier and
> specialization, how large is the residual gender salary gap that
> qualifications cannot explain — and which features drive it?*

A self-contained sub-pipeline that reuses the main study's cleaned data
(`outputs/processed*.parquet`) and writes everything under `outputs/gap/`.

## Run
```bash
cd code
./run_gap.sh            # or run gap_common.py -> gap01..gap04 in order
```
Prerequisite: `s01_data_prep.py` must have produced `processed.parquet` and
`processed_aux.parquet` (already present in `outputs/`).

## Stages
| File | What it does | Key outputs |
|------|--------------|-------------|
| `gap_common.py` | Assembles the wage-gap table: winsorized `log_salary`, gender group, interpretable merit covariates. Holds the analysis config. | `outputs/gap/gap_table.parquet` |
| `gap01_oaxaca.py` | Classical Oaxaca–Blinder threefold + twofold decomposition with a 4-way reference-structure sensitivity table and bootstrap CIs. | `oaxaca_*.csv`, `oaxaca.json`, `oaxaca_decomposition.png` |
| `gap02_ml_counterfactual.py` | Gradient-boosted log-salary model + SHAP attribution + ceteris-paribus gender-flip counterfactual. | `ml_counterfactual.json`, `shap_*.png`, `counterfactual_gender_flip.png` |
| `gap03_fairness_audit.py` | Fairlearn audit of an above-median-salary classifier (gender withheld): disparate impact, equalized odds, per-group TPR/FPR. | `fairness_by_group.csv`, `fairness.json`, `fairness_rates.png` |
| `gap04_compile.py` | Merges all three arms into one master table + headline figure. | `MASTER_summary.csv/.tex`, `HEADLINE_residual_gap.png` |

## The one decision you own
`gap_common.REFERENCE_STRUCTURE` (`"male" | "female" | "pooled" | "neumark"`)
sets the non-discriminatory wage structure for the **twofold** split. It
materially affects the "unexplained" share, so `gap01` always prints a 4-way
sensitivity table (`oaxaca_reference_sensitivity.csv`) regardless of the
headline choice. Default: `"neumark"` (pooled regression with a group dummy —
the most common modern default).

## Headline result (current data)
Both independent methods agree on a **~6–7% qualification-adjusted female
penalty**, even though the *raw* gap is only ~3.6% — because women in this
sample are on average **better qualified**, so the explained component is
negative (their characteristics predict they should out-earn men). The
fairness audit is consistent: a merit-only classifier *selects women for
high-salary at a higher rate* (DI ratio 0.86, passes the 80% rule), yet
realized wages penalise them — the contrast is the paper's core narrative.

All estimates are observational; frame as association under unconfoundedness,
not proof of causation.
