# Locked Method & Evidence — Merit-Reward Reversal + DGLC (AMEO 2015)

## The contribution (two linked, novel parts)

**1. Merit-Reward Reversal (MRR) — a diagnostic.** A bootstrapped test that
classifies a group fairness gap as *label-resident* vs *model-resident*. The
signature: a merit-only model selects the protected group at a **higher** rate
than the realized label rewards them. Triangulated by Oaxaca + counterfactual.
Code: `gap05_diagnostic.py`.

**2. Decomposition-Guided Label Correction (DGLC) — the remedy.** An
interpretable, alpha-tunable correction of the supervision *target* grounded in
the Oaxaca reference (non-discriminatory) wage structure. Self-validating
(re-decomposing the corrected target zeroes the residual it was built from).
Code: `gap06_dglc.py`.

## Locked evidence (reproduced from raw AMEO_2015_train.xlsx)

| Quantity | Value | 95% CI / note |
|---|---|---|
| Realized reward gap (F−M) | −0.036 | [−0.070, −0.000] — women penalised by label |
| Merit selection gap (F−M) | +0.061 | [+0.027, +0.099] — women favoured on merit |
| Reversal magnitude | +0.097 | [+0.057, +0.139] — **label-resident** |
| Oaxaca unexplained (triangulation) | +0.070 log pts | [+0.037, +0.105] |
| Counterfactual gender effect (triangulation) | −0.063 log pts | agrees |
| Women selected — A baseline | 0.470 | reference |
| Women selected — B post-hoc parity | 0.435 | **levels DOWN −0.036** |
| Women selected — C DGLC (ours) | 0.549 | **rewards merit +0.078** |
| Realized AUROC: baseline → DGLC | 0.741 → 0.742 | predictive validity preserved |
| Self-validation: Oaxaca unexplained after DGLC | +0.0000 | collapses from +0.0699 |

DGLC corrected base-rate gap (F−M) stays positive across all four reference
structures (male/female/pooled/neumark: 0.057/0.050/0.038/0.042) — robust.

## Headline narrative

Women are better qualified; a merit model favours them; the realized-wage label
penalises them ~7%. **Enforcing parity against that biased label harms the group
it claims to protect (levels women down); correcting the label at its source
(DGLC) rewards merit at no cost to realized predictive accuracy.**

## Positioning (cite — do not claim as ours)
- Leveling-down pathology of parity constraints: Mittelstadt et al., 2023.
- Label-bias correction by reweighting: Jiang & Nachum, 2020.
- Oaxaca (1973); Blinder (1973); Neumark (1988) — decomposition.
Our novelty = the decomposition-to-ML bridge, the MRR diagnostic, the
alpha-frontier, and the dual (realized vs merit) audit.

## Caveats (state plainly)
Observational (association under unconfoundedness, not causation). Wage-model
CV R²≈0.18 — report the counterfactual as one of three converging arms.
Corrected- vs realized-label metrics are not directly comparable; the win is the
selection-direction contrast + AUROC preservation + self-validation.

## Reproduce
```
cd code && python3 s01_data_prep.py && ./run_gap.sh
```
