"""
Idea 1 - Stage 10 (sensitivity): covariate-partition robustness of the
Merit-Reward Reversal (Concern 1).

The Oaxaca decomposition, the GBM counterfactual, and the MRR diagnostic all
condition on ONE merit-covariate partition. Two of those covariates are
contestable as "merit": CollegeTier (tracks socioeconomic background) and
spec_group (specialization sorting is gendered). This script re-runs the MRR
diagnostic (and its Oaxaca residual) under three partitions to test whether the
reversal's sign and label-resident verdict survive dropping them.

  P0 (canonical) : num = NUMERIC_COV,  cat = [CollegeTier, spec_group]
  P1 (drop tier) : num = NUMERIC_COV,  cat = [spec_group]
  P2 (drop tier+spec): num = NUMERIC_COV, cat = []

Everything else identical: same AMEO rows/spec, same gender-withholding, same
XGBoost estimator, same out-of-fold protocol, same SEED, same 2000-sample
bootstrap. Reuses gap_core.run_mrr unchanged (the canonical computation), so P0
must reproduce the canonical headline exactly.

Read-only w.r.t. the canonical pipeline: writes ONLY a new file
  outputs/gap/tables/mrr_partition_sensitivity.csv
and prints the table. No existing output is overwritten.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config as C
import gap_common as G
import gap_core as core
import gap_datasets as D
import gap02_ml_counterfactual as CF   # reuse canonical estimator + flip logic

C.set_seed()

# canonical reference (CANONICAL_NUMBERS.md section 1) that P0 must reproduce
CANON = dict(delta_label=-0.03561, delta_merit=+0.06147, R=+0.09708,
            ci_lo=+0.05730, ci_hi=+0.13862)
CANON_CF = -0.06254294514656067   # ml_counterfactual.json (P0 must reproduce)


# --------------------------------------------------------------------------- #
# GBM gender-flip counterfactual under a chosen categorical partition.
# Mirrors gap02_ml_counterfactual._encode/_model exactly; for the full CATEG_COV
# it is byte-identical to the canonical arm, so P0 reproduces the stored value.
# --------------------------------------------------------------------------- #
def _encode_partition(df, cat):
    num = [c for c in G.NUMERIC_COV if c in df.columns]
    X = df[num].astype(float).copy()
    X[CF.IS_FEMALE] = (df[G.GROUP_COL] == G.OTHER_GROUP).astype(int).values
    cat = [c for c in cat if c in df.columns]
    if cat:
        X = pd.concat([X, pd.get_dummies(df[cat].astype(str),
                                         drop_first=True, dtype=float)], axis=1)
    y = df[G.OUTCOME].astype(float).values
    return X, y

def counterfactual(df, cat):
    """Ceteris-paribus mean female-minus-male effect on predicted log-wage
    (full-data fit, exactly as gap02)."""
    X, y = _encode_partition(df, cat)
    model = CF._model().fit(X, y)
    X_flip = X.copy(); X_flip[CF.IS_FEMALE] = 1 - X_flip[CF.IS_FEMALE]
    pred_obs = model.predict(X); pred_cf = model.predict(X_flip)
    female_mask = X[CF.IS_FEMALE].values == 1
    delta = np.where(female_mask, pred_obs - pred_cf, pred_cf - pred_obs)
    return float(np.mean(delta))


def main():
    spec = D.load_ameo()
    df, outcome, group, ref, oth = (spec["df"], spec["outcome"], spec["group"],
                                    spec["ref"], spec["oth"])
    num = list(spec["num"])                       # NUMERIC_COV (unchanged in all)
    full_cat = list(spec["cat"])                  # [CollegeTier, spec_group]

    partitions = [
        ("P0_canonical", full_cat, "none"),
        ("P1_drop_tier", [c for c in full_cat if c != "CollegeTier"], "CollegeTier"),
        ("P2_drop_tier_spec", [c for c in full_cat if c not in ("CollegeTier", "spec_group")],
         "CollegeTier + spec_group"),
    ]

    rows = []
    for name, cat, dropped in partitions:
        r = core.run_mrr(df, outcome, group, ref, oth, num, cat)
        cf = counterfactual(df, cat)                          # third arm
        lo, hi = r["reversal_magnitude_ci"]
        holds = bool(r["merit_reward_reversal"] and lo > 0)   # sign right AND CI excludes 0
        unexpl = r["oaxaca_unexplained"]
        all_three = bool(holds and unexpl > 0 and cf < 0)     # full label-resident verdict
        rows.append(dict(
            partition=name, dropped=dropped, cat=";".join(cat) if cat else "(numeric only)",
            delta_label=round(r["delta_label"], 5),
            delta_merit=round(r["delta_merit"], 5),
            R=round(r["reversal_magnitude"], 5),
            R_ci_lo=round(lo, 5), R_ci_hi=round(hi, 5),
            oaxaca_unexpl=round(unexpl, 5),
            counterfactual=round(cf, 5),
            reversal_sign_ok=bool(r["merit_reward_reversal"]),
            ci_excludes_zero=bool(lo > 0),
            reversal_holds=("Y" if holds else "N"),
            counterfactual_negative=("Y" if cf < 0 else "N"),
            all_three_label_resident=("Y" if all_three else "N")))
    out = pd.DataFrame(rows)
    out.to_csv(f"{G.GAP_TAB}/mrr_partition_sensitivity.csv", index=False)

    # ---- P0 must reproduce canonical (reversal + counterfactual), else STOP --
    p0 = rows[0]
    ok = (abs(p0["delta_label"] - CANON["delta_label"]) < 5e-4 and
          abs(p0["delta_merit"] - CANON["delta_merit"]) < 5e-4 and
          abs(p0["R"] - CANON["R"]) < 5e-4 and
          abs(p0["R_ci_lo"] - CANON["ci_lo"]) < 5e-4 and
          abs(p0["R_ci_hi"] - CANON["ci_hi"]) < 5e-4 and
          abs(p0["counterfactual"] - round(CANON_CF, 5)) < 5e-4)

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("==== Three-arm covariate-partition sensitivity (AMEO) ====")
    print(out[["partition", "dropped", "delta_merit", "R", "R_ci_lo", "R_ci_hi",
               "oaxaca_unexpl", "counterfactual", "reversal_holds",
               "counterfactual_negative", "all_three_label_resident"]].to_string(index=False))
    print()
    if ok:
        print(f"P0 reproduces canonical: delta_label={p0['delta_label']}, "
              f"delta_merit={p0['delta_merit']}, R={p0['R']} "
              f"[{p0['R_ci_lo']},{p0['R_ci_hi']}], counterfactual={p0['counterfactual']} "
              f"(canonical {round(CANON_CF,5)})  -> MATCHES CANONICAL.")
    else:
        print("*** STOP: P0 does NOT reproduce canonical "
              f"(got dl={p0['delta_label']}, dm={p0['delta_merit']}, R={p0['R']} "
              f"[{p0['R_ci_lo']},{p0['R_ci_hi']}]; expected "
              f"dl={CANON['delta_label']}, dm={CANON['delta_merit']}, "
              f"R={CANON['R']} [{CANON['ci_lo']},{CANON['ci_hi']}]). "
              "Sensitivity harness differs from canonical; do NOT trust P1/P2. ***")
    print(f"\nsaved -> {G.GAP_TAB}/mrr_partition_sensitivity.csv")


if __name__ == "__main__":
    main()
