"""
Idea 1 - Stage 1: Oaxaca-Blinder gender wage-gap decomposition (classical arm).

Splits the mean log-salary gap into:
  * THREEFOLD : endowments (E) + coefficients (C) + interaction (I)
  * TWOFOLD   : explained (Q) + unexplained (U), under a chosen reference
                wage structure (gap_common.REFERENCE_STRUCTURE).

The "explained" part is what differing qualifications justify; the
"unexplained" part is the residual the data cannot attribute to merit -- the
quantity of policy interest. A four-way sensitivity table (male / female /
pooled / Neumark reference structures) is reported so the headline number is
never an artefact of one modelling choice.

Outputs:
  outputs/gap/tables/oaxaca_aggregate.csv      threefold + twofold + bootstrap CI
  outputs/gap/tables/oaxaca_detailed.csv       per-variable explained/unexplained
  outputs/gap/tables/oaxaca_reference_sensitivity.csv
  outputs/gap/metrics/oaxaca.json
  outputs/gap/figures/oaxaca_decomposition.png
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

import config as C
from utils import set_plot_style, save_json
import gap_common as G

C.set_seed()


# --------------------------------------------------------------------------- #
# Design matrix (consistent columns across both groups)
# --------------------------------------------------------------------------- #
def _design(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X with constant + dummies, y).  Columns are stable across groups
    because dummies are built on the full sample passed in."""
    num = [c for c in G.NUMERIC_COV if c in df.columns]
    cat = [c for c in G.CATEG_COV if c in df.columns]
    X_num = df[num].astype(float)
    X_cat = pd.get_dummies(df[cat].astype(str), drop_first=True, dtype=float)
    X = pd.concat([X_num, X_cat], axis=1)
    X = sm.add_constant(X, has_constant="add")
    y = df[G.OUTCOME].astype(float)
    return X, y


def _ols(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    return sm.OLS(y.values, X.values).fit().params


def _reference_betas(Xr, yr, Xo, yo, X_pool, y_pool, ref_dummy, structure: str):
    """Return the reference coefficient vector b* for the twofold split."""
    if structure == "male":          # advantaged group's structure
        return _ols(Xr, yr)
    if structure == "female":        # disadvantaged group's structure
        return _ols(Xo, yo)
    if structure == "pooled":        # pooled, no group indicator
        return _ols(X_pool, y_pool)
    if structure == "neumark":       # Neumark (1988): pooled incl. group dummy
        Xp = X_pool.copy()
        Xp["_grp"] = ref_dummy.values
        b = _ols(Xp, y_pool)
        return b[:-1]                # drop the group-dummy coef -> b* over X
    raise ValueError(f"unknown reference structure: {structure}")


def _twofold(mean_r, mean_o, b_r, b_o, b_star):
    explained   = float((mean_r - mean_o) @ b_star)
    unexplained = float(mean_r @ (b_r - b_star) + mean_o @ (b_star - b_o))
    return explained, unexplained


def _threefold(mean_r, mean_o, b_r, b_o):
    E = float((mean_r - mean_o) @ b_o)
    Cc = float(mean_o @ (b_r - b_o))
    I = float((mean_r - mean_o) @ (b_r - b_o))
    return E, Cc, I


def _detailed(cols, mean_r, mean_o, b_r, b_o, b_star):
    """Per-coefficient explained/unexplained, aggregated back to variables."""
    expl = (mean_r - mean_o) * b_star
    unex = mean_r * (b_r - b_star) + mean_o * (b_star - b_o)
    rows = []
    for c, e, u in zip(cols, expl, unex):
        var = c.split("_")[0] if c not in ("const",) else "const"
        rows.append((c, var, float(e), float(u)))
    d = pd.DataFrame(rows, columns=["term", "variable", "explained", "unexplained"])
    return d.groupby("variable", as_index=False)[["explained", "unexplained"]].sum()


def _bootstrap_ci(df, structure, n_boot=1000, alpha=0.05, seed=C.SEED):
    rng = np.random.default_rng(seed)
    ref = df[df[G.GROUP_COL] == G.REFERENCE_GROUP].reset_index(drop=True)
    oth = df[df[G.GROUP_COL] == G.OTHER_GROUP].reset_index(drop=True)
    expl_b, unex_b = [], []
    for _ in range(n_boot):
        rs = ref.iloc[rng.integers(0, len(ref), len(ref))]
        os_ = oth.iloc[rng.integers(0, len(oth), len(oth))]
        boot = pd.concat([rs, os_], ignore_index=True)
        try:
            e, u, *_ = _decompose(boot, structure, with_detail=False)
            expl_b.append(e); unex_b.append(u)
        except Exception:
            continue
    q = lambda a: (float(np.quantile(a, alpha/2)), float(np.quantile(a, 1-alpha/2)))
    return {"explained_ci": q(expl_b), "unexplained_ci": q(unex_b)}


def _decompose(df, structure, with_detail=True):
    ref = df[df[G.GROUP_COL] == G.REFERENCE_GROUP]
    oth = df[df[G.GROUP_COL] == G.OTHER_GROUP]
    Xall, yall = _design(df)
    cols = list(Xall.columns)
    ref_dummy = (df[G.GROUP_COL] == G.REFERENCE_GROUP).astype(float)

    Xr, yr = Xall.loc[ref.index], yall.loc[ref.index]
    Xo, yo = Xall.loc[oth.index], yall.loc[oth.index]
    b_r, b_o = _ols(Xr, yr), _ols(Xo, yo)
    b_star = _reference_betas(Xr, yr, Xo, yo, Xall, yall, ref_dummy, structure)

    mean_r = Xr.mean(axis=0).values
    mean_o = Xo.mean(axis=0).values
    explained, unexplained = _twofold(mean_r, mean_o, b_r, b_o, b_star)
    if not with_detail:
        return explained, unexplained
    E, Cc, I = _threefold(mean_r, mean_o, b_r, b_o)
    detail = _detailed(cols, mean_r, mean_o, b_r, b_o, b_star)
    raw = float(yr.mean() - yo.mean())
    return explained, unexplained, raw, (E, Cc, I), detail


def main() -> None:
    set_plot_style()
    df = G.load_gap_table()
    structure = G.REFERENCE_STRUCTURE

    explained, unexplained, raw, (E, Cc, I), detail = _decompose(df, structure)
    ci = _bootstrap_ci(df, structure)

    pct = lambda x: 100 * x / raw if raw else float("nan")
    agg = pd.DataFrame([
        ("raw_gap",            raw,          100.0),
        ("explained",          explained,    pct(explained)),
        ("unexplained",        unexplained,  pct(unexplained)),
        ("threefold_endowments", E,          pct(E)),
        ("threefold_coefficients", Cc,       pct(Cc)),
        ("threefold_interaction",  I,        pct(I)),
    ], columns=["component", "log_points", "pct_of_raw_gap"])
    agg["approx_pct_INR"] = (np.expm1(agg["log_points"]) * 100).round(2)
    agg.to_csv(f"{G.GAP_TAB}/oaxaca_aggregate.csv", index=False)

    detail = detail.sort_values("explained", key=abs, ascending=False)
    detail.to_csv(f"{G.GAP_TAB}/oaxaca_detailed.csv", index=False)

    # ---- four-way reference-structure sensitivity -------------------------
    sens = []
    for s in ["male", "female", "pooled", "neumark"]:
        e, u = _decompose(df, s, with_detail=False)
        sens.append((s, e, u, pct(e), pct(u)))
    sens = pd.DataFrame(sens, columns=["reference_structure", "explained",
                        "unexplained", "explained_pct", "unexplained_pct"])
    sens.to_csv(f"{G.GAP_TAB}/oaxaca_reference_sensitivity.csv", index=False)

    save_json({
        "reference_structure": structure,
        "raw_gap_logpts": raw,
        "explained_logpts": explained,
        "unexplained_logpts": unexplained,
        "explained_pct_of_gap": pct(explained),
        "unexplained_pct_of_gap": pct(unexplained),
        "bootstrap_95ci": ci,
        "threefold": {"endowments": E, "coefficients": Cc, "interaction": I},
    }, f"{G.GAP_MET}/oaxaca.json")

    # ---- figure: stacked explained/unexplained + top driver bars ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.bar(["Explained\n(qualifications)"], [explained], color="#4C8CBF")
    ax1.bar(["Unexplained\n(residual)"], [unexplained], color="#D1495B")
    ax1.axhline(raw, ls="--", c="grey", lw=1, label=f"raw gap = {raw:+.3f}")
    ax1.set_ylabel("log-salary gap (male − female)")
    ax1.set_title(f"Twofold decomposition ({structure} reference)")
    ax1.legend(fontsize=8)
    top = detail.reindex(detail["explained"].abs().sort_values(ascending=True).index).tail(7)
    ax2.barh(top["variable"], top["explained"], color="#4C8CBF", label="explained")
    ax2.barh(top["variable"], top["unexplained"], left=top["explained"],
             color="#D1495B", alpha=.8, label="unexplained")
    ax2.set_title("Per-variable contribution")
    ax2.set_xlabel("log-salary points")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{G.GAP_FIG}/oaxaca_decomposition.png", dpi=C.PLOT_DPI)
    plt.close(fig)

    print(f"[oaxaca] reference structure : {structure}")
    print(f"[oaxaca] raw gap        : {raw:+.4f} log pts")
    print(f"[oaxaca] explained      : {explained:+.4f} ({pct(explained):.1f}% of gap)")
    print(f"[oaxaca] unexplained    : {unexplained:+.4f} ({pct(unexplained):.1f}% of gap)")
    print(f"[oaxaca] 95% CI unexpl. : {ci['unexplained_ci']}")
    print(f"[oaxaca] saved tables/figure under {G.GAP_DIR}")


if __name__ == "__main__":
    main()
