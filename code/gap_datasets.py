"""
Idea 1 - dataset adapters. Each loader returns a canonical spec so the SAME
MRR + DGLC core (gap_core.py) runs unchanged on every dataset:

    {name, df, outcome, group, ref, oth, num, cat}

where `outcome` is a continuous winsorized log-wage column, `group` holds
'male'/'female', and num/cat are pre-labour-market merit covariates.

Datasets
  ameo        : AMEO 2015 (primary)            -> reuses gap_common.load_gap_table()
  campus      : Kaggle Campus Recruitment      -> <ROOT>/campus_recruitment.csv (placed only)
  acs         : folktables ACSIncome (one US state) -> auto-download via folktables (census.gov)
  acs_openml  : ACSIncome via fairlearn/OpenML -> avoids census.gov (use if Census is blocked)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import config as C
import gap_common as G

WINSOR_P = 0.01

def _winsor_log(s):
    s = pd.Series(s).astype(float)
    lo, hi = s.quantile(WINSOR_P), s.quantile(1 - WINSOR_P)
    return np.log1p(s.clip(lo, hi))


# --------------------------------------------------------------------------- #
def load_ameo():
    df = G.load_gap_table()
    return dict(name="AMEO 2015 (India, eng. grads)", df=df, outcome=G.OUTCOME,
                group=G.GROUP_COL, ref="male", oth="female",
                num=list(G.NUMERIC_COV), cat=list(G.CATEG_COV))


# --------------------------------------------------------------------------- #
def load_campus(path=None):
    path = path or os.path.join(C.ROOT, "campus_recruitment.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Campus Recruitment CSV not found at {path}. Download from Kaggle "
            "(benroshan/factors-affecting-campus-placement) and save it there.")
    d = pd.read_csv(path)
    d.columns = [c.strip().lower() for c in d.columns]
    d["gender"] = (d["gender"].astype(str).str.upper().str.strip()
                   .map({"M": "male", "F": "female"}))
    d = d[d["salary"].notna()].copy()                  # placed students only
    d["log_salary_w"] = _winsor_log(d["salary"])
    num = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
    cat = [c for c in ["hsc_s", "degree_t", "specialisation", "workex"] if c in d.columns]
    d = d.dropna(subset=["log_salary_w", "gender"]).reset_index(drop=True)
    return dict(name="Campus Recruitment (India, MBA)", df=d, outcome="log_salary_w",
                group="gender", ref="male", oth="female", num=num, cat=cat)


# --------------------------------------------------------------------------- #
def load_acs(state="CA", year="2018", max_rows=None):
    """ACSIncome for one US state via folktables (downloads from census.gov).
    Continuous PINCP -> log wage; SEX -> group. Merit covariates kept PRE-MARKET
    (age, education) by default; occupation / hours are deliberately excluded
    from the main spec (they are channels of sorting). Use load_acs_openml if
    census.gov is blocked on your network.
    """
    from folktables import ACSDataSource, adult_filter
    ds = ACSDataSource(survey_year=year, horizon="1-Year", survey="person")
    raw = ds.get_data(states=[state], download=True)
    raw = adult_filter(raw)                            # age>16, income>=100, hrs>=1
    d = raw[["PINCP", "SEX", "AGEP", "SCHL"]].dropna().copy()
    d = d[d["PINCP"] > 0]
    if max_rows and len(d) > max_rows:
        d = d.sample(max_rows, random_state=C.SEED)
    d["gender"] = d["SEX"].map({1: "male", 2: "female"})
    d["AGEP"] = d["AGEP"].astype(float)
    d["SCHL"] = d["SCHL"].astype(float)                # ordinal education code
    d["log_salary_w"] = _winsor_log(d["PINCP"])
    d = d.dropna(subset=["gender", "log_salary_w"]).reset_index(drop=True)
    return dict(name=f"ACSIncome {state} {year} (US)", df=d, outcome="log_salary_w",
                group="gender", ref="male", oth="female",
                num=["AGEP", "SCHL"], cat=[])


# --------------------------------------------------------------------------- #
def load_acs_openml(max_rows=None):
    """ACSIncome via fairlearn/OpenML (avoids census.gov downloads).
    National sample (all states, 2018). Same canonical schema as load_acs:
    continuous PINCP -> log wage; SEX -> group; pre-market merit covariates
    (age, education). Use this when census.gov is blocked by VPN/firewall.
    """
    from fairlearn.datasets import fetch_acs_income
    data = fetch_acs_income(as_frame=True)
    df = data.frame.copy()
    df = df[df["PINCP"] > 0].dropna(subset=["PINCP", "SEX", "AGEP", "SCHL"]).copy()
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=C.SEED)
    df["gender"] = df["SEX"].map({1.0: "male", 2.0: "female", 1: "male", 2: "female"})
    df["AGEP"] = df["AGEP"].astype(float)
    df["SCHL"] = df["SCHL"].astype(float)              # ordinal education code
    df["log_salary_w"] = _winsor_log(df["PINCP"])
    df = df.dropna(subset=["gender", "log_salary_w"]).reset_index(drop=True)
    return dict(name="ACSIncome (US, OpenML)", df=df, outcome="log_salary_w",
                group="gender", ref="male", oth="female",
                num=["AGEP", "SCHL"], cat=[])


LOADERS = {"ameo": load_ameo, "campus": load_campus,
           "acs": load_acs, "acs_openml": load_acs_openml}

def load(name, **kw):
    if name not in LOADERS:
        raise ValueError(f"unknown dataset '{name}'; choose from {list(LOADERS)}")
    return LOADERS[name](**kw) if kw else LOADERS[name]()