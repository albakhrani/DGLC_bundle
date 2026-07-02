"""
Shared data access layer.

Guarantees that every model in the study (linear, tree, gradient-boosting,
MLP and the DeBERTa table-to-text transformer) sees the SAME stratified
train/test partition and the SAME target, so the comparison is fair.

Exposes:
  load_frame()                 -> cleaned modelling DataFrame
  get_splits()                 -> dict with train/test indices (cached to disk)
  build_preprocessor()         -> sklearn ColumnTransformer (fit on train only)
  feature_columns()            -> (numeric_cols, categorical_cols)
"""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import config as C

_NON_PREDICTORS = {C.TARGET, "log_salary", "designation_text", "DOB"}
_CATEGORICAL = ["Gender", "Degree", "Specialization", "CollegeState",
                "10board", "12board", "CollegeTier", "CollegeCityTier"]
SPLIT_FILE = os.path.join(C.OUT, "split_indices.json")


def load_frame() -> pd.DataFrame:
    return pd.read_parquet(C.PROC_PARQUET).reset_index(drop=True)


def feature_columns(df: pd.DataFrame):
    predictors = [c for c in df.columns if c not in _NON_PREDICTORS
                  and not pd.api.types.is_datetime64_any_dtype(df[c])]
    categorical = [c for c in _CATEGORICAL if c in predictors]
    numeric = [c for c in predictors
               if c not in categorical and pd.api.types.is_numeric_dtype(df[c])]
    # any leftover object columns -> treat as categorical
    leftover = [c for c in predictors if c not in categorical and c not in numeric]
    categorical += leftover
    return numeric, categorical


def get_splits(df: pd.DataFrame) -> dict:
    """Stratified 80/20 split, cached so DeBERTa reuses identical indices."""
    if os.path.exists(SPLIT_FILE):
        with open(SPLIT_FILE) as f:
            return json.load(f)
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=C.TEST_SIZE, random_state=C.SEED,
                              stratify=df[C.TARGET].values)
    splits = {"train": tr.tolist(), "test": te.tolist()}
    with open(SPLIT_FILE, "w") as f:
        json.dump(splits, f)
    return splits


def build_preprocessor(numeric, categorical) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=10, max_categories=15,
                                 sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical),
    ], remainder="drop", verbose_feature_names_out=True)


def get_Xy(df: pd.DataFrame):
    numeric, categorical = feature_columns(df)
    X = df[numeric + categorical].copy()
    y = df[C.TARGET].astype(int).values
    return X, y, numeric, categorical


if __name__ == "__main__":
    df = load_frame()
    X, y, num, cat = get_Xy(df)
    sp = get_splits(df)
    print(f"n={len(df)}  numeric={len(num)}  categorical={len(cat)}")
    print(f"train={len(sp['train'])}  test={len(sp['test'])}")
    print("categorical cardinalities:",
          {c: int(df[c].nunique()) for c in cat})
