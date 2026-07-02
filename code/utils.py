"""
Shared utilities: plotting style, metric computation, bootstrap CIs,
table-to-text serialisation, and small I/O helpers.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    balanced_accuracy_score, matthews_corrcoef, brier_score_loss,
    precision_score, recall_score,
)

# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def set_plot_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
        "savefig.bbox": "tight",
    })

# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Full suite of threshold-free and thresholded classification metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "AUROC":      roc_auc_score(y_true, y_prob),
        "AUPRC":      average_precision_score(y_true, y_prob),
        "Accuracy":   accuracy_score(y_true, y_pred),
        "BalAcc":     balanced_accuracy_score(y_true, y_pred),
        "F1":         f1_score(y_true, y_pred, zero_division=0),
        "Precision":  precision_score(y_true, y_pred, zero_division=0),
        "Recall":     recall_score(y_true, y_pred, zero_division=0),
        "MCC":        matthews_corrcoef(y_true, y_pred),
        "Brier":      brier_score_loss(y_true, y_prob),
    }
    return out


def bootstrap_ci(y_true, y_prob, metric="AUROC", n_boot=2000, seed=42, alpha=0.05):
    """Percentile bootstrap CI for a probability-based metric on a test set."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    fn = {"AUROC": roc_auc_score, "AUPRC": average_precision_score}[metric]
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(fn(y_true[idx], y_prob[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(stats)), float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Table -> text serialisation (for the DeBERTa table-to-text model)
# --------------------------------------------------------------------------- #
def row_to_text(row: pd.Series, feature_cols, pretty: dict | None = None) -> str:
    """
    Serialise one tabular record into a natural-language description, the
    standard 'text template' serialisation used for LLM/transformer models on
    tabular data (Hegselmann et al., 2023, TabLLM).
    """
    pretty = pretty or {}
    parts = []
    for c in feature_cols:
        v = row[c]
        if pd.isna(v):
            continue
        name = pretty.get(c, c)
        if isinstance(v, float):
            v = round(v, 2)
        parts.append(f"{name} is {v}")
    return ". ".join(parts) + "."


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def save_json(obj, path) -> None:
    def _default(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)


def df_to_latex(df: pd.DataFrame, path: str, caption: str, label: str,
                float_fmt: str = "%.3f") -> None:
    """Write a page-fitting LaTeX booktabs table (resizebox-wrapped)."""
    try:
        body = df.to_latex(index=False, float_format=lambda x: float_fmt % x,
                           escape=True,
                           column_format="l" + "r" * (df.shape[1] - 1))
        tex = ("\\begin{table}[t]\n\\centering\\small\n"
               "\\caption{%s}\\label{%s}\n"
               "\\resizebox{\\textwidth}{!}{%%\n%s}\n\\end{table}\n"
               % (caption, label, body))
    except Exception:
        tex = ("\\begin{table}[t]\\centering\\caption{%s}\\label{%s}\n"
               "\\begin{verbatim}\n%s\n\\end{verbatim}\\end{table}\n"
               % (caption, label, df.to_string(index=False)))
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)
