"""
Figure 1 (fig:framework) -- the dual-process (System 1 / System 2) framework.

Publication-quality flat-design schematic: two leakage-controlled inputs feed a
Predictive stage (System 1) whose two model families produce a calibrated
probability; an Auditing stage (System 2) then supplies attribution, recourse,
and a fairness audit that routes to the label-side MRR / DGLC path, ending in a
gated decision. Exports BOTH a vector SVG (scales to print) and a 300-dpi PNG.

Not a data plot; text is fixed and must match the manuscript exactly.
Output:
  <template>/fig1.svg   (vector source)
  <template>/fig1.png   (raster used by \\includegraphics)
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# --- output location: the IEEE Access template dir (fig1 = framework) --------
HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.abspath(os.path.join(
    HERE, "..", "..", "ACCESS_latex_template_20260513"))

# --- restrained editorial palette -------------------------------------------
SLATE_FILL, SLATE_EDGE, SLATE_TX = "#EAF0F6", "#4A6D91", "#22384C"   # inputs
AMB_FILL,  AMB_EDGE,  AMB_TITLE = "#FCF5E8", "#D2A54A", "#8A6412"    # predictive
AMB_SUBED, SUB_TX               = "#E4C583", "#3A3626"
AMB_ACC                         = "#B07C27"                          # calib highlight
SAGE_FILL, SAGE_EDGE, SAGE_TITLE= "#EDF4EC", "#7BA074", "#3E5C39"    # auditing
SAGE_SUBED, SAGE_TX             = "#B6D0AE", "#2A3A28"
DARK                            = "#33414C"                          # final box
ANNOT                           = "#5B5B5B"
ARROW                           = "#5A6570"
WHITE                           = "#FFFFFF"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "svg.fonttype": "none",   # keep text as text in the SVG (editable, crisp)
})

SHADOW = [pe.withSimplePatchShadow(offset=(1.4, -1.4), alpha=0.10,
                                   shadow_rgbFace="#5A6570")]


def rbox(ax, x, y, w, h, fill, edge, lw=1.4, rs=1.3, shadow=False, z=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={rs}",
                       linewidth=lw, edgecolor=edge, facecolor=fill,
                       mutation_aspect=1.0, zorder=z)
    if shadow:
        p.set_path_effects(SHADOW)
    ax.add_patch(p)


def arrow(ax, p0, p1, color=ARROW, lw=1.3, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=11, linewidth=lw,
        color=color, connectionstyle=f"arc3,rad={rad}",
        shrinkA=1.5, shrinkB=1.5, zorder=1))


def lines(ax, cx, y0, rows, lh, fs, color, ha="center", weight="normal"):
    """Place a stack of text rows; rows may be (text, weight, color) tuples."""
    for i, r in enumerate(rows):
        t, w, c = (r if isinstance(r, tuple) else (r, weight, color))
        ax.text(cx, y0 + i * lh, t, ha=ha, va="center", fontsize=fs,
                color=c, fontweight=w, zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(8.0, 9.16))
    ax.set_xlim(0, 100)
    ax.set_ylim(114.5, 0)          # inverted: y increases downward (top-down)
    ax.axis("off")

    # ---------- 1. inputs ---------------------------------------------------
    rbox(ax, 5, 3, 90, 8, SLATE_FILL, SLATE_EDGE, shadow=True)
    lines(ax, 50, 5.9, [
        "Candidate record (AMEO 2015): AMCAT cognitive / technical scores,",
        "personality, academic history, demographics",
    ], lh=2.6, fs=10, color=SLATE_TX)

    arrow(ax, (50, 11), (50, 13))

    rbox(ax, 5, 13, 90, 8, SLATE_FILL, SLATE_EDGE, shadow=True)
    lines(ax, 50, 15.8, [
        "Leakage-controlled preprocessing",
        "(exclude post-hire fields; impute, scale, encode)",
    ], lh=2.6, fs=10, color=SLATE_TX)

    arrow(ax, (50, 21), (50, 23.7))

    # ---------- 2. predictive stage (System 1) ------------------------------
    rbox(ax, 3, 24, 94, 35, AMB_FILL, AMB_EDGE, lw=1.6, rs=2.0, shadow=True, z=1)
    ax.text(50, 27.0, "Predictive stage  (System 1)", ha="center", va="center",
            fontsize=13, fontweight="bold", color=AMB_TITLE, zorder=4)
    ax.text(50, 29.0, "13 + 4 = 17 models benchmarked on identical cached splits",
            ha="center", va="center", fontsize=8.3, color="#9A7A3A",
            style="italic", zorder=4)

    # classical / ensemble / glass-box (13)
    rbox(ax, 6, 31, 52, 16.5, WHITE, AMB_SUBED, lw=1.2, rs=1.2, z=2)
    lines(ax, 8.5, 33.6, [
        ("Classical / ensemble / glass-box (13):", "bold", SUB_TX),
        ("LR, NB, kNN, SVM-RBF, DT, RF,", "normal", SUB_TX),
        ("ExtraTrees, HistGB, XGBoost,", "normal", SUB_TX),
        ("CatBoost, LightGBM, MLP, EBM", "normal", SUB_TX),
    ], lh=3.4, fs=9.3, color=SUB_TX, ha="left")

    # deep tabular (4)
    rbox(ax, 60, 31, 34, 16.5, WHITE, AMB_SUBED, lw=1.2, rs=1.2, z=2)
    lines(ax, 62.5, 34.8, [
        ("Deep tabular (4):", "bold", SUB_TX),
        ("FT-Transformer, TabNet,", "normal", SUB_TX),
        ("TabTransformer, DeBERTa", "normal", SUB_TX),
    ], lh=3.4, fs=9.3, color=SUB_TX, ha="left")

    # converging arrows into the calibrated-probability box
    arrow(ax, (32, 47.5), (44, 50.3), rad=0.18)
    arrow(ax, (77, 47.5), (56, 50.3), rad=-0.18)

    rbox(ax, 24, 50.3, 52, 6.8, AMB_ACC, AMB_ACC, lw=0, rs=1.4, shadow=True, z=3)
    ax.text(50, 53.7, "Calibrated probability   P(high salary | x)",
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color=WHITE, zorder=5)

    # ---------- between-stage annotation + handoff --------------------------
    arrow(ax, (50, 59), (50, 63.2))
    ax.text(63.5, 61.0, "deliberate before the prediction is trusted",
            ha="left", va="center", fontsize=9.5, style="italic", color=ANNOT,
            zorder=4)

    # ---------- 3. auditing stage (System 2) --------------------------------
    rbox(ax, 3, 63.2, 94, 37, SAGE_FILL, SAGE_EDGE, lw=1.6, rs=2.0,
         shadow=True, z=1)
    ax.text(50, 66.2, "Auditing stage  (System 2)", ha="center", va="center",
            fontsize=13, fontweight="bold", color=SAGE_TITLE, zorder=4)

    rbox(ax, 6, 68.8, 88, 6.6, WHITE, SAGE_SUBED, lw=1.2, rs=1.1, z=2)
    ax.text(9, 72.1, "(a)  Attribution: global + local SHAP",
            ha="left", va="center", fontsize=9.6, color=SAGE_TX, zorder=4)

    rbox(ax, 6, 77.0, 88, 6.6, WHITE, SAGE_SUBED, lw=1.2, rs=1.1, z=2)
    ax.text(9, 80.3, "(b)  Counterfactual recourse: minimal-cost, "
                     "actionability-constrained",
            ha="left", va="center", fontsize=9.6, color=SAGE_TX, zorder=4)

    rbox(ax, 6, 85.2, 88, 11.2, WHITE, SAGE_SUBED, lw=1.2, rs=1.1, z=2)
    ax.text(9, 88.4, "(c)  Fairness audit: group + intersectional metrics",
            ha="left", va="center", fontsize=9.6, color=SAGE_TX, zorder=4)
    ax.text(12.2, 92.6, "→  label-side diagnosis & correction (MRR / DGLC)",
            ha="left", va="center", fontsize=9.6, color=SAGE_TITLE,
            fontweight="bold", zorder=4)

    arrow(ax, (50, 100.2), (50, 103.3))

    # ---------- 4. gated decision ------------------------------------------
    rbox(ax, 5, 103.3, 90, 7.8, DARK, DARK, lw=0, rs=1.5, shadow=True, z=3)
    ax.text(50, 107.2,
            "Gated decision + explanation + recourse + fairness report",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=WHITE, zorder=5)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.995, bottom=0.005)
    svg = os.path.join(TEMPLATE_DIR, "fig1.svg")
    png = os.path.join(TEMPLATE_DIR, "fig1.png")
    fig.savefig(svg)                      # vector
    fig.savefig(png, dpi=300)             # raster used by LaTeX
    plt.close(fig)
    print("saved ->", svg)
    print("saved ->", png)


if __name__ == "__main__":
    main()
