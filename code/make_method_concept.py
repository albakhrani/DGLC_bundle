"""
Conceptual schematic of the MRR -> DGLC method logic (a diagram, not a data plot).

Four stages, left to right, with the label-resident vs model-resident branch and
the DGLC self-validation loop:

  1. Observed gap
  2. DIAGNOSE  (Oaxaca-Blinder decomposition + ML counterfactual; three arms)
  3. CLASSIFY  (Merit-Reward Reversal test -> label-resident | model-resident)
  4. CORRECT + SELF-VALIDATE (DGLC; re-target label, loop: re-decompose -> residual ~ 0)

Muted, journal-appropriate palette; sans-serif; no data, no gradients, no clip-art.
Does NOT encode the refuted "levels-down/lifts-up" contrast. DGLC stage carries an
honest annotation that it is judged by merit-alignment + self-validation, not by
realized-label fairness.

Output: outputs/gap/figures/fig_method_concept.png (300 dpi, ~7 in wide).
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import config as C
import gap_common as G

# ----- palette (3 muted colours) ------------------------------------------- #
BOX_FILL = "#EDF1F5"   # very light slate
BOX_EDGE = "#34495E"   # dark slate
MAIN     = "#2E6F40"   # muted green  -> the label-resident main path
OFF      = "#9C6B70"   # muted mauve  -> the model-resident off-path
TEXT     = "#243B53"
SUB      = "#4A5A6A"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "svg.fonttype": "none",
})


def rbox(ax, cx, cy, w, h, fc=BOX_FILL, ec=BOX_EDGE, lw=1.6):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))


def arrow(ax, p0, p1, color=MAIN, lw=2.2, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=16,
        linewidth=lw, color=color, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", zorder=1,
        shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(7.5, 4.05))
    ax.set_xlim(0, 15.4); ax.set_ylim(0, 7.9)
    ax.axis("off")

    # header
    ax.text(7.7, 7.55, "Diagnose where a fairness gap lives, then correct it at the source",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color=TEXT)

    # main-row box geometry (non-uniform gaps: extra room before stage 4 for its branch label)
    cy = 3.9; w = 2.7; h = 1.5
    cx = [1.75, 5.25, 8.75, 13.0]
    edges_r = [c + w / 2 for c in cx]
    edges_l = [c - w / 2 for c in cx]

    for c in cx:
        rbox(ax, c, cy, w, h)

    # stage 1 - observed gap
    ax.text(cx[0], cy + 0.46, "1  Observed gap", ha="center", va="center",
            fontsize=9.0, fontweight="bold", color=TEXT)
    ax.text(cx[0], cy - 0.24, "Group fairness gap\nin a deployed model",
            ha="center", va="center", fontsize=8.0, color=SUB)

    # stage 2 - diagnose
    ax.text(cx[1], cy + 0.50, "2  DIAGNOSE", ha="center", va="center",
            fontsize=9.0, fontweight="bold", color=TEXT)
    ax.text(cx[1], cy - 0.28,
            "Oaxaca–Blinder wage\ndecomposition +\nML counterfactual\n(three converging arms)",
            ha="center", va="center", fontsize=7.7, color=SUB)

    # stage 3 - classify
    ax.text(cx[2], cy + 0.50, "3  CLASSIFY (MRR)", ha="center", va="center",
            fontsize=9.0, fontweight="bold", color=TEXT)
    ax.text(cx[2], cy - 0.28,
            "Merit-Reward Reversal:\ndoes a merit-only model\nfavour the group the\nlabel penalises?",
            ha="center", va="center", fontsize=7.7, color=SUB)

    # stage 4 - correct + self-validate
    ax.text(cx[3], cy + 0.50, "4  CORRECT (DGLC)", ha="center", va="center",
            fontsize=9.0, fontweight="bold", color=TEXT)
    ax.text(cx[3], cy - 0.28,
            "Re-target the label to\nthe merit-implied rate\n(non-discriminatory\nreference structure)",
            ha="center", va="center", fontsize=7.7, color=SUB)

    # ---- main-path arrows (green) ----------------------------------------- #
    arrow(ax, (edges_r[0], cy), (edges_l[1], cy))
    arrow(ax, (edges_r[1], cy), (edges_l[2], cy))
    # classify -> correct : the label-resident branch (wide gap holds a one-line label)
    arrow(ax, (edges_r[2], cy), (edges_l[3], cy))
    ax.text((edges_r[2] + edges_l[3]) / 2, cy + 0.24, "label-resident",
            ha="center", va="center", fontsize=7.4, color=MAIN, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))

    # ---- model-resident off-path (mauve, dashed) -------------------------- #
    off_cy = 1.15; off_w = 4.4; off_h = 1.0
    rbox(ax, cx[2], off_cy, off_w, off_h, fc="#F4ECEE", ec=OFF, lw=1.4)
    ax.text(cx[2], off_cy + 0.20, "model-resident → use model-side mitigation",
            ha="center", va="center", fontsize=8.0, color=OFF, fontweight="bold")
    ax.text(cx[2], off_cy - 0.24, "(ThresholdOptimizer, ExponentiatedGradient)",
            ha="center", va="center", fontsize=7.4, color=OFF)
    arrow(ax, (cx[2], cy - h / 2), (cx[2], off_cy + off_h / 2), color=OFF, lw=1.8, ls=(0, (4, 2)))
    ax.text(cx[2] + 0.18, (cy - h / 2 + off_cy + off_h / 2) / 2, "model-\nresident",
            ha="left", va="center", fontsize=7.4, color=OFF, fontweight="bold")

    # ---- DGLC self-validation loop (green, above stage 4) ----------------- #
    top = cy + h / 2
    arrow(ax, (cx[3] + 0.95, top + 0.02), (cx[3] - 0.95, top + 0.02),
          color=MAIN, lw=2.2, rad=-1.15)
    ax.text(cx[3], top + 1.82, "self-validation loop",
            ha="center", va="center", fontsize=8.2, color=MAIN, fontweight="bold")
    ax.text(cx[3], top + 1.36,
            "re-decompose → unexplained\nresidual  +0.070 → ≈0",
            ha="center", va="center", fontsize=7.4, color=SUB)

    # ---- honest annotation on the DGLC stage ------------------------------ #
    ax.text(cx[3], cy - h / 2 - 0.36,
            "judged by merit-alignment + self-validation,\nnot by realized-label fairness",
            ha="center", va="center", fontsize=7.1, color=SUB, style="italic")

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    out = f"{G.GAP_FIG}/fig_method_concept.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[concept] saved -> {out}")


if __name__ == "__main__":
    main()
