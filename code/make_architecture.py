"""Generate the System 1 / System 2 framework architecture diagram."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import config as C

plt.rcParams.update({"font.size": 9, "savefig.dpi": 300})
fig, ax = plt.subplots(figsize=(8.4, 9.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 13); ax.axis("off")

def box(x, y, w, h, text, fc, ec="#2b3a55", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)

def arrow(x1, y1, x2, y2, style="-|>", color="#2b3a55", lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=16, color=color, lw=lw))

# --- Data + preprocessing ---
box(1.5, 12.0, 7, 0.9, "Candidate record (AMEO 2015): AMCAT cognitive/technical\nscores, personality, academic history, demographics", "#eaf2fb", fs=8.5)
box(1.5, 10.8, 7, 0.85, "Leakage-controlled preprocessing\n(exclude post-hire fields; -1 -> missing + indicators; impute, scale, encode)", "#eaf2fb", fs=8.5)
arrow(5, 12.0, 5, 11.65)

# --- System 1 ---
box(0.6, 8.0, 8.8, 2.3, "", "#fdf3e7", ec="#d98c2b")
ax.text(5, 10.05, "SYSTEM 1  -  fast, intuitive prediction", ha="center",
        fontsize=10.5, fontweight="bold", color="#b5651d")
box(0.95, 8.35, 4.0, 1.35, "Model panel (14):\nLR, NB, kNN, SVM, DT, RF, ExtraTrees,\nHistGB, XGBoost, CatBoost, MLP", "#fff", ec="#d98c2b", fs=7.8)
box(5.15, 8.35, 3.95, 1.35, "Deep tabular models:\nDeBERTa (table-to-text),\nFT-Transformer, TabNet", "#fff", ec="#d98c2b", fs=7.8)
arrow(5, 10.8, 5, 10.3)
box(3.0, 6.9, 4, 0.8, "Calibrated probability  P(high salary | x)", "#fde9d0", ec="#d98c2b", bold=True, fs=8.5)
arrow(5, 8.0, 5, 7.7)

# --- gate ---
ax.text(5, 6.5, "deliberate before the prediction is trusted", ha="center",
        fontsize=8, style="italic", color="#555")
arrow(5, 6.9, 5, 6.0)

# --- System 2 ---
box(0.6, 1.3, 8.8, 4.5, "", "#eaf6ee", ec="#2e8b57")
ax.text(5, 5.55, "SYSTEM 2  -  slow, deliberative auditing", ha="center",
        fontsize=10.5, fontweight="bold", color="#1f7a44")
box(0.95, 4.35, 8.1, 0.95, "(a) Attribution: global + local SHAP feature importance", "#fff", ec="#2e8b57", fs=8.5)
box(0.95, 3.25, 8.1, 0.95, "(b) Counterfactual recourse: minimal-cost, actionability-constrained\nchange over mutable features (feasibility rule)", "#fff", ec="#2e8b57", fs=8.3)
box(0.95, 2.15, 8.1, 0.95, "(c) Fairness audit: group + intersectional metrics, then\nmitigation assessment (post-hoc threshold optimization)", "#fff", ec="#2e8b57", fs=8.3)
arrow(5, 4.35, 5, 4.2, style="-")
arrow(5, 3.25, 5, 3.1, style="-")

# --- output ---
box(1.5, 0.2, 7, 0.8, "Gated decision  +  explanation  +  recourse  +  fairness report",
    "#d7ecdc", ec="#1f7a44", bold=True, fs=9)
arrow(5, 1.3, 5, 1.0)

ax.set_title("Dual-process (System 1 / System 2) deployment framework",
             fontsize=12, fontweight="bold", pad=10)
fig.savefig(f"{C.FIG_DIR}/fig_architecture.png", bbox_inches="tight"); plt.close()
print("saved fig_architecture.png")
