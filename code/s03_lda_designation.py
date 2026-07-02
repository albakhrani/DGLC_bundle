"""
Stage 3 - Latent Dirichlet Allocation over job designations (DESCRIPTIVE).

Rationale: 'Designation' is assigned at hiring and therefore cannot be used as
a predictor of salary without leakage. We instead use it descriptively: LDA
recovers latent occupational themes in the labour-market outcomes, and we relate
those themes to earnings and to gender composition. This supplies the
'structure discovery' that motivates the fairness analysis in Stage 6.

Model selection: we sweep K and pick the number of topics by held-out
perplexity and topic coherence (UMass) trade-off.

Outputs:
  outputs/tables/lda_topics.csv             top terms per topic
  outputs/tables/lda_topic_profiles.csv     salary & gender profile per topic
  outputs/metrics/lda_selection.json        K-sweep diagnostics
  outputs/figures/fig_lda_topics.png, fig_lda_topic_salary_gender.png
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import config as C
from utils import set_plot_style, save_json

C.set_seed()
set_plot_style()

STOP = {"engineer", "engineers", "and", "of", "the", "in", "at", "for",
        "trainee", "executive", "junior", "senior", "associate"}  # too generic


def umass_coherence(model, X, feature_names, topn=10, eps=1e-12):
    """Mean UMass coherence across topics (higher = more coherent)."""
    Xb = (X > 0).astype(int)
    doc_freq = np.asarray(Xb.sum(axis=0)).ravel()
    co = Xb.T @ Xb        # term-term co-document counts
    scores = []
    for k in range(model.n_components):
        top = np.argsort(model.components_[k])[::-1][:topn]
        s = 0.0
        for i in range(1, len(top)):
            for j in range(i):
                wi, wj = top[i], top[j]
                s += np.log((co[wi, wj] + 1) / (doc_freq[wj] + eps))
        scores.append(s)
    return float(np.mean(scores))


def main():
    aux = pd.read_parquet(C.PROC_PARQUET.replace("processed", "processed_aux"))
    text = aux["designation_text"].fillna("").astype(str)
    text = text[text.str.len() > 1]
    aux = aux.loc[text.index].copy()

    vec = CountVectorizer(stop_words=list(STOP), max_df=0.6, min_df=8,
                          token_pattern=r"[a-z]{3,}")
    X = vec.fit_transform(text)
    vocab = np.array(vec.get_feature_names_out())

    # ---- choose K via perplexity + coherence ------------------------------
    sweep = {}
    for k in [5, 6, 8, 10, 12]:
        lda_k = LatentDirichletAllocation(n_components=k, learning_method="batch",
                                          max_iter=30, random_state=C.SEED)
        lda_k.fit(X)
        sweep[k] = {"perplexity": float(lda_k.perplexity(X)),
                    "coherence_umass": umass_coherence(lda_k, X, vocab)}
    save_json(sweep, f"{C.MET_DIR}/lda_selection.json")
    # pick K maximising coherence (typical practical criterion)
    K = max(sweep, key=lambda k: sweep[k]["coherence_umass"])
    print("K sweep:", {k: round(v["coherence_umass"], 2) for k, v in sweep.items()})
    print("Selected K =", K)

    lda = LatentDirichletAllocation(n_components=K, learning_method="batch",
                                    max_iter=60, random_state=C.SEED)
    W = lda.fit_transform(X)
    aux["topic"] = W.argmax(axis=1)

    # ---- top terms ---------------------------------------------------------
    rows = []
    for k in range(K):
        top = np.argsort(lda.components_[k])[::-1][:10]
        rows.append({"topic": k, "top_terms": ", ".join(vocab[top])})
    topics_df = pd.DataFrame(rows)
    topics_df.to_csv(f"{C.TAB_DIR}/lda_topics.csv", index=False)
    print(topics_df.to_string(index=False))

    # ---- topic profiles: salary + gender ----------------------------------
    prof = (aux.groupby("topic")
            .agg(n=("Salary", "size"),
                 median_salary=("Salary", "median"),
                 pct_high_salary=(C.TARGET, "mean"),
                 pct_female=("Gender", lambda s: (s == "female").mean()))
            .reset_index())
    prof = prof.merge(topics_df, on="topic")
    prof.to_csv(f"{C.TAB_DIR}/lda_topic_profiles.csv", index=False)

    _plot_topics(lda, vocab, K)
    _plot_profiles(prof)
    print("\nLDA stage complete. Topic profiles:\n", prof[
        ["topic", "n", "median_salary", "pct_high_salary", "pct_female"]].round(3))


def _plot_topics(lda, vocab, K):
    ncol = 2
    nrow = int(np.ceil(K / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.0 * nrow))
    axes = np.array(axes).ravel()
    for k in range(K):
        top = np.argsort(lda.components_[k])[::-1][:8]
        w = lda.components_[k][top]
        axes[k].barh(range(len(top)), w[::-1], color="#3182bd")
        axes[k].set_yticks(range(len(top)))
        axes[k].set_yticklabels(vocab[top][::-1], fontsize=8)
        axes[k].set_title(f"Topic {k}", fontsize=9)
    for j in range(K, len(axes)):
        axes[j].axis("off")
    fig.suptitle("LDA occupational topics (job designations)", y=1.01)
    fig.savefig(f"{C.FIG_DIR}/fig_lda_topics.png"); plt.close(fig)


def _plot_profiles(prof):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(prof["topic"], prof["median_salary"], color="#31a354")
    ax[0].set_xlabel("Topic"); ax[0].set_ylabel("Median salary (INR)")
    ax[0].set_title("Earnings by occupational topic")
    ax[1].bar(prof["topic"], 100 * prof["pct_female"], color="#de2d26")
    ax[1].axhline(100 * prof["pct_female"].mean(), ls="--", c="k", lw=0.8,
                  label="overall")
    ax[1].set_xlabel("Topic"); ax[1].set_ylabel("% female")
    ax[1].set_title("Gender composition by topic"); ax[1].legend(fontsize=8)
    fig.savefig(f"{C.FIG_DIR}/fig_lda_topic_salary_gender.png"); plt.close(fig)


if __name__ == "__main__":
    main()
