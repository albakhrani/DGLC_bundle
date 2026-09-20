"""
Stage 9 - Modern tabular deep-learning baselines.

Adds two state-of-the-art tabular neural architectures as fair competitors to the
gradient-boosted trees and the DeBERTa table-to-text model, all on the identical
stratified split:
  * FT-Transformer (Gorishniy et al.) - feature tokenizer + Transformer.
  * TabNet (Arik & Pfister)            - sequential attention with feature masks.

This directly addresses whether deep models tailored to tabular data (rather than
a serialized language model) can beat gradient boosting here.

Outputs:
  outputs/metrics/test_probs_FT-Transformer.npy, test_probs_TabNet.npy
  outputs/metrics/tabular_dl_results.json
  outputs/figures/fig_tabular_dl.png
"""
from __future__ import annotations
import warnings, json, os
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import matplotlib.pyplot as plt

import config as C
from utils import classification_metrics, bootstrap_ci, save_json, set_plot_style
import data_module as D

C.set_seed()
set_plot_style()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def prepare(df, tr, te):
    numeric, categorical = D.feature_columns(df)
    # numeric: median impute + standardize (fit on train)
    num_imp = SimpleImputer(strategy="median").fit(df.iloc[tr][numeric])
    scaler = StandardScaler().fit(num_imp.transform(df.iloc[tr][numeric]))
    def num(idx):
        return scaler.transform(num_imp.transform(df.iloc[idx][numeric])).astype(np.float32)
    # categorical: most-frequent impute + ordinal encode (unknown -> new index)
    cat_imp = SimpleImputer(strategy="most_frequent").fit(df.iloc[tr][categorical].astype(str))
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                         encoded_missing_value=-1).fit(cat_imp.transform(df.iloc[tr][categorical].astype(str)))
    cards = [len(c) + 1 for c in enc.categories_]      # +1 for unknown bucket
    def cat(idx):
        x = enc.transform(cat_imp.transform(df.iloc[idx][categorical].astype(str)))
        x = x.astype(np.int64)
        last = np.array(cards) - 1                       # per-column unknown bucket
        for j in range(x.shape[1]):
            x[x[:, j] < 0, j] = last[j]                  # map unknown/missing -> last index
        return x
    return (num(tr), cat(tr)), (num(te), cat(te)), cards, numeric, categorical


# --------------------------------------------------------------------------- #
# FT-Transformer (rtdl_revisiting_models)
# --------------------------------------------------------------------------- #
def train_ft(Xtr, Xctr, ytr, Xva, Xcva, yva, Xte, Xcte, cards, epochs=80):
    """Epoch selected on a VALIDATION split (no test peeking); test scored once."""
    from rtdl_revisiting_models import FTTransformer
    Xn = torch.tensor(Xtr).to(DEVICE); Xc = torch.tensor(Xctr).to(DEVICE)
    y = torch.tensor(ytr, dtype=torch.float32).to(DEVICE)
    Xn_va = torch.tensor(Xva).to(DEVICE); Xc_va = torch.tensor(Xcva).to(DEVICE)
    Xn_te = torch.tensor(Xte).to(DEVICE); Xc_te = torch.tensor(Xcte).to(DEVICE)
    kw = FTTransformer.get_default_kwargs(n_blocks=3)
    model = FTTransformer(n_cont_features=Xtr.shape[1], cat_cardinalities=cards,
                          d_out=1, **kw).to(DEVICE)
    opt = model.make_default_optimizer() if hasattr(model, "make_default_optimizer") \
        else torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    lossfn = nn.BCEWithLogitsLoss()
    ds = TensorDataset(Xn, Xc, y); dl = DataLoader(ds, batch_size=128, shuffle=True)
    best_val, best_test = -1, None
    for ep in range(epochs):
        model.train()
        for xn, xc, yb in dl:
            opt.zero_grad()
            loss = lossfn(model(xn, xc).squeeze(-1), yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xn_va, Xc_va).squeeze(-1)).cpu().numpy()
            pt = torch.sigmoid(model(Xn_te, Xc_te).squeeze(-1)).cpu().numpy()
        vauc = classification_metrics(yva, pv)["AUROC"]
        if vauc > best_val:                 # select on validation
            best_val, best_test = vauc, pt
        if (ep + 1) % 20 == 0:
            print(f"  FT-Transformer ep{ep+1}: val AUROC={vauc:.4f}")
    print(f"  FT-Transformer selected val AUROC={best_val:.4f}")
    return best_test


# --------------------------------------------------------------------------- #
# TabNet (pytorch-tabnet)
# --------------------------------------------------------------------------- #
def train_tabnet(Xtr, Xctr, ytr, Xva, Xcva, yva, Xte, Xcte, cards):
    """Early stopping on VALIDATION split; test scored once."""
    from pytorch_tabnet.tab_model import TabNetClassifier
    Xtr_all = np.hstack([Xtr, Xctr.astype(np.float32)])
    Xva_all = np.hstack([Xva, Xcva.astype(np.float32)])
    Xte_all = np.hstack([Xte, Xcte.astype(np.float32)])
    n_num = Xtr.shape[1]
    cat_idxs = list(range(n_num, n_num + len(cards)))
    clf = TabNetClassifier(cat_idxs=cat_idxs, cat_dims=list(cards),
                           n_d=24, n_a=24, n_steps=4, gamma=1.5,
                           seed=C.SEED, verbose=0,
                           optimizer_params=dict(lr=2e-2))
    clf.fit(Xtr_all, ytr, eval_set=[(Xva_all, yva)], eval_metric=["auc"],
            max_epochs=150, patience=25, batch_size=256, virtual_batch_size=128)
    return clf.predict_proba(Xte_all)[:, 1]


def train_tabtransformer(Xtr, Xctr, ytr, Xva, Xcva, yva, Xte, Xcte, cards, epochs=80):
    """TabTransformer (Huang et al., 2020): categorical embeddings through a
    Transformer, concatenated with continuous features. Validation-selected."""
    from tab_transformer_pytorch import TabTransformer
    Xn = torch.tensor(Xtr).to(DEVICE); Xc = torch.tensor(Xctr).to(DEVICE)
    y = torch.tensor(ytr, dtype=torch.float32).to(DEVICE)
    Xn_va = torch.tensor(Xva).to(DEVICE); Xc_va = torch.tensor(Xcva).to(DEVICE)
    Xn_te = torch.tensor(Xte).to(DEVICE); Xc_te = torch.tensor(Xcte).to(DEVICE)
    model = TabTransformer(categories=tuple(cards), num_continuous=Xtr.shape[1],
                           dim=32, depth=4, heads=8, dim_out=1,
                           mlp_hidden_mults=(4, 2), mlp_act=nn.ReLU(),
                           attn_dropout=0.1, ff_dropout=0.1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    lossfn = nn.BCEWithLogitsLoss()
    ds = TensorDataset(Xn, Xc, y); dl = DataLoader(ds, batch_size=128, shuffle=True)
    best_val, best_test = -1, None
    for ep in range(epochs):
        model.train()
        for xn, xc, yb in dl:
            opt.zero_grad()
            loss = lossfn(model(xc, xn).squeeze(-1), yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xc_va, Xn_va).squeeze(-1)).cpu().numpy()
            pt = torch.sigmoid(model(Xc_te, Xn_te).squeeze(-1)).cpu().numpy()
        vauc = classification_metrics(yva, pv)["AUROC"]
        if vauc > best_val:
            best_val, best_test = vauc, pt
        if (ep + 1) % 20 == 0:
            print(f"  TabTransformer ep{ep+1}: val AUROC={vauc:.4f}")
    print(f"  TabTransformer selected val AUROC={best_val:.4f}")
    return best_test


def main():
    from sklearn.model_selection import train_test_split
    df = D.load_frame()
    y = df[C.TARGET].astype(int).values
    sp = D.get_splits(df)
    tr_all, te = np.array(sp["train"]), np.array(sp["test"])
    # carve a validation split from train for epoch / early-stopping selection
    tr, va = train_test_split(tr_all, test_size=0.15, random_state=C.SEED,
                              stratify=y[tr_all])
    (Xtr, Xctr), (Xva, Xcva), cards, numeric, categorical = prepare(df, tr, va)
    (_, _), (Xte, Xcte), _, _, _ = prepare(df, tr, te)
    print(f"FT/TabNet inputs: {Xtr.shape[1]} numeric + {len(cards)} categorical; "
          f"train={len(tr)} val={len(va)} test={len(te)}")

    results = {}
    print("Training FT-Transformer ...")
    p_ft = train_ft(Xtr, Xctr, y[tr], Xva, Xcva, y[va], Xte, Xcte, cards)
    np.save(f"{C.MET_DIR}/test_probs_FT-Transformer.npy", p_ft)

    print("Training TabNet ...")
    p_tn = train_tabnet(Xtr, Xctr, y[tr], Xva, Xcva, y[va], Xte, Xcte, cards)
    np.save(f"{C.MET_DIR}/test_probs_TabNet.npy", p_tn)

    print("Training TabTransformer ...")
    p_tt = train_tabtransformer(Xtr, Xctr, y[tr], Xva, Xcva, y[va], Xte, Xcte, cards)
    np.save(f"{C.MET_DIR}/test_probs_TabTransformer.npy", p_tt)

    for name, p in [("FT-Transformer", p_ft), ("TabNet", p_tn),
                    ("TabTransformer", p_tt)]:
        m = classification_metrics(y[te], p)
        _, lo, hi = bootstrap_ci(y[te], p, "AUROC", seed=C.SEED)
        m.update({"AUROC_lo": lo, "AUROC_hi": hi})
        results[name] = m
        print(f"{name}: AUROC={m['AUROC']:.4f} (95% CI {lo:.3f}-{hi:.3f}) F1={m['F1']:.3f}")
    # include DeBERTa (trained in s05) so the figure shows ALL four deep models
    deb = f"{C.MET_DIR}/test_probs_DeBERTa.npy"
    if os.path.exists(deb):
        pd_ = np.load(deb); m = classification_metrics(y[te], pd_)
        _, lo, hi = bootstrap_ci(y[te], pd_, "AUROC", seed=C.SEED)
        m.update({"AUROC_lo": lo, "AUROC_hi": hi}); results["DeBERTa"] = m
    save_json(results, f"{C.MET_DIR}/tabular_dl_results.json")

    order = sorted(results, key=lambda n: results[n]["AUROC"], reverse=True)
    plt.figure(figsize=(5.4, 4))
    aucs = [results[n]["AUROC"] for n in order]
    err = [[results[n]["AUROC"] - results[n]["AUROC_lo"] for n in order],
           [results[n]["AUROC_hi"] - results[n]["AUROC"] for n in order]]
    plt.bar(order, aucs, yerr=err, color="#6a51a3", capsize=4, alpha=0.85)
    # Reference line: CatBoost held-out test AUROC from the s04 panel artifact
    # (run_all.sh runs s04 before s09, so the file exists in a full run).
    s1_path = f"{C.MET_DIR}/system1_test.csv"
    try:
        gbdt_auc = float(pd.read_csv(s1_path, index_col="Model").loc["CatBoost", "AUROC"])
    except (FileNotFoundError, KeyError) as e:
        print(f"WARNING: CatBoost test AUROC unavailable from {s1_path} ({e!r}); "
              "omitting best-GBDT reference line - run s04_system1_models.py first.")
        gbdt_auc = None
    if gbdt_auc is not None:
        plt.axhline(gbdt_auc, ls="--", c="k", lw=0.9,
                    label=f"best GBDT (CatBoost, {gbdt_auc:.3f})")
    plt.ylim(0.70, 0.80); plt.ylabel("Test AUROC")
    plt.xticks(rotation=15); plt.legend(fontsize=8)
    plt.title("Deep-learning baselines vs.\\ best boosted tree")
    plt.savefig(f"{C.FIG_DIR}/fig_tabular_dl.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("Stage 9 complete.")


if __name__ == "__main__":
    main()
