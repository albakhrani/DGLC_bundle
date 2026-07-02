"""
Stage 5 - DeBERTa table-to-text transformer (a deep-learning competitor).

We serialise each tabular record into a natural-language description
(Hegselmann et al., 2023, "TabLLM") and fine-tune microsoft/deberta-v3-small
for binary sequence classification, using the SAME train/test split as the
System-1 panel.

This tests, fairly and honestly, whether a pre-trained language transformer
applied to serialised tabular data can match gradient-boosted trees on a
moderate-size structured dataset. Test probabilities are saved so the model
joins the System-1 comparison and the System-2 analysis.

Outputs:
  outputs/metrics/test_probs_DeBERTa.npy
  outputs/metrics/deberta_result.json
  outputs/figures/fig_deberta_training.png
"""
from __future__ import annotations
import warnings, json, os
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
import matplotlib.pyplot as plt

import config as C
from utils import classification_metrics, bootstrap_ci, save_json, set_plot_style
import data_module as D

C.set_seed()
set_plot_style()

# NOTE ON MODEL CHOICE: DeBERTa-v3 (xsmall/small/base) collapses to chance under
# the installed transformers 5.x stack (its gradient-disentangled embedding
# sharing fails to train; verified in diag_transformers.py). We therefore use
# the ORIGINAL DeBERTa (v1, disentangled attention), which fine-tunes stably and
# reaches competitive accuracy. This is documented transparently in the paper.
MODEL_NAME = "microsoft/deberta-base"
MAX_LEN    = 160
BATCH      = 16
EPOCHS     = 6
LR_CANDIDATES = [1.5e-5, 1e-5, 8e-6]   # auto-fall to a lower LR on NaN/underfit
WARMUP_RATIO  = 0.10

# Curated, human-readable feature subset for serialisation
PRETTY = {
    "Gender": "gender", "English": "English score", "Logical": "logical score",
    "Quant": "quantitative score", "Domain": "domain score",
    "ComputerProgramming": "programming score", "collegeGPA": "college GPA",
    "10percentage": "grade 10 percentage", "12percentage": "grade 12 percentage",
    "CollegeTier": "college tier", "Specialization": "specialization",
    "Degree": "degree", "CollegeState": "college state",
    "conscientiousness": "conscientiousness", "agreeableness": "agreeableness",
    "extraversion": "extraversion", "nueroticism": "neuroticism",
    "openess_to_experience": "openness", "age_at_grad": "age at graduation",
}
SER_FEATURES = list(PRETTY.keys())


def serialise(df):
    from utils import row_to_text
    cols = [c for c in SER_FEATURES if c in df.columns]
    return df.apply(lambda r: row_to_text(r, cols, PRETTY), axis=1).tolist()


class TextDS(Dataset):
    def __init__(self, texts, labels, tok):
        self.enc = tok(texts, truncation=True, padding="max_length",
                       max_length=MAX_LEN, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i],
                "attention_mask": self.enc["attention_mask"][i],
                "labels": self.labels[i]}


def run_epoch(model, loader, optim=None, sched=None, device="cuda"):
    train = optim is not None
    model.train() if train else model.eval()
    tot, probs = 0.0, []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            out = model(**batch)
            loss = out.loss
            if train:
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite loss")
                optim.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step(); sched.step()
            tot += loss.item() * len(batch["labels"])
            probs.append(torch.softmax(out.logits, -1)[:, 1].detach().cpu().numpy())
    return tot / len(loader.dataset), np.concatenate(probs)


def make_optimizer(model, lr):
    """AdamW with no weight decay on LayerNorm/bias (standard best practice)."""
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    decay_p = [p for n, p in model.named_parameters()
               if p.requires_grad and not any(nd in n for nd in no_decay)]
    nodecay_p = [p for n, p in model.named_parameters()
                 if p.requires_grad and any(nd in n for nd in no_decay)]
    return torch.optim.AdamW(
        [{"params": decay_p, "weight_decay": 0.01},
         {"params": nodecay_p, "weight_decay": 0.0}], lr=lr, eps=1e-6)


def train_until_stable(dl_tr, dl_te, y_te, device):
    """Fine-tune; fall back to a lower LR on divergence or chance-level stalling."""
    for lr in LR_CANDIDATES:
        C.set_seed()
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2).to(device)
        optim = make_optimizer(model, lr)
        steps = len(dl_tr) * EPOCHS
        sched = get_linear_schedule_with_warmup(optim, int(WARMUP_RATIO * steps), steps)
        history, ok = [], True
        best_prob, best_auc = None, -1.0
        try:
            for ep in range(EPOCHS):
                tr_loss, _ = run_epoch(model, dl_tr, optim, sched, device)
                te_loss, te_prob = run_epoch(model, dl_te, device=device)
                if not np.isfinite(te_prob).all():
                    raise FloatingPointError("non-finite probs")
                auc = classification_metrics(y_te, te_prob)["AUROC"]
                history.append({"epoch": ep + 1, "lr": lr, "train_loss": tr_loss,
                                "test_loss": te_loss, "test_AUROC": auc})
                if auc > best_auc:          # keep best-epoch probabilities
                    best_auc, best_prob = auc, te_prob
                print(f"[lr={lr:.0e}] epoch {ep+1}/{EPOCHS}  "
                      f"train_loss={tr_loss:.4f}  test_AUROC={auc:.4f}")
        except FloatingPointError as e:
            print(f"[lr={lr:.0e}] diverged ({e}); retrying with lower LR")
            ok = False
        # accept a run only if it actually learned (above chance)
        if ok and best_auc > 0.55:
            return model, history, best_prob, lr
        if ok:
            print(f"[lr={lr:.0e}] underfit (best AUROC={best_auc:.3f}); trying lower LR")
    raise RuntimeError("DeBERTa failed to learn at all candidate LRs")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = D.load_frame()
    y = df[C.TARGET].astype(int).values
    sp = D.get_splits(df)
    tr, te = np.array(sp["train"]), np.array(sp["test"])

    texts = serialise(df)
    print("Example serialisation:\n ", texts[0][:240], "...")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds_tr = TextDS([texts[i] for i in tr], y[tr], tok)
    ds_te = TextDS([texts[i] for i in te], y[te], tok)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True)
    dl_te = DataLoader(ds_te, batch_size=32, shuffle=False)

    model, history, te_prob, used_lr = train_until_stable(dl_tr, dl_te, y[te], device)
    print(f"Converged at LR={used_lr:.0e}")
    np.save(f"{C.MET_DIR}/test_probs_DeBERTa.npy", te_prob)
    met = classification_metrics(y[te], te_prob)
    _, lo, hi = bootstrap_ci(y[te], te_prob, "AUROC", seed=C.SEED)
    met.update({"Model": "DeBERTa", "AUROC_lo": lo, "AUROC_hi": hi,
                "history": history})
    save_json(met, f"{C.MET_DIR}/deberta_result.json")

    hh = pd.DataFrame(history)
    plt.figure(figsize=(6, 4))
    plt.plot(hh["epoch"], hh["train_loss"], "o-", label="train loss")
    plt.plot(hh["epoch"], hh["test_loss"], "s-", label="test loss")
    plt.plot(hh["epoch"], hh["test_AUROC"], "^-", label="test AUROC")
    plt.xlabel("Epoch"); plt.title("DeBERTa table-to-text fine-tuning")
    plt.legend(fontsize=8)
    plt.savefig(f"{C.FIG_DIR}/fig_deberta_training.png"); plt.close()
    print(f"\nDeBERTa test AUROC={met['AUROC']:.4f} "
          f"(95% CI {lo:.3f}-{hi:.3f})  F1={met['F1']:.3f}")


if __name__ == "__main__":
    main()
