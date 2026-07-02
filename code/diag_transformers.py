"""Fast diagnostic: does each cached transformer learn the serialised task?
3 epochs, global LR 2e-5, same loop. Prints test AUROC per epoch."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
import config as C, data_module as D
from utils import classification_metrics
from s05_deberta import serialise, TextDS, run_epoch
C.set_seed()

device = "cuda"
df = D.load_frame(); y = df[C.TARGET].astype(int).values
sp = D.get_splits(df); tr, te = np.array(sp["train"]), np.array(sp["test"])
texts = serialise(df)

for name in ["microsoft/deberta-v3-small", "roberta-base", "bert-base-uncased"]:
    C.set_seed()
    tok = AutoTokenizer.from_pretrained(name)
    dl_tr = DataLoader(TextDS([texts[i] for i in tr], y[tr], tok), batch_size=16, shuffle=True)
    dl_te = DataLoader(TextDS([texts[i] for i in te], y[te], tok), batch_size=32)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    steps = len(dl_tr) * 3
    sch = get_linear_schedule_with_warmup(opt, int(0.1*steps), steps)
    aucs = []
    try:
        for ep in range(3):
            run_epoch(model, dl_tr, opt, sch, device)
            _, p = run_epoch(model, dl_te, device=device)
            a = classification_metrics(y[te], p)["AUROC"] if np.isfinite(p).all() else float("nan")
            aucs.append(round(a, 4))
    except Exception as e:
        aucs.append(f"FAILED:{type(e).__name__}")
    print(f"{name:32s} AUROC/epoch = {aucs}", flush=True)
