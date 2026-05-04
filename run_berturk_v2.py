"""
BERTurk Fine-Tuning v2 — Iyilestirilmis
- Sinif agirlikli CrossEntropy (notr sinifi icin)
- LR: 1e-5 (2e-5 yerine)
- 6 epoch + erken durdurma (patience=3, val macro-F1 izlenir)
- Label smoothing 0.1
- Effective batch 32 (batch=8, accum=4)
"""
import os, json, random, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def log(msg): print(msg, flush=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
    logging as hf_logging,
)
hf_logging.set_verbosity_error()
from sklearn.metrics import accuracy_score, f1_score, classification_report
from pathlib import Path

MODEL_NAME   = "dbmdz/bert-base-turkish-cased"
NUM_LABELS   = 3
LR           = 1e-5
BATCH_SIZE   = 8
ACCUM_STEPS  = 4        # effective batch = 32
EPOCHS       = 6
PATIENCE     = 3        # early stopping
MAX_LEN      = 128
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
LABEL_SMOOTH = 0.1
SEED         = 42
CKPT_DIR     = Path("models/berturk_finetuned")
RESULTS_DIR  = Path("results")
LABEL_NAMES  = ["olumsuz", "notr", "olumlu"]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Cihaz : {DEVICE}")
if torch.cuda.is_available():
    log(f"GPU   : {torch.cuda.get_device_name(0)}")
    log(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

log("Veri yukleniyor...")
train_df = pd.read_csv("data/processed/train.csv")
val_df   = pd.read_csv("data/processed/val.csv")
test_df  = pd.read_csv("data/processed/test.csv")
log(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

# Sinif agirlikları hesapla
label_counts = train_df["label"].value_counts().sort_index().values
total        = label_counts.sum()
class_weights = torch.tensor(
    total / (NUM_LABELS * label_counts), dtype=torch.float32
).to(DEVICE)
log(f"Sinif agirliklari: {class_weights.cpu().tolist()}")

log(f"Tokenizer yukleniyor: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class ReviewDataset(Dataset):
    def __init__(self, df):
        self.texts  = df["Review"].astype(str).tolist()
        self.labels = df["label"].tolist()
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        enc = tokenizer(
            self.texts[i], truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}, torch.tensor(self.labels[i])

train_ds = ReviewDataset(train_df)
val_ds   = ReviewDataset(val_df)
test_ds  = ReviewDataset(test_df)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
val_dl   = DataLoader(val_ds,   batch_size=32,          shuffle=False, num_workers=0, pin_memory=False)
test_dl  = DataLoader(test_ds,  batch_size=32,          shuffle=False, num_workers=0, pin_memory=False)

log(f"Model yukleniyor: {MODEL_NAME}")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model.to(DEVICE)
log(f"VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Label smoothing + sinif agirlikli loss
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)

optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps  = (len(train_dl) // ACCUM_STEPS) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
log(f"Egitim: {EPOCHS} epoch | {total_steps} adim | {warmup_steps} warmup")
log(f"LR={LR} | Effective batch={BATCH_SIZE*ACCUM_STEPS} | Label smooth={LABEL_SMOOTH}")


def evaluate(loader):
    model.eval()
    preds, all_labels, probas = [], [], []
    with torch.no_grad():
        for batch, labels in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out   = model(**batch)
            prob  = torch.softmax(out.logits, dim=-1).cpu().tolist()
            pred  = out.logits.argmax(-1).cpu().tolist()
            preds.extend(pred); all_labels.extend(labels.tolist()); probas.extend(prob)
    acc = accuracy_score(all_labels, preds)
    f1  = f1_score(all_labels, preds, average="macro")
    return acc, f1, preds, all_labels, probas


CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

log("\n" + "=" * 60)
log("EGITIM BASLIYOR (v2)")
log("=" * 60)

best_val_f1   = 0.0
best_state    = None
patience_cnt  = 0
history       = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (batch, labels) in enumerate(train_dl):
        batch  = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = labels.to(DEVICE)
        logits = model(**batch).logits
        loss   = criterion(logits, labels) / ACCUM_STEPS
        loss.backward()
        total_loss += loss.item() * ACCUM_STEPS

        if (step + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % 200 == 0:
            log(f"  Epoch {epoch+1} | Step {step+1}/{len(train_dl)} | Loss: {total_loss/(step+1):.4f}")

    avg_loss = total_loss / len(train_dl)
    val_acc, val_f1, _, _, _ = evaluate(val_dl)
    history.append({"epoch": epoch+1, "loss": avg_loss, "val_acc": val_acc, "val_f1": val_f1})

    log(f"\nEpoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1  = val_f1
        best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_cnt = 0
        log(f"  >> En iyi model guncellendi (F1={val_f1:.4f})")
    else:
        patience_cnt += 1
        log(f"  -- patience {patience_cnt}/{PATIENCE}")
        if patience_cnt >= PATIENCE:
            log(f"  >> Erken durdurma! (epoch {epoch+1})")
            break
    log("")

log("=" * 60)
log("TEST DEGERLENDIRME")
log("=" * 60)

model.load_state_dict(best_state)
del best_state
torch.cuda.empty_cache()

model.save_pretrained(str(CKPT_DIR))
tokenizer.save_pretrained(str(CKPT_DIR))
log(f"Checkpoint kaydedildi: {CKPT_DIR}")

test_acc, test_f1, test_preds, test_labels, test_probas = evaluate(test_dl)
log(f"Test Accuracy : {test_acc:.4f}")
log(f"Test Macro-F1 : {test_f1:.4f}")
log("")
log(classification_report(test_labels, test_preds, target_names=LABEL_NAMES, digits=4))

from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, roc_auc_score
from sklearn.preprocessing import label_binarize
mcc   = matthews_corrcoef(test_labels, test_preds)
kappa = cohen_kappa_score(test_labels, test_preds)
y_bin = label_binarize(test_labels, classes=[0, 1, 2])
auc   = roc_auc_score(y_bin, np.array(test_probas), average="macro", multi_class="ovr")
log(f"MCC        : {mcc:.4f}")
log(f"Cohen Kappa: {kappa:.4f}")
log(f"ROC-AUC    : {auc:.4f}")

results = {
    "berturk_finetuned": {
        "accuracy":    round(test_acc, 4),
        "macro_f1":    round(test_f1, 4),
        "mcc":         round(mcc, 4),
        "kappa":       round(kappa, 4),
        "roc_auc":     round(auc, 4),
        "best_val_f1": round(best_val_f1, 4),
        "history":     history,
        "config":      {
            "lr": LR, "epochs_run": len(history),
            "effective_batch": BATCH_SIZE * ACCUM_STEPS,
            "label_smoothing": LABEL_SMOOTH,
            "class_weights": class_weights.cpu().tolist(),
        },
    }
}
with open(RESULTS_DIR / "berturk_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
log("Sonuclar kaydedildi: results/berturk_results.json")
log("EGITIM TAMAMLANDI.")
