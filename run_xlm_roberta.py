"""
XLM-RoBERTa Fine-Tuning — BERTurk ile karşılaştırma için
Model: xlm-roberta-base (270M parametre, 100 dil)
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
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    logging as hf_logging,
)
hf_logging.set_verbosity_error()
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
from src.evaluation.metrics import compute_classification_metrics, print_metrics

MODEL_NAME   = "xlm-roberta-base"
NUM_LABELS   = 3
LR           = 1e-5
BATCH_SIZE   = 8
ACCUM_STEPS  = 4       # effective batch = 32
EPOCHS       = 6
PATIENCE     = 3
MAX_LEN      = 128
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
LABEL_SMOOTH = 0.1
SEED         = 42
CKPT_DIR    = Path("models/xlm_roberta_finetuned")
RESULTS_DIR = Path("results")
LABEL_NAMES = ["olumsuz", "notr", "olumlu"]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Cihaz: {DEVICE} | Model: {MODEL_NAME}")

train_df = pd.read_csv("data/processed/train.csv")
val_df   = pd.read_csv("data/processed/val.csv")
test_df  = pd.read_csv("data/processed/test.csv")
log(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

# Sinif agirlikları
label_counts = train_df["label"].value_counts().sort_index().values
class_weights = torch.tensor(
    label_counts.sum() / (NUM_LABELS * label_counts), dtype=torch.float32
).to(DEVICE)
log(f"Sinif agirliklari: {class_weights.cpu().tolist()}")

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

train_dl = DataLoader(ReviewDataset(train_df), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
val_dl   = DataLoader(ReviewDataset(val_df),   batch_size=32,          shuffle=False, num_workers=0, pin_memory=False)
test_dl  = DataLoader(ReviewDataset(test_df),  batch_size=32,          shuffle=False, num_workers=0, pin_memory=False)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model.to(DEVICE)
log(f"Parametre: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
log(f"VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

criterion    = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTH)
optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
total_steps  = (len(train_dl) // ACCUM_STEPS) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
scaler       = torch.amp.GradScaler("cuda")
log(f"LR={LR} | Effective batch={BATCH_SIZE*ACCUM_STEPS} | Label smooth={LABEL_SMOOTH} | AMP=ON")


def evaluate(loader):
    model.eval()
    preds, labels, probas = [], [], []
    with torch.no_grad():
        for batch, y in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.amp.autocast("cuda"):
                out = model(**batch)
            proba = torch.softmax(out.logits.float(), dim=-1).cpu().tolist()
            pred  = out.logits.argmax(-1).cpu().tolist()
            preds.extend(pred); labels.extend(y.tolist()); probas.extend(proba)
    return preds, labels, probas


best_val_f1, best_state = 0.0, None
patience_cnt = 0
history      = []
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

log(f"\nEgitim: {EPOCHS} epoch | {total_steps} adim\n{'='*55}")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (batch, labels) in enumerate(train_dl):
        batch  = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = labels.to(DEVICE)
        with torch.amp.autocast("cuda"):
            logits = model(**batch).logits
            loss   = criterion(logits, labels) / ACCUM_STEPS
        scaler.scale(loss).backward()
        total_loss += loss.item() * ACCUM_STEPS

        if (step + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            scheduler.step(); optimizer.zero_grad()

        if (step + 1) % 200 == 0:
            log(f"  Epoch {epoch+1} | Step {step+1}/{len(train_dl)} | Loss: {total_loss/(step+1):.4f}")

    val_preds, val_labels, _ = evaluate(val_dl)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1  = f1_score(val_labels, val_preds, average="macro")
    history.append({"epoch": epoch+1, "loss": total_loss/len(train_dl), "val_acc": val_acc, "val_f1": val_f1})
    log(f"\nEpoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_dl):.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

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

# Test
model.load_state_dict(best_state)
del best_state; torch.cuda.empty_cache()
model.save_pretrained(str(CKPT_DIR))
tokenizer.save_pretrained(str(CKPT_DIR))

test_preds, test_labels, test_probas = evaluate(test_dl)
metrics = compute_classification_metrics(test_labels, test_preds, test_probas)
print_metrics(metrics, "XLM-RoBERTa (fine-tuned)")
log("")

metrics["history"] = history
results = {"xlm_roberta": metrics}
with open(RESULTS_DIR / "xlm_roberta_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
log("Sonuclar: results/xlm_roberta_results.json")
log("EGITIM TAMAMLANDI.")
