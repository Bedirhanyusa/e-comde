"""
XLM-RoBERTa v2 — Optimize edilmis egitim
MAX_LEN=192, daha dusuk LR, cosine schedule, label smoothing, 8 epoch
Onceki: 74.26% F1 (128 token, 6 epoch, linear schedule)
Hedef : >%78 F1
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
    get_cosine_schedule_with_warmup,
    logging as hf_logging,
)
hf_logging.set_verbosity_error()
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    matthews_corrcoef, cohen_kappa_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from pathlib import Path

MODEL_NAME  = "xlm-roberta-base"
NUM_LABELS  = 3
MAX_LEN     = 192
SEED        = 42
LR          = 6e-6
EPOCHS      = 8
BATCH_SIZE  = 8
ACCUM       = 4      # effective batch = 32
PATIENCE    = 4
WARMUP_RATIO= 0.08
LABEL_SMOOTH= 0.04

CKPT_DIR    = Path("models/xlm_roberta_v2_finetuned")
RESULTS_DIR = Path("results")
LABEL_NAMES = ["olumsuz", "notr", "olumlu"]

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

set_seed(SEED)
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
scaler  = torch.amp.GradScaler("cuda") if USE_AMP else None

log(f"Cihaz: {DEVICE} | Model: {MODEL_NAME}")
if torch.cuda.is_available():
    log(f"GPU  : {torch.cuda.get_device_name(0)}")
    log(f"VRAM : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

# ── Veri ─────────────────────────────────────────────────────────
v4_dir = Path("data/processed_v4")
if not v4_dir.exists():
    log("HATA: data/processed_v4 bulunamadi.")
    exit(1)

train_df = pd.read_csv(v4_dir / "train.csv")
val_df   = pd.read_csv(v4_dir / "val.csv")
test_df  = pd.read_csv(v4_dir / "test.csv")
log(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

log(f"Tokenizer yukleniyor: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class ReviewDataset(Dataset):
    def __init__(self, df):
        texts  = df["Review"].astype(str).tolist()
        self.labels = torch.tensor(df["label"].tolist(), dtype=torch.long)
        ids_list, mask_list = [], []
        for i in range(0, len(texts), 256):
            enc = tokenizer(texts[i:i+256], truncation=True, padding="max_length",
                            max_length=MAX_LEN, return_tensors="pt")
            ids_list.append(enc["input_ids"]); mask_list.append(enc["attention_mask"])
        self.input_ids      = torch.cat(ids_list)
        self.attention_mask = torch.cat(mask_list)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i], "attention_mask": self.attention_mask[i]}, self.labels[i]

train_dl = DataLoader(ReviewDataset(train_df), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
val_dl   = DataLoader(ReviewDataset(val_df),   batch_size=24,         shuffle=False, num_workers=0, pin_memory=False)
test_dl  = DataLoader(ReviewDataset(test_df),  batch_size=24,         shuffle=False, num_workers=0, pin_memory=False)

log(f"Model yukleniyor: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model.to(DEVICE)
if torch.cuda.is_available():
    log(f"VRAM kullanim: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

counts  = train_df["label"].value_counts().sort_index().values
weights = torch.tensor(counts.sum() / (NUM_LABELS * counts), dtype=torch.float32)

criterion    = nn.CrossEntropyLoss(weight=weights.to(DEVICE), label_smoothing=LABEL_SMOOTH)
optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps  = (len(train_dl) // ACCUM) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

log(f"\nEgitim basliyor | LR={LR} | MAX_LEN={MAX_LEN} | Eff.batch={BATCH_SIZE*ACCUM} | AMP={USE_AMP}")
log(f"Adimlar: {total_steps} | Warmup: {warmup_steps}")

def evaluate(model, loader):
    model.eval()
    preds, labels_all, probas_all = [], [], []
    with torch.no_grad():
        for batch, y in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                out = model(**batch)
            prob = torch.softmax(out.logits.float(), dim=-1).cpu().tolist()
            pred = out.logits.argmax(-1).cpu().tolist()
            preds.extend(pred); labels_all.extend(y.tolist()); probas_all.extend(prob)
    return accuracy_score(labels_all, preds), f1_score(labels_all, preds, average="macro"), preds, labels_all, probas_all

best_f1, best_state, pat_cnt = 0.0, None, 0
history = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (batch, labels) in enumerate(train_dl):
        batch  = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = labels.to(DEVICE)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(**batch).logits
            loss   = criterion(logits, labels) / ACCUM

        if USE_AMP:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * ACCUM

        if (step + 1) % ACCUM == 0:
            if USE_AMP:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % 300 == 0:
            log(f"  E{epoch+1} | Step {step+1}/{len(train_dl)} | Loss: {total_loss/(step+1):.4f}")

    avg_loss = total_loss / len(train_dl)
    val_acc, val_f1, _, _, _ = evaluate(model, val_dl)
    history.append({"epoch": epoch+1, "loss": round(avg_loss,4),
                     "val_acc": round(val_acc,4), "val_f1": round(val_f1,4)})
    log(f"\nEpoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1    = val_f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        pat_cnt    = 0
        log(f"  >> En iyi model guncellendi (F1={val_f1:.4f})")
    else:
        pat_cnt += 1
        log(f"  -- patience {pat_cnt}/{PATIENCE}")
        if pat_cnt >= PATIENCE:
            log(f"  >> Erken durdurma (epoch {epoch+1})")
            break
    log("")

model.load_state_dict(best_state)
del best_state; torch.cuda.empty_cache()

# ── Test degerlendirme ────────────────────────────────────────────
log("\n" + "="*60)
log("TEST DEGERLENDIRME")
log("="*60)

test_acc, test_f1, test_preds, test_labels, test_probas = evaluate(model, test_dl)
log(f"Test Accuracy : {test_acc:.4f}")
log(f"Test Macro-F1 : {test_f1:.4f}")
log(classification_report(test_labels, test_preds, target_names=LABEL_NAMES, digits=4))

mcc   = matthews_corrcoef(test_labels, test_preds)
kappa = cohen_kappa_score(test_labels, test_preds)
y_bin = label_binarize(test_labels, classes=[0,1,2])
auc   = roc_auc_score(y_bin, np.array(test_probas), average="macro", multi_class="ovr")
per_class_f1 = f1_score(test_labels, test_preds, average=None, labels=[0,1,2])
per_class    = {LABEL_NAMES[i]: round(float(per_class_f1[i]),4) for i in range(3)}

log(f"MCC        : {mcc:.4f}")
log(f"Cohen Kappa: {kappa:.4f}")
log(f"ROC-AUC    : {auc:.4f}")
log(f"Per-class F1: {per_class}")

CKPT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(CKPT_DIR))
tokenizer.save_pretrained(str(CKPT_DIR))
log(f"\nCheckpoint: {CKPT_DIR}")

results = {
    "xlm_roberta_v2": {
        "accuracy":      round(test_acc, 4),
        "macro_f1":      round(test_f1, 4),
        "mcc":           round(mcc, 4),
        "kappa":         round(kappa, 4),
        "roc_auc_macro": round(auc, 4),
        "per_class":     per_class,
        "best_val_f1":   round(best_f1, 4),
        "history":       history,
        "config": {"lr": LR, "epochs": EPOCHS, "max_len": MAX_LEN,
                   "label_smooth": LABEL_SMOOTH, "base_model": MODEL_NAME},
    }
}
with open(RESULTS_DIR / "xlm_roberta_v2_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
log("Sonuclar: results/xlm_roberta_v2_results.json")
log("\nEGITIM TAMAMLANDI (XLM-RoBERTa v2).")
