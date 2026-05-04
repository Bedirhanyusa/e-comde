"""
Aşama 2: Baseline modelleri çalıştır ve sonuçları kaydet.
TF-IDF + LR  ve  BiLSTM — MCC, Kappa, ROC-AUC dahil tam metrikler
"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def log(msg): print(msg, flush=True)

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              matthews_corrcoef, cohen_kappa_score, roc_auc_score)
from sklearn.preprocessing import label_binarize

LABEL_NAMES = ["olumsuz", "notr", "olumlu"]

# ── Veri yükle ──────────────────────────────────────────────────
train = pd.read_csv("data/processed/train.csv")
val   = pd.read_csv("data/processed/val.csv")
test  = pd.read_csv("data/processed/test.csv")

X_train, y_train = train["Review"].astype(str), train["label"]
X_val,   y_val   = val["Review"].astype(str),   val["label"]
X_test,  y_test  = test["Review"].astype(str),  test["label"]

results = {}

# ── 1. TF-IDF + Logistic Regression ─────────────────────────────
print("=" * 60)
print("1. TF-IDF + Logistic Regression")
print("=" * 60)

tfidf_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=100_000,
        analyzer="char_wb",       # Türkçe morfoloji için karakter n-gram
        sublinear_tf=True,
        min_df=2,
    )),
    ("clf", LogisticRegression(
        C=5.0,
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
        n_jobs=-1,
    )),
])

tfidf_pipeline.fit(X_train, y_train)

for split_name, X, y in [("Val", X_val, y_val), ("Test", X_test, y_test)]:
    preds = tfidf_pipeline.predict(X)
    acc = accuracy_score(y, preds)
    mf1 = f1_score(y, preds, average="macro")
    print(f"\n{split_name} — Accuracy: {acc:.4f} | Macro-F1: {mf1:.4f}")
    print(classification_report(y, preds, target_names=LABEL_NAMES, digits=4))

test_preds = tfidf_pipeline.predict(X_test)
test_probas = tfidf_pipeline.predict_proba(X_test)
y_bin = label_binarize(y_test, classes=[0, 1, 2])
results["tfidf_lr"] = {
    "accuracy": round(accuracy_score(y_test, test_preds), 4),
    "macro_f1": round(f1_score(y_test, test_preds, average="macro"), 4),
    "mcc":      round(matthews_corrcoef(y_test, test_preds), 4),
    "kappa":    round(cohen_kappa_score(y_test, test_preds), 4),
    "roc_auc_macro":  round(roc_auc_score(y_bin, test_probas, multi_class="ovr", average="macro"), 4),
}

import joblib
Path("models").mkdir(exist_ok=True)
joblib.dump(tfidf_pipeline, "models/tfidf_lr.pkl")
print("Model kaydedildi: models/tfidf_lr.pkl")

# ── 2. BiLSTM ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. BiLSTM (PyTorch)")
print("=" * 60)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cihaz: {DEVICE}")

# Vocab
def build_vocab(texts, min_freq=2):
    counter = Counter(t for text in texts for t in text.lower().split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, c in counter.items():
        if c >= min_freq:
            vocab[w] = len(vocab)
    return vocab

vocab = build_vocab(X_train.tolist())
print(f"Vocab boyutu: {len(vocab):,}")

MAX_LEN = 64

class ReviewDS(Dataset):
    def __init__(self, texts, labels):
        self.labels = labels.tolist()
        self.seqs = []
        for t in texts:
            ids = [vocab.get(w, 1) for w in t.lower().split()[:MAX_LEN]]
            ids += [0] * (MAX_LEN - len(ids))
            self.seqs.append(ids)

    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return (torch.tensor(self.seqs[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long))

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden=192, num_classes=3, dropout=0.4):
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True,
                            bidirectional=True, num_layers=2, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        e = self.drop(self.emb(x))
        _, (h, _) = self.lstm(e)
        out = torch.cat([h[-2], h[-1]], dim=-1)
        return self.fc(self.drop(out))

train_ds = ReviewDS(X_train, y_train)
val_ds   = ReviewDS(X_val,   y_val)
test_ds  = ReviewDS(X_test,  y_test)

train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=256)
test_dl  = DataLoader(test_ds,  batch_size=256)

model = BiLSTM(len(vocab)).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
criterion = nn.CrossEntropyLoss()

best_val_f1 = 0.0
best_state  = None

for epoch in range(15):
    model.train()
    total_loss = 0
    for x, y in train_dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    model.eval()
    preds_all, true_all = [], []
    with torch.no_grad():
        for x, y in val_dl:
            p = model(x.to(DEVICE)).argmax(1).cpu().tolist()
            preds_all.extend(p); true_all.extend(y.tolist())

    val_f1  = f1_score(true_all, preds_all, average="macro")
    val_acc = accuracy_score(true_all, preds_all)
    print(f"Epoch {epoch+1:2d}/15 | Loss: {total_loss/len(train_dl):.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state  = {k: v.clone() for k, v in model.state_dict().items()}

# Test evaluation with best model
model.load_state_dict(best_state)
model.eval()
preds_all, true_all = [], []
with torch.no_grad():
    for x, y in test_dl:
        p = model(x.to(DEVICE)).argmax(1).cpu().tolist()
        preds_all.extend(p); true_all.extend(y.tolist())

test_acc = accuracy_score(true_all, preds_all)
test_f1  = f1_score(true_all, preds_all, average="macro")
print(f"\nTest — Accuracy: {test_acc:.4f} | Macro-F1: {test_f1:.4f}")
print(classification_report(true_all, preds_all, target_names=LABEL_NAMES, digits=4))

bilstm_probas_all = []
model.eval()
with torch.no_grad():
    for x, _ in test_dl:
        logits = model(x.to(DEVICE))
        proba = torch.softmax(logits, dim=-1).cpu().tolist()
        bilstm_probas_all.extend(proba)

y_bin2 = label_binarize(true_all, classes=[0, 1, 2])
results["bilstm"] = {
    "accuracy": round(test_acc, 4),
    "macro_f1": round(test_f1, 4),
    "mcc":      round(matthews_corrcoef(true_all, preds_all), 4),
    "kappa":    round(cohen_kappa_score(true_all, preds_all), 4),
    "roc_auc_macro":  round(roc_auc_score(y_bin2, bilstm_probas_all, multi_class="ovr", average="macro"), 4),
}

torch.save(best_state, "models/bilstm.pt")
print("Model kaydedildi: models/bilstm.pt")

# ── Özet ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BASELINE SONUÇLARI (Test Seti)")
print("=" * 60)
print(f"{'Model':<30} {'Accuracy':>10} {'Macro-F1':>10} {'MCC':>8} {'ROC-AUC':>8}")
print("-" * 70)
for name, r in results.items():
    print(f"{name:<30} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} {r.get('mcc',0):>8.4f} {r.get('roc_auc_macro',0):>8.4f}")

Path("results").mkdir(exist_ok=True)
with open("results/baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSonuclar kaydedildi: results/baseline_results.json")
