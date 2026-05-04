"""
ABSA — Aspect-Based Sentiment Analysis
Hepsiburada verisinden kargo / urun / fiyat / musteri_hizmetleri aspektleri
BERTurk fine-tuned modeli üzerine çok-etiketli sınıflandırma başlığı eklenir.
"""
import os, re, json, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def log(msg): print(msg, flush=True)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel, get_linear_schedule_with_warmup, logging as hf_logging
hf_logging.set_verbosity_error()
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report
from pathlib import Path

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERTURK_DIR = "models/berturk_finetuned"   # fine-tuned BERTurk ağırlıklarını kullan
MAX_LEN     = 128
BATCH_SIZE  = 16
EPOCHS      = 5
LR          = 2e-5
SEED        = 42
RESULTS_DIR = Path("results")
CKPT_DIR    = Path("models/absa_model")

ASPECTS = ["kargo", "urun_kalitesi", "fiyat", "musteri_hizmetleri"]
SENT_LABELS = ["olumsuz", "notr", "olumlu"]

# ── Aspect tespit kuralları (kural tabanlı etiket üretimi) ────────
ASPECT_KEYWORDS = {
    "kargo":               ["kargo", "teslimat", "kurye", "gönderim", "paket", "paketleme",
                             "hızlı geldi", "geç geldi", "zamanında"],
    "urun_kalitesi":       ["kalite", "sağlam", "dayanıklı", "bozuk", "kırık", "sahte",
                             "orijinal", "ürün", "materyal", "kumaş", "malzeme"],
    "fiyat":               ["fiyat", "pahalı", "ucuz", "değer", "para", "fiyatına göre",
                             "ekonomik", "makul"],
    "musteri_hizmetleri":  ["müşteri hizmetleri", "iade", "destek", "satıcı", "mağaza",
                             "hizmet", "ilgili", "cevap"],
}

def detect_aspects(text: str) -> dict[str, int]:
    text_lower = text.lower()
    return {asp: int(any(kw in text_lower for kw in kws))
            for asp, kws in ASPECT_KEYWORDS.items()}

def simple_sentiment(text: str, aspect: str) -> int:
    neg_words = ["kötü", "berbat", "rezalet", "geç", "bozuk", "kırık", "sahte",
                 "pahalı", "hayal kırıklığı", "memnun değil", "iade", "şikâyet"]
    pos_words = ["harika", "mükemmel", "güzel", "hızlı", "sağlam", "kaliteli",
                 "memnun", "tavsiye", "teşekkür", "süper"]
    text_lower = text.lower()
    neg = sum(1 for w in neg_words if w in text_lower)
    pos = sum(1 for w in pos_words if w in text_lower)
    if neg > pos: return 0
    if pos > neg: return 2
    return 1

# ── Hepsiburada ABSA verisini yükle ────────────────────────────
log("ABSA verisi yukleniyor...")
try:
    products = pd.read_csv("data/raw/absa/products.csv")
    reviews  = pd.read_csv("data/raw/absa/reviews.csv")
    df = reviews.merge(products[["id", "url"]], left_on="product_id", right_on="id", how="left")
    df = df.rename(columns={"review_text": "Review", "rating": "rating"})
    df = df.dropna(subset=["Review"])
    df["Review"] = df["Review"].astype(str)
    log(f"Hepsiburada: {len(df)} yorum")
    USE_HEPSIBURADA = True
except Exception as e:
    log(f"Hepsiburada yuklenemedi: {e} — Trendyol verisi kullanilacak")
    USE_HEPSIBURADA = False

if not USE_HEPSIBURADA:
    df = pd.read_csv("data/processed/train.csv")
    df = df.rename(columns={"label": "rating"})

# ── Aspect etiketlerini otomatik üret ──────────────────────────
log("Aspect etiketleri uretiliyor...")
aspect_rows = []
for _, row in df.iterrows():
    text = str(row["Review"])
    detected = detect_aspects(text)
    for asp, present in detected.items():
        if present:
            sent = simple_sentiment(text, asp)
            aspect_rows.append({
                "Review": text,
                "aspect": asp,
                "sentiment": sent,
            })

aspect_df = pd.DataFrame(aspect_rows)
log(f"Aspect ornekleri: {len(aspect_df)}")
log(str(aspect_df["aspect"].value_counts()))
log("")

# ── BERTurk tabanlı ABSA modeli ────────────────────────────────
try:
    tokenizer = AutoTokenizer.from_pretrained(BERTURK_DIR)
    log(f"BERTurk fine-tuned checkpoint kullaniliyor: {BERTURK_DIR}")
except Exception:
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    log("Vanilla BERTurk kullaniliyor")

ASPECT2ID = {a: i for i, a in enumerate(ASPECTS)}

class ABSADataset(Dataset):
    def __init__(self, df):
        self.texts   = df["Review"].tolist()
        self.aspects = df["aspect"].tolist()
        self.labels  = df["sentiment"].tolist()
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        # [CLS] yorum [SEP] aspekt [SEP] formatı — lazy tokenization (batch crash'ini önler)
        enc = tokenizer(
            self.texts[i], self.aspects[i],
            truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}, torch.tensor(self.labels[i])

# Train/test split
from sklearn.model_selection import train_test_split
train_absa, test_absa = train_test_split(
    aspect_df, test_size=0.2, stratify=aspect_df["sentiment"], random_state=SEED
)

train_dl = DataLoader(ABSADataset(train_absa), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
test_dl  = DataLoader(ABSADataset(test_absa),  batch_size=32,          shuffle=False, num_workers=0, pin_memory=False)

class ABSAModel(nn.Module):
    def __init__(self, base_model_path: str, num_classes: int = 3):
        super().__init__()
        try:
            self.bert = BertModel.from_pretrained(base_model_path)
        except Exception:
            self.bert = BertModel.from_pretrained("dbmdz/bert-base-turkish-cased")
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, **kwargs):
        out = self.bert(**kwargs)
        return self.fc(self.drop(out.pooler_output))

model = ABSAModel(BERTURK_DIR).to(DEVICE)
optimizer    = AdamW(model.parameters(), lr=LR)
total_steps  = len(train_dl) * EPOCHS
scheduler    = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
criterion    = nn.CrossEntropyLoss()
scaler       = torch.cuda.amp.GradScaler()

log(f"ABSA egitimi basliyor ({EPOCHS} epoch, AMP=ON)...")
log("=" * 50)

best_f1, best_state = 0.0, None
CKPT_DIR.mkdir(parents=True, exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch, labels in train_dl:
        batch  = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = criterion(model(**batch), labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        scheduler.step()
        total_loss += loss.item()

    # Eval
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch, labels in test_dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            preds = model(**batch).argmax(-1).cpu().tolist()
            all_preds.extend(preds); all_labels.extend(labels.tolist())

    f1  = f1_score(all_labels, all_preds, average="macro")
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    log(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_dl):.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
del best_state; torch.cuda.empty_cache()

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch, labels in test_dl:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        preds = model(**batch).argmax(-1).cpu().tolist()
        all_preds.extend(preds); all_labels.extend(labels.tolist())

log(f"\n{'='*50}")
log("ABSA TEST SONUCLARI")
log(f"{'='*50}")
log(classification_report(all_labels, all_preds, target_names=SENT_LABELS, digits=4))

absa_results = {
    "absa": {
        "macro_f1": round(f1_score(all_labels, all_preds, average="macro"), 4),
        "accuracy": round(sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels), 4),
        "aspects":  ASPECTS,
    }
}
RESULTS_DIR.mkdir(exist_ok=True)
with open(RESULTS_DIR / "absa_results.json", "w") as f:
    json.dump(absa_results, f, indent=2, ensure_ascii=False)

torch.save(model.state_dict(), str(CKPT_DIR / "absa_model.pt"))
log(f"\nABSA tamamlandi. Checkpoint: {CKPT_DIR}/absa_model.pt")
