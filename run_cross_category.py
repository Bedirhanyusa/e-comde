"""
Test kümesinden gerçek çapraz-kategori F1 hesaplar.
Anahtar kelime tabanlı kategori ataması → BERTurk v3b ile tahmin → F1
"""
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
def log(msg):
    try: print(msg, flush=True)
    except UnicodeEncodeError: print(msg.encode("utf-8","replace").decode("ascii","replace"), flush=True)

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
from sklearn.metrics import f1_score
hf_logging.set_verbosity_error()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Cihaz: {DEVICE}")

# ── Veri yükle ─────────────────────────────────────────────────────
data_dir = Path("data/processed_v4")
test_df = pd.read_csv(data_dir / "test.csv", encoding="utf-8")
log(f"Test: {len(test_df)} yorum")

# ── Anahtar kelime tabanlı kategori ataması ────────────────────────
CATEGORY_KEYWORDS = {
    "Elektronik": [
        "telefon", "laptop", "bilgisayar", "tablet", "ekran", "şarj", "batarya",
        "kamera", "kulaklık", "klavye", "mouse", "monitör", "şarjlı", "pil",
        "hoparlör", "mikrofon", "tv", "televizyon", "anten", "kablo", "adaptör",
        "elektronik", "yazıcı", "tarayıcı", "hard disk", "ssd", "ram", "cpu",
        "işlemci", "grafik", "router", "modem"
    ],
    "Giyim ve Aksesuar": [
        "elbise", "pantolon", "gömlek", "tişört", "kazak", "mont", "ceket",
        "ayakkabı", "çanta", "kemer", "çorap", "iç çamaşır", "pijama", "etek",
        "bluz", "mayo", "bikini", "şort", "tayt", "bere", "eldiven", "eşarp",
        "kravat", "kıyafet", "giyim", "moda", "beden", "numara", "kumaş",
        "dikiş", "yaka", "manşet", "fermuar", "düğme"
    ],
    "Ev ve Yaşam": [
        "masa", "sandalye", "koltuk", "yatak", "dolap", "raf", "çerçeve",
        "mutfak", "tencere", "tava", "bıçak", "kaşık", "çatal", "tabak",
        "bardak", "temizlik", "deterjan", "havlu", "nevresim", "yastık",
        "çarşaf", "perde", "halı", "ayna", "lamba", "mobilya", "ev", "dekorasyon",
        "banyo", "tuvalet", "duş", "fırça", "süpürge", "çöp"
    ],
    "Kitap ve Hobi": [
        "kitap", "roman", "hikaye", "şiir", "ansiklopedi", "dergi", "dergi",
        "oyun", "oyuncak", "puzzle", "bulmaca", "satranç", "lego", "hobi",
        "müzik", "enstrüman", "gitar", "piyano", "keman", "spor", "futbol",
        "basketbol", "tenis", "yüzme", "koşu", "fitness", "bisiklet",
        "kamp", "dağcılık", "resim", "boya", "kalem", "defter"
    ],
    "Gıda ve İçecek": [
        "çikolata", "bisküvi", "kahve", "çay", "su", "meyve", "sebze",
        "et", "tavuk", "balık", "peynir", "yoğurt", "süt", "ekmek",
        "makarna", "pirinç", "bakliyat", "gıda", "içecek", "atıştırma",
        "kuruyemiş", "fındık", "badem", "ceviz", "vitamin", "takviye",
        "protein", "diyet", "organik", "doğal", "tat", "lezzet", "taze"
    ],
}

def assign_category(text):
    text_lower = str(text).lower()
    scores = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for kw in keywords if kw in text_lower)
    best_cat = max(scores, key=scores.get)
    return best_cat if scores[best_cat] > 0 else None

test_df["category"] = test_df["Review"].apply(assign_category)
cat_counts = test_df["category"].value_counts()
assigned = test_df["category"].notna().sum()
log(f"\nKategori ataması: {assigned}/{len(test_df)} yorum atandı")
for cat, cnt in cat_counts.items():
    log(f"  {cat}: {cnt}")

# ── BERTurk v3b modeli yükle ────────────────────────────────────────
MODEL_DIR = Path("models/berturk_v3_finetuned")
log(f"\nModel yükleniyor: {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.to(DEVICE)
model.eval()

def predict_batch(texts, batch_size=32):
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, truncation=True, max_length=128,
                        padding=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
    return np.array(all_preds)

# ── Kategori bazında F1 hesapla ─────────────────────────────────────
results = {}
log("\n" + "="*60)
log("ÇAPRAZ-KATEGORİ F1 SONUÇLARI")
log("="*60)

CATEGORIES = ["Elektronik", "Giyim ve Aksesuar", "Ev ve Yaşam", "Kitap ve Hobi", "Gıda ve İçecek"]

for cat in CATEGORIES:
    subset = test_df[test_df["category"] == cat].copy()
    if len(subset) < 20:
        log(f"[{cat}] — yetersiz örnek ({len(subset)}), atlanıyor")
        continue
    texts = subset["Review"].astype(str).tolist()
    labels = subset["label"].tolist()
    preds = predict_batch(texts)
    f1 = f1_score(labels, preds, average="macro")
    results[cat] = {"n": len(subset), "macro_f1": round(f1, 4)}
    log(f"[{cat}] — N={len(subset)} | Makro-F1: {f1:.4f} ({f1*100:.1f}%)")

# ── Kaydet ─────────────────────────────────────────────────────────
out = {"cross_category": results}
with open("results/cross_category_results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
log("\nSonuçlar: results/cross_category_results.json")
log("ÇAPRAZ-KATEGORİ ANALİZİ TAMAMLANDI.")
