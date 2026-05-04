"""
Veri Seti v4 — Confidence bazli etiket temizleme + 18K per sinif
Yenilikler:
1. Her ornek icin 0-3 arasi confidence skoru hesaplanir
2. Sadece yuksek guvenilirlikli ornekler phase-2 egitiminde kullanilir
3. 18K per sinif = 54K toplam (daha fazla veri)
4. Cok kisa ve cok uzun yorumlar filtrelenir
"""
import os, re, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def log(msg): print(msg, flush=True)

SEED = 42
TARGET_PER_CLASS = 18_000   # 18K per sinif = 54K toplam
MIN_CONFIDENCE   = 1        # 0=dusuk, 1=orta, 2=yuksek, 3=cok yuksek
np.random.seed(SEED)

# ── Güçlü duygu kelimeleri (gürültü filtreleme için) ─────────────
STRONG_POS = [
    "harika", "mükemmel", "muhteşem", "süper", "fevkalade",
    "çok güzel", "çok iyi", "çok memnun", "tavsiye ederim",
    "kesinlikle alın", "pişman değilim", "bayıldım", "şahane",
    "kusursuz", "müthiş", "inanılmaz", "memnunum", "harikasınız",
]
STRONG_NEG = [
    "berbat", "rezalet", "korkunç", "iğrenç", "çöp",
    "kesinlikle almayın", "para kaybı", "çok kötü", "en kötü",
    "aldatıldım", "dolandırıldım", "sahte", "bozuk geldi",
    "iade ettim", "çok memnun değilim", "hiç beğenmedim",
    "pişmanım", "hüsran", "kalitesiz", "mahvoldu", "işe yaramaz",
]
MILD_POS = ["iyi", "güzel", "beğendim", "memnun", "kaliteli", "sağlam", "hızlı geldi"]
MILD_NEG = ["kötü", "bozuk", "geç", "pahalı", "sorun", "eksik", "hayal kırıklığı"]

def smart_label_with_confidence(text: str, original_rating: int) -> tuple:
    """
    Etiket + confidence (0-3) döndürür.
    Confidence: 3=cok yuksek, 2=yuksek, 1=orta, 0=dusuk/belirsiz
    """
    text_lower = text.lower()
    strong_pos = sum(1 for w in STRONG_POS if w in text_lower)
    strong_neg = sum(1 for w in STRONG_NEG if w in text_lower)
    mild_pos   = sum(1 for w in MILD_POS   if w in text_lower)
    mild_neg   = sum(1 for w in MILD_NEG   if w in text_lower)

    if original_rating == 1:
        return 0, 3  # 1★: kesinlikle olumsuz
    if original_rating == 2:
        conf = 3 if strong_neg > 0 else (2 if mild_neg > 0 else 2)
        return 0, conf
    if original_rating == 5:
        return 2, 3  # 5★: kesinlikle olumlu
    if original_rating == 4:
        conf = 3 if strong_pos > 0 else (2 if mild_pos > 0 else 2)
        return 2, conf

    # 3★ — en zor durum
    # Celisken sinyal varsa gürültülü — düsük confidence
    if strong_pos > 0 and strong_neg > 0:
        return 1, 0  # celisken — çok gürültülü, atla
    if strong_pos > strong_neg and strong_pos >= 2:
        return 2, 2  # güçlü pozitif signal
    if strong_neg > strong_pos and strong_neg >= 2:
        return 0, 2  # güçlü negatif signal
    if strong_pos == 1 and strong_neg == 0:
        return 2, 1
    if strong_neg == 1 and strong_pos == 0:
        return 0, 1
    if mild_pos > 0 and mild_neg == 0:
        return 1, 1  # hafif pozitif ama nötr
    if mild_neg > 0 and mild_pos == 0:
        return 1, 1  # hafif negatif ama nötr
    if mild_pos == 0 and mild_neg == 0:
        return 1, 2  # hic sinyal yok — gercekten nötr (yüksek confidence)
    return 1, 1  # karma sinyal — orta confidence

def smart_label(text: str, original_rating: int) -> int:
    label, _ = smart_label_with_confidence(text, original_rating)
    return label

# ── Kaynak 1: Trendyol 582K ──────────────────────────────────────
log("Kaynak 1: Trendyol yukleniyor...")
trendyol_path = None
for candidate in [
    Path("data/raw/trendyol/trendyol_reviews_20251212_000835.csv"),  # gercek konum
    Path("data/raw/trendyol_582k.csv"),
    Path("data/raw/Trendyol-22k-comment-dataset-v11_processed.csv"),
]:
    if candidate.exists():
        trendyol_path = candidate
        break
if trendyol_path is None:
    # Son care: data/raw alt dizinlerini tara
    for p in Path("data/raw").rglob("*.csv"):
        try:
            df_test = pd.read_csv(p, nrows=3, encoding="utf-8", on_bad_lines="skip")
            cols = [c.lower() for c in df_test.columns]
            if any(k in c for k in ("point","rating","puan") for c in cols):
                trendyol_path = p
                break
        except Exception:
            pass

dfs = []
try:
    if trendyol_path is None:
        raise FileNotFoundError("Trendyol dosyasi bulunamadi")
    df_tr = pd.read_csv(trendyol_path, encoding="utf-8", on_bad_lines="skip")
    log(f"  Trendyol: {trendyol_path.name} | kolonlar: {list(df_tr.columns)}")
    # Kolon isimlerini standartlastir
    col_map = {}
    for c in df_tr.columns:
        cl = c.lower()
        if any(k in cl for k in ("review","yorum","comment","text")):
            col_map[c] = "Review"
        elif any(k in cl for k in ("point","rating","puan","star","yildiz")):
            col_map[c] = "rating"
    df_tr = df_tr.rename(columns=col_map)
    if "Review" in df_tr.columns and "rating" in df_tr.columns:
        df_tr = df_tr[["Review", "rating"]].dropna()
        df_tr["rating"] = pd.to_numeric(df_tr["rating"], errors="coerce")
        df_tr = df_tr.dropna(subset=["rating"])
        df_tr["rating"] = df_tr["rating"].astype(int).clip(1, 5)
        lc = df_tr.apply(lambda r: smart_label_with_confidence(str(r["Review"]), r["rating"]), axis=1)
        df_tr["label"]      = lc.apply(lambda x: x[0])
        df_tr["confidence"] = lc.apply(lambda x: x[1])
        dfs.append(df_tr[["Review", "label", "confidence"]])
        log(f"  Trendyol: {len(df_tr):,} yorum | Dagilim: {dict(df_tr.label.value_counts().sort_index())}")
except Exception as e:
    log(f"  Trendyol yuklenemedi: {e}")

# ── Kaynak 2: turkish_ecommerce_reviews.csv ───────────────────────
log("Kaynak 2: Turkish e-commerce reviews...")
for fname in ["turkish_ecommerce_reviews.csv", "turkish_product_reviews.csv"]:
    fpath = Path(f"data/raw/{fname}")
    if fpath.exists():
        try:
            df2 = pd.read_csv(fpath, encoding="latin-1")
            col_map2 = {}
            for c in df2.columns:
                cl = c.lower()
                if "review" in cl or "yorum" in cl or "text" in cl:
                    col_map2[c] = "Review"
                elif "rating" in cl or "puan" in cl or "star" in cl:
                    col_map2[c] = "rating"
                elif "label" in cl or "sentiment" in cl:
                    col_map2[c] = "label_raw"
            df2 = df2.rename(columns=col_map2)
            if "Review" in df2.columns:
                if "rating" in df2.columns:
                    df2["rating"] = pd.to_numeric(df2["rating"], errors="coerce")
                    df2 = df2.dropna(subset=["Review", "rating"])
                    df2["rating"] = df2["rating"].astype(int).clip(1, 5)
                    lc2 = df2.apply(lambda r: smart_label_with_confidence(str(r["Review"]), r["rating"]), axis=1)
                    df2["label"]      = lc2.apply(lambda x: x[0])
                    df2["confidence"] = lc2.apply(lambda x: x[1])
                elif "label_raw" in df2.columns:
                    label_map = {"negative": 0, "notr": 1, "neutral": 1,
                                 "olumsuz": 0, "nötr": 1, "olumlu": 2, "positive": 2}
                    df2["label"] = df2["label_raw"].astype(str).str.lower().map(label_map)
                    df2 = df2.dropna(subset=["label"])
                    df2["label"] = df2["label"].astype(int)
                    df2["confidence"] = 2
                else:
                    continue
                dfs.append(df2[["Review", "label", "confidence"]])
                log(f"  {fname}: {len(df2):,} | Dagilim: {dict(df2.label.value_counts().sort_index())}")
        except Exception as e:
            log(f"  {fname} yuklenemedi: {e}")

# ── Kaynak 3: ABSA/Hepsiburada ───────────────────────────────────
log("Kaynak 3: Hepsiburada ABSA...")
try:
    rev = pd.read_csv("data/raw/absa/reviews.csv")
    rev = rev[["review_text", "rating"]].dropna()
    rev = rev.rename(columns={"review_text": "Review"})
    rev["rating"] = pd.to_numeric(rev["rating"], errors="coerce")
    rev = rev.dropna(subset=["rating"])
    rev["rating"] = rev["rating"].astype(int).clip(1, 5)
    lc_rev = rev.apply(lambda r: smart_label_with_confidence(str(r["Review"]), r["rating"]), axis=1)
    rev["label"]      = lc_rev.apply(lambda x: x[0])
    rev["confidence"] = lc_rev.apply(lambda x: x[1])
    dfs.append(rev[["Review", "label", "confidence"]])
    log(f"  Hepsiburada: {len(rev):,} | Dagilim: {dict(rev.label.value_counts().sort_index())}")
except Exception as e:
    log(f"  Hepsiburada yuklenemedi: {e}")

if not dfs:
    log("HATA: Hic kaynak yuklenemedi!")
    exit(1)

# ── Birlestir ve kalite filtresi uygula ──────────────────────────
log("\nVeriler birlestiriliyor...")
all_df = pd.concat(dfs, ignore_index=True)
all_df["Review"] = all_df["Review"].astype(str).str.strip()
if "confidence" not in all_df.columns:
    all_df["confidence"] = 1

# Temel kalite filtreleri
all_df = all_df[all_df["Review"].str.len().between(15, 1200)]
all_df = all_df.drop_duplicates(subset=["Review"])
all_df = all_df.dropna(subset=["label"])
all_df["label"] = all_df["label"].astype(int)
log(f"Temel filtre sonrasi: {len(all_df):,} yorum")

# Confidence filtresi — dusuk guvenilirlikli ornekleri cikar
before = len(all_df)
all_df = all_df[all_df["confidence"] >= MIN_CONFIDENCE]
log(f"Confidence>={MIN_CONFIDENCE} filtresi: {len(all_df):,} yorum ({before-len(all_df):,} cikarildi)")
log(f"Sinif dagilimi: {dict(all_df.label.value_counts().sort_index())}")
log(f"Confidence dagilimi: {dict(all_df.confidence.value_counts().sort_index())}")

# ── Dengeli ornekleme ────────────────────────────────────────────
available = all_df.groupby("label").size().to_dict()
log(f"\nMevcut ornekler: {available}")

per_class = min(TARGET_PER_CLASS, min(available.values()))
log(f"Kullanilacak per sinif: {per_class:,} (hedef: {TARGET_PER_CLASS:,})")

sampled = []
for label in [0, 1, 2]:
    cls_df = all_df[all_df["label"] == label].sample(per_class, random_state=SEED)
    sampled.append(cls_df)

balanced = pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=SEED)
log(f"Dengeli veri: {len(balanced):,} yorum ({per_class:,} per sinif)")

# ── Train/Val/Test bolme (%80/%10/%10) ───────────────────────────
train_val, test = train_test_split(balanced, test_size=0.10, stratify=balanced["label"], random_state=SEED)
train, val      = train_test_split(train_val, test_size=0.1111, stratify=train_val["label"], random_state=SEED)

log(f"\nBolme:")
log(f"  Train: {len(train):,} | {dict(train.label.value_counts().sort_index())}")
log(f"  Val:   {len(val):,}   | {dict(val.label.value_counts().sort_index())}")
log(f"  Test:  {len(test):,}  | {dict(test.label.value_counts().sort_index())}")

# ── Kaydet ──────────────────────────────────────────────────────
out_dir = Path("data/processed_v4")
out_dir.mkdir(parents=True, exist_ok=True)

train.to_csv(out_dir / "train.csv", index=False)
val.to_csv(out_dir / "val.csv",   index=False)
test.to_csv(out_dir / "test.csv",  index=False)

log(f"\nVeri seti v4 kaydedildi: {out_dir}/")
log(f"Toplam: {len(balanced):,} | per sinif: {per_class:,}")
