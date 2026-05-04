"""
Veri Hazırlama v2 — Trendyol 582K (kategori destekli)
Strateji:
  - Trendyol ana veri seti: kategori + yorum + puan
  - 5 kategori: Elektronik, Giyim ve Aksesuar, Kitap/Hobi, Kozmetik, (Genel)
  - Her kategoriden dengeli sentiment örnekleme
  - Toplam hedef: ~22,000 (jüri için daha güçlü)
"""

import re
import unicodedata
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

TRENDYOL_PATH = "data/raw/trendyol/trendyol_reviews_20251212_000835.csv"
OUT_DIR = Path("data/processed")
SEED = 42

CATEGORY_MAP = {
    "Elektronik": "Elektronik",
    "Giyim": "Giyim ve Aksesuar",
    "Aksesuar": "Giyim ve Aksesuar",
    "Kitap": "Kitap / Hobi",
    "Kozmetik & Kişisel Bakım": "Kozmetik",
}

# Her kategori için hedef örnekler (sentiment başına)
# Hedef toplam ~15,000 - bildiriyle eslesen
PER_CATEGORY_PER_SENTIMENT = {
    "Elektronik":       {"olumsuz": 1200, "notr": 833, "olumlu": 1800},
    "Giyim ve Aksesuar":{"olumsuz": 1200, "notr": 833, "olumlu": 1800},
    "Kitap / Hobi":     {"olumsuz": 707,  "notr": 833, "olumlu": 1800},
    "Kozmetik":         {"olumsuz": 1200, "notr": 833, "olumlu": 1800},
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def point_to_label(p) -> int | None:
    try:
        p = int(float(p))
    except (ValueError, TypeError):
        return None
    if p <= 2:
        return 0   # olumsuz
    if p == 3:
        return 1   # notr
    return 2       # olumlu


LABEL_NAME = {0: "olumsuz", 1: "notr", 2: "olumlu"}

print("Trendyol veri seti yukleniyor...")
df = pd.read_csv(TRENDYOL_PATH, encoding="utf-8", on_bad_lines="skip",
                 usecols=["category", "point", "comment"])

df = df.dropna(subset=["category", "point", "comment"])
df["category_mapped"] = df["category"].map(CATEGORY_MAP)
df = df.dropna(subset=["category_mapped"])
df["Review"] = df["comment"].apply(normalize)
df = df[df["Review"].str.len() > 5]
df["label"] = df["point"].apply(point_to_label)
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)
df["category"] = df["category_mapped"]

print(f"Toplam kullanilabilir: {len(df):,}")
print("\nKategori x Sentiment dagilimi:")
pivot = df.groupby(["category", "label"]).size().unstack(fill_value=0)
pivot.columns = ["olumsuz", "notr", "olumlu"]
print(pivot.to_string())

# Stratified sampling
sampled_parts = []
for cat, targets in PER_CATEGORY_PER_SENTIMENT.items():
    cat_df = df[df["category"] == cat].copy()
    for lbl_name, n in targets.items():
        lbl_id = {"olumsuz": 0, "notr": 1, "olumlu": 2}[lbl_name]
        pool = cat_df[cat_df["label"] == lbl_id]
        take = min(n, len(pool))
        sampled_parts.append(pool.sample(n=take, random_state=SEED))

final = pd.concat(sampled_parts).drop_duplicates()
final = final[["Review", "label", "category"]].reset_index(drop=True)

print(f"\nSampling sonrasi toplam: {len(final):,}")
print("\nFinal dagilim:")
print(final["label"].map(LABEL_NAME).value_counts())
print("\nKategori dagilimi:")
print(final["category"].value_counts())

# Pct check
pct = final["label"].value_counts(normalize=True).sort_index() * 100
for i, name in LABEL_NAME.items():
    print(f"  {name}: %{pct.get(i, 0):.1f}")

# Train / Val / Test split (80/10/10), stratify by label
train, temp = train_test_split(final, test_size=0.20, stratify=final["label"], random_state=SEED)
val, test   = train_test_split(temp,  test_size=0.50, stratify=temp["label"],  random_state=SEED)

OUT_DIR.mkdir(parents=True, exist_ok=True)
train.to_csv(OUT_DIR / "train.csv", index=False)
val.to_csv(OUT_DIR / "val.csv",   index=False)
test.to_csv(OUT_DIR / "test.csv",  index=False)

print(f"\nBolme:")
print(f"  train : {len(train):,}")
print(f"  val   : {len(val):,}")
print(f"  test  : {len(test):,}")
print("\nTAMAMLANDI — data/processed/ yazildi.")
