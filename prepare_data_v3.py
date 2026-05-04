"""
Veri Hazırlama v3 — 3 kaynak birlestirilmis, ~72K dengeli veri seti
Kaynaklar:
  1. turkish_ecommerce_reviews.csv (49K, rating 1-5)
  2. Trendyol 582K (point 1-5, kategori bilgisi)
  3. absa/reviews.csv (54K, rating 1-5)

Strateji: her siniftan esit ornek al, olumlu:notr:olumsuz = 1:1:1
Hedef: ~72K toplam (24K x 3 sinif)
"""
import re, unicodedata
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

SEED    = 42
OUT_DIR = Path("data/processed")
TARGET_PER_CLASS = 10000   # her siniftan — toplam ~30K, egitim ~24K

LABEL_MAP = {0: "olumsuz", 1: "notr", 2: "olumlu"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rating_to_label(r) -> int | None:
    try:
        r = int(float(r))
    except (ValueError, TypeError):
        return None
    if r <= 2: return 0
    if r == 3: return 1
    return 2


# ── 1. turkish_ecommerce_reviews.csv ────────────────────────────
print("1. turkish_ecommerce_reviews.csv yukleniyor...")
try:
    df1 = pd.read_csv("data/raw/turkish_ecommerce_reviews.csv", encoding="utf-8")
except UnicodeDecodeError:
    df1 = pd.read_csv("data/raw/turkish_ecommerce_reviews.csv", encoding="latin-1")

df1 = df1.dropna(subset=["comment", "rating"])
df1["Review"]   = df1["comment"].apply(normalize)
df1["label"]    = df1["rating"].apply(rating_to_label)
df1["category"] = "Genel"
df1["source"]   = "turkish_ec"
df1 = df1[df1["Review"].str.len() > 5].dropna(subset=["label"])
df1["label"] = df1["label"].astype(int)
print(f"   {len(df1):,} satir | {df1['label'].value_counts().sort_index().to_dict()}")


# ── 2. Trendyol 582K ────────────────────────────────────────────
print("2. Trendyol yukleniyor...")
CATEGORY_MAP = {
    "Elektronik":              "Elektronik",
    "Giyim":                   "Giyim ve Aksesuar",
    "Aksesuar":                "Giyim ve Aksesuar",
    "Kitap":                   "Kitap / Hobi",
    "Kozmetik & Kişisel Bakım":"Kozmetik",
}
df2 = pd.read_csv(
    "data/raw/trendyol/trendyol_reviews_20251212_000835.csv",
    encoding="utf-8", on_bad_lines="skip",
    usecols=["category", "point", "comment"]
)
df2 = df2.dropna(subset=["category", "point", "comment"])
df2["category"] = df2["category"].map(CATEGORY_MAP)
df2 = df2.dropna(subset=["category"])
df2["Review"]   = df2["comment"].apply(normalize)
df2["label"]    = df2["point"].apply(rating_to_label)
df2["source"]   = "trendyol"
df2 = df2[df2["Review"].str.len() > 5].dropna(subset=["label"])
df2["label"] = df2["label"].astype(int)
print(f"   {len(df2):,} satir | {df2['label'].value_counts().sort_index().to_dict()}")


# ── 3. absa/reviews.csv ─────────────────────────────────────────
print("3. ABSA reviews yukleniyor...")
df3 = pd.read_csv("data/raw/absa/reviews.csv", encoding="utf-8")
df3 = df3.dropna(subset=["review_text", "rating"])
df3["Review"]   = df3["review_text"].apply(normalize)
df3["label"]    = df3["rating"].apply(rating_to_label)
df3["category"] = "Genel"
df3["source"]   = "absa"
df3 = df3[df3["Review"].str.len() > 5].dropna(subset=["label"])
df3["label"] = df3["label"].astype(int)
print(f"   {len(df3):,} satir | {df3['label'].value_counts().sort_index().to_dict()}")


# ── Birlestir ────────────────────────────────────────────────────
cols = ["Review", "label", "category", "source"]
all_df = pd.concat([df1[cols], df2[cols], df3[cols]], ignore_index=True)
all_df = all_df.drop_duplicates(subset=["Review"]).reset_index(drop=True)
print(f"\nBirlesmis toplam: {len(all_df):,}")
print(f"Sinif dagilimi: {all_df['label'].value_counts().sort_index().to_dict()}")

# ── Dengeli ornekleme ────────────────────────────────────────────
parts = []
for lbl in [0, 1, 2]:
    pool = all_df[all_df["label"] == lbl]
    n    = min(TARGET_PER_CLASS, len(pool))
    parts.append(pool.sample(n=n, random_state=SEED))
    print(f"  {LABEL_MAP[lbl]}: {n:,} alinacak ({len(pool):,} mevcut)")

final = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
final = final[["Review", "label", "category"]].copy()

print(f"\nFinal veri seti: {len(final):,} satir")
print(f"Sinif: {final['label'].value_counts().sort_index().to_dict()}")
print(f"Kategori: {final['category'].value_counts().to_dict()}")

# ── Train / Val / Test split 80/10/10 ────────────────────────────
train, temp = train_test_split(final, test_size=0.20, stratify=final["label"], random_state=SEED)
val, test   = train_test_split(temp,  test_size=0.50, stratify=temp["label"],  random_state=SEED)

OUT_DIR.mkdir(parents=True, exist_ok=True)
train.to_csv(OUT_DIR / "train.csv", index=False)
val.to_csv(OUT_DIR   / "val.csv",   index=False)
test.to_csv(OUT_DIR  / "test.csv",  index=False)

print(f"\nBolme:")
print(f"  train : {len(train):,}")
print(f"  val   : {len(val):,}")
print(f"  test  : {len(test):,}")
print("\nTAMAMLANDI — data/processed/ yazildi.")
