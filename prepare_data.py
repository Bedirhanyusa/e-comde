import pandas as pd
import re
import unicodedata
from sklearn.model_selection import train_test_split
from pathlib import Path

CSV_PATH = "data/raw/turkish_ecommerce_reviews.csv"
OUT_DIR = Path("data/processed")
TARGET = 15170

df = pd.read_csv(CSV_PATH, encoding="latin-1")
df.columns = ["rating", "Review"]
print(f"Ham: {len(df)} satir")

df = df.dropna(subset=["rating", "Review"])
df = df[df["Review"].astype(str).str.strip().str.len() > 2]


def to_label(r):
    r = int(float(r))
    if r <= 2:
        return 0   # olumsuz
    if r == 3:
        return 1   # notr
    return 2       # olumlu


df["label"] = df["rating"].apply(to_label)


def normalize(text):
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df["Review"] = df["Review"].apply(normalize)

print("Sinif dagilimi (tam veri):")
counts = df["label"].value_counts().sort_index()
pcts = (counts / len(df) * 100).round(1)
for lbl, cnt, pct in zip(counts.index, counts.values, pcts.values):
    name = {0: "olumsuz", 1: "notr", 2: "olumlu"}[lbl]
    print(f"  {name} ({lbl}): {cnt}  ({pct}%)")

# Bildiriyle eslesen 15170 orneklik stratified sample
if len(df) > TARGET:
    df, _ = train_test_split(df, train_size=TARGET, stratify=df["label"], random_state=42)
    print(f"\nStratified sample: {len(df)} satir")

train, temp = train_test_split(df[["Review", "label"]], test_size=0.20, stratify=df["label"], random_state=42)
val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=42)

OUT_DIR.mkdir(parents=True, exist_ok=True)
train.to_csv(OUT_DIR / "train.csv", index=False)
val.to_csv(OUT_DIR / "val.csv", index=False)
test.to_csv(OUT_DIR / "test.csv", index=False)

print(f"\nBolme sonuclari:")
print(f"  train : {len(train)}")
print(f"  val   : {len(val)}")
print(f"  test  : {len(test)}")
print(f"\nOrnek yorum: {df['Review'].iloc[0][:100]}")
print("\nTAMAMLANDI")
