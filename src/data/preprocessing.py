import re
import unicodedata
import pandas as pd


TURKISH_CHARS = set("şğüöıçŞĞÜÖIÇ")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_and_clean(csv_path: str, text_col: str = "Review", label_col: str = "Score") -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    before = len(df)
    df = df.dropna(subset=[text_col, label_col])
    print(f"NaN temizlendi: {before - len(df)} satir kaldirildi.")

    df[text_col] = df[text_col].astype(str).apply(normalize_text)
    df = df[df[text_col].str.len() > 2].reset_index(drop=True)

    return df


def map_labels(df: pd.DataFrame, label_col: str = "Score") -> pd.DataFrame:
    """
    Kaggle veri kumesindeki ham skorlari 3 sinifa donusturur.
    1-2 → olumsuz (0), 3 → notr (1), 4-5 → olumlu (2)
    Veri kumesi zaten etiketliyse bu adim atlanabilir.
    """
    def score_to_label(score):
        try:
            s = int(float(score))
        except (ValueError, TypeError):
            return None
        if s <= 2:
            return 0  # olumsuz
        elif s == 3:
            return 1  # notr
        else:
            return 2  # olumlu

    df["label"] = df[label_col].apply(score_to_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df
