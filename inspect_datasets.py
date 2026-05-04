import pandas as pd

def inspect(path, enc='utf-8', label=""):
    try:
        df = pd.read_csv(path, encoding=enc, nrows=3)
    except Exception:
        try:
            df = pd.read_csv(path, encoding='latin-1', nrows=3)
            enc = 'latin-1'
        except Exception as e:
            print(f"{label}: HATA - {e}")
            return

    full = pd.read_csv(path, encoding=enc)
    print(f"\n{'='*60}")
    print(f"{label}  [{enc}]")
    print(f"Satir: {len(full)}  |  Kolonlar: {full.columns.tolist()}")
    print(full.head(2).to_string())

inspect("data/raw/turkish_ecommerce_reviews.csv", 'latin-1', "ANA VERI SETI")
inspect("data/raw/absa/products.csv", label="ABSA - products")
inspect("data/raw/absa/reviews.csv", label="ABSA - reviews")
inspect("data/raw/trendyol/trendyol_reviews_20251212_000835.csv", label="TRENDYOL 582K")
