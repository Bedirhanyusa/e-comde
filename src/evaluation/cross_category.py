"""
Tablo 5 (bildirideki): Urun kategorisi bazinda makro-F1.
Test CSV'sinde 'category' kolonu olmasi gerekir.
"""
import pandas as pd
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from src.data.dataset import TurkishReviewDataset
from src.evaluation.metrics import compute_classification_metrics


CATEGORIES = ["Elektronik", "Giyim ve Aksesuar", "Ev ve Yasam", "Kitap / Hobi", "Gida ve Icecek"]


def evaluate_by_category(
    model_path: str = "models/berturk_finetuned",
    test_csv: str = "data/processed/test.csv",
    category_col: str = "category",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=3)
    model.to(device)
    model.eval()

    df = pd.read_csv(test_csv)
    if category_col not in df.columns:
        print(f"'{category_col}' kolonu bulunamadi. Test CSV'sine kategori bilgisi ekleyin.")
        return

    print(f"{'Kategori':<25} {'Ornek':>8} {'Makro-F1':>10}")
    print("-" * 45)

    for cat in CATEGORIES:
        sub = df[df[category_col] == cat]
        if len(sub) == 0:
            continue

        texts = sub["Review"].tolist()
        labels = sub["label"].tolist()
        ds = TurkishReviewDataset(texts, labels, tokenizer)
        loader = DataLoader(ds, batch_size=32)

        preds = []
        with torch.no_grad():
            for batch, _ in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                preds.extend(model(**batch).logits.argmax(dim=-1).tolist())

        metrics = compute_classification_metrics(labels, preds)
        print(f"{cat:<25} {len(sub):>8} {metrics['macro_f1']:>10.4f}")


if __name__ == "__main__":
    evaluate_by_category()
