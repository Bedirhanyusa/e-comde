"""
Ablasyon calismasini calistirir ve Tablo 4'u (bildirideki) reproduce eder.
Konfigurasyonlar:
  0: BERTurk sifir-atis (fine-tune yok)
  1: BERTurk genel ince ayar (haber + sosyal medya)
  2: BERTurk alana ozgu ince ayar — E-ComDe (ana model)
"""
import torch
import pandas as pd
from transformers import AutoTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from src.data.dataset import TurkishReviewDataset
from src.evaluation.metrics import compute_classification_metrics


def evaluate_model(model_name_or_path: str, test_csv: str, max_len: int = 128, batch_size: int = 32) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = BertForSequenceClassification.from_pretrained(model_name_or_path, num_labels=3)
    model.to(device)
    model.eval()

    df = pd.read_csv(test_csv)
    texts = df["Review"].tolist()
    labels = df["label"].tolist()

    ds = TurkishReviewDataset(texts, labels, tokenizer, max_len)
    loader = DataLoader(ds, batch_size=batch_size)

    all_preds = []
    with torch.no_grad():
        for batch, _ in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            all_preds.extend(logits.argmax(dim=-1).tolist())

    return compute_classification_metrics(labels, all_preds)


def run_ablation(test_csv: str = "data/processed/test.csv"):
    configs = [
        ("BERTurk sifir-atis", "dbmdz/bert-base-turkish-cased"),
        ("BERTurk genel ince ayar", "models/berturk_general_finetuned"),
        ("BERTurk alana ozgu (E-ComDe)", "models/berturk_finetuned"),
    ]

    print(f"{'Yapilandirma':<40} {'Dogruluk':>10} {'Makro-F1':>10}")
    print("-" * 62)

    results = []
    for name, path in configs:
        try:
            metrics = evaluate_model(path, test_csv)
            print(f"{name:<40} {metrics['accuracy']:>10.4f} {metrics['macro_f1']:>10.4f}")
            results.append({"name": name, **metrics})
        except Exception as e:
            print(f"{name:<40} HATA: {e}")

    return results


if __name__ == "__main__":
    run_ablation()
