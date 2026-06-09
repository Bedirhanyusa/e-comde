"""
BERTurk v3b test seti confusion matrix hesaplama.
Çıktı: frontend/public/berturk_v3_metrics.json
"""
import json
import sys
import os

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(__file__))
from src.models.sentiment_classifier import SentimentClassifier

MODEL_DIR = "models/berturk_v3_finetuned"
TEST_CSV  = "data/processed_v4/test.csv"
OUT_PATH  = "frontend/public/berturk_v3_metrics.json"
BATCH     = 64
MAX_LEN   = 192

LABEL_MAP = {0: "olumsuz", 1: "notr", 2: "olumlu"}

def batched_predict(clf, texts, batch_size=64):
    all_results = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        all_results.extend(clf.predict(chunk, max_len=MAX_LEN))
        print(f"  {min(i+batch_size, len(texts))}/{len(texts)} yorumlar işlendi...", flush=True)
    return all_results

def main():
    print("Test seti yükleniyor...")
    df = pd.read_csv(TEST_CSV)
    texts = df["Review"].astype(str).tolist()
    true_ids = df["label"].tolist()  # 0/1/2 int

    print(f"Model yükleniyor: {MODEL_DIR}")
    clf = SentimentClassifier(model_dir=MODEL_DIR)

    print("Tahmin yapılıyor (batch=64)...")
    results = batched_predict(clf, texts, BATCH)

    label_to_id = {"olumsuz": 0, "notr": 1, "olumlu": 2}
    pred_ids = [label_to_id[r["label"]] for r in results]

    cm = confusion_matrix(true_ids, pred_ids, labels=[0, 1, 2])
    report = classification_report(
        true_ids, pred_ids,
        labels=[0, 1, 2],
        target_names=["olumsuz", "notr", "olumlu"],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "confusion_matrix": cm.tolist(),
        "classes": ["olumsuz", "notr", "olumlu"],
        "per_class": {
            "olumsuz": {
                "precision": round(report["olumsuz"]["precision"], 4),
                "recall":    round(report["olumsuz"]["recall"],    4),
                "f1":        round(report["olumsuz"]["f1-score"],  4),
                "support":   int(report["olumsuz"]["support"]),
            },
            "notr": {
                "precision": round(report["notr"]["precision"], 4),
                "recall":    round(report["notr"]["recall"],    4),
                "f1":        round(report["notr"]["f1-score"],  4),
                "support":   int(report["notr"]["support"]),
            },
            "olumlu": {
                "precision": round(report["olumlu"]["precision"], 4),
                "recall":    round(report["olumlu"]["recall"],    4),
                "f1":        round(report["olumlu"]["f1-score"],  4),
                "support":   int(report["olumlu"]["support"]),
            },
        },
        "accuracy":  round(report["accuracy"], 4),
        "macro_f1":  round(report["macro avg"]["f1-score"], 4),
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nKaydedildi: {OUT_PATH}")
    print(f"Accuracy: {metrics['accuracy']}  |  Macro-F1: {metrics['macro_f1']}")
    print("Confusion matrix:")
    for row, name in zip(cm.tolist(), ["olumsuz", "notr", "olumlu"]):
        print(f"  {name}: {row}")

if __name__ == "__main__":
    main()
