"""
Ensemble: BERTurk + XLM-RoBERTa softmax averaging
Çalıştır: her iki model checkpoint'i hazır olduktan sonra
"""
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
hf_logging.set_verbosity_error()

from sklearn.metrics import accuracy_score, f1_score
from src.evaluation.metrics import compute_classification_metrics, print_metrics
from pathlib import Path

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results")
LABEL_NAMES = ["olumsuz", "nötr", "olumlu"]
MAX_LEN     = 128

test_df = pd.read_csv("data/processed/test.csv")
print(f"Test seti: {len(test_df)} ornek | Cihaz: {DEVICE}")


def get_probas(model_dir: str) -> np.ndarray:
    print(f"  Yukleniyor: {model_dir}")
    tok   = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=3)
    model.to(DEVICE).eval()

    texts  = test_df["Review"].astype(str).tolist()
    labels = test_df["label"].tolist()
    all_probas = []

    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tok(batch_texts, truncation=True, padding="max_length",
                  max_length=MAX_LEN, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out   = model(**enc)
            proba = torch.softmax(out.logits, dim=-1).cpu().numpy()
        all_probas.append(proba)

    del model; torch.cuda.empty_cache()
    return np.vstack(all_probas), labels


berturk_dir = "models/berturk_finetuned"
xlm_dir     = "models/xlm_roberta_finetuned"

missing = [d for d in [berturk_dir, xlm_dir] if not Path(d).exists()]
if missing:
    print(f"Eksik checkpoint'ler: {missing}")
    print("Once run_berturk.py ve run_xlm_roberta.py tamamlayin.")
    exit(1)

print("Ensemble olasiliklari hesaplaniyor...")
probas_bert, labels = get_probas(berturk_dir)
probas_xlm,  _      = get_probas(xlm_dir)

# Simple average ensemble
ensemble_probas = (probas_bert + probas_xlm) / 2.0
ensemble_preds  = ensemble_probas.argmax(axis=1).tolist()
labels          = list(labels)

metrics = compute_classification_metrics(labels, ensemble_preds, ensemble_probas.tolist())
print_metrics(metrics, "Ensemble (BERTurk + XLM-RoBERTa)")

RESULTS_DIR.mkdir(exist_ok=True)
results = {"ensemble_avg": metrics}
with open(RESULTS_DIR / "ensemble_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Sonuclar: results/ensemble_results.json")
