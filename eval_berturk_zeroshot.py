"""
BERTurk sifir atis (zero-shot) degerlendirme.
Fine-tuning OLMADAN dogrudan test setinde calistirir.
Ablasyon tablosu icin "ince ayarsiz" satirini doldurur.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def log(msg): print(msg, flush=True)

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification, logging as hf_logging
hf_logging.set_verbosity_error()
from pathlib import Path
from src.evaluation.metrics import compute_classification_metrics, print_metrics

MODEL_NAME  = "dbmdz/bert-base-turkish-cased"
RESULTS_DIR = Path("results")
MAX_LEN     = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Cihaz: {DEVICE}")

log("Zero-shot BERTurk yukleniyor (fine-tuning YOK)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model.to(DEVICE)
model.eval()
log("Model hazir.")

test_df = pd.read_csv("data/processed/test.csv")
log(f"Test set: {len(test_df)} ornek")

class ReviewDataset(Dataset):
    def __init__(self, df):
        self.texts  = df["Review"].astype(str).tolist()
        self.labels = df["label"].tolist()
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        enc = tokenizer(
            self.texts[i], truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}, torch.tensor(self.labels[i])

test_dl = DataLoader(ReviewDataset(test_df), batch_size=32, shuffle=False,
                     num_workers=0, pin_memory=False)

all_preds, all_labels, all_probas = [], [], []
with torch.no_grad():
    for batch, labels in test_dl:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out   = model(**batch)
        proba = torch.softmax(out.logits, dim=-1).cpu().tolist()
        pred  = out.logits.argmax(-1).cpu().tolist()
        all_preds.extend(pred)
        all_labels.extend(labels.tolist())
        all_probas.extend(proba)

metrics = compute_classification_metrics(all_labels, all_preds, all_probas)
print_metrics(metrics, "BERTurk ZERO-SHOT (ince ayarsiz)")

RESULTS_DIR.mkdir(exist_ok=True)
with open(RESULTS_DIR / "berturk_zeroshot_results.json", "w", encoding="utf-8") as f:
    json.dump({"berturk_zeroshot": metrics}, f, indent=2, ensure_ascii=False)
log("Sonuclar: results/berturk_zeroshot_results.json")
