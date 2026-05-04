"""
Tam Değerlendirme — Tüm modelleri karşılaştırır
Accuracy, Macro-F1, MCC, Cohen Kappa, ROC-AUC
Çalıştırma zamanı: BERTurk + XLM-RoBERTa eğitimi bittikten sonra
"""
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def log(msg): print(msg, flush=True)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
hf_logging.set_verbosity_error()
from pathlib import Path

from src.evaluation.metrics import compute_classification_metrics, print_metrics

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results")
LABEL_NAMES = ["olumsuz", "notr", "olumlu"]

test_df = pd.read_csv("data/processed/test.csv")


class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts     = df["Review"].astype(str).tolist()
        self.labels    = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i], truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}, torch.tensor(self.labels[i])


def eval_transformer(model_dir: str, name: str) -> dict:
    log(f"\n{name} yukleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=3)
    model.to(DEVICE).eval()

    ds = ReviewDataset(test_df, tokenizer)
    dl = DataLoader(ds, batch_size=32, num_workers=0, pin_memory=False)

    preds, labels, probas = [], [], []
    with torch.no_grad():
        for batch, y in dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out   = model(**batch)
            proba = torch.softmax(out.logits, dim=-1).cpu().tolist()
            pred  = out.logits.argmax(-1).cpu().tolist()
            preds.extend(pred); labels.extend(y.tolist()); probas.extend(proba)

    del model; torch.cuda.empty_cache()
    return compute_classification_metrics(labels, preds, probas)


all_results = {}

# ── Baseline sonuçları ───────────────────────────────────────────
try:
    with open(RESULTS_DIR / "baseline_results.json") as f:
        baselines = json.load(f)
    all_results.update(baselines)
except FileNotFoundError:
    log("Baseline sonuclari bulunamadi.")

# ── BERTurk ─────────────────────────────────────────────────────
berturk_dir = "models/berturk_finetuned"
if Path(berturk_dir).exists():
    metrics = eval_transformer(berturk_dir, "BERTurk")
    all_results["berturk_finetuned"] = metrics
    print_metrics(metrics, "BERTurk (E-ComDe)")
else:
    log("BERTurk checkpoint bulunamadi.")

# ── XLM-RoBERTa ─────────────────────────────────────────────────
xlm_dir = "models/xlm_roberta_finetuned"
if Path(xlm_dir).exists():
    metrics = eval_transformer(xlm_dir, "XLM-RoBERTa")
    all_results["xlm_roberta"] = metrics
    print_metrics(metrics, "XLM-RoBERTa")
else:
    log("XLM-RoBERTa checkpoint bulunamadi.")

# ── Ensemble sonuclari (varsa) ───────────────────────────────────
try:
    with open(RESULTS_DIR / "ensemble_results.json") as f:
        ens = json.load(f)
    all_results.update(ens)
    log("Ensemble sonuclari yuklendi.")
except FileNotFoundError:
    pass

# ── BERTurk v3 sonuclari (varsa) ─────────────────────────────────
try:
    with open(RESULTS_DIR / "berturk_v3_results.json") as f:
        v3 = json.load(f)
    all_results.update(v3)
    log("BERTurk v3 sonuclari yuklendi.")
except FileNotFoundError:
    pass

# ── Ensemble v2 sonuclari (varsa) ────────────────────────────────
try:
    with open(RESULTS_DIR / "ensemble_v2_results.json") as f:
        ens2 = json.load(f)
    all_results.update(ens2)
    log("Ensemble v2 sonuclari yuklendi.")
except FileNotFoundError:
    pass

# ── Karşılaştırma tablosu ────────────────────────────────────────
log(f"\n{'='*70}")
log("NIHAI MODEL KARSILASTIRMASI")
log(f"{'='*70}")
log(f"{'Model':<30} {'Acc':>8} {'F1':>8} {'MCC':>8} {'Kappa':>8} {'AUC':>8}")
log("-" * 70)

model_order = ["tfidf_lr", "bilstm", "xlm_roberta", "berturk_finetuned",
               "ensemble_avg", "berturk_v3", "ensemble_v2_avg"]
for name in model_order:
    if name not in all_results:
        continue
    r = all_results[name]
    auc = r.get("roc_auc_macro", float("nan"))
    mcc = r.get("mcc", float("nan"))
    kap = r.get("kappa", float("nan"))
    auc_str = f"{auc:.4f}" if not (isinstance(auc, float) and np.isnan(auc)) else "  N/A "
    mcc_str = f"{mcc:.4f}" if not (isinstance(mcc, float) and np.isnan(mcc)) else "  N/A "
    kap_str = f"{kap:.4f}" if not (isinstance(kap, float) and np.isnan(kap)) else "  N/A "
    log(f"{name:<30} {r['accuracy']:>8.4f} {r['macro_f1']:>8.4f} {mcc_str:>8} {kap_str:>8} {auc_str:>8}")

# ── Kaydet ──────────────────────────────────────────────────────
with open(RESULTS_DIR / "full_evaluation.json", "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
log(f"\nTum sonuclar: results/full_evaluation.json")
