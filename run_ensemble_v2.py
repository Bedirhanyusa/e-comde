"""
Ensemble v2: BERTurk v3 + XLM-RoBERTa — agirlikli softmax ortalama
Agirliklar: val_f1 oranina gore otomatik hesaplanir (en iyi model daha agir basar)
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
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    matthews_corrcoef, cohen_kappa_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from pathlib import Path

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results")
LABEL_NAMES = ["olumsuz", "notr", "olumlu"]
MAX_LEN     = 192  # BERTurk v3 ile eslesen pencere

v4_test   = Path("data/processed_v4/test.csv")
v3_test   = Path("data/processed/test.csv")
test_path = v4_test if v4_test.exists() else v3_test
test_df   = pd.read_csv(test_path)
log(f"Test seti: {test_path} | {len(test_df)} ornek | Cihaz: {DEVICE}")


class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts  = df["Review"].astype(str).tolist()
        self.labels = df["label"].tolist()
        self.tok    = tokenizer
        self.mlen   = max_len
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        enc = self.tok(
            self.texts[i], truncation=True, padding="max_length",
            max_length=self.mlen, return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}, torch.tensor(self.labels[i])


def get_probas(model_dir: str, max_len: int = 128) -> tuple:
    log(f"  Yukleniyor: {model_dir} (max_len={max_len})")
    tok   = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=3)
    model.to(DEVICE).eval()

    dl = DataLoader(ReviewDataset(test_df, tok, max_len),
                    batch_size=24, num_workers=0, pin_memory=False)
    all_probas, all_labels = [], []

    with torch.no_grad():
        for batch, y in dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out   = model(**batch)
            prob  = torch.softmax(out.logits.float(), dim=-1).cpu().numpy()
            all_probas.append(prob)
            all_labels.extend(y.tolist())

    del model; torch.cuda.empty_cache()
    return np.vstack(all_probas), all_labels


berturk_v3_dir = "models/berturk_v3_finetuned"
xlm_dir        = "models/xlm_roberta_finetuned"

missing = [d for d in [berturk_v3_dir, xlm_dir] if not Path(d).exists()]
if missing:
    log(f"Eksik checkpoint: {missing}")
    exit(1)

# Val F1'lerini oku — agirlik icin kullan
v3_val_f1  = 0.0
xlm_val_f1 = 0.0
try:
    with open(RESULTS_DIR / "berturk_v3_results.json", encoding="utf-8") as f:
        v3_val_f1 = json.load(f)["berturk_v3"].get("best_val_f1", 0.0)
except Exception:
    pass
try:
    with open(RESULTS_DIR / "xlm_roberta_results.json", encoding="utf-8") as f:
        xlm_data = json.load(f)
        xlm_val_f1 = xlm_data.get("xlm_roberta", xlm_data).get("macro_f1", 0.0)
except Exception:
    pass

# Agirlik: val F1 oranina gore; eger ikisi de 0 ise esit agirlik
total_f1 = v3_val_f1 + xlm_val_f1
if total_f1 > 0.01:
    w_v3  = v3_val_f1  / total_f1
    w_xlm = xlm_val_f1 / total_f1
else:
    w_v3 = w_xlm = 0.5

log(f"\nAgirlıklar — BERTurk v3: {w_v3:.3f} (val_f1={v3_val_f1:.4f})  "
    f"XLM-RoBERTa: {w_xlm:.3f} (val_f1={xlm_val_f1:.4f})")

log("\nBERTurk v3 olasiliklari hesaplaniyor...")
probas_v3, labels = get_probas(berturk_v3_dir, max_len=192)

log("XLM-RoBERTa olasiliklari hesaplaniyor...")
probas_xlm, _     = get_probas(xlm_dir, max_len=128)

# Agirlikli ortalama
ensemble_probas = w_v3 * probas_v3 + w_xlm * probas_xlm
ensemble_preds  = ensemble_probas.argmax(axis=1).tolist()

acc   = accuracy_score(labels, ensemble_preds)
f1    = f1_score(labels, ensemble_preds, average="macro")
mcc   = matthews_corrcoef(labels, ensemble_preds)
kappa = cohen_kappa_score(labels, ensemble_preds)
y_bin = label_binarize(labels, classes=[0, 1, 2])
auc   = roc_auc_score(y_bin, ensemble_probas, average="macro", multi_class="ovr")

per_class_f1 = f1_score(labels, ensemble_preds, average=None, labels=[0, 1, 2])
per_class    = {LABEL_NAMES[i]: round(float(per_class_f1[i]), 4) for i in range(3)}

log(f"\n{'='*55}")
log(f"Ensemble v2 (BERTurk v3 x{w_v3:.2f} + XLM-RoBERTa x{w_xlm:.2f})")
log(f"{'='*55}")
log(f"  Accuracy : {acc:.4f}")
log(f"  Macro-F1 : {f1:.4f}")
log(f"  MCC      : {mcc:.4f}")
log(f"  Kappa    : {kappa:.4f}")
log(f"  ROC-AUC  : {auc:.4f}")
log(f"  Per-class: {per_class}")
log("")
log(classification_report(labels, ensemble_preds, target_names=LABEL_NAMES, digits=4))

RESULTS_DIR.mkdir(exist_ok=True)
results = {
    "ensemble_v2_avg": {
        "accuracy":      round(acc, 4),
        "macro_f1":      round(f1, 4),
        "mcc":           round(mcc, 4),
        "kappa":         round(kappa, 4),
        "roc_auc_macro": round(auc, 4),
        "per_class":     per_class,
        "weights":       {"berturk_v3": round(w_v3, 4), "xlm_roberta": round(w_xlm, 4)},
        "test_set":      str(test_path),
    }
}
with open(RESULTS_DIR / "ensemble_v2_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
log("Sonuclar: results/ensemble_v2_results.json")
