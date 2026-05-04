"""
Ensemble v3: BERTurk v3 + XLM-RoBERTa v2 + savasy — 3 model agirlikli ensemble
Agirliklar: her modelin val_f1'ine oranla otomatik hesaplanir
En guclu kombinasyon — hedef: >%85 macro-F1
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


def get_probas(model_dir: str, max_len: int) -> tuple:
    log(f"  Yukleniyor: {model_dir} (max_len={max_len})")
    tok   = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=3)
    model.to(DEVICE).eval()

    dl = DataLoader(ReviewDataset(test_df, tok, max_len),
                    batch_size=20, num_workers=0, pin_memory=False)
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


# ── Model dizinleri ve max_len eslesmesi ─────────────────────────
MODELS = [
    ("models/berturk_v3_finetuned",   192, "berturk_v3_results.json",    "berturk_v3"),
    ("models/xlm_roberta_v2_finetuned", 192, "xlm_roberta_v2_results.json", "xlm_roberta_v2"),
    ("models/savasy_finetuned",        192, "savasy_results.json",          "savasy_finetuned"),
]

# Fallback: xlm_roberta v1 yoksa v2, savasy yoksa atla
fallback = {
    "models/xlm_roberta_v2_finetuned": ("models/xlm_roberta_finetuned", 128),
}

active_models = []
for ckpt, mlen, res_file, key in MODELS:
    if Path(ckpt).exists():
        active_models.append((ckpt, mlen, res_file, key))
    elif ckpt in fallback:
        fb_ckpt, fb_len = fallback[ckpt]
        if Path(fb_ckpt).exists():
            log(f"  Fallback: {ckpt} -> {fb_ckpt}")
            active_models.append((fb_ckpt, fb_len, res_file, key.replace("_v2", "")))
        else:
            log(f"  Atlanıyor (checkpoint yok): {ckpt}")
    else:
        log(f"  Atlanıyor (checkpoint yok): {ckpt}")

if len(active_models) < 2:
    log("HATA: En az 2 model gerekli.")
    exit(1)

# ── Val F1 bazli agirliklar ──────────────────────────────────────
val_f1s = []
for _, _, res_file, key in active_models:
    try:
        with open(RESULTS_DIR / res_file, encoding="utf-8") as f:
            data = json.load(f)
        vf1 = data.get(key, data).get("best_val_f1", data.get(key, data).get("macro_f1", 0.75))
        val_f1s.append(vf1)
    except Exception:
        val_f1s.append(0.75)

total_f1 = sum(val_f1s)
weights  = [v / total_f1 for v in val_f1s]

log("\n--- Model Agirliklari ---")
for (ckpt, _, _, key), vf1, w in zip(active_models, val_f1s, weights):
    log(f"  {key}: val_f1={vf1:.4f} -> agirlik={w:.3f}")

# ── Inference ───────────────────────────────────────────────────
all_probas_list = []
labels = None

for (ckpt, mlen, _, key), w in zip(active_models, weights):
    log(f"\n{key} inference...")
    probas, lbs = get_probas(ckpt, mlen)
    all_probas_list.append((probas, w))
    if labels is None:
        labels = lbs

ensemble_probas = sum(p * w for p, w in all_probas_list)
ensemble_preds  = ensemble_probas.argmax(axis=1).tolist()

# ── Metrikler ────────────────────────────────────────────────────
acc   = accuracy_score(labels, ensemble_preds)
f1    = f1_score(labels, ensemble_preds, average="macro")
mcc   = matthews_corrcoef(labels, ensemble_preds)
kappa = cohen_kappa_score(labels, ensemble_preds)
y_bin = label_binarize(labels, classes=[0,1,2])
auc   = roc_auc_score(y_bin, ensemble_probas, average="macro", multi_class="ovr")
per_class_f1 = f1_score(labels, ensemble_preds, average=None, labels=[0,1,2])
per_class    = {LABEL_NAMES[i]: round(float(per_class_f1[i]),4) for i in range(3)}

model_names = [key for _, _, _, key in active_models]
log(f"\n{'='*60}")
log(f"Ensemble v3 ({' + '.join(model_names)})")
log(f"{'='*60}")
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
    "ensemble_v3": {
        "accuracy":      round(acc, 4),
        "macro_f1":      round(f1, 4),
        "mcc":           round(mcc, 4),
        "kappa":         round(kappa, 4),
        "roc_auc_macro": round(auc, 4),
        "per_class":     per_class,
        "models":        model_names,
        "weights":       {k: round(w, 4) for k, w in zip(model_names, weights)},
        "test_set":      str(test_path),
    }
}
with open(RESULTS_DIR / "ensemble_v3_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
log("Sonuclar: results/ensemble_v3_results.json")
