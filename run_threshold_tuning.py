"""
Threshold Tuning — Ensemble v3 üzerinde sınıf bazlı eşik optimizasyonu
Val seti üzerinde her sınıf için en iyi threshold bulunur, test setine uygulanır.
Hedef: nötr sınıfı F1'ini artırarak makro F1'i maksimize etmek.
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
from sklearn.metrics import f1_score, accuracy_score, classification_report, matthews_corrcoef, cohen_kappa_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from pathlib import Path
from itertools import product

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results")
LABEL_NAMES = ["olumsuz", "notr", "olumlu"]

v4_dir  = Path("data/processed_v4")
val_df  = pd.read_csv(v4_dir / "val.csv")
test_df = pd.read_csv(v4_dir / "test.csv")
log(f"Val: {len(val_df)} | Test: {len(test_df)} | Cihaz: {DEVICE}")


class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts  = df["Review"].astype(str).tolist()
        self.labels = df["label"].tolist()
        self.tok    = tokenizer
        self.mlen   = max_len
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, padding="max_length",
                       max_length=self.mlen, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}, torch.tensor(self.labels[i])


def get_probas(model_dir, max_len, df):
    tok   = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=3)
    model.to(DEVICE).eval()
    dl = DataLoader(ReviewDataset(df, tok, max_len), batch_size=32, num_workers=0)
    all_probas, all_labels = [], []
    with torch.no_grad():
        for batch, y in dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            prob  = torch.softmax(model(**batch).logits.float(), dim=-1).cpu().numpy()
            all_probas.append(prob)
            all_labels.extend(y.tolist())
    del model; torch.cuda.empty_cache()
    return np.vstack(all_probas), np.array(all_labels)


# ── Mevcut modelleri belirle ─────────────────────────────────────
MODELS = [
    ("models/berturk_v3_finetuned",     192, "berturk_v3_results.json",    "berturk_v3"),
    ("models/xlm_roberta_v2_finetuned", 192, "xlm_roberta_v2_results.json","xlm_roberta_v2"),
    ("models/savasy_finetuned",         192, "savasy_results.json",         "savasy_finetuned"),
    ("models/xlm_roberta_finetuned",    128, "xlm_roberta_results.json",    "xlm_roberta"),
]

active = [(d, ml, rf, k) for d, ml, rf, k in MODELS if Path(d).exists()]
if len(active) < 2:
    log("Yeterli model yok (en az 2 gerekli)."); exit(1)

# val_f1 ağırlıkları
def get_val_f1(res_file, key):
    try:
        with open(RESULTS_DIR / res_file, encoding="utf-8") as f:
            data = json.load(f)
        return data.get(key, data).get("best_val_f1", data.get(key, data).get("macro_f1", 0.75))
    except Exception:
        return 0.75

val_f1s  = [get_val_f1(rf, k) for _, _, rf, k in active]
total_f1 = sum(val_f1s)
weights  = [v / total_f1 for v in val_f1s]

log("\n--- Kullanılan modeller ---")
for (d, _, _, k), w in zip(active, weights): log(f"  {k}: ağırlık={w:.3f}")

# ── Val ve test probalarını al ───────────────────────────────────
log("\nVal seti inference...")
val_probas_list = []
for (d, ml, _, k), w in zip(active, weights):
    log(f"  {k}...")
    vp, val_labels = get_probas(d, ml, val_df)
    val_probas_list.append(vp * w)
val_ensemble = sum(val_probas_list)

log("\nTest seti inference...")
test_probas_list = []
for (d, ml, _, k), w in zip(active, weights):
    log(f"  {k}...")
    tp, test_labels = get_probas(d, ml, test_df)
    test_probas_list.append(tp * w)
test_ensemble = sum(test_probas_list)

# ── Baseline (argmax) ────────────────────────────────────────────
baseline_preds = val_ensemble.argmax(axis=1)
baseline_f1    = f1_score(val_labels, baseline_preds, average="macro")
log(f"\nBaseline (argmax) val Macro-F1: {baseline_f1:.4f}")

# ── Threshold arama — nötr için eşik ────────────────────────────
# Strateji: p_notr > threshold ise notr tahmin et, aksi halde argmax
log("\nThreshold arama (nötr eşiği 0.20-0.45)...")

best_f1, best_thr = baseline_f1, None
for thr in np.arange(0.20, 0.46, 0.01):
    preds = []
    for probs in val_ensemble:
        if probs[1] >= thr:
            preds.append(1)  # notr
        else:
            preds.append(probs.argmax())
    f1 = f1_score(val_labels, preds, average="macro")
    if f1 > best_f1:
        best_f1  = f1
        best_thr = thr

if best_thr is not None:
    log(f"En iyi eşik: {best_thr:.2f} → val Macro-F1: {best_f1:.4f} "
        f"(+{(best_f1-baseline_f1)*100:.2f} puan)")
else:
    log("Threshold tuning iyileşme sağlamadı, argmax kullanılıyor.")
    best_thr = None

# ── Test setine uygula ───────────────────────────────────────────
log("\n" + "="*60)
log("TEST DEGERLENDIRME")
log("="*60)

if best_thr is not None:
    test_preds = []
    for probs in test_ensemble:
        if probs[1] >= best_thr:
            test_preds.append(1)
        else:
            test_preds.append(probs.argmax())
    mode_str = f"threshold_notr={best_thr:.2f}"
else:
    test_preds = test_ensemble.argmax(axis=1).tolist()
    mode_str   = "argmax"

acc   = accuracy_score(test_labels, test_preds)
f1    = f1_score(test_labels, test_preds, average="macro")
mcc   = matthews_corrcoef(test_labels, test_preds)
kappa = cohen_kappa_score(test_labels, test_preds)
y_bin = label_binarize(test_labels, classes=[0,1,2])
auc   = roc_auc_score(y_bin, test_ensemble, average="macro", multi_class="ovr")
per_class_f1 = f1_score(test_labels, test_preds, average=None, labels=[0,1,2])
per_class    = {LABEL_NAMES[i]: round(float(per_class_f1[i]),4) for i in range(3)}

log(f"Mod       : {mode_str}")
log(f"Accuracy  : {acc:.4f}")
log(f"Macro-F1  : {f1:.4f}")
log(f"MCC       : {mcc:.4f}")
log(f"Kappa     : {kappa:.4f}")
log(f"ROC-AUC   : {auc:.4f}")
log(f"Per-class : {per_class}")
log("")
log(classification_report(test_labels, test_preds, target_names=LABEL_NAMES, digits=4))

results = {
    "ensemble_tuned": {
        "accuracy":      round(acc, 4),
        "macro_f1":      round(f1, 4),
        "mcc":           round(mcc, 4),
        "kappa":         round(kappa, 4),
        "roc_auc_macro": round(auc, 4),
        "per_class":     per_class,
        "threshold_notr": round(float(best_thr), 2) if best_thr else None,
        "models":        [k for _, _, _, k in active],
        "weights":       {k: round(w,4) for (_, _, _, k), w in zip(active, weights)},
        "mode":          mode_str,
    }
}
RESULTS_DIR.mkdir(exist_ok=True)
with open(RESULTS_DIR / "ensemble_tuned_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
log("Sonuclar: results/ensemble_tuned_results.json")
log("\nTHRESHOLD TUNING TAMAMLANDI.")
