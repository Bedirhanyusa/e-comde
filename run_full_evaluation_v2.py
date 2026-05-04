"""
Tam Degerlendirme v2 — Tum modelleri JSON sonuclarindan yukler ve karsilastirir
(Yeniden inference yapmaz — tum sonuclar onceki scriptlerle kaydedilmistir)
"""
import os, json, warnings, numpy as np
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def log(msg): print(msg, flush=True)

from pathlib import Path

RESULTS_DIR = Path("results")

all_results = {}

LOAD_MAP = {
    "baseline_results.json":         None,
    "berturk_results.json":          "berturk_finetuned",
    "berturk_zeroshot_results.json": None,
    "xlm_roberta_results.json":      "xlm_roberta",
    "ensemble_results.json":         None,
    "berturk_v3_results.json":       None,
    "xlm_roberta_v2_results.json":   None,
    "savasy_results.json":           None,
    "ensemble_v2_results.json":      None,
    "ensemble_v3_results.json":      None,
    "ensemble_tuned_results.json":   None,
    "absa_results.json":             None,
    "rouge_results.json":            None,
}

for fname, key in LOAD_MAP.items():
    fpath = RESULTS_DIR / fname
    if not fpath.exists():
        log(f"Atlanıyor (bulunamadı): {fname}")
        continue
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)
    if key:
        # json dogrudan dict of metrics ise icine sar
        if key not in data:
            data = {key: data}
    all_results.update(data)
    log(f"Yuklendi: {fname} ({list(data.keys())})")

# ── Karsilastirma tablosu ────────────────────────────────────────
METRIC_MODELS = [
    "tfidf_lr", "bilstm",
    "berturk_zeroshot",
    "berturk_finetuned",
    "xlm_roberta",
    "ensemble_avg",
    "berturk_v3",
    "xlm_roberta_v2",
    "savasy_finetuned",
    "ensemble_v2_avg",
    "ensemble_v3",
    "ensemble_tuned",
]

log(f"\n{'='*75}")
log("NIHAI MODEL KARSILASTIRMASI")
log(f"{'='*75}")
log(f"{'Model':<28} {'Acc':>8} {'F1':>8} {'MCC':>8} {'Kappa':>8} {'AUC':>8}")
log("-" * 75)

for name in METRIC_MODELS:
    if name not in all_results:
        continue
    r = all_results[name]
    def fmt(v):
        if isinstance(v, float) and np.isnan(v): return "  N/A "
        if v is None: return "  N/A "
        return f"{v:.4f}"
    log(f"{name:<28} {fmt(r.get('accuracy')):>8} {fmt(r.get('macro_f1')):>8} "
        f"{fmt(r.get('mcc')):>8} {fmt(r.get('kappa')):>8} {fmt(r.get('roc_auc_macro')):>8}")

log(f"{'='*75}")

# ── Kaydet ──────────────────────────────────────────────────────
out_path = RESULTS_DIR / "full_evaluation_v2.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
log(f"\nTum sonuclar: {out_path}")
