"""
BERTurk v3 bittikten sonra calistir:
1. Summarizer (rouge)
2. Ensemble v2 (BERTurk v3 + XLM-Ro v1)
3. XLM-RoBERTa v2 (v4+Trendyol verisi ile)
4. savasy fine-tune
5. Ensemble v3 (3 model)
6. Full evaluation v2
7. Paper update
"""
import subprocess, sys, os, time, json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PYTHON = sys.executable
os.chdir(Path(__file__).parent)
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def run_step(script, desc, critical=False, skip_if_exists=None):
    if skip_if_exists and Path(skip_if_exists).exists():
        print(f"ATLANDI: {desc}", flush=True)
        return True
    log_path = LOG_DIR / script.replace(".py", "_post.log")
    print(f"\n{'='*60}\nADIM: {desc}\n{'='*60}", flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            [PYTHON, "-u", script],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            lf.write(line)
        proc.wait()
    elapsed = (time.time() - t0) / 60
    ok = proc.returncode in (0, -1073740791)
    print(f"\n>> {'TAMAMLANDI' if ok else f'HATA(exit={proc.returncode})'} ({elapsed:.1f} dk)", flush=True)
    if not ok and critical:
        sys.exit(1)
    return ok

print("POST-BERTURK-V3 PIPELINE BASLIYOR", flush=True)

# Onkosul: berturk_v3 sonuclari mevcut olmali
if not Path("results/berturk_v3_results.json").exists():
    print("HATA: berturk_v3_results.json bulunamadi!", flush=True)
    sys.exit(1)

# Sonucu goster
with open("results/berturk_v3_results.json", encoding="utf-8") as f:
    v3 = json.load(f)["berturk_v3"]
print(f"\nBERTurk v3 sonucu: Acc={v3['accuracy']:.4f} | F1={v3['macro_f1']:.4f} | MCC={v3['mcc']:.4f}", flush=True)

# 1. Summarizer
run_step("run_summarizer.py", "mT5 ozetleyici + ROUGE",
         skip_if_exists="results/rouge_results.json")

# 2. Ensemble v2 (BERTurk v3 + XLM-Ro v1, agirlikli)
run_step("run_ensemble_v2.py", "Ensemble v2 (BERTurk v3 + XLM-Ro v1)",
         skip_if_exists="results/ensemble_v2_results.json")

# 3. XLM-RoBERTa v2 (v4+Trendyol, MAX_LEN=192)
run_step("run_xlm_roberta_v2.py", "XLM-RoBERTa v2 (MAX_LEN=192, v4 veri)",
         skip_if_exists="results/xlm_roberta_v2_results.json")

# 4. savasy fine-tune
run_step("run_savasy.py", "savasy Turkce sentiment fine-tune",
         skip_if_exists="results/savasy_results.json")

# 5. Ensemble v3 (3 model agirlikli)
run_step("run_ensemble_v3.py", "Ensemble v3 (BERTurk v3 + XLM-Ro v2 + savasy)",
         skip_if_exists="results/ensemble_v3_results.json")

# 6. Full evaluation
run_step("run_full_evaluation_v2.py", "Tam degerlendirme v2")

# 7. Paper update
run_step("update_paper.py", "Konferans bildirisi guncelleme")

# Ozet
print("\n" + "="*60, flush=True)
print("TUM PIPELINE TAMAMLANDI!", flush=True)
print("="*60, flush=True)

model_order = [
    ("ensemble_v3_results.json",    "ensemble_v3"),
    ("ensemble_v2_results.json",    "ensemble_v2_avg"),
    ("savasy_results.json",         "savasy_finetuned"),
    ("berturk_v3_results.json",     "berturk_v3"),
    ("xlm_roberta_v2_results.json", "xlm_roberta_v2"),
    ("xlm_roberta_results.json",    "xlm_roberta"),
    ("berturk_results.json",        "berturk_finetuned"),
    ("baseline_results.json",       "tfidf_lr"),
]

best_f1, best_name = 0.0, "?"
print(f"\n{'Model':<30} {'Acc':>8} {'F1':>8} {'MCC':>8}", flush=True)
print("-" * 58, flush=True)
for fname, key in model_order:
    fpath = Path("results") / fname
    if not fpath.exists(): continue
    try:
        data = json.load(open(fpath, encoding="utf-8"))
        r    = data.get(key, data)
        acc  = r.get("accuracy", 0)
        f1   = r.get("macro_f1", 0)
        mcc  = r.get("mcc", 0)
        print(f"{key:<30} {acc:>8.4f} {f1:>8.4f} {mcc:>8.4f}", flush=True)
        if f1 > best_f1:
            best_f1, best_name = f1, key
    except Exception:
        pass

print(f"\n{'='*58}", flush=True)
print(f"EN IYI: {best_name} | Macro-F1 = {best_f1:.4f} ({best_f1*100:.1f}%)", flush=True)
if best_f1 >= 0.85:
    print("HEDEF TUTTURULDU! >= %85", flush=True)
elif best_f1 >= 0.80:
    print("%80+ — hedefe yakin.", flush=True)
