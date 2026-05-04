"""
V3 Pipeline — context-safe continuation script
Runs: ABSA → summarizer → prepare_data_v4 → BERTurk v3 training
Run this whenever you need to continue the pipeline autonomously.
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
        print(f"ATLANDI (zaten mevcut): {desc} [{skip_if_exists}]", flush=True)
        return True

    log_path = LOG_DIR / script.replace(".py", "_v3pipeline.log")
    print(f"\n{'='*60}", flush=True)
    print(f"ADIM: {desc}", flush=True)
    print(f"{'='*60}", flush=True)
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
    ok = proc.returncode == 0
    status = "TAMAMLANDI" if ok else f"HATA (exit={proc.returncode})"
    print(f"\n>> {status} ({elapsed:.1f} dk)", flush=True)
    if not ok and critical:
        print(f"KRITIK HATA: {script} basarisiz, durduruluyor.", flush=True)
        sys.exit(1)
    return ok

print("V3 Pipeline Basliyor", flush=True)
print(f"Python: {PYTHON}", flush=True)

# 1. ABSA
run_step("run_absa.py", "ABSA modulu",
         skip_if_exists="results/absa_results.json")

# 2. Summarizer
run_step("run_summarizer.py", "mT5 ozetleme + ROUGE",
         skip_if_exists="results/rouge_results.json")

# 3. V4 data prep
run_step("prepare_data_v4.py", "V4 veri seti (15K/sinif, akilli etiket)",
         skip_if_exists="data/processed_v4/train.csv")

# 4. BERTurk v3 two-phase training (CRITICAL - main improvement step)
run_step("run_berturk_v3.py", "BERTurk v3 iki asamali egitim (hedef: %82-87)",
         critical=True,
         skip_if_exists="results/berturk_v3_results.json")

# 5. Rebuild ensemble with v3 model if it exists
v3_ckpt = Path("models/berturk_v3_finetuned")
if v3_ckpt.exists():
    print("\nBERTurk v3 checkpoint mevcut — ensemble v2 olusturuluyor...", flush=True)
    run_step("run_ensemble_v2.py", "Ensemble v2 (BERTurk v3 + XLM-Ro)",
             skip_if_exists="results/ensemble_v2_results.json")

# 6. Full evaluation update
run_step("run_full_evaluation_v2.py", "Tam degerlendirme v2",
         skip_if_exists="results/full_evaluation_v2.json")

# 7. Paper update
run_step("update_paper.py", "Bildiri guncelleme")

print("\n" + "="*60, flush=True)
print("V3 PIPELINE TAMAMLANDI!", flush=True)

# Final pipeline otomatik baslar (XLM-Ro v2 + savasy + 3-model ensemble)
print("\nFinal pipeline baslatiliyor...", flush=True)
run_step("run_final_pipeline.py", "Final pipeline (XLM-Ro v2 + savasy + ensemble v3)")
