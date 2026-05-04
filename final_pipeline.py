"""
Son adimlar — baselines.py bittikten sonra calistirilir:
1. eval_berturk_zeroshot  — ablasyon (hizli)
2. run_xlm_roberta        — XLM-RoBERTa egitimi (uzun ~2-3 saat)
3. run_absa               — ABSA modulu
4. run_summarizer         — mT5 ozetleme + ROUGE
5. run_full_evaluation    — tum modeller karsilastirma
6. update_paper           — bildiri guncelleme
"""
import subprocess, sys, os, time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PYTHON  = sys.executable
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

STEPS = [
    ("eval_berturk_zeroshot.py", "BERTurk zero-shot ablasyon",   False),
    ("run_xlm_roberta.py",       "XLM-RoBERTa ince ayar",         True),
    ("run_ensemble.py",          "Ensemble (BERTurk + XLM-Ro)",   False),
    ("run_absa.py",              "ABSA modulu",                    False),
    ("run_summarizer.py",        "mT5 ozetleme",                   False),
    ("run_full_evaluation.py",   "Tam degerlendirme",              True),
    ("update_paper.py",          "Bildiri guncelleme",             True),
]

def run_step(script, desc, critical):
    log_path = LOG_DIR / script.replace(".py", "_final.log")
    print(f"\n{'='*60}", flush=True)
    print(f"ADIM: {desc}", flush=True)
    print(f"Script: {script}", flush=True)
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            [PYTHON, script],
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
        print(f"KRITIK HATA: {script} basarisiz, pipeline durduruluyor.", flush=True)
        sys.exit(1)
    return ok

os.chdir(Path(__file__).parent)
print("Final Pipeline Basliyor", flush=True)
print(f"Python: {PYTHON}", flush=True)

for script, desc, critical in STEPS:
    run_step(script, desc, critical)

print("\n" + "="*60, flush=True)
print("TUM PIPELINE TAMAMLANDI!", flush=True)
print("Sonuclar: e-comde/results/", flush=True)
