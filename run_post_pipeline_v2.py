"""
Master pipeline bittikten sonra çalıştırılır:
Summarizer (ROUGE) → update_paper_v2 (final)
"""
import subprocess, sys, time, os
from pathlib import Path

os.chdir(Path(__file__).parent)
def log(msg): print(msg, flush=True)
PYTHON = sys.executable

# Master pipeline bitene kadar bekle
MASTER_DONE_MARKER = Path("results/full_evaluation_v2.json")
if not MASTER_DONE_MARKER.exists():
    log("Master pipeline bekleniyor (results/full_evaluation_v2.json)...")
    waited = 0
    while not MASTER_DONE_MARKER.exists():
        time.sleep(60)
        waited += 1
        log(f"  ... {waited} dakika beklendi")
    log("Master pipeline tamamlandi!")
else:
    log("Master pipeline zaten bitti, devam ediliyor...")

STEPS = [
    ("Summarizer v3 (mT5 fine-tuned)", "run_summarizer_v3.py", Path("results/rouge_results.json")),
    ("Update paper v2",    "update_paper_v2.py",  None),
]

for name, script, result_path in STEPS:
    if result_path and result_path.exists():
        log(f"\n[ATLA] {name} — sonuc mevcut")
        continue
    log(f"\n{'='*60}\n[BASLIYOR] {name}\n{'='*60}")
    r = subprocess.run([PYTHON, script], check=False)
    if r.returncode != 0:
        log(f"[HATA] {name} (kod: {r.returncode})")
    else:
        log(f"[TAMAMLANDI] {name}")

log("\n" + "="*60)
log("POST-PIPELINE TAMAMLANDI — Her sey hazir!")
log("="*60)
