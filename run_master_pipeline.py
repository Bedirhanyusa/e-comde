"""
Master Pipeline — BERTurk v3b bittikten sonra tam sıralı çalışma:
XLM-Ro v2 → savasy → ensemble v2 → ensemble v3 → full eval v2 → update paper v2
"""
import subprocess, sys, time, os
from pathlib import Path

os.chdir(Path(__file__).parent)

def log(msg): print(msg, flush=True)

PYTHON = sys.executable

STEPS = [
    ("XLM-RoBERTa v2",    "run_xlm_roberta_v2.py",      Path("results/xlm_roberta_v2_results.json")),
    ("savasy",             "run_savasy.py",                Path("results/savasy_results.json")),
    ("Ensemble v2",        "run_ensemble_v2.py",           Path("results/ensemble_v2_results.json")),
    ("Ensemble v3",        "run_ensemble_v3.py",           Path("results/ensemble_v3_results.json")),
    ("Threshold Tuning",   "run_threshold_tuning.py",      Path("results/ensemble_tuned_results.json")),
    ("Full evaluation v2", "run_full_evaluation_v2.py",    Path("results/full_evaluation_v2.json")),
    ("Update paper v2",    "update_paper_v2.py",           None),
]

# ── BERTurk v3b tamamlanana kadar bekle ─────────────────────────────
V3B_RESULT = Path("results/berturk_v3_results.json")
if not V3B_RESULT.exists():
    log("BERTurk v3b bekleniyor (results/berturk_v3_results.json)...")
    waited = 0
    while not V3B_RESULT.exists():
        time.sleep(60)
        waited += 1
        log(f"  ... {waited} dakika beklendi")
    log("BERTurk v3b tamamlandi!")
else:
    log("BERTurk v3b zaten tamamlanmis, devam ediliyor...")

# ── Sıralı adımlar ───────────────────────────────────────────────────
for name, script, result_path in STEPS:
    if result_path and result_path.exists():
        log(f"\n[ATLA] {name} — sonuc mevcut: {result_path}")
        continue

    log(f"\n{'='*60}")
    log(f"[BASLIYOR] {name}")
    log(f"{'='*60}")

    result = subprocess.run([PYTHON, script], check=False)
    if result.returncode != 0:
        log(f"[HATA] {name} basarisiz (kod: {result.returncode})")
        if script.startswith("run_xlm") or script.startswith("run_savasy"):
            log("Kritik egitim adimi basarisiz, pipeline durduruluyor.")
            sys.exit(1)
        log("Kritik olmayan adim, devam ediliyor...")
    else:
        log(f"[TAMAMLANDI] {name}")

log("\n" + "=" * 60)
log("TUM ADIMLAR TAMAMLANDI!")
log("=" * 60)
