"""
Pipeline bittikten sonra calistirilacak son adimlar:
1. Baseline'lari yeni 30K test setiyle yeniden hesapla
2. BERTurk zero-shot degerlendirme (ablasyon)
3. Tam degerlendirmeyi yeniden calistir (dogru baseline dahil)
4. Bildiriyi gercek sonuclarla guncelle
"""
import subprocess, sys, os, time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PYTHON  = sys.executable
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

STEPS = [
    ("run_baselines.py",          "Baseline'lar (yeni 30K test seti)"),
    ("eval_berturk_zeroshot.py",  "BERTurk zero-shot (ablasyon)"),
    ("run_full_evaluation.py",    "Tam degerlendirme (guncellenmis)"),
    ("update_paper.py",           "Bildiri guncelleme"),
]

def run_step(script, desc):
    log_path = LOG_DIR / (script.replace(".py", "_post.log"))
    print(f"\n{'='*60}", flush=True)
    print(f"ADIM: {desc}", flush=True)
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
    return ok

os.chdir(Path(__file__).parent)
print("Post-Pipeline Adimlar Basliyor", flush=True)

for script, desc in STEPS:
    run_step(script, desc)

print("\n" + "="*60, flush=True)
print("TUM POST-PIPELINE ADIMLAR TAMAMLANDI!", flush=True)
