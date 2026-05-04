"""
Tam Pipeline — Her şeyi sırayla çalıştırır:
1. prepare_data_v3    — 3 kaynak birleşik 30K veri seti
2. run_berturk_v2     — BERTurk ince ayar (sınıf ağırlıklı, LR 1e-5)
3. run_xlm_roberta    — XLM-RoBERTa ince ayar
4. run_absa           — ABSA modülü
5. run_summarizer     — mT5 özetleme + ROUGE
6. run_full_evaluation— Tüm modeller karşılaştırma tablosu
"""
import subprocess, sys, os, time
from pathlib import Path

# Windows cp1254 encode hatasını önle
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PYTHON  = sys.executable
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

STEPS = [
    # prepare_data_v3 tamamlandi (train=24K, val=3K, test=3K)
    ("run_berturk_v2.py",          "BERTurk v2 egitimi"),
    ("run_xlm_roberta.py",         "XLM-RoBERTa egitimi"),
    ("eval_berturk_zeroshot.py",   "BERTurk zero-shot (ablasyon icin)"),
    ("run_absa.py",                "ABSA modulu"),
    ("run_summarizer.py",          "mT5 ozetleme"),
    ("run_full_evaluation.py",     "Tam degerlendirme"),
    ("update_paper.py",            "Bildiri guncelleme"),
]

def run_step(script, desc):
    log_path = LOG_DIR / (script.replace(".py", ".log"))
    print(f"\n{'='*60}", flush=True)
    print(f"ADIM: {desc}", flush=True)
    print(f"Script: {script} | Log: {log_path}", flush=True)
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            [PYTHON, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            lf.write(line)
        proc.wait()

    elapsed = (time.time() - t0) / 60
    if proc.returncode == 0:
        print(f"\n>> TAMAMLANDI ({elapsed:.1f} dk)", flush=True)
        return True
    else:
        print(f"\n>> HATA! Exit code={proc.returncode} ({elapsed:.1f} dk)", flush=True)
        return False

os.chdir(Path(__file__).parent)
print("E-ComDe Tam Pipeline Basliyor", flush=True)
print(f"Python: {PYTHON}", flush=True)

for script, desc in STEPS:
    ok = run_step(script, desc)
    if not ok and script not in ("run_absa.py", "run_summarizer.py"):
        print(f"\nKRITIK HATA: {script} basarisiz, pipeline durduruluyor.", flush=True)
        sys.exit(1)

print("\n" + "="*60, flush=True)
print("TUM PIPELINE TAMAMLANDI!", flush=True)
print("Sonuclar: e-comde/results/", flush=True)
print("Loglar:   e-comde/logs/", flush=True)
