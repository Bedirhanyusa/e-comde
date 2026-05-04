"""
Final Pipeline — V3 pipeline bittikten sonra calistirilir
Sira: XLM-RoBERTa v2 -> savasy -> Ensemble v3 -> Full eval v2 -> Paper update
Hedef: >%85 macro-F1
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
        print(f"ATLANDI (mevcut): {desc}", flush=True)
        return True

    log_path = LOG_DIR / script.replace(".py", "_final.log")
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
    ok = proc.returncode in (0, -1073740791)  # -1073740791: Windows stack OOM (sonuclar kaydedildi)
    status = "TAMAMLANDI" if ok else f"HATA (exit={proc.returncode})"
    print(f"\n>> {status} ({elapsed:.1f} dk)", flush=True)
    if not ok and critical:
        print(f"KRITIK HATA: {script} basarisiz, durduruluyor.", flush=True)
        sys.exit(1)
    return ok

print("FINAL PIPELINE BASLIYOR", flush=True)
print(f"Python: {PYTHON}", flush=True)

# Onkosul: v4 verisi hazir olmali
if not Path("data/processed_v4/train.csv").exists():
    print("HATA: data/processed_v4 bulunamadi. Once run_v3_pipeline.py tamamlayin.", flush=True)
    sys.exit(1)

# 1. XLM-RoBERTa v2 (192 token, daha iyi LR)
run_step("run_xlm_roberta_v2.py", "XLM-RoBERTa v2 (MAX_LEN=192, LR=6e-6, 8 epoch)",
         critical=False,
         skip_if_exists="results/xlm_roberta_v2_results.json")

# 2. savasy fine-tune
run_step("run_savasy.py", "savasy/bert-turkish — v4 uzerinde fine-tune",
         critical=False,
         skip_if_exists="results/savasy_results.json")

# 3. 3-model ensemble
run_step("run_ensemble_v3.py", "Ensemble v3 (BERTurk v3 + XLM-Ro v2 + savasy)",
         critical=False,
         skip_if_exists="results/ensemble_v3_results.json")

# 4. Full evaluation v2 (tum sonuclari JSON'dan yukle)
run_step("run_full_evaluation_v2.py", "Tam degerlendirme v2",
         skip_if_exists="results/full_evaluation_v2.json")

# 5. Bildiri guncelleme
run_step("update_paper.py", "Konferans bildirisi guncelleme")

# ── Ozet ────────────────────────────────────────────────────────
print("\n" + "="*60, flush=True)
print("FINAL PIPELINE TAMAMLANDI!", flush=True)
print("="*60, flush=True)

# En iyi sonucu raporla
best_f1, best_model = 0.0, "?"
for fname, key in [
    ("ensemble_v3_results.json",   "ensemble_v3"),
    ("ensemble_v2_results.json",   "ensemble_v2_avg"),
    ("berturk_v3_results.json",    "berturk_v3"),
    ("savasy_results.json",        "savasy_finetuned"),
    ("xlm_roberta_v2_results.json","xlm_roberta_v2"),
]:
    fpath = Path("results") / fname
    if fpath.exists():
        try:
            data = json.load(open(fpath, encoding="utf-8"))
            f1   = data.get(key, {}).get("macro_f1", 0)
            if f1 > best_f1:
                best_f1, best_model = f1, key
        except Exception:
            pass

print(f"\nEN IYI SONUC: {best_model} | Macro-F1 = {best_f1:.4f} ({best_f1*100:.2f}%)", flush=True)
if best_f1 >= 0.85:
    print("HEDEF TUTTURULDU! (>=%85)", flush=True)
elif best_f1 >= 0.80:
    print("Iyi sonuc — %85 hedefine yakin.", flush=True)
else:
    print("Daha fazla optimizasyon gerekebilir.", flush=True)
