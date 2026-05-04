"""
E-ComDe Master Pipeline
Tum modelleri sirayla calistirir, her adimda ilerlemeyi kaydeder.
Yeniden baslatildiginda tamamlanmis adimlari atlar.

Kullanim: python run_pipeline.py
"""
import os, sys, json, subprocess, time
from pathlib import Path

PIPELINE_STATE = Path("results/pipeline_state.json")
RESULTS_DIR    = Path("results")
PYTHON         = sys.executable


def load_state() -> dict:
    if PIPELINE_STATE.exists():
        with open(PIPELINE_STATE) as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(PIPELINE_STATE, "w") as f:
        json.dump(state, f, indent=2)


def run_step(name: str, script: str, state: dict) -> bool:
    if state.get(name) == "done":
        print(f"[ATLA] {name} — zaten tamamlandi")
        return True

    print(f"\n{'='*60}")
    print(f"[BASLIYOR] {name}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(
        [PYTHON, script],
        cwd=Path(__file__).parent,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "TOKENIZERS_PARALLELISM": "false"},
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        state[name] = "done"
        state[f"{name}_elapsed_s"] = round(elapsed)
        save_state(state)
        print(f"[TAMAM] {name} — {elapsed/60:.1f} dakika")
        return True
    else:
        state[name] = "failed"
        state[f"{name}_returncode"] = result.returncode
        save_state(state)
        print(f"[HATA] {name} — return code {result.returncode}")
        return False


def main():
    state = load_state()
    print("E-ComDe Master Pipeline")
    print(f"Mevcut durum: {state}\n")

    steps = [
        ("baselines",        "run_baselines.py"),
        ("berturk",          "run_berturk.py"),
        ("xlm_roberta",      "run_xlm_roberta.py"),
        ("absa",             "run_absa.py"),
        ("summarizer",       "run_summarizer.py"),
        ("ensemble",         "run_ensemble.py"),
        ("full_evaluation",  "run_full_evaluation.py"),
        ("visualizations",   "run_visualizations.py"),
    ]

    failed = []
    for name, script in steps:
        if not Path(script).exists():
            print(f"[ATLA] {script} bulunamadi")
            continue
        ok = run_step(name, script, state)
        if not ok:
            failed.append(name)
            print(f"[UYARI] {name} basarisiz, pipeline devam ediyor...")

    print(f"\n{'='*60}")
    print("PIPELINE TAMAMLANDI")
    print(f"{'='*60}")
    if failed:
        print(f"Basarisiz adimlar: {failed}")
    else:
        print("Tum adimlar basariyla tamamlandi!")

    # Nihai ozet tablosu
    full_eval = RESULTS_DIR / "full_evaluation.json"
    if full_eval.exists():
        with open(full_eval) as f:
            results = json.load(f)
        print(f"\n{'Model':<30} {'Acc':>8} {'F1':>8} {'MCC':>8}")
        print("-" * 58)
        for model, r in results.items():
            if isinstance(r, dict) and "accuracy" in r:
                print(f"{model:<30} {r['accuracy']:>8.4f} {r['macro_f1']:>8.4f} "
                      f"{r.get('mcc', float('nan')):>8.4f}")


if __name__ == "__main__":
    main()
