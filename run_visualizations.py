"""
Tum gorsellestirmeleri uretir — pipeline'in son adimi
Gerekli: results/full_evaluation.json mevcut olmali
"""
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")
FIGS_DIR    = Path("results/figures")
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DISPLAY_NAMES = {
    "tfidf_lr":          "TF-IDF + LR",
    "bilstm":            "BiLSTM",
    "xlm_roberta":       "XLM-RoBERTa",
    "berturk_finetuned": "BERTurk",
    "ensemble_avg":      "Ensemble",
}
MODEL_ORDER = ["tfidf_lr", "bilstm", "xlm_roberta", "berturk_finetuned", "ensemble_avg"]
LABEL_NAMES = ["Olumsuz", "Nötr", "Olumlu"]


def load_all_results() -> dict:
    combined = {}
    for fname in ["baseline_results.json", "berturk_results.json",
                  "xlm_roberta_results.json", "ensemble_results.json",
                  "full_evaluation.json"]:
        p = RESULTS_DIR / fname
        if p.exists():
            with open(p) as f:
                combined.update(json.load(f))
    return combined


# ── 1. Model karşılaştırma bar chart ─────────────────────────────
def plot_model_comparison(results: dict):
    present  = [k for k in MODEL_ORDER if k in results and "accuracy" in results[k]]
    if not present:
        print("Karsilastirma verisi yok, atlaniyor.")
        return

    labels   = [DISPLAY_NAMES.get(k, k) for k in present]
    accuracy = [results[k]["accuracy"] for k in present]
    macro_f1 = [results[k]["macro_f1"] for k in present]
    mcc      = [results[k].get("mcc", 0) for k in present]

    x, w = np.arange(len(labels)), 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w,   accuracy, w, label="Accuracy",  color="#2980B9", alpha=0.88)
    ax.bar(x,       macro_f1, w, label="Macro-F1", color="#27AE60", alpha=0.88)
    ax.bar(x + w,   mcc,      w, label="MCC",       color="#8E44AD", alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("E-ComDe: Model Performans Karşılaştırması", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for i, (a, f, m) in enumerate(zip(accuracy, macro_f1, mcc)):
        ax.text(i - w,   a + 0.01, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i,       f + 0.01, f"{f:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w,   m + 0.01, f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = FIGS_DIR / "model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Kaydedildi: {out}")


# ── 2. Per-class F1 radar chart ───────────────────────────────────
def plot_per_class_f1(results: dict):
    present = [k for k in MODEL_ORDER if k in results and "per_class" in results[k]]
    if not present:
        return

    fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4))
    if len(present) == 1:
        axes = [axes]

    class_keys = ["olumsuz", "notr", "olumlu"]
    for ax, key in zip(axes, present):
        pc = results[key]["per_class"]
        p_vals = [pc[c]["precision"] for c in class_keys]
        r_vals = [pc[c]["recall"]    for c in class_keys]
        f_vals = [pc[c]["f1"]        for c in class_keys]

        x = np.arange(len(LABEL_NAMES))
        w = 0.25
        ax.bar(x - w, p_vals, w, label="Precision", color="#2980B9", alpha=0.85)
        ax.bar(x,     r_vals, w, label="Recall",    color="#27AE60", alpha=0.85)
        ax.bar(x + w, f_vals, w, label="F1",        color="#E67E22", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(LABEL_NAMES, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_title(DISPLAY_NAMES.get(key, key), fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Sınıf Bazlı Metrikler", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = FIGS_DIR / "per_class_metrics.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Kaydedildi: {out}")


# ── 3. Confusion matrix heatmaps ──────────────────────────────────
def plot_confusion_matrices(results: dict):
    from sklearn.metrics import ConfusionMatrixDisplay
    import numpy as np

    present = [k for k in MODEL_ORDER if k in results and "confusion_matrix" in results[k]]
    if not present:
        print("Confusion matrix verisi yok, atlaniyor.")
        return

    for key in present:
        cm = np.array(results[key]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {DISPLAY_NAMES.get(key, key)}",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        out = FIGS_DIR / f"cm_{key}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Kaydedildi: {out}")


# ── 4. ROC-AUC summary ────────────────────────────────────────────
def plot_auc_comparison(results: dict):
    present = [k for k in MODEL_ORDER if k in results and "roc_auc_macro" in results[k]]
    if len(present) < 2:
        print("AUC karsilastirmasi icin yeterli model yok, atlaniyor.")
        return

    labels = [DISPLAY_NAMES.get(k, k) for k in present]
    aucs   = [results[k]["roc_auc_macro"] for k in present]
    kappas = [results[k].get("kappa", 0)  for k in present]

    x, w = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, aucs,   w, label="ROC-AUC",     color="#16A085", alpha=0.88)
    ax.bar(x + w/2, kappas, w, label="Cohen Kappa", color="#2C3E50", alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("ROC-AUC ve Cohen Kappa Karşılaştırması", fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for i, (a, k) in enumerate(zip(aucs, kappas)):
        ax.text(i - w/2, a + 0.01, f"{a:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w/2, k + 0.01, f"{k:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = FIGS_DIR / "auc_kappa_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Kaydedildi: {out}")


if __name__ == "__main__":
    results = load_all_results()
    print(f"Yuklenen model sayisi: {len(results)}")
    print(f"Modeller: {list(results.keys())}\n")

    plot_model_comparison(results)
    plot_per_class_f1(results)
    plot_confusion_matrices(results)
    plot_auc_comparison(results)

    print(f"\nTum gorseller: {FIGS_DIR}/")
