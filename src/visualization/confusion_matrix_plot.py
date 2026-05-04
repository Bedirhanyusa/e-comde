"""Saves confusion matrix + per-class metrics chart as PNG files."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path


LABEL_NAMES = ["Olumsuz", "Nötr", "Olumlu"]
COLORS      = ["#E74C3C", "#F39C12", "#27AE60"]


def plot_confusion_matrix(y_true, y_pred, title: str, out_path: str) -> None:
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Kaydedildi: {out_path}")


def plot_model_comparison(results_json: str, out_path: str) -> None:
    with open(results_json, encoding="utf-8") as f:
        results = json.load(f)

    display_names = {
        "tfidf_lr":          "TF-IDF + LR",
        "bilstm":            "BiLSTM",
        "xlm_roberta":       "XLM-RoBERTa",
        "berturk_finetuned": "BERTurk",
    }
    order   = ["tfidf_lr", "bilstm", "xlm_roberta", "berturk_finetuned"]
    present = [k for k in order if k in results]

    labels   = [display_names.get(k, k) for k in present]
    accuracy = [results[k]["accuracy"] for k in present]
    macro_f1 = [results[k]["macro_f1"] for k in present]

    x   = np.arange(len(labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, accuracy, w, label="Accuracy",  color="#2980B9", alpha=0.85)
    ax.bar(x + w/2, macro_f1, w, label="Macro-F1", color="#27AE60", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("E-ComDe Model Karşılaştırması", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for i, (a, f) in enumerate(zip(accuracy, macro_f1)):
        ax.text(i - w/2, a + 0.005, f"{a:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w/2, f + 0.005, f"{f:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Kaydedildi: {out_path}")


if __name__ == "__main__":
    plot_model_comparison("results/full_evaluation.json", "results/figures/model_comparison.png")
