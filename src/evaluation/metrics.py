import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

LABEL_NAMES = ["olumsuz", "notr", "olumlu"]


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_proba: list[list[float]] | None = None,
) -> dict:
    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    mcc      = matthews_corrcoef(y_true, y_pred)
    kappa    = cohen_kappa_score(y_true, y_pred)
    report   = classification_report(y_true, y_pred, target_names=LABEL_NAMES, output_dict=True)
    cm       = confusion_matrix(y_true, y_pred)

    result = {
        "accuracy":  round(acc, 4),
        "macro_f1":  round(macro_f1, 4),
        "mcc":       round(mcc, 4),       # Matthews Correlation Coefficient
        "kappa":     round(kappa, 4),     # Cohen's Kappa
        "per_class": {
            name: {
                "precision": round(report[name]["precision"], 4),
                "recall":    round(report[name]["recall"], 4),
                "f1":        round(report[name]["f1-score"], 4),
                "support":   int(report[name]["support"]),
            }
            for name in LABEL_NAMES
        },
        "confusion_matrix": cm.tolist(),
    }

    # ROC-AUC (softmax olasılıkları varsa)
    if y_proba is not None:
        try:
            y_bin = label_binarize(y_true, classes=[0, 1, 2])
            auc_macro = roc_auc_score(y_bin, np.array(y_proba), average="macro", multi_class="ovr")
            result["roc_auc_macro"] = round(auc_macro, 4)

            per_class_auc = {}
            for i, name in enumerate(LABEL_NAMES):
                auc = roc_auc_score(y_bin[:, i], np.array(y_proba)[:, i])
                per_class_auc[name] = round(auc, 4)
            result["roc_auc_per_class"] = per_class_auc
        except Exception:
            pass

    return result


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    import evaluate
    rouge = evaluate.load("rouge")
    result = rouge.compute(predictions=predictions, references=references)
    return {k: round(v, 4) for k, v in result.items()}


def compute_bertscore(predictions: list[str], references: list[str], lang: str = "tr") -> dict:
    from bert_score import score as bs_score
    P, R, F1 = bs_score(predictions, references, lang=lang, verbose=False)
    return {
        "bertscore_precision": round(P.mean().item(), 4),
        "bertscore_recall":    round(R.mean().item(), 4),
        "bertscore_f1":        round(F1.mean().item(), 4),
    }


def print_metrics(metrics: dict, model_name: str = "Model"):
    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  Macro-F1       : {metrics['macro_f1']:.4f}")
    print(f"  MCC            : {metrics['mcc']:.4f}")
    print(f"  Cohen Kappa    : {metrics['kappa']:.4f}")
    if "roc_auc_macro" in metrics:
        print(f"  ROC-AUC (macro): {metrics['roc_auc_macro']:.4f}")
    print()
    print(f"  {'Sinif':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*44}")
    for name, vals in metrics["per_class"].items():
        print(f"  {name:<12} {vals['precision']:>10.4f} {vals['recall']:>10.4f} {vals['f1']:>10.4f}")
    if "roc_auc_per_class" in metrics:
        print()
        print(f"  ROC-AUC per sinif:")
        for name, auc in metrics["roc_auc_per_class"].items():
            print(f"    {name}: {auc:.4f}")
