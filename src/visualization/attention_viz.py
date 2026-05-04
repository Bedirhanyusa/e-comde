"""Attention-based token importance visualization for BERTurk predictions."""
import torch
import numpy as np
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification


LABEL_NAMES = ["olumsuz", "nötr", "olumlu"]


def get_attention_scores(text: str, model, tokenizer, device: str = "cpu") -> dict:
    """Returns token-level importance scores via attention rollout."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    pred   = outputs.logits.argmax(-1).item()
    proba  = torch.softmax(outputs.logits, dim=-1)[0].tolist()

    # Average attention across all heads, last layer
    attn       = outputs.attentions[-1][0]          # [heads, seq, seq]
    attn_mean  = attn.mean(dim=0)[0].cpu().numpy()  # CLS row → [seq]
    # Normalize to [0, 1]
    scores     = attn_mean / (attn_mean.max() + 1e-9)

    return {
        "text":       text,
        "tokens":     tokens,
        "scores":     scores.tolist(),
        "prediction": LABEL_NAMES[pred],
        "probabilities": {LABEL_NAMES[i]: round(p, 4) for i, p in enumerate(proba)},
    }


def explain_batch(texts: list[str], model_dir: str, out_path: str, device: str = "cpu") -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=3)
    model.to(device).eval()

    results = [get_attention_scores(t, model, tokenizer, device) for t in texts]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Attention explanations saved: {out_path}")


if __name__ == "__main__":
    samples = [
        "Ürün çok kaliteliydi, hızlı kargo ile geldi. Kesinlikle tavsiye ederim.",
        "Berbat bir ürün, tamamen bozuk geldi. Param gitti.",
        "Fiyatına göre idare eder, ne iyi ne kötü.",
    ]
    explain_batch(samples, "models/berturk_finetuned", "results/attention_explanations.json")
