import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL_MAP = {0: "olumsuz", 1: "notr", 2: "olumlu"}


class SentimentClassifier:
    def __init__(self, model_dir: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts: list[str], max_len: int = 192, confidence_threshold: float = 0.75):
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        with torch.no_grad():
            logits = self.model(**encodings).logits
            probs = torch.softmax(logits, dim=-1)
            confidences, pred_ids = probs.max(dim=-1)

        results = []
        for text, pred_id, conf in zip(texts, pred_ids.tolist(), confidences.tolist()):
            results.append({
                "text": text,
                "label": LABEL_MAP[pred_id],
                "confidence": round(conf, 4),
                "flagged": conf < confidence_threshold,
            })
        return results
