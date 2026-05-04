import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


SENTIMENT_PREFIX = {
    "olumlu":  "olumlu yorumlari ozetle: ",
    "olumsuz": "olumsuz yorumlari ozetle: ",
    "notr":    "notr yorumlari ozetle: ",
}

DEFAULT_MODEL_DIR = "models/mt5_summarizer"


class ReviewSummarizer:
    def __init__(self, model_name: str = DEFAULT_MODEL_DIR, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def summarize(
        self,
        reviews: list[str],
        sentiment: str = "genel",
        max_input_len: int = 384,
        max_new_tokens: int = 128,
        num_beams: int = 8,
        top_reviews: int = 15,
    ) -> str:
        prefix = SENTIMENT_PREFIX.get(sentiment, "yorumlari ozetle: ")
        combined = " [SEP] ".join(reviews[:top_reviews])
        input_text = prefix + combined

        inputs = self.tokenizer(
            input_text,
            truncation=True,
            max_length=max_input_len,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.2,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def summarize_by_sentiment(self, reviews: list[str], labels: list[str]) -> dict[str, str]:
        grouped: dict[str, list[str]] = {"olumlu": [], "olumsuz": [], "notr": []}
        for text, label in zip(reviews, labels):
            if label in grouped:
                grouped[label].append(text)

        return {
            sentiment: self.summarize(texts, sentiment)
            for sentiment, texts in grouped.items()
            if texts
        }
