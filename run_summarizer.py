"""
Aşama 4: mT5 özetleme modülü değerlendirmesi
ROUGE metrikleri + nitel örnek çıktı
"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def log(msg): print(msg, flush=True)

import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging as hf_logging
hf_logging.set_verbosity_error()
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "google/mt5-small"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SENTIMENT_PREFIX = {
    "olumlu":  "olumlu yorumlari ozetle",
    "olumsuz": "olumsuz yorumlari ozetle",
    "notr":    "notr yorumlari ozetle",
}
LABEL_MAP = {0: "olumsuz", 1: "notr", 2: "olumlu"}

log(f"mT5-small yukleniyor ({DEVICE})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()
log("Model yuklendi.")


def summarize(reviews: list[str], sentiment: str, max_input: int = 512, max_new: int = 128) -> str:
    prefix   = SENTIMENT_PREFIX.get(sentiment, "ozetle")
    combined = " [SEP] ".join(reviews[:50])
    inp_text = f"{prefix}: {combined}"

    inputs = tokenizer(
        inp_text, truncation=True, max_length=max_input, return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ── Veri yükle ──────────────────────────────────────────────────
v4_test = Path("data/processed_v4/test.csv")
v3_test = Path("data/processed/test.csv")
test_df = pd.read_csv(v4_test if v4_test.exists() else v3_test)
test_df["sentiment"] = test_df["label"].map(LABEL_MAP)

# ── Her duygu için özet üret ─────────────────────────────────────
log("\n" + "=" * 60)
log("DUYGU BAZLI OZET URETIMI")
log("=" * 60)

summaries = {}
for sentiment in ["olumlu", "olumsuz", "notr"]:
    reviews = test_df[test_df["sentiment"] == sentiment]["Review"].astype(str).tolist()
    log(f"\n[{sentiment.upper()}] — {len(reviews)} yorum")
    summary = summarize(reviews, sentiment)
    summaries[sentiment] = summary
    log(f"Ozet: {summary}")

# ── Elektronik olumsuz özelinde örnek (bildiri Tablo 6 eşdeğeri) ─
log("\n" + "=" * 60)
log("KATEGORI BAZLI ORNEK: Elektronik - Olumsuz")
log("=" * 60)
if "category" in test_df.columns:
    elektr_neg = test_df[
        (test_df["category"] == "Elektronik") & (test_df["sentiment"] == "olumsuz")
    ]["Review"].astype(str).tolist()
    if elektr_neg:
        summary_elekt = summarize(elektr_neg, "olumsuz")
        log(f"Yorum sayisi: {len(elektr_neg)}")
        log(f"Ozet: {summary_elekt}")

# ── ROUGE değerlendirmesi ────────────────────────────────────────
log("\n" + "=" * 60)
log("ROUGE DEGERLENDIRMESI")
log("=" * 60)

try:
    import evaluate
    rouge = evaluate.load("rouge")

    # Her duygu için 5'er örnek oluştur, modelin özetini referansla karşılaştır
    predictions, references = [], []
    for sentiment in ["olumlu", "olumsuz", "notr"]:
        reviews = test_df[test_df["sentiment"] == sentiment]["Review"].astype(str).tolist()
        # 5 farklı 10'luk batch üret
        for i in range(0, min(50, len(reviews)), 10):
            batch = reviews[i:i+10]
            if len(batch) < 3:
                continue
            pred = summarize(batch, sentiment, max_input=256, max_new=64)
            # Referans: ilk 3 yorumu birleştir (gerçek içerik)
            ref = " ".join(batch[:3])[:200]
            predictions.append(pred)
            references.append(ref)

    if predictions:
        scores = rouge.compute(predictions=predictions, references=references)
        log(f"ROUGE-1 : {scores['rouge1']:.4f}")
        log(f"ROUGE-2 : {scores['rouge2']:.4f}")
        log(f"ROUGE-L : {scores['rougeL']:.4f}")

        rouge_results = {k: round(v, 4) for k, v in scores.items()}
        with open(RESULTS_DIR / "rouge_results.json", "w") as f:
            json.dump({"rouge": rouge_results, "summaries": summaries}, f, indent=2, ensure_ascii=False)
        log("Sonuclar kaydedildi: results/rouge_results.json")
    else:
        log("Yeterli veri yok.")

except Exception as e:
    log(f"ROUGE hesaplanamadi: {e}")
    with open(RESULTS_DIR / "rouge_results.json", "w") as f:
        json.dump({"summaries": summaries}, f, indent=2, ensure_ascii=False)

log("\nOzetleme tamamlandi.")
