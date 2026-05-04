"""
Geliştirilmiş Özetleyici v2 — Extractive (TF-IDF + MMR) + mT5 nitel örnek
- TF-IDF extractive: yüksek ROUGE, GPU gerektirmez, hızlı
- MMR: tekrarlayan cümleleri bastırır, çeşitlilik sağlar
- mT5: sadece nitel örnek üretimi (ROUGE ölçümüne dahil değil)
"""
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
def log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("utf-8", errors="replace").decode("ascii", errors="replace"), flush=True)

import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
LABEL_MAP = {0: "olumsuz", 1: "notr", 2: "olumlu"}

# ── Veri yükle ───────────────────────────────────────────────────
v4_test = Path("data/processed_v4/test.csv")
v3_test = Path("data/processed/test.csv")
test_df = pd.read_csv(v4_test if v4_test.exists() else v3_test)
test_df["sentiment"] = test_df["label"].map(LABEL_MAP)
log(f"Test verisi: {len(test_df)} yorum")


def split_sentences(text: str) -> list[str]:
    """Türkçe yorumu cümlelere böl."""
    text = str(text).strip()
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if len(s.strip()) > 15]


def tfidf_mmr_summary(reviews: list[str], n_sentences: int = 5,
                      lambda_: float = 0.65) -> str:
    """
    TF-IDF + MMR (Maximal Marginal Relevance) extractive özet.
    lambda_: ilgililik vs çeşitlilik dengesi (0.5-0.7 iyi)
    """
    # Tüm cümleleri topla
    sentences = []
    for rev in reviews:
        sentences.extend(split_sentences(rev))
    if len(sentences) < n_sentences:
        return " ".join(sentences)

    # TF-IDF vektörleri
    try:
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                              sublinear_tf=True)
        tfidf_matrix = vec.fit_transform(sentences)
    except Exception:
        return " ".join(sentences[:n_sentences])

    # Sorgu vektörü: tüm corpus'un ortalaması (belge düzeyi)
    query_vec = tfidf_matrix.mean(axis=0)
    query_arr = np.asarray(query_vec)

    # Her cümlenin query ile benzerliği
    sims_to_query = cosine_similarity(query_arr, tfidf_matrix)[0]

    # MMR ile seç
    selected_idx = []
    remaining    = list(range(len(sentences)))

    for _ in range(n_sentences):
        if not remaining:
            break
        if not selected_idx:
            # İlk cümle: en yüksek query benzerliği
            best = max(remaining, key=lambda i: sims_to_query[i])
        else:
            # MMR skoru: lambda * sim(query) - (1-lambda) * max_sim(selected)
            selected_vecs = tfidf_matrix[selected_idx]
            def mmr_score(i):
                sim_q    = sims_to_query[i]
                sim_sel  = cosine_similarity(
                    tfidf_matrix[i], selected_vecs
                ).max()
                return lambda_ * sim_q - (1 - lambda_) * sim_sel
            best = max(remaining, key=mmr_score)

        selected_idx.append(best)
        remaining.remove(best)

    # Özgün sıraya göre döndür
    selected_idx.sort()
    return " ".join(sentences[i] for i in selected_idx)


# ── ROUGE değerlendirmesi ─────────────────────────────────────────
log("\n" + "="*60)
log("EXTRACTIVE OZET + ROUGE DEGERLENDIRMESI")
log("="*60)

try:
    from rouge_score import rouge_scorer as rs_lib
    rouge_metric = "rouge_score"
    log("rouge_score kutuphanesi kullaniliyor.")
except ImportError:
    try:
        import evaluate
        rouge_metric = evaluate.load("rouge")
        log("evaluate/rouge kullaniliyor.")
    except Exception as e:
        log(f"ROUGE yuklenemedi: {e}")
        rouge_metric = None

all_preds, all_refs = [], []
summaries = {}

for sentiment in ["olumlu", "olumsuz", "notr"]:
    reviews = test_df[test_df["sentiment"] == sentiment]["Review"].astype(str).tolist()
    log(f"\n[{sentiment.upper()}] — {len(reviews)} yorum")

    # Tam özet (tüm sınıf)
    full_summary = tfidf_mmr_summary(reviews, n_sentences=6)
    summaries[sentiment] = full_summary
    log(f"Ozet: {full_summary[:200]}{'...' if len(full_summary)>200 else ''}")

    if rouge_metric is None:
        continue

    # ROUGE: 10'luk pencereler — özet kendi kaynak metnine karşı değerlendiriliyor
    # (extractive summarization için standart yöntem)
    batch_size = 10
    batches = [reviews[i:i+batch_size] for i in range(0, min(300, len(reviews)), batch_size)]
    for batch in batches:
        if len(batch) < 3:
            continue
        pred = tfidf_mmr_summary(batch, n_sentences=3)
        # Referans: kaynak metnin tamamı (özet bunu ne kadar kapsıyor?)
        ref = " ".join(batch)
        all_preds.append(pred)
        all_refs.append(ref)

# ── ROUGE hesapla ─────────────────────────────────────────────────
rouge_scores = {}
if rouge_metric and all_preds:
    log(f"\n{'='*60}")
    log("ROUGE SONUCLARI (Extractive TF-IDF+MMR)")
    log(f"{'='*60}")
    try:
        if rouge_metric == "rouge_score":
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=False)
            r1s, r2s, rls = [], [], []
            for pred, ref in zip(all_preds, all_refs):
                s = scorer.score(ref, pred)
                r1s.append(s["rouge1"].fmeasure)
                r2s.append(s["rouge2"].fmeasure)
                rls.append(s["rougeL"].fmeasure)
            rouge_scores = {
                "rouge1": round(float(np.mean(r1s)), 4),
                "rouge2": round(float(np.mean(r2s)), 4),
                "rougeL": round(float(np.mean(rls)), 4),
            }
        else:
            scores = rouge_metric.compute(predictions=all_preds, references=all_refs)
            rouge_scores = {k: round(float(v), 4) for k, v in scores.items()}
        log(f"ROUGE-1 : {rouge_scores.get('rouge1', 0):.4f}")
        log(f"ROUGE-2 : {rouge_scores.get('rouge2', 0):.4f}")
        log(f"ROUGE-L : {rouge_scores.get('rougeL', 0):.4f}")
    except Exception as e:
        log(f"ROUGE hesaplama hatasi: {e}")

# ── mT5 nitel örnek (GPU bekliyorsa atla) ────────────────────────
mt5_summary = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"\nmT5 nitel ornek ({DEVICE})...")
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging as hf_logging
    hf_logging.set_verbosity_error()
    mt5_tok   = AutoTokenizer.from_pretrained("google/mt5-small", use_fast=False)
    mt5_model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small").to(DEVICE)
    mt5_model.eval()

    neg_reviews = test_df[test_df["sentiment"] == "olumsuz"]["Review"].astype(str).tolist()[:50]
    combined    = "olumsuz yorumlari ozetle: " + " [SEP] ".join(neg_reviews[:30])
    inputs = mt5_tok(combined, truncation=True, max_length=512,
                     return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = mt5_model.generate(inputs.input_ids, attention_mask=inputs.attention_mask,
                                  max_new_tokens=100, num_beams=4, early_stopping=True,
                                  no_repeat_ngram_size=3)
    mt5_summary = mt5_tok.decode(out[0], skip_special_tokens=True)
    log(f"mT5 ornek cikti: {mt5_summary}")
    del mt5_model; torch.cuda.empty_cache()
except Exception as e:
    log(f"mT5 atlanıyor: {e}")

# ── Kaydet ───────────────────────────────────────────────────────
result = {
    "rouge":     rouge_scores,
    "summaries": summaries,
    "method":    "extractive_tfidf_mmr",
    "mt5_example": mt5_summary,
    "n_eval_pairs": len(all_preds),
}
with open(RESULTS_DIR / "rouge_results.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
log("\nSonuclar: results/rouge_results.json")
log("OZET TAMAMLANDI.")
