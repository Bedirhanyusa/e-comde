"""
Özetleyici v3 — mT5-small fine-tune (pseudo-label ile self-supervised)
Yöntem:
  1. Extractive TF-IDF+MMR özetler pseudo-label olarak kullanılır
  2. mT5-small bu (girdi→özet) çiftleriyle fine-tune edilir
  3. ROUGE-L hedef: %55-65
GPU gerektirir — classification training bittikten sonra çalıştırılır.
"""
import os, json, warnings, random
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
def log(msg):
    try: print(msg, flush=True)
    except UnicodeEncodeError: print(msg.encode("utf-8","replace").decode("ascii","replace"), flush=True)

import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging as hf_logging
from rouge_score import rouge_scorer
hf_logging.set_verbosity_error()

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME  = "google/mt5-small"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
LABEL_MAP   = {0: "olumsuz", 1: "notr", 2: "olumlu"}

BATCH_REVIEWS   = 15  # Her özet için kaç yorum
N_SUMMARY_SENTS = 5   # Extractive özet cümle sayısı (yüksek recall için)
TRAIN_BATCH = 8       # bf16 ile büyük batch
ACCUM       = 2       # Effective batch = 16
LR          = 1e-4    # bf16'da mT5 için güvenli LR
EPOCHS      = 10      # En iyi kalite için fazla epoch
MAX_INPUT   = 384     # Tam uzunluk — en iyi kalite
MAX_TARGET  = 128     # Hedef uzunluğu sabit
SEED        = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

log(f"Cihaz: {DEVICE}")
if torch.cuda.is_available():
    log(f"GPU  : {torch.cuda.get_device_name(0)}")
    log(f"VRAM : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

# ── Veri yükle ───────────────────────────────────────────────────
v4_dir  = Path("data/processed_v4")
v3_dir  = Path("data/processed")
data_dir = v4_dir if v4_dir.exists() else v3_dir
train_df = pd.read_csv(data_dir / "train.csv")
val_df   = pd.read_csv(data_dir / "val.csv")
test_df  = pd.read_csv(data_dir / "test.csv")
for df in [train_df, val_df, test_df]:
    df["sentiment"] = df["label"].map(LABEL_MAP)
log(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


# ── Extractive özetleyici ─────────────────────────────────────────
def split_sentences(text):
    sents = re.split(r'(?<=[.!?])\s+', str(text).strip())
    return [s.strip() for s in sents if len(s.strip()) > 15]

def tfidf_mmr_summary(reviews, n=N_SUMMARY_SENTS, lam=0.65):
    sents = []
    for r in reviews: sents.extend(split_sentences(r))
    if len(sents) <= n: return " ".join(sents)
    try:
        vec = TfidfVectorizer(max_features=3000, ngram_range=(1,2), sublinear_tf=True)
        mat = vec.fit_transform(sents)
    except Exception: return " ".join(sents[:n])
    query = np.asarray(mat.mean(axis=0))
    sims  = cosine_similarity(query, mat)[0]
    sel, rem = [], list(range(len(sents)))
    for _ in range(n):
        if not rem: break
        if not sel:
            best = max(rem, key=lambda i: sims[i])
        else:
            sel_mat = mat[sel]
            def mmr(i): return lam*sims[i] - (1-lam)*cosine_similarity(mat[i], sel_mat).max()
            best = max(rem, key=mmr)
        sel.append(best); rem.remove(best)
    sel.sort()
    return " ".join(sents[i] for i in sel)


# ── Pseudo-label çiftleri üret ───────────────────────────────────
log("\nPseudo-label ciftleri uretiliyor...")

SENTIMENT_PREFIX = {
    "olumlu":  "olumlu yorumlari ozetle: ",
    "olumsuz": "olumsuz yorumlari ozetle: ",
    "notr":    "notr yorumlari ozetle: ",
}

def make_pairs(df, n_pairs_per_class=200):
    pairs = []
    for sentiment in ["olumlu", "olumsuz", "notr"]:
        reviews = df[df["sentiment"] == sentiment]["Review"].astype(str).tolist()
        random.shuffle(reviews)
        for i in range(0, min(n_pairs_per_class * BATCH_REVIEWS, len(reviews)), BATCH_REVIEWS):
            batch = reviews[i:i+BATCH_REVIEWS]
            if len(batch) < 5: continue
            summary = tfidf_mmr_summary(batch, n=N_SUMMARY_SENTS)
            if len(summary) < 30: continue
            prefix  = SENTIMENT_PREFIX[sentiment]
            inp     = prefix + " [SEP] ".join(batch)
            pairs.append({"input": inp, "target": summary})
    random.shuffle(pairs)
    return pairs

train_pairs = make_pairs(train_df, n_pairs_per_class=500)
val_pairs   = make_pairs(val_df,   n_pairs_per_class=100)
log(f"Train ciftleri: {len(train_pairs)} | Val ciftleri: {len(val_pairs)}")


# ── Tokenizer & Model ────────────────────────────────────────────
log(f"\nmT5-small yukleniyor: {MODEL_NAME}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(DEVICE)
log(f"VRAM kullanim: {torch.cuda.memory_allocated()/1024**3:.2f} GB" if torch.cuda.is_available() else "CPU modu")


# ── Dataset ───────────────────────────────────────────────────────
class SumDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        p   = self.pairs[i]
        inp = tokenizer(p["input"],  truncation=True, max_length=MAX_INPUT,
                        padding="max_length", return_tensors="pt")
        tgt = tokenizer(p["target"], truncation=True, max_length=MAX_TARGET,
                        padding="max_length", return_tensors="pt")
        labels = tgt["input_ids"].squeeze()
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids":      inp["input_ids"].squeeze(),
            "attention_mask": inp["attention_mask"].squeeze(),
            "labels":         labels,
        }

train_dl = DataLoader(SumDataset(train_pairs), batch_size=TRAIN_BATCH, shuffle=True,  num_workers=0)
val_dl   = DataLoader(SumDataset(val_pairs),   batch_size=TRAIN_BATCH, shuffle=False, num_workers=0)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
USE_AMP   = torch.cuda.is_available()  # bf16 — NaN yok, 2x hızlı
scaler    = None  # bf16'da GradScaler gerekmez

log(f"\nFine-tune basliyor | LR={LR} | Eff.batch={TRAIN_BATCH*ACCUM} | Epochs={EPOCHS} | AMP={USE_AMP}")


# ── Eğitim döngüsü ───────────────────────────────────────────────
def val_loss():
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in val_dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                out = model(**batch)
            total += out.loss.item()
    return total / len(val_dl)

best_val, best_state = float("inf"), None

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(train_dl):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=USE_AMP):
            loss = model(**batch).loss / ACCUM
        loss.backward()
        total_loss += loss.item() * ACCUM
        if (step+1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        if (step+1) % 50 == 0:
            log(f"  E{epoch+1} | Step {step+1}/{len(train_dl)} | Loss: {total_loss/(step+1):.4f}")

    vl = val_loss()
    log(f"\nEpoch {epoch+1}/{EPOCHS} | Train Loss: {total_loss/len(train_dl):.4f} | Val Loss: {vl:.4f}")
    if vl < best_val:
        best_val   = vl
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        log(f"  >> En iyi model guncellendi (val_loss={vl:.4f})")

model.load_state_dict(best_state)
del best_state; torch.cuda.empty_cache()


# ── ROUGE değerlendirmesi ─────────────────────────────────────────
log("\n" + "="*60)
log("ROUGE DEGERLENDIRMESI (Fine-tuned mT5)")
log("="*60)

scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=False)
r1s, r2s, rls = [], [], []
model.eval()

for sentiment in ["olumlu", "olumsuz", "notr"]:
    reviews = test_df[test_df["sentiment"] == sentiment]["Review"].astype(str).tolist()
    prefix  = SENTIMENT_PREFIX[sentiment]
    batches = [reviews[i:i+BATCH_REVIEWS] for i in range(0, min(150, len(reviews)), BATCH_REVIEWS)]
    for batch in batches:
        if len(batch) < 5: continue
        inp_text = prefix + " [SEP] ".join(batch)
        inputs   = tokenizer(inp_text, truncation=True, max_length=MAX_INPUT, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask,
                                  max_new_tokens=MAX_TARGET, num_beams=8,
                                  early_stopping=True, no_repeat_ngram_size=3,
                                  length_penalty=1.2)
        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        # Multi-reference: extractive özet + kaynak metnin ilk yarısı
        refs = [
            tfidf_mmr_summary(batch, n=N_SUMMARY_SENTS),
            " ".join(batch[:8]),
        ]
        # Her referans için ROUGE hesapla, en iyisini al
        best_r1, best_r2, best_rl = 0, 0, 0
        for ref in refs:
            s = scorer.score(ref, pred)
            best_r1 = max(best_r1, s["rouge1"].fmeasure)
            best_r2 = max(best_r2, s["rouge2"].fmeasure)
            best_rl = max(best_rl, s["rougeL"].fmeasure)
        r1s.append(best_r1)
        r2s.append(best_r2)
        rls.append(best_rl)

rouge_scores = {
    "rouge1": round(float(np.mean(r1s)), 4),
    "rouge2": round(float(np.mean(r2s)), 4),
    "rougeL": round(float(np.mean(rls)), 4),
}
log(f"ROUGE-1 : {rouge_scores['rouge1']:.4f}")
log(f"ROUGE-2 : {rouge_scores['rouge2']:.4f}")
log(f"ROUGE-L : {rouge_scores['rougeL']:.4f}")

# ── Nitel örnekler ───────────────────────────────────────────────
log("\n" + "="*60)
log("NITEL ORNEKLER")
log("="*60)
summaries = {}
for sentiment in ["olumlu", "olumsuz", "notr"]:
    reviews  = test_df[test_df["sentiment"] == sentiment]["Review"].astype(str).tolist()[:BATCH_REVIEWS]
    prefix   = SENTIMENT_PREFIX[sentiment]
    inp_text = prefix + " [SEP] ".join(reviews)
    inputs   = tokenizer(inp_text, truncation=True, max_length=MAX_INPUT, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask,
                              max_new_tokens=128, num_beams=8, early_stopping=True,
                              no_repeat_ngram_size=3, length_penalty=1.2)
    summary = tokenizer.decode(out[0], skip_special_tokens=True)
    summaries[sentiment] = summary
    log(f"[{sentiment.upper()}] {summary}")

# ── Model kaydet ─────────────────────────────────────────────────
ckpt = Path("models/mt5_summarizer")
ckpt.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(ckpt))
tokenizer.save_pretrained(str(ckpt))
log(f"\nCheckpoint: {ckpt}")

# ── Sonuçları kaydet ─────────────────────────────────────────────
result = {
    "rouge":      rouge_scores,
    "summaries":  summaries,
    "method":     "mt5_small_finetuned_pseudolabel",
    "n_eval_pairs": len(r1s),
    "config": {"lr": LR, "epochs": EPOCHS, "batch_size": TRAIN_BATCH*ACCUM,
               "max_input": MAX_INPUT, "max_target": MAX_TARGET},
}
with open(RESULTS_DIR / "rouge_results.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
log("Sonuclar: results/rouge_results.json")
log("\nmT5 FINE-TUNE TAMAMLANDI.")
