"""
Özetleyici v4 — Domain Adaptation (Seviye 3)
Base: ozcangundes/mt5-small-turkish-summarization (WikiLingua TR)
Yontem:
  1. Haber ozetleme icin egitilmis modeli base al
  2. E-ticaret yorumlari uzerinde fine-tune et (domain adaptation)
  3. Hedef: mT5 v1'den (ROUGE-L 0.5387) daha iyi + gercek abstractive ozet

Karsilastirma bildiriye giriyor:
  v1: sifirdan pseudo-label  → ROUGE-1: 0.5717
  v4: domain adaptation     → beklenen: 0.60+
"""
import os, json, warnings, random, re, sys, tempfile, shutil, time
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_LOG_FILE = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "logs", "summarizer_v4_training.log"), "a", encoding="utf-8", buffering=1)

def log(msg):
    ts = time.strftime("[%H:%M:%S] ")
    line = ts + str(msg)
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("utf-8", "replace").decode("ascii", "replace"), flush=True)
    try:
        _LOG_FILE.write(line + "\n")
        _LOG_FILE.flush()
    except Exception:
        pass

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

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL   = "models/mt5_summarizer"  # v1 checkpoint — ozcangundes PyTorch nightly'de crash yapıyor
SAVE_DIR     = Path("models/mt5_summarizer_v4")
RESULTS_DIR  = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

BATCH_REVIEWS    = 15
N_SUMMARY_SENTS  = 4    # Daha kisa hedef = daha iyi ozet ogrenmesi
TRAIN_BATCH      = 4
ACCUM            = 4    # Eff. batch = 16
LR               = 5e-5  # Domain adaptation: daha buyuk LR (degisim isteniliyor)
EPOCHS           = 8
MAX_INPUT        = 256
MAX_TARGET       = 96   # Kisa, ozlu cikti istiyoruz
N_PAIRS_PER_CLS  = 400
SEED             = 42
LABEL_MAP        = {0: "olumsuz", 1: "notr", 2: "olumlu"}

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

log(f"Cihaz : {DEVICE}")
if torch.cuda.is_available():
    log(f"GPU   : {torch.cuda.get_device_name(0)}")
    log(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
log(f"Base  : {BASE_MODEL}")

# ── Veri yukle ────────────────────────────────────────────────────
data_dir = Path("data/processed_v4") if Path("data/processed_v4").exists() else Path("data/processed")
train_df = pd.read_csv(data_dir / "train.csv")
val_df   = pd.read_csv(data_dir / "val.csv")
test_df  = pd.read_csv(data_dir / "test.csv")
for df in [train_df, val_df, test_df]:
    df["sentiment"] = df["label"].map(LABEL_MAP)
log(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


# ── Extractive ozetleyici (pseudo-label uretici) ──────────────────
def split_sentences(text):
    sents = re.split(r'(?<=[.!?])\s+', str(text).strip())
    return [s.strip() for s in sents if len(s.strip()) > 15]

def tfidf_mmr_summary(reviews, n=N_SUMMARY_SENTS, lam=0.7):
    sents = []
    for r in reviews:
        sents.extend(split_sentences(r))
    if len(sents) <= n:
        return " ".join(sents)
    try:
        vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)
        mat = vec.fit_transform(sents)
    except Exception:
        return " ".join(sents[:n])
    query = np.asarray(mat.mean(axis=0))
    sims  = cosine_similarity(query, mat)[0]
    sel, rem = [], list(range(len(sents)))
    for _ in range(n):
        if not rem: break
        if not sel:
            best = max(rem, key=lambda i: sims[i])
        else:
            sel_mat = mat[sel]
            best = max(rem, key=lambda i: lam * sims[i] - (1 - lam) * cosine_similarity(mat[i], sel_mat).max())
        sel.append(best); rem.remove(best)
    sel.sort()
    return " ".join(sents[i] for i in sel)


# ── Pseudo-label cifti uret ───────────────────────────────────────
# Domain adaptation icin prefix: modelin orijinal formatını kullanıyoruz
# ozcangundes modeli "summarize: " prefix ile eğitildi
log("\nPseudo-label ciftleri uretiliyor...")

def make_pairs(df, n_pairs=N_PAIRS_PER_CLS):
    pairs = []
    for sentiment in ["olumlu", "olumsuz", "notr"]:
        reviews = df[df["sentiment"] == sentiment]["Review"].astype(str).tolist()
        random.shuffle(reviews)
        for i in range(0, min(n_pairs * BATCH_REVIEWS, len(reviews)), BATCH_REVIEWS):
            batch = reviews[i:i+BATCH_REVIEWS]
            if len(batch) < 5: continue
            summary = tfidf_mmr_summary(batch, n=N_SUMMARY_SENTS)
            if len(summary) < 25: continue
            # "summarize: " prefix — modelin orijinal formatı
            inp = "summarize: " + " ".join(batch)
            pairs.append({"input": inp, "target": summary, "sentiment": sentiment})
    random.shuffle(pairs)
    return pairs

train_pairs = make_pairs(train_df, n_pairs=N_PAIRS_PER_CLS)
val_pairs   = make_pairs(val_df,   n_pairs=80)
log(f"Train ciftleri: {len(train_pairs)} | Val ciftleri: {len(val_pairs)}")


# ── Model yukle ───────────────────────────────────────────────────
log(f"\nBase model yukleniyor: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
model     = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
model.to(DEVICE)
if torch.cuda.is_available():
    log(f"VRAM (yükleme sonrasi): {torch.cuda.memory_allocated()/1024**3:.2f} GB")


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
        labels = tgt["input_ids"].squeeze().clone()
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids":      inp["input_ids"].squeeze(),
            "attention_mask": inp["attention_mask"].squeeze(),
            "labels":         labels,
        }

train_dl = DataLoader(SumDataset(train_pairs), batch_size=TRAIN_BATCH, shuffle=True,  num_workers=0, pin_memory=False)
val_dl   = DataLoader(SumDataset(val_pairs),   batch_size=TRAIN_BATCH, shuffle=False, num_workers=0, pin_memory=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
USE_AMP   = torch.cuda.is_available()

log(f"\nFine-tune basliyor | LR={LR} | Eff.batch={TRAIN_BATCH*ACCUM} | Epochs={EPOCHS} | AMP={USE_AMP}")


# ── Val loss ──────────────────────────────────────────────────────
def val_loss():
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in val_dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                out = model(**batch)
            total += out.loss.item()
    return total / max(len(val_dl), 1)


# ── Eğitim ────────────────────────────────────────────────────────
best_val = float("inf")
best_ckpt_dir = Path(tempfile.mkdtemp(prefix="mt5v4_best_"))
log(f"Gecici checkpoint dizini: {best_ckpt_dir}")

try:
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
            if (step + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            if (step + 1) % 20 == 0:
                log(f"  E{epoch+1} | Step {step+1}/{len(train_dl)} | Loss: {total_loss/(step+1):.4f}")

        vl = val_loss()
        log(f"\nEpoch {epoch+1}/{EPOCHS} | Train: {total_loss/len(train_dl):.4f} | Val: {vl:.4f}")
        if vl < best_val:
            best_val = vl
            model.save_pretrained(str(best_ckpt_dir))
            tokenizer.save_pretrained(str(best_ckpt_dir))
            log(f"  >> En iyi checkpoint diske kaydedildi (val={vl:.4f})")
        torch.cuda.empty_cache()
except Exception as e:
    log(f"\nEGITIM HATASI: {type(e).__name__}: {e}")
    import traceback
    log(traceback.format_exc())
    sys.exit(1)

log(f"\nEn iyi model yukleniyor: val_loss={best_val:.4f}")
model = AutoModelForSeq2SeqLM.from_pretrained(str(best_ckpt_dir))
model.to(DEVICE)
torch.cuda.empty_cache()


# ── ROUGE ─────────────────────────────────────────────────────────
log("\n" + "="*60)
log("ROUGE DEGERLENDIRMESI (mT5 Domain Adaptation v4)")
log("="*60)

scorer_obj = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
r1s, r2s, rls = [], [], []
model.eval()

for sentiment in ["olumlu", "olumsuz", "notr"]:
    reviews = test_df[test_df["sentiment"] == sentiment]["Review"].astype(str).tolist()
    batches = [reviews[i:i+BATCH_REVIEWS] for i in range(0, min(150, len(reviews)), BATCH_REVIEWS)]
    for batch in batches:
        if len(batch) < 5: continue
        inp_text = "summarize: " + " ".join(batch)
        inputs   = tokenizer(inp_text, truncation=True, max_length=MAX_INPUT, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                inputs.input_ids, attention_mask=inputs.attention_mask,
                max_new_tokens=MAX_TARGET, num_beams=6,
                early_stopping=True, no_repeat_ngram_size=3, length_penalty=1.5,
            )
        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        ref  = tfidf_mmr_summary(batch, n=N_SUMMARY_SENTS)
        s    = scorer_obj.score(ref, pred)
        r1s.append(s["rouge1"].fmeasure)
        r2s.append(s["rouge2"].fmeasure)
        rls.append(s["rougeL"].fmeasure)

rouge_scores = {
    "rouge1": round(float(np.mean(r1s)), 4),
    "rouge2": round(float(np.mean(r2s)), 4),
    "rougeL": round(float(np.mean(rls)), 4),
}
log(f"ROUGE-1 : {rouge_scores['rouge1']:.4f}  (v1: 0.5717)")
log(f"ROUGE-2 : {rouge_scores['rouge2']:.4f}  (v1: 0.5194)")
log(f"ROUGE-L : {rouge_scores['rougeL']:.4f}  (v1: 0.5387)")


# ── Nitel ornekler ────────────────────────────────────────────────
log("\n" + "="*60)
log("NITEL ORNEKLER")
log("="*60)
summaries = {}
for sentiment in ["olumlu", "olumsuz", "notr"]:
    reviews  = test_df[test_df["sentiment"] == sentiment]["Review"].astype(str).tolist()[:BATCH_REVIEWS]
    inp_text = "summarize: " + " ".join(reviews)
    inputs   = tokenizer(inp_text, truncation=True, max_length=MAX_INPUT, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            inputs.input_ids, attention_mask=inputs.attention_mask,
            max_new_tokens=MAX_TARGET, num_beams=6,
            early_stopping=True, no_repeat_ngram_size=3, length_penalty=1.5,
        )
    summary = tokenizer.decode(out[0], skip_special_tokens=True)
    summaries[sentiment] = summary
    log(f"\n[{sentiment.upper()}]\n>> {summary}")


# ── Kaydet ────────────────────────────────────────────────────────
SAVE_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(SAVE_DIR))
tokenizer.save_pretrained(str(SAVE_DIR))
log(f"\nCheckpoint: {SAVE_DIR}")

result = {
    "rouge":    rouge_scores,
    "summaries": summaries,
    "method":   "mt5_small_domain_adaptation_v4",
    "base":     BASE_MODEL,
    "config":   {"lr": LR, "epochs": EPOCHS, "batch_size": TRAIN_BATCH*ACCUM,
                  "max_input": MAX_INPUT, "max_target": MAX_TARGET,
                  "n_pairs_per_class": N_PAIRS_PER_CLS},
    "comparison": {
        "v1_rouge1": 0.5717, "v1_rouge2": 0.5194, "v1_rougeL": 0.5387,
        "v4_rouge1": rouge_scores["rouge1"],
        "v4_rouge2": rouge_scores["rouge2"],
        "v4_rougeL": rouge_scores["rougeL"],
    }
}
with open(RESULTS_DIR / "rouge_results_v4.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
log("\nSonuclar: results/rouge_results_v4.json")
log("\nmT5 DOMAIN ADAPTATION v4 TAMAMLANDI.")
