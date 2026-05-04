"""
BERTurk Fine-Tuning v3 — Iki Asamali Egitim + AMP
Asama 1: Tum buyuk veri seti (gürültülü, ~90K) ile 3 epoch — domain adaptation
Asama 2: Temiz dengeli v4 verisi (15K/sinif) ile 10 epoch — görev ince ayari
Hedef: %84-88 macro-F1
"""
import os, json, random, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def log(msg): print(msg, flush=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    get_cosine_schedule_with_warmup,
    logging as hf_logging,
)
hf_logging.set_verbosity_error()
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    matthews_corrcoef, cohen_kappa_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from pathlib import Path

# ── Konfigürasyon ─────────────────────────────────────────────────
MODEL_NAME  = "dbmdz/bert-base-turkish-cased"
NUM_LABELS  = 3
MAX_LEN     = 192   # 128→192: daha uzun yorumlar icin daha iyi baglam
SEED        = 42

# Asama 1 — buyuk gürültülü veri ile domain adaptation
P1_LR           = 2e-5
P1_EPOCHS       = 3    # 2→3: daha iyi domain adaptation
P1_BATCH        = 8
P1_ACCUM        = 4    # effective batch = 32
P1_LABEL_SMOOTH = 0.10 # 0.15→0.10: daha az smoothing, daha net ogrenim
P1_WARMUP_RATIO = 0.06

# Asama 2 — temiz v4 verisi ile gorev ince ayari
P2_LR           = 4e-6  # 8e-6→4e-6: daha hassas ince ayar
P2_EPOCHS       = 10    # 6→10: patience erken durdurur, daha genis arama
P2_BATCH        = 8
P2_ACCUM        = 4     # effective batch = 32
P2_LABEL_SMOOTH = 0.03  # 0.05→0.03: temiz veride daha az smoothing
P2_PATIENCE     = 5     # 4→5: erken durdurmada daha sabir
P2_WARMUP_RATIO = 0.08

CKPT_DIR    = Path("models/berturk_v3_finetuned")
RESULTS_DIR = Path("results")
LABEL_NAMES = ["olumsuz", "notr", "olumlu"]

# Gürültülü sifreleme kelimeleri
STRONG_POS = [
    "harika", "mükemmel", "muhteşem", "süper", "fevkalade",
    "çok güzel", "çok iyi", "çok memnun", "tavsiye ederim",
    "kesinlikle alın", "pişman değilim", "bayıldım", "şahane"
]
STRONG_NEG = [
    "berbat", "rezalet", "korkunç", "iğrenç", "çöp",
    "kesinlikle almayın", "para kaybı", "çok kötü", "en kötü",
    "aldatıldım", "dolandırıldım", "sahte", "bozuk geldi",
    "iade ettim", "çok memnun değilim", "hiç beğenmedim"
]

def smart_label(text: str, rating: int) -> int:
    if rating <= 2: return 0
    if rating >= 4: return 2
    t = text.lower()
    p = sum(1 for w in STRONG_POS if w in t)
    n = sum(1 for w in STRONG_NEG if w in t)
    if p > n and p >= 1: return 2
    if n > p and n >= 1: return 0
    return 1

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Cihaz: {DEVICE} | Model: {MODEL_NAME}")
if torch.cuda.is_available():
    log(f"GPU  : {torch.cuda.get_device_name(0)}")
    log(f"VRAM : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

USE_AMP = torch.cuda.is_available()
scaler  = torch.amp.GradScaler("cuda") if USE_AMP else None

# ── Veri yükleme yardımcısı ───────────────────────────────────────
def load_raw_sources(cap_per_class: int = None) -> pd.DataFrame:
    dfs = []

    def add(df, src):
        df = df[["Review", "label"]].dropna()
        df["Review"] = df["Review"].astype(str).str.strip()
        df = df[df["Review"].str.len().between(10, 1000)]
        df = df.drop_duplicates(subset=["Review"])
        log(f"  {src}: {len(df):,} | dist={dict(df.label.value_counts().sort_index())}")
        dfs.append(df)

    # Trendyol
    for tpath in [
        Path("data/raw/trendyol/trendyol_reviews_20251212_000835.csv"),  # gercek konum
        Path("data/raw/trendyol_582k.csv"),
        Path("data/raw/Trendyol-22k-comment-dataset-v11_processed.csv"),
    ]:
        if tpath.exists():
            try:
                df = pd.read_csv(tpath, encoding="utf-8", on_bad_lines="skip")
                col_map = {}
                for c in df.columns:
                    cl = c.lower()
                    if any(k in cl for k in ("comment","review","yorum","text")): col_map[c] = "Review"
                    elif any(k in cl for k in ("point","puan","rating","star","yildiz")): col_map[c] = "rating"
                df = df.rename(columns=col_map)
                if "Review" in df.columns and "rating" in df.columns:
                    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
                    df = df.dropna(subset=["rating"])
                    df["rating"] = df["rating"].astype(int).clip(1, 5)
                    df["label"]  = df.apply(lambda r: smart_label(str(r["Review"]), r["rating"]), axis=1)
                    add(df, tpath.name)
            except Exception as e:
                log(f"  Trendyol hata: {e}")
            break

    # Turkish e-commerce
    for fname in ["turkish_ecommerce_reviews.csv", "turkish_product_reviews.csv"]:
        fpath = Path(f"data/raw/{fname}")
        if fpath.exists():
            try:
                df = pd.read_csv(fpath, encoding="latin-1")
                col_map = {}
                for c in df.columns:
                    cl = c.lower()
                    if any(k in cl for k in ("review","yorum","text")): col_map[c] = "Review"
                    elif any(k in cl for k in ("rating","puan","star")):  col_map[c] = "rating"
                    elif any(k in cl for k in ("label","sentiment")):     col_map[c] = "label_raw"
                df = df.rename(columns=col_map)
                if "Review" in df.columns:
                    if "rating" in df.columns:
                        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
                        df = df.dropna(subset=["rating"])
                        df["rating"] = df["rating"].astype(int).clip(1, 5)
                        df["label"] = df.apply(lambda r: smart_label(str(r["Review"]), r["rating"]), axis=1)
                    elif "label_raw" in df.columns:
                        m = {"negative":0,"notr":1,"neutral":1,"olumsuz":0,"nötr":1,"olumlu":2,"positive":2}
                        df["label"] = df["label_raw"].astype(str).str.lower().map(m)
                        df = df.dropna(subset=["label"])
                        df["label"] = df["label"].astype(int)
                    else:
                        continue
                    add(df, fname)
            except Exception as e:
                log(f"  {fname} hata: {e}")

    # Hepsiburada ABSA
    absa_path = Path("data/raw/absa/reviews.csv")
    if absa_path.exists():
        try:
            df = pd.read_csv(absa_path)
            df = df[["review_text", "rating"]].rename(columns={"review_text": "Review"}).dropna()
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
            df = df.dropna(subset=["rating"])
            df["rating"] = df["rating"].astype(int).clip(1, 5)
            df["label"]  = df.apply(lambda r: smart_label(str(r["Review"]), r["rating"]), axis=1)
            add(df, "hepsiburada_absa")
        except Exception as e:
            log(f"  Hepsiburada hata: {e}")

    if not dfs:
        raise RuntimeError("Hic kaynak yuklenemedi!")

    all_df = pd.concat(dfs, ignore_index=True)
    all_df["Review"] = all_df["Review"].astype(str).str.strip()
    all_df = all_df[all_df["Review"].str.len().between(10, 1000)]
    all_df = all_df.drop_duplicates(subset=["Review"]).dropna()

    if cap_per_class:
        avail = all_df.groupby("label").size().to_dict()
        per = min(cap_per_class, min(avail.values()))
        all_df = pd.concat([
            all_df[all_df["label"] == lbl].sample(per, random_state=SEED)
            for lbl in [0, 1, 2]
        ], ignore_index=True).sample(frac=1, random_state=SEED)
        log(f"Dengelendi: {len(all_df):,} ({per:,}/sinif)")

    return all_df


# ── Dataset ───────────────────────────────────────────────────────
log(f"Tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class ReviewDataset(Dataset):
    """Tum metinleri onceden tokenize eder — egitim sirasinda tokenizer cagrisi yok."""
    def __init__(self, df, max_len=128):
        texts  = df["Review"].astype(str).tolist()
        labels = df["label"].tolist()
        self.labels = torch.tensor(labels, dtype=torch.long)
        # Batch halinde tokenize et (256'lik parcalar)
        ids_list, mask_list, tti_list = [], [], []
        chunk = 256
        for i in range(0, len(texts), chunk):
            enc = tokenizer(
                texts[i:i+chunk], truncation=True, padding="max_length",
                max_length=max_len, return_tensors="pt",
            )
            ids_list.append(enc["input_ids"])
            mask_list.append(enc["attention_mask"])
            if "token_type_ids" in enc:
                tti_list.append(enc["token_type_ids"])
        self.input_ids      = torch.cat(ids_list)
        self.attention_mask = torch.cat(mask_list)
        self.token_type_ids = torch.cat(tti_list) if tti_list else None

    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        d = {"input_ids": self.input_ids[i], "attention_mask": self.attention_mask[i]}
        if self.token_type_ids is not None:
            d["token_type_ids"] = self.token_type_ids[i]
        return d, self.labels[i]


def make_loaders(df, batch_size, max_len=128, val_split=0.10):
    from sklearn.model_selection import train_test_split
    tr, va = train_test_split(df, test_size=val_split, stratify=df["label"], random_state=SEED)
    log(f"  Tokenize ediliyor: train={len(tr):,} val={len(va):,} (max_len={max_len})...")
    tr_dl = DataLoader(ReviewDataset(tr, max_len), batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    va_dl = DataLoader(ReviewDataset(va, max_len), batch_size=16,          shuffle=False, num_workers=0, pin_memory=False)
    return tr_dl, va_dl, tr, va


# ── Değerlendirme ─────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    preds, labels_all, probas_all = [], [], []
    with torch.no_grad():
        for batch, labels in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                out = model(**batch)
            prob = torch.softmax(out.logits.float(), dim=-1).cpu().tolist()
            pred = out.logits.argmax(-1).cpu().tolist()
            preds.extend(pred); labels_all.extend(labels.tolist()); probas_all.extend(prob)
    acc = accuracy_score(labels_all, preds)
    f1  = f1_score(labels_all, preds, average="macro")
    return acc, f1, preds, labels_all, probas_all


# ── Egitim döngüsü (tek faz) ──────────────────────────────────────
def train_phase(model, train_dl, val_dl, epochs, lr, accum, label_smooth,
                patience, warmup_ratio, class_weights, phase_name):
    criterion   = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE),
                                       label_smoothing=label_smooth)
    optimizer   = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_dl) // accum) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler   = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    log(f"\n{'='*60}")
    log(f"ASAMA: {phase_name}")
    log(f"{'='*60}")
    log(f"LR={lr} | Eff.batch={P1_BATCH*accum} | Smooth={label_smooth} | AMP={USE_AMP}")
    log(f"Adimlar: {total_steps} | Warmup: {warmup_steps}")

    best_f1    = 0.0
    best_state = None
    pat_cnt    = 0
    history    = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, (batch, labels) in enumerate(train_dl):
            batch  = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = labels.to(DEVICE)

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(**batch).logits
                loss   = criterion(logits, labels) / accum

            if USE_AMP:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * accum

            if (step + 1) % accum == 0:
                if USE_AMP:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % 300 == 0:
                log(f"  E{epoch+1} | Step {step+1}/{len(train_dl)} | Loss: {total_loss/(step+1):.4f}")

        avg_loss = total_loss / len(train_dl)
        val_acc, val_f1, _, _, _ = evaluate(model, val_dl)
        history.append({"epoch": epoch+1, "loss": round(avg_loss,4),
                         "val_acc": round(val_acc,4), "val_f1": round(val_f1,4)})
        log(f"\nEpoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_cnt    = 0
            log(f"  >> En iyi model guncellendi (F1={val_f1:.4f})")
        else:
            pat_cnt += 1
            log(f"  -- patience {pat_cnt}/{patience}")
            if patience > 0 and pat_cnt >= patience:
                log(f"  >> Erken durdurma! (epoch {epoch+1})")
                break
        log("")

    if best_state:
        model.load_state_dict(best_state)
        del best_state
        torch.cuda.empty_cache()

    return model, best_f1, history


# ── ANA AKIŞ ─────────────────────────────────────────────────────

# ASAMA 1: Büyük gürültülü veri (30K/sinif)
log("\nASAMA 1 VERiSi YUKLENIYOR...")
log("Tum kaynaklardan veri okunuyor (cap=30K/sinif)...")
phase1_df = load_raw_sources(cap_per_class=30_000)
log(f"Asama 1 toplam: {len(phase1_df):,}")
p1_train_dl, p1_val_dl, _, _ = make_loaders(phase1_df, P1_BATCH, max_len=128)  # P1: 128 token (VRAM tasarrufu)
log(f"train batches: {len(p1_train_dl)} | val batches: {len(p1_val_dl)}")

# Sinif agirliklari (asama 1)
p1_counts = phase1_df["label"].value_counts().sort_index().values
p1_weights = torch.tensor(p1_counts.sum() / (NUM_LABELS * p1_counts), dtype=torch.float32)

# Model yükle
log(f"\nModel yukleniyor: {MODEL_NAME}")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model.to(DEVICE)
if torch.cuda.is_available():
    log(f"VRAM kullanim: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Asama 1 egitimi
model, p1_best_f1, p1_history = train_phase(
    model, p1_train_dl, p1_val_dl,
    epochs=P1_EPOCHS, lr=P1_LR, accum=P1_ACCUM,
    label_smooth=P1_LABEL_SMOOTH, patience=0,
    warmup_ratio=P1_WARMUP_RATIO, class_weights=p1_weights,
    phase_name="ASAMA 1 — Buyuk gürültülü veri (domain adaptation)"
)
log(f"\nAsama 1 tamamlandi. En iyi Val F1: {p1_best_f1:.4f}")

# Asama 1 sonrasi bellek temizle
del phase1_df, p1_train_dl, p1_val_dl
torch.cuda.empty_cache()

# ASAMA 2: Temiz dengeli v4 verisi
v4_dir = Path("data/processed_v4")
if not v4_dir.exists():
    log("\ndata/processed_v4 bulunamadi — prepare_data_v4.py calistiriliyor...")
    import subprocess, sys
    r = subprocess.run([sys.executable, "prepare_data_v4.py"], capture_output=True, text=True)
    log(r.stdout[-2000:] if r.stdout else "(no output)")
    if r.returncode != 0:
        log(f"prepare_data_v4 HATA: {r.stderr[-500:]}")

log("\nASAMA 2 VERiSi YUKLENIYOR...")
train2 = pd.read_csv(v4_dir / "train.csv")
val2   = pd.read_csv(v4_dir / "val.csv")
test2  = pd.read_csv(v4_dir / "test.csv")
log(f"v4 train={len(train2):,} val={len(val2):,} test={len(test2):,}")

p2_train_dl = DataLoader(ReviewDataset(train2, MAX_LEN), batch_size=P2_BATCH, shuffle=True,  num_workers=0, pin_memory=False)
p2_val_dl   = DataLoader(ReviewDataset(val2,   MAX_LEN), batch_size=16,        shuffle=False, num_workers=0, pin_memory=False)
p2_test_dl  = DataLoader(ReviewDataset(test2,  MAX_LEN), batch_size=16,        shuffle=False, num_workers=0, pin_memory=False)

# Sinif agirliklari (asama 2 — dengeli oldugundan ~1.0)
p2_counts  = train2["label"].value_counts().sort_index().values
p2_weights = torch.tensor(p2_counts.sum() / (NUM_LABELS * p2_counts), dtype=torch.float32)

model, p2_best_f1, p2_history = train_phase(
    model, p2_train_dl, p2_val_dl,
    epochs=P2_EPOCHS, lr=P2_LR, accum=P2_ACCUM,
    label_smooth=P2_LABEL_SMOOTH, patience=P2_PATIENCE,
    warmup_ratio=P2_WARMUP_RATIO, class_weights=p2_weights,
    phase_name="ASAMA 2 — Temiz dengeli v4 veri (gorev ince ayari)"
)
log(f"\nAsama 2 tamamlandi. En iyi Val F1: {p2_best_f1:.4f}")

# ── Test değerlendirme ────────────────────────────────────────────
log("\n" + "="*60)
log("TEST DEGERLENDIRME (v4 test seti)")
log("="*60)

test_acc, test_f1, test_preds, test_labels, test_probas = evaluate(model, p2_test_dl)
log(f"Test Accuracy : {test_acc:.4f}")
log(f"Test Macro-F1 : {test_f1:.4f}")
log("")
log(classification_report(test_labels, test_preds, target_names=LABEL_NAMES, digits=4))

mcc   = matthews_corrcoef(test_labels, test_preds)
kappa = cohen_kappa_score(test_labels, test_preds)
y_bin = label_binarize(test_labels, classes=[0, 1, 2])
auc   = roc_auc_score(y_bin, np.array(test_probas), average="macro", multi_class="ovr")
log(f"MCC        : {mcc:.4f}")
log(f"Cohen Kappa: {kappa:.4f}")
log(f"ROC-AUC    : {auc:.4f}")

# Per-class F1
from sklearn.metrics import f1_score as f1s
per_class_f1 = f1s(test_labels, test_preds, average=None, labels=[0,1,2])
per_class = {LABEL_NAMES[i]: round(float(per_class_f1[i]), 4) for i in range(3)}
log(f"Per-class F1: {per_class}")

# ── Kaydet ───────────────────────────────────────────────────────
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
model.save_pretrained(str(CKPT_DIR))
tokenizer.save_pretrained(str(CKPT_DIR))
log(f"\nCheckpoint kaydedildi: {CKPT_DIR}")

results = {
    "berturk_v3": {
        "accuracy":      round(test_acc, 4),
        "macro_f1":      round(test_f1, 4),
        "mcc":           round(mcc, 4),
        "kappa":         round(kappa, 4),
        "roc_auc_macro": round(auc, 4),
        "per_class":     per_class,
        "best_val_f1":   round(p2_best_f1, 4),
        "phase1_history": p1_history,
        "phase2_history": p2_history,
        "config": {
            "p1_lr": P1_LR, "p1_epochs": P1_EPOCHS,
            "p2_lr": P2_LR, "p2_epochs": P2_EPOCHS,
            "effective_batch": P2_BATCH * P2_ACCUM,
            "p1_label_smooth": P1_LABEL_SMOOTH,
            "p2_label_smooth": P2_LABEL_SMOOTH,
            "data_version": "v4",
        },
    }
}
with open(RESULTS_DIR / "berturk_v3_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
log("Sonuclar kaydedildi: results/berturk_v3_results.json")
log("\nEGITIM TAMAMLANDI (v3).")
