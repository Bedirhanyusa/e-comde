"""mT5-small modelini indir ve tokenizer'ı doğrula"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging as hf_logging
import sentencepiece as spm

hf_logging.set_verbosity_info()
MODEL_NAME = "google/mt5-small"

print("mT5-small tokenizer indiriliyor...", flush=True)
try:
    # use_fast=False ile indir
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    print(f"Tokenizer OK: vocab_size={tokenizer.vocab_size}", flush=True)
except Exception as e:
    print(f"use_fast=False basarisiz: {e}", flush=True)
    print("Alternatif deneniyor: sentencepiece dogrudan...", flush=True)

    # Cache'deki spiece.model'i bul
    import glob
    files = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--google--mt5-small/**/*.model"), recursive=True)
    for f in files:
        size = Path(f).stat().st_size
        print(f"  {f}: {size} bytes", flush=True)
        try:
            sp = spm.SentencePieceProcessor()
            sp.Load(f)
            print(f"  -> SP OK: {sp.GetPieceSize()} pieces", flush=True)
        except Exception as se:
            print(f"  -> SP HATALI: {se} — siliniyor", flush=True)
            Path(f).unlink()

    print("Yeniden indiriliyor...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, force_download=True)
    print(f"Tokenizer OK (force_download): vocab_size={tokenizer.vocab_size}", flush=True)

print("Model ağırlıkları indiriliyor...", flush=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print("Model OK!", flush=True)

print("\nmT5-small hazır. run_summarizer_v3.py çalıştırılabilir.", flush=True)
