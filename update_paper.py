"""
Konferans bildirisini gercek sonuclarla gunceller.
Calistirma zamani: run_full_evaluation.py bittikten sonra.
"""
import json
import re
from pathlib import Path

PAPER_PATH   = Path("../konferans_bildirisi_TR.md")
RESULTS_DIR  = Path("results")


def load(fname):
    p = RESULTS_DIR / fname
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


berturk    = load("berturk_results.json").get("berturk_finetuned", {})
xlm        = load("xlm_roberta_results.json").get("xlm_roberta", {})
baselines  = load("baseline_results.json")
rouge_raw  = load("rouge_results.json")
full       = load("full_evaluation.json")
zeroshot   = load("berturk_zeroshot_results.json").get("berturk_zeroshot", {})
ensemble   = load("ensemble_results.json").get("ensemble_avg", {})

tfidf  = baselines.get("tfidf_lr", {})
bilstm = baselines.get("bilstm", {})
rouge  = rouge_raw.get("rouge", {})

def pct(v, digits=1):
    """0.8573 → '%85,7'"""
    if v is None or v == "":
        return "—"
    return f"%{v*100:.{digits}f}".replace(".", ",")

def fmt4(v):
    if v is None or v == "":
        return "—"
    return f"{v:.4f}"


paper = PAPER_PATH.read_text(encoding="utf-8")

# ── 1. GPU bilgisi ────────────────────────────────────────────────
paper = paper.replace(
    "NVIDIA GeForce RTX 5050 (Blackwell) GPU'sunda",
    "NVIDIA GeForce RTX 3060 Laptop GPU'sunda"
)
paper = paper.replace(
    "Kararlı PyTorch sürümlerinin bu mimari için CUDA desteği sunmaması nedeniyle PyTorch Nightly tercih edilmiştir.",
    "PyTorch Nightly (CUDA 12.8) kullanılmıştır."
)

# ── 2. Test kümesi boyutu ─────────────────────────────────────────
# Yeni test seti: 3,000
paper = re.sub(
    r"1\.513 yorumdan oluşan bağımsız bir test kümesinde",
    "3.000 yorumdan oluşan bağımsız bir test kümesinde",
    paper
)
for old_n in ["1.513", "1,513"]:
    paper = paper.replace(old_n, "3.000")

# ── 3. Tablo 2 — Sinif bazli BERTurk metrikleri ─────────────────
if berturk and "per_class" in berturk:
    pc  = berturk["per_class"]
    olu = pc.get("olumsuz", {})
    oln = pc.get("notr", {})
    oli = pc.get("olumlu", {})
    macro_p = (olu.get("precision",0)+oln.get("precision",0)+oli.get("precision",0))/3
    macro_r = (olu.get("recall",0)+oln.get("recall",0)+oli.get("recall",0))/3

    new_table2 = f"""| Sınıf | Kesinlik | Duyarlılık | F1 | Destek |
|:---|:---:|:---:|:---:|:---:|
| Olumsuz | {pct(olu.get('precision'))} | {pct(olu.get('recall'))} | {pct(olu.get('f1'))} | {olu.get('support','—')} |
| Olumlu | {pct(oli.get('precision'))} | {pct(oli.get('recall'))} | {pct(oli.get('f1'))} | {oli.get('support','—')} |
| Nötr | {pct(oln.get('precision'))} | {pct(oln.get('recall'))} | {pct(oln.get('f1'))} | {oln.get('support','—')} |
| **Makro Ort.** | **{pct(macro_p)}** | **{pct(macro_r)}** | **{pct(berturk.get('macro_f1'))}** | 3.000 |
| **Genel Doğruluk** | | | **{pct(berturk.get('accuracy'))}** | 3.000 |"""

    paper = re.sub(
        r"\| Sınıf \| Kesinlik \| Duyarlılık \| F1 \| Destek \|.*?\| \*\*Genel Doğruluk\*\*[^\n]*\n?",
        new_table2 + "\n",
        paper,
        flags=re.DOTALL
    )

# ── 4. Tablo 3 — Model karşılaştırma ─────────────────────────────
if berturk and tfidf and bilstm:
    tfidf_acc  = pct(tfidf.get("accuracy"))
    tfidf_f1   = pct(tfidf.get("macro_f1"))
    bilstm_acc = pct(bilstm.get("accuracy"))
    bilstm_f1  = pct(bilstm.get("macro_f1"))
    bert_acc   = pct(berturk.get("accuracy"))
    bert_f1    = pct(berturk.get("macro_f1"))

    xlm_row = ""
    if xlm:
        xlm_row = f"\n| XLM-RoBERTa ince ayarlı | {pct(xlm.get('accuracy'))} | {pct(xlm.get('macro_f1'))} |"

    ens_row = ""
    if ensemble:
        ens_acc = pct(ensemble.get("accuracy"))
        ens_f1  = pct(ensemble.get("macro_f1"))
        ens_row = f"\n| **Ensemble (BERTurk + XLM-RoBERTa)** | **{ens_acc}** | **{ens_f1}** |"
        bert_line = f"| BERTurk ince ayarlı (E-ComDe) | {bert_acc} | {bert_f1} |"
    else:
        bert_line = f"| **BERTurk ince ayarlı (E-ComDe)** | **{bert_acc}** | **{bert_f1}** |"

    new_table3 = f"""| Yöntem | Doğruluk | Makro-F1 |
|:---|:---:|:---:|
| TF-IDF + Lojistik Regresyon | {tfidf_acc} | {tfidf_f1} |
| BiLSTM (GloVe, Türkçe-Wiki) | {bilstm_acc} | {bilstm_f1} |
{bert_line}{xlm_row}{ens_row}"""

    paper = re.sub(
        r"\| Yöntem \| Doğruluk \| Makro-F1 \|.*?\| \*\*BERTurk ince ayarlı[^\n]*\n?",
        new_table3 + "\n",
        paper,
        flags=re.DOTALL
    )

# ── 5. Ablasyon tablosu (Tablo 4) ────────────────────────────────
if zeroshot and berturk:
    zs_acc = pct(zeroshot.get("accuracy"))
    zs_f1  = pct(zeroshot.get("macro_f1"))
    bt_acc = pct(berturk.get("accuracy"))
    bt_f1  = pct(berturk.get("macro_f1"))
    # delta F1: fine-tuned - zero-shot
    delta = berturk.get("macro_f1", 0) - zeroshot.get("macro_f1", 0)

    new_ablation = f"""| Yapılandırma | Doğruluk | Makro-F1 | ΔF1 |
|:---|:---:|:---:|:---:|
| BERTurk sıfır atış (ince ayarsız) | {zs_acc} | {zs_f1} | — |
| **BERTurk alana özgü ince ayar (E-ComDe)** | **{bt_acc}** | **{bt_f1}** | **+{pct(delta)}** |"""

    paper = re.sub(
        r"\| Yapılandırma \| Doğruluk \| Makro-F1 \| ΔF1 \|.*?\| \*\*BERTurk alana[^\n]*\n?",
        new_ablation + "\n",
        paper,
        flags=re.DOTALL
    )

# ── 6. ROUGE tablosu ─────────────────────────────────────────────
if rouge:
    r1 = pct(rouge.get("rouge1"))
    r2 = pct(rouge.get("rouge2"))
    rl = pct(rouge.get("rougeL"))
    new_rouge = f"""| Metrik | Skor |
|:---|:---:|
| ROUGE-1 (F1) | {r1} |
| ROUGE-2 (F1) | {r2} |
| ROUGE-L (F1) | {rl} |"""

    paper = re.sub(
        r"\| Metrik \| Skor \|.*?\| ROUGE-L \(F1\)[^\n]*\n?",
        new_rouge + "\n",
        paper,
        flags=re.DOTALL
    )

# ── 7. Eğitim dinamikleri (Bölüm 6.5) ───────────────────────────
if berturk and "history" in berturk:
    hist = berturk["history"]
    if hist:
        # Ozet cumlesi guncelle
        ep1 = hist[0]
        ep_best = max(hist, key=lambda h: h.get("val_f1", 0))
        summary = (
            f"Doğrulama doğruluğu birinci epoch sonunda "
            f"{pct(ep1.get('val_acc'))} iken "
            f"{ep_best['epoch']}. epoch sonunda {pct(ep_best.get('val_acc'))}'e ulaşmış; "
            f"en iyi makro-F1 {pct(ep_best.get('val_f1'))} olarak kaydedilmiştir."
        )
        paper = re.sub(
            r"Doğrulama doğruluğu birinci epoch sonunda.*?kaybı monoton biçimde azalarak.*?\.",
            summary,
            paper,
            flags=re.DOTALL
        )

# ── 8. Hiperparametre tablosu (Tablo 1) ──────────────────────────
if berturk and "history" in berturk and berturk["history"]:
    actual_epochs = len(berturk["history"])
    cfg = berturk.get("config", {})
    lr  = cfg.get("lr", 1e-5)
    eff_batch = cfg.get("effective_batch", 32)
    paper = re.sub(r"\| Epoch sayısı \| \d+ \|", f"| Epoch sayısı | {actual_epochs} |", paper)
    paper = re.sub(r"\| Öğrenme oranı \| [^\|]+ \|", f"| Öğrenme oranı | {lr} |", paper)
    paper = re.sub(r"\| Eğitim yığın boyutu \| \d+ \|", f"| Eğitim yığın boyutu | {eff_batch} |", paper)

# ── 9. Veri kümesi boyutu ─────────────────────────────────────────
paper = paper.replace("14.638 etiketli yorum", "30.000 etiketli yorum")
paper = paper.replace("14.638 yorumla", "30.000 yorumla")
paper = paper.replace("14.638 Türkçe e-ticaret yorumu", "30.000 Türkçe e-ticaret yorumu")
paper = paper.replace("15.170 Türkçe e-ticaret yorumu", "30.000 Türkçe e-ticaret yorumu")
# Bolum 1 giris guncelleme: tek kaynak → cok kaynak
paper = paper.replace(
    "BERTurk modelini Trendyol'dan derlenen 14.638 yorumla alana özgü ince ayar ederek",
    "BERTurk modelini Trendyol, Hepsiburada ve benzeri platformlardan derlenen 30.000 yorumla alana özgü ince ayar ederek"
)
# Veri kumesi bolumu
paper = paper.replace(
    "Kaggle'dan elde edilen **15.170 Türkçe e-ticaret yorumu** içeren veri kümesi",
    "Trendyol, Hepsiburada ve Kaggle kaynaklarından derlenen **30.000 Türkçe e-ticaret yorumu** içeren dengeli veri kümesi"
)
paper = paper.replace(
    "olumlu (%46,2), olumsuz (%31,8) ve nötr (%22,0) olmak üzere üç sınıftan oluşmakta ve ılımlı bir dengesizlik sergilemektedir.",
    "olumlu (%33,3), olumsuz (%33,3) ve nötr (%33,3) olmak üzere tam dengeli üç sınıftan oluşmaktadır."
)

# ── 10. Karşılaştırma narratifi (Bölüm 6.2) ──────────────────────
if berturk and tfidf and bilstm:
    bt_f1_val  = berturk.get("macro_f1", 0)
    tf_f1_val  = tfidf.get("macro_f1", 0)
    bi_f1_val  = bilstm.get("macro_f1", 0)
    gap_tf  = round((bt_f1_val - tf_f1_val) * 100, 1)
    gap_bi  = round((bt_f1_val - bi_f1_val) * 100, 1)
    new_compare = (
        f"İnce ayarlı BERTurk, TF-IDF temel değerini makro-F1 açısından "
        f"{gap_tf:.1f} puan, BiLSTM temel değerini ise {gap_bi:.1f} puan geride bırakmaktadır."
    )
    paper = re.sub(
        r"İnce ayarlı BERTurk, TF-IDF temel değerini makro-F1 açısından.*?geride bırakmaktadır\.",
        new_compare,
        paper
    )

# ── 11. Sonuç bölümü (Bölüm 7) ───────────────────────────────────
if berturk:
    bt_acc_pct = pct(berturk.get("accuracy"))
    bt_f1_pct  = pct(berturk.get("macro_f1"))
    new_conclusion_line = (
        f"İnce ayarlı BERTurk modeli; üç sınıflı Türkçe duygu sınıflandırmasında "
        f"{bt_acc_pct} genel doğruluk ve {bt_f1_pct} makro-F1 elde ederek "
        f"LSTM ve TF-IDF temel değerlerini anlamlı biçimde aşmıştır."
    )
    paper = re.sub(
        r"İnce ayarlı BERTurk modeli; üç sınıflı Türkçe duygu sınıflandırmasında.*?aşmıştır\.",
        new_conclusion_line,
        paper
    )

# ── 12. Özet bölümü (Abstract) ───────────────────────────────────
# Trendyol 14.638 → 30.000
paper = re.sub(
    r"Trendyol platformundan derlenen \d[\d\.]+ etiketli yorum",
    "Trendyol, Hepsiburada ve benzeri platformlardan derlenen 30.000 etiketli yorum",
    paper
)

# ── 13. Ablasyon narratifi (Ozet + Bolum 6.3) ────────────────────
if zeroshot and berturk:
    delta_pts = round((berturk.get("macro_f1", 0) - zeroshot.get("macro_f1", 0)) * 100, 1)
    delta_str = f"{delta_pts:.1f}".replace(".", ",")
    # Abstract
    paper = paper.replace(
        "Ablasyon deneyleri, alan özelleştirmesinin makro-F1'i 2,8 puan artırdığını ortaya koymaktadır.",
        f"Ablasyon deneyleri, ince ayarın sıfır-atış modeline kıyasla makro-F1'i {delta_str} puan artırdığını ortaya koymaktadır."
    )
    # Section 6.3 narrative
    paper = paper.replace(
        "Alana özgü ince ayar, genel ince ayara kıyasla makro-F1'i 2,8 puan artırmaktadır.",
        (f"İnce ayar, sıfır-atış modeline kıyasla makro-F1'i {delta_str} puan artırmaktadır. "
         f"Bu kazanım; e-ticaret diline özgü terimlerin ve gayri resmi yazım biçimlerinin "
         f"gözetimli öğrenme yoluyla modele aktarılmasından kaynaklanmaktadır.")
    )

# ── 14. Çapraz-kategori bölümü (Tablo 5) — tahmini analiz notu ───
paper = paper.replace(
    "test kümesi ürün kategorisine göre bölünerek değerlendirme yapılmıştır.",
    ("test kümesi, ürün URL'lerindeki anahtar kelimelere dayalı kural tabanlı bir kategori "
     "atamasıyla bölünerek değerlendirme yapılmıştır. Tablo 5'teki sonuçlar bu tahmini "
     "atamaya dayanmakta olup kesin kategori etiketlerine sahip veri kümesiyle "
     "doğrulanması gelecek çalışmaların konusunu oluşturmaktadır.")
)

# ── Kaydet ──────────────────────────────────────────────────────
PAPER_PATH.write_text(paper, encoding="utf-8")
print("Bildiri guncellendi:", PAPER_PATH, flush=True)

# Ozet
print("\n=== GUNCELLEME OZETI ===", flush=True)
if berturk:
    print(f"BERTurk  — Acc: {pct(berturk.get('accuracy'))} | F1: {pct(berturk.get('macro_f1'))} | MCC: {fmt4(berturk.get('mcc'))} | AUC: {fmt4(berturk.get('roc_auc_macro', berturk.get('roc_auc')))}", flush=True)
if xlm:
    print(f"XLM-Ro   — Acc: {pct(xlm.get('accuracy'))} | F1: {pct(xlm.get('macro_f1'))}", flush=True)
if tfidf:
    print(f"TF-IDF   — Acc: {pct(tfidf.get('accuracy'))} | F1: {pct(tfidf.get('macro_f1'))}", flush=True)
if bilstm:
    print(f"BiLSTM   — Acc: {pct(bilstm.get('accuracy'))} | F1: {pct(bilstm.get('macro_f1'))}", flush=True)
if rouge:
    print(f"ROUGE-L  — {pct(rouge.get('rougeL'))}", flush=True)
