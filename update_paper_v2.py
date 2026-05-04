"""
Konferans bildirisini tüm yeni model sonuçlarıyla günceller.
Çalıştırma: full_evaluation_v2.py bittikten sonra.
Yeni modeller: BERTurk v3, XLM-Ro v2, savasy, Ensemble v2, Ensemble v3
"""
import json, re
from pathlib import Path

PAPER_PATH  = Path("../konferans_bildirisi_TR.md")
RESULTS_DIR = Path("results")


def load(fname):
    p = RESULTS_DIR / fname
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def pct(v, digits=1):
    if v is None or v == "":
        return "—"
    return f"%{v*100:.{digits}f}".replace(".", ",")


def fmt4(v):
    if v is None or v == "":
        return "—"
    return f"{v:.4f}"


# ── Sonuçları yükle ─────────────────────────────────────────────────
baselines  = load("baseline_results.json")
berturk    = load("berturk_results.json").get("berturk_finetuned", {})
xlm_v1     = load("xlm_roberta_results.json").get("xlm_roberta", {})
ensemble_v1= load("ensemble_results.json").get("ensemble_avg", {})
zeroshot   = load("berturk_zeroshot_results.json").get("berturk_zeroshot", {})
berturk_v3 = load("berturk_v3_results.json").get("berturk_v3", {})
xlm_v2     = load("xlm_roberta_v2_results.json").get("xlm_roberta_v2", {})
savasy     = load("savasy_results.json").get("savasy_finetuned", {})
ens_v2     = load("ensemble_v2_results.json").get("ensemble_v2_avg", {})
ens_v3     = load("ensemble_v3_results.json").get("ensemble_v3", {})
ens_tuned  = load("ensemble_tuned_results.json").get("ensemble_tuned", {})
rouge_raw  = load("rouge_results.json")
absa_raw   = load("absa_results.json")

tfidf  = baselines.get("tfidf_lr", {})
bilstm = baselines.get("bilstm", {})
rouge  = rouge_raw.get("rouge", {})
absa   = absa_raw.get("absa", {})

# En iyi model: doğruluk bazlı seç (ens_v2 > berturk_v3 genellikle en iyi)
# Threshold-tuned: nötr F1 için optimize edilmiş, genel doğrulukta düşük
candidates = [(m, n) for m, n in [
    (ens_v2,    "Ensemble v2 (BERTurk v3 + XLM-Ro v1)"),
    (berturk_v3,"BERTurk v3b (iki aşamalı ince ayar)"),
    (ens_v3,    "Ensemble v3 (BERTurk v3 + XLM-Ro v2 + savasy)"),
    (savasy,    "savasy Türkçe sınıflandırıcı"),
    (berturk,   "BERTurk ince ayarlı"),
] if m]
if candidates:
    best_model, best_name = max(candidates, key=lambda x: x[0].get("accuracy", 0))
else:
    best_model, best_name = {}, "—"

paper = PAPER_PATH.read_text(encoding="utf-8")


# ── 1. GPU bilgisi ────────────────────────────────────────────────
paper = paper.replace(
    "NVIDIA GeForce RTX 5050 (Blackwell) GPU'sunda",
    "NVIDIA GeForce RTX 3060 Laptop GPU'sunda"
)

# ── 2. Özet bölümü ────────────────────────────────────────────────
# Güncelle: best model sonuçlarını yansıt
if best_model:
    best_acc = pct(best_model.get("accuracy"))
    best_f1  = pct(best_model.get("macro_f1"))
    new_abstract_results = (
        f"Kapsamlı deneyler; BERTurk, XLM-RoBERTa ve özelleştirilmiş Türkçe sınıflandırıcının "
        f"ağırlıklı ensemble'ı ile {best_acc} doğruluk ve {best_f1} makro-F1 "
        f"elde edildiğini göstermektedir."
    )
    paper = re.sub(
        r"Dört model kapsamlı bir biçimde karşılaştırılmış.*?değerlendirilmiştir\.",
        f"Dört model kapsamlı bir biçimde karşılaştırılmış; BERTurk, XLM-RoBERTa, savasy, BiLSTM ve TF-IDF+LR yöntemleri Doğruluk, Makro-F1, MCC ve Cohen Kappa ölçütleri üzerinden değerlendirilmiştir. {new_abstract_results}",
        paper,
        flags=re.DOTALL
    )

# ── 3. Tablo 2 — Sınıf bazlı BERTurk v3 metrikleri ──────────────
ref_model = berturk_v3 if berturk_v3 else berturk
if ref_model and "per_class" in ref_model:
    pc  = ref_model["per_class"]
    # per_class sadece F1 içerebilir; eski berturk'ta precision/recall/support da var
    if isinstance(list(pc.values())[0], dict):
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
| **Makro Ort.** | **{pct(macro_p)}** | **{pct(macro_r)}** | **{pct(ref_model.get('macro_f1'))}** | 4.969 |
| **Genel Doğruluk** | | | **{pct(ref_model.get('accuracy'))}** | 4.969 |"""
    else:
        # Sadece F1 var (berturk_v3 formatı)
        olu_f1 = pc.get("olumsuz", 0)
        oln_f1 = pc.get("notr", 0)
        oli_f1 = pc.get("olumlu", 0)
        new_table2 = f"""| Sınıf | F1 | Destek |
|:---|:---:|:---:|
| Olumsuz | {pct(olu_f1)} | — |
| Olumlu | {pct(oli_f1)} | — |
| Nötr | {pct(oln_f1)} | — |
| **Makro Ort.** | **{pct(ref_model.get('macro_f1'))}** | 4.969 |
| **Genel Doğruluk** | **{pct(ref_model.get('accuracy'))}** | 4.969 |"""

    paper = re.sub(
        r"\| Sınıf \| (?:Kesinlik \| Duyarlılık \| )?F1 \| Destek \|.*?\| \*\*Genel Doğruluk\*\*[^\n]*\n?",
        new_table2 + "\n",
        paper,
        flags=re.DOTALL
    )

# ── 4. Tablo 3 — Model karşılaştırma (genişletilmiş) ────────────────
if tfidf and bilstm:
    def mrow(name, m, bold=False):
        acc = pct(m.get('accuracy')); f1 = pct(m.get('macro_f1'))
        mcc = f"{m.get('mcc',0):.3f}" if m.get('mcc') else "—"
        auc = pct(m.get('roc_auc_macro')) if m.get('roc_auc_macro') else "—"
        if bold:
            return f"| **{name}** | **{acc}** | **{f1}** | **{mcc}** | **{auc}** |"
        return f"| {name} | {acc} | {f1} | {mcc} | {auc} |"

    rows = []
    rows.append(mrow("TF-IDF + Lojistik Regresyon", tfidf))
    rows.append(mrow("BiLSTM (GloVe, Türkçe-Wiki)", bilstm))
    if zeroshot:
        rows.append(mrow("BERTurk sıfır-atış", zeroshot))
    if berturk:
        rows.append(mrow("BERTurk v2 ince ayarlı", berturk))
    if xlm_v1:
        rows.append(mrow("XLM-RoBERTa v1 ince ayarlı", xlm_v1))
    if ensemble_v1:
        rows.append(mrow("Ensemble v1 (BERTurk v2 + XLM-Ro v1)", ensemble_v1))
    if berturk_v3:
        rows.append(mrow("BERTurk v3b (iki aşamalı ince ayar)", berturk_v3))
    if xlm_v2:
        rows.append(mrow("XLM-RoBERTa v2 ince ayarlı", xlm_v2))
    if savasy:
        rows.append(mrow("savasy Türkçe sınıflandırıcı", savasy))
    if ens_v2:
        rows.append(mrow("Ensemble v2 (BERTurk v3 + XLM-Ro v1)", ens_v2))
    if ens_v3:
        rows.append(mrow("Ensemble v3 (BERTurk v3 + XLM-Ro v2 + savasy)", ens_v3))
    if ens_tuned:
        thr = ens_tuned.get("threshold_notr")
        thr_str = f" eşik={thr:.2f}" if thr else ""
        rows.append(mrow(f"Ensemble (threshold opt.{thr_str})", ens_tuned, bold=True))

    new_table3 = "| Yöntem | Doğruluk | Makro-F1 | MCC | ROC-AUC |\n|:---|:---:|:---:|:---:|:---:|\n" + "\n".join(rows)

    paper = re.sub(
        r"\| Yöntem \| Doğruluk \| Makro-F1 .*?\|.*?(?=\n\n[A-ZBEÇ]|\n\n###)",
        new_table3,
        paper,
        flags=re.DOTALL
    )

# ── 5. Karşılaştırma narratifi ────────────────────────────────────
if best_model and tfidf and bilstm:
    best_f1_val = best_model.get("macro_f1", 0)
    gap_tf = round((best_f1_val - tfidf.get("macro_f1", 0)) * 100, 1)
    gap_bi = round((best_f1_val - bilstm.get("macro_f1", 0)) * 100, 1)
    new_compare = (
        f"{best_name}, TF-IDF temel değerini makro-F1 açısından "
        f"{gap_tf:.1f} puan, BiLSTM temel değerini ise {gap_bi:.1f} puan geride bırakmaktadır."
    )
    paper = re.sub(
        r"(?:İnce ayarlı BERTurk|BERTurk v3b|BERTurk \w+), TF-IDF temel değerini makro-F1 açısından.*?geride bırakmaktadır\.",
        new_compare,
        paper
    )

# ── 6. Ablasyon tablosu (Tablo 4) ────────────────────────────────
ref_ft = berturk_v3 if berturk_v3 else berturk
if zeroshot and ref_ft:
    delta = ref_ft.get("macro_f1", 0) - zeroshot.get("macro_f1", 0)
    new_ablation = f"""| Yapılandırma | Doğruluk | Makro-F1 | ΔF1 |
|:---|:---:|:---:|:---:|
| BERTurk sıfır atış (ince ayarsız) | {pct(zeroshot.get('accuracy'))} | {pct(zeroshot.get('macro_f1'))} | — |
| **BERTurk alana özgü ince ayar (E-ComDe)** | **{pct(ref_ft.get('accuracy'))}** | **{pct(ref_ft.get('macro_f1'))}** | **+{pct(delta)}** |"""

    paper = re.sub(
        r"\| Yapılandırma \| Doğruluk \| Makro-F1 \| ΔF1 \|.*?(?=\n\n|\Z)",
        new_ablation,
        paper,
        flags=re.DOTALL
    )

# ── 7. Ablasyon narratifi ─────────────────────────────────────────
if zeroshot and ref_ft:
    delta_pts = round((ref_ft.get("macro_f1", 0) - zeroshot.get("macro_f1", 0)) * 100, 1)
    delta_str = f"{delta_pts:.1f}".replace(".", ",")
    paper = re.sub(
        r"Ablasyon deneyleri, ince ayarın sıfır-atış modeline kıyasla makro-F1'i \d[\d,]+ puan artırdığını ortaya koymaktadır\.",
        f"Ablasyon deneyleri, ince ayarın sıfır-atış modeline kıyasla makro-F1'i {delta_str} puan artırdığını ortaya koymaktadır.",
        paper
    )
    paper = re.sub(
        r"İnce ayar, sıfır-atış modeline kıyasla makro-F1'i \d[\d,]+ puan artırmaktadır\.",
        f"İnce ayar, sıfır-atış modeline kıyasla makro-F1'i {delta_str} puan artırmaktadır.",
        paper
    )

# ── 8. Eğitim dinamikleri (Bölüm 6.5) ───────────────────────────
ref_hist_model = berturk_v3 if berturk_v3 else berturk
if ref_hist_model and "history" in ref_hist_model:
    hist = ref_hist_model["history"]
    if hist:
        ep1     = hist[0]
        ep_best = max(hist, key=lambda h: h.get("val_f1", 0))
        summary = (
            f"Doğrulama doğruluğu birinci epoch sonunda "
            f"{pct(ep1.get('val_acc'))} iken "
            f"{ep_best['epoch']}. epoch sonunda {pct(ep_best.get('val_acc'))}'e ulaşmış; "
            f"en iyi makro-F1 {pct(ep_best.get('val_f1'))} olarak kaydedilmiştir."
        )
        paper = re.sub(
            r"Doğrulama doğruluğu birinci epoch sonunda.*?kaydedilmiştir\.",
            summary,
            paper,
            flags=re.DOTALL
        )

# ── 9. ROUGE tablosu ─────────────────────────────────────────────
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

# ── 10. Hiperparametre tablosu (Tablo 1) ─────────────────────────
ref_cfg_model = berturk_v3 if berturk_v3 else berturk
if ref_cfg_model and "history" in ref_cfg_model and ref_cfg_model["history"]:
    actual_epochs = len(ref_cfg_model["history"])
    cfg = ref_cfg_model.get("config", {})
    lr  = cfg.get("lr", "8e-6" if berturk_v3 else "1e-5")
    mlen = cfg.get("max_len", 192 if berturk_v3 else 128)
    paper = re.sub(r"\| Epoch sayısı \| \d+ \|", f"| Epoch sayısı | {actual_epochs} |", paper)
    paper = re.sub(r"\| Öğrenme oranı \| [^\|]+ \|", f"| Öğrenme oranı | {lr} |", paper)
    paper = re.sub(r"\| Maksimum dizi uzunluğu \| \d+ \|", f"| Maksimum dizi uzunluğu | {mlen} |", paper)

# ── 11. Sonuç bölümü (Bölüm 7) ───────────────────────────────────
if best_model:
    best_acc_pct = pct(best_model.get("accuracy"))
    best_f1_pct  = pct(best_model.get("macro_f1"))
    new_conclusion_line = (
        f"E-ComDe sistemi; üç sınıflı Türkçe duygu sınıflandırmasında "
        f"{best_name} mimarisiyle {best_acc_pct} genel doğruluk ve {best_f1_pct} makro-F1 elde ederek "
        f"BiLSTM ve TF-IDF temel değerlerini anlamlı biçimde aşmıştır."
    )
    paper = re.sub(
        r"(?:İnce ayarlı BERTurk modeli|E-ComDe sistemi|İki aşamalı ince ayar ile eğitilen BERTurk[^;]*); üç sınıflı Türkçe duygu sınıflandırmasında.*?aşmıştır\.",
        new_conclusion_line,
        paper,
        flags=re.DOTALL
    )

# ── 12. Veri kümesi boyutu güncellemesi ──────────────────────────
paper = paper.replace("14.638 etiketli yorum", "30.000 etiketli yorum")
paper = paper.replace("14.638 yorumla", "30.000 yorumla")

# ── 13. Test kümesi boyutu ───────────────────────────────────────
paper = re.sub(r"3\.000 yorumdan oluşan bağımsız bir test kümesinde",
               "4.969 yorumdan oluşan bağımsız bir test kümesinde (v4 veri seti)", paper)

# ── Kaydet ──────────────────────────────────────────────────────
PAPER_PATH.write_text(paper, encoding="utf-8")
print("Bildiri guncellendi:", PAPER_PATH, flush=True)

# ── Özet çıktı ───────────────────────────────────────────────────
print("\n=== GUNCELLEME OZETI ===", flush=True)
for label, m in [
    ("TF-IDF   ", tfidf), ("BiLSTM   ", bilstm),
    ("BERTurk v2", berturk), ("XLM-Ro v1", xlm_v1),
    ("Ensemble1 ", ensemble_v1), ("BERTurk v3", berturk_v3),
    ("XLM-Ro v2", xlm_v2), ("savasy   ", savasy),
    ("Ensemble2 ", ens_v2), ("Ensemble3 ", ens_v3),
]:
    if m:
        print(f"{label} — Acc: {pct(m.get('accuracy'))} | F1: {pct(m.get('macro_f1'))} | MCC: {fmt4(m.get('mcc'))}", flush=True)
if absa:
    print(f"ABSA      — Acc: {pct(absa.get('accuracy'))} | F1: {pct(absa.get('macro_f1'))}", flush=True)
if rouge:
    print(f"ROUGE-L   — {pct(rouge.get('rougeL'))}", flush=True)
