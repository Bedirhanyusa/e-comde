"""
Claude claude-haiku-4-5 tabanlı Türkçe özet ve rehber üretici.
BERTurk sentiment sınıflandırmasının çıktısını alır, anlamlı özetler
ve "Bu ürünü almalı mısın?" rehber kartı üretir.
"""

from __future__ import annotations
import json
import os
import anthropic

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_MODEL = "claude-haiku-4-5-20251001"

_SYS_SUMMARIZE = (
    "Sen bir Türk e-ticaret platformunun yapay zeka yorum analistisisin. "
    "Müşteri yorumlarını okuyup TAM 2 cümlelik, somut ve akıcı Türkçe özetler yazarsın.\n"
    "Kurallar:\n"
    "- Geniş zaman kullan: 'öne çıkıyor', 'sıklıkla belirtiliyor', 'dikkat çekiyor'.\n"
    "- Spesifik özellik adlarını kullan (örn: 'ses kalitesi', 'pil ömrü').\n"
    "- Yüzde/sıklık hissi ver: 'büyük çoğunluk', 'pek çok kullanıcı'.\n"
    "- TAM 2 cümle — ne fazla ne az.\n"
    "- Liste, numara, madde işareti, tire, başlık YASAK."
)

_SYS_FULL = (
    "Sen bir Türk e-ticaret platformunun yapay zeka yorum analistisisin. "
    "YALNIZCA geçerli JSON döndür, başka hiçbir şey yazma."
)

_SYS_ADVISOR = (
    "Sen bir Türk e-ticaret platformunun yapay zeka alışveriş danışmanısın. "
    "Sana ürün yorumları ve sentiment analizi verilir. "
    "Alıcıya net, dürüst ve faydalı tavsiye verirsin. "
    "YALNIZCA geçerli JSON döndür, başka hiçbir şey yazma."
)

_SYS_EMOTION = (
    "Sen bir Türk e-ticaret platformunun yapay zeka duygu analistisisin. "
    "YALNIZCA geçerli JSON döndür, başka hiçbir şey yazma."
)


class ReviewSummarizer:
    def __init__(self, api_key: str = ""):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY bulunamadı. "
                ".env dosyasına ANTHROPIC_API_KEY=sk-ant-... ekleyin."
            )
        self.client = anthropic.Anthropic(api_key=key)
        print("[Summarizer] Claude API hazır.", flush=True)

    # ── Özet ─────────────────────────────────────────────────────────────

    def summarize(
        self,
        reviews: list[str],
        sentiment: str = "genel",
        top_reviews: int = 15,
        **_kwargs,
    ) -> str:
        if not reviews:
            return ""

        label_map = {"olumlu": "olumlu", "olumsuz": "olumsuz", "notr": "nötr"}
        label = label_map.get(sentiment, "genel")
        subset = reviews[:top_reviews]
        reviews_text = "\n".join(f"- {r.strip()}" for r in subset)

        prompt = (
            f"Aşağıdaki {len(subset)} {label} müşteri yorumunu oku "
            f"ve TAM 2 cümleyle özetle:\n{reviews_text}\nÖzet:"
        )

        try:
            msg = self.client.messages.create(
                model=_MODEL,
                max_tokens=200,
                system=_SYS_SUMMARIZE,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception as e:
            err = str(e)
            if "credit" in err.lower() or "balance" in err.lower():
                return "⚠️ API kredisi yetersiz. console.anthropic.com adresinden kredi yükleyin."
            return f"Özet üretilemedi: {err[:120]}"

    def summarize_by_sentiment(
        self, reviews: list[str], labels: list[str]
    ) -> dict[str, str]:
        grouped: dict[str, list[str]] = {"olumlu": [], "olumsuz": [], "notr": []}
        for text, label in zip(reviews, labels):
            if label in grouped:
                grouped[label].append(text)
        return {s: self.summarize(texts, s) for s, texts in grouped.items() if texts}

    # ── Tam analiz (özet + madde listeleri tek çağrıda) ──────────────────

    def get_full_analysis(
        self, reviews: list[str], labels: list[str]
    ) -> dict:
        """
        Tek API çağrısında döndürür:
        {
          "summaries": {"olumlu": "...", "olumsuz": "...", "notr": "..."},
          "overall": "2-3 cümle genel özet",
          "pros": ["artı 1", "artı 2", "artı 3"],
          "cons": ["eksi 1", "eksi 2", "eksi 3"]
        }
        """
        olumlu = [r for r, l in zip(reviews, labels) if l == "olumlu"]
        olumsuz = [r for r, l in zip(reviews, labels) if l == "olumsuz"]
        notr    = [r for r, l in zip(reviews, labels) if l == "notr"]

        def sample(lst, n=12):
            return "\n".join(f"- {r.strip()}" for r in lst[:n]) or "(yok)"

        prompt = f"""Ürün yorumları aşağıda verilmiştir. Analiz et ve YALNIZCA şu JSON formatında yanıt ver:

{{
  "summaries": {{
    "olumlu": "TAM 2 cümle olumlu özet. Geniş zaman, spesifik özellik adları kullan.",
    "olumsuz": "TAM 2 cümle olumsuz özet. Geniş zaman, spesifik şikayet konuları belirt.",
    "notr": "TAM 2 cümle nötr özet (nötr yorum yoksa boş string döndür)."
  }},
  "overall": "Bu ürünü 2-3 cümleyle değerlendir: güçlü yönleri ve zayıf yönleri dengeli şekilde aktar. Doğal, güvenilir bir üslup kullan.",
  "pros": ["Kısa ve net artı madde 1", "Kısa ve net artı madde 2", "Kısa ve net artı madde 3"],
  "cons": ["Kısa ve net eksi madde 1", "Kısa ve net eksi madde 2", "Kısa ve net eksi madde 3"]
}}

Kurallar:
- Tüm metinler Türkçe olacak
- "pros" ve "cons" her biri maksimum 10 kelime, net ve somut olacak
- Yorum yoksa o alan için makul bir fallback yaz
- Başka hiçbir metin ekleme

OLUMLU YORUMLAR ({len(olumlu)} adet):
{sample(olumlu)}

OLUMSUZ YORUMLAR ({len(olumsuz)} adet):
{sample(olumsuz)}

NÖTR YORUMLAR ({len(notr)} adet):
{sample(notr)}"""

        try:
            msg = self.client.messages.create(
                model=_MODEL,
                max_tokens=800,
                system=_SYS_FULL,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            # Zorunlu alanları doğrula
            data.setdefault("summaries", {})
            data.setdefault("overall", "")
            data.setdefault("pros", [])
            data.setdefault("cons", [])
            return data
        except Exception as e:
            # Fallback: sentiment-by-sentiment
            summaries = self.summarize_by_sentiment(reviews, labels)
            overall = " ".join(filter(None, [
                summaries.get("olumlu", ""),
                summaries.get("notr", ""),
            ]))[:400]
            return {
                "summaries": summaries,
                "overall": overall,
                "pros": [],
                "cons": [],
            }

    # ── Rehber / Danışman ────────────────────────────────────────────────

    def get_advisor(
        self,
        reviews: list[str],
        labels: list[str],
        score: float,
    ) -> dict:
        """
        Döndürür:
        {
          "verdict": "Tavsiye Edilir" | "Dikkatli Olun" | "Tavsiye Edilmez",
          "verdict_score": 3 | 2 | 1,
          "target_audience": "Kim için ideal? (1 cümle)",
          "buy_if": ["durum1", "durum2"],
          "watch_out": ["uyarı1", "uyarı2"]
        }
        """
        if not reviews:
            return _fallback_advisor(score)

        olumlu = [r for r, l in zip(reviews, labels) if l == "olumlu"]
        olumsuz = [r for r, l in zip(reviews, labels) if l == "olumsuz"]
        notr = [r for r, l in zip(reviews, labels) if l == "notr"]

        sample_pos = "\n".join(f"- {r}" for r in olumlu[:6])
        sample_neg = "\n".join(f"- {r}" for r in olumsuz[:6])
        sample_neu = "\n".join(f"- {r}" for r in notr[:3])

        prompt = f"""Ürün analiz verileri:
Sentiment skoru: {score:.2f} / 5.00
Olumlu yorum sayısı: {len(olumlu)}
Olumsuz yorum sayısı: {len(olumsuz)}
Nötr yorum sayısı: {len(notr)}

Örnek olumlu yorumlar:
{sample_pos or "(yok)"}

Örnek olumsuz yorumlar:
{sample_neg or "(yok)"}

Örnek nötr yorumlar:
{sample_neu or "(yok)"}

Bu verilere dayanarak alıcıya Türkçe tavsiye ver. YALNIZCA şu JSON formatında yanıt ver:
{{
  "verdict": "Tavsiye Edilir" | "Dikkatli Olun" | "Tavsiye Edilmez",
  "verdict_score": 3 | 2 | 1,
  "target_audience": "Bu ürün ... için ideal. (1 cümle)",
  "buy_if": ["durum 1", "durum 2", "durum 3"],
  "watch_out": ["uyarı 1", "uyarı 2", "uyarı 3"]
}}"""

        try:
            msg = self.client.messages.create(
                model=_MODEL,
                max_tokens=500,
                system=_SYS_ADVISOR,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
        except Exception:
            return _fallback_advisor(score)
        # JSON'u güvenli parse et
        try:
            # Bazen model markdown kod bloğu içine alır
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            # Zorunlu alanları kontrol et
            for key in ("verdict", "verdict_score", "target_audience", "buy_if", "watch_out"):
                if key not in data:
                    raise ValueError(f"Eksik alan: {key}")
            return data
        except Exception:
            return _fallback_advisor(score)

    # ── Duygu Derinliği ──────────────────────────────────────────────────

    def get_emotions(self, reviews: list[str], labels: list[str]) -> dict:
        """
        BERTurk'ün 3 sınıfını 5 duygusal kategoriye alt-sınıflandırır:
        coskulu | memnun | notr | hayal_kirikligi | ofkeli
        """
        olumlu = [r for r, l in zip(reviews, labels) if l == "olumlu"][:12]
        olumsuz = [r for r, l in zip(reviews, labels) if l == "olumsuz"][:12]
        notr_count = sum(1 for l in labels if l == "notr")

        if not olumlu and not olumsuz:
            return {"coskulu": 0, "memnun": 0, "notr": notr_count,
                    "hayal_kirikligi": 0, "ofkeli": 0}

        parts = []
        if olumlu:
            parts.append("OLUMLU YORUMLAR:\n" + "\n".join(f"- {r}" for r in olumlu))
        if olumsuz:
            parts.append("OLUMSUZ YORUMLAR:\n" + "\n".join(f"- {r}" for r in olumsuz))

        prompt = f"""Bu Türkçe yorumları oku.

Olumlu yorumları "coskulu" veya "memnun" olarak sınıflandır:
- coskulu: Çok heyecanlı, "harika!", "bayıldım", "süper", aşırı memnun
- memnun: Normal memnuniyet, "güzel", "iyi", "beğendim", tatmin olmuş

Olumsuz yorumları "hayal_kirikligi" veya "ofkeli" olarak sınıflandır:
- hayal_kirikligi: Hayal kırıklığı, beklenti karşılanmadı, "beklediğimden kötü"
- ofkeli: Kızgın, sinirli, "berbat", "rezalet", "çöp", sert eleştiri

{chr(10).join(parts)}

Nötr yorum sayısı: {notr_count} (bunları saymana gerek yok, direkt {notr_count} yaz)

YALNIZCA şu JSON formatında yanıt ver:
{{"coskulu": <sayı>, "memnun": <sayı>, "notr": {notr_count}, "hayal_kirikligi": <sayı>, "ofkeli": <sayı>}}"""

        try:
            msg = self.client.messages.create(
                model=_MODEL,
                max_tokens=120,
                system=_SYS_EMOTION,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            for key in ("coskulu", "memnun", "notr", "hayal_kirikligi", "ofkeli"):
                data.setdefault(key, 0)
            return data
        except Exception:
            return _fallback_emotions(len(olumlu), len(olumsuz), notr_count)


def _fallback_emotions(olumlu_count: int, olumsuz_count: int, notr_count: int) -> dict:
    return {
        "coskulu": olumlu_count // 2,
        "memnun": olumlu_count - olumlu_count // 2,
        "notr": notr_count,
        "hayal_kirikligi": olumsuz_count // 2,
        "ofkeli": olumsuz_count - olumsuz_count // 2,
    }


def _fallback_advisor(score: float) -> dict:
    if score >= 4.0:
        return {
            "verdict": "Tavsiye Edilir",
            "verdict_score": 3,
            "target_audience": "Bu ürün genel kullanım için uygundur.",
            "buy_if": ["Güvenilir bir ürün arıyorsanız", "Fiyat-performans önemliyse"],
            "watch_out": ["Bireysel beklentilerinizi göz önünde bulundurun"],
        }
    elif score >= 3.0:
        return {
            "verdict": "Dikkatli Olun",
            "verdict_score": 2,
            "target_audience": "Belirli kullanım senaryoları için uygun olabilir.",
            "buy_if": ["Beklentileriniz düşükse", "Fiyatı uygunsa"],
            "watch_out": ["Yorumları dikkatlice okuyun", "İade politikasını kontrol edin"],
        }
    else:
        return {
            "verdict": "Tavsiye Edilmez",
            "verdict_score": 1,
            "target_audience": "Bu ürün çoğu kullanıcı için uygun değil.",
            "buy_if": ["Başka alternatif yoksa ve fiyatı çok düşükse"],
            "watch_out": ["Müşteri memnuniyeti düşük", "Kalite sorunları bildiriliyor"],
        }
