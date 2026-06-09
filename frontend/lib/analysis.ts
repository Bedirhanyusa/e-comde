import { ReviewResult, Sentiment } from "./types";

export function calcReviewQuality(text: string): number {
  const t = text.trim();
  if (!t) return 0;

  const len = t.length;
  const lenScore = len < 20 ? 0.1 : len < 50 ? 0.4 : len < 150 ? 0.7 : 1.0;

  const wordCount = t.split(/\s+/).filter(Boolean).length;
  const wordScore = wordCount <= 3 ? 0.2 : wordCount <= 10 ? 0.5 : 1.0;

  let score = (lenScore + wordScore) / 2;

  // Sayısal/ölçüsel bilgi içeriyorsa daha bilgi yoğun kabul et
  if (/\d+\s*(cm|mm|saat|gün|ay|yıl|kg|gr|ml|lt|tl|₺)/i.test(t)) score += 0.2;

  // Karşılaştırma/itiraz ifadesi içeriyorsa daha analitik
  if (/\b(ama|ancak|fakat|beklediğim|beklenenden|oysa|ne yazık|maalesef|aslında)\b/i.test(t)) score += 0.15;

  return Math.min(score, 1);
}

export function calcQualityStats(results: ReviewResult[]): { highQuality: number; total: number } {
  const highQuality = results.filter(r => calcReviewQuality(r.text) > 0.6).length;
  return { highQuality, total: results.length };
}

export function calcSentimentScore(results: ReviewResult[]): number {
  if (!results.length) return 0;
  const scoreMap: Record<string, number> = { olumlu: 5, notr: 3, olumsuz: 1 };
  let weightedSum = 0;
  let totalWeight = 0;
  for (const r of results) {
    const quality = calcReviewQuality(r.text);
    const weight = 0.4 + 0.6 * quality; // min 0.4, max 1.0
    weightedSum += (scoreMap[r.label] ?? 3) * weight;
    totalWeight += weight;
  }
  return totalWeight > 0 ? weightedSum / totalWeight : 0;
}

export function splitPoints(text: string, max = 4): string[] {
  if (!text) return [];
  return text
    .split(/\.\s+/)
    .map(s => s.trim().replace(/^[-•*]\s*/, ""))
    .filter(s => s.length > 15)
    .slice(0, max)
    .map(s => (s.endsWith(".") ? s : s + "."));
}

const CATEGORY_DEFS = [
  { name: "Kargo & Teslimat",    keys: ["kargo", "teslimat", "paket", "ambalaj", "hızlı geldi", "geç geldi"] },
  { name: "Ürün Kalitesi",       keys: ["kalite", "sağlam", "dayanıklı", "bozuk", "kırık", "malzeme", "ince"] },
  { name: "Fiyat/Performans",    keys: ["fiyat", "ücret", "para", "değer", "pahalı", "ucuz", "ekonomik"] },
  { name: "Müşteri Hizmetleri",  keys: ["hizmet", "destek", "iade", "müşteri", "satıcı", "yardım"] },
  { name: "Ürün Görünümü",       keys: ["renk", "şık", "güzel", "estetik", "tasarım", "görünüm"] },
];

export function calcCategoryScores(
  results: ReviewResult[]
): Array<{ name: string; score: number }> {
  return CATEGORY_DEFS
    .map(({ name, keys }) => {
      const match = results.filter(r =>
        keys.some(k => r.text.toLowerCase().includes(k))
      );
      if (match.length < 2) return null;
      const o = match.filter(r => r.label === "olumlu").length;
      const n = match.filter(r => r.label === "notr").length;
      const u = match.filter(r => r.label === "olumsuz").length;
      return { name, score: (o * 5 + n * 3 + u * 1) / match.length };
    })
    .filter(Boolean) as Array<{ name: string; score: number }>;
}

export function groupFlaggedByReason(
  suspicious: Array<{ review: ReviewResult; reasons: string[] }>
): Array<{ reason: string; count: number }> {
  const counts: Record<string, number> = {};
  for (const { reasons } of suspicious) {
    for (const r of reasons) {
      counts[r] = (counts[r] || 0) + 1;
    }
  }
  return Object.entries(counts)
    .map(([reason, count]) => ({ reason, count }))
    .sort((a, b) => b.count - a.count);
}

export function getTopReviews(
  results: ReviewResult[],
  sentiment: Sentiment,
  n = 2
): ReviewResult[] {
  return [...results]
    .filter(r => r.label === sentiment)
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, n);
}
