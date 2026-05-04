import { ReviewResult, Sentiment } from "./types";

export function calcSentimentScore(results: ReviewResult[]): number {
  if (!results.length) return 0;
  const o = results.filter(r => r.label === "olumlu").length;
  const n = results.filter(r => r.label === "notr").length;
  const u = results.filter(r => r.label === "olumsuz").length;
  return (o * 5 + n * 3 + u * 1) / results.length;
}

export function splitPoints(text: string, max = 4): string[] {
  if (!text) return [];
  return text
    .split(/\.\s+/)
    .map(s => s.trim().replace(/^[-•*]\s*/, ""))
    .filter(s => s.length > 18)
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
