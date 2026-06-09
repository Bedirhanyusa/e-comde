import { SentimentResponse, SummarizeResponse, AdvisorResponse, EmotionResponse, ShopCategory, ShopProduct, ShopAnalysisResponse } from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function analyzeSentiment(texts: string[]): Promise<SentimentResponse> {
  const res = await fetch(`${BASE_URL}/api/v1/sentiment/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ texts }),
  });
  if (!res.ok) throw new Error("Sentiment analizi başarısız.");
  return res.json();
}

export async function summarizeReviews(
  reviews: string[],
  labels: string[]
): Promise<SummarizeResponse> {
  const res = await fetch(`${BASE_URL}/api/v1/summarize/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reviews, labels }),
  });
  if (!res.ok) throw new Error("Özetleme başarısız.");
  return res.json();
}

export async function getAdvisor(
  reviews: string[],
  labels: string[],
  score: number
): Promise<AdvisorResponse> {
  const res = await fetch(`${BASE_URL}/api/v1/summarize/advisor`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reviews, labels, score }),
  });
  if (!res.ok) throw new Error("Danışman analizi başarısız.");
  return res.json();
}

export async function getEmotions(
  reviews: string[],
  labels: string[]
): Promise<EmotionResponse> {
  const res = await fetch(`${BASE_URL}/api/v1/summarize/emotions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reviews, labels }),
  });
  if (!res.ok) throw new Error("Duygu analizi başarısız.");
  return res.json();
}

export async function getShopProducts(category?: string): Promise<{ categories: ShopCategory[]; products: ShopProduct[] }> {
  const url = category
    ? `${BASE_URL}/api/v1/shop/products?category=${encodeURIComponent(category)}`
    : `${BASE_URL}/api/v1/shop/products`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("Ürünler yüklenemedi.");
  return res.json();
}

export async function analyzeShopProduct(productId: string): Promise<ShopAnalysisResponse> {
  const res = await fetch(`${BASE_URL}/api/v1/shop/products/${productId}/analyze`, {
    method: "POST",
  });
  if (!res.ok) throw new Error("Ürün analizi başarısız.");
  return res.json();
}

export async function analyzeByUrl(url: string): Promise<ShopAnalysisResponse> {
  const res = await fetch(`${BASE_URL}/api/v1/shop/analyze-by-url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { detail?: string }).detail ?? "URL analizi başarısız.");
  }
  return res.json();
}

export async function uploadCSV(file: File): Promise<SentimentResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE_URL}/api/v1/ingest/csv`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error("CSV yükleme başarısız.");
  return res.json();
}
