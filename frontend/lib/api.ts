import { SentimentResponse, SummarizeResponse } from "./types";

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
