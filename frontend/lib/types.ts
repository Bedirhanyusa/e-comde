export type Sentiment = "olumlu" | "olumsuz" | "notr";

export interface ReviewResult {
  text: string;
  label: Sentiment;
  confidence: number;
  flagged: boolean;
}

export interface SentimentResponse {
  results: ReviewResult[];
  total: number;
  flagged_count: number;
}

export interface SummarizeResponse {
  summaries: Record<Sentiment, string>;
  overall: string;
  pros: string[];
  cons: string[];
}

export interface AdvisorResponse {
  verdict: "Tavsiye Edilir" | "Dikkatli Olun" | "Tavsiye Edilmez";
  verdict_score: 1 | 2 | 3;
  target_audience: string;
  buy_if: string[];
  watch_out: string[];
}

export interface EmotionResponse {
  coskulu: number;
  memnun: number;
  notr: number;
  hayal_kirikligi: number;
  ofkeli: number;
}

// Shop types
export interface ShopCategory {
  id: string;
  label: string;
  icon: string;
  color: string;
  product_count: number;
}

export interface ShopProduct {
  id: string;
  category_id: string;
  name: string;
  brand: string;
  description: string;
  price: string;
  features: string[];
  image: string;
  trendyol_url: string;
  avg_rating: number;
  total_reviews: number;
}

export interface ShopAnalysisResponse {
  product: ShopProduct;
  results: ReviewResult[];
  total: number;
  flagged_count: number;
  sentiment_distribution: Record<Sentiment, number>;
  summaries: Record<Sentiment, string>;
  overall: string;
  pros: string[];
  cons: string[];
  avg_confidence: number;
}
