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
}
