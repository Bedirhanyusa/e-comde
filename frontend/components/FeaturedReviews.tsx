"use client";

import { MessageSquareQuote, ThumbsUp, ThumbsDown, Minus } from "lucide-react";
import { ReviewResult, Sentiment } from "@/lib/types";

const SENTIMENT_ICON = {
  olumlu: { Icon: ThumbsUp,  color: "text-green-600" },
  olumsuz: { Icon: ThumbsDown, color: "text-red-500" },
  notr:    { Icon: Minus,     color: "text-gray-400" },
};

const SENTIMENT_LABEL = { olumlu: "Olumlu", olumsuz: "Olumsuz", notr: "Nötr" };

function ReviewCard({ review }: { review: ReviewResult }) {
  const { Icon, color } = SENTIMENT_ICON[review.label];
  const initial = review.text.trim()[0]?.toUpperCase() ?? "?";
  // Fake a short display name from first words
  const name = review.text.split(" ").slice(0, 2).join(" ").substring(0, 14) + "…";

  return (
    <div className="bg-white dark:bg-[var(--surface)] border border-[var(--border)] rounded-2xl p-5 flex flex-col gap-3">
      <div className="flex items-center gap-2.5">
        <div className="w-8 h-8 rounded-full bg-violet-100 dark:bg-violet-900/40 flex items-center justify-center text-violet-700 dark:text-violet-300 font-bold text-sm flex-shrink-0">
          {initial}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-semibold text-[var(--text)] truncate">{name}</p>
          <span className={`flex items-center gap-1 text-xs ${color}`}>
            <Icon className="w-3 h-3" />
            {SENTIMENT_LABEL[review.label]}
          </span>
        </div>
        <span className="text-xs text-[var(--text-muted)] tabular-nums">{(review.confidence * 100).toFixed(0)}%</span>
      </div>
      <p className="text-sm text-[var(--text)] leading-relaxed italic">
        &ldquo;{review.text}&rdquo;
      </p>
    </div>
  );
}

interface Props {
  reviews: ReviewResult[];
}

export function FeaturedReviews({ reviews }: Props) {
  const pick = (s: Sentiment, n: number) =>
    [...reviews].filter(r => r.label === s).sort((a, b) => b.confidence - a.confidence).slice(0, n);

  const featured = [
    ...pick("olumlu", 2),
    ...pick("notr", 1),
    ...pick("olumsuz", 2),
  ].slice(0, 6);

  if (!featured.length) return null;

  return (
    <div className="bg-white dark:bg-[var(--surface)] border border-[var(--border)] rounded-3xl shadow-sm p-6">
      <div className="flex items-center gap-2 mb-5">
        <MessageSquareQuote className="w-4 h-4 text-violet-600" />
        <h3 className="font-bold text-base text-[var(--text)]">Öne Çıkan Yorumlar</h3>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {featured.map((r, i) => (
          <ReviewCard key={i} review={r} />
        ))}
      </div>
    </div>
  );
}
