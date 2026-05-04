"use client";

import { Sparkles, Users } from "lucide-react";

interface Props {
  score: number;
  totalReviews: number;
  overallSummary: string;
  targetAudience?: string;
}

function StarRating({ score }: { score: number }) {
  return (
    <div className="flex gap-1">
      {[1, 2, 3, 4, 5].map((i) => {
        const filled = score >= i - 0.25;
        return (
          <svg
            key={i}
            width="22"
            height="22"
            viewBox="0 0 20 20"
            style={{ width: 22, height: 22, flexShrink: 0 }}
          >
            <path
              d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"
              fill={filled ? "#F59E0B" : "#E5E7EB"}
            />
          </svg>
        );
      })}
    </div>
  );
}

export function ScoreSummaryCard({ score, totalReviews, overallSummary, targetAudience }: Props) {
  return (
    <div className="bg-white dark:bg-[var(--surface)] border border-[var(--border)] rounded-3xl shadow-sm p-6">
      <div className="grid grid-cols-1 md:grid-cols-[200px_1fr] gap-8">

        {/* Left: score */}
        <div className="flex flex-col items-center justify-center gap-3 py-4 border-b md:border-b-0 md:border-r border-[var(--border)]">
          <div
            className="tabular-nums leading-none text-[var(--text)]"
            style={{ fontSize: 72, fontWeight: 800, lineHeight: 1 }}
          >
            {score.toFixed(1)}
          </div>
          <StarRating score={score} />
          <p className="text-xs text-[var(--text-muted)]">
            {totalReviews.toLocaleString("tr-TR")} yorum
          </p>
        </div>

        {/* Right: AI summary */}
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-violet-600" />
            <h3 className="font-bold text-base text-[var(--text)]">Yapay Zeka Özeti</h3>
          </div>
          <p className="text-sm text-[var(--text)] leading-relaxed">{overallSummary}</p>

          {targetAudience && (
            <div className="mt-auto pt-4 border-t border-[var(--border)] flex items-start gap-2">
              <Users className="w-4 h-4 text-violet-500 flex-shrink-0 mt-0.5" />
              <div>
                <span className="text-xs font-semibold text-[var(--text-muted)]">Kimler Almalı? </span>
                <span className="text-xs text-[var(--text)]">{targetAudience}</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
