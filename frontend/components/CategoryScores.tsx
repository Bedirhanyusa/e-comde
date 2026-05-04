"use client";

import { BarChart2 } from "lucide-react";

interface Props {
  scores: Array<{ name: string; score: number }>;
}

function barColor(s: number): string {
  if (s >= 4.0) return "#22c55e";
  if (s >= 2.8) return "#fbbf24";
  return "#ef4444";
}

export function CategoryScores({ scores }: Props) {
  if (!scores.length) return null;
  return (
    <div className="bg-white dark:bg-[var(--surface)] border border-[var(--border)] rounded-3xl shadow-sm p-6 h-full">
      <div className="flex items-center gap-2 mb-5">
        <BarChart2 className="w-4 h-4 text-violet-600" />
        <h3 className="font-bold text-base text-[var(--text)]">Kategori Puanları</h3>
      </div>
      <div className="space-y-4">
        {scores.map(({ name, score }) => (
          <div key={name}>
            <div className="flex justify-between items-center mb-1.5">
              <span className="text-sm text-[var(--text)]">{name}</span>
              <span className="text-sm font-bold text-[var(--text)] tabular-nums">
                {score.toFixed(1)} / 5
              </span>
            </div>
            <div
              className="w-full rounded-full"
              style={{ background: "#F3F4F6", height: 10 }}
            >
              <div
                className="rounded-full transition-all duration-700"
                style={{
                  width: `${(score / 5) * 100}%`,
                  height: 10,
                  background: barColor(score),
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
