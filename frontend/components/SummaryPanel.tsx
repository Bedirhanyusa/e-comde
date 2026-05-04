"use client";

import { ThumbsUp, ThumbsDown, Minus, Bot } from "lucide-react";
import { Sentiment } from "@/lib/types";

const CONFIG: Record<Sentiment, { label: string; icon: typeof ThumbsUp; border: string; iconColor: string; bg: string }> = {
  olumlu: {
    label: "Olumlu Yorumlar",
    icon: ThumbsUp,
    border: "border-green-200 dark:border-green-800",
    iconColor: "text-green-500",
    bg: "bg-green-50 dark:bg-green-900/20",
  },
  olumsuz: {
    label: "Olumsuz Yorumlar",
    icon: ThumbsDown,
    border: "border-red-200 dark:border-red-800",
    iconColor: "text-red-500",
    bg: "bg-red-50 dark:bg-red-900/20",
  },
  notr: {
    label: "Nötr Yorumlar",
    icon: Minus,
    border: "border-amber-200 dark:border-amber-800",
    iconColor: "text-amber-500",
    bg: "bg-amber-50 dark:bg-amber-900/20",
  },
};

interface Props {
  summaries: Partial<Record<Sentiment, string>>;
}

export function SummaryPanel({ summaries }: Props) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {(["olumlu", "olumsuz", "notr"] as Sentiment[]).map((label) => {
        const cfg = CONFIG[label];
        const Icon = cfg.icon;
        return summaries[label] ? (
          <div key={label} className={`rounded-2xl border p-5 ${cfg.border} ${cfg.bg}`}>
            <div className="flex items-center gap-2 mb-3">
              <Icon className={`w-4 h-4 ${cfg.iconColor}`} />
              <h3 className="font-semibold text-sm">{cfg.label}</h3>
              <Bot className="w-3 h-3 text-[var(--text-muted)] ml-auto" />
            </div>
            <p className="text-sm text-[var(--text)] leading-relaxed">{summaries[label]}</p>
          </div>
        ) : null;
      })}
    </div>
  );
}
