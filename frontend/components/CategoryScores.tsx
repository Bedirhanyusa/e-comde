"use client";

import { BarChart2 } from "lucide-react";
import { motion } from "framer-motion";

interface Props {
  scores: Array<{ name: string; score: number }>;
}

function barColor(s: number): string {
  if (s >= 4.0) return "#22c55e";
  if (s >= 2.8) return "#f59e0b";
  return "#ef4444";
}

export function CategoryScores({ scores }: Props) {
  if (!scores.length) return null;
  return (
    <div className="bg-[var(--surface)] border border-[var(--border)] rounded-3xl shadow-sm p-6 h-full">
      <div className="flex items-center gap-2 mb-5">
        <div className="p-1.5 rounded-lg bg-violet-100 dark:bg-violet-900/30">
          <BarChart2 className="w-3.5 h-3.5 text-violet-600 dark:text-violet-400" />
        </div>
        <h3 className="font-bold text-sm text-[var(--text)] uppercase tracking-wide">Kategori Puanları</h3>
      </div>
      <div className="space-y-4">
        {scores.map(({ name, score }, i) => (
          <motion.div
            key={name}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.08, duration: 0.4 }}
          >
            <div className="flex justify-between items-center mb-1.5">
              <span className="text-sm text-[var(--text)]">{name}</span>
              <span className="text-sm font-bold tabular-nums" style={{ color: barColor(score) }}>
                {score.toFixed(1)}
              </span>
            </div>
            <div className="w-full h-2 rounded-full bg-[var(--border)] overflow-hidden">
              <motion.div
                className="h-full rounded-full"
                style={{ background: barColor(score) }}
                initial={{ width: 0 }}
                animate={{ width: `${(score / 5) * 100}%` }}
                transition={{ delay: 0.2 + i * 0.08, duration: 0.8, ease: "easeOut" }}
              />
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
