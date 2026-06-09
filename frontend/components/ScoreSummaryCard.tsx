"use client";

import { Sparkles } from "lucide-react";
import { motion, useMotionValue, useTransform, animate } from "framer-motion";
import { useEffect } from "react";

interface Props {
  score: number;
  totalReviews: number;
  overallSummary: string;
}

function AnimatedScore({ target }: { target: number }) {
  const count = useMotionValue(0);
  const rounded = useTransform(count, (v) => v.toFixed(1));

  useEffect(() => {
    const controls = animate(count, target, { duration: 1.4, ease: "easeOut" });
    return controls.stop;
  }, [target, count]);

  return <motion.span>{rounded}</motion.span>;
}

function StarRating({ score }: { score: number }) {
  return (
    <div className="flex gap-1">
      {[1, 2, 3, 4, 5].map((i) => {
        const filled = score >= i - 0.25;
        const half = !filled && score >= i - 0.75;
        return (
          <motion.svg
            key={i}
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.8 + i * 0.08, type: "spring", stiffness: 300 }}
            width="20" height="20" viewBox="0 0 20 20"
          >
            <path
              d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"
              fill={filled ? "#F59E0B" : half ? "url(#half)" : "#E5E7EB"}
            />
          </motion.svg>
        );
      })}
    </div>
  );
}

export function ScoreSummaryCard({ score, totalReviews, overallSummary }: Props) {
  const scoreColor = score >= 4 ? "#22c55e" : score >= 3 ? "#f59e0b" : "#ef4444";
  const circumference = 2 * Math.PI * 36;
  const pct = ((score - 1) / 4);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="bg-[var(--surface)] border border-[var(--border)] rounded-3xl shadow-sm overflow-hidden"
    >
      {/* Top accent line */}
      <div className="h-0.5 w-full bg-gradient-to-r from-violet-500 via-purple-500 to-indigo-500" />

      <div className="p-6 grid grid-cols-1 md:grid-cols-[220px_1fr] gap-8">
        {/* Left: Score */}
        <div className="flex flex-col items-center justify-center gap-4 py-2">
          {/* Circular progress */}
          <div className="relative w-28 h-28">
            <svg className="w-28 h-28 -rotate-90" viewBox="0 0 80 80">
              <circle cx="40" cy="40" r="36" fill="none" stroke="var(--border)" strokeWidth="6" />
              <motion.circle
                cx="40" cy="40" r="36"
                fill="none"
                stroke={scoreColor}
                strokeWidth="6"
                strokeLinecap="round"
                strokeDasharray={circumference}
                initial={{ strokeDashoffset: circumference }}
                animate={{ strokeDashoffset: circumference * (1 - pct) }}
                transition={{ duration: 1.4, ease: "easeOut", delay: 0.2 }}
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-3xl font-extrabold tabular-nums leading-none" style={{ color: scoreColor }}>
                <AnimatedScore target={score} />
              </span>
              <span className="text-xs text-[var(--text-muted)] font-medium">/ 5</span>
            </div>
          </div>
          <StarRating score={score} />
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.2 }}
            className="text-xs text-[var(--text-muted)]"
          >
            {totalReviews.toLocaleString("tr-TR")} yorum analiz edildi
          </motion.p>
        </div>

        {/* Right: Summary */}
        <motion.div
          initial={{ opacity: 0, x: 10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3, duration: 0.5 }}
          className="flex flex-col gap-3 justify-center"
        >
          <div className="flex items-center gap-2">
            <div className="p-1.5 rounded-lg bg-violet-100 dark:bg-violet-900/30">
              <Sparkles className="w-3.5 h-3.5 text-violet-600 dark:text-violet-400" />
            </div>
            <h3 className="font-bold text-sm text-[var(--text)] uppercase tracking-wide">
              Yapay Zeka Özeti
            </h3>
          </div>
          <p className="text-sm text-[var(--text)] leading-relaxed">
            {overallSummary}
          </p>
        </motion.div>
      </div>
    </motion.div>
  );
}
