"use client";

import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";
import { motion } from "framer-motion";
import { EmotionResponse } from "@/lib/types";

const EMOTIONS: { key: keyof EmotionResponse; label: string; emoji: string; color: string }[] = [
  { key: "coskulu",        label: "Coşkulu",         emoji: "😍", color: "#8B5CF6" },
  { key: "memnun",         label: "Memnun",           emoji: "😊", color: "#22c55e" },
  { key: "notr",           label: "Nötr",             emoji: "😐", color: "#9CA3AF" },
  { key: "hayal_kirikligi",label: "Hayal Kırıklığı",  emoji: "😞", color: "#F97316" },
  { key: "ofkeli",         label: "Öfkeli",           emoji: "😠", color: "#ef4444" },
];

interface Props { data: EmotionResponse }

export function EmotionDistributionChart({ data }: Props) {
  const chartData = EMOTIONS
    .map(e => ({ ...e, value: data[e.key] }))
    .filter(e => e.value > 0);

  const total = Object.values(data).reduce((a, b) => a + b, 0);
  if (total === 0) return null;

  return (
    <div className="flex flex-col items-center gap-4">
      <ResponsiveContainer width="100%" height={220}>
        <PieChart>
          <Pie
            data={chartData}
            dataKey="value"
            nameKey="label"
            cx="50%" cy="50%"
            innerRadius={68}
            outerRadius={98}
            paddingAngle={3}
            strokeWidth={0}
            animationBegin={0}
            animationDuration={900}
          >
            {chartData.map((entry) => (
              <Cell key={entry.key} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              borderRadius: 12,
              fontSize: 12,
              border: "1px solid var(--border)",
              background: "var(--surface)",
              color: "var(--text)",
              boxShadow: "0 4px 20px rgba(0,0,0,0.08)",
            }}
            formatter={(v: number, _: string, entry: { payload?: { emoji?: string; label?: string } }) => [
              `${entry.payload?.emoji ?? ""} ${v} yorum (${((v / total) * 100).toFixed(1)}%)`,
              entry.payload?.label ?? "",
            ]}
          />
        </PieChart>
      </ResponsiveContainer>

      <div className="flex flex-wrap justify-center gap-x-4 gap-y-2">
        {EMOTIONS.filter(e => data[e.key] > 0).map((e, i) => (
          <motion.div
            key={e.key}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 + i * 0.08 }}
            className="flex flex-col items-center gap-0.5"
          >
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full" style={{ background: e.color }} />
              <span className="text-xs text-[var(--text-muted)] font-medium">
                {e.emoji} {e.label}
              </span>
            </div>
            <span className="text-sm font-bold text-[var(--text)]">
              %{((data[e.key] / total) * 100).toFixed(0)}
            </span>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
