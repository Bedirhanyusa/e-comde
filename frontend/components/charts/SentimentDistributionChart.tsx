"use client";

import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";
import { motion } from "framer-motion";
import { ReviewResult } from "@/lib/types";

const COLORS: Record<string, string> = {
  olumlu: "#22c55e",
  olumsuz: "#ef4444",
  notr: "#9CA3AF",
};
const LABELS: Record<string, string> = {
  olumlu: "Pozitif",
  olumsuz: "Negatif",
  notr: "Nötr",
};

interface Props { results: ReviewResult[] }

function CenterLabel({ total, dominant }: { total: number; dominant: string }) {
  return (
    <g>
      <text x="50%" y="46%" textAnchor="middle" dominantBaseline="middle"
        style={{ fontSize: 22, fontWeight: 800, fill: COLORS[dominant] || "#7C3AED" }}>
        {total}
      </text>
      <text x="50%" y="58%" textAnchor="middle" dominantBaseline="middle"
        style={{ fontSize: 10, fill: "#9CA3AF", fontWeight: 500 }}>
        yorum
      </text>
    </g>
  );
}

export function SentimentDistributionChart({ results }: Props) {
  const counts = results.reduce<Record<string, number>>((acc, r) => {
    acc[r.label] = (acc[r.label] || 0) + 1;
    return acc;
  }, {});

  const total = results.length;
  const data = (["olumlu", "olumsuz", "notr"] as const)
    .filter(k => counts[k])
    .map(k => ({ key: k, name: LABELS[k], value: counts[k] }));

  const dominant = data.reduce((a, b) => a.value > b.value ? a : b, data[0])?.key ?? "olumlu";

  return (
    <div className="flex flex-col items-center gap-4">
      <ResponsiveContainer width="100%" height={220}>
        <PieChart>
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            cx="50%" cy="50%"
            innerRadius={68}
            outerRadius={98}
            paddingAngle={4}
            strokeWidth={0}
            animationBegin={0}
            animationDuration={900}
          >
            {data.map((entry) => (
              <Cell key={entry.key} fill={COLORS[entry.key]} />
            ))}
            <CenterLabel total={total} dominant={dominant} />
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
            formatter={(v: number) => [`${v} yorum (${((v / total) * 100).toFixed(1)}%)`, ""]}
          />
        </PieChart>
      </ResponsiveContainer>

      {/* Animated legend */}
      <div className="flex items-center gap-5">
        {data.map((d, i) => (
          <motion.div
            key={d.key}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 + i * 0.1 }}
            className="flex flex-col items-center gap-1"
          >
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full" style={{ background: COLORS[d.key] }} />
              <span className="text-xs text-[var(--text-muted)] font-medium">{d.name}</span>
            </div>
            <span className="text-sm font-bold text-[var(--text)]">
              %{((d.value / total) * 100).toFixed(0)}
            </span>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
