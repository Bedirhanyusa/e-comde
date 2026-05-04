"use client";

import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";
import { ReviewResult } from "@/lib/types";

const COLORS: Record<string, string> = {
  olumlu: "#22c55e",
  olumsuz: "#ef4444",
  notr:    "#9CA3AF",
};

const LABELS: Record<string, string> = {
  olumlu: "Pozitif",
  olumsuz: "Negatif",
  notr: "Nötr",
};

interface Props {
  results: ReviewResult[];
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

  return (
    <div className="flex flex-col items-center gap-4">
      <ResponsiveContainer width="100%" height={220}>
        <PieChart>
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            cx="50%" cy="50%"
            innerRadius={65}
            outerRadius={100}
            paddingAngle={3}
            strokeWidth={0}
          >
            {data.map((entry) => (
              <Cell key={entry.key} fill={COLORS[entry.key]} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{ borderRadius: 10, fontSize: 12, border: "1px solid #e5e7eb" }}
            formatter={(v: number) => [`${v} yorum (${((v / total) * 100).toFixed(1)}%)`, ""]}
          />
        </PieChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex items-center gap-5">
        {data.map(d => (
          <div key={d.key} className="flex items-center gap-1.5 text-sm text-[var(--text-muted)]">
            <span className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: COLORS[d.key] }} />
            <span className="font-medium">{d.name}</span>
            <span className="text-xs font-bold text-[var(--text)]">
              %{((d.value / total) * 100).toFixed(0)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
