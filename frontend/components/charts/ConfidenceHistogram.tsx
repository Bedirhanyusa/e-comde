"use client";

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { ReviewResult } from "@/lib/types";

interface Props { results: ReviewResult[] }

function binColor(range: string): string {
  const low = parseInt(range.split("-")[0]);
  if (low >= 80) return "#22c55e";
  if (low >= 60) return "#f59e0b";
  return "#9CA3AF";
}

export function ConfidenceHistogram({ results }: Props) {
  const bins = Array.from({ length: 10 }, (_, i) => ({
    range: `${i * 10}-${(i + 1) * 10}%`,
    count: 0,
  }));

  results.forEach((r) => {
    const idx = Math.min(Math.floor(r.confidence * 10), 9);
    bins[idx].count++;
  });

  const visible = bins.filter(b => b.count > 0).length > 0 ? bins : bins;

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={visible} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}
        barCategoryGap="20%">
        <XAxis
          dataKey="range"
          tick={{ fontSize: 9, fill: "var(--text-muted)" }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          tick={{ fontSize: 10, fill: "var(--text-muted)" }}
          axisLine={false}
          tickLine={false}
          width={30}
        />
        <Tooltip
          cursor={{ fill: "rgba(124,58,237,0.06)", radius: 6 }}
          contentStyle={{
            borderRadius: 10,
            fontSize: 12,
            border: "1px solid var(--border)",
            background: "var(--surface)",
            color: "var(--text)",
            boxShadow: "0 4px 20px rgba(0,0,0,0.08)",
          }}
          formatter={(v: number) => [`${v} yorum`, "Sayı"]}
        />
        <Bar dataKey="count" radius={[5, 5, 0, 0]} animationDuration={800}>
          {visible.map((entry, i) => (
            <Cell key={i} fill={binColor(entry.range)} fillOpacity={0.85} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
