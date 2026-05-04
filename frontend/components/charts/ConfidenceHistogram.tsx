"use client";

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { ReviewResult } from "@/lib/types";

interface Props {
  results: ReviewResult[];
}

export function ConfidenceHistogram({ results }: Props) {
  const bins = Array.from({ length: 10 }, (_, i) => ({
    range: `${(i * 10).toString().padStart(2, "0")}-${((i + 1) * 10).toString().padStart(2, "0")}%`,
    count: 0,
  }));

  results.forEach((r) => {
    const idx = Math.min(Math.floor(r.confidence * 10), 9);
    bins[idx].count++;
  });

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={bins} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="range" tick={{ fontSize: 10 }} />
        <YAxis tick={{ fontSize: 11 }} />
        <Tooltip
          cursor={{ fill: "rgba(124,58,237,0.08)" }}
          contentStyle={{ borderRadius: 8, fontSize: 12 }}
          formatter={(v: number) => [v, "Yorum"]}
        />
        <Bar dataKey="count" fill="#7C3AED" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
