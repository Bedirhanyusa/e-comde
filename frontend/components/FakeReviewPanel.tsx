"use client";

import { ShieldAlert, ShieldCheck, AlertCircle, ChevronDown, ChevronUp, Copy } from "lucide-react";
import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { ReviewResult } from "@/lib/types";
import { groupFlaggedByReason } from "@/lib/analysis";

interface SuspiciousReview {
  review: ReviewResult;
  reasons: string[];
}

function detectSuspicious(reviews: ReviewResult[]): SuspiciousReview[] {
  const texts = reviews.map(r => r.text.trim().toLowerCase());
  const results: SuspiciousReview[] = [];

  reviews.forEach((r, i) => {
    const reasons: string[] = [];
    const text = r.text.trim();

    if (r.confidence < 0.70) reasons.push("Düşük güven skoru");
    if (text.length < 15) reasons.push("Çok kısa yorum");
    if (text === text.toUpperCase() && text.length > 5) reasons.push("Tamamı büyük harf");
    if (/^[\W\d\s]+$/.test(text)) reasons.push("Anlamsız içerik");

    const dupIdx = texts.findIndex((t, j) => j !== i && t === texts[i]);
    if (dupIdx !== -1 && dupIdx < i) reasons.push("Tekrarlayan yorum");

    const wordCount = text.split(/\s+/).filter(Boolean).length;
    if (wordCount <= 2) reasons.push("Çok az kelime");

    if (reasons.length > 0) results.push({ review: r, reasons });
  });

  return results;
}

function getDuplicateClusters(reviews: ReviewResult[]): Array<{ text: string; count: number }> {
  const freq: Record<string, number> = {};
  for (const r of reviews) {
    const key = r.text.trim().toLowerCase();
    freq[key] = (freq[key] || 0) + 1;
  }
  return Object.entries(freq)
    .filter(([, c]) => c >= 2)
    .map(([text, count]) => ({ text, count }))
    .sort((a, b) => b.count - a.count);
}

interface Props {
  reviews: ReviewResult[];
  flaggedCount: number;
}

export function FakeReviewPanel({ reviews, flaggedCount }: Props) {
  const [expanded, setExpanded] = useState(false);
  const suspicious = detectSuspicious(reviews);
  const total = reviews.length;
  const pct = total > 0 ? Math.round((suspicious.length / total) * 100) : 0;
  const riskLevel = pct >= 20 ? "yüksek" : pct >= 10 ? "orta" : "düşük";
  const riskColor = pct >= 20
    ? "border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20"
    : pct >= 10
    ? "border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-900/20"
    : "border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-900/20";

  const iconColor = pct >= 20 ? "text-red-500" : pct >= 10 ? "text-amber-500" : "text-emerald-500";
  const labelColor = pct >= 20
    ? "text-red-700 dark:text-red-300"
    : pct >= 10
    ? "text-amber-700 dark:text-amber-300"
    : "text-emerald-700 dark:text-emerald-300";

  const barColor = pct >= 20 ? "#ef4444" : pct >= 10 ? "#f59e0b" : "#22c55e";

  const reasonGroups = groupFlaggedByReason(suspicious);
  const duplicateClusters = getDuplicateClusters(reviews);

  // Otomatik özet metni
  const topReason = reasonGroups[0];
  const summaryText = suspicious.length > 0
    ? `Bu üründe %${pct} oranında şüpheli yorum tespit edildi.${topReason ? ` En yaygın pattern: ${topReason.reason.toLowerCase()} (${topReason.count} adet).` : ""}`
    : "";

  if (suspicious.length === 0) {
    return (
      <div className="flex items-center gap-3 px-5 py-4 rounded-2xl border border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-900/20">
        <ShieldCheck className="w-5 h-5 text-emerald-500 shrink-0" />
        <div>
          <p className="text-sm font-semibold text-emerald-700 dark:text-emerald-300">Şüpheli yorum tespit edilmedi</p>
          <p className="text-xs text-emerald-600/70 dark:text-emerald-400/70 mt-0.5">{total} yorum analiz edildi, tümü güvenilir görünüyor.</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`rounded-3xl border ${riskColor} overflow-hidden`}>
      {/* Header */}
      <button
        onClick={() => setExpanded(v => !v)}
        className="w-full flex items-center gap-4 px-6 py-4 text-left"
      >
        <ShieldAlert className={`w-5 h-5 shrink-0 ${iconColor}`} />
        <div className="flex-1 min-w-0">
          <p className={`text-sm font-bold ${labelColor}`}>
            {suspicious.length} şüpheli yorum tespit edildi
            <span className="ml-2 font-normal opacity-70">({pct}% · Risk: {riskLevel})</span>
          </p>
          <p className="text-xs text-[var(--text-muted)] mt-0.5">{summaryText}</p>
        </div>
        {/* Risk bar */}
        <div className="hidden sm:flex flex-col items-end gap-1 shrink-0 mr-2">
          <div className="w-24 h-1.5 rounded-full bg-black/10 dark:bg-white/10 overflow-hidden">
            <div
              className={`h-full rounded-full ${pct >= 20 ? "bg-red-500" : pct >= 10 ? "bg-amber-500" : "bg-emerald-500"}`}
              style={{ width: `${Math.min(pct * 3, 100)}%` }}
            />
          </div>
          <span className="text-xs text-[var(--text-muted)]">{pct}% şüpheli</span>
        </div>
        {expanded ? <ChevronUp className="w-4 h-4 text-[var(--text-muted)] shrink-0" /> : <ChevronDown className="w-4 h-4 text-[var(--text-muted)] shrink-0" />}
      </button>

      {/* Expanded content */}
      {expanded && (
        <div className="border-t border-black/5 dark:border-white/5 px-6 pb-6 pt-4 space-y-5">

          {/* Sebep Dağılımı Grafiği */}
          {reasonGroups.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-[var(--text)] mb-3">Şüphe Sebebi Dağılımı</p>
              <ResponsiveContainer width="100%" height={Math.max(reasonGroups.length * 36, 80)}>
                <BarChart
                  data={reasonGroups}
                  layout="vertical"
                  margin={{ top: 0, right: 16, bottom: 0, left: 8 }}
                >
                  <XAxis type="number" hide />
                  <YAxis
                    type="category"
                    dataKey="reason"
                    width={148}
                    tick={{ fontSize: 11, fill: "var(--text-muted)" }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      borderRadius: 10,
                      fontSize: 12,
                      border: "1px solid var(--border)",
                      background: "var(--surface)",
                      color: "var(--text)",
                    }}
                    formatter={(v: number) => [`${v} yorum`, "Adet"]}
                  />
                  <Bar dataKey="count" radius={[0, 6, 6, 0]} maxBarSize={18}>
                    {reasonGroups.map((_, i) => (
                      <Cell key={i} fill={barColor} fillOpacity={0.75 - i * 0.08} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Kopya Yorum Kümeleri */}
          {duplicateClusters.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-[var(--text)] mb-2 flex items-center gap-1.5">
                <Copy className="w-3.5 h-3.5 text-amber-500" />
                Tekrarlayan Yorum Kümeleri ({duplicateClusters.length})
              </p>
              <div className="space-y-2">
                {duplicateClusters.slice(0, 5).map(({ text, count }, i) => (
                  <div key={i} className="flex items-start gap-2.5 p-3 rounded-xl bg-white/60 dark:bg-black/10">
                    <span className="shrink-0 text-xs font-bold px-2 py-0.5 rounded-full bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300">
                      {count}×
                    </span>
                    <p className="text-xs text-[var(--text)] line-clamp-2 leading-relaxed">
                      &ldquo;{text}&rdquo;
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Şüpheli Yorum Listesi */}
          <div>
            <p className="text-xs font-semibold text-[var(--text)] mb-2">Şüpheli Yorumlar</p>
            <div className="space-y-2 max-h-52 overflow-y-auto">
              {suspicious.map(({ review, reasons }, i) => (
                <div key={i} className="flex items-start gap-3 p-3 rounded-2xl bg-white/60 dark:bg-black/10">
                  <AlertCircle className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-[var(--text)] leading-snug line-clamp-2">{review.text}</p>
                    <div className="flex flex-wrap gap-1.5 mt-2">
                      {reasons.map((r, j) => (
                        <span key={j} className="text-xs px-2 py-0.5 rounded-full bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300 font-medium">
                          {r}
                        </span>
                      ))}
                      <span className="text-xs px-2 py-0.5 rounded-full bg-[var(--border)] text-[var(--text-muted)] font-medium">
                        %{Math.round(review.confidence * 100)} güven
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
