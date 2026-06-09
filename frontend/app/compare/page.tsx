"use client";

import { useState, useCallback } from "react";
import {
  Link2, Loader2, CheckCircle2, XCircle, AlertTriangle, Scale,
  TrendingUp, ShieldAlert, Star, BarChart3, Users, Zap,
} from "lucide-react";
import { motion } from "framer-motion";
import { Navbar } from "@/components/Navbar";
import { analyzeByUrl, getAdvisor, summarizeReviews } from "@/lib/api";
import {
  calcSentimentScore, calcCategoryScores, calcQualityStats, calcReviewQuality,
} from "@/lib/analysis";
import {
  SentimentResponse, SummarizeResponse, AdvisorResponse, ShopAnalysisResponse,
} from "@/lib/types";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, Legend,
} from "recharts";

// ── Tipler ──────────────────────────────────────────────────────────────────

interface ProductAnalysis {
  sentimentData: SentimentResponse;
  summaryData: SummarizeResponse | null;
  advisorData: AdvisorResponse | null;
  score: number;
  url: string;
  name?: string;
  categoryScores: Array<{ name: string; score: number }>;
  avgConfidence: number;
  fakeReviewPct: number;
  qualityPct: number;
  sentimentBreakdown: { olumlu: number; notr: number; olumsuz: number };
}

// ── Yardımcılar ─────────────────────────────────────────────────────────────

function detectFakePct(results: SentimentResponse["results"]): number {
  const texts = results.map(r => r.text.trim().toLowerCase());
  let count = 0;
  results.forEach((r, i) => {
    const text = r.text.trim();
    const isDup = texts.findIndex((t, j) => j !== i && t === texts[i]) !== -1;
    if (
      r.confidence < 0.70 ||
      text.length < 15 ||
      text.split(/\s+/).filter(Boolean).length <= 2 ||
      isDup
    ) count++;
  });
  return results.length > 0 ? Math.round((count / results.length) * 100) : 0;
}

function avgConf(results: SentimentResponse["results"]): number {
  if (!results.length) return 0;
  return results.reduce((s, r) => s + r.confidence, 0) / results.length;
}

// ── Mini bileşenler ──────────────────────────────────────────────────────────

function VerdictBadge({ verdict }: { verdict: string }) {
  const cfg: Record<string, { icon: typeof CheckCircle2; cls: string }> = {
    "Tavsiye Edilir":  { icon: CheckCircle2,  cls: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300" },
    "Dikkatli Olun":   { icon: AlertTriangle, cls: "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300" },
    "Tavsiye Edilmez": { icon: XCircle,       cls: "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300" },
  };
  const { icon: Icon, cls } = cfg[verdict] ?? { icon: AlertTriangle, cls: "bg-gray-100 text-gray-700" };
  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-bold ${cls}`}>
      <Icon className="w-4 h-4" /> {verdict}
    </span>
  );
}

function ScorePill({ score, best }: { score: number; best: boolean }) {
  const color = score >= 4 ? "text-emerald-600" : score >= 3 ? "text-amber-600" : "text-red-500";
  return (
    <span className={`text-2xl font-extrabold ${color} ${best ? "underline decoration-dotted underline-offset-4" : ""}`}>
      {score.toFixed(2)}
      {best && <span className="text-xs ml-1">🏆</span>}
    </span>
  );
}

function PctBar({ pct, color }: { pct: number; color: string }) {
  return (
    <div className="w-full h-1.5 rounded-full bg-black/10 dark:bg-white/10 overflow-hidden">
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${pct}%` }}
        transition={{ duration: 0.7, ease: "easeOut" }}
        className="h-full rounded-full"
        style={{ background: color }}
      />
    </div>
  );
}

// ── Baş-başa metrik tablosu ──────────────────────────────────────────────────

interface MetricRow {
  label: string;
  icon: typeof TrendingUp;
  valA: string | React.ReactNode;
  valB: string | React.ReactNode;
  winnerA?: boolean; // A kazandı mı
  winnerB?: boolean; // B kazandı mı
  neutral?: boolean;
}

function HeadToHead({ a, b }: { a: ProductAnalysis; b: ProductAnalysis }) {
  const verdictScore = (v: string) =>
    v === "Tavsiye Edilir" ? 3 : v === "Dikkatli Olun" ? 2 : 1;

  const rows: MetricRow[] = [
    {
      label: "BERTurk Sentiment Skoru",
      icon: TrendingUp,
      valA: <ScorePill score={a.score} best={a.score > b.score} />,
      valB: <ScorePill score={b.score} best={b.score > a.score} />,
      winnerA: a.score > b.score + 0.1,
      winnerB: b.score > a.score + 0.1,
      neutral: Math.abs(a.score - b.score) <= 0.1,
    },
    {
      label: "Pozitif Yorum Oranı",
      icon: Star,
      valA: `%${Math.round((a.sentimentBreakdown.olumlu / a.sentimentData.total) * 100)}`,
      valB: `%${Math.round((b.sentimentBreakdown.olumlu / b.sentimentData.total) * 100)}`,
      winnerA: a.sentimentBreakdown.olumlu / a.sentimentData.total > b.sentimentBreakdown.olumlu / b.sentimentData.total + 0.03,
      winnerB: b.sentimentBreakdown.olumlu / b.sentimentData.total > a.sentimentBreakdown.olumlu / a.sentimentData.total + 0.03,
      neutral: Math.abs(a.sentimentBreakdown.olumlu / a.sentimentData.total - b.sentimentBreakdown.olumlu / b.sentimentData.total) <= 0.03,
    },
    {
      label: "Ortalama Model Güveni",
      icon: Zap,
      valA: `%${Math.round(a.avgConfidence * 100)}`,
      valB: `%${Math.round(b.avgConfidence * 100)}`,
      winnerA: a.avgConfidence > b.avgConfidence + 0.02,
      winnerB: b.avgConfidence > a.avgConfidence + 0.02,
      neutral: Math.abs(a.avgConfidence - b.avgConfidence) <= 0.02,
    },
    {
      label: "Şüpheli Yorum Oranı",
      icon: ShieldAlert,
      valA: `%${a.fakeReviewPct}`,
      valB: `%${b.fakeReviewPct}`,
      winnerA: a.fakeReviewPct < b.fakeReviewPct - 2,
      winnerB: b.fakeReviewPct < a.fakeReviewPct - 2,
      neutral: Math.abs(a.fakeReviewPct - b.fakeReviewPct) <= 2,
    },
    {
      label: "Yüksek Kaliteli Yorum",
      icon: BarChart3,
      valA: `%${a.qualityPct}`,
      valB: `%${b.qualityPct}`,
      winnerA: a.qualityPct > b.qualityPct + 3,
      winnerB: b.qualityPct > a.qualityPct + 3,
      neutral: Math.abs(a.qualityPct - b.qualityPct) <= 3,
    },
    {
      label: "Toplam Yorum",
      icon: Users,
      valA: a.sentimentData.total.toLocaleString("tr-TR"),
      valB: b.sentimentData.total.toLocaleString("tr-TR"),
      neutral: true,
    },
    ...(a.advisorData && b.advisorData ? [{
      label: "Yapay Zeka Kararı",
      icon: CheckCircle2,
      valA: <VerdictBadge verdict={a.advisorData.verdict} />,
      valB: <VerdictBadge verdict={b.advisorData.verdict} />,
      winnerA: verdictScore(a.advisorData.verdict) > verdictScore(b.advisorData.verdict),
      winnerB: verdictScore(b.advisorData.verdict) > verdictScore(a.advisorData.verdict),
      neutral: verdictScore(a.advisorData.verdict) === verdictScore(b.advisorData.verdict),
    }] : []),
  ];

  const winsA = rows.filter(r => r.winnerA).length;
  const winsB = rows.filter(r => r.winnerB).length;

  return (
    <div className="bg-white dark:bg-[var(--surface)] border border-[var(--border)] rounded-3xl overflow-hidden shadow-sm">
      {/* Başlık */}
      <div className="px-6 py-4 border-b border-[var(--border)] flex items-center justify-between">
        <h2 className="font-bold text-base text-[var(--text)]">Metrik Karşılaştırması</h2>
        <p className="text-xs text-[var(--text-muted)]">BERTurk temel model · Claude AI yardımcı</p>
      </div>

      {/* Başlık satırı */}
      <div className="grid grid-cols-[1fr_1fr_1fr] border-b border-[var(--border)] bg-[var(--bg)]">
        <div className="px-4 py-2.5 text-xs font-bold text-[var(--text-muted)] uppercase tracking-wide">Metrik</div>
        <div className="px-4 py-2.5 text-xs font-bold text-violet-600 uppercase tracking-wide border-l border-[var(--border)] text-center">
          Ürün A <span className="text-[var(--text-muted)] font-normal ml-1">({winsA} kazanım)</span>
        </div>
        <div className="px-4 py-2.5 text-xs font-bold text-indigo-600 uppercase tracking-wide border-l border-[var(--border)] text-center">
          Ürün B <span className="text-[var(--text-muted)] font-normal ml-1">({winsB} kazanım)</span>
        </div>
      </div>

      {/* Satırlar */}
      {rows.map((row, i) => {
        const Icon = row.icon;
        return (
          <div key={i} className={`grid grid-cols-[1fr_1fr_1fr] border-b border-[var(--border)] last:border-0 ${i % 2 === 0 ? "" : "bg-[var(--bg)]/40"}`}>
            <div className="px-4 py-3.5 flex items-center gap-2">
              <Icon className="w-3.5 h-3.5 text-violet-500 shrink-0" />
              <span className="text-xs font-medium text-[var(--text)]">{row.label}</span>
            </div>
            <div className={`px-4 py-3.5 border-l border-[var(--border)] flex items-center justify-center ${row.winnerA ? "bg-emerald-50 dark:bg-emerald-900/15" : ""}`}>
              <span className="text-sm font-bold text-[var(--text)]">{row.valA}</span>
              {row.winnerA && <span className="ml-1.5 text-emerald-500 text-xs">✓</span>}
            </div>
            <div className={`px-4 py-3.5 border-l border-[var(--border)] flex items-center justify-center ${row.winnerB ? "bg-emerald-50 dark:bg-emerald-900/15" : ""}`}>
              <span className="text-sm font-bold text-[var(--text)]">{row.valB}</span>
              {row.winnerB && <span className="ml-1.5 text-emerald-500 text-xs">✓</span>}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Duygu Dağılımı yan yana ─────────────────────────────────────────────────

function SentimentCompareChart({ a, b }: { a: ProductAnalysis; b: ProductAnalysis }) {
  const toRow = (p: ProductAnalysis) => {
    const t = p.sentimentData.total || 1;
    return {
      olumlu: Math.round((p.sentimentBreakdown.olumlu / t) * 100),
      notr: Math.round((p.sentimentBreakdown.notr / t) * 100),
      olumsuz: Math.round((p.sentimentBreakdown.olumsuz / t) * 100),
    };
  };
  const ra = toRow(a);
  const rb = toRow(b);

  const rows = [
    { label: "Pozitif", key: "olumlu" as const, color: "#22c55e" },
    { label: "Nötr",    key: "notr"    as const, color: "#9CA3AF" },
    { label: "Negatif", key: "olumsuz" as const, color: "#ef4444" },
  ];

  return (
    <div className="bg-white dark:bg-[var(--surface)] border border-[var(--border)] rounded-3xl p-6 shadow-sm">
      <h2 className="font-bold text-base text-[var(--text)] mb-5">BERTurk Duygu Dağılımı</h2>
      <div className="space-y-5">
        {rows.map(({ label, key, color }) => (
          <div key={key}>
            <div className="flex justify-between mb-1.5">
              <span className="text-xs font-semibold text-[var(--text-muted)]">{label}</span>
              <div className="flex gap-4">
                <span className="text-xs font-bold text-violet-600">A: %{ra[key]}</span>
                <span className="text-xs font-bold text-indigo-600">B: %{rb[key]}</span>
              </div>
            </div>
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <span className="text-xs text-[var(--text-muted)] w-6">A</span>
                <div className="flex-1 h-2 rounded-full bg-black/10 dark:bg-white/10 overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${ra[key]}%` }}
                    transition={{ duration: 0.7, ease: "easeOut" }}
                    className="h-full rounded-full opacity-80"
                    style={{ background: color }}
                  />
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-[var(--text-muted)] w-6">B</span>
                <div className="flex-1 h-2 rounded-full bg-black/10 dark:bg-white/10 overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${rb[key]}%` }}
                    transition={{ duration: 0.7, ease: "easeOut", delay: 0.1 }}
                    className="h-full rounded-full opacity-60"
                    style={{ background: color }}
                  />
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Kategori skoru karşılaştırması ──────────────────────────────────────────

function CategoryCompareChart({ a, b }: { a: ProductAnalysis; b: ProductAnalysis }) {
  const allCats = Array.from(new Set([
    ...a.categoryScores.map(c => c.name),
    ...b.categoryScores.map(c => c.name),
  ]));

  if (allCats.length === 0) return null;

  const data = allCats.map(name => ({
    name: name.split("&")[0].trim(),
    A: a.categoryScores.find(c => c.name === name)?.score ?? 0,
    B: b.categoryScores.find(c => c.name === name)?.score ?? 0,
  }));

  return (
    <div className="bg-white dark:bg-[var(--surface)] border border-[var(--border)] rounded-3xl p-6 shadow-sm">
      <h2 className="font-bold text-base text-[var(--text)] mb-1">Kategori Bazlı Karşılaştırma</h2>
      <p className="text-xs text-[var(--text-muted)] mb-5">BERTurk anahtar kelime eşleşmesi ile hesaplanır (1–5 ölçeği)</p>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 0, right: 8, left: -20, bottom: 0 }}>
          <XAxis dataKey="name" tick={{ fontSize: 10, fill: "var(--text-muted)" }} axisLine={false} tickLine={false} />
          <YAxis domain={[0, 5]} tick={{ fontSize: 10, fill: "var(--text-muted)" }} axisLine={false} tickLine={false} />
          <Tooltip
            contentStyle={{
              borderRadius: 12, fontSize: 12,
              border: "1px solid var(--border)",
              background: "var(--surface)",
              color: "var(--text)",
            }}
            formatter={(v: number, name: string) => [`${v.toFixed(2)} / 5`, `Ürün ${name}`]}
          />
          <Legend formatter={(v) => `Ürün ${v}`} wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="A" fill="#7C3AED" fillOpacity={0.8} radius={[4, 4, 0, 0]} maxBarSize={28} />
          <Bar dataKey="B" fill="#6366F1" fillOpacity={0.6} radius={[4, 4, 0, 0]} maxBarSize={28} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Radar grafiği (genel profil) ────────────────────────────────────────────

function RadarCompare({ a, b }: { a: ProductAnalysis; b: ProductAnalysis }) {
  const data = [
    { metric: "Sentiment", A: a.score, B: b.score },
    { metric: "Güven", A: a.avgConfidence * 5, B: b.avgConfidence * 5 },
    { metric: "Kalite", A: (a.qualityPct / 100) * 5, B: (b.qualityPct / 100) * 5 },
    { metric: "Güvenilirlik", A: ((100 - a.fakeReviewPct) / 100) * 5, B: ((100 - b.fakeReviewPct) / 100) * 5 },
    { metric: "Pozitiflik", A: (a.sentimentBreakdown.olumlu / a.sentimentData.total) * 5, B: (b.sentimentBreakdown.olumlu / b.sentimentData.total) * 5 },
  ];

  return (
    <div className="bg-white dark:bg-[var(--surface)] border border-[var(--border)] rounded-3xl p-6 shadow-sm">
      <h2 className="font-bold text-base text-[var(--text)] mb-1">Genel Profil Karşılaştırması</h2>
      <p className="text-xs text-[var(--text-muted)] mb-4">BERTurk modelinden türetilen çok boyutlu skor (0–5)</p>
      <ResponsiveContainer width="100%" height={240}>
        <RadarChart data={data}>
          <PolarGrid stroke="var(--border)" />
          <PolarAngleAxis dataKey="metric" tick={{ fontSize: 10, fill: "var(--text-muted)" }} />
          <Radar name="Ürün A" dataKey="A" stroke="#7C3AED" fill="#7C3AED" fillOpacity={0.25} strokeWidth={2} />
          <Radar name="Ürün B" dataKey="B" stroke="#6366F1" fill="#6366F1" fillOpacity={0.15} strokeWidth={2} strokeDasharray="4 2" />
          <Legend formatter={(v) => v} wrapperStyle={{ fontSize: 11 }} />
          <Tooltip
            contentStyle={{ borderRadius: 10, fontSize: 11, border: "1px solid var(--border)", background: "var(--surface)", color: "var(--text)" }}
            formatter={(v: number, name: string) => [`${v.toFixed(2)}`, name]}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Detay kartları ──────────────────────────────────────────────────────────

function DetailColumn({ data, label, accentColor }: { data: ProductAnalysis; label: string; accentColor: string }) {
  const pros = data.summaryData?.pros ?? [];
  const cons = data.summaryData?.cons ?? [];

  return (
    <div className="space-y-4">
      {/* Başlık */}
      <div className="bg-white dark:bg-[var(--surface)] border border-[var(--border)] rounded-3xl p-5">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-2.5 h-2.5 rounded-full" style={{ background: accentColor }} />
          <p className="text-xs font-bold uppercase tracking-wide" style={{ color: accentColor }}>{label}</p>
        </div>
        {data.name && <p className="text-sm font-semibold text-[var(--text)] mb-3 line-clamp-2">{data.name}</p>}

        {/* Sentiment mini bar */}
        {(() => {
          const t = data.sentimentData.total || 1;
          const oPct = (data.sentimentBreakdown.olumlu / t) * 100;
          const nPct = (data.sentimentBreakdown.notr / t) * 100;
          const uPct = (data.sentimentBreakdown.olumsuz / t) * 100;
          return (
            <div className="overflow-hidden rounded-full h-2 flex gap-0.5">
              <div className="h-full bg-emerald-500 rounded-l-full" style={{ width: `${oPct}%` }} />
              <div className="h-full bg-gray-400" style={{ width: `${nPct}%` }} />
              <div className="h-full bg-red-500 rounded-r-full" style={{ width: `${uPct}%` }} />
            </div>
          );
        })()}
        <div className="flex justify-between mt-1.5">
          <span className="text-xs text-emerald-600">%{Math.round((data.sentimentBreakdown.olumlu / (data.sentimentData.total || 1)) * 100)} pozitif</span>
          <span className="text-xs text-red-500">%{Math.round((data.sentimentBreakdown.olumsuz / (data.sentimentData.total || 1)) * 100)} negatif</span>
        </div>
      </div>

      {/* Advisor */}
      {data.advisorData && (
        <div className="bg-white dark:bg-[var(--surface)] border border-[var(--border)] rounded-3xl p-5">
          <p className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wide mb-2">Yapay Zeka Kararı</p>
          <VerdictBadge verdict={data.advisorData.verdict} />
          <p className="text-sm text-[var(--text)] mt-3 leading-relaxed">{data.advisorData.target_audience}</p>
          {data.advisorData.buy_if.length > 0 && (
            <div className="mt-3">
              <p className="text-xs text-emerald-600 font-semibold mb-1">Şu durumlarda al</p>
              <ul className="space-y-1">
                {data.advisorData.buy_if.map((b, i) => (
                  <li key={i} className="text-xs text-[var(--text)] flex items-start gap-1.5">
                    <CheckCircle2 className="w-3 h-3 text-emerald-500 shrink-0 mt-0.5" /> {b}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {data.advisorData.watch_out.length > 0 && (
            <div className="mt-3">
              <p className="text-xs text-red-500 font-semibold mb-1">Dikkat et</p>
              <ul className="space-y-1">
                {data.advisorData.watch_out.map((w, i) => (
                  <li key={i} className="text-xs text-[var(--text)] flex items-start gap-1.5">
                    <AlertTriangle className="w-3 h-3 text-amber-500 shrink-0 mt-0.5" /> {w}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* AI Özeti */}
      {data.summaryData?.overall && (
        <div className="bg-white dark:bg-[var(--surface)] border border-[var(--border)] rounded-3xl p-5">
          <p className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wide mb-2">Claude AI Özeti</p>
          <p className="text-sm text-[var(--text)] leading-relaxed">{data.summaryData.overall}</p>
        </div>
      )}

      {/* Artılar / Eksiler */}
      {(pros.length > 0 || cons.length > 0) && (
        <div className="bg-white dark:bg-[var(--surface)] border border-[var(--border)] rounded-3xl p-5 space-y-3">
          {pros.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-emerald-600 uppercase tracking-wide mb-2">Artılar</p>
              <ul className="space-y-1.5">
                {pros.map((p, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-[var(--text)]">
                    <CheckCircle2 className="w-3 h-3 text-emerald-500 shrink-0 mt-0.5" /> {p}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {cons.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-red-500 uppercase tracking-wide mb-2">Eksiler</p>
              <ul className="space-y-1.5">
                {cons.map((c, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-[var(--text)]">
                    <XCircle className="w-3 h-3 text-red-500 shrink-0 mt-0.5" /> {c}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Winner Banner ────────────────────────────────────────────────────────────

function WinnerBanner({ a, b }: { a: ProductAnalysis; b: ProductAnalysis }) {
  const diff = Math.abs(a.score - b.score);
  if (diff < 0.1) {
    return (
      <div className="text-center py-4 px-6 rounded-2xl bg-[var(--surface)] border border-[var(--border)]">
        <p className="text-sm font-bold text-[var(--text)]">⚖️ Eşit Seviyede Ürünler</p>
        <p className="text-xs text-[var(--text-muted)] mt-1">Fark 0.1 puanın altında — detaylara bakın</p>
      </div>
    );
  }
  const winner = a.score > b.score ? { label: "Ürün A", name: a.name } : { label: "Ürün B", name: b.name };
  return (
    <div className="text-center py-4 px-6 rounded-2xl bg-violet-50 dark:bg-violet-900/20 border border-violet-200 dark:border-violet-800">
      <p className="text-sm font-bold text-violet-700 dark:text-violet-300">
        🏆 {winner.label}{winner.name ? ` — ${winner.name}` : ""} daha iyi
      </p>
      <p className="text-xs text-violet-500 dark:text-violet-400 mt-1">
        Fark: {diff.toFixed(2)} puan · BERTurk kalite ağırlıklı skor
      </p>
    </div>
  );
}

// ── Ana sayfa ────────────────────────────────────────────────────────────────

export default function ComparePage() {
  const [urlA, setUrlA] = useState("");
  const [urlB, setUrlB] = useState("");
  const [errorA, setErrorA] = useState<string | null>(null);
  const [errorB, setErrorB] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<[ProductAnalysis, ProductAnalysis] | null>(null);

  const analyzeProduct = useCallback(async (url: string): Promise<ProductAnalysis> => {
    const shopMatch = url.match(/\/shop\/products\/(\d+)/);
    let data: ShopAnalysisResponse;
    if (shopMatch) {
      const { analyzeShopProduct } = await import("@/lib/api");
      data = await analyzeShopProduct(shopMatch[1]);
    } else {
      data = await analyzeByUrl(url);
    }

    const sentimentData: SentimentResponse = {
      results: data.results,
      total: data.total,
      flagged_count: data.flagged_count,
    };
    const score = calcSentimentScore(data.results);
    const texts = data.results.map(r => r.text);
    const labels = data.results.map(r => r.label);

    const [summaryRes, advisorRes] = await Promise.allSettled([
      summarizeReviews(texts, labels),
      getAdvisor(texts, labels, score),
    ]);

    const qs = calcQualityStats(data.results);
    const fakePct = detectFakePct(data.results);

    return {
      sentimentData,
      summaryData: summaryRes.status === "fulfilled" ? summaryRes.value : null,
      advisorData: advisorRes.status === "fulfilled" ? advisorRes.value : null,
      score,
      url,
      name: data.product?.name,
      categoryScores: calcCategoryScores(data.results),
      avgConfidence: avgConf(data.results),
      fakeReviewPct: fakePct,
      qualityPct: qs.total > 0 ? Math.round((qs.highQuality / qs.total) * 100) : 0,
      sentimentBreakdown: {
        olumlu: data.results.filter(r => r.label === "olumlu").length,
        notr: data.results.filter(r => r.label === "notr").length,
        olumsuz: data.results.filter(r => r.label === "olumsuz").length,
      },
    };
  }, []);

  const handleCompare = async () => {
    setErrorA(null);
    setErrorB(null);
    if (!urlA.trim() || !urlB.trim()) return;
    setIsLoading(true);
    setResults(null);

    const [resA, resB] = await Promise.allSettled([
      analyzeProduct(urlA.trim()),
      analyzeProduct(urlB.trim()),
    ]);

    if (resA.status === "rejected") setErrorA("Bu link analiz edilemedi.");
    if (resB.status === "rejected") setErrorB("Bu link analiz edilemedi.");
    if (resA.status === "fulfilled" && resB.status === "fulfilled") {
      setResults([resA.value, resB.value]);
    }
    setIsLoading(false);
  };

  return (
    <div className="min-h-screen bg-[var(--bg)]">
      <Navbar />
      <main className="max-w-6xl mx-auto px-4 sm:px-6 py-10">

        {/* Başlık */}
        <div className="flex items-center gap-3 mb-8">
          <div className="p-2.5 rounded-2xl bg-violet-100 dark:bg-violet-900/30">
            <Scale className="w-5 h-5 text-violet-600 dark:text-violet-400" />
          </div>
          <div>
            <h1 className="text-2xl font-extrabold text-[var(--text)]">Ürün Karşılaştırma</h1>
            <p className="text-sm text-[var(--text-muted)] mt-0.5">
              BERTurk sentiment modeli ile derinlemesine karşılaştır · Claude AI yardımcı
            </p>
          </div>
        </div>

        {/* URL Girişleri */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          {[
            { label: "Ürün A", url: urlA, setUrl: setUrlA, error: errorA, color: "violet" },
            { label: "Ürün B", url: urlB, setUrl: setUrlB, error: errorB, color: "indigo" },
          ].map(({ label, url, setUrl, error, color }, idx) => (
            <div key={idx}>
              <label className={`text-xs font-bold text-${color}-600 uppercase tracking-wide mb-2 block`}>{label}</label>
              <div className="flex gap-2 bg-white dark:bg-[var(--surface)] border border-[var(--border)] rounded-2xl p-2 focus-within:border-violet-400 transition-colors">
                <div className="flex items-center pl-2 shrink-0">
                  <Link2 className="w-4 h-4 text-violet-500" />
                </div>
                <input
                  type="url"
                  value={url}
                  onChange={e => setUrl(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && handleCompare()}
                  placeholder="İstünShop veya Trendyol ürün linki..."
                  className="flex-1 bg-transparent text-sm text-[var(--text)] placeholder:text-[var(--text-muted)] outline-none py-2 px-1"
                />
              </div>
              {error && <p className="mt-1 text-xs text-red-500">{error}</p>}
            </div>
          ))}
        </div>

        <div className="flex justify-center mb-8">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
            onClick={handleCompare}
            disabled={!urlA.trim() || !urlB.trim() || isLoading}
            className="flex items-center gap-2 px-8 py-3 rounded-2xl text-sm font-bold text-white bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-700 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-sm"
          >
            {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Scale className="w-4 h-4" />}
            {isLoading ? "Analiz ediliyor..." : "Karşılaştır"}
          </motion.button>
        </div>

        {/* Yükleniyor */}
        {isLoading && (
          <div className="flex flex-col items-center gap-4 py-16">
            <div className="relative w-16 h-16 flex items-center justify-center">
              <div className="absolute inset-0 rounded-full border-4 border-violet-100 dark:border-violet-900/40" />
              <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-violet-600 animate-spin" />
              <Loader2 className="w-6 h-6 text-violet-600 animate-spin" />
            </div>
            <div className="text-center">
              <p className="text-sm font-bold text-[var(--text)]">BERTurk analiz ediyor</p>
              <p className="text-xs text-[var(--text-muted)] mt-1 animate-pulse">Her iki ürün paralel işleniyor...</p>
            </div>
          </div>
        )}

        {/* Sonuçlar */}
        {results && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="space-y-6"
          >
            <WinnerBanner a={results[0]} b={results[1]} />
            <HeadToHead a={results[0]} b={results[1]} />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <SentimentCompareChart a={results[0]} b={results[1]} />
              <RadarCompare a={results[0]} b={results[1]} />
            </div>

            <CategoryCompareChart a={results[0]} b={results[1]} />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <DetailColumn data={results[0]} label="Ürün A" accentColor="#7C3AED" />
              <DetailColumn data={results[1]} label="Ürün B" accentColor="#6366F1" />
            </div>
          </motion.div>
        )}
      </main>
    </div>
  );
}
