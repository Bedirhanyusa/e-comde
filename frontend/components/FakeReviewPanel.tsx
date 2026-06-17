"use client";

import { ShieldAlert, ShieldCheck, AlertCircle, ChevronDown, ChevronUp, Copy, Info } from "lucide-react";
import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { ReviewResult } from "@/lib/types";
import { groupFlaggedByReason } from "@/lib/analysis";

// ── Kural tanımları (akademik dayanaklı) ────────────────────────────────────

interface Rule {
  id: string;
  label: string;
  category: "model" | "linguistic" | "statistical";
  description: string;
  weight: number;
  check: (r: ReviewResult, ctx: ReviewContext) => boolean;
}

interface ReviewContext {
  text: string;
  lower: string;
  wordCount: number;
  freq: Record<string, number>;
  idx: number;
  texts: string[];
}

const GENERIC_PHRASES = [
  "çok güzel", "harika", "süper", "mükemmel", "bayıldım", "tavsiye ederim",
  "teşekkürler", "iyiydi", "güzel", "beğendim", "kaliteli", "sağolun",
];

const RULES: Rule[] = [
  {
    id: "low_confidence",
    label: "Düşük model güveni",
    category: "model",
    description: "BERTurk modeli bu yorumu sınıflandırırken güven skoru %70 altında kaldı. Model belirsizliği, içeriğin tutarsız, anlamsız veya yanıltıcı olabileceğine işaret eder.",
    weight: 3,
    check: (r) => r.confidence < 0.70,
  },
  {
    id: "very_short",
    label: "Çok kısa içerik",
    category: "linguistic",
    description: "12 karakterden kısa yorumlar yeterli bilgi içermez. Araştırmalar, sahte yorumların gerçek yorumlardan daha kısa olduğunu gösterir (Jindal & Liu, 2008).",
    weight: 2,
    check: (_, ctx) => ctx.text.length < 12,
  },
  {
    id: "few_words",
    label: "Yetersiz kelime sayısı",
    category: "linguistic",
    description: "2 kelime veya daha az içeren yorumlar detaylı bir deneyim aktaramaz. Düşük bilgi yoğunluğu sahte yorum göstergesidir.",
    weight: 2,
    check: (_, ctx) => ctx.wordCount <= 2 && ctx.text.length >= 3,
  },
  {
    id: "all_caps",
    label: "Tamamı büyük harf",
    category: "linguistic",
    description: "Tüm harfleri büyük yazılmış yorumlar dikkat çekme amaçlı spam taktiği olabilir. Doğal kullanıcı davranışıyla uyuşmaz.",
    weight: 1,
    check: (_, ctx) => ctx.text.length > 4 && ctx.text === ctx.text.toUpperCase() && /[A-ZÇĞİÖŞÜ]/.test(ctx.text),
  },
  {
    id: "meaningless",
    label: "Anlamsız içerik",
    category: "linguistic",
    description: "Sadece sembol, rakam veya özel karakter içeren yorumlar — gerçek bir ürün deneyimi aktarmaz.",
    weight: 3,
    check: (_, ctx) => /^[\W\d\s]+$/.test(ctx.text) && ctx.text.length > 1,
  },
  {
    id: "duplicate",
    label: "Tekrarlayan yorum",
    category: "statistical",
    description: "Birebir aynı metin birden fazla kez gönderilmiş. Toplu sahte yorum kampanyalarında sıkça görülen bir pattern (Mukherjee et al., 2012).",
    weight: 3,
    check: (_, ctx) => ctx.freq[ctx.lower] >= 2 && ctx.texts.findIndex((t, j) => j < ctx.idx && t === ctx.lower) !== -1,
  },
  {
    id: "char_repeat",
    label: "Aşırı karakter tekrarı",
    category: "linguistic",
    description: "Aynı karakterin 5+ kez art arda tekrarı (örn. 'aaaa', '!!!!!') — doğal yazım davranışıyla bağdaşmaz.",
    weight: 1,
    check: (_, ctx) => /(.)\1{4,}/.test(ctx.lower),
  },
  {
    id: "generic_positive",
    label: "Jenerik olumlu ifade",
    category: "linguistic",
    description: "Kısa, şablonvari olumlu ifadeler ('harika', 'süper', 'teşekkürler') spesifik bir deneyim aktarmaz. Sahte yorum üreticileri genellikle bu tür kısa, genel ifadeler kullanır (Ott et al., 2011).",
    weight: 2,
    check: (_, ctx) => ctx.wordCount <= 4 && GENERIC_PHRASES.some(p => ctx.lower.includes(p)),
  },
  {
    id: "spam_content",
    label: "Spam içerik",
    category: "statistical",
    description: "Sadece rakamlardan oluşan veya URL içeren yorumlar — reklam veya bot kaynaklı olabilir.",
    weight: 3,
    check: (_, ctx) => /^\d+$/.test(ctx.text.replace(/\s/g, "")) || /https?:\/\//.test(ctx.text),
  },
];

const CATEGORY_LABELS: Record<string, { label: string; description: string }> = {
  model: { label: "Model Tabanlı", description: "BERTurk modelinin sınıflandırma güveninden türetilir" },
  linguistic: { label: "Dilbilimsel", description: "Metin yapısı ve dil kalıpları analizi" },
  statistical: { label: "İstatistiksel", description: "Frekans ve tekrar örüntüleri tespiti" },
};

// ── Tespit motoru ───────────────────────────────────────────────────────────

interface SuspiciousReview {
  review: ReviewResult;
  reasons: string[];
  riskScore: number;
  matchedRules: Rule[];
}

function detectSuspicious(reviews: ReviewResult[]): SuspiciousReview[] {
  const texts = reviews.map(r => r.text.trim().toLowerCase());
  const freq: Record<string, number> = {};
  texts.forEach(t => { freq[t] = (freq[t] || 0) + 1; });
  const results: SuspiciousReview[] = [];

  reviews.forEach((r, i) => {
    const text = r.text.trim();
    const ctx: ReviewContext = {
      text,
      lower: texts[i],
      wordCount: text.split(/\s+/).filter(Boolean).length,
      freq,
      idx: i,
      texts,
    };

    const matchedRules = RULES.filter(rule => rule.check(r, ctx));
    if (matchedRules.length > 0) {
      const riskScore = Math.min(matchedRules.reduce((s, r) => s + r.weight, 0) * 12, 100);
      results.push({
        review: r,
        reasons: matchedRules.map(r => r.label),
        riskScore,
        matchedRules,
      });
    }
  });

  return results.sort((a, b) => b.riskScore - a.riskScore);
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

// ── Bileşen ─────────────────────────────────────────────────────────────────

interface Props {
  reviews: ReviewResult[];
  flaggedCount: number;
}

export function FakeReviewPanel({ reviews }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [showMethodology, setShowMethodology] = useState(false);
  const [selectedReview, setSelectedReview] = useState<SuspiciousReview | null>(null);
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

  const reasonGroups = groupFlaggedByReason(suspicious.map(s => ({ review: s.review, reasons: s.reasons })));
  const duplicateClusters = getDuplicateClusters(reviews);

  const avgRisk = suspicious.length > 0
    ? Math.round(suspicious.reduce((s, r) => s + r.riskScore, 0) / suspicious.length)
    : 0;

  if (suspicious.length === 0) {
    return (
      <div className="flex items-center gap-3 px-5 py-4 rounded-2xl border border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-900/20">
        <ShieldCheck className="w-5 h-5 text-emerald-500 shrink-0" />
        <div>
          <p className="text-sm font-semibold text-emerald-700 dark:text-emerald-300">Şüpheli yorum tespit edilmedi</p>
          <p className="text-xs text-emerald-600/70 dark:text-emerald-400/70 mt-0.5">{total} yorum {RULES.length} kural ile analiz edildi, tümü güvenilir görünüyor.</p>
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
            <span className="ml-2 font-normal opacity-70">(%{pct} · Ort. risk: {avgRisk}/100 · Seviye: {riskLevel})</span>
          </p>
          <p className="text-xs text-[var(--text-muted)] mt-0.5">
            {RULES.length} kural tabanlı analiz · 3 kategori (model, dilbilimsel, istatistiksel)
          </p>
        </div>
        <div className="hidden sm:flex flex-col items-end gap-1 shrink-0 mr-2">
          <div className="w-24 h-1.5 rounded-full bg-black/10 dark:bg-white/10 overflow-hidden">
            <div
              className={`h-full rounded-full ${pct >= 20 ? "bg-red-500" : pct >= 10 ? "bg-amber-500" : "bg-emerald-500"}`}
              style={{ width: `${Math.min(pct * 3, 100)}%` }}
            />
          </div>
          <span className="text-xs text-[var(--text-muted)]">%{pct} şüpheli</span>
        </div>
        {expanded ? <ChevronUp className="w-4 h-4 text-[var(--text-muted)] shrink-0" /> : <ChevronDown className="w-4 h-4 text-[var(--text-muted)] shrink-0" />}
      </button>

      {/* Expanded content */}
      {expanded && (
        <div className="border-t border-black/5 dark:border-white/5 px-6 pb-6 pt-4 space-y-5">

          {/* Metodoloji açıklama */}
          <div>
            <button
              onClick={() => setShowMethodology(v => !v)}
              className="flex items-center gap-2 text-xs font-semibold text-violet-600 dark:text-violet-400 hover:underline mb-2"
            >
              <Info className="w-3.5 h-3.5" />
              Nasıl Çalışır? (Metodoloji)
              {showMethodology ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>

            {showMethodology && (
              <div className="p-4 rounded-2xl bg-white/80 dark:bg-black/15 border border-[var(--border)] space-y-3">
                <p className="text-xs text-[var(--text)] leading-relaxed">
                  Sahte yorum tespiti, <strong>kural tabanlı çok boyutlu analiz</strong> yaklaşımı kullanır.
                  Her yorum {RULES.length} kurala karşı test edilir ve eşleşen kuralların ağırlıklı toplamı
                  ile <strong>risk skoru (0–100)</strong> hesaplanır. Kurallar 3 akademik kategoriye ayrılır:
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                  {Object.entries(CATEGORY_LABELS).map(([key, { label, description }]) => {
                    const count = RULES.filter(r => r.category === key).length;
                    return (
                      <div key={key} className="p-3 rounded-xl bg-[var(--bg)] border border-[var(--border)]">
                        <p className="text-xs font-bold text-[var(--text)]">{label}</p>
                        <p className="text-[10px] text-[var(--text-muted)] mt-0.5">{description}</p>
                        <p className="text-[10px] text-violet-500 font-semibold mt-1">{count} kural</p>
                      </div>
                    );
                  })}
                </div>
                <p className="text-[10px] text-[var(--text-muted)] italic">
                  Referanslar: Jindal & Liu (2008) — Opinion Spam Detection; Ott et al. (2011) — Finding Deceptive Opinion Spam;
                  Mukherjee et al. (2012) — Spotting Fake Reviewer Groups
                </p>
              </div>
            )}
          </div>

          {/* Sebep Dağılımı Grafiği */}
          {reasonGroups.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-[var(--text)] mb-3">Kural Eşleşme Dağılımı</p>
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
                    width={168}
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
                    formatter={(v: number) => [`${v} yorum`, "Eşleşme"]}
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
            <p className="text-xs font-semibold text-[var(--text)] mb-2">
              Şüpheli Yorumlar
              <span className="font-normal text-[var(--text-muted)] ml-1">(tıkla → detaylı açıklama)</span>
            </p>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {suspicious.map((item, i) => (
                <div key={i}>
                  <button
                    onClick={() => setSelectedReview(selectedReview === item ? null : item)}
                    className="w-full flex items-start gap-3 p-3 rounded-2xl bg-white/60 dark:bg-black/10 hover:bg-white/90 dark:hover:bg-black/20 transition-colors text-left"
                  >
                    <AlertCircle className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-[var(--text)] leading-snug line-clamp-2">{item.review.text}</p>
                      <div className="flex flex-wrap items-center gap-1.5 mt-2">
                        {item.reasons.map((r, j) => (
                          <span key={j} className="text-xs px-2 py-0.5 rounded-full bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300 font-medium">
                            {r}
                          </span>
                        ))}
                        <span className="text-xs px-2 py-0.5 rounded-full bg-[var(--border)] text-[var(--text-muted)] font-medium">
                          Risk: {item.riskScore}/100
                        </span>
                        <span className="text-xs px-2 py-0.5 rounded-full bg-[var(--border)] text-[var(--text-muted)] font-medium">
                          %{Math.round(item.review.confidence * 100)} güven
                        </span>
                      </div>
                    </div>
                  </button>

                  {/* Detaylı açıklama (tıklanınca) */}
                  {selectedReview === item && (
                    <div className="mt-1 ml-7 p-3 rounded-xl bg-violet-50/80 dark:bg-violet-900/15 border border-violet-200/60 dark:border-violet-800/40 space-y-2">
                      <p className="text-xs font-bold text-violet-700 dark:text-violet-300">Neden Şüpheli?</p>
                      {item.matchedRules.map((rule, j) => (
                        <div key={j} className="flex items-start gap-2">
                          <span className={`shrink-0 text-[10px] font-bold px-1.5 py-0.5 rounded mt-0.5 ${
                            rule.category === "model" ? "bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300"
                            : rule.category === "linguistic" ? "bg-purple-100 dark:bg-purple-900/40 text-purple-700 dark:text-purple-300"
                            : "bg-orange-100 dark:bg-orange-900/40 text-orange-700 dark:text-orange-300"
                          }`}>
                            {CATEGORY_LABELS[rule.category].label}
                          </span>
                          <div>
                            <p className="text-xs font-semibold text-[var(--text)]">{rule.label} <span className="font-normal text-[var(--text-muted)]">(ağırlık: {rule.weight}/3)</span></p>
                            <p className="text-[11px] text-[var(--text-muted)] leading-relaxed mt-0.5">{rule.description}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
