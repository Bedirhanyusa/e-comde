"use client";

import { motion } from "framer-motion";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from "recharts";
import { Navbar } from "@/components/Navbar";
import { Trophy, FlaskConical, BarChart3, Grid3X3 } from "lucide-react";

// ── Statik veriler ────────────────────────────────────────────────────────────

const MODEL_TABLE = [
  { name: "BERTurk (zero-shot)", type: "Transformer", accuracy: 0.373,  f1: 0.3045, mcc: 0.068, roc: 0.525 },
  { name: "BiLSTM",              type: "Derin Öğrenme", accuracy: 0.698,  f1: 0.6943, mcc: 0.548, roc: 0.859 },
  { name: "TF-IDF + LR",         type: "Baseline",     accuracy: 0.7023, f1: 0.6996, mcc: 0.554, roc: 0.859 },
  { name: "XLM-RoBERTa v1",      type: "Transformer",  accuracy: 0.7453, f1: 0.7426, mcc: 0.619, roc: 0.888 },
  { name: "BERTurk v2",          type: "Transformer",  accuracy: 0.754,  f1: 0.7538, mcc: 0.632, roc: 0.897 },
  { name: "Ensemble v1",         type: "Ensemble",     accuracy: 0.758,  f1: 0.7563, mcc: 0.638, roc: 0.898 },
  { name: "XLM-RoBERTa v2",      type: "Transformer",  accuracy: 0.7664, f1: 0.7643, mcc: 0.650, roc: 0.912 },
  { name: "Savasy Fine-tuned",   type: "Transformer",  accuracy: 0.7704, f1: 0.7696, mcc: 0.656, roc: 0.914 },
  { name: "Ensemble v3 (3-model)",type: "Ensemble",    accuracy: 0.7786, f1: 0.7776, mcc: 0.668, roc: 0.921 },
  { name: "BERTurk v3",          type: "Transformer",  accuracy: 0.7822, f1: 0.7813, mcc: 0.674, roc: 0.922, best: true },
  { name: "Ensemble v2 (Ağırlıklı)", type: "Ensemble", accuracy: 0.7827, f1: 0.7804, mcc: 0.675, roc: 0.921 },
];

const TYPE_COLORS: Record<string, string> = {
  "Baseline":     "#9CA3AF",
  "Derin Öğrenme":"#60A5FA",
  "Transformer":  "#7C3AED",
  "Ensemble":     "#F59E0B",
};

const CROSS_CATEGORY = [
  { name: "Elektronik",        f1: 0.8071, n: 176  },
  { name: "Kitap ve Hobi",     f1: 0.7962, n: 140  },
  { name: "Ev ve Yaşam",       f1: 0.7830, n: 628  },
  { name: "Gıda ve İçecek",    f1: 0.7816, n: 1111 },
  { name: "Giyim ve Aksesuar", f1: 0.7682, n: 791  },
];

const PER_CLASS = [
  { name: "Olumsuz", f1: 0.8005, precision: 0.804,  recall: 0.797, support: 1656, color: "#ef4444" },
  { name: "Nötr",    f1: 0.6877, precision: 0.699,  recall: 0.677, support: 1656, color: "#9CA3AF" },
  { name: "Olumlu",  f1: 0.8556, precision: 0.839,  recall: 0.873, support: 1657, color: "#22c55e" },
];

// confusion matrix: rows = gerçek, cols = tahmin
const CM = {
  classes: ["Olumsuz", "Nötr", "Olumlu"],
  matrix: [
    [1320, 284,  52 ],
    [310,  1121, 225],
    [12,   199,  1446],
  ],
  totals: [1656, 1656, 1657],
};

// ── Yardımcı bileşenler ───────────────────────────────────────────────────────

function SectionCard({ title, subtitle, icon: Icon, children }: {
  title: string; subtitle?: string; icon: React.ElementType; children: React.ReactNode;
}) {
  return (
    <div className="bg-white dark:bg-[var(--surface)] border border-[var(--border)] rounded-3xl shadow-sm p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 rounded-xl bg-violet-100 dark:bg-violet-900/30">
          <Icon className="w-4 h-4 text-violet-600 dark:text-violet-400" />
        </div>
        <div>
          <h2 className="font-bold text-base text-[var(--text)]">{title}</h2>
          {subtitle && <p className="text-xs text-[var(--text-muted)] mt-0.5">{subtitle}</p>}
        </div>
      </div>
      {children}
    </div>
  );
}

function BadgeType({ type }: { type: string }) {
  const color = TYPE_COLORS[type] ?? "#9CA3AF";
  return (
    <span
      className="inline-block px-2 py-0.5 rounded-full text-[10px] font-semibold text-white"
      style={{ background: color }}
    >
      {type}
    </span>
  );
}

// ── Confusion Matrix Heatmap ──────────────────────────────────────────────────

function ConfusionMatrixHeatmap() {
  const maxVal = Math.max(...CM.matrix.flat());

  function cellOpacity(val: number, isDiag: boolean) {
    const ratio = val / maxVal;
    if (isDiag) return `rgba(124,58,237,${0.15 + ratio * 0.65})`;
    return `rgba(239,68,68,${0.05 + ratio * 0.45})`;
  }

  return (
    <div className="overflow-x-auto">
      {/* Başlık satırı: tahmin edilen */}
      <div className="mb-2 text-center">
        <span className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-widest">
          Tahmin Edilen →
        </span>
      </div>
      <div className="flex">
        {/* Sol etiket: Gerçek */}
        <div className="flex flex-col items-center justify-center mr-3 w-6 shrink-0">
          <span
            className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-widest"
            style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
          >
            Gerçek ↑
          </span>
        </div>

        <div className="flex-1">
          {/* Kolon başlıkları */}
          <div className="grid grid-cols-4 gap-1 mb-1">
            <div /> {/* boş sol hücre */}
            {CM.classes.map(c => (
              <div key={c} className="text-center text-xs font-bold text-[var(--text)] py-1">{c}</div>
            ))}
          </div>

          {/* Satırlar */}
          {CM.matrix.map((row, ri) => (
            <div key={ri} className="grid grid-cols-4 gap-1 mb-1">
              {/* Satır etiketi */}
              <div className="flex items-center justify-end pr-2">
                <span className="text-xs font-bold text-[var(--text)]">{CM.classes[ri]}</span>
              </div>
              {row.map((val, ci) => {
                const isDiag = ri === ci;
                const pct = ((val / CM.totals[ri]) * 100).toFixed(1);
                return (
                  <motion.div
                    key={ci}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: (ri * 3 + ci) * 0.04 }}
                    className="rounded-xl p-3 flex flex-col items-center justify-center gap-0.5 cursor-default"
                    style={{ background: cellOpacity(val, isDiag) }}
                    title={`Gerçek: ${CM.classes[ri]} → Tahmin: ${CM.classes[ci]}: ${val} yorum`}
                  >
                    <span className={`text-base font-extrabold tabular-nums ${isDiag ? "text-violet-700 dark:text-violet-300" : "text-[var(--text)]"}`}>
                      {val.toLocaleString("tr-TR")}
                    </span>
                    <span className="text-[10px] text-[var(--text-muted)] font-medium">%{pct}</span>
                  </motion.div>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Açıklama */}
      <div className="mt-4 flex flex-wrap gap-3">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-sm bg-violet-500 opacity-70" />
          <span className="text-[11px] text-[var(--text-muted)]">Doğru tahmin (diyagonal)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-sm bg-red-400 opacity-50" />
          <span className="text-[11px] text-[var(--text-muted)]">Yanlış tahmin</span>
        </div>
      </div>
    </div>
  );
}

// ── Ana sayfa ─────────────────────────────────────────────────────────────────

export default function ModelsPage() {
  return (
    <div className="min-h-screen bg-[var(--bg)]">
      <Navbar />

      <main className="max-w-6xl mx-auto px-4 sm:px-6 py-10 space-y-8">

        {/* Başlık */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center py-6"
        >
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-violet-100 dark:bg-violet-900/30 border border-violet-200 dark:border-violet-800 text-violet-600 dark:text-violet-400 text-xs font-semibold mb-4">
            <FlaskConical className="w-3 h-3" />
            54.000 Türkçe E-Ticaret Yorumu · %80/10/10 Bölünmüş Test Seti
          </div>
          <h1 className="text-4xl font-extrabold text-[var(--text)] tracking-tight mb-3">
            Model <span className="text-gradient">Metrikleri</span>
          </h1>
          <p className="text-[var(--text-muted)] text-base max-w-xl mx-auto">
            TF-IDF'ten BERTurk v3'e model geliştirme yolculuğu, confusion matrix ve kategori bazlı analiz.
          </p>
        </motion.div>

        {/* Model karşılaştırma tablosu */}
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
          <SectionCard title="Model Karşılaştırması" subtitle="Test seti metrikleri — artan Macro-F1 sırasıyla" icon={Trophy}>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-[var(--border)]">
                    <th className="text-left py-2.5 pr-4 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wide">Model</th>
                    <th className="text-left py-2.5 pr-3 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wide">Tür</th>
                    <th className="text-right py-2.5 pr-3 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wide">Accuracy</th>
                    <th className="text-right py-2.5 pr-3 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wide">Macro-F1</th>
                    <th className="text-right py-2.5 pr-3 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wide">MCC</th>
                    <th className="text-right py-2.5 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wide">ROC-AUC</th>
                  </tr>
                </thead>
                <tbody>
                  {MODEL_TABLE.map((m, i) => (
                    <motion.tr
                      key={m.name}
                      initial={{ opacity: 0, x: -8 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.05 + i * 0.04 }}
                      className={`border-b border-[var(--border)] last:border-0 transition-colors hover:bg-violet-50/40 dark:hover:bg-violet-900/10 ${
                        m.best ? "bg-violet-50 dark:bg-violet-900/20" : ""
                      }`}
                    >
                      <td className="py-3 pr-4">
                        <span className={`font-semibold ${m.best ? "text-violet-700 dark:text-violet-300" : "text-[var(--text)]"}`}>
                          {m.name}
                        </span>
                        {m.best && (
                          <span className="ml-2 text-[10px] bg-violet-600 text-white px-1.5 py-0.5 rounded-full font-bold">
                            Seçilen
                          </span>
                        )}
                      </td>
                      <td className="py-3 pr-3"><BadgeType type={m.type} /></td>
                      <td className="py-3 pr-3 text-right tabular-nums font-mono text-[var(--text)]">{(m.accuracy * 100).toFixed(2)}%</td>
                      <td className={`py-3 pr-3 text-right tabular-nums font-mono font-bold ${m.best ? "text-violet-600 dark:text-violet-400" : "text-[var(--text)]"}`}>
                        {(m.f1 * 100).toFixed(2)}%
                      </td>
                      <td className="py-3 pr-3 text-right tabular-nums font-mono text-[var(--text)]">{m.mcc.toFixed(3)}</td>
                      <td className="py-3 text-right tabular-nums font-mono text-[var(--text)]">{m.roc.toFixed(3)}</td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* F1 gelişim mini bar'ı */}
            <div className="mt-6 pt-5 border-t border-[var(--border)]">
              <p className="text-xs font-semibold text-[var(--text-muted)] mb-3 uppercase tracking-wide">Macro-F1 Gelişimi</p>
              <div className="flex items-end gap-1 h-16">
                {MODEL_TABLE.map((m, i) => {
                  const pct = (m.f1 / 0.82) * 100;
                  return (
                    <motion.div
                      key={m.name}
                      title={`${m.name}: ${(m.f1 * 100).toFixed(2)}%`}
                      className={`flex-1 rounded-t-md cursor-default transition-opacity ${m.best ? "opacity-100" : "opacity-60 hover:opacity-80"}`}
                      style={{ height: `${pct}%`, background: m.best ? "#7C3AED" : TYPE_COLORS[m.type] ?? "#9CA3AF" }}
                      initial={{ height: 0 }}
                      animate={{ height: `${pct}%` }}
                      transition={{ delay: 0.3 + i * 0.05, duration: 0.5, ease: "easeOut" }}
                    />
                  );
                })}
              </div>
              <div className="flex gap-1 mt-1">
                {MODEL_TABLE.map((m) => (
                  <div key={m.name} className="flex-1 text-center text-[8px] text-[var(--text-muted)] truncate" title={m.name}>
                    {(m.f1 * 100).toFixed(0)}
                  </div>
                ))}
              </div>
            </div>
          </SectionCard>
        </motion.div>

        {/* Confusion Matrix + Per-class F1 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
            <SectionCard title="Confusion Matrix" subtitle="BERTurk v3 — 4.969 test yorumu" icon={Grid3X3}>
              <ConfusionMatrixHeatmap />
            </SectionCard>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}>
            <SectionCard title="Sınıf Bazlı Metrikler" subtitle="BERTurk v3 — precision, recall, F1" icon={BarChart3}>
              <div className="space-y-5">
                {PER_CLASS.map((cls, i) => (
                  <motion.div key={cls.name} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 + i * 0.1 }}>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: cls.color }} />
                        <span className="text-sm font-semibold text-[var(--text)]">{cls.name}</span>
                        <span className="text-xs text-[var(--text-muted)]">({cls.support.toLocaleString("tr-TR")} yorum)</span>
                      </div>
                      <span className="text-sm font-bold tabular-nums" style={{ color: cls.color }}>
                        F1: {(cls.f1 * 100).toFixed(1)}%
                      </span>
                    </div>
                    {/* Precision / Recall mini bar'lar */}
                    <div className="space-y-1.5">
                      {[
                        { label: "Precision", val: cls.precision },
                        { label: "Recall",    val: cls.recall    },
                        { label: "F1",        val: cls.f1        },
                      ].map(({ label, val }) => (
                        <div key={label} className="flex items-center gap-2">
                          <span className="text-[10px] w-14 text-[var(--text-muted)] font-medium">{label}</span>
                          <div className="flex-1 h-1.5 rounded-full bg-[var(--border)] overflow-hidden">
                            <motion.div
                              className="h-full rounded-full"
                              style={{ background: cls.color }}
                              initial={{ width: 0 }}
                              animate={{ width: `${val * 100}%` }}
                              transition={{ delay: 0.4 + i * 0.1, duration: 0.7, ease: "easeOut" }}
                            />
                          </div>
                          <span className="text-[10px] text-[var(--text-muted)] tabular-nums w-9 text-right">{(val * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                ))}

                {/* Özet */}
                <div className="mt-4 pt-4 border-t border-[var(--border)] grid grid-cols-3 gap-3 text-center">
                  {[
                    { label: "Accuracy",  val: "78.22%" },
                    { label: "Macro-F1",  val: "78.13%" },
                    { label: "ROC-AUC",   val: "0.922"  },
                  ].map(({ label, val }) => (
                    <div key={label} className="p-2 rounded-xl bg-violet-50 dark:bg-violet-900/20">
                      <p className="text-base font-extrabold text-violet-700 dark:text-violet-300">{val}</p>
                      <p className="text-[10px] text-[var(--text-muted)] mt-0.5">{label}</p>
                    </div>
                  ))}
                </div>
              </div>
            </SectionCard>
          </motion.div>
        </div>

        {/* Cross-category analiz */}
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
          <SectionCard title="Kategori Bazlı Genelleme" subtitle="BERTurk v3 — 5 ürün kategorisinde Macro-F1 (test seti)" icon={BarChart3}>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={CROSS_CATEGORY} margin={{ top: 8, right: 16, left: -8, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 11, fill: "var(--text-muted)" }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    domain={[0.70, 0.83]}
                    tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                    tick={{ fontSize: 11, fill: "var(--text-muted)" }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      borderRadius: 12,
                      fontSize: 12,
                      border: "1px solid var(--border)",
                      background: "var(--surface)",
                      color: "var(--text)",
                    }}
                    formatter={(v: number) => [
                      `${(v * 100).toFixed(2)}%`,
                      "Macro-F1",
                    ]}
                  />
                  <Bar dataKey="f1" radius={[8, 8, 0, 0]} maxBarSize={80}>
                    {CROSS_CATEGORY.map((_, i) => (
                      <Cell key={i} fill={i === 0 ? "#7C3AED" : "#A78BFA"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-4 grid grid-cols-2 sm:grid-cols-5 gap-3">
              {CROSS_CATEGORY.map((cat) => (
                <div key={cat.name} className="p-3 rounded-xl bg-[var(--bg)] border border-[var(--border)] text-center">
                  <p className="text-base font-extrabold text-violet-700 dark:text-violet-300">
                    {(cat.f1 * 100).toFixed(1)}%
                  </p>
                  <p className="text-[10px] text-[var(--text-muted)] mt-0.5 font-medium">{cat.name}</p>
                  <p className="text-[10px] text-[var(--text-muted)]">n={cat.n}</p>
                </div>
              ))}
            </div>
          </SectionCard>
        </motion.div>

      </main>
    </div>
  );
}
