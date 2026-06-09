"use client";

import { CheckCircle2, AlertTriangle, XCircle, Users, ShoppingCart, Eye } from "lucide-react";
import { motion } from "framer-motion";
import { AdvisorResponse } from "@/lib/types";

interface Props {
  advisor: AdvisorResponse;
}

const VERDICT_CONFIG = {
  "Tavsiye Edilir": {
    icon: CheckCircle2,
    gradient: "from-emerald-500 to-green-400",
    bg: "bg-emerald-50 dark:bg-emerald-900/10",
    border: "border-emerald-200 dark:border-emerald-800/60",
    badge: "bg-emerald-500",
    text: "text-emerald-700 dark:text-emerald-300",
    glow: "shadow-emerald-500/20",
  },
  "Dikkatli Olun": {
    icon: AlertTriangle,
    gradient: "from-amber-500 to-yellow-400",
    bg: "bg-amber-50 dark:bg-amber-900/10",
    border: "border-amber-200 dark:border-amber-800/60",
    badge: "bg-amber-500",
    text: "text-amber-700 dark:text-amber-300",
    glow: "shadow-amber-500/20",
  },
  "Tavsiye Edilmez": {
    icon: XCircle,
    gradient: "from-red-500 to-rose-400",
    bg: "bg-red-50 dark:bg-red-900/10",
    border: "border-red-200 dark:border-red-800/60",
    badge: "bg-red-500",
    text: "text-red-700 dark:text-red-300",
    glow: "shadow-red-500/20",
  },
} as const;

const listVariants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.08 } },
};
const listItem = {
  hidden: { opacity: 0, x: -6 },
  show: { opacity: 1, x: 0, transition: { duration: 0.3 } },
};

export function AdvisorCard({ advisor }: Props) {
  const cfg = VERDICT_CONFIG[advisor.verdict] ?? VERDICT_CONFIG["Dikkatli Olun"];
  const VerdictIcon = cfg.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={`rounded-3xl border ${cfg.border} ${cfg.bg} overflow-hidden shadow-lg ${cfg.glow}`}
    >
      {/* Gradient top bar */}
      <div className={`h-1 w-full bg-gradient-to-r ${cfg.gradient}`} />

      <div className="p-6">
        {/* Header */}
        <div className="flex items-center gap-3 mb-5">
          <div className="p-2 rounded-2xl bg-white/60 dark:bg-black/20">
            <ShoppingCart className="w-4 h-4 text-violet-600 dark:text-violet-400" />
          </div>
          <div>
            <h2 className="font-bold text-base text-[var(--text)]">Bu Ürünü Almalı mısın?</h2>
            <p className="text-xs text-[var(--text-muted)] mt-0.5">Yapay zeka alışveriş danışmanı</p>
          </div>
        </div>

        {/* Verdict */}
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
          className="mb-5"
        >
          <span className={`inline-flex items-center gap-2 px-4 py-2 rounded-2xl text-sm font-bold text-white shadow-md ${cfg.badge}`}>
            <VerdictIcon className="w-4 h-4" />
            {advisor.verdict}
          </span>
        </motion.div>

        {/* Target audience */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="flex items-start gap-2.5 mb-5 px-4 py-3 rounded-2xl bg-white/50 dark:bg-black/10"
        >
          <Users className="w-4 h-4 text-violet-500 mt-0.5 shrink-0" />
          <p className="text-sm text-[var(--text)]">{advisor.target_audience}</p>
        </motion.div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {advisor.buy_if.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-2.5">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                <span className="text-xs font-bold text-[var(--text-muted)] uppercase tracking-wide">Şu Durumlarda Al</span>
              </div>
              <motion.ul variants={listVariants} initial="hidden" animate="show" className="space-y-1.5">
                {advisor.buy_if.map((item, i) => (
                  <motion.li key={i} variants={listItem} className="flex items-start gap-2 text-sm text-[var(--text)]">
                    <span className="mt-2 w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0" />
                    {item}
                  </motion.li>
                ))}
              </motion.ul>
            </div>
          )}

          {advisor.watch_out.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-2.5">
                <Eye className="w-3.5 h-3.5 text-amber-500" />
                <span className="text-xs font-bold text-[var(--text-muted)] uppercase tracking-wide">Dikkat Et</span>
              </div>
              <motion.ul variants={listVariants} initial="hidden" animate="show" className="space-y-1.5">
                {advisor.watch_out.map((item, i) => (
                  <motion.li key={i} variants={listItem} className="flex items-start gap-2 text-sm text-[var(--text)]">
                    <span className="mt-2 w-1.5 h-1.5 rounded-full bg-amber-400 shrink-0" />
                    {item}
                  </motion.li>
                ))}
              </motion.ul>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
