"use client";

import { ThumbsUp, ThumbsDown, CheckCircle2, XCircle } from "lucide-react";
import { motion } from "framer-motion";

interface Props {
  pros: string[];
  cons: string[];
}

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.07 } },
};

const item = {
  hidden: { opacity: 0, x: -8 },
  show: { opacity: 1, x: 0, transition: { duration: 0.3 } },
};

export function ProConSection({ pros, cons }: Props) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* Pros */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="bg-[var(--surface)] border border-[var(--border)] rounded-3xl shadow-sm p-6 overflow-hidden relative"
      >
        <div className="absolute top-0 right-0 w-24 h-24 bg-emerald-400/5 rounded-full -translate-y-8 translate-x-8" />
        <div className="flex items-center gap-2 mb-4">
          <div className="p-1.5 rounded-lg bg-emerald-100 dark:bg-emerald-900/30">
            <ThumbsUp className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
          </div>
          <h3 className="font-bold text-sm text-[var(--text)] uppercase tracking-wide">En Çok Övülenler</h3>
        </div>
        <motion.ul variants={container} initial="hidden" animate="show" className="space-y-2.5">
          {pros.length > 0 ? pros.map((p, i) => (
            <motion.li key={i} variants={item} className="flex items-start gap-2.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-500 flex-shrink-0 mt-0.5" />
              <span className="text-sm text-[var(--text)] leading-relaxed">{p}</span>
            </motion.li>
          )) : (
            <p className="text-sm text-[var(--text-muted)]">Yeterli veri yok.</p>
          )}
        </motion.ul>
      </motion.div>

      {/* Cons */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.1 }}
        className="bg-[var(--surface)] border border-[var(--border)] rounded-3xl shadow-sm p-6 overflow-hidden relative"
      >
        <div className="absolute top-0 right-0 w-24 h-24 bg-red-400/5 rounded-full -translate-y-8 translate-x-8" />
        <div className="flex items-center gap-2 mb-4">
          <div className="p-1.5 rounded-lg bg-red-100 dark:bg-red-900/30">
            <ThumbsDown className="w-3.5 h-3.5 text-red-500 dark:text-red-400" />
          </div>
          <h3 className="font-bold text-sm text-[var(--text)] uppercase tracking-wide">En Çok Şikayet Edilenler</h3>
        </div>
        <motion.ul variants={container} initial="hidden" animate="show" className="space-y-2.5">
          {cons.length > 0 ? cons.map((c, i) => (
            <motion.li key={i} variants={item} className="flex items-start gap-2.5">
              <XCircle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
              <span className="text-sm text-[var(--text)] leading-relaxed">{c}</span>
            </motion.li>
          )) : (
            <p className="text-sm text-[var(--text-muted)]">Yeterli veri yok.</p>
          )}
        </motion.ul>
      </motion.div>
    </div>
  );
}
