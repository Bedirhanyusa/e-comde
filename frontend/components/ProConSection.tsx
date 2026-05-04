"use client";

import { ThumbsUp, ThumbsDown, CheckCircle2, XCircle } from "lucide-react";

interface Props {
  pros: string[];
  cons: string[];
}

export function ProConSection({ pros, cons }: Props) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* Pros */}
      <div className="bg-white dark:bg-[var(--surface)] border border-[var(--border)] rounded-3xl shadow-sm p-6">
        <div className="flex items-center gap-2 mb-4">
          <ThumbsUp className="w-4 h-4 text-green-600" />
          <h3 className="font-bold text-base text-[var(--text)]">En Çok Övülenler</h3>
        </div>
        <ul className="space-y-3">
          {pros.length > 0 ? pros.map((p, i) => (
            <li key={i} className="flex items-start gap-2.5">
              <CheckCircle2 className="w-4 h-4 text-green-500 flex-shrink-0 mt-0.5" />
              <span className="text-sm text-[var(--text)] leading-relaxed">{p}</span>
            </li>
          )) : (
            <p className="text-sm text-[var(--text-muted)]">Yeterli veri yok.</p>
          )}
        </ul>
      </div>

      {/* Cons */}
      <div className="bg-white dark:bg-[var(--surface)] border border-[var(--border)] rounded-3xl shadow-sm p-6">
        <div className="flex items-center gap-2 mb-4">
          <ThumbsDown className="w-4 h-4 text-red-500" />
          <h3 className="font-bold text-base text-[var(--text)]">En Çok Şikayet Edilenler</h3>
        </div>
        <ul className="space-y-3">
          {cons.length > 0 ? cons.map((c, i) => (
            <li key={i} className="flex items-start gap-2.5">
              <XCircle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
              <span className="text-sm text-[var(--text)] leading-relaxed">{c}</span>
            </li>
          )) : (
            <p className="text-sm text-[var(--text-muted)]">Yeterli veri yok.</p>
          )}
        </ul>
      </div>
    </div>
  );
}
