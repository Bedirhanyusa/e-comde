"use client";

import { useTheme } from "next-themes";
import { Moon, Sun, Sparkles, ShoppingBag, RotateCcw } from "lucide-react";
import { useEffect, useState } from "react";

interface Props {
  onNewAnalysis?: () => void;
}

export function Navbar({ onNewAnalysis }: Props) {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  return (
    <nav className="sticky top-0 z-50 bg-white/80 dark:bg-[var(--surface)]/80 backdrop-blur-md border-b border-[var(--border)]">
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-xl bg-violet-600 flex items-center justify-center shadow-sm">
            <Sparkles className="w-4 h-4 text-white" strokeWidth={2.2} />
          </div>
          <span className="font-bold text-xl tracking-tight text-[var(--text)]">E-ComDe</span>
        </div>

        <div className="flex items-center gap-3">
          {onNewAnalysis && (
            <button
              onClick={onNewAnalysis}
              className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm font-medium bg-[var(--bg)] border border-[var(--border)] hover:border-violet-400 text-[var(--text-muted)] transition"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Yeni Analiz
            </button>
          )}
          <div className="hidden sm:flex items-center gap-1.5 text-sm text-[var(--text-muted)]">
            <ShoppingBag className="w-4 h-4" />
            <span>Akıllı Ürün Analizi</span>
          </div>
          {mounted && (
            <button
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              className="p-2 rounded-xl hover:bg-[var(--border)] transition text-[var(--text-muted)]"
              aria-label="Tema değiştir"
            >
              {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
          )}
        </div>
      </div>
    </nav>
  );
}
