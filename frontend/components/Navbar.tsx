"use client";

import { useTheme } from "next-themes";
import { Moon, Sun, Sparkles, RotateCcw, Scale, FlaskConical } from "lucide-react";
import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";

interface Props {
  onNewAnalysis?: () => void;
}

export function Navbar({ onNewAnalysis }: Props) {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    setMounted(true);
    const onScroll = () => setScrolled(window.scrollY > 10);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const isShop = pathname === "/shop";
  const isCompare = pathname === "/compare";
  const isModels = pathname === "/models";

  return (
    <motion.nav
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className={`sticky top-0 z-50 transition-all duration-300 ${
        scrolled
          ? "glass border-b border-[var(--border)] shadow-sm"
          : "bg-transparent border-b border-transparent"
      }`}
    >
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        {/* Logo */}
        <Link
          href={isShop ? "/" : "/shop"}
          className="flex items-center gap-2.5 group"
        >
          <motion.div
            whileHover={{ scale: 1.05, rotate: 5 }}
            whileTap={{ scale: 0.95 }}
            className="w-9 h-9 rounded-xl bg-gradient-to-br from-violet-500 to-violet-700 flex items-center justify-center shadow-md group-hover:shadow-violet-500/30 transition-shadow"
          >
            <Sparkles className="w-4.5 h-4.5 text-white" strokeWidth={2} />
          </motion.div>
          <span className="font-bold text-xl tracking-tight text-[var(--text)]">
            {isShop ? (
              <>E-Com<span className="text-gradient">De</span></>
            ) : (
              <>İstün<span className="text-gradient">Shop</span></>
            )}
          </span>
        </Link>

        {/* Nav links */}
        <div className="flex items-center gap-2">
          <AnimatePresence mode="wait">
            {!isModels && (
              <motion.div
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
              >
                <Link
                  href="/models"
                  className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold text-violet-600 dark:text-violet-400 border border-violet-200 dark:border-violet-800/60 hover:bg-violet-50 dark:hover:bg-violet-900/20 transition-colors"
                >
                  <FlaskConical className="w-3.5 h-3.5" />
                  Model Metrikleri
                </Link>
              </motion.div>
            )}
          </AnimatePresence>
          <AnimatePresence mode="wait">
            {!isCompare && (
              <motion.div
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
              >
                <Link
                  href="/compare"
                  className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold text-violet-600 dark:text-violet-400 border border-violet-200 dark:border-violet-800/60 hover:bg-violet-50 dark:hover:bg-violet-900/20 transition-colors"
                >
                  <Scale className="w-3.5 h-3.5" />
                  Karşılaştır
                </Link>
              </motion.div>
            )}
          </AnimatePresence>

          {onNewAnalysis && (
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.97 }}
              onClick={onNewAnalysis}
              className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm font-medium bg-[var(--surface)] border border-[var(--border)] hover:border-violet-400 text-[var(--text-muted)] transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Yeni Analiz
            </motion.button>
          )}

          {mounted && (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
              className="p-2 rounded-xl hover:bg-[var(--border)] transition-colors text-[var(--text-muted)]"
              aria-label="Tema değiştir"
            >
              <AnimatePresence mode="wait" initial={false}>
                <motion.div
                  key={theme}
                  initial={{ rotate: -90, opacity: 0 }}
                  animate={{ rotate: 0, opacity: 1 }}
                  exit={{ rotate: 90, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                >
                  {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
                </motion.div>
              </AnimatePresence>
            </motion.button>
          )}
        </div>
      </div>
    </motion.nav>
  );
}
