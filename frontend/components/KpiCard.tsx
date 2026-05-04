"use client";

import { LucideIcon } from "lucide-react";
import clsx from "clsx";

interface Props {
  label: string;
  value: string | number;
  icon: LucideIcon;
  color: "indigo" | "green" | "red" | "amber";
  sub?: string;
}

const colorMap = {
  indigo: "bg-violet-50 dark:bg-violet-900/30 text-violet-600 dark:text-violet-400",
  green:  "bg-green-50  dark:bg-green-900/30  text-green-600  dark:text-green-400",
  red:    "bg-red-50    dark:bg-red-900/30    text-red-600    dark:text-red-400",
  amber:  "bg-amber-50  dark:bg-amber-900/30  text-amber-600  dark:text-amber-400",
};

const valueColor = {
  indigo: "text-violet-600 dark:text-violet-400",
  green:  "text-green-600  dark:text-green-400",
  red:    "text-red-600    dark:text-red-400",
  amber:  "text-amber-600  dark:text-amber-400",
};

export function KpiCard({ label, value, icon: Icon, color, sub }: Props) {
  return (
    <div className="bg-white dark:bg-[var(--surface)] border border-[var(--border)] rounded-2xl p-5 flex items-center gap-4 shadow-sm hover:shadow-md transition">
      <div className={clsx("p-3 rounded-xl", colorMap[color])}>
        <Icon className="w-6 h-6" />
      </div>
      <div>
        <div className={clsx("text-3xl font-bold tabular-nums", valueColor[color])}>{value}</div>
        <div className="text-sm text-[var(--text-muted)] font-medium">{label}</div>
        {sub && <div className="text-xs text-[var(--text-muted)] mt-0.5">{sub}</div>}
      </div>
    </div>
  );
}
