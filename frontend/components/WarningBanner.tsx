"use client";

import { AlertTriangle } from "lucide-react";

interface Props {
  points: string[];
}

export function WarningBanner({ points }: Props) {
  if (!points.length) return null;
  return (
    <div className="rounded-2xl border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-900/20 p-5">
      <div className="flex items-center gap-2 mb-3">
        <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400 flex-shrink-0" />
        <h3 className="font-bold text-sm text-amber-700 dark:text-amber-300">
          Satın Almadan Önce Dikkat!
        </h3>
      </div>
      <ul className="space-y-1.5">
        {points.map((p, i) => (
          <li key={i} className="flex items-start gap-2 text-sm text-amber-800 dark:text-amber-200">
            <span className="mt-1 w-1.5 h-1.5 rounded-full bg-amber-500 flex-shrink-0" />
            {p}
          </li>
        ))}
      </ul>
    </div>
  );
}
