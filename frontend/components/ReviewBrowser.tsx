"use client";

import { useState, useMemo } from "react";
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  flexRender,
  ColumnDef,
  SortingState,
} from "@tanstack/react-table";
import { Search, Download, ChevronUp, ChevronDown, Flag } from "lucide-react";
import { ReviewResult, Sentiment } from "@/lib/types";
import clsx from "clsx";

const BADGE: Record<Sentiment, string> = {
  olumlu: "bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-300",
  olumsuz: "bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300",
  notr: "bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300",
};

const FILTER_LABELS: Record<string, string> = {
  hepsi: "Tümü",
  olumlu: "Olumlu",
  olumsuz: "Olumsuz",
  notr: "Nötr",
};

interface Props {
  results: ReviewResult[];
}

export function ReviewBrowser({ results }: Props) {
  const [sentimentFilter, setSentimentFilter] = useState<Sentiment | "hepsi">("hepsi");
  const [showFlagged, setShowFlagged] = useState(false);
  const [globalFilter, setGlobalFilter] = useState("");
  const [sorting, setSorting] = useState<SortingState>([]);

  const filtered = useMemo(() => results.filter((r) => {
    if (sentimentFilter !== "hepsi" && r.label !== sentimentFilter) return false;
    if (showFlagged && !r.flagged) return false;
    return true;
  }), [results, sentimentFilter, showFlagged]);

  const columns: ColumnDef<ReviewResult>[] = [
    {
      accessorKey: "text",
      header: "Yorum",
      cell: (info) => (
        <span className="block max-w-xs truncate" title={info.getValue<string>()}>
          {info.getValue<string>()}
        </span>
      ),
    },
    {
      accessorKey: "label",
      header: "Etiket",
      cell: (info) => {
        const label = info.getValue<Sentiment>();
        return (
          <span className={clsx("px-2 py-0.5 rounded-full text-xs font-semibold", BADGE[label])}>
            {FILTER_LABELS[label]}
          </span>
        );
      },
    },
    {
      accessorKey: "confidence",
      header: "Güven",
      cell: (info) => {
        const val = info.getValue<number>();
        return (
          <div className="flex items-center gap-2">
            <div className="w-16 bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
              <div
                className="bg-violet-500 h-1.5 rounded-full"
                style={{ width: `${(val * 100).toFixed(0)}%` }}
              />
            </div>
            <span className="text-xs tabular-nums">{(val * 100).toFixed(1)}%</span>
          </div>
        );
      },
    },
    {
      accessorKey: "flagged",
      header: "Durum",
      cell: (info) =>
        info.getValue<boolean>() ? (
          <span className="flex items-center gap-1 text-red-500 text-xs font-medium">
            <Flag className="w-3 h-3" /> İnceleme
          </span>
        ) : (
          <span className="text-green-500 text-xs">✓ Otomatik</span>
        ),
    },
  ];

  const table = useReactTable({
    data: filtered,
    columns,
    state: { sorting, globalFilter },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    initialState: { pagination: { pageSize: 15 } },
  });

  const exportCSV = () => {
    const rows = [
      ["Yorum", "Etiket", "Güven", "İşaretli"],
      ...filtered.map((r) => [
        `"${r.text.replace(/"/g, '""')}"`,
        r.label,
        (r.confidence * 100).toFixed(1) + "%",
        r.flagged ? "Evet" : "Hayır",
      ]),
    ];
    const blob = new Blob([rows.map((r) => r.join(",")).join("\n")], { type: "text/csv;charset=utf-8;" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "e-comde-yorumlar.csv";
    a.click();
  };

  return (
    <div>
      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <div className="relative flex-1 min-w-[180px]">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--text-muted)]" />
          <input
            value={globalFilter}
            onChange={(e) => setGlobalFilter(e.target.value)}
            placeholder="Yorumlarda ara..."
            className="w-full pl-9 pr-3 py-2 text-sm rounded-xl border border-[var(--border)] bg-white dark:bg-[var(--surface)] focus:outline-none focus:ring-2 focus:ring-violet-400"
          />
        </div>
        <div className="flex gap-2">
          {(["hepsi", "olumlu", "olumsuz", "notr"] as const).map((f) => (
            <button
              key={f}
              onClick={() => setSentimentFilter(f)}
              className={clsx(
                "px-3 py-1.5 rounded-xl text-xs font-medium border transition",
                sentimentFilter === f
                  ? "bg-violet-600 text-white border-violet-600"
                  : "bg-white dark:bg-[var(--surface)] border-[var(--border)] text-[var(--text-muted)] hover:border-violet-400"
              )}
            >
              {FILTER_LABELS[f]}
            </button>
          ))}
        </div>
        <label className="flex items-center gap-1.5 text-xs text-[var(--text-muted)] cursor-pointer ml-auto">
          <input
            type="checkbox"
            checked={showFlagged}
            onChange={(e) => setShowFlagged(e.target.checked)}
            className="rounded"
          />
          Sadece işaretliler
        </label>
        <button
          onClick={exportCSV}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium bg-[var(--surface)] border border-[var(--border)] hover:border-indigo-400 transition text-[var(--text-muted)]"
        >
          <Download className="w-3.5 h-3.5" /> CSV İndir
        </button>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-xl border border-[var(--border)]">
        <table className="w-full text-sm">
          <thead className="bg-[var(--bg)] border-b border-[var(--border)]">
            {table.getHeaderGroups().map((hg) => (
              <tr key={hg.id}>
                {hg.headers.map((header) => (
                  <th
                    key={header.id}
                    className="text-left px-4 py-3 font-semibold text-[var(--text-muted)] text-xs uppercase tracking-wide cursor-pointer select-none"
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    <div className="flex items-center gap-1">
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      {header.column.getIsSorted() === "asc" && <ChevronUp className="w-3 h-3" />}
                      {header.column.getIsSorted() === "desc" && <ChevronDown className="w-3 h-3" />}
                    </div>
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row, i) => (
              <tr
                key={row.id}
                className={clsx(
                  "border-b border-[var(--border)] hover:bg-violet-50 dark:hover:bg-violet-900/10 transition",
                  i % 2 === 0 ? "bg-[var(--surface)]" : "bg-[var(--bg)]"
                )}
              >
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="px-4 py-3">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {table.getRowModel().rows.length === 0 && (
          <p className="text-center text-[var(--text-muted)] py-10">Sonuç bulunamadı.</p>
        )}
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between mt-3 text-xs text-[var(--text-muted)]">
        <span>
          {table.getFilteredRowModel().rows.length} yorum
          {showFlagged ? " (işaretli)" : ""} · Sayfa {table.getState().pagination.pageIndex + 1}/{table.getPageCount()}
        </span>
        <div className="flex gap-2">
          <button
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
            className="px-3 py-1 rounded-lg border border-[var(--border)] disabled:opacity-40 hover:border-violet-400 transition"
          >
            ← Önceki
          </button>
          <button
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
            className="px-3 py-1 rounded-lg border border-[var(--border)] disabled:opacity-40 hover:border-violet-400 transition"
          >
            Sonraki →
          </button>
        </div>
      </div>
    </div>
  );
}
