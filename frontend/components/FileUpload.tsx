"use client";

import { useRef, useState } from "react";
import { Upload, Loader2, FileText } from "lucide-react";
import { uploadCSV } from "@/lib/api";
import { SentimentResponse } from "@/lib/types";
import clsx from "clsx";

interface Props {
  onResult: (data: SentimentResponse) => void;
  dark?: boolean;
}

export function FileUpload({ onResult, dark }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = async (file: File) => {
    setLoading(true);
    setError(null);
    try {
      const data = await uploadCSV(file);
      onResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Hata oluştu.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      onClick={() => !loading && inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
      }}
      className={clsx(
        "relative flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed p-10 text-center cursor-pointer transition-all duration-200 select-none",
        loading && "pointer-events-none",
        dark
          ? clsx(
              "border-indigo-400/40 hover:border-indigo-300",
              dragging ? "bg-white/10 border-white/60" : "bg-white/5"
            )
          : clsx(
              "border-[var(--border)] hover:border-indigo-400 hover:bg-indigo-50/50 dark:hover:bg-indigo-900/10",
              dragging ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-900/20" : "bg-[var(--bg)]"
            )
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        className="hidden"
        onChange={(e) => { if (e.target.files?.[0]) handleFile(e.target.files[0]); }}
      />

      {loading ? (
        <>
          <div className={clsx("p-4 rounded-2xl", dark ? "bg-white/10" : "bg-indigo-100 dark:bg-indigo-900/40")}>
            <Loader2 className={clsx("w-7 h-7 animate-spin", dark ? "text-white" : "text-indigo-600")} />
          </div>
          <div>
            <p className={clsx("font-semibold text-sm", dark ? "text-white" : "text-[var(--text)]")}>Analiz ediliyor...</p>
            <p className={clsx("text-xs mt-0.5", dark ? "text-indigo-200" : "text-[var(--text-muted)]")}>BERTurk her yorumu sınıflandırıyor</p>
          </div>
        </>
      ) : (
        <>
          <div className={clsx("p-4 rounded-2xl transition", dark ? "bg-white/10" : "bg-indigo-100 dark:bg-indigo-900/40")}>
            {dragging
              ? <FileText className={clsx("w-7 h-7", dark ? "text-white" : "text-indigo-600")} />
              : <Upload className={clsx("w-7 h-7", dark ? "text-indigo-200" : "text-indigo-500")} />
            }
          </div>
          <div>
            <p className={clsx("font-semibold text-sm", dark ? "text-white" : "text-[var(--text)]")}>
              {dragging ? "Bırakın, yükleniyor!" : "CSV dosyasını sürükleyin veya tıklayın"}
            </p>
            <p className={clsx("text-xs mt-1", dark ? "text-indigo-200" : "text-[var(--text-muted)]")}>
              <span className="font-medium">Review</span> kolonlu CSV · maksimum 50.000 satır
            </p>
          </div>
          <div className={clsx(
            "mt-1 px-5 py-2 rounded-xl text-sm font-medium transition",
            dark
              ? "bg-white text-indigo-700 hover:bg-indigo-50"
              : "bg-indigo-600 text-white hover:bg-indigo-700"
          )}>
            Dosya Seç
          </div>
        </>
      )}

      {error && (
        <p className={clsx("text-sm mt-1", dark ? "text-red-300" : "text-red-500")}>
          {error}
        </p>
      )}
    </div>
  );
}
