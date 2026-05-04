"use client";

import { useRef, useState } from "react";
import { Search, Loader2, Paperclip } from "lucide-react";
import { uploadCSV } from "@/lib/api";
import { SentimentResponse } from "@/lib/types";

interface Props {
  onResult: (data: SentimentResponse) => void;
  onLoadingChange?: (loading: boolean) => void;
}

export function UploadBar({ onResult, onLoadingChange }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = async (file: File) => {
    setLoading(true);
    onLoadingChange?.(true);
    setError(null);
    try {
      const data = await uploadCSV(file);
      onResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Hata oluştu.");
    } finally {
      setLoading(false);
      onLoadingChange?.(false);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const file = e.dataTransfer.files[0];
          if (file) handleFile(file);
        }}
        className={`
          relative flex items-center gap-3 bg-white dark:bg-[var(--surface)] rounded-2xl
          border-2 transition-all duration-200 px-5 py-4
          ${dragging ? "border-violet-500 ring-4 ring-violet-100 dark:ring-violet-900/30" : "border-transparent"}
        `}
        style={{ boxShadow: "0 4px 32px 0 rgba(0,0,0,0.10)" }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={(e) => { if (e.target.files?.[0]) handleFile(e.target.files[0]); }}
        />

        <Search className="w-5 h-5 text-[var(--text-muted)] flex-shrink-0" />

        <span
          onClick={() => !loading && inputRef.current?.click()}
          className="flex-1 text-base text-[var(--text-muted)] cursor-pointer select-none truncate"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin text-violet-600" />
              <span>Analiz ediliyor...</span>
            </span>
          ) : dragging ? (
            <span className="text-violet-600 font-medium">Bırakın, yükleniyor!</span>
          ) : (
            "CSV dosyanızı sürükleyin veya tıklayın..."
          )}
        </span>

        <button
          onClick={() => !loading && inputRef.current?.click()}
          disabled={loading}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-violet-600 hover:bg-violet-700 active:bg-violet-800 text-white text-sm font-semibold transition-all duration-150 disabled:opacity-60 flex-shrink-0"
        >
          {loading
            ? <Loader2 className="w-4 h-4 animate-spin" />
            : <Paperclip className="w-4 h-4" />
          }
          Analiz Et
        </button>
      </div>

      {error && (
        <p className="text-center text-red-500 text-sm mt-3">{error}</p>
      )}

      <p className="text-center text-xs text-[var(--text-muted)] mt-3">
        <span className="font-medium">Review</span> kolonlu CSV · maksimum 50.000 satır · BERTurk v3 · %78.3 doğruluk
      </p>
    </div>
  );
}
