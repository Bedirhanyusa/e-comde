"use client";

import { useState } from "react";
import { Loader2, Clock } from "lucide-react";
import { Navbar } from "@/components/Navbar";
import { UploadBar } from "@/components/UploadBar";
import { WarningBanner } from "@/components/WarningBanner";
import { ScoreSummaryCard } from "@/components/ScoreSummaryCard";
import { CategoryScores } from "@/components/CategoryScores";
import { ProConSection } from "@/components/ProConSection";
import { FeaturedReviews } from "@/components/FeaturedReviews";
import { ReviewBrowser } from "@/components/ReviewBrowser";
import { SentimentDistributionChart } from "@/components/charts/SentimentDistributionChart";
import { ConfidenceHistogram } from "@/components/charts/ConfidenceHistogram";
import { summarizeReviews } from "@/lib/api";
import { SentimentResponse, SummarizeResponse } from "@/lib/types";
import {
  calcSentimentScore,
  splitPoints,
  calcCategoryScores,
} from "@/lib/analysis";

function Card({
  title,
  subtitle,
  children,
  className = "",
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`bg-white dark:bg-[var(--surface)] border border-[var(--border)] rounded-3xl shadow-sm p-6 ${className}`}>
      <div className="mb-5">
        <h2 className="font-bold text-base text-[var(--text)]">{title}</h2>
        {subtitle && <p className="text-xs text-[var(--text-muted)] mt-0.5">{subtitle}</p>}
      </div>
      {children}
    </div>
  );
}

export default function DashboardPage() {
  const [sentimentData, setSentimentData] = useState<SentimentResponse | null>(null);
  const [summaryData, setSummaryData] = useState<SummarizeResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isSummarizing, setIsSummarizing] = useState(false);

  const handleResult = async (data: SentimentResponse) => {
    setSentimentData(data);
    setSummaryData(null);
    setIsSummarizing(true);
    try {
      const texts = data.results.map((r) => r.text);
      const labels = data.results.map((r) => r.label);
      const summary = await summarizeReviews(texts, labels);
      setSummaryData(summary);
    } catch {
      // sessiz geç
    } finally {
      setIsSummarizing(false);
    }
  };

  const handleReset = () => {
    setSentimentData(null);
    setSummaryData(null);
    setIsAnalyzing(false);
    setIsSummarizing(false);
  };

  // Derived values
  const score = sentimentData ? calcSentimentScore(sentimentData.results) : 0;
  const categoryScores = sentimentData ? calcCategoryScores(sentimentData.results) : [];
  const warnings = summaryData?.summaries?.olumsuz ? splitPoints(summaryData.summaries.olumsuz, 3) : [];
  const pros = summaryData?.summaries?.olumlu ? splitPoints(summaryData.summaries.olumlu, 4) : [];
  const cons = summaryData?.summaries?.olumsuz ? splitPoints(summaryData.summaries.olumsuz, 4) : [];
  const overallSummary = summaryData
    ? [summaryData.summaries?.olumlu, summaryData.summaries?.notr]
        .filter(Boolean)
        .join(" ")
        .slice(0, 400)
    : "";
  const targetAudience = summaryData?.summaries?.olumlu
    ? splitPoints(summaryData.summaries.olumlu, 1)[0]
    : undefined;

  const total = sentimentData?.total ?? 0;

  return (
    <div className="min-h-screen bg-[var(--bg)]">
      <Navbar onNewAnalysis={sentimentData ? handleReset : undefined} />

      <main className="max-w-6xl mx-auto px-4 sm:px-6">

        {/* ── LANDING ── */}
        {!sentimentData && (
          <div className="flex flex-col items-center justify-center min-h-[calc(100vh-64px)] text-center py-16">
            <h1 className="text-5xl sm:text-6xl font-extrabold text-[var(--text)] leading-[1.12] mb-5 max-w-3xl tracking-tight">
              Yorumları{" "}
              <span className="text-violet-600 dark:text-violet-400">yapay zeka</span>
              {" "}ile saniyeler içinde analiz edin.
            </h1>
            <p className="text-[var(--text-muted)] text-lg mb-10 max-w-xl leading-relaxed">
              CSV dosyanızı yükleyin. Yapay zeka her yorumu sınıflandırır, saçma yorumları ayıklar ve net bir özet sunar.
            </p>
            <UploadBar onResult={handleResult} onLoadingChange={setIsAnalyzing} />

            {/* Loading animation */}
            {isAnalyzing && (
              <div className="mt-16 flex flex-col items-center gap-4">
                <div className="relative w-20 h-20 flex items-center justify-center">
                  <div className="absolute inset-0 rounded-full border-4 border-violet-100 dark:border-violet-900/40" />
                  <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-violet-600 animate-spin" />
                  <Loader2 className="w-8 h-8 text-violet-600 animate-spin" />
                </div>
                <div>
                  <p className="font-bold text-base text-[var(--text)]">Yorumlar Analiz Ediliyor</p>
                  <p className="text-sm text-[var(--text-muted)] mt-1 max-w-xs">
                    Yapay zeka yüzlerce yorumu tarıyor, duygu kategorilerini puanlıyor...
                  </p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── RESULTS ── */}
        {sentimentData && (
          <div className="py-10 space-y-6">

            {/* Warning Banner */}
            {warnings.length > 0 && <WarningBanner points={warnings} />}

            {/* Score + AI Summary */}
            {isSummarizing ? (
              <div className="bg-white dark:bg-[var(--surface)] border border-[var(--border)] rounded-3xl p-10 flex flex-col items-center gap-3 text-[var(--text-muted)]">
                <div className="p-3 rounded-2xl bg-violet-50 dark:bg-violet-900/30">
                  <Loader2 className="w-6 h-6 animate-spin text-violet-500" />
                </div>
                <p className="text-sm font-medium">mT5 özet hazırlanıyor<span className="animate-pulse">...</span></p>
              </div>
            ) : overallSummary ? (
              <ScoreSummaryCard
                score={score}
                totalReviews={total}
                overallSummary={overallSummary}
                targetAudience={targetAudience}
              />
            ) : null}

            {/* Category Scores + Sentiment Distribution */}
            {(categoryScores.length > 0) && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <CategoryScores scores={categoryScores} />
                <div className="bg-white dark:bg-[var(--surface)] border border-[var(--border)] rounded-3xl shadow-sm p-6 flex flex-col h-full">
                  <div className="mb-5">
                    <h3 className="font-bold text-base text-[var(--text)] flex items-center gap-2">
                      <Clock className="w-4 h-4 text-violet-600" />
                      Duygu Dağılımı
                    </h3>
                  </div>
                  <SentimentDistributionChart results={sentimentData.results} />
                </div>
              </div>
            )}

            {/* If no category scores, show charts in full width */}
            {categoryScores.length === 0 && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card title="Duygu Dağılımı" subtitle="Sınıflandırma sonuçları">
                  <SentimentDistributionChart results={sentimentData.results} />
                </Card>
                <Card title="Güven Skoru Dağılımı" subtitle="Model tahmin güveni histogramı">
                  <ConfidenceHistogram results={sentimentData.results} />
                </Card>
              </div>
            )}

            {/* Confidence Histogram (when we have category scores) */}
            {categoryScores.length > 0 && (
              <Card title="Güven Skoru Dağılımı" subtitle="BERTurk model tahmin güveni histogramı">
                <ConfidenceHistogram results={sentimentData.results} />
              </Card>
            )}

            {/* Pros / Cons */}
            {(pros.length > 0 || cons.length > 0) && (
              <ProConSection pros={pros} cons={cons} />
            )}

            {/* Featured Reviews */}
            <FeaturedReviews reviews={sentimentData.results} />

            {/* Full Review Browser */}
            <Card
              title="Tüm Yorumlar"
              subtitle={`${total.toLocaleString("tr-TR")} yorum · arama, filtre ve CSV export`}
            >
              <ReviewBrowser results={sentimentData.results} />
            </Card>
          </div>
        )}
      </main>
    </div>
  );
}
