"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Loader2, Clock, Link2, ArrowRight, Store, FileDown, Scale, Zap, BarChart3, MessageSquare } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useSearchParams, useRouter } from "next/navigation";
import { Navbar } from "@/components/Navbar";
import { UploadBar } from "@/components/UploadBar";
import { WarningBanner } from "@/components/WarningBanner";
import { ScoreSummaryCard } from "@/components/ScoreSummaryCard";
import { CategoryScores } from "@/components/CategoryScores";
import { ProConSection } from "@/components/ProConSection";
import { FeaturedReviews } from "@/components/FeaturedReviews";
import { ReviewBrowser } from "@/components/ReviewBrowser";
import { FakeReviewPanel } from "@/components/FakeReviewPanel";
import { SentimentDistributionChart } from "@/components/charts/SentimentDistributionChart";
import { ConfidenceHistogram } from "@/components/charts/ConfidenceHistogram";
import { EmotionDistributionChart } from "@/components/charts/EmotionDistributionChart";
import { summarizeReviews, analyzeByUrl, getAdvisor, getEmotions } from "@/lib/api";
import { SentimentResponse, SummarizeResponse, AdvisorResponse, EmotionResponse, ShopAnalysisResponse } from "@/lib/types";
import { AdvisorCard } from "@/components/AdvisorCard";
import {
  calcSentimentScore,
  splitPoints,
  calcCategoryScores,
  calcQualityStats,
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
  const searchParams = useSearchParams();
  const router = useRouter();
  const [sentimentData, setSentimentData] = useState<SentimentResponse | null>(null);
  const [summaryData, setSummaryData] = useState<SummarizeResponse | null>(null);
  const [advisorData, setAdvisorData] = useState<AdvisorResponse | null>(null);
  const [emotionData, setEmotionData] = useState<EmotionResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [shopProduct, setShopProduct] = useState<ShopAnalysisResponse["product"] | null>(null);
  const [urlInput, setUrlInput] = useState("");
  const [urlError, setUrlError] = useState<string | null>(null);
  const analyzedUrl = useRef<string | null>(null);

  // URL param'dan gelen ürün analizi (İstünShop'tan navigate)
  useEffect(() => {
    const productUrl = searchParams.get("product_url");
    if (!productUrl || analyzedUrl.current === productUrl) return;
    analyzedUrl.current = productUrl;
    setUrlInput(productUrl);
    handleUrlAnalyze(productUrl);
    // URL temizle (history'e product_url kalmasın)
    router.replace("/", { scroll: false });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams]);

  const handleUrlAnalyze = useCallback(async (url?: string) => {
    const target = (url ?? urlInput).trim();
    if (!target) return;
    setUrlError(null);
    setIsAnalyzing(true);
    setSentimentData(null);
    setSummaryData(null);
    setAdvisorData(null);
    setEmotionData(null);
    setShopProduct(null);
    try {
      // /shop/products/{id} formatını tanı (İstünShop linki)
      const shopMatch = target.match(/\/shop\/products\/(\d+)/);
      let data: ShopAnalysisResponse;
      if (shopMatch) {
        const { analyzeShopProduct } = await import("@/lib/api");
        data = await analyzeShopProduct(shopMatch[1]);
      } else {
        data = await analyzeByUrl(target);
      }
      setSentimentData({ results: data.results, total: data.total, flagged_count: data.flagged_count });
      setSummaryData({ summaries: data.summaries, overall: data.overall ?? "", pros: data.pros ?? [], cons: data.cons ?? [] });
      setShopProduct(data.product);
      // Advisor + Emotions: arka planda paralel çağır
      const sentScore = calcSentimentScore(data.results);
      const texts = data.results.map(r => r.text);
      const lbls = data.results.map(r => r.label);
      getAdvisor(texts, lbls, sentScore).then(setAdvisorData).catch(() => {});
      getEmotions(texts, lbls).then(setEmotionData).catch(() => {});
    } catch (e: unknown) {
      setUrlError(e instanceof Error ? e.message : "Bu link analiz edilemedi. İstünShop'tan bir ürün linki kopyaladığınızdan emin olun.");
    } finally {
      setIsAnalyzing(false);
    }
  }, [urlInput]);

  const handleResult = async (data: SentimentResponse) => {
    setSentimentData(data);
    setSummaryData(null);
    setAdvisorData(null);
    setIsSummarizing(true);
    try {
      const texts = data.results.map((r) => r.text);
      const labels = data.results.map((r) => r.label);
      const sentScore = calcSentimentScore(data.results);
      const [summary, advisor, emotions] = await Promise.allSettled([
        summarizeReviews(texts, labels),
        getAdvisor(texts, labels, sentScore),
        getEmotions(texts, labels),
      ]);
      if (summary.status === "fulfilled") setSummaryData(summary.value);
      if (advisor.status === "fulfilled") setAdvisorData(advisor.value);
      if (emotions.status === "fulfilled") setEmotionData(emotions.value);
    } catch {
      // sessiz geç
    } finally {
      setIsSummarizing(false);
    }
  };

  const handleReset = () => {
    setSentimentData(null);
    setSummaryData(null);
    setAdvisorData(null);
    setEmotionData(null);
    setIsAnalyzing(false);
    setIsSummarizing(false);
    setShopProduct(null);
  };

  // Derived values
  const score = sentimentData ? calcSentimentScore(sentimentData.results) : 0;
  const categoryScores = sentimentData ? calcCategoryScores(sentimentData.results) : [];
  const qualityStats = sentimentData ? calcQualityStats(sentimentData.results) : null;
  const pros = summaryData?.pros ?? [];
  const cons = summaryData?.cons ?? [];
  const warnings = cons.slice(0, 3);
  const total = sentimentData?.total ?? 0;
  const overallSummary = summaryData?.overall ?? "";

  return (
    <div className="min-h-screen bg-[var(--bg)]">
      <Navbar onNewAnalysis={sentimentData ? handleReset : undefined} />

      <main className="max-w-6xl mx-auto px-4 sm:px-6">

        {/* ── LANDING ── */}
        <AnimatePresence>
        {!sentimentData && (
          <div className="flex flex-col items-center justify-center min-h-[calc(100vh-64px)] text-center py-16 relative">
            {/* Background glow blobs */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
              <div className="absolute -top-20 left-1/2 -translate-x-1/2 w-[600px] h-[400px] bg-violet-500/5 dark:bg-violet-500/10 rounded-full blur-3xl" />
              <div className="absolute top-1/2 -left-20 w-[300px] h-[300px] bg-indigo-400/5 dark:bg-indigo-400/8 rounded-full blur-3xl" />
              <div className="absolute top-1/2 -right-20 w-[300px] h-[300px] bg-purple-400/5 dark:bg-purple-400/8 rounded-full blur-3xl" />
            </div>

            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              className="flex items-center gap-2 px-4 py-1.5 rounded-full bg-violet-100 dark:bg-violet-900/30 border border-violet-200 dark:border-violet-800/60 text-violet-600 dark:text-violet-400 text-xs font-semibold mb-6"
            >
              <Zap className="w-3 h-3" />
              BERTurk + Claude AI · Türkçe E-Ticaret Analizi
            </motion.div>

            {/* Heading */}
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="text-5xl sm:text-6xl font-extrabold text-[var(--text)] leading-[1.1] mb-5 max-w-3xl tracking-tight"
            >
              Yorumları{" "}
              <span className="text-gradient">yapay zeka</span>
              {" "}ile{" "}
              <br className="hidden sm:block" />
              saniyeler içinde analiz edin.
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="text-[var(--text-muted)] text-lg mb-8 max-w-xl leading-relaxed"
            >
              Ürün linki yapıştırın ya da CSV yükleyin. BERTurk her yorumu sınıflandırır,
              yapay zeka özet ve alışveriş tavsiyesi üretir.
            </motion.p>

            {/* Feature pills */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
              className="flex flex-wrap justify-center gap-2 mb-10"
            >
              {[
                { icon: BarChart3, label: "Duygu Analizi" },
                { icon: MessageSquare, label: "AI Özet" },
                { icon: Scale, label: "Ürün Karşılaştırma" },
                { icon: FileDown, label: "PDF Rapor" },
              ].map(({ icon: Icon, label }, i) => (
                <motion.span
                  key={label}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.35 + i * 0.06 }}
                  className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-[var(--surface)] border border-[var(--border)] text-[var(--text-muted)]"
                >
                  <Icon className="w-3 h-3 text-violet-500" />
                  {label}
                </motion.span>
              ))}
            </motion.div>

            {/* URL Input */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.35 }}
              className="w-full max-w-2xl mb-4"
            >
              <div className="flex gap-2 bg-[var(--surface)] border border-[var(--border)] rounded-2xl p-2 shadow-lg focus-within:border-violet-400 focus-within:shadow-violet-500/10 focus-within:shadow-xl transition-all duration-300">
                <div className="flex items-center pl-2 shrink-0">
                  <Link2 className="w-4 h-4 text-violet-500" />
                </div>
                <input
                  type="url"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleUrlAnalyze()}
                  placeholder="İstünShop veya Trendyol ürün linkini yapıştırın..."
                  className="flex-1 bg-transparent text-sm text-[var(--text)] placeholder:text-[var(--text-muted)] outline-none py-2 px-1"
                />
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.97 }}
                  onClick={() => handleUrlAnalyze()}
                  disabled={!urlInput.trim() || isAnalyzing}
                  className="flex items-center gap-1.5 px-5 py-2 rounded-xl text-sm font-semibold text-white bg-gradient-to-r from-violet-600 to-violet-500 hover:from-violet-700 hover:to-violet-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-sm shrink-0"
                >
                  <ArrowRight className="w-4 h-4" />
                  Analiz Et
                </motion.button>
              </div>
              {urlError && (
                <motion.p
                  initial={{ opacity: 0, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-2 text-sm text-red-500 text-left px-2"
                >
                  {urlError}
                </motion.p>
              )}
              <div className="flex items-center gap-2 mt-2 px-2">
                <Store className="w-3.5 h-3.5 text-violet-400" />
                <p className="text-xs text-[var(--text-muted)]">
                  Hazır ürün kataloğu için{" "}
                  <a href="/shop" className="text-violet-500 hover:underline font-medium">İstünShop</a>
                  &apos;u ziyaret edin.
                </p>
              </div>
            </motion.div>

            {/* Divider */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.45 }}
              className="flex items-center gap-3 w-full max-w-2xl my-4"
            >
              <div className="flex-1 h-px bg-[var(--border)]" />
              <span className="text-xs text-[var(--text-muted)] font-medium px-2">veya CSV yükle</span>
              <div className="flex-1 h-px bg-[var(--border)]" />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
            >
              <UploadBar onResult={handleResult} onLoadingChange={setIsAnalyzing} />
            </motion.div>

            {/* Loading */}
            {isAnalyzing && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="mt-16 flex flex-col items-center gap-4"
              >
                <div className="relative w-20 h-20 flex items-center justify-center">
                  <div className="absolute inset-0 rounded-full border-4 border-violet-100 dark:border-violet-900/40" />
                  <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-violet-600 animate-spin" />
                  <Loader2 className="w-8 h-8 text-violet-600 animate-spin" />
                </div>
                <div>
                  <p className="font-bold text-base text-[var(--text)]">Analiz Ediliyor</p>
                  <p className="text-sm text-[var(--text-muted)] mt-1 max-w-xs animate-pulse">
                    BERTurk yorumları tarıyor, Claude özet hazırlıyor...
                  </p>
                </div>
              </motion.div>
            )}
          </div>
        )}
        </AnimatePresence>

        {/* ── RESULTS ── */}
        {sentimentData && (
          <motion.div
            id="analysis-results"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.4 }}
            className="py-10 space-y-6"
          >

            {/* Toolbar: PDF + Compare */}
            <div className="flex items-center justify-end gap-2">
              <a
                href="/compare"
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold text-violet-600 dark:text-violet-400 border border-violet-200 dark:border-violet-800 hover:bg-violet-50 dark:hover:bg-violet-900/20 transition-colors"
              >
                <Scale className="w-3.5 h-3.5" />
                Karşılaştır
              </a>
              <button
                onClick={async () => {
                  const { downloadAnalysisPDF } = await import("@/lib/pdf");
                  const olumlu = sentimentData!.results.filter(r => r.label === "olumlu").length;
                  const olumsuz = sentimentData!.results.filter(r => r.label === "olumsuz").length;
                  const notr = sentimentData!.results.filter(r => r.label === "notr").length;
                  downloadAnalysisPDF({
                    score,
                    totalReviews: total,
                    flaggedCount: sentimentData!.flagged_count,
                    overallSummary,
                    pros,
                    cons,
                    summaries: {
                      olumlu: summaryData?.summaries?.olumlu,
                      olumsuz: summaryData?.summaries?.olumsuz,
                      notr: summaryData?.summaries?.notr,
                    },
                    advisor: advisorData,
                    sentimentCounts: { olumlu, olumsuz, notr },
                    shopProduct: shopProduct ? {
                      name: (shopProduct as { name: string; avg_rating: number; total_reviews: number }).name,
                      avg_rating: (shopProduct as { name: string; avg_rating: number; total_reviews: number }).avg_rating,
                      total_reviews: (shopProduct as { name: string; avg_rating: number; total_reviews: number }).total_reviews,
                    } : null,
                  });
                }}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold text-violet-600 dark:text-violet-400 border border-violet-200 dark:border-violet-800 hover:bg-violet-50 dark:hover:bg-violet-900/20 transition-colors"
              >
                <FileDown className="w-3.5 h-3.5" />
                PDF İndir
              </button>
            </div>

            {/* İstünShop ürün banner */}
            {shopProduct && (
              <div className="flex items-center gap-3 px-5 py-3.5 rounded-2xl bg-violet-50 dark:bg-violet-900/20 border border-violet-200 dark:border-violet-800">
                <div className="w-8 h-8 rounded-xl bg-violet-600 flex items-center justify-center shrink-0">
                  <span className="text-white text-xs font-bold">İS</span>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-violet-500 font-semibold uppercase tracking-wide">İstünShop Ürün Analizi</p>
                  <p className="text-sm font-semibold text-[var(--text)] truncate">{shopProduct.name}</p>
                </div>
                <div className="text-right shrink-0">
                  <p className="text-xs text-[var(--text-muted)]">{shopProduct.total_reviews.toLocaleString("tr-TR")} toplam yorum</p>
                  <p className="text-xs text-violet-600 font-semibold">★ {shopProduct.avg_rating.toFixed(1)}</p>
                </div>
              </div>
            )}

            {/* Warning Banner */}
            {warnings.length > 0 && <WarningBanner points={warnings} />}

            {/* Sahte Yorum Tespiti */}
            <FakeReviewPanel reviews={sentimentData.results} flaggedCount={sentimentData.flagged_count} />

            {/* Score + AI Summary */}
            {isSummarizing ? (
              <div className="bg-white dark:bg-[var(--surface)] border border-[var(--border)] rounded-3xl p-10 flex flex-col items-center gap-3 text-[var(--text-muted)]">
                <div className="p-3 rounded-2xl bg-violet-50 dark:bg-violet-900/30">
                  <Loader2 className="w-6 h-6 animate-spin text-violet-500" />
                </div>
                <p className="text-sm font-medium">Yapay zeka analiz hazırlıyor<span className="animate-pulse">...</span></p>
              </div>
            ) : overallSummary ? (
              <ScoreSummaryCard
                score={score}
                totalReviews={total}
                overallSummary={overallSummary}
              />
            ) : null}

            {/* Yorum Kalitesi göstergesi */}
            {qualityStats && qualityStats.total > 0 && (
              <div className="flex items-center gap-3 px-5 py-3 rounded-2xl bg-[var(--surface)] border border-[var(--border)]">
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-xs font-semibold text-[var(--text)]">Yorum Kalite Analizi</span>
                    <span className="text-xs text-[var(--text-muted)]">
                      {qualityStats.highQuality} / {qualityStats.total} yüksek kaliteli yorum
                    </span>
                  </div>
                  <div className="w-full h-1.5 rounded-full bg-black/10 dark:bg-white/10 overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${(qualityStats.highQuality / qualityStats.total) * 100}%` }}
                      transition={{ duration: 0.8, ease: "easeOut" }}
                      className="h-full rounded-full bg-gradient-to-r from-violet-500 to-violet-400"
                    />
                  </div>
                  <p className="text-xs text-[var(--text-muted)] mt-1.5">
                    Detaylı, spesifik yorumlar analizde daha yüksek ağırlık taşır.
                  </p>
                </div>
              </div>
            )}

            {/* Advisor Card */}
            {advisorData && <AdvisorCard advisor={advisorData} />}

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
              <div className={`grid grid-cols-1 ${emotionData ? "md:grid-cols-2" : ""} gap-6`}>
                <Card title="Güven Skoru Dağılımı" subtitle="BERTurk model tahmin güveni histogramı">
                  <ConfidenceHistogram results={sentimentData.results} />
                </Card>
                {emotionData && (
                  <Card title="Duygu Derinliği" subtitle="5 duygusal kategori — Claude AI ile sınıflandırma">
                    <EmotionDistributionChart data={emotionData} />
                  </Card>
                )}
              </div>
            )}

            {/* Emotion chart when no category scores */}
            {categoryScores.length === 0 && emotionData && (
              <Card title="Duygu Derinliği" subtitle="5 duygusal kategori — Claude AI ile sınıflandırma">
                <EmotionDistributionChart data={emotionData} />
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
          </motion.div>
        )}
      </main>
    </div>
  );
}
