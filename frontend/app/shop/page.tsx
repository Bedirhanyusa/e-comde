"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  Cpu, Shirt, BookOpen, Sparkles, Heart,
  Star, Store, ShoppingBag, Zap, Scale, ArrowLeft,
} from "lucide-react";
import { Navbar } from "@/components/Navbar";
import { getShopProducts } from "@/lib/api";
import { ShopCategory, ShopProduct } from "@/lib/types";
import { motion } from "framer-motion";

const ICONS: Record<string, React.ElementType> = {
  cpu: Cpu, shirt: Shirt, book: BookOpen, sparkles: Sparkles, heart: Heart,
};

const container = {
  hidden: {},
  show: { transition: { staggerChildren: 0.07 } },
};
const cardItem = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4 } },
};

function StarRating({ rating, size = "sm" }: { rating: number; size?: "sm" | "md" }) {
  const sz = size === "md" ? "w-4 h-4" : "w-3.5 h-3.5";
  return (
    <div className="flex items-center gap-0.5">
      {[1, 2, 3, 4, 5].map((s) => (
        <Star key={s} className={`${sz} ${
          s <= Math.round(rating)
            ? "text-amber-400 fill-amber-400"
            : "text-[var(--border)] fill-[var(--border)]"
        }`} />
      ))}
      <span className={`ml-1 font-bold tabular-nums ${size === "md" ? "text-sm" : "text-xs"} text-[var(--text)]`}>
        {rating.toFixed(1)}
      </span>
    </div>
  );
}

function RatingBadge({ rating }: { rating: number }) {
  if (rating >= 4.5) return (
    <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-emerald-500 text-white shadow-sm shadow-emerald-500/30">
      ⭐ Çok İyi
    </span>
  );
  if (rating >= 4.0) return (
    <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-blue-500 text-white shadow-sm shadow-blue-500/30">
      👍 İyi
    </span>
  );
  if (rating >= 3.3) return (
    <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-amber-500 text-white shadow-sm shadow-amber-500/30">
      Orta
    </span>
  );
  return (
    <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-red-500 text-white shadow-sm shadow-red-500/30">
      Düşük
    </span>
  );
}

function ProductCard({
  product, category, compareSlot, onCompareSelect,
}: {
  product: ShopProduct;
  category: ShopCategory | undefined;
  compareSlot?: "A" | "B" | null;
  onCompareSelect?: (product: ShopProduct) => void;
}) {
  const router = useRouter();
  const [imgError, setImgError] = useState(false);
  const Icon = category ? (ICONS[category.icon] ?? ShoppingBag) : ShoppingBag;
  const color = category?.color ?? "#8B5CF6";

  const handleAnalyze = () => {
    router.push(`/?product_url=/shop/products/${product.id}`);
  };

  return (
    <motion.div
      variants={cardItem}
      whileHover={{ y: -5, transition: { duration: 0.2 } }}
      className="group bg-[var(--surface)] border border-[var(--border)] rounded-3xl overflow-hidden shadow-sm hover:shadow-2xl hover:border-transparent transition-all duration-300 flex flex-col"
      style={{ ["--hover-shadow" as string]: `0 20px 60px ${color}15` }}
    >
      {/* Product Image */}
      <div className="relative h-44 overflow-hidden bg-gradient-to-br"
        style={{ background: `linear-gradient(135deg, ${color}10, ${color}22)` }}>
        {product.image && !imgError ? (
          <img
            src={product.image}
            alt={product.name}
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
            onError={() => setImgError(true)}
            loading="lazy"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <div className="w-16 h-16 rounded-2xl flex items-center justify-center"
              style={{ background: `${color}20`, border: `1px solid ${color}30` }}>
              <Icon className="w-8 h-8" style={{ color }} />
            </div>
          </div>
        )}

        {/* Overlay badges */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
        <div className="absolute top-3 left-3">
          <RatingBadge rating={product.avg_rating} />
        </div>
        <div className="absolute top-3 right-3">
          <span className="text-[10px] font-semibold px-2 py-1 rounded-full bg-black/40 text-white backdrop-blur-sm">
            {product.total_reviews.toLocaleString("tr-TR")} yorum
          </span>
        </div>
        {product.price && (
          <div className="absolute bottom-3 right-3">
            <span className="text-sm font-bold px-3 py-1 rounded-full bg-white/90 dark:bg-black/70 text-[var(--text)] backdrop-blur-sm shadow-sm">
              {product.price}
            </span>
          </div>
        )}
      </div>

      {/* Content */}
      <div className="flex flex-col flex-1 p-4 gap-2.5">
        {/* Brand */}
        {product.brand && (
          <span className="text-xs font-semibold text-violet-500 dark:text-violet-400 uppercase tracking-wide">
            {product.brand}
          </span>
        )}

        {/* Name */}
        <h3 className="font-bold text-sm text-[var(--text)] leading-snug line-clamp-2 group-hover:text-violet-600 dark:group-hover:text-violet-400 transition-colors">
          {product.name}
        </h3>

        {/* Rating */}
        <StarRating rating={product.avg_rating} />

        {/* Description */}
        {product.description && (
          <p className="text-xs text-[var(--text-muted)] leading-relaxed line-clamp-2">
            {product.description}
          </p>
        )}

        {/* Features */}
        {product.features && product.features.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {product.features.slice(0, 3).map((f, i) => (
              <span key={i} className="text-[10px] font-medium px-1.5 py-0.5 rounded-md bg-[var(--bg)] border border-[var(--border)] text-[var(--text-muted)]">
                {f}
              </span>
            ))}
          </div>
        )}

        {/* CTA */}
        <div className="mt-auto pt-2 flex flex-col gap-2">
          {compareSlot ? (
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.97 }}
              onClick={() => onCompareSelect?.(product)}
              className="flex items-center justify-center gap-2 w-full py-2.5 rounded-2xl text-sm font-bold text-white transition-all shadow-md"
              style={{ background: `linear-gradient(135deg, #6366F1, #7C3AED)` }}
            >
              <Scale className="w-3.5 h-3.5" />
              Ürün {compareSlot} Olarak Seç
            </motion.button>
          ) : (
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
            onClick={handleAnalyze}
            className={`flex items-center justify-center gap-2 w-full py-2.5 rounded-2xl text-sm font-bold transition-all ${
              false
                ? "bg-emerald-500 text-white shadow-lg shadow-emerald-500/25"
                : "text-white shadow-md hover:shadow-lg"
            }`}
            style={{
              background: `linear-gradient(135deg, ${color}, ${color}cc)`,
              boxShadow: `0 4px 15px ${color}35`,
            }}
          >
            <Zap className="w-3.5 h-3.5" />Analiz Et
          </motion.button>
          )}
        </div>
      </div>
    </motion.div>
  );
}

export default function ShopPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const compareSlot = (searchParams.get("compareSlot") as "A" | "B" | null) ?? null;

  const [categories, setCategories] = useState<ShopCategory[]>([]);
  const [products, setProducts] = useState<ShopProduct[]>([]);
  const [selectedCat, setSelectedCat] = useState<string>("all");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    getShopProducts()
      .then((d) => { setCategories(d.categories); setProducts(d.products); setLoading(false); })
      .catch(() => { setError("Ürünler yüklenemedi. Backend çalışıyor mu?"); setLoading(false); });
  }, []);

  const handleCompareSelect = (product: ShopProduct) => {
    const url = `/shop/products/${product.id}`;
    if (compareSlot === "A") router.push(`/compare?urlA=${encodeURIComponent(url)}`);
    else if (compareSlot === "B") router.push(`/compare?urlB=${encodeURIComponent(url)}`);
  };

  const filtered = selectedCat === "all" ? products : products.filter((p) => p.category_id === selectedCat);
  const catMap = Object.fromEntries(categories.map((c) => [c.id, c]));

  return (
    <div className="min-h-screen bg-[var(--bg)]">
      <Navbar />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-10">

        {/* Compare Mode Banner */}
        {compareSlot && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center gap-3 mb-6 px-5 py-3.5 rounded-2xl bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800"
          >
            <Scale className="w-5 h-5 text-indigo-600 dark:text-indigo-400 shrink-0" />
            <div className="flex-1">
              <p className="text-sm font-bold text-indigo-700 dark:text-indigo-300">
                Karşılaştırma Modu — Ürün {compareSlot}
              </p>
              <p className="text-xs text-indigo-600/70 dark:text-indigo-400/70">
                Karşılaştırmak istediğin ürünü seç
              </p>
            </div>
            <a href="/compare" className="flex items-center gap-1.5 text-xs font-semibold text-indigo-600 dark:text-indigo-400 hover:underline">
              <ArrowLeft className="w-3.5 h-3.5" />Geri Dön
            </a>
          </motion.div>
        )}

        {/* Hero */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-10"
        >
          {/* Top bar */}
          <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-6 mb-6">
            <div>
              <div className="flex items-center gap-2.5 mb-2">
                <motion.div
                  whileHover={{ rotate: 10, scale: 1.05 }}
                  className="w-10 h-10 rounded-2xl bg-gradient-to-br from-violet-500 to-violet-700 flex items-center justify-center shadow-md shadow-violet-500/20"
                >
                  <Store className="w-5 h-5 text-white" />
                </motion.div>
                <div>
                  <p className="text-xs text-violet-500 font-bold uppercase tracking-widest">Demo Mağaza</p>
                  <span className="text-2xl font-extrabold tracking-tight text-[var(--text)]">
                    İstün<span className="text-gradient">Shop</span>
                  </span>
                </div>
              </div>
              <h1 className="text-2xl sm:text-3xl font-extrabold text-[var(--text)] leading-tight">
                Ürünü seç, AI ile analiz et
              </h1>
              <p className="text-[var(--text-muted)] mt-1 max-w-xl text-sm">
                Bir ürüne tıkla → <span className="text-violet-500 font-semibold">Analiz Et</span> → Yapay zeka saniyeler içinde tüm yorumları analiz eder.
              </p>
            </div>
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 }}
              className="flex items-center gap-3 shrink-0"
            >
              <div className="text-center px-4 py-2.5 bg-[var(--surface)] border border-[var(--border)] rounded-2xl shadow-sm">
                <p className="text-2xl font-extrabold text-violet-600">{products.length}</p>
                <p className="text-xs text-[var(--text-muted)]">Ürün</p>
              </div>
              <div className="text-center px-4 py-2.5 bg-[var(--surface)] border border-[var(--border)] rounded-2xl shadow-sm">
                <p className="text-2xl font-extrabold text-violet-600">{categories.length}</p>
                <p className="text-xs text-[var(--text-muted)]">Kategori</p>
              </div>
            </motion.div>
          </div>

          {/* How it works */}
          <div className="flex items-center gap-2 px-5 py-3.5 bg-gradient-to-r from-violet-50 to-indigo-50 dark:from-violet-900/10 dark:to-indigo-900/10 border border-violet-200/60 dark:border-violet-800/40 rounded-2xl">
            {[
              { icon: Store, num: "1", text: "Aşağıdan ürün seç" },
              { icon: Zap, num: "2", text: '"Analiz Et" butonuna bas' },
              { icon: Scale, num: "3", text: "Yapay zeka sonuçları gösterir" },
            ].map(({ icon: Icon, num, text }, i) => (
              <div key={i} className="flex items-center gap-2">
                <div className="flex items-center gap-1.5">
                  <span className="w-6 h-6 rounded-full bg-violet-600 text-white text-[10px] font-extrabold flex items-center justify-center shrink-0">
                    {num}
                  </span>
                  <span className="text-xs font-medium text-[var(--text)] hidden sm:block">{text}</span>
                </div>
                {i < 2 && <span className="text-violet-300 dark:text-violet-700 mx-1">›</span>}
              </div>
            ))}
          </div>
        </motion.div>

        {/* Category tabs */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="flex flex-wrap gap-2 mb-8"
        >
          <button
            onClick={() => setSelectedCat("all")}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm font-semibold transition-all border ${
              selectedCat === "all"
                ? "bg-violet-600 text-white border-violet-600 shadow-md shadow-violet-500/20"
                : "bg-[var(--surface)] border-[var(--border)] text-[var(--text-muted)] hover:border-violet-300 hover:text-[var(--text)]"
            }`}
          >
            <Store className="w-3.5 h-3.5" />
            Tüm Ürünler
            <span className={`text-[10px] px-1.5 py-0.5 rounded-md font-bold ${selectedCat === "all" ? "bg-white/20" : "bg-[var(--bg)]"}`}>
              {products.length}
            </span>
          </button>
          {categories.map((cat) => {
            const Icon = ICONS[cat.icon] ?? ShoppingBag;
            const active = selectedCat === cat.id;
            const count = products.filter(p => p.category_id === cat.id).length;
            return (
              <motion.button
                key={cat.id}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.97 }}
                onClick={() => setSelectedCat(cat.id)}
                className={`flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm font-semibold transition-all border ${
                  active
                    ? "text-white border-transparent shadow-md"
                    : "bg-[var(--surface)] border-[var(--border)] text-[var(--text-muted)] hover:border-violet-300 hover:text-[var(--text)]"
                }`}
                style={active ? {
                  background: `linear-gradient(135deg, ${cat.color}, ${cat.color}cc)`,
                  boxShadow: `0 4px 15px ${cat.color}30`,
                } : {}}
              >
                <Icon className="w-3.5 h-3.5" />
                {cat.label}
                <span className={`text-[10px] px-1.5 py-0.5 rounded-md font-bold ${active ? "bg-white/20" : "bg-[var(--bg)]"}`}>
                  {count}
                </span>
              </motion.button>
            );
          })}
        </motion.div>

        {/* Error */}
        {error && (
          <div className="mb-6 p-4 rounded-2xl bg-red-50 dark:bg-red-900/10 border border-red-200 text-red-600 text-sm font-medium">
            ⚠️ {error}
          </div>
        )}

        {/* Loading skeleton */}
        {loading && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="bg-[var(--surface)] border border-[var(--border)] rounded-3xl overflow-hidden">
                <div className="h-44 bg-[var(--bg)] shimmer-bg" />
                <div className="p-4 space-y-3">
                  <div className="h-3 w-16 rounded bg-[var(--bg)] shimmer-bg" />
                  <div className="h-4 w-full rounded bg-[var(--bg)] shimmer-bg" />
                  <div className="h-4 w-3/4 rounded bg-[var(--bg)] shimmer-bg" />
                  <div className="h-9 w-full rounded-2xl bg-[var(--bg)] shimmer-bg" />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Product grid */}
        {!loading && filtered.length === 0 && !error && (
          <div className="flex flex-col items-center justify-center py-24 text-[var(--text-muted)]">
            <ShoppingBag className="w-12 h-12 mb-4 opacity-20" />
            <p className="text-sm">Bu kategoride ürün yok.</p>
          </div>
        )}

        {!loading && filtered.length > 0 && (
          <>
            <p className="text-xs text-[var(--text-muted)] mb-4 font-medium">
              {filtered.length} ürün gösteriliyor
            </p>
            <motion.div
              key={selectedCat}
              variants={container}
              initial="hidden"
              animate="show"
              className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5"
            >
              {filtered.map((product) => (
                <ProductCard key={product.id} product={product} category={catMap[product.category_id]} compareSlot={compareSlot} onCompareSelect={handleCompareSelect} />
              ))}
            </motion.div>
          </>
        )}
      </main>
    </div>
  );
}
