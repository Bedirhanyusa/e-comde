import jsPDF from "jspdf";

export interface PDFReportData {
  score: number;
  totalReviews: number;
  flaggedCount: number;
  overallSummary: string;
  pros: string[];
  cons: string[];
  summaries: { olumlu?: string; olumsuz?: string; notr?: string };
  advisor?: {
    verdict: string;
    target_audience: string;
    buy_if: string[];
    watch_out: string[];
  } | null;
  sentimentCounts: { olumlu: number; olumsuz: number; notr: number };
  shopProduct?: { name: string; avg_rating: number; total_reviews: number } | null;
}

// ── Turkce karakter normalizasyonu ──────────────────────────────────────────

function t(text: string): string {
  return text
    .replace(/ğ/g, "g").replace(/Ğ/g, "G")
    .replace(/ş/g, "s").replace(/Ş/g, "S")
    .replace(/ı/g, "i").replace(/İ/g, "I")
    .replace(/ö/g, "o").replace(/Ö/g, "O")
    .replace(/ü/g, "u").replace(/Ü/g, "U")
    .replace(/ç/g, "c").replace(/Ç/g, "C")
    .replace(/â/g, "a").replace(/Â/g, "A")
    .replace(/î/g, "i").replace(/Î/g, "I")
    .replace(/û/g, "u").replace(/Û/g, "U");
}

function scoreColor(score: number): [number, number, number] {
  if (score >= 4) return [22, 163, 74];
  if (score >= 3) return [217, 119, 6];
  return [220, 38, 38];
}

function verdictColor(verdict: string): [number, number, number] {
  if (verdict.includes("Tavsiye Edilir")) return [22, 163, 74];
  if (verdict.includes("Dikkatli")) return [217, 119, 6];
  return [220, 38, 38];
}

// ── PDF Engine ─────────────────────────────────────────────────────────────

export async function downloadAnalysisPDF(
  data: PDFReportData,
  filename = "ecomde-analiz-raporu.pdf"
) {
  const pdf = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  const W = pdf.internal.pageSize.getWidth();
  const H = pdf.internal.pageSize.getHeight();
  const M = 14;
  const CW = W - M * 2;
  let y = 0;

  const ensureSpace = (needed: number) => {
    if (y + needed > H - 18) {
      pdf.addPage();
      y = 14;
    }
  };

  const sectionTitle = (title: string, icon?: string) => {
    ensureSpace(18);
    y += 3;
    pdf.setFillColor(109, 40, 217);
    pdf.roundedRect(M, y, 1.2, 8, 0.6, 0.6, "F");
    pdf.setFontSize(11);
    pdf.setFont("helvetica", "bold");
    pdf.setTextColor(30, 30, 30);
    pdf.text((icon ? icon + "  " : "") + t(title), M + 4, y + 6);
    y += 12;
  };

  // ═══════════════════════════════════════════════════════════════════════════
  //  HEADER BAR
  // ═══════════════════════════════════════════════════════════════════════════
  pdf.setFillColor(109, 40, 217);
  pdf.rect(0, 0, W, 22, "F");

  // Logo box
  pdf.setFillColor(139, 92, 246);
  pdf.roundedRect(M, 4.5, 13, 13, 3, 3, "F");
  pdf.setFontSize(9);
  pdf.setTextColor(255, 255, 255);
  pdf.setFont("helvetica", "bold");
  pdf.text("E", M + 4.8, 12.5);

  // Title
  pdf.setFontSize(14);
  pdf.text("E-ComDe", M + 16, 11);
  pdf.setFontSize(8);
  pdf.setFont("helvetica", "normal");
  pdf.text("Yapay Zeka Analiz Raporu", M + 16, 16);

  // Date right
  pdf.setFontSize(8);
  const now = new Date();
  const dateStr = t(now.toLocaleDateString("tr-TR", { day: "2-digit", month: "long", year: "numeric" }));
  const timeStr = now.toLocaleTimeString("tr-TR", { hour: "2-digit", minute: "2-digit" });
  pdf.text(dateStr + "  " + timeStr, W - M, 11, { align: "right" });
  pdf.setFontSize(7);
  pdf.text("BERTurk v3 + Claude Haiku", W - M, 16, { align: "right" });

  y = 28;

  // ═══════════════════════════════════════════════════════════════════════════
  //  PRODUCT INFO (if shop product)
  // ═══════════════════════════════════════════════════════════════════════════
  if (data.shopProduct) {
    pdf.setFillColor(248, 246, 255);
    pdf.roundedRect(M, y, CW, 16, 3, 3, "F");
    pdf.setDrawColor(139, 92, 246);
    pdf.setLineWidth(0.4);
    pdf.roundedRect(M, y, CW, 16, 3, 3, "S");

    pdf.setFontSize(7);
    pdf.setTextColor(109, 40, 217);
    pdf.setFont("helvetica", "bold");
    pdf.text("ISTUNSHOP URUN ANALIZI", M + 5, y + 5.5);

    pdf.setTextColor(30, 30, 30);
    pdf.setFontSize(10);
    pdf.setFont("helvetica", "bold");
    const pName = t(data.shopProduct.name.length > 65 ? data.shopProduct.name.slice(0, 65) + "..." : data.shopProduct.name);
    pdf.text(pName, M + 5, y + 12);

    pdf.setFontSize(9);
    pdf.setTextColor(109, 40, 217);
    pdf.setFont("helvetica", "bold");
    const rating = data.shopProduct.avg_rating.toFixed(1) + " / 5";
    const reviews = data.shopProduct.total_reviews.toLocaleString("tr-TR") + " yorum";
    pdf.text(rating + "   |   " + reviews, W - M - 5, y + 12, { align: "right" });
    y += 22;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  SCORE + STATS ROW
  // ═══════════════════════════════════════════════════════════════════════════
  const sc = scoreColor(data.score);

  // Score circle
  pdf.setFillColor(...sc);
  pdf.circle(M + 14, y + 14, 14, "F");
  pdf.setFontSize(16);
  pdf.setFont("helvetica", "bold");
  pdf.setTextColor(255, 255, 255);
  pdf.text(data.score.toFixed(1), M + 14, y + 15.5, { align: "center" });
  pdf.setFontSize(6);
  pdf.text("/ 5.0", M + 14, y + 20, { align: "center" });

  // Score label
  const scoreLabel = data.score >= 4 ? "OLUMLU EGILIM" : data.score >= 3 ? "KARMA EGILIM" : "OLUMSUZ EGILIM";
  pdf.setTextColor(...sc);
  pdf.setFontSize(8);
  pdf.setFont("helvetica", "bold");
  pdf.text(scoreLabel, M + 32, y + 6);

  // Summary stats boxes
  const total = data.sentimentCounts.olumlu + data.sentimentCounts.olumsuz + data.sentimentCounts.notr;
  const stats = [
    { label: "Toplam Yorum", value: total.toLocaleString("tr-TR") },
    { label: "Olumlu", value: `%${total > 0 ? Math.round((data.sentimentCounts.olumlu / total) * 100) : 0}` },
    { label: "Olumsuz", value: `%${total > 0 ? Math.round((data.sentimentCounts.olumsuz / total) * 100) : 0}` },
    { label: t("Supheli"), value: String(data.flaggedCount) },
  ];

  const boxW = (CW - 34) / 4;
  stats.forEach((s, i) => {
    const bx = M + 32 + i * (boxW + 2);
    pdf.setFillColor(248, 249, 250);
    pdf.roundedRect(bx, y + 10, boxW, 18, 2, 2, "F");
    pdf.setFontSize(13);
    pdf.setFont("helvetica", "bold");
    pdf.setTextColor(30, 30, 30);
    pdf.text(s.value, bx + boxW / 2, y + 20, { align: "center" });
    pdf.setFontSize(6.5);
    pdf.setFont("helvetica", "normal");
    pdf.setTextColor(107, 114, 128);
    pdf.text(s.label, bx + boxW / 2, y + 25, { align: "center" });
  });

  y += 34;

  // ═══════════════════════════════════════════════════════════════════════════
  //  SENTIMENT DISTRIBUTION BARS
  // ═══════════════════════════════════════════════════════════════════════════
  sectionTitle("Duygu Dagilimi");

  const sentBars = [
    { label: "Olumlu", count: data.sentimentCounts.olumlu, color: [22, 163, 74] as [number, number, number] },
    { label: t("Notr"), count: data.sentimentCounts.notr, color: [156, 163, 175] as [number, number, number] },
    { label: "Olumsuz", count: data.sentimentCounts.olumsuz, color: [220, 38, 38] as [number, number, number] },
  ];

  const barMaxW = CW * 0.50;
  sentBars.forEach(({ label, count, color }) => {
    const pct = total > 0 ? count / total : 0;
    const barW = Math.max(pct * barMaxW, pct > 0 ? 2 : 0);

    pdf.setFontSize(8);
    pdf.setFont("helvetica", "normal");
    pdf.setTextColor(107, 114, 128);
    pdf.text(label, M + 22, y + 4.5, { align: "right" });

    pdf.setFillColor(240, 240, 240);
    pdf.roundedRect(M + 24, y, barMaxW, 6, 2, 2, "F");
    if (barW > 0) {
      pdf.setFillColor(...color);
      pdf.roundedRect(M + 24, y, barW, 6, 2, 2, "F");
    }

    pdf.setTextColor(30, 30, 30);
    pdf.setFont("helvetica", "bold");
    pdf.setFontSize(8);
    pdf.text(`${(pct * 100).toFixed(1)}%  (${count})`, M + 24 + barMaxW + 4, y + 4.5);
    y += 10;
  });
  y += 2;

  // ═══════════════════════════════════════════════════════════════════════════
  //  AI SUMMARY
  // ═══════════════════════════════════════════════════════════════════════════
  if (data.overallSummary) {
    sectionTitle("Yapay Zeka Genel Ozeti");
    pdf.setFillColor(250, 249, 255);
    const lines = pdf.splitTextToSize(t(data.overallSummary), CW - 10);
    const h = lines.length * 4.8 + 8;
    ensureSpace(h + 4);
    pdf.roundedRect(M, y, CW, h, 3, 3, "F");
    pdf.setDrawColor(200, 180, 255);
    pdf.setLineWidth(0.3);
    pdf.line(M + 1, y + 3, M + 1, y + h - 3);
    pdf.setFontSize(8.5);
    pdf.setFont("helvetica", "normal");
    pdf.setTextColor(55, 65, 81);
    pdf.text(lines, M + 5, y + 6);
    y += h + 4;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  SENTIMENT SUMMARIES (olumlu / olumsuz)
  // ═══════════════════════════════════════════════════════════════════════════
  const hasOlumlu = data.summaries.olumlu?.trim();
  const hasOlumsuz = data.summaries.olumsuz?.trim();

  if (hasOlumlu || hasOlumsuz) {
    sectionTitle("Duygu Bazli Ozetler");

    if (hasOlumlu) {
      const lines = pdf.splitTextToSize(t(data.summaries.olumlu!), CW - 10);
      const h = lines.length * 4.5 + 10;
      ensureSpace(h + 2);
      pdf.setFillColor(240, 253, 244);
      pdf.roundedRect(M, y, CW, h, 3, 3, "F");
      pdf.setDrawColor(22, 163, 74);
      pdf.setLineWidth(0.5);
      pdf.line(M + 1, y + 3, M + 1, y + h - 3);
      pdf.setFontSize(7.5);
      pdf.setFont("helvetica", "bold");
      pdf.setTextColor(22, 163, 74);
      pdf.text("OLUMLU", M + 5, y + 5);
      pdf.setFont("helvetica", "normal");
      pdf.setTextColor(55, 65, 81);
      pdf.setFontSize(8);
      pdf.text(lines, M + 5, y + 10);
      y += h + 3;
    }

    if (hasOlumsuz) {
      const lines = pdf.splitTextToSize(t(data.summaries.olumsuz!), CW - 10);
      const h = lines.length * 4.5 + 10;
      ensureSpace(h + 2);
      pdf.setFillColor(254, 242, 242);
      pdf.roundedRect(M, y, CW, h, 3, 3, "F");
      pdf.setDrawColor(220, 38, 38);
      pdf.setLineWidth(0.5);
      pdf.line(M + 1, y + 3, M + 1, y + h - 3);
      pdf.setFontSize(7.5);
      pdf.setFont("helvetica", "bold");
      pdf.setTextColor(220, 38, 38);
      pdf.text("OLUMSUZ", M + 5, y + 5);
      pdf.setFont("helvetica", "normal");
      pdf.setTextColor(55, 65, 81);
      pdf.setFontSize(8);
      pdf.text(lines, M + 5, y + 10);
      y += h + 3;
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  PROS & CONS (two columns)
  // ═══════════════════════════════════════════════════════════════════════════
  if (data.pros.length > 0 || data.cons.length > 0) {
    sectionTitle("One Cikan Arti ve Eksiler");

    const colW = (CW - 6) / 2;

    // Calculate heights first
    let prosH = 10;
    data.pros.forEach(p => { prosH += pdf.splitTextToSize(t(p), colW - 12).length * 4 + 3; });
    let consH = 10;
    data.cons.forEach(c => { consH += pdf.splitTextToSize(t(c), colW - 12).length * 4 + 3; });
    const maxH = Math.max(prosH, consH) + 2;
    ensureSpace(maxH + 4);

    // Pros box
    pdf.setFillColor(240, 253, 244);
    pdf.roundedRect(M, y, colW, maxH, 3, 3, "F");
    pdf.setFontSize(8);
    pdf.setFont("helvetica", "bold");
    pdf.setTextColor(22, 163, 74);
    pdf.text("+  Artilar", M + 5, y + 7);
    let py = y + 13;
    data.pros.forEach(p => {
      pdf.setFont("helvetica", "normal");
      pdf.setTextColor(55, 65, 81);
      pdf.setFontSize(7.5);
      const lines = pdf.splitTextToSize("  " + t(p), colW - 12);
      pdf.text(lines, M + 5, py);
      py += lines.length * 4 + 3;
    });

    // Cons box
    const cx = M + colW + 6;
    pdf.setFillColor(254, 242, 242);
    pdf.roundedRect(cx, y, colW, maxH, 3, 3, "F");
    pdf.setFontSize(8);
    pdf.setFont("helvetica", "bold");
    pdf.setTextColor(220, 38, 38);
    pdf.text("-  Eksiler", cx + 5, y + 7);
    let cy2 = y + 13;
    data.cons.forEach(c => {
      pdf.setFont("helvetica", "normal");
      pdf.setTextColor(55, 65, 81);
      pdf.setFontSize(7.5);
      const lines = pdf.splitTextToSize("  " + t(c), colW - 12);
      pdf.text(lines, cx + 5, cy2);
      cy2 += lines.length * 4 + 3;
    });

    y += maxH + 4;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  AI ADVISOR
  // ═══════════════════════════════════════════════════════════════════════════
  if (data.advisor) {
    sectionTitle("Yapay Zeka Alisveris Danismani");

    // Verdict badge
    const vc = verdictColor(data.advisor.verdict);
    ensureSpace(60);

    pdf.setFillColor(...vc);
    const verdictText = t(data.advisor.verdict);
    const vW = pdf.getTextWidth(verdictText) * 1.1 + 14;
    pdf.roundedRect(M, y, vW, 10, 3, 3, "F");
    pdf.setFontSize(10);
    pdf.setFont("helvetica", "bold");
    pdf.setTextColor(255, 255, 255);
    pdf.text(verdictText, M + 7, y + 7);
    y += 14;

    // Target audience
    pdf.setFillColor(248, 246, 255);
    const tgtLines = pdf.splitTextToSize(t(data.advisor.target_audience), CW - 10);
    const tgtH = tgtLines.length * 4.5 + 6;
    pdf.roundedRect(M, y, CW, tgtH, 3, 3, "F");
    pdf.setFontSize(8);
    pdf.setFont("helvetica", "normal");
    pdf.setTextColor(55, 65, 81);
    pdf.text(tgtLines, M + 5, y + 5);
    y += tgtH + 4;

    // Buy if / Watch out (two columns)
    const colW2 = (CW - 6) / 2;
    let buyH = 10;
    data.advisor.buy_if.forEach(b => { buyH += pdf.splitTextToSize(t(b), colW2 - 12).length * 4 + 3; });
    let watchH = 10;
    data.advisor.watch_out.forEach(w => { watchH += pdf.splitTextToSize(t(w), colW2 - 12).length * 4 + 3; });
    const advH = Math.max(buyH, watchH) + 2;
    ensureSpace(advH + 4);

    if (data.advisor.buy_if.length > 0) {
      pdf.setFillColor(240, 253, 244);
      pdf.roundedRect(M, y, colW2, advH, 3, 3, "F");
      pdf.setFontSize(7.5);
      pdf.setFont("helvetica", "bold");
      pdf.setTextColor(22, 163, 74);
      pdf.text("Al:", M + 5, y + 7);
      let by2 = y + 13;
      data.advisor.buy_if.forEach(item => {
        pdf.setFont("helvetica", "normal");
        pdf.setTextColor(55, 65, 81);
        const lines = pdf.splitTextToSize("> " + t(item), colW2 - 12);
        pdf.text(lines, M + 5, by2);
        by2 += lines.length * 4 + 3;
      });
    }

    if (data.advisor.watch_out.length > 0) {
      const wx = M + colW2 + 6;
      pdf.setFillColor(255, 247, 237);
      pdf.roundedRect(wx, y, colW2, advH, 3, 3, "F");
      pdf.setFontSize(7.5);
      pdf.setFont("helvetica", "bold");
      pdf.setTextColor(217, 119, 6);
      pdf.text("Dikkat:", wx + 5, y + 7);
      let wy2 = y + 13;
      data.advisor.watch_out.forEach(item => {
        pdf.setFont("helvetica", "normal");
        pdf.setTextColor(55, 65, 81);
        const lines = pdf.splitTextToSize("> " + t(item), colW2 - 12);
        pdf.text(lines, wx + 5, wy2);
        wy2 += lines.length * 4 + 3;
      });
    }

    y += advH + 4;
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  FOOTER (all pages)
  // ═══════════════════════════════════════════════════════════════════════════
  const totalPages = (pdf as unknown as { internal: { getNumberOfPages: () => number } }).internal.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    pdf.setPage(i);

    // Footer bar
    pdf.setFillColor(109, 40, 217);
    pdf.rect(0, H - 10, W, 10, "F");
    pdf.setFontSize(7);
    pdf.setFont("helvetica", "normal");
    pdf.setTextColor(255, 255, 255);
    pdf.text("E-ComDe  |  BERTurk v3 + Claude Haiku  |  Istanbul Saglik ve Teknoloji Universitesi", M, H - 4);
    pdf.text(`${i} / ${totalPages}`, W - M, H - 4, { align: "right" });

    // Top accent line (pages after 1)
    if (i > 1) {
      pdf.setFillColor(109, 40, 217);
      pdf.rect(0, 0, W, 2, "F");
    }
  }

  pdf.save(filename);
}
