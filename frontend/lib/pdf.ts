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

// ── Yardımcı fonksiyonlar ──────────────────────────────────────────────────

function wrapText(pdf: jsPDF, text: string, x: number, maxWidth: number, lineHeight: number): number {
  const lines = pdf.splitTextToSize(text, maxWidth);
  pdf.text(lines, x, 0);
  return lines.length * lineHeight;
}

function drawHRule(pdf: jsPDF, y: number, pageW: number, margin: number, color = [229, 231, 235] as [number, number, number]) {
  pdf.setDrawColor(...color);
  pdf.setLineWidth(0.2);
  pdf.line(margin, y, pageW - margin, y);
}

function stars(score: number): string {
  const filled = Math.round(score);
  return "★".repeat(filled) + "☆".repeat(5 - filled) + `  ${score.toFixed(1)} / 5.0`;
}

function scoreColor(score: number): [number, number, number] {
  if (score >= 4) return [34, 197, 94];
  if (score >= 3) return [245, 158, 11];
  return [239, 68, 68];
}

function verdictColor(verdict: string): [number, number, number] {
  if (verdict === "Tavsiye Edilir") return [34, 197, 94];
  if (verdict === "Dikkatli Olun") return [245, 158, 11];
  return [239, 68, 68];
}

// ── Ana PDF üreteci ────────────────────────────────────────────────────────

export async function downloadAnalysisPDF(
  data: PDFReportData,
  filename = "ecomde-analiz-raporu.pdf"
) {
  const pdf = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  const pageW = pdf.internal.pageSize.getWidth();
  const pageH = pdf.internal.pageSize.getHeight();
  const margin = 16;
  const contentW = pageW - margin * 2;
  let y = 0;

  // ── HEADER ─────────────────────────────────────────────────────────────
  pdf.setFillColor(109, 40, 217);
  pdf.rect(0, 0, pageW, 20, "F");

  // Logo arka plan
  pdf.setFillColor(139, 92, 246);
  pdf.roundedRect(margin, 4, 12, 12, 2, 2, "F");
  pdf.setFontSize(7);
  pdf.setTextColor(255, 255, 255);
  pdf.setFont("helvetica", "bold");
  pdf.text("EC", margin + 3.8, 11.5);

  // Başlık
  pdf.setFontSize(12);
  pdf.setFont("helvetica", "bold");
  pdf.text("E-ComDe", margin + 15, 9);
  pdf.setFontSize(8);
  pdf.setFont("helvetica", "normal");
  pdf.text("Yapay Zeka Ürün Analiz Raporu", margin + 15, 14);

  // Tarih
  pdf.setFontSize(8);
  const dateStr = new Date().toLocaleDateString("tr-TR", { day: "2-digit", month: "long", year: "numeric" });
  pdf.text(dateStr, pageW - margin, 11, { align: "right" });

  y = 28;

  // ── ÜRÜN BİLGİSİ (varsa) ───────────────────────────────────────────────
  if (data.shopProduct) {
    pdf.setFillColor(245, 243, 255);
    pdf.roundedRect(margin, y, contentW, 14, 3, 3, "F");
    pdf.setFontSize(8);
    pdf.setTextColor(109, 40, 217);
    pdf.setFont("helvetica", "bold");
    pdf.text("İSTÜNSHOP ÜRÜN ANALİZİ", margin + 4, y + 5);
    pdf.setTextColor(30, 30, 30);
    pdf.setFont("helvetica", "normal");
    pdf.setFontSize(9);
    const productName = data.shopProduct.name.length > 70
      ? data.shopProduct.name.slice(0, 70) + "..."
      : data.shopProduct.name;
    pdf.text(productName, margin + 4, y + 10.5);
    pdf.setFontSize(8);
    pdf.setTextColor(109, 40, 217);
    pdf.text(`${data.shopProduct.total_reviews.toLocaleString("tr-TR")} yorum  ·  ★ ${data.shopProduct.avg_rating.toFixed(1)}`, pageW - margin - 4, y + 10.5, { align: "right" });
    y += 20;
  }

  // ── SKOR ───────────────────────────────────────────────────────────────
  const sc = scoreColor(data.score);
  pdf.setFillColor(...sc);
  pdf.circle(margin + 12, y + 12, 12, "F");
  pdf.setFontSize(13);
  pdf.setFont("helvetica", "bold");
  pdf.setTextColor(255, 255, 255);
  pdf.text(data.score.toFixed(1), margin + 12, y + 13.5, { align: "center" });

  pdf.setTextColor(30, 30, 30);
  pdf.setFontSize(14);
  pdf.setFont("helvetica", "bold");
  pdf.text(stars(data.score), margin + 28, y + 9);
  pdf.setFontSize(9);
  pdf.setFont("helvetica", "normal");
  pdf.setTextColor(107, 114, 128);
  pdf.text(`${data.totalReviews.toLocaleString("tr-TR")} müşteri yorumu analiz edildi`, margin + 28, y + 15);
  if (data.flaggedCount > 0) {
    pdf.setTextColor(245, 158, 11);
    pdf.text(`⚠ ${data.flaggedCount} şüpheli yorum tespit edildi`, margin + 28, y + 20);
  }
  y += 32;
  drawHRule(pdf, y, pageW, margin);
  y += 6;

  // ── DUYGU DAĞILIMI ─────────────────────────────────────────────────────
  const total = data.sentimentCounts.olumlu + data.sentimentCounts.olumsuz + data.sentimentCounts.notr;
  pdf.setFontSize(10);
  pdf.setFont("helvetica", "bold");
  pdf.setTextColor(30, 30, 30);
  pdf.text("Duygu Dağılımı", margin, y + 4);
  y += 8;

  const barData = [
    { label: "Olumlu", count: data.sentimentCounts.olumlu, color: [34, 197, 94] as [number, number, number] },
    { label: "Nötr", count: data.sentimentCounts.notr, color: [156, 163, 175] as [number, number, number] },
    { label: "Olumsuz", count: data.sentimentCounts.olumsuz, color: [239, 68, 68] as [number, number, number] },
  ];

  const barH = 7;
  const barMaxW = contentW * 0.55;
  const labelW = 22;
  const numW = 18;

  barData.forEach(({ label, count, color }) => {
    const pct = total > 0 ? count / total : 0;
    const barW = Math.max(pct * barMaxW, pct > 0 ? 2 : 0);
    const pctStr = `${(pct * 100).toFixed(1)}%`;

    pdf.setFontSize(8);
    pdf.setFont("helvetica", "normal");
    pdf.setTextColor(107, 114, 128);
    pdf.text(label, margin + labelW - 1, y + barH / 2 + 2, { align: "right" });

    // Bar background
    pdf.setFillColor(243, 244, 246);
    pdf.roundedRect(margin + labelW + 2, y, barMaxW, barH, 2, 2, "F");
    // Bar fill
    if (barW > 0) {
      pdf.setFillColor(...color);
      pdf.roundedRect(margin + labelW + 2, y, barW, barH, 2, 2, "F");
    }
    // Percentage + count
    pdf.setTextColor(30, 30, 30);
    pdf.setFont("helvetica", "bold");
    pdf.text(`${pctStr}  (${count})`, margin + labelW + barMaxW + 5, y + barH / 2 + 2);
    y += barH + 4;
  });
  y += 4;
  drawHRule(pdf, y, pageW, margin);
  y += 6;

  // ── YAPAY ZEKA ÖZETİ ───────────────────────────────────────────────────
  if (data.overallSummary) {
    pdf.setFontSize(10);
    pdf.setFont("helvetica", "bold");
    pdf.setTextColor(30, 30, 30);
    pdf.text("Yapay Zeka Genel Özeti", margin, y + 4);
    y += 8;

    pdf.setFillColor(250, 250, 255);
    const summaryLines = pdf.splitTextToSize(data.overallSummary, contentW - 8);
    const summaryH = summaryLines.length * 5 + 6;
    pdf.roundedRect(margin, y, contentW, summaryH, 3, 3, "F");
    pdf.setFontSize(9);
    pdf.setFont("helvetica", "normal");
    pdf.setTextColor(55, 65, 81);
    pdf.text(summaryLines, margin + 4, y + 5);
    y += summaryH + 6;
    drawHRule(pdf, y, pageW, margin);
    y += 6;
  }

  // ── OLUMLU ÖZET ────────────────────────────────────────────────────────
  const hasOlumlu = data.summaries.olumlu && data.summaries.olumlu.trim();
  const hasOlumsuz = data.summaries.olumsuz && data.summaries.olumsuz.trim();

  if (hasOlumlu || hasOlumsuz) {
    pdf.setFontSize(10);
    pdf.setFont("helvetica", "bold");
    pdf.setTextColor(30, 30, 30);
    pdf.text("Duygu Bazlı Özetler", margin, y + 4);
    y += 8;

    if (hasOlumlu) {
      pdf.setFillColor(240, 253, 244);
      const lines = pdf.splitTextToSize("+ " + data.summaries.olumlu!, contentW - 8);
      const h = lines.length * 5 + 6;
      pdf.roundedRect(margin, y, contentW, h, 3, 3, "F");
      pdf.setFontSize(8);
      pdf.setFont("helvetica", "bold");
      pdf.setTextColor(21, 128, 61);
      pdf.text("Olumlu Özet", margin + 4, y + 4);
      pdf.setFont("helvetica", "normal");
      pdf.setTextColor(55, 65, 81);
      pdf.text(lines, margin + 4, y + 9);
      y += h + 4;
    }

    if (hasOlumsuz) {
      pdf.setFillColor(254, 242, 242);
      const lines = pdf.splitTextToSize("- " + data.summaries.olumsuz!, contentW - 8);
      const h = lines.length * 5 + 6;
      pdf.roundedRect(margin, y, contentW, h, 3, 3, "F");
      pdf.setFontSize(8);
      pdf.setFont("helvetica", "bold");
      pdf.setTextColor(185, 28, 28);
      pdf.text("Olumsuz Özet", margin + 4, y + 4);
      pdf.setFont("helvetica", "normal");
      pdf.setTextColor(55, 65, 81);
      pdf.text(lines, margin + 4, y + 9);
      y += h + 4;
    }
    y += 2;
    drawHRule(pdf, y, pageW, margin);
    y += 6;
  }

  // ── ARTILLAR VE EKSİLER ────────────────────────────────────────────────
  if (data.pros.length > 0 || data.cons.length > 0) {
    // Yeni sayfaya geç gerekirse
    if (y > pageH - 80) { pdf.addPage(); y = 16; }

    pdf.setFontSize(10);
    pdf.setFont("helvetica", "bold");
    pdf.setTextColor(30, 30, 30);
    pdf.text("Öne Çıkan Artı ve Eksiler", margin, y + 4);
    y += 8;

    const colW = (contentW - 6) / 2;

    // Sol: Artılar
    let leftY = y;
    pdf.setFillColor(240, 253, 244);
    pdf.setFontSize(8.5);
    pdf.setFont("helvetica", "bold");
    pdf.setTextColor(21, 128, 61);
    pdf.text("✓  En Çok Övülenler", margin, leftY + 4);
    leftY += 8;
    data.pros.forEach((p) => {
      const lines = pdf.splitTextToSize("• " + p, colW - 2);
      pdf.setFont("helvetica", "normal");
      pdf.setTextColor(55, 65, 81);
      pdf.setFontSize(8);
      pdf.text(lines, margin, leftY);
      leftY += lines.length * 4.5 + 1;
    });

    // Sağ: Eksiler
    let rightY = y;
    pdf.setFontSize(8.5);
    pdf.setFont("helvetica", "bold");
    pdf.setTextColor(185, 28, 28);
    pdf.text("✗  En Çok Şikayet Edilenler", margin + colW + 6, rightY + 4);
    rightY += 8;
    data.cons.forEach((c) => {
      const lines = pdf.splitTextToSize("• " + c, colW - 2);
      pdf.setFont("helvetica", "normal");
      pdf.setTextColor(55, 65, 81);
      pdf.setFontSize(8);
      pdf.text(lines, margin + colW + 6, rightY);
      rightY += lines.length * 4.5 + 1;
    });

    y = Math.max(leftY, rightY) + 4;
    drawHRule(pdf, y, pageW, margin);
    y += 6;
  }

  // ── DANIŞMAN KARTI ─────────────────────────────────────────────────────
  if (data.advisor) {
    if (y > pageH - 70) { pdf.addPage(); y = 16; }

    pdf.setFontSize(10);
    pdf.setFont("helvetica", "bold");
    pdf.setTextColor(30, 30, 30);
    pdf.text("Yapay Zeka Alışveriş Danışmanı", margin, y + 4);
    y += 8;

    // Verdict badge
    const vc = verdictColor(data.advisor.verdict);
    pdf.setFillColor(...vc);
    const vW = pdf.getTextWidth(data.advisor.verdict) + 12;
    pdf.roundedRect(margin, y, vW, 9, 2, 2, "F");
    pdf.setFontSize(9);
    pdf.setFont("helvetica", "bold");
    pdf.setTextColor(255, 255, 255);
    pdf.text(data.advisor.verdict, margin + 6, y + 6);
    y += 13;

    // Hedef kitle
    pdf.setFillColor(248, 245, 255);
    const targetLines = pdf.splitTextToSize(data.advisor.target_audience, contentW - 8);
    const targetH = targetLines.length * 5 + 6;
    pdf.roundedRect(margin, y, contentW, targetH, 3, 3, "F");
    pdf.setFontSize(8);
    pdf.setFont("helvetica", "normal");
    pdf.setTextColor(55, 65, 81);
    pdf.text(targetLines, margin + 4, y + 5);
    y += targetH + 6;

    // Buy if + Watch out
    const colW2 = (contentW - 6) / 2;

    if (data.advisor.buy_if.length > 0) {
      pdf.setFontSize(8);
      pdf.setFont("helvetica", "bold");
      pdf.setTextColor(21, 128, 61);
      pdf.text("Şu Durumlarda Al:", margin, y + 4);
      let by = y + 8;
      data.advisor.buy_if.forEach((item) => {
        pdf.setFont("helvetica", "normal");
        pdf.setTextColor(55, 65, 81);
        const lines = pdf.splitTextToSize("› " + item, colW2);
        pdf.text(lines, margin, by);
        by += lines.length * 4.5;
      });
    }

    if (data.advisor.watch_out.length > 0) {
      pdf.setFontSize(8);
      pdf.setFont("helvetica", "bold");
      pdf.setTextColor(185, 28, 28);
      pdf.text("Dikkat Et:", margin + colW2 + 6, y + 4);
      let wy = y + 8;
      data.advisor.watch_out.forEach((item) => {
        pdf.setFont("helvetica", "normal");
        pdf.setTextColor(55, 65, 81);
        const lines = pdf.splitTextToSize("› " + item, colW2);
        pdf.text(lines, margin + colW2 + 6, wy);
        wy += lines.length * 4.5;
      });
    }
    y += 50;
  }

  // ── FOOTER (tüm sayfalarda) ────────────────────────────────────────────
  const totalPages = (pdf as unknown as { internal: { getNumberOfPages: () => number } }).internal.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    pdf.setPage(i);
    pdf.setFillColor(109, 40, 217);
    pdf.rect(0, pageH - 10, pageW, 10, "F");
    pdf.setFontSize(7);
    pdf.setFont("helvetica", "normal");
    pdf.setTextColor(255, 255, 255);
    pdf.text("E-ComDe · Yapay Zeka E-Ticaret Analiz Sistemi · BERTurk + Claude AI", margin, pageH - 4);
    pdf.text(`Sayfa ${i} / ${totalPages}`, pageW - margin, pageH - 4, { align: "right" });
  }

  pdf.save(filename);
}
