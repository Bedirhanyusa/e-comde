import asyncio, json
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Literal

from backend.dependencies import get_sentiment_classifier, get_summarizer
from backend.config import settings
from src.models.sentiment_classifier import SentimentClassifier
from src.models.summarizer import ReviewSummarizer

router = APIRouter(prefix="/api/v1/shop", tags=["shop"])

CATALOG_PATH = Path("data/shop_catalog.json")
_catalog: dict | None = None


def _load_catalog() -> dict:
    global _catalog
    if _catalog is None:
        if not CATALOG_PATH.exists():
            raise HTTPException(status_code=503, detail="Shop katalogu bulunamadi.")
        with open(CATALOG_PATH, encoding="utf-8") as f:
            _catalog = json.load(f)
    return _catalog


# ── Schemas ──────────────────────────────────────────────────────────

class CategoryOut(BaseModel):
    id: str
    label: str
    icon: str
    color: str
    product_count: int


class ProductOut(BaseModel):
    id: str
    category_id: str
    name: str
    brand: str = ""
    description: str = ""
    price: str = ""
    features: list[str] = []
    image: str = ""
    trendyol_url: str
    avg_rating: float
    total_reviews: int


class ReviewResult(BaseModel):
    text: str
    label: Literal["olumlu", "olumsuz", "notr"]
    confidence: float
    flagged: bool


class ShopAnalysisResponse(BaseModel):
    product: ProductOut
    results: list[ReviewResult]
    total: int
    flagged_count: int
    sentiment_distribution: dict[str, int]
    summaries: dict[str, str]
    overall: str = ""
    pros: list[str] = []
    cons: list[str] = []
    avg_confidence: float


# ── Endpoints ────────────────────────────────────────────────────────

class AnalyzeByUrlRequest(BaseModel):
    url: str


@router.post("/analyze-by-url", response_model=ShopAnalysisResponse)
async def analyze_by_url(
    request: AnalyzeByUrlRequest,
    classifier: SentimentClassifier = Depends(get_sentiment_classifier),
    summarizer: ReviewSummarizer = Depends(get_summarizer),
):
    catalog = _load_catalog()
    url = request.url.strip().rstrip("/")
    product = next(
        (p for p in catalog["products"] if p["trendyol_url"].rstrip("/") == url),
        None,
    )
    if not product:
        raise HTTPException(status_code=404, detail="Bu URL katalogda bulunamadı.")

    return await analyze_product(product["id"], classifier, summarizer)


@router.get("/products", response_model=dict)
def list_products(category: str | None = None):
    catalog = _load_catalog()
    products = catalog["products"]
    if category:
        products = [p for p in products if p["category_id"] == category]
    return {
        "categories": catalog["categories"],
        "products": [
            {k: v for k, v in p.items() if k != "reviews"}
            for p in products
        ],

    }


@router.post("/products/{product_id}/analyze", response_model=ShopAnalysisResponse)
async def analyze_product(
    product_id: str,
    classifier: SentimentClassifier = Depends(get_sentiment_classifier),
    summarizer: ReviewSummarizer = Depends(get_summarizer),
):
    catalog = _load_catalog()
    product = next((p for p in catalog["products"] if p["id"] == product_id), None)
    if not product:
        raise HTTPException(status_code=404, detail="Urun bulunamadi.")

    reviews = product["reviews"]
    loop = asyncio.get_event_loop()

    # Sentiment analizi
    raw = await loop.run_in_executor(
        None,
        lambda: classifier.predict(reviews, confidence_threshold=settings.confidence_threshold),
    )

    results = [ReviewResult(**r) for r in raw]
    labels = [r.label for r in results]

    dist: dict[str, int] = {"olumlu": 0, "olumsuz": 0, "notr": 0}
    for lbl in labels:
        dist[lbl] += 1

    avg_conf = round(sum(r.confidence for r in results) / len(results), 4) if results else 0.0
    flagged = sum(1 for r in results if r.flagged)

    # Özetleme — tam analiz (summaries + overall + pros + cons)
    full = await loop.run_in_executor(
        None,
        lambda: summarizer.get_full_analysis(reviews, labels),
    )

    product_out = ProductOut(
        id=product["id"],
        category_id=product["category_id"],
        name=product["name"],
        brand=product.get("brand", ""),
        description=product.get("description", ""),
        price=product.get("price", ""),
        features=product.get("features", []),
        image=product.get("image", ""),
        trendyol_url=product["trendyol_url"],
        avg_rating=product["avg_rating"],
        total_reviews=product["total_reviews"],
    )

    return ShopAnalysisResponse(
        product=product_out,
        results=results,
        total=len(results),
        flagged_count=flagged,
        sentiment_distribution=dist,
        summaries=full.get("summaries", {}),
        overall=full.get("overall", ""),
        pros=full.get("pros", []),
        cons=full.get("cons", []),
        avg_confidence=avg_conf,
    )
