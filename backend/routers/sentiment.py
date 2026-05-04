import asyncio
from fastapi import APIRouter, Depends
from backend.schemas.sentiment import ReviewRequest, SentimentResponse, ReviewResult
from backend.dependencies import get_sentiment_classifier
from backend.config import settings
from src.models.sentiment_classifier import SentimentClassifier

router = APIRouter(prefix="/api/v1/sentiment", tags=["sentiment"])


@router.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(
    request: ReviewRequest,
    classifier: SentimentClassifier = Depends(get_sentiment_classifier),
):
    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(
        None,
        lambda: classifier.predict(request.texts, confidence_threshold=settings.confidence_threshold),
    )
    results = [ReviewResult(**r) for r in raw]
    return SentimentResponse(
        results=results,
        total=len(results),
        flagged_count=sum(1 for r in results if r.flagged),
    )
