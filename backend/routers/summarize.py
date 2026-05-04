import asyncio
from fastapi import APIRouter, Depends
from backend.schemas.summarize import SummarizeRequest, SummarizeResponse
from backend.dependencies import get_summarizer
from src.models.summarizer import ReviewSummarizer

router = APIRouter(prefix="/api/v1/summarize", tags=["summarize"])


@router.post("/", response_model=SummarizeResponse)
async def summarize_reviews(
    request: SummarizeRequest,
    summarizer: ReviewSummarizer = Depends(get_summarizer),
):
    loop = asyncio.get_event_loop()
    summaries = await loop.run_in_executor(
        None,
        lambda: summarizer.summarize_by_sentiment(request.reviews, request.labels),
    )
    return SummarizeResponse(summaries=summaries)
