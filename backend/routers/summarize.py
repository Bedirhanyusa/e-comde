import asyncio
from fastapi import APIRouter, Depends
from backend.schemas.summarize import (
    SummarizeRequest, SummarizeResponse,
    AdvisorRequest, AdvisorResponse,
    EmotionRequest, EmotionResponse,
)
from backend.dependencies import get_summarizer
from src.models.summarizer import ReviewSummarizer

router = APIRouter(prefix="/api/v1/summarize", tags=["summarize"])


@router.post("/", response_model=SummarizeResponse)
async def summarize_reviews(
    request: SummarizeRequest,
    summarizer: ReviewSummarizer = Depends(get_summarizer),
):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: summarizer.get_full_analysis(request.reviews, request.labels),
    )
    return SummarizeResponse(**result)


@router.post("/advisor", response_model=AdvisorResponse)
async def get_advisor(
    request: AdvisorRequest,
    summarizer: ReviewSummarizer = Depends(get_summarizer),
):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: summarizer.get_advisor(request.reviews, request.labels, request.score),
    )
    return AdvisorResponse(**result)


@router.post("/emotions", response_model=EmotionResponse)
async def get_emotions(
    request: EmotionRequest,
    summarizer: ReviewSummarizer = Depends(get_summarizer),
):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: summarizer.get_emotions(request.reviews, request.labels),
    )
    return EmotionResponse(**result)
