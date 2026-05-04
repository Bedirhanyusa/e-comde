from pydantic import BaseModel
from typing import Literal


class ReviewRequest(BaseModel):
    texts: list[str]


class ReviewResult(BaseModel):
    text: str
    label: Literal["olumlu", "olumsuz", "notr"]
    confidence: float
    flagged: bool


class SentimentResponse(BaseModel):
    results: list[ReviewResult]
    total: int
    flagged_count: int
