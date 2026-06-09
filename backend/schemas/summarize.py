from pydantic import BaseModel


class SummarizeRequest(BaseModel):
    reviews: list[str]
    labels: list[str]


class SummarizeResponse(BaseModel):
    summaries: dict[str, str]
    overall: str = ""
    pros: list[str] = []
    cons: list[str] = []


class AdvisorRequest(BaseModel):
    reviews: list[str]
    labels: list[str]
    score: float


class AdvisorResponse(BaseModel):
    verdict: str           # "Tavsiye Edilir" | "Dikkatli Olun" | "Tavsiye Edilmez"
    verdict_score: int     # 3 | 2 | 1
    target_audience: str
    buy_if: list[str]
    watch_out: list[str]


class EmotionRequest(BaseModel):
    reviews: list[str]
    labels: list[str]


class EmotionResponse(BaseModel):
    coskulu: int = 0
    memnun: int = 0
    notr: int = 0
    hayal_kirikligi: int = 0
    ofkeli: int = 0
