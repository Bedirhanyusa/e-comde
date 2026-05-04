from pydantic import BaseModel


class SummarizeRequest(BaseModel):
    reviews: list[str]
    labels: list[str]


class SummarizeResponse(BaseModel):
    summaries: dict[str, str]
