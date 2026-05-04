import io
import json
import pandas as pd
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from backend.schemas.sentiment import SentimentResponse, ReviewResult
from backend.dependencies import get_sentiment_classifier
from backend.config import settings
from src.models.sentiment_classifier import SentimentClassifier

router = APIRouter(prefix="/api/v1/ingest", tags=["ingest"])


def _parse_csv_texts(content: bytes, text_col: str) -> list[str]:
    for enc in ("utf-8-sig", "utf-8", "cp1254", "latin-1"):
        try:
            decoded = content.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        decoded = content.decode("utf-8", errors="replace")

    # Standard parse first
    try:
        df = pd.read_csv(io.StringIO(decoded))
        if text_col in df.columns:
            return df[text_col].dropna().astype(str).tolist()
    except pd.errors.ParserError:
        pass

    # Fallback: line-by-line (handles unquoted commas in single-column CSVs)
    lines = [l.strip() for l in decoded.splitlines() if l.strip()]
    if not lines:
        return []
    header = lines[0].strip().strip('"')
    if header == text_col:
        return [l.strip().strip('"') for l in lines[1:] if l.strip()]

    raise ValueError(f"'{text_col}' kolonu bulunamadı.")


@router.post("/csv", response_model=SentimentResponse)
async def ingest_csv(
    file: UploadFile = File(...),
    text_col: str = "Review",
    classifier: SentimentClassifier = Depends(get_sentiment_classifier),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Sadece CSV dosyası kabul edilir.")
    content = await file.read()
    try:
        texts = _parse_csv_texts(content, text_col)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not texts:
        raise HTTPException(status_code=400, detail="CSV dosyasında yorum bulunamadı.")
    raw = classifier.predict(texts, confidence_threshold=settings.confidence_threshold)
    results = [ReviewResult(**r) for r in raw]
    return SentimentResponse(results=results, total=len(results), flagged_count=sum(1 for r in results if r.flagged))


@router.post("/json", response_model=SentimentResponse)
async def ingest_json(
    file: UploadFile = File(...),
    classifier: SentimentClassifier = Depends(get_sentiment_classifier),
):
    content = await file.read()
    data = json.loads(content)
    texts = [item.get("text", "") for item in data if item.get("text")]
    raw = classifier.predict(texts, confidence_threshold=settings.confidence_threshold)
    results = [ReviewResult(**r) for r in raw]
    return SentimentResponse(results=results, total=len(results), flagged_count=sum(1 for r in results if r.flagged))
