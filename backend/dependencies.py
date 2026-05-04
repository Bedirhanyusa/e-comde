from functools import lru_cache
from src.models.sentiment_classifier import SentimentClassifier
from src.models.summarizer import ReviewSummarizer
from backend.config import settings


@lru_cache(maxsize=1)
def get_sentiment_classifier() -> SentimentClassifier:
    return SentimentClassifier(settings.sentiment_model_dir)


@lru_cache(maxsize=1)
def get_summarizer() -> ReviewSummarizer:
    return ReviewSummarizer(settings.summarizer_model_name)
