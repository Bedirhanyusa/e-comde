from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.dependencies import get_sentiment_classifier, get_summarizer
from backend.routers import sentiment, summarize, ingest, shop


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Modelleri uygulama baslarken tek sefer yukle
    get_sentiment_classifier()
    get_summarizer()
    yield


app = FastAPI(title="E-ComDe API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sentiment.router)
app.include_router(summarize.router)
app.include_router(ingest.router)
app.include_router(shop.router)


@app.get("/health")
def health():
    return {"status": "ok"}
