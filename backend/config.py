from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    sentiment_model_dir: str = "models/berturk_v3_finetuned"
    summarizer_model_dir: str = "models/mt5_summarizer"
    summarizer_model_name: str = "models/mt5_summarizer"
    confidence_threshold: float = 0.75
    max_len: int = 192
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"]

    class Config:
        env_file = ".env"


settings = Settings()
