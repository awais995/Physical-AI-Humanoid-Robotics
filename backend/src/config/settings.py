import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    """
    # Cohere Configuration
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")

    # Qdrant Configuration
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "book_content")

    # Neon Postgres Configuration
    neon_database_url: str = os.getenv("NEON_DATABASE_URL", "")

    # Application Settings
    app_name: str = "RAG Chatbot API"
    app_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    app_env: str = os.getenv("APP_ENV", "development")

    # Content Processing Settings
    chunk_size: int = 512  # tokens
    chunk_overlap: int = 50  # tokens
    max_chunk_size: int = 1024  # maximum tokens per chunk
    min_chunk_size: int = 128   # minimum tokens per chunk

    # Retrieval Settings
    top_k: int = 5  # number of passages to retrieve
    similarity_threshold: float = 0.3  # minimum similarity score
    rerank_enabled: bool = True  # whether to use Cohere re-ranking
    rerank_top_k: int = 3  # number of passages to re-rank

    # Generation Settings
    max_tokens: int = 500  # max tokens in response
    temperature: float = 0.1  # generation temperature (low for factual responses)
    hallucination_check_enabled: bool = True

    # Performance Settings
    response_timeout: int = 30  # seconds
    max_concurrent_requests: int = 10

    # Caching Settings
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour in seconds

    # Rate Limiting
    rate_limit_requests: int = 100  # requests per minute
    rate_limit_window: int = 60  # seconds

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # This will ignore extra fields that are not defined in the model

# Create settings instance
settings = Settings()

# Validate required settings
def validate_settings():
    """
    Validate that all required settings are present
    """
    errors = []

    if not settings.cohere_api_key:
        errors.append("COHERE_API_KEY is required")

    if not settings.qdrant_url:
        errors.append("QDRANT_URL is required")

    if not settings.neon_database_url:
        errors.append("NEON_DATABASE_URL is required")

    if errors:
        raise ValueError(f"Missing required settings: {', '.join(errors)}")

# Validate settings on import
validate_settings()