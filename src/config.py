from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # General app settings
    app_name: str = "AI Knowledge API"
    env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # LLM / AI settings
    llm_provider: str = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    database_url: str
    
    # AWS S3 Configuration
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str = "us-east-2"
    s3_bucket_name: str = "bioseekersbucket"

    # Search / retrieval
    default_k: int = 5
    default_threshold: float = 0.75
    embedded_dim: int = 1024

    # Storage
    storage_type: str = "s3"
    local_storage_path: str = "./data"
    
    # Embedder
    HF_TOKEN: str

    # Pydantic v2 configuration
    model_config = {
        "env_file": ".env",
        "extra": "forbid"
    }

@lru_cache()
def get_settings() -> Settings:
    return Settings()