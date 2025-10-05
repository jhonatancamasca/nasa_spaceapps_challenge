from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # General app settings
    app_name: str = "AI Knowledge API"
    env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1  # <-- Agregado para Docker/env

    # LLM / AI settings
    llm_provider: str = "openai"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    # Database
    database_url: str | None = None

    # Search / retrieval
    default_k: int = 5
    default_threshold: float = 0.75

    # Storage
    storage_type: str = "local"
    local_storage_path: str = "./data"

    # Pydantic v2 configuration
    model_config = {
        "env_file": ".env",
        "extra": "forbid"  # Evita variables desconocidas
    }

@lru_cache()
def get_settings() -> Settings:
    return Settings()
