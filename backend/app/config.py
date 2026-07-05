from pydantic_settings import BaseSettings
from pydantic import field_validator
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://user:pass@localhost:5432/hypothesis_db"
    qdrant_url: str = "http://localhost:6333"
    
    llm_provider: str = "yandex"
    
    yandex_api_key: str = ""
    yandex_folder_id: str = ""
    yandex_api_base: str = "https://llm.api.cloud.yandex.net/foundationModels/v1"
    yandex_model: str = "yandexgpt/latest"
    
    llm_api_key: str = ""
    llm_api_base: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-flash"
    
    embedding_api_key: str = ""
    embedding_api_base: str = "https://foundation-models.api.cloud.ru/v1"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 1024
    
    chunk_size: int = 2000
    chunk_overlap: int = 200
    search_top_k: int = 10
    mistral_api_key: str = ""
    mistral_max_file_size_mb: int = 50
    ocr_provider: str = "mistral"
    glm_ocr_base_url: str = "http://localhost:8080"
    glm_ocr_model: str = "zai-org/GLM-OCR"
    glm_ocr_dpi: int = 200
    upload_dir: str = "/app/data/uploads"

    @field_validator("database_url", mode="before")
    @classmethod
    def fix_db_url(cls, v: str) -> str:
        if v.startswith("postgresql://") and "+asyncpg" not in v:
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
