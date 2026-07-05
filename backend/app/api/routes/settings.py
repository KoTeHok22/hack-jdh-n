from fastapi import APIRouter
from pydantic import BaseModel
from typing import Literal
from app.config import get_settings
import os

router = APIRouter(prefix="/settings", tags=["settings"])

class EmbeddingProviderRequest(BaseModel):
    provider: Literal["local", "api", "auto"]

@router.get("/embedding")
async def get_embedding_provider():
    s = get_settings()
    return {"provider": s.embedding_provider}

@router.post("/embedding")
async def set_embedding_provider(request: EmbeddingProviderRequest):
    os.environ["EMBEDDING_PROVIDER"] = request.provider
    get_settings.cache_clear()
    s = get_settings()
    return {"provider": s.embedding_provider, "status": "ok"}
