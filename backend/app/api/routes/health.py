import asyncio
from fastapi import APIRouter
from app.db.qdrant_client import get_qdrant
from app.db.database import engine

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health_check():
    checks = {}

    try:
        from sqlalchemy import text
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        checks["postgres"] = "healthy"
    except Exception as e:
        checks["postgres"] = f"unhealthy: {str(e)}"

    try:
        client = get_qdrant()
        if client:
            await client.get_collections()
        collections = [c.name for c in (await client.get_collections()).collections]
        checks["qdrant"] = "healthy"
        checks["qdrant_collections"] = collections
    except Exception as e:
        checks["qdrant"] = f"unhealthy: {str(e)}"

    try:
        from app.services.embedding_local import is_model_loaded
        checks["embedding_local"] = "ready" if is_model_loaded() else "pending: not yet loaded"
    except Exception as e:
        checks["embedding_local"] = f"unavailable: {str(e)}"

    try:
        from app.config import get_settings
        s = get_settings()
        checks["embedding_provider"] = s.embedding_provider
        checks["embedding_fallback_dim"] = s.embedding_fallback_dim
        checks["ensemble_llm"] = f"enabled={s.llm_ensemble_enabled}"
    except Exception as e:
        checks["config_error"] = str(e)

    core_ok = all(
        checks.get(k, "").startswith("healthy")
        for k in ["postgres", "qdrant"]
    )

    return {
        "status": "healthy" if core_ok else "degraded",
        "checks": checks,
    }
