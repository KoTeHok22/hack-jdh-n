from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from app.services.search import hybrid_search, semantic_search, bm25_search

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    filters: Optional[dict] = None
    top_k: int = 10
    mode: str = "hybrid"


@router.post("")
async def search(request: SearchRequest):
    if request.mode == "semantic":
        results = await semantic_search(request.query, request.top_k, request.filters)
    elif request.mode == "bm25":
        results = bm25_search(request.query, request.top_k)
    else:
        results = await hybrid_search(request.query, request.top_k, request.filters)

    return {"results": results, "count": len(results)}
