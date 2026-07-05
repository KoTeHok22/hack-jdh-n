from qdrant_client.models import Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi
from app.db.qdrant_client import get_qdrant, COLLECTION_NAME
from app.services.embedding import embed_text
from app.config import get_settings

settings = get_settings()

_bm25_index = None
_bm25_docs = []
_bm25_doc_map = {}
_all_bm25_docs: list[dict] = []


async def semantic_search(query: str, top_k: int = 10, filters: dict = None) -> list[dict]:
    client = get_qdrant()
    if client is None:
        return []
    query_vector = await embed_text(query)

    qdrant_filter = None
    if filters:
        conditions = []
        for key, value in filters.items():
            conditions.append(FieldCondition(key=f"payload.{key}", match=MatchValue(value=value)))
        if conditions:
            qdrant_filter = Filter(must=conditions)

    results = await client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        query_filter=qdrant_filter,
    )

    return [
        {
            "id": str(r.id),
            "content": r.payload.get("content", ""),
            "document_id": r.payload.get("document_id", ""),
            "source_title": r.payload.get("source_title", ""),
            "page": r.payload.get("page"),
            "section": r.payload.get("section"),
            "score": r.score,
        }
        for r in results
    ]


def build_bm25_index(chunks: list[dict]):
    global _bm25_index, _bm25_docs, _bm25_doc_map, _all_bm25_docs
    if not chunks:
        return
    _all_bm25_docs.extend(chunks)
    _bm25_docs = _all_bm25_docs
    tokenized = [doc["content"].lower().split() for doc in _bm25_docs]
    _bm25_index = BM25Okapi(tokenized)
    _bm25_doc_map = {i: doc for i, doc in enumerate(_bm25_docs)}


def bm25_search(query: str, top_k: int = 10) -> list[dict]:
    if _bm25_index is None:
        return []
    tokens = query.lower().split()
    scores = _bm25_index.get_scores(tokens)
    top_indices = scores.argsort()[::-1][:top_k]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            doc = _bm25_doc_map[idx].copy()
            doc["score"] = float(scores[idx])
            results.append(doc)
    return results


def reciprocal_rank_fusion(results_list: list[list[dict]], k: int = 60) -> list[dict]:
    scores = {}
    for results in results_list:
        for rank, doc in enumerate(results):
            doc_id = doc.get("id", doc.get("content", "")[:100])
            if doc_id not in scores:
                scores[doc_id] = {"doc": doc, "score": 0.0}
            scores[doc_id]["score"] += 1.0 / (k + rank + 1)

    sorted_results = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [r["doc"] | {"score": r["score"]} for r in sorted_results]


async def hybrid_search(query: str, top_k: int = 10, filters: dict = None) -> list[dict]:
    semantic_results = await semantic_search(query, top_k=top_k, filters=filters)
    bm25_results = bm25_search(query, top_k=top_k)

    if not bm25_results:
        return semantic_results

    fused = reciprocal_rank_fusion([semantic_results, bm25_results])
    return fused[:top_k]


def build_context(chunks: list[dict], max_tokens: int = 4000) -> str:
    context_parts = []
    total_tokens = 0

    for chunk in chunks:
        content = chunk.get("content", "")
        source = chunk.get("source_title", "Unknown")
        page = chunk.get("page", "?")
        section = chunk.get("section", "")

        header = f"[Источник: {source}"
        if page:
            header += f", стр. {page}"
        if section:
            header += f", раздел: {section}"
        header += "]"

        entry = f"{header}\n{content}\n"
        entry_tokens = len(entry) // 4

        if total_tokens + entry_tokens > max_tokens:
            break

        context_parts.append(entry)
        total_tokens += entry_tokens

    return "\n---\n".join(context_parts)
