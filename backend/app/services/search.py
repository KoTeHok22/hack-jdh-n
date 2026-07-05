import math
from datetime import datetime, timedelta

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


def build_context_with_refs(chunks: list[dict], max_tokens: int = 4000) -> tuple[str, list[dict]]:
    refs = []
    context_parts = []
    total_tokens = 0
    
    for i, chunk in enumerate(chunks, 1):
        content = chunk.get("content", "")
        source = chunk.get("source_title", "Unknown")
        page = chunk.get("page", "?")
        section = chunk.get("section", "")
        
        refs.append({
            "n": i,
            "chunk_id": chunk.get("id", f"chunk_{i}"),
            "title": source,
            "page": page,
            "section": section,
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
        })
        
        header = f"[{i}] Источник: {source}"
        if page:
            header += f", стр. {page}"
        if section:
            header += f", раздел: {section}"
        
        entry = f"{header}\n{content}\n"
        entry_tokens = len(entry) // 4
        
        if total_tokens + entry_tokens > max_tokens:
            break
        
        context_parts.append(entry)
        total_tokens += entry_tokens
    
    context = "\n---\n".join(context_parts)
    return context, refs


def build_context(chunks: list[dict], max_tokens: int = 4000) -> str:
    context, _ = build_context_with_refs(chunks, max_tokens)
    return context


async def semantic_search(query: str, top_k: int = 10, filters: dict = None) -> list[dict]:
    client = get_qdrant()
    if client is None:
        return []
    query_vector = await embed_text(query)

    collection = COLLECTION_NAME
    fallback_collection = f"{COLLECTION_NAME}_fallback"
    if len(query_vector) == settings.embedding_fallback_dim:
        collection, fallback_collection = fallback_collection, collection

    qdrant_filter = None
    if filters:
        conditions = []
        for key, value in filters.items():
            conditions.append(FieldCondition(key=f"payload.{key}", match=MatchValue(value=value)))
        if conditions:
            qdrant_filter = Filter(must=conditions)

    results = await client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        query_filter=qdrant_filter,
    )

    if not results:
        try:
            from app.services.embedding_local import embed_text_local
            fb_vector = await embed_text_local(query)
            fb_results = await client.search(
                collection_name=fallback_collection,
                query_vector=fb_vector.tolist(),
                limit=top_k,
                query_filter=qdrant_filter,
            )
            results = fb_results
        except Exception:
            pass

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


def apply_time_decay(results: list[dict], days_weight: float = 0.95) -> list[dict]:
    now = datetime.utcnow()
    decayed = []
    
    for doc in results:
        score = doc.get("score", 0.0)
        created_at = doc.get("created_at")
        
        if created_at:
            try:
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                
                days_old = (now - created_at).days
                decay = math.pow(days_weight, days_old)
                doc["score"] = score * decay
            except (ValueError, TypeError):
                pass
        
        decayed.append(doc)
    
    return decayed


def apply_metadata_boost(results: list[dict], boost_factors: dict = None) -> list[dict]:
    if not boost_factors:
        return results
    
    boosted = []
    for doc in results:
        score = doc.get("score", 0.0)
        multiplier = 1.0
        
        for field, boosts in boost_factors.items():
            field_value = doc.get(field)
            if field_value in boosts:
                multiplier *= boosts[field_value]
        
        doc["score"] = score * multiplier
        boosted.append(doc)
    
    return boosted


def diversify_results(results: list[dict], lambda_param: float = 0.7, k: int = 10) -> list[dict]:

    if not results or k >= len(results):
        return results[:k]
    
    selected = []
    candidates = results.copy()
    
    while len(selected) < k and candidates:
        if not selected:
            best = max(candidates, key=lambda x: x.get("score", 0))
        else:
            best = None
            best_mmr = float("-inf")
            
            for candidate in candidates:
                relevance = candidate.get("score", 0)
                
                max_sim = 0
                for sel in selected:
                    sim = _calculate_similarity(candidate, sel)
                    max_sim = max(max_sim, sim)
                
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                
                if mmr > best_mmr:
                    best_mmr = mmr
                    best = candidate
        
        selected.append(best)
        candidates.remove(best)
    
    return selected


def _calculate_similarity(doc1: dict, doc2: dict) -> float:
    content1 = set(doc1.get("content", "").lower().split())
    content2 = set(doc2.get("content", "").lower().split())
    
    if not content1 or not content2:
        return 0.0
    
    intersection = len(content1 & content2)
    union = len(content1 | content2)
    
    return intersection / union if union > 0 else 0.0


async def advanced_hybrid_search(
    query: str,
    top_k: int = 10,
    filters: dict = None,
    use_time_decay: bool = False,
    use_diversification: bool = True,
    diversity_lambda: float = 0.7
) -> list[dict]:
    results = await hybrid_search(query, top_k=top_k * 2, filters=filters)
    
    if use_time_decay:
        results = apply_time_decay(results, days_weight=0.95)
    
    results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
    
    if use_diversification and len(results) > top_k:
        results = diversify_results(results, lambda_param=diversity_lambda, k=top_k)
    else:
        results = results[:top_k]
    
    return results
