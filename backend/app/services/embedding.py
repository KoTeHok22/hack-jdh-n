import logging
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from app.config import get_settings
from app.services.embedding_local import embed_text_local, embed_texts_local

logger = logging.getLogger(__name__)

_client = None

def _get_settings():
    return get_settings()

def _get_semaphore():
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(3)
    return _semaphore

def _get_client():
    global _client
    if _client is None:
        s = _get_settings()
        _client = OpenAI(
            api_key=s.embedding_api_key,
            base_url=s.embedding_api_base,
        )
    return _client

@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=30))
async def _embed_one(text: str) -> list[float]:
    s = _get_settings()
    client = _get_client()
    response = client.embeddings.create(
        model=s.embedding_model,
        input=[text]
    )
    return response.data[0].embedding

async def _embed_one_with_limit(text: str) -> list[float]:
    sem = _get_semaphore()
    async with sem:
        return await _embed_one(text)

async def embed_text(text: str) -> list[float]:
    s = _get_settings()
    if s.embedding_provider == "local":
        return await embed_text_local(text)

    try:
        return await _embed_one(text)
    except Exception as e:
        logger.warning(f"API embedding failed, falling back to local: {e}")
        return await embed_text_local(text)

async def embed_texts(texts: list[str]) -> list[list[float]]:
    s = _get_settings()
    if s.embedding_provider == "local":
        return await embed_texts_local(texts)

    try:
        logger.info(f"Embedding {len(texts)} texts via API (concurrent)...")
        tasks = [_embed_one_with_limit(t) for t in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = [r for r in results if not isinstance(r, Exception)]
        if len(successful) < len(texts) * 0.5:
            raise Exception(f"API failed for {len(results) - len(successful)}/{len(texts)} texts")

        final_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"Failed embedding text {i + 1}: {r}")
                local_emb = await embed_text_local(texts[i])
                api_dim = s.embedding_dim
                if len(local_emb) < api_dim:
                    local_emb += [0.0] * (api_dim - len(local_emb))
                elif len(local_emb) > api_dim:
                    local_emb = local_emb[:api_dim]
                final_results.append(local_emb)
            else:
                final_results.append(r)

        logger.info(f"Embedding done: {len(final_results)} vectors")
        return final_results
    except Exception as e:
        logger.warning(f"Batch API embedding failed, falling back to local: {e}")
        return await embed_texts_local(texts)
