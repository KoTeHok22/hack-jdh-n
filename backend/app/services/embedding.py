import logging
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

_semaphore = None
_client = None


def _get_semaphore():
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(3)
    return _semaphore


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_api_base,
        )
    return _client


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=30))
async def _embed_one(text: str) -> list[float]:
    client = _get_client()
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=[text]
    )
    return response.data[0].embedding


async def _embed_one_with_limit(text: str) -> list[float]:
    sem = _get_semaphore()
    async with sem:
        return await _embed_one(text)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    logger.info(f"Embedding {len(texts)} texts via Cloud.ru API (concurrent)...")
    tasks = [_embed_one_with_limit(t) for t in texts]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    final_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error(f"Failed embedding text {i + 1}: {r}")
            final_results.append([0.0] * settings.embedding_dim)
        else:
            final_results.append(r)

    logger.info(f"Embedding done: {len(final_results)} vectors")
    return final_results


async def embed_text(text: str) -> list[float]:
    return await _embed_one(text)
