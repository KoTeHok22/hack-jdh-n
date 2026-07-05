from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.config import get_settings

settings = get_settings()

client: AsyncQdrantClient = None
COLLECTION_NAME = "hypothesis_chunks"


async def init_qdrant():
    global client
    client = AsyncQdrantClient(url=settings.qdrant_url)

    collections = [c.name for c in (await client.get_collections()).collections]
    if COLLECTION_NAME not in collections:
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
    
    fallback_collection = f"{COLLECTION_NAME}_fallback"
    if fallback_collection not in collections:
        await client.create_collection(
            collection_name=fallback_collection,
            vectors_config=VectorParams(
                size=settings.embedding_fallback_dim,
                distance=Distance.COSINE,
            ),
        )


def get_qdrant() -> AsyncQdrantClient:
    return client
