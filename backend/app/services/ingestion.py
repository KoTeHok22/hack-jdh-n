import uuid
import os
import logging
from pathlib import Path
from typing import Callable, Optional

from app.services.parsers import parse_file
from app.services.chunking import chunk_text
from app.services.embedding import embed_texts
from app.services.search import build_bm25_index
from app.db.qdrant_client import get_qdrant, COLLECTION_NAME
from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()
UPLOAD_DIR = Path(settings.upload_dir)


async def ingest_file(
    file_path: str,
    title: str,
    doc_type: str = "textbook",
    ocr_provider: str = None,
    progress: Optional[Callable] = None,
    file_size: Optional[int] = None,
) -> dict:
    def _emit(stage: str, percent: float, detail: str = ""):
        if progress:
            progress(stage, percent, detail)

    _emit("uploading", 5, "Файл сохранён")

    logger.info(f"Начало обработки файла: {title}")
    _emit("processing", 10, "Начато распознавание...")

    text = await parse_file(file_path, ocr_provider=ocr_provider)
    logger.info(f"Файл распарсен: {len(text)} символов")
    _emit("ocr", 30, f"Распознано {len(text):,d} символов")

    _emit("chunking", 35, "Разбиение на чанки...")
    chunks = chunk_text(
        text,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        source_title=title,
    )
    logger.info(f"Создано {len(chunks)} чанков")
    _emit("chunks", 45, f"Создано {len(chunks)} чанков")

    if not chunks:
        return {"chunks_count": 0, "status": "error"}

    texts_to_embed = [c["content"] for c in chunks]
    logger.info(f"Начало embedding {len(texts_to_embed)} текстов...")
    _emit("embedding", 50, f"Векторизация {len(texts_to_embed)} чанков...")

    embeddings = await embed_texts(texts_to_embed)
    logger.info(f"Embedding завершен")
    _emit("embedding", 70, f"Векторизация завершена")

    doc_id = str(uuid.uuid4())
    points = []
    bm25_chunks = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = str(uuid.uuid4())
        points.append({
            "id": point_id,
            "vector": embedding,
            "payload": {
                "document_id": doc_id,
                "content": chunk["content"],
                "source_title": title,
                "doc_type": doc_type,
                "page": chunk.get("page"),
                "section": chunk.get("section"),
                "position": chunk.get("position", i),
                "char_count": chunk.get("char_count", 0),
                "token_count": chunk.get("token_count", 0),
            },
        })

        bm25_chunks.append({
            "id": point_id,
            "content": chunk["content"],
            "document_id": doc_id,
            "source_title": title,
            "page": chunk.get("page"),
            "section": chunk.get("section"),
        })

    _emit("indexing", 72, "Запись в Qdrant...")
    client = get_qdrant()
    fallback_dim = get_settings().embedding_fallback_dim
    collection = COLLECTION_NAME
    if embeddings and len(embeddings[0]) == fallback_dim:
        collection = f"{COLLECTION_NAME}_fallback"
    batch_size = 100
    total = len(points)
    for batch_idx, i in enumerate(range(0, total, batch_size)):
        batch = points[i:i + batch_size]
        from qdrant_client.models import PointStruct
        await client.upsert(
            collection_name=collection,
            points=[PointStruct(**p) for p in batch],
        )
        pct = 72 + int(23 * (batch_idx + 1) * batch_size / total)
        _emit("indexing", min(pct, 95), f"Индексация: {min(i + batch_size, total)}/{total} векторов")

    _emit("bm25", 96, "Построение BM25 индекса...")
    build_bm25_index(bm25_chunks)
    logger.info(f"BM25 индекс построен для {len(bm25_chunks)} чанков")
    _emit("done", 100, "Готово")

    return {
        "document_id": doc_id,
        "chunks_count": len(chunks),
        "status": "indexed",
        "chunks": bm25_chunks,
    }


async def ingest_directory(dir_path: str, doc_type: str = "textbook") -> list[dict]:
    results = []
    path = Path(dir_path)

    for file_path in sorted(path.rglob("*")):
        if file_path.suffix.lower() in (".md", ".txt", ".pdf", ".docx", ".xlsx", ".csv"):
            if file_path.stat().st_size < 50:
                continue
            try:
                result = await ingest_file(str(file_path), file_path.name, doc_type)
                results.append({"file": file_path.name, **result})
            except Exception as e:
                results.append({"file": file_path.name, "status": "error", "error": str(e)})

    return results
