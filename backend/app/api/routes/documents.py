from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Literal
from io import BytesIO
from fpdf import FPDF
from pathlib import Path
from sqlalchemy import select, delete
import hashlib
import logging
import os
import shutil
import time
import uuid

from app.services.ingestion import ingest_file, ingest_directory, UPLOAD_DIR
from app.services.processing_queue import get_processing_queue
from app.config import get_settings
from app.db.database import get_session
from app.db.orm import Document, Chunk
from app.db.qdrant_client import get_qdrant, COLLECTION_NAME
from qdrant_client.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)

router = APIRouter()

upload_progress: dict = {}


def compute_file_hash(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


async def _ingest_task(
    upload_id: str,
    file_path: str,
    title: str,
    doc_type: str,
    ocr_provider: Optional[str],
    file_hash: str,
):
    def _progress(stage: str, percent: float, detail: str):
        upload_progress[upload_id] = {
            "status": "processing",
            "stage": stage,
            "percent": int(percent),
            "detail": detail,
            "queued_at": upload_progress[upload_id].get("queued_at"),
        }

    upload_progress[upload_id] = {
        "status": "processing",
        "stage": "uploading",
        "percent": 5,
        "detail": "Файл сохранён",
    }

    try:
        result = await ingest_file(
            str(file_path),
            title,
            doc_type,
            ocr_provider=ocr_provider,
            progress=_progress,
            file_size=None,
        )

        async with get_session() as session:
            doc = Document(
                id=result["document_id"],
                title=title,
                source=str(file_path),
                doc_type=doc_type,
                status="indexed",
                chunks_count=result["chunks_count"],
                index_completed=True,
                file_hash=file_hash,
            )
            session.add(doc)

            for i, chunk_data in enumerate(result.get("chunks", [])):
                chunk_db = Chunk(
                    id=chunk_data["id"],
                    document_id=result["document_id"],
                    content=chunk_data["content"],
                    page=chunk_data.get("page"),
                    section=chunk_data.get("section"),
                    position=i,
                )
                session.add(chunk_db)

            await session.commit()

        upload_progress[upload_id] = {
            "status": "done",
            "document_id": result["document_id"],
            "chunks_count": result["chunks_count"],
        }
    except Exception as e:
        logger.exception(f"Ingest error for {upload_id}: {e}")
        upload_progress[upload_id] = {"status": "error", "detail": str(e)}


@router.get("/upload-progress/{upload_id}")
async def get_upload_progress(upload_id: str):
    progress = upload_progress.get(upload_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Upload not found")
    return progress


settings = get_settings()


@router.get("/queue")
async def get_queue_status():
    queue = get_processing_queue()
    return queue.get_status()


@router.get("/ocr-providers")
async def list_ocr_providers():
    from app.services.glm_health import is_glm_available, get_glm_unavailable_reason
    from app.services.paddle_health import is_paddle_available, get_paddle_unavailable_reason

    glm_available = is_glm_available()
    paddle_available = is_paddle_available()

    return {
        "current": get_settings().ocr_provider,
        "available": [
            {
                "id": "mistral",
                "name": "Mistral OCR",
                "description": "Платный облачный OCR через Mistral API",
                "requires_api_key": True,
                "available": True,
                "unavailable_reason": None,
            },
            {
                "id": "glm",
                "name": "GLM-OCR",
                "description": "Локальный OCR через модель GLM-OCR (zai-org/GLM-OCR)",
                "requires_api_key": False,
                "available": glm_available,
                "unavailable_reason": get_glm_unavailable_reason() if not glm_available else None,
            },
            {
                "id": "paddle",
                "name": "PaddleOCR",
                "description": "Локальный OCR через PaddleOCR (Baidu) - высокое качество",
                "requires_api_key": False,
                "available": paddle_available,
                "unavailable_reason": get_paddle_unavailable_reason() if not paddle_available else None,
            },
        ],
    }


@router.get("")
async def list_documents():
    async with get_session() as session:
        stmt = select(Document).order_by(Document.created_at.desc())
        result = await session.execute(stmt)
        docs = result.scalars().all()
        
        documents_list = [
            {
                "id": str(doc.id),
                "title": doc.title,
                "doc_type": doc.doc_type,
                "status": doc.status,
                "chunks_count": doc.chunks_count,
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "source": doc.source,
            }
            for doc in docs
        ]
        
        return {
            "documents": documents_list,
            "total": len(documents_list)
        }


@router.get("/{document_id}")
async def get_document(document_id: str):
    async with get_session() as session:
        stmt = select(Document).where(Document.id == document_id)
        result = await session.execute(stmt)
        doc = result.scalar_one_or_none()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return {
            "id": str(doc.id),
            "title": doc.title,
            "doc_type": doc.doc_type,
            "status": doc.status,
            "chunks_count": doc.chunks_count,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
            "source": doc.source,
        }


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    logger.info(f"Deleting document: {document_id}")
    async with get_session() as session:
        stmt = select(Document).where(Document.id == document_id)
        result = await session.execute(stmt)
        doc = result.scalar_one_or_none()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        if doc.source and Path(doc.source).exists():
            try:
                os.unlink(doc.source)
            except OSError as e:
                logger.warning(f"Failed to delete file {doc.source}: {e}")

        stmt_chunks = delete(Chunk).where(Chunk.document_id == document_id)
        await session.execute(stmt_chunks)
        
        try:
            qdrant = get_qdrant()
            await qdrant.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            )
            logger.info(f"Deleted vectors for document {document_id}")
        except Exception as e:
            logger.warning(f"Failed to delete vectors for {document_id}: {e}")
        
        stmt = delete(Document).where(Document.id == document_id)
        await session.execute(stmt)
        await session.commit()

    return {"deleted": True, "document_id": document_id}


@router.get("/{document_id}/download")
async def download_document(document_id: str):
    async with get_session() as session:
        stmt = select(Document).where(Document.id == document_id)
        result = await session.execute(stmt)
        doc = result.scalar_one_or_none()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        file_path = doc.source
        if not file_path or not Path(file_path).exists():
            raise HTTPException(status_code=404, detail="File not found on disk")

        return FileResponse(
            path=file_path,
            filename=doc.title or file_path.split("/")[-1],
            media_type="application/octet-stream",
        )


@router.get("/{document_id}/ocr-pdf")
async def download_ocr_text_pdf(document_id: str):
    async with get_session() as session:
        doc = await session.get(Document, document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        stmt = select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.page)
        result = await session.execute(stmt)
        chunks = result.scalars().all()
        
        if not chunks:
            raise HTTPException(status_code=404, detail="No OCR content available for this document")
        
        pdf = FPDF()
        pdf.add_page()
        
        pdf.add_font("DejaVu", "", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=11)
        
        pdf.set_font("DejaVu", size=14)
        pdf.cell(0, 10, f"OCR Текст: {doc.title}", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(5)
        
        pdf.set_font("DejaVu", size=11)
        for chunk in chunks:
            if chunk.page:
                pdf.set_font("DejaVu", size=10)
                pdf.set_text_color(128, 128, 128)
                pdf.cell(0, 5, f"--- Страница {chunk.page} ---", new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)
                pdf.set_font("DejaVu", size=11)
                pdf.set_text_color(0, 0, 0)
            
            text = chunk.content or ""
            pdf.multi_cell(0, 6, text)
            pdf.ln(3)
        
        pdf_bytes = BytesIO()
        pdf.output(pdf_bytes)
        pdf_bytes.seek(0)
        
        base_name = Path(doc.title).stem if doc.title else "document"
        filename = f"{base_name}_ocr.pdf"
        
        return StreamingResponse(
            pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )


class IngestDirectoryRequest(BaseModel):
    directory_path: str
    doc_type: str = "textbook"
    ocr_provider: Optional[Literal["mistral", "glm", "paddle"]] = None


class DocumentResponse(BaseModel):
    document_id: str
    title: str
    chunks_count: int
    status: str


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    doc_type: str = "textbook",
    ocr_provider: Optional[str] = Query(None),
):
    logger.info(f"Upload: {file.filename}, {file.size} bytes, type={doc_type}, ocr={ocr_provider}")

    from app.services.glm_health import is_glm_available
    from app.services.paddle_health import is_paddle_available

    if not ocr_provider or ocr_provider == "auto":
        if is_glm_available():
            ocr_provider = "glm"
        elif is_paddle_available():
            ocr_provider = "paddle"
        else:
            ocr_provider = "mistral"
        logger.info(f"Auto-selected OCR provider: {ocr_provider}")

    if ocr_provider == "glm" and not is_glm_available():
        logger.warning("GLM-OCR requested but unavailable, falling back to mistral")
        ocr_provider = "mistral"

    if ocr_provider == "paddle" and not is_paddle_available():
        logger.warning("PaddleOCR requested but unavailable, falling back to mistral")
        ocr_provider = "mistral"

    logger.info(f"OCR provider: {ocr_provider}")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size = file_path.stat().st_size
    file_hash = compute_file_hash(str(file_path))
    logger.info(f"File saved: {file_path} ({file_size} bytes), hash: {file_hash[:16]}...")

    async with get_session() as session:
        stmt = select(Document).where(Document.file_hash == file_hash)
        result = await session.execute(stmt)
        existing_doc = result.scalar_one_or_none()
        if existing_doc:
            os.unlink(file_path)
            raise HTTPException(
                status_code=409,
                detail=f"Документ '{existing_doc.title}' уже загружен (ID: {existing_doc.id})"
            )

    queue = get_processing_queue()
    queue.set_ingest_fn(_ingest_task)

    upload_progress[file_id] = {
        "status": "queued",
        "stage": "waiting",
        "percent": 0,
        "detail": f"В очереди ({file_size} байт)",
        "queued_at": time.time(),
        "title": file.filename,
        "size": file_size,
    }

    queue.submit(
        upload_id=file_id,
        file_path=str(file_path),
        title=file.filename,
        doc_type=doc_type,
        ocr_provider=ocr_provider,
        file_size=file_size,
        file_hash=file_hash,
    )

    return {"upload_id": file_id, "status": "queued", "queue_size": queue.get_status()["queue_size"]}


@router.post("/ingest-directory")
async def ingest_dir(request: IngestDirectoryRequest):
    if not Path(request.directory_path).exists():
        raise HTTPException(status_code=404, detail="Directory not found")

    results = await ingest_directory(request.directory_path, request.doc_type)
    return {"results": results, "total": len(results)}


@router.post("/ocr-text")
async def get_raw_ocr_text(
    file: UploadFile = File(...),
    ocr_provider: Literal["mistral", "glm", "paddle"] = Query(...),
):
    from app.services.mistral_ocr import ocr_pdf_file
    from app.services.glm_ocr import ocr_pdf_file_glm
    from app.services.paddle_ocr import ocr_pdf_file_paddle
    import tempfile
    import time

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        start = time.time()
        if ocr_provider == "mistral":
            text = await ocr_pdf_file(tmp_path)
        elif ocr_provider == "paddle":
            text = await ocr_pdf_file_paddle(tmp_path)
        else:
            text = await ocr_pdf_file_glm(tmp_path)
        elapsed = time.time() - start
    finally:
        os.unlink(tmp_path)

    return {
        "ocr_provider": ocr_provider,
        "text": text,
        "char_count": len(text),
        "processing_time": elapsed,
    }
