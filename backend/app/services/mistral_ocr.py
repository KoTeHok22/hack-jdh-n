import logging
import os
import tempfile
from typing import List, Dict, Any
import httpx
import fitz
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
OCR_MODEL = "mistral-ocr-latest"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
async def process_pdf_with_mistral(pdf_path: str) -> List[Dict[str, Any]]:
    url = f"{MISTRAL_BASE_URL}/ocr"
    headers = {
        "Authorization": f"Bearer {settings.mistral_api_key}",
        "Content-Type": "application/json"
    }
    
    with open(pdf_path, "rb") as f:
        import base64
        pdf_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {
        "model": OCR_MODEL,
        "document": {
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{pdf_base64}"
        }
    }
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        if not response.is_success:
            logger.error(f"Mistral OCR error {response.status_code}: {response.text}")
        response.raise_for_status()
        data = response.json()
        
        pages = []
        for page_data in data.get("pages", []):
            pages.append({
                "index": page_data.get("index"),
                "markdown": page_data.get("markdown", ""),
                "tables": page_data.get("tables", [])
            })
        return pages


def split_pdf_by_pages(pdf_path: str, max_pages_per_chunk: int = 20) -> List[str]:
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    chunk_paths = []
    
    for i in range(0, total_pages, max_pages_per_chunk):
        start_page = i
        end_page = min(i + max_pages_per_chunk, total_pages)
        
        chunk_doc = fitz.open()
        for page_idx in range(start_page, end_page):
            chunk_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
        
        chunk_path = tempfile.mktemp(suffix=f"_chunk_{i}.pdf")
        chunk_doc.save(chunk_path)
        chunk_doc.close()
        chunk_paths.append(chunk_path)
    
    doc.close()
    return chunk_paths


async def ocr_pdf_file(file_path: str) -> str:
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    logger.info(f"Processing PDF: {file_path}, size: {file_size_mb:.2f} MB")
    
    if file_size_mb <= settings.mistral_max_file_size_mb:
        logger.info("File size within limits, processing directly")
        pages = await process_pdf_with_mistral(file_path)
        return combine_pages(pages)
    
    logger.info(f"File too large ({file_size_mb:.2f} MB), splitting into chunks")
    chunk_paths = split_pdf_by_pages(file_path, max_pages_per_chunk=15)
    
    all_pages = []
    for i, chunk_path in enumerate(chunk_paths):
        logger.info(f"Processing chunk {i+1}/{len(chunk_paths)}")
        try:
            pages = await process_pdf_with_mistral(chunk_path)
            all_pages.extend(pages)
        except Exception as e:
            logger.error(f"Failed to process chunk {i+1}: {e}")
        finally:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
    
    result = combine_pages(all_pages)
    logger.info(f"OCR completed: {len(all_pages)} pages processed")
    return result


def combine_pages(pages: List[Dict[str, Any]]) -> str:
    combined = []
    for page in sorted(pages, key=lambda x: x["index"]):
        combined.append(f"# Page {page['index'] + 1}\n\n{page['markdown']}")
    return "\n\n---\n\n".join(combined)
