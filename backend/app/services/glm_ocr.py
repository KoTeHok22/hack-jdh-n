import logging
import os
import base64
from typing import List
import httpx
import fitz
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def pdf_page_to_base64_image(pdf_path: str, page_index: int, dpi: int = 200) -> str:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_index)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    img_data = pix.tobytes("png")
    doc.close()
    return base64.b64encode(img_data).decode("utf-8")


def get_pdf_page_count(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
async def process_image_with_glm_ocr(image_base64: str, prompt: str = "Text Recognition:") -> str:
    url = f"{settings.glm_ocr_base_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": settings.glm_ocr_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 8192,
        "temperature": 0.0,
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def ocr_pdf_file_glm(pdf_path: str) -> str:
    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    total_pages = get_pdf_page_count(pdf_path)
    logger.info(f"GLM-OCR: Processing PDF {pdf_path}, size: {file_size_mb:.2f} MB, pages: {total_pages}")

    pages_text: List[str] = []
    for page_idx in range(total_pages):
        logger.info(f"GLM-OCR: Processing page {page_idx + 1}/{total_pages}")
        try:
            img_b64 = pdf_page_to_base64_image(pdf_path, page_idx, dpi=settings.glm_ocr_dpi)
            text = await process_image_with_glm_ocr(img_b64)
            pages_text.append(f"# Page {page_idx + 1}\n\n{text}")
        except Exception as e:
            logger.error(f"GLM-OCR: Failed to process page {page_idx + 1}: {e}")
            pages_text.append(f"# Page {page_idx + 1}\n\n[OCR Error: {str(e)}]")

    result = "\n\n---\n\n".join(pages_text)
    logger.info(f"GLM-OCR: Completed {total_pages} pages")
    return result


async def ocr_image_file_glm(file_path: str) -> str:
    logger.info(f"GLM-OCR: Processing image {file_path}")
    with open(file_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    text = await process_image_with_glm_ocr(img_b64)
    return text
