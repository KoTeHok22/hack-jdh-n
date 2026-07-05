import logging
import fitz
from docx import Document as DocxDocument
import pandas as pd
from pathlib import Path
from io import BytesIO
from typing import Optional
from app.services.mistral_ocr import ocr_pdf_file
from app.services.glm_ocr import ocr_pdf_file_glm
from app.services.paddle_ocr import ocr_pdf_file_paddle
from app.config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


MIN_TEXT_CHARS = 100


async def parse_pdf(file_path: str, ocr_provider: Optional[str] = None) -> str:
    provider = ocr_provider or _settings.ocr_provider

    doc = fitz.open(file_path)
    text_parts = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            text_parts.append(f"[Page {page_num + 1}]\n{text}")
    doc.close()

    result = "\n\n".join(text_parts)
    if len(result.strip()) >= MIN_TEXT_CHARS:
        logger.info(f"PDF is text-based, extracted {len(result)} chars via PyMuPDF (no OCR needed)")
        return result

    logger.info(f"PDF appears to be a scan ({len(result.strip())} chars of text), using OCR via {provider}")
    return await _ocr_pdf(file_path, provider=provider)


async def _ocr_pdf(file_path: str, provider: Optional[str] = None) -> str:
    active = provider or _settings.ocr_provider
    if active == "glm":
        return await ocr_pdf_file_glm(file_path)
    if active == "paddle":
        return await ocr_pdf_file_paddle(file_path)
    return await ocr_pdf_file(file_path)


def parse_docx(file_path: str) -> str:
    doc = DocxDocument(file_path)
    paragraphs = []
    for para in doc.paragraphs:
        if para.text.strip():
            paragraphs.append(para.text)
    return "\n\n".join(paragraphs)


def parse_txt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8")


def parse_md(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8")


def parse_xlsx(file_path: str) -> str:
    xls = pd.ExcelFile(file_path)
    parts = []
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        parts.append(f"=== Sheet: {sheet_name} ===\n{df.to_string(index=False)}")
    return "\n\n".join(parts)


def parse_csv(file_path: str) -> str:
    df = pd.read_csv(file_path)
    return df.to_string(index=False)


async def parse_file(file_path: str, ocr_provider: Optional[str] = None) -> str:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return await parse_pdf(file_path, ocr_provider=ocr_provider)
    parsers = {
        ".docx": parse_docx,
        ".txt": parse_txt,
        ".md": parse_md,
        ".xlsx": parse_xlsx,
        ".csv": parse_csv,
    }
    parser = parsers.get(ext)
    if not parser:
        raise ValueError(f"Unsupported file type: {ext}")
    return parser(file_path)
