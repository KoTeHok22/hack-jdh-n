import fitz
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)


async def ocr_pdf_file_paddle(file_path: str) -> str:
    """OCR PDF using PaddleOCR"""
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        raise RuntimeError("PaddleOCR не установлен. Установите: pip install paddleocr paddlepaddle")
    
    ocr = PaddleOCR(use_textline_orientation=True, lang='ru')
    
    doc = fitz.open(file_path)
    page_count = len(doc)
    logger.info(f"PaddleOCR: processing PDF {file_path}, pages: {page_count}")
    
    text_parts = []
    
    for page_num in range(page_count):
        logger.info(f"PaddleOCR: processing page {page_num + 1}/{page_count}")
        page = doc[page_num]
        
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img_path = f"/tmp/paddle_page_{page_num}.png"
        pix.save(img_path)
        
        try:
            result = ocr.ocr(img_path, cls=True)
            
            if result and result[0]:
                page_text = []
                for line in result[0]:
                    text = line[1][0]
                    page_text.append(text)
                text_parts.append(f"[Page {page_num + 1}]\n" + "\n".join(page_text))
            else:
                text_parts.append(f"[Page {page_num + 1}]\n(No text detected)")
        
        finally:
            if Path(img_path).exists():
                Path(img_path).unlink()
    
    doc.close()
    logger.info(f"PaddleOCR: completed {page_count} pages")
    
    return "\n\n".join(text_parts)
