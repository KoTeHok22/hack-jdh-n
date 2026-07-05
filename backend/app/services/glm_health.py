import logging
import httpx
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_glm_available: bool = False
_glm_unavailable_reason: str = ""


def is_glm_available() -> bool:
    return _glm_available


def get_glm_unavailable_reason() -> str:
    return _glm_unavailable_reason


async def check_glm_health():
    global _glm_available, _glm_unavailable_reason

    url = f"{settings.glm_ocr_base_url}/v1/models"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            _glm_available = True
            _glm_unavailable_reason = ""
            logger.info(f"GLM-OCR health check passed: {settings.glm_ocr_base_url}")
    except httpx.ConnectError as e:
        _glm_available = False
        _glm_unavailable_reason = f"Нет подключения к серверу GLM-OCR ({settings.glm_ocr_base_url}): {e}"
        logger.warning(f"GLM-OCR unavailable: {_glm_unavailable_reason}")
    except httpx.HTTPStatusError as e:
        _glm_available = False
        _glm_unavailable_reason = f"Ошибка HTTP от GLM-OCR сервера: {e.response.status_code}"
        logger.warning(f"GLM-OCR unavailable: {_glm_unavailable_reason}")
    except Exception as e:
        _glm_available = False
        _glm_unavailable_reason = f"Неизвестная ошибка при подключении к GLM-OCR: {e}"
        logger.warning(f"GLM-OCR unavailable: {_glm_unavailable_reason}")
