import logging

logger = logging.getLogger(__name__)

_paddle_available: bool = False
_paddle_unavailable_reason: str = ""


def is_paddle_available() -> bool:
    return _paddle_available


def get_paddle_unavailable_reason() -> str:
    return _paddle_unavailable_reason


def check_paddle_health():
    global _paddle_available, _paddle_unavailable_reason

    try:
        from paddleocr import PaddleOCR
        _paddle_available = True
        _paddle_unavailable_reason = ""
        logger.info("PaddleOCR health check passed")
    except ImportError as e:
        _paddle_available = False
        _paddle_unavailable_reason = f"PaddleOCR не установлен: {e}"
        logger.warning(f"PaddleOCR unavailable: {_paddle_unavailable_reason}")
    except Exception as e:
        _paddle_available = False
        _paddle_unavailable_reason = f"Ошибка инициализации PaddleOCR: {e}"
        logger.warning(f"PaddleOCR unavailable: {_paddle_unavailable_reason}")
