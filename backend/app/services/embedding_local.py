import logging
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

_model = None
_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_DIM = 384


def _get_model():
    global _model
    if _model is None:
        logger.info(f"Loading local embedding model {_MODEL_NAME}...")
        _model = TextEmbedding(_MODEL_NAME)
        logger.info("Local embedding model loaded")
    return _model


def is_model_loaded() -> bool:
    return _model is not None


async def embed_text_local(text: str) -> list[float]:
    model = _get_model()
    return next(model.embed([text])).tolist()


async def embed_texts_local(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    return [emb.tolist() for emb in model.embed(texts)]


async def test_local_embeddings() -> bool:
    try:
        model = _get_model()
        result = next(model.embed(["test"]))
        return len(result.tolist()) == _DIM
    except Exception as e:
        logger.error(f"Local embeddings test failed: {e}")
        return False
