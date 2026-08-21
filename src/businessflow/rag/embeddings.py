"""The one embedding model, shared by ingestion and retrieval. Both must
use the exact same model, or the vectors they produce aren't comparable --
this is the single place that loads it, so there's no way for the two to
drift apart.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

_EMBED_MODEL = "intfloat/multilingual-e5-small"


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    # int8-quantized ONNX -- same RAM/speed reasoning as Whisper and the
    # reranker elsewhere in this project.
    return SentenceTransformer(
        _EMBED_MODEL, backend="onnx", model_kwargs={"file_name": "onnx/model_qint8_avx512_vnni.onnx"}
    )


def embed_passages(texts: list[str]):
    """For text being stored/indexed. e5 models require the "passage: "
    prefix on indexed text -- dropping it measurably hurts retrieval
    quality, it's not decoration."""
    return _model().encode([f"passage: {t}" for t in texts], normalize_embeddings=True)


def embed_query(text: str):
    """For a search query. Deliberately a different prefix than
    embed_passages -- e5 was trained on that asymmetry between what's
    being searched for and what's being stored."""
    return _model().encode([f"query: {text}"], normalize_embeddings=True)[0]
