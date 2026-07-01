"""Local vector index: sentence-transformers embeddings + NumPy cosine similarity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from legal_assistant.config import (
    BI_ENCODER_POOL,
    CHUNKS_FILE,
    CROSS_ENCODER_MAX_PASSAGE_CHARS,
    CROSS_ENCODER_MODEL,
    EMBEDDING_MODEL,
    EMBEDDINGS_FILE,
    INDEX_DIR,
    TOP_K,
    USE_CROSS_ENCODER,
)

_model: SentenceTransformer | None = None
_ce_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_cross_encoder():
    global _ce_model
    if _ce_model is None:
        from sentence_transformers import CrossEncoder

        _ce_model = CrossEncoder(CROSS_ENCODER_MODEL)
    return _ce_model


def _clip_passage_for_ce(text: str) -> str:
    if len(text) <= CROSS_ENCODER_MAX_PASSAGE_CHARS:
        return text
    return text[:CROSS_ENCODER_MAX_PASSAGE_CHARS]


def _rerank_with_cross_encoder(question: str, candidates: list[str], top_k: int) -> list[str]:
    if len(candidates) <= top_k:
        return candidates[:top_k]
    ce = _get_cross_encoder()
    pairs = [[question, _clip_passage_for_ce(t)] for t in candidates]
    scores = ce.predict(pairs, show_progress_bar=False)
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    return [candidates[int(i)] for i in order[:top_k]]


def index_paths(base: Path | None = None) -> tuple[Path, Path]:
    root = base or INDEX_DIR
    return root / EMBEDDINGS_FILE, root / CHUNKS_FILE


def build_index(chunks: list[str], base: Path | None = None) -> None:
    if not chunks:
        raise ValueError("No chunks to index.")
    root = base or INDEX_DIR
    root.mkdir(parents=True, exist_ok=True)
    model = _get_model()
    emb = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    np.save(index_paths(root)[0], emb.astype(np.float32))
    path_chunks = index_paths(root)[1]
    with path_chunks.open("w", encoding="utf-8") as f:
        for i, text in enumerate(chunks):
            f.write(json.dumps({"id": i, "text": text}, ensure_ascii=False) + "\n")


def load_index(base: Path | None = None) -> tuple[np.ndarray, list[str]] | None:
    root = base or INDEX_DIR
    pe, pc = index_paths(root)
    if not pe.is_file() or not pc.is_file():
        return None
    emb = np.load(pe)
    texts: list[str] = []
    with pc.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
    if len(texts) != emb.shape[0]:
        return None
    return emb, texts


def query_index(question: str, base: Path | None = None, top_k: int = TOP_K) -> list[str]:
    loaded = load_index(base)
    if loaded is None:
        return []
    emb, texts = loaded
    model = _get_model()
    q = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = emb @ q

    pool = min(BI_ENCODER_POOL if USE_CROSS_ENCODER else top_k, len(texts))
    pool = max(pool, min(top_k, len(texts)))
    k = min(pool, len(texts))
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    candidates = [texts[int(i)] for i in idx]

    if USE_CROSS_ENCODER and len(candidates) > top_k:
        return _rerank_with_cross_encoder(question, candidates, top_k)
    return candidates[:top_k]


def index_size(base: Path | None = None) -> int:
    loaded = load_index(base)
    if loaded is None:
        return 0
    return len(loaded[1])
