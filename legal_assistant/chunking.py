from __future__ import annotations

from legal_assistant.config import CHUNK_OVERLAP_WORDS, CHUNK_WORDS


def chunk_by_words(
    text: str,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """Split text into overlapping word windows (default ~500 words)."""
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_words - overlap_words)
    chunks: list[str] = []
    i = 0
    while i < len(words):
        piece = words[i : i + chunk_words]
        chunks.append(" ".join(piece))
        i += step
    return chunks
