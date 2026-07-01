"""
RAG pipeline: semantic search (local MiniLM index) then Gemini summarization.

Flow matches the usual agent pattern: retrieve → generate.
"""

from __future__ import annotations

from legal_assistant.config import (
    GEMINI_MAX_EXCERPT_CHARS,
    GEMINI_MAX_HISTORY_MESSAGES,
    INDEX_DIR,
    TOP_K,
)
from legal_assistant.gemini_rest import generate_text
from legal_assistant.local_index import index_size, query_index

SYSTEM_PROMPT = """You are a helpful assistant for Indian law as published in the Bharatiya Nyaya Sanhita (BNS), 2023.

Rules:
- You ONLY use the "Retrieved excerpts" below for specific section numbers, definitions, and legal wording. If the excerpts do not contain enough information, say clearly that the excerpts do not cover it and avoid inventing sections.
- Give practical procedural guidelines (steps, which documents may be needed, where to seek official help) when the excerpts support it. You are NOT a lawyer; remind the user that this is general information, not legal advice.
- Prefer simple English; if the user writes in Hindi, reply in Hindi.
- Quote or paraphrase closely from the excerpts when stating rules, and mention page markers if they appear in the text (e.g. "--- Page 12 ---").
"""


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated to save API quota]..."


def retrieve_excerpts(question: str) -> str:
    if index_size(INDEX_DIR) == 0:
        return "(No index found. Run: python -m legal_assistant.ingest)"
    docs = query_index(question, INDEX_DIR, top_k=TOP_K)
    parts: list[str] = []
    for i, d in enumerate(docs):
        if not d:
            continue
        parts.append(f"[Excerpt {i + 1}]\n{_clip(d, GEMINI_MAX_EXCERPT_CHARS)}")
    return "\n\n".join(parts)


def build_prompt(retrieved: str, history: list[dict]) -> str:
    # Drop oldest turns so free-tier token limits are not exceeded.
    tail = history[-GEMINI_MAX_HISTORY_MESSAGES:] if history else []
    lines: list[str] = []
    for turn in tail:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            lines.append(f"User: {_clip(str(content), 2000)}")
        elif role == "assistant":
            lines.append(f"Assistant: {_clip(str(content), 2000)}")
    convo = "\n".join(lines) if lines else "(empty)"
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Retrieved excerpts from BNS 2023 PDF:\n{retrieved}\n\n"
        f"Conversation:\n{convo}\n\n"
        "Respond to the user's latest message following the rules above."
    )


def chat_reply(history: list[dict], user_message: str) -> str:
    """history: prior turns [{role, content}, ...]; user_message is the new user text."""
    retrieved = retrieve_excerpts(user_message)
    full_history = list(history) + [{"role": "user", "content": user_message}]
    prompt = build_prompt(retrieved, full_history)
    return generate_text(prompt)
