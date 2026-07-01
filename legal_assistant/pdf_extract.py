from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def extract_text_by_page(pdf_path: Path) -> list[tuple[int, str]]:
    """Return list of (1-based page number, text) for each page."""
    reader = PdfReader(str(pdf_path))
    out: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        out.append((i + 1, text.strip()))
    return out


def build_marked_document(pdf_path: Path) -> str:
    """Single string with page markers for better citations in chunks."""
    parts: list[str] = []
    for page_no, text in extract_text_by_page(pdf_path):
        if not text:
            continue
        parts.append(f"--- Page {page_no} ---\n{text}")
    return "\n\n".join(parts)
