"""Chunk BNS PDF and build local embedding index."""

from __future__ import annotations

import argparse
import shutil

from legal_assistant.chunking import chunk_by_words
from legal_assistant.config import BNS_PDF_PATH, INDEX_DIR
from legal_assistant.local_index import build_index, index_paths
from legal_assistant.pdf_extract import build_marked_document


def run_ingest(reset: bool = False) -> int:
    if not BNS_PDF_PATH.is_file():
        raise FileNotFoundError(
            f"PDF not found at {BNS_PDF_PATH}. Set BNS_PDF_PATH in .env or place the file there."
        )

    doc = build_marked_document(BNS_PDF_PATH)
    chunks = chunk_by_words(doc)
    if not chunks:
        raise ValueError("No text extracted from PDF. The file may be image-only or encrypted.")

    if reset and INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)

    build_index(chunks, INDEX_DIR)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest BNS 2023 PDF into local vector index")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing index folder before indexing",
    )
    args = parser.parse_args()
    n = run_ingest(reset=args.reset)
    emb_path, _ = index_paths(INDEX_DIR)
    print(f"Ingested {n} chunks. Index at {INDEX_DIR} (embeddings: {emb_path})")


if __name__ == "__main__":
    main()
