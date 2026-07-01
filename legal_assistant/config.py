import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PDF = PROJECT_ROOT / "data" / "BNS2023.pdf"
_index_env = os.getenv("INDEX_DIR")
if _index_env:
    _p = Path(_index_env)
    INDEX_DIR = _p.resolve() if _p.is_absolute() else (PROJECT_ROOT / _p).resolve()
else:
    INDEX_DIR = PROJECT_ROOT / "bns_index"
BNS_PDF_PATH = Path(os.getenv("BNS_PDF_PATH", _DEFAULT_PDF))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Flash-Lite uses fewer tokens / lower rate pressure on free tier than full Flash.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "4"))
# Cap each retrieved chunk to limit input tokens (free tier is tight).
GEMINI_MAX_EXCERPT_CHARS = int(os.getenv("GEMINI_MAX_EXCERPT_CHARS", "3200"))
# Only include the last N user+assistant turns in the prompt (excluding the new user line).
GEMINI_MAX_HISTORY_MESSAGES = int(os.getenv("GEMINI_MAX_HISTORY_MESSAGES", "8"))

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS = 500
CHUNK_OVERLAP_WORDS = 75
TOP_K = 3

# Cross-encoder reranking (optional): bi-encoder pulls a pool, cross-encoder rescores pairs.
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


USE_CROSS_ENCODER = _env_bool("USE_CROSS_ENCODER", False)
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
# How many bi-encoder hits to consider before reranking (must be >= TOP_K).
BI_ENCODER_POOL = max(TOP_K, int(os.getenv("BI_ENCODER_POOL", "24")))
# Cross-encoders use short contexts; trim only for scoring, full chunk is still returned.
CROSS_ENCODER_MAX_PASSAGE_CHARS = int(os.getenv("CROSS_ENCODER_MAX_PASSAGE_CHARS", "2000"))

EMBEDDINGS_FILE = "embeddings.npy"
CHUNKS_FILE = "chunks.jsonl"
