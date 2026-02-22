import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
DB_DIR = DATA_DIR / "db"

MODEL_PROVIDER = "openai"
EMBED_PROVIDER = "openai"
ROUTER_PROVIDER = "openai"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o").strip()
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()
OPENAI_ROUTER_MODEL = os.getenv("OPENAI_ROUTER_MODEL", "gpt-4o-mini").strip()

# Core retrieval/chat settings
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
TOP_K = int(os.getenv("TOP_K", "6"))
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").strip().lower()
ENABLE_RERANKER = _env_bool("ENABLE_RERANKER", True)
RERANK_MODEL_ID = os.getenv("RERANK_MODEL_ID", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "auto")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "20"))

def ensure_dirs() -> None:
    for path in (UPLOAD_DIR, PROCESSED_DIR, INDEX_DIR, DB_DIR):
        path.mkdir(parents=True, exist_ok=True)
