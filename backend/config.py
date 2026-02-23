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
SQLITE_DB_PATH = Path(os.getenv("SQLITE_DB_PATH", str(DB_DIR / "app.db"))).resolve()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DB_BACKEND = os.getenv("DB_BACKEND", "postgres" if DATABASE_URL else "sqlite").strip().lower()
if DB_BACKEND not in {"sqlite", "postgres"}:
    DB_BACKEND = "sqlite"
PGVECTOR_DIM = int(os.getenv("PGVECTOR_DIM", "1536"))
PG_CONNECT_TIMEOUT_SEC = float(os.getenv("PG_CONNECT_TIMEOUT_SEC", "5"))

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
ENABLE_AI_INGEST_SUMMARIES = _env_bool("ENABLE_AI_INGEST_SUMMARIES", False)
ENABLE_PADDLE_OCR = _env_bool("ENABLE_PADDLE_OCR", True)
PADDLE_OCR_LANG = os.getenv("PADDLE_OCR_LANG", "en").strip() or "en"
PADDLE_OCR_USE_GPU = _env_bool("PADDLE_OCR_USE_GPU", False)
PADDLE_OCR_MIN_CONFIDENCE = float(os.getenv("PADDLE_OCR_MIN_CONFIDENCE", "0.50"))
PADDLE_OCR_MAX_RETRIES = max(0, int(os.getenv("PADDLE_OCR_MAX_RETRIES", "1")))
OCR_WORKER_TIMEOUT_SEC = float(os.getenv("OCR_WORKER_TIMEOUT_SEC", "60"))
OCR_WORKER_STARTUP_TIMEOUT_SEC = float(os.getenv("OCR_WORKER_STARTUP_TIMEOUT_SEC", "120"))
OCR_NATIVE_TEXT_MIN_CHARS = int(os.getenv("OCR_NATIVE_TEXT_MIN_CHARS", "1"))
PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "220"))
ENABLE_PPTX_SLIDE_RENDER = _env_bool("ENABLE_PPTX_SLIDE_RENDER", True)
ENABLE_PPTX_RELATIONSHIP_GRAPH = _env_bool("ENABLE_PPTX_RELATIONSHIP_GRAPH", True)
PPTX_GRAPH_MAX_EDGES = int(os.getenv("PPTX_GRAPH_MAX_EDGES", "120"))
ENABLE_DIAGRAM_PIPELINE = _env_bool("ENABLE_DIAGRAM_PIPELINE", True)
ENABLE_YOLO_DIAGRAM_DETECTOR = _env_bool("ENABLE_YOLO_DIAGRAM_DETECTOR", True)
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo26n.pt").strip() or "yolo26n.pt"
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "auto").strip() or "auto"
YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.25"))
YOLO_IOU_THRESHOLD = float(os.getenv("YOLO_IOU_THRESHOLD", "0.45"))
YOLO_IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "960"))
DIAGRAM_MIN_NODE_AREA = int(os.getenv("DIAGRAM_MIN_NODE_AREA", "1600"))
DIAGRAM_MIN_EDGE_LENGTH = int(os.getenv("DIAGRAM_MIN_EDGE_LENGTH", "48"))
DIAGRAM_MAX_NODES = int(os.getenv("DIAGRAM_MAX_NODES", "80"))
DIAGRAM_MAX_EDGES = int(os.getenv("DIAGRAM_MAX_EDGES", "180"))
DIAGRAM_MAX_NODE_CHUNKS = int(os.getenv("DIAGRAM_MAX_NODE_CHUNKS", "24"))
DIAGRAM_MAX_EDGE_CHUNKS = int(os.getenv("DIAGRAM_MAX_EDGE_CHUNKS", "24"))
DIAGRAM_TOP_K_FLOOR = int(os.getenv("DIAGRAM_TOP_K_FLOOR", "14"))
DIAGRAM_PER_DOC_LIMIT_FLOOR = int(os.getenv("DIAGRAM_PER_DOC_LIMIT_FLOOR", "4"))
DIAGRAM_MIN_EVIDENCE_SOURCE_TYPES = int(os.getenv("DIAGRAM_MIN_EVIDENCE_SOURCE_TYPES", "3"))
DIAGRAM_MIN_GRAPH_CHUNKS = int(os.getenv("DIAGRAM_MIN_GRAPH_CHUNKS", "1"))
DIAGRAM_MIN_OCR_CHUNKS = int(os.getenv("DIAGRAM_MIN_OCR_CHUNKS", "2"))
DIAGRAM_MIN_NODE_CHUNKS = int(os.getenv("DIAGRAM_MIN_NODE_CHUNKS", "2"))
DIAGRAM_MIN_EDGE_CHUNKS = int(os.getenv("DIAGRAM_MIN_EDGE_CHUNKS", "2"))
DIAGRAM_MIN_SLIDE_GRAPH_CHUNKS = int(os.getenv("DIAGRAM_MIN_SLIDE_GRAPH_CHUNKS", "0"))
DIAGRAM_MIXED_EVIDENCE_LIMIT = int(os.getenv("DIAGRAM_MIXED_EVIDENCE_LIMIT", "72"))

def ensure_dirs() -> None:
    for path in (UPLOAD_DIR, PROCESSED_DIR, INDEX_DIR, DB_DIR):
        path.mkdir(parents=True, exist_ok=True)
