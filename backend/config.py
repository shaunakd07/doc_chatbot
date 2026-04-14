import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def _normalize_doc_type(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_\- ]+", " ", str(value or "").strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip().replace("-", "_").replace(" ", "_")
    return cleaned


def _parse_tenant_thresholds(raw: str) -> dict[str, float]:
    mapping: dict[str, float] = {}
    text = str(raw or "").strip()
    if not text:
        return mapping
    for item in text.split(","):
        entry = str(item or "").strip()
        if not entry or ":" not in entry:
            continue
        tenant, threshold = entry.split(":", 1)
        tenant_id = str(tenant or "").strip()
        if not tenant_id:
            continue
        try:
            value = float(threshold)
        except Exception:
            continue
        mapping[tenant_id] = max(0.01, min(0.99, value))
    return mapping


def _parse_equivalent_types(raw: str) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    text = str(raw or "").strip()
    if not text:
        return mapping
    for item in text.split(";"):
        entry = str(item or "").strip()
        if not entry or "=" not in entry:
            continue
        left, right = entry.split("=", 1)
        key = _normalize_doc_type(left)
        if not key:
            continue
        values: set[str] = set()
        for raw_value in right.split("|"):
            normalized = _normalize_doc_type(raw_value)
            if normalized:
                values.add(normalized)
        if values:
            mapping[key] = values
    return mapping


def _parse_csv_types(raw: str) -> set[str]:
    values: set[str] = set()
    for item in str(raw or "").split(","):
        normalized = _normalize_doc_type(item)
        if normalized:
            values.add(normalized)
    return values


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
DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "default").strip() or "default"
INTERNAL_EMAIL_DOMAINS = [
    item.strip().lower()
    for item in os.getenv("INTERNAL_EMAIL_DOMAINS", "").split(",")
    if item.strip()
]

DOC_TYPE_CLASSIFIER_PROVIDER = os.getenv("DOC_TYPE_CLASSIFIER_PROVIDER", "heuristic").strip().lower() or "heuristic"
DOC_TYPE_SEMANTIC_MODEL = os.getenv("DOC_TYPE_SEMANTIC_MODEL", OPENAI_ROUTER_MODEL).strip() or OPENAI_ROUTER_MODEL
DOC_TYPE_SEMANTIC_TIMEOUT_SEC = float(os.getenv("DOC_TYPE_SEMANTIC_TIMEOUT_SEC", "20"))
AZURE_DOC_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT", "").strip()
AZURE_DOC_INTELLIGENCE_API_KEY = os.getenv("AZURE_DOC_INTELLIGENCE_API_KEY", "").strip()
AZURE_DOC_INTELLIGENCE_CLASSIFIER_ID = os.getenv("AZURE_DOC_INTELLIGENCE_CLASSIFIER_ID", "").strip()
AZURE_DOC_INTELLIGENCE_API_VERSION = os.getenv("AZURE_DOC_INTELLIGENCE_API_VERSION", "2024-11-30")
AZURE_DOC_INTELLIGENCE_TIMEOUT_SEC = float(os.getenv("AZURE_DOC_INTELLIGENCE_TIMEOUT_SEC", "45"))
AZURE_DOC_INTELLIGENCE_POLL_INTERVAL_SEC = float(os.getenv("AZURE_DOC_INTELLIGENCE_POLL_INTERVAL_SEC", "0.8"))

DOC_TYPE_REVIEW_ENABLED = _env_bool("DOC_TYPE_REVIEW_ENABLED", True)
DOC_TYPE_REVIEW_CONFIDENCE_THRESHOLD = float(os.getenv("DOC_TYPE_REVIEW_CONFIDENCE_THRESHOLD", "0.9"))
DOC_TYPE_REVIEW_MIN_SCORE_RATIO = float(os.getenv("DOC_TYPE_REVIEW_MIN_SCORE_RATIO", "1.3"))
DOC_TYPE_REVIEW_TENANT_THRESHOLDS = _parse_tenant_thresholds(
    os.getenv("DOC_TYPE_REVIEW_TENANT_THRESHOLDS", "")
)
DOC_TYPE_REVIEW_EQUIVALENT_TYPES = _parse_equivalent_types(
    os.getenv("DOC_TYPE_REVIEW_EQUIVALENT_TYPES", "contract=nda|statement_of_work;nda=contract")
)
DOC_TYPE_REVIEW_IGNORED_PREDICTED_TYPES = _parse_csv_types(
    os.getenv("DOC_TYPE_REVIEW_IGNORED_PREDICTED_TYPES", "unknown")
)

MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "16000"))
TOP_K = int(os.getenv("TOP_K", "6"))
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").strip().lower()
HYBRID_METADATA_SEMANTIC = _env_bool("HYBRID_METADATA_SEMANTIC", False)
HYBRID_METADATA_TOP_K = max(6, int(os.getenv("HYBRID_METADATA_TOP_K", "14")))
HYBRID_METADATA_PER_DOC_LIMIT = max(1, int(os.getenv("HYBRID_METADATA_PER_DOC_LIMIT", "3")))
HYBRID_METADATA_MAX_CANDIDATES = max(2, int(os.getenv("HYBRID_METADATA_MAX_CANDIDATES", "48")))
HYBRID_METADATA_MIN_EVIDENCE_SCORE = float(os.getenv("HYBRID_METADATA_MIN_EVIDENCE_SCORE", "0.28"))
ENABLE_HAYSTACK_RETRIEVAL = _env_bool("ENABLE_HAYSTACK_RETRIEVAL", False)
ENABLE_HAYSTACK_QUERY_EXPANSION = _env_bool("ENABLE_HAYSTACK_QUERY_EXPANSION", True)
CHAT_MEMORY_ENABLED = _env_bool("CHAT_MEMORY_ENABLED", True)
CHAT_MEMORY_RECENT_TURNS = max(1, int(os.getenv("CHAT_MEMORY_RECENT_TURNS", "6")))
CHAT_MEMORY_MAX_MESSAGES = max(CHAT_MEMORY_RECENT_TURNS * 2, int(os.getenv("CHAT_MEMORY_MAX_MESSAGES", "24")))
CHAT_MEMORY_REWRITE_MAX_CHARS = max(300, int(os.getenv("CHAT_MEMORY_REWRITE_MAX_CHARS", "2200")))
CHAT_MEMORY_SUMMARY_TARGET_CHARS = max(240, int(os.getenv("CHAT_MEMORY_SUMMARY_TARGET_CHARS", "700")))
CHAT_MEMORY_SUMMARY_MAX_CHARS = max(
    CHAT_MEMORY_SUMMARY_TARGET_CHARS,
    int(os.getenv("CHAT_MEMORY_SUMMARY_MAX_CHARS", "1400")),
)
ENABLE_RERANKER = _env_bool("ENABLE_RERANKER", True)
RERANK_MODEL_ID = os.getenv("RERANK_MODEL_ID", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "auto")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "36"))
INGEST_MAX_WORKERS = max(1, int(os.getenv("INGEST_MAX_WORKERS", "2")))
INGEST_PROGRESS_MIN_INTERVAL_SEC = float(os.getenv("INGEST_PROGRESS_MIN_INTERVAL_SEC", "0.75"))
INGEST_PROGRESS_MIN_DELTA = max(1, int(os.getenv("INGEST_PROGRESS_MIN_DELTA", "3")))
ENABLE_AI_INGEST_SUMMARIES = _env_bool("ENABLE_AI_INGEST_SUMMARIES", False)
_ENABLE_OCR_RAW = os.getenv("ENABLE_OCR")
ENABLE_OCR = _env_bool("ENABLE_OCR", True) if _ENABLE_OCR_RAW is not None else _env_bool("ENABLE_PADDLE_OCR", True)
OCR_ENGINE = os.getenv("OCR_ENGINE", "tesseract").strip().lower()
if OCR_ENGINE not in {"tesseract", "paddle"}:
    OCR_ENGINE = "tesseract"
PADDLE_OCR_LANG = os.getenv("PADDLE_OCR_LANG", "en").strip() or "en"
PADDLE_OCR_USE_GPU = _env_bool("PADDLE_OCR_USE_GPU", False)
PADDLE_OCR_MIN_CONFIDENCE = float(os.getenv("PADDLE_OCR_MIN_CONFIDENCE", "0.50"))
PADDLE_OCR_MAX_RETRIES = max(0, int(os.getenv("PADDLE_OCR_MAX_RETRIES", "1")))
PADDLE_OCR_REINIT_ON_FAILURE = _env_bool("PADDLE_OCR_REINIT_ON_FAILURE", False)
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = _env_bool("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", True)
OCR_TESSERACT_CMD = os.getenv("OCR_TESSERACT_CMD", "").strip()
OCR_TESSERACT_LANG = os.getenv("OCR_TESSERACT_LANG", "eng").strip() or "eng"
OCR_TESSERACT_OEM = int(os.getenv("OCR_TESSERACT_OEM", "1"))
OCR_TESSERACT_PSM = int(os.getenv("OCR_TESSERACT_PSM", "3"))
OCR_WORKER_TIMEOUT_SEC = float(os.getenv("OCR_WORKER_TIMEOUT_SEC", "60"))
OCR_WORKER_STARTUP_TIMEOUT_SEC = float(os.getenv("OCR_WORKER_STARTUP_TIMEOUT_SEC", "120"))
OCR_NATIVE_TEXT_MIN_CHARS = int(os.getenv("OCR_NATIVE_TEXT_MIN_CHARS", "1"))
OCR_MAX_IMAGE_SIDE = max(0, int(os.getenv("OCR_MAX_IMAGE_SIDE", "2200")))
OCR_DIAGRAM_CROP_MAX_SIDE = max(0, int(os.getenv("OCR_DIAGRAM_CROP_MAX_SIDE", "800")))
PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "220"))
PDF_RENDER_MODE = os.getenv("PDF_RENDER_MODE", "needed").strip().lower()
if PDF_RENDER_MODE not in {"all", "needed", "none"}:
    PDF_RENDER_MODE = "needed"
ENABLE_PPTX_SLIDE_RENDER = _env_bool("ENABLE_PPTX_SLIDE_RENDER", True)
PPTX_SLIDE_RENDER_POLICY = os.getenv("PPTX_SLIDE_RENDER_POLICY", "auto").strip().lower()
if PPTX_SLIDE_RENDER_POLICY not in {"always", "auto", "never"}:
    PPTX_SLIDE_RENDER_POLICY = "auto"
PPTX_MIN_TEXT_CHARS_FOR_SKIP_RENDER = max(0, int(os.getenv("PPTX_MIN_TEXT_CHARS_FOR_SKIP_RENDER", "160")))
ENABLE_PPTX_RELATIONSHIP_GRAPH = _env_bool("ENABLE_PPTX_RELATIONSHIP_GRAPH", True)
PPTX_GRAPH_MAX_EDGES = int(os.getenv("PPTX_GRAPH_MAX_EDGES", "120"))
ENABLE_DIAGRAM_PIPELINE = _env_bool("ENABLE_DIAGRAM_PIPELINE", True)
DIAGRAM_PARSE_POLICY = os.getenv("DIAGRAM_PARSE_POLICY", "auto").strip().lower()
if DIAGRAM_PARSE_POLICY not in {"auto", "always", "never"}:
    DIAGRAM_PARSE_POLICY = "auto"
DIAGRAM_MAX_IMAGES_PER_DOC = max(0, int(os.getenv("DIAGRAM_MAX_IMAGES_PER_DOC", "12")))
ENABLE_YOLO_DIAGRAM_DETECTOR = _env_bool("ENABLE_YOLO_DIAGRAM_DETECTOR", True)
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo26n.pt").strip() or "yolo26n.pt"
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "auto").strip() or "auto"
YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.25"))
YOLO_IOU_THRESHOLD = float(os.getenv("YOLO_IOU_THRESHOLD", "0.45"))
YOLO_IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "960"))
DIAGRAM_MIN_NODE_AREA = int(os.getenv("DIAGRAM_MIN_NODE_AREA", "1600"))
DIAGRAM_NODE_OCR_TOP_K = max(0, int(os.getenv("DIAGRAM_NODE_OCR_TOP_K", "12")))
DIAGRAM_NODE_OCR_MIN_AREA = max(0, int(os.getenv("DIAGRAM_NODE_OCR_MIN_AREA", "2400")))
DIAGRAM_NODE_OCR_MIN_SCORE = float(os.getenv("DIAGRAM_NODE_OCR_MIN_SCORE", "0.30"))
DIAGRAM_MIN_EDGE_LENGTH = int(os.getenv("DIAGRAM_MIN_EDGE_LENGTH", "48"))
DIAGRAM_MAX_NODES = int(os.getenv("DIAGRAM_MAX_NODES", "80"))
DIAGRAM_MAX_EDGES = int(os.getenv("DIAGRAM_MAX_EDGES", "180"))
DIAGRAM_MAX_NODE_CHUNKS = int(os.getenv("DIAGRAM_MAX_NODE_CHUNKS", "24"))
DIAGRAM_MAX_EDGE_CHUNKS = int(os.getenv("DIAGRAM_MAX_EDGE_CHUNKS", "24"))
DIAGRAM_AUX_CHUNK_LIMIT_PER_PAGE = max(1, int(os.getenv("DIAGRAM_AUX_CHUNK_LIMIT_PER_PAGE", "16")))
DIAGRAM_AUX_CHUNK_GLOBAL_LIMIT = max(1, int(os.getenv("DIAGRAM_AUX_CHUNK_GLOBAL_LIMIT", "96")))
DIAGRAM_TOP_K_FLOOR = int(os.getenv("DIAGRAM_TOP_K_FLOOR", "14"))
DIAGRAM_PER_DOC_LIMIT_FLOOR = int(os.getenv("DIAGRAM_PER_DOC_LIMIT_FLOOR", "4"))
DIAGRAM_MIN_EVIDENCE_SOURCE_TYPES = int(os.getenv("DIAGRAM_MIN_EVIDENCE_SOURCE_TYPES", "3"))
DIAGRAM_MIN_GRAPH_CHUNKS = int(os.getenv("DIAGRAM_MIN_GRAPH_CHUNKS", "1"))
DIAGRAM_MIN_OCR_CHUNKS = int(os.getenv("DIAGRAM_MIN_OCR_CHUNKS", "2"))
DIAGRAM_MIN_NODE_CHUNKS = int(os.getenv("DIAGRAM_MIN_NODE_CHUNKS", "2"))
DIAGRAM_MIN_EDGE_CHUNKS = int(os.getenv("DIAGRAM_MIN_EDGE_CHUNKS", "2"))
DIAGRAM_MIN_SLIDE_GRAPH_CHUNKS = int(os.getenv("DIAGRAM_MIN_SLIDE_GRAPH_CHUNKS", "0"))
DIAGRAM_MIXED_EVIDENCE_LIMIT = int(os.getenv("DIAGRAM_MIXED_EVIDENCE_LIMIT", "72"))
CHUNK_DEDUP_ENABLED = _env_bool("CHUNK_DEDUP_ENABLED", True)
CHUNK_DEDUP_MIN_CHARS = max(8, int(os.getenv("CHUNK_DEDUP_MIN_CHARS", "40")))


def ensure_dirs() -> None:
    for path in (UPLOAD_DIR, PROCESSED_DIR, INDEX_DIR, DB_DIR):
        path.mkdir(parents=True, exist_ok=True)
