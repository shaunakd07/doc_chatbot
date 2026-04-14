# doc_chatbot Workflow and Architecture Guide

This guide describes the current implementation in this repository.

## 1. What the app is

`doc_chatbot` is a document ingestion and retrieval application with source-grounded chat on top of uploaded files.

High-level flow:

1. A file upload or Drive import creates a document record.
2. The ingestion queue extracts text, tables, images, OCR output, and optional graph structure.
3. The pipeline chunks the extracted content and stores embeddings and metadata.
4. A chat request is routed, retrieved against the selected documents, and answered from the retrieved evidence.
5. The API returns an answer plus sources and route metadata.

The backend is the system of record. The frontend is a thin client that calls the API and renders results.

## 2. Main runtime components

- API app: `backend/app.py`
- Config and env parsing: `backend/config.py`
- Persistence layer: `backend/storage.py`
- Ingestion queue: `backend/services/ingestion_queue.py`
- Ingestion pipeline: `backend/ingestion/pipeline.py`
- File extractors: `backend/ingestion/extractors.py`
- OCR controller and worker: `backend/ingestion/ocr.py`, `backend/ingestion/ocr_worker.py`
- Diagram parser: `backend/ingestion/diagram_parser.py`
- Dense index and sparse index: `backend/index/vector_index.py`, `backend/index/sparse_index.py`
- Retrieval orchestration: `backend/services/retrieval_service.py`
- Optional Haystack search backend: `backend/services/haystack_backend.py`
- Chat orchestration: `backend/services/chat_service.py`
- OpenAI router: `backend/services/openai_router_service.py`
- Document classification and review: `backend/services/document_classifier.py`, `backend/services/out_of_place_detection.py`, `backend/services/review_workflow_service.py`
- Frontend: `frontend/index.html`, `frontend/app.js`, `frontend/styles.css`

## 3. Storage and persisted state

Configured directories from `backend/config.py`:

- `data/uploads/`: raw source files stored under a per-document folder
- `data/processed/`: derived artifacts such as page images and extracted images
- `data/db/`: local SQLite path when SQLite is used
- `data/index/`: local index directory

Database backend:

- `DB_BACKEND=postgres` is the current recommended mode and the default in `.env.example`
- `DB_BACKEND=sqlite` remains supported for simpler local usage

`backend/storage.py` currently manages these tables:

- `documents`
- `chunks`
- `embeddings`
- `diagram_graphs`
- `chat_sessions`
- `chat_messages`

Important detail:

- In PostgreSQL mode, embeddings are stored in `pgvector`
- In SQLite mode, embeddings are stored as float32 blobs and dense search is handled in process

## 4. API surface

Primary endpoints in `backend/app.py`:

- `GET /api/health`
- `GET /api/documents`
- `GET /api/documents/{doc_id}`
- `GET /api/documents/{doc_id}/diagram-graphs`
- `POST /api/documents`
- `POST /api/documents/drive`
- `DELETE /api/documents/{doc_id}`
- `DELETE /api/documents`
- `POST /api/chat`
- `GET /api/folders/{folder_id}/review-flags`
- `GET /api/folders/{folder_id}/review-summary`
- `POST /api/folders/{folder_id}/review-decisions`
- `GET /api/folders/{folder_id}/out-of-place`

The frontend calls these endpoints directly and does not maintain its own persistent state.

## 5. Upload and ingestion lifecycle

### 5.1 Document creation

`POST /api/documents`:

1. Creates a `doc_id`
2. Inserts a `documents` row with `status=queued`
3. Saves the file under `data/uploads/<doc_id>/...`
4. Submits ingestion to `IngestionQueue`

`POST /api/documents/drive`:

1. Downloads a file or folder through `gdown`
2. Creates one document record per discovered file
3. Copies each file into the upload area
4. Queues each file for ingestion

### 5.2 Status transitions

The normal state flow is:

- `queued`
- `processing`
- `ready`

On pipeline failure the document is marked:

- `failed`

The ingestion pipeline writes progress metadata such as:

- `ingest_progress`
- `ingest_stage`
- `ingest_message`
- `ingest_updated_at`
- `ingest_error`

The frontend polls `GET /api/documents` and uses this metadata to render progress.

## 6. Ingestion pipeline details

The main entrypoint is `ingest_file()` in `backend/ingestion/pipeline.py`.

### 6.1 Extraction dispatch

Current extractor dispatch is extension-based with a best-effort generic fallback.

Supported categories include:

- PDF
- PPTX
- DOCX
- XLSX, XLSM, XLTX, XLTM, XLS
- text-like formats such as TXT, MD, CSV, JSON, YAML, LOG, INI, CFG, TOML
- HTML, XML, RTF
- ODT, ODS, ODP
- image formats such as PNG, JPG, TIFF, BMP, GIF, WEBP

Fallback behavior in `extract_generic()`:

1. Try format-specific helpers for HTML, XML, RTF, spreadsheet, and ODF files
2. Detect OOXML/ODF payloads inside zip containers
3. Attempt PDF extraction
4. Fall back to plain text read

### 6.2 OCR behavior

OCR is mediated by `backend/ingestion/ocr.py` and executed in the worker subprocess in `backend/ingestion/ocr_worker.py`.

Why the worker exists:

- isolate OCR runtime dependencies
- reduce conflicts with the rest of the Python process
- allow timeout/retry handling per OCR request

Current OCR options:

- `OCR_ENGINE=tesseract` by default
- optional `OCR_ENGINE=paddle`

Common OCR metadata written on image-derived content includes:

- `ocr_status`
- `ocr_engine`
- `ocr_confidence`
- `ocr_line_count`
- `ocr_error`

### 6.3 Diagram and graph extraction

When image-like content appears diagram-like, `parse_image_diagram()` can extract:

- graph summary chunks
- node chunks
- edge chunks
- persisted graph JSON records in `diagram_graphs`

Detection and graph-building use a mix of:

- Ultralytics YOLO when enabled
- OpenCV contour and edge logic as fallback
- OCR over node crops
- NetworkX metrics when available

### 6.4 Chunking and embedding

Chunking is character-window based in `backend/ingestion/text_chunker.py`.

The pipeline:

1. Builds content blocks
2. Splits text into overlapping chunks
3. Persists chunk rows
4. Computes embeddings in batch
5. Persists embeddings
6. Updates dense and sparse retrieval structures

The pipeline also writes derived metadata such as:

- `auto_tags`
- `doc_type`
- `doc_type_confidence`
- `doc_type_scores`

## 7. File-type behavior summary

### PDF

- native text extraction with PyMuPDF
- page rendering to images
- OCR and diagram handling where needed

### PPTX

- shape text extraction
- table extraction
- embedded image extraction
- slide relationship graph generation
- optional full-slide rendering through LibreOffice for OCR fallback

### DOCX

- paragraph extraction
- table extraction
- embedded image extraction
- OOXML fallback if `python-docx` is unavailable

### Spreadsheets

- table-style serialization of rows
- sheet names preserved in extracted text
- embedded image extraction for supported workbook formats

### Images

- stored as image-derived content
- OCR and diagram parsing run from the ingestion pipeline

## 8. Retrieval and chat flow

The main answer path is in `ChatService.answer()`.

### 8.1 Conversation memory

If `CHAT_MEMORY_ENABLED=true`, the backend:

- loads recent chat state from `chat_sessions` and `chat_messages`
- can rewrite the current user question using recent conversation context
- persists the new turn after a successful answer

### 8.2 Route planning

`OpenAIRouterService` returns a structured route with fields such as:

- `task_type`
- `needs_cross_doc`
- `needs_numeric_extraction`
- `needs_image_reasoning`
- `retrieval_plan`
- `analysis_plan`
- `expected_answer_type`
- `confidence`

The router is OpenAI-backed in the current implementation.

### 8.3 Retrieval modes

Current retrieval paths include:

- dense semantic retrieval
- sparse BM25 retrieval
- hybrid reciprocal-rank fusion
- balanced per-document retrieval for cross-document coverage

Optional enrichments:

- cross-encoder reranking
- metadata-semantic adaptation
- Haystack-backed in-memory retrieval flow when enabled

### 8.4 Metadata-aware queries

The chat service now has a dedicated metadata-query path for questions such as:

- counts
- latest or earliest uploads
- date-filtered document questions
- author/editor/uploader-role questions
- version-change comparisons

That path is implemented in `backend/services/chat_service.py`; it is not just a prompt trick.

### 8.5 Context assembly and answer generation

After retrieval:

1. chunks are deduplicated and ordered
2. source tags are added
3. optional document summaries are added
4. image evidence can be attached for multimodal generation
5. `OpenAIChatModel` sends the request to the configured chat model

Returned API payloads include:

- `answer`
- `sources`
- `intent`
- `route`
- `conversation_id`

## 9. Folder review and document classification workflow

The backend supports advisory review of folder uploads where an expected document type is known.

Current provider options in `backend/services/document_classifier.py`:

- `heuristic`
- `semantic_openai`
- `azure_document_intelligence`

Current default from `.env.example`:

- `DOC_TYPE_CLASSIFIER_PROVIDER=heuristic`

Flow:

1. upload or Drive import stores `folder_id` and `expected_doc_type`
2. ingestion completes and writes base document-type metadata
3. `IngestionQueue` triggers `detectOutOfPlaceDocuments()`
4. review metadata is written back onto the affected documents
5. reviewers can query summary state or apply decisions through the folder review endpoints

Decision types currently supported:

- `dismiss`
- `accept`
- `whitelist`
- `reopen`

Important current limitation:

- the backend supports single-file review metadata through request headers
- the frontend currently renders review inputs but does not send those headers during upload
- Drive import is the most complete built-in API path for review metadata today

## 10. Current frontend behavior

The browser UI exposes three tabs:

- `Chat`
- `Documents`
- `Logs`

Current capabilities:

- upload files
- upload folders
- import from Google Drive
- poll ingestion progress
- select ready documents for chat scope
- delete one or all documents
- inspect client-side activity logs

The frontend is intentionally lightweight and most business logic remains in the backend.

## 11. Important configuration areas

High-impact environment variables:

- OpenAI models:
  - `OPENAI_CHAT_MODEL`
  - `OPENAI_EMBED_MODEL`
  - `OPENAI_ROUTER_MODEL`
- retrieval:
  - `RETRIEVAL_MODE`
  - `ENABLE_RERANKER`
  - `RERANK_TOP_N`
  - `HYBRID_METADATA_SEMANTIC`
  - `ENABLE_HAYSTACK_RETRIEVAL`
- chat memory:
  - `CHAT_MEMORY_ENABLED`
  - `CHAT_MEMORY_RECENT_TURNS`
  - `CHAT_MEMORY_MAX_MESSAGES`
- OCR:
  - `ENABLE_OCR`
  - `OCR_ENGINE`
  - `OCR_WORKER_TIMEOUT_SEC`
- diagrams:
  - `ENABLE_DIAGRAM_PIPELINE`
  - `ENABLE_YOLO_DIAGRAM_DETECTOR`
  - `YOLO_*`
  - `DIAGRAM_*`
- review workflow:
  - `DOC_TYPE_REVIEW_ENABLED`
  - `DOC_TYPE_REVIEW_CONFIDENCE_THRESHOLD`
  - `DOC_TYPE_REVIEW_MIN_SCORE_RATIO`
  - `DOC_TYPE_CLASSIFIER_PROVIDER`
- persistence:
  - `DB_BACKEND`
  - `DATABASE_URL`
  - `SQLITE_DB_PATH`

## 12. Known implementation caveats

- The health payload still reports some legacy names such as `vlm_enabled`, but the current stack is OpenAI-backed rather than the older local-VLM setup.
- `data/index/` still exists in config and on disk, but the authoritative persisted state is the database plus processed artifacts.
- Folder review UI is only partially wired on the frontend.
- PPTX full-slide rendering depends on LibreOffice being available on the machine.
- OCR and diagram-heavy ingestion quality remains environment-sensitive and should be validated with the local test corpus after major config changes.
