# Infrastructure and Runtime Notes

This document summarizes the current runtime shape of the repository and the operational dependencies required to run it.

## 1. Runtime topology

The application is currently a single FastAPI service process with a few in-process and subprocess helpers:

- FastAPI app in `backend/app.py`
- frontend assets served by the same app from `frontend/`
- ingestion handled by a thread-pool queue in `backend/services/ingestion_queue.py`
- OCR executed through a separate worker subprocess

There is no separate frontend server, job queue service, or external vector database service in the repo today.

## 2. Persistence and state

Filesystem state:

- raw uploads: `data/uploads/<doc_id>/...`
- processed artifacts: `data/processed/<doc_id>/...`
- local SQLite file when SQLite is used: `data/db/app.db`
- auxiliary local index directory: `data/index/`

Database tables created in `backend/storage.py`:

- `documents`
- `chunks`
- `embeddings`
- `diagram_graphs`
- `chat_sessions`
- `chat_messages`

Backend modes:

- PostgreSQL mode:
  - recommended for regular development and concurrent ingestion
  - uses `pgvector`
  - can create an IVF Flat index for embeddings
- SQLite mode:
  - simpler local fallback
  - embeddings stored as blobs
  - dense search handled in process

## 3. External services

Current external service dependency:

- OpenAI API

OpenAI is used for:

- embeddings
- route planning
- answer generation
- optional semantic document classification

Relevant modules:

- `backend/index/embeddings.py`
- `backend/services/openai_router_service.py`
- `backend/models/openai_chat.py`
- `backend/services/document_classifier.py`

Optional external service dependency:

- Azure Document Intelligence for document classification when `DOC_TYPE_CLASSIFIER_PROVIDER=azure`

Google Drive import uses:

- `gdown`

That is a client-side library dependency, not a separately managed service inside this repo.

## 4. Local runtime dependencies

Python packages from `requirements.txt` power the current stack:

- FastAPI and Uvicorn
- OpenAI SDK
- PyMuPDF, python-pptx, python-docx, openpyxl, xlrd, BeautifulSoup, striprtf
- NumPy
- sentence-transformers and PyTorch
- OpenCV and NetworkX
- Ultralytics
- Haystack
- psycopg
- pytesseract

System-level dependencies that materially affect behavior:

- Tesseract
  - default OCR engine
  - required unless OCR is disabled or the engine is switched to Paddle
- LibreOffice / `soffice`
  - required for PPTX full-slide rendering fallback
- PostgreSQL with `vector` extension
  - required only when `DB_BACKEND=postgres`

Optional acceleration:

- CUDA-enabled PyTorch for GPU acceleration

## 5. Startup sequence

App initialization in `backend/app.py` currently performs:

1. ensure configured directories exist
2. initialize database schema
3. create embedder
4. load dense and sparse indexes
5. create reranker
6. create chat model and router
7. create retrieval service and chat service
8. create the ingestion queue

The app serves:

- `/` -> `frontend/index.html`
- `/app.js`
- `/styles.css`
- `/icons/{filename}`

## 6. Operational health

The canonical runtime check is:

- `GET /api/health`

Current health response includes:

- general status
- model ids
- DB backend
- retrieval mode
- reranker status
- diagram pipeline status
- ingestion queue depth
- review workflow config
- CUDA visibility
- last route and last generation error

Use this endpoint after boot, after config changes, and during long ingest runs.

## 7. Ingestion infrastructure details

Ingestion is not processed inline with the upload request.

Current model:

- upload endpoint saves the file and returns quickly
- `IngestionQueue` runs `ingest_file()` in a thread pool
- dense and sparse index mutation is guarded by an internal lock
- post-ingestion folder-review detection runs in the queue callback when folder metadata is present

Operational implication:

- long OCR or diagram workloads do not block the request thread
- ingestion throughput is controlled primarily by `INGEST_MAX_WORKERS`

## 8. Chat and retrieval infrastructure details

The chat path is request/response and synchronous from the API client's perspective.

Runtime stages:

1. optional conversation-state load
2. question rewrite using chat memory
3. route planning
4. metadata query handling when applicable
5. retrieval and reranking
6. context assembly
7. multimodal OpenAI generation
8. conversation persistence

Current retrieval building blocks:

- dense search
- sparse BM25 search
- hybrid fusion
- optional reranking
- optional metadata-semantic adaptation
- optional Haystack-backed search path

## 9. Folder review infrastructure

The backend supports folder-level mismatch review.

Implemented pieces:

- metadata capture on upload/import
- post-ingestion detection
- document-type classifier provider abstraction
- summary and decision endpoints
- audit trail persisted inside `documents.metadata`

Current operational caveat:

- the frontend exposes review fields visually, but `frontend/app.js` does not yet send the single-file review headers to the backend

## 10. Default configuration posture

`.env.example` currently defaults to:

- `DB_BACKEND=postgres`
- `OPENAI_CHAT_MODEL=gpt-4o-mini`
- `OPENAI_EMBED_MODEL=text-embedding-3-small`
- `OPENAI_ROUTER_MODEL=gpt-4o-mini`
- `DOC_TYPE_CLASSIFIER_PROVIDER=heuristic`
- `RETRIEVAL_MODE=hybrid`
- `ENABLE_RERANKER=true`
- `ENABLE_OCR=true`
- `OCR_ENGINE=tesseract`
- `ENABLE_DIAGRAM_PIPELINE=true`
- `ENABLE_YOLO_DIAGRAM_DETECTOR=true`

If you want the lightest local setup:

- switch to SQLite
- keep OpenAI configured
- install Tesseract
- accept that PPTX full-slide rendering will be skipped unless LibreOffice is installed

## 11. Current non-goals / not present in repo

The repo does not currently include:

- a separate CI/CD pipeline definition
- a dedicated worker service outside the FastAPI process
- a managed message queue
- FAISS, Milvus, Pinecone, or another standalone vector DB
- the older local InternVL/Qwen model runtime described in stale docs

Those were part of earlier exploration, not the current default implementation.
