# doc_chatbot Workflow and Architecture Guide

This guide explains how `doc_chatbot` works end-to-end in plain technical language. It is based on the current code in this repository.

## 1) What this app is

`doc_chatbot` is a Retrieval-Augmented Generation (RAG) system for your uploaded files.

High level flow:

1. You upload documents in the UI.
2. The backend extracts text, tables, images, and diagram structure.
3. The backend chunks and indexes the extracted content.
4. When you ask a question, the backend retrieves the most relevant chunks.
5. The model answers using only that retrieved context.
6. The UI shows the answer plus sources.

Important: retrieval happens before generation. The model does not search your files by itself.

## 2) Main components

- API server: FastAPI (`backend/app.py`)
- Ingestion pipeline: `backend/ingestion/pipeline.py`
- Extractors per file type: `backend/ingestion/extractors.py`
- OCR subsystem: `backend/ingestion/ocr.py` + `backend/ingestion/ocr_worker.py`
- Diagram parsing: `backend/ingestion/diagram_parser.py`
- Storage: SQLite wrapper (`backend/storage.py`)
- Dense vector search: `backend/index/vector_index.py`
- Sparse lexical search (BM25): `backend/index/sparse_index.py`
- Retrieval orchestration: `backend/services/retrieval_service.py`
- Chat orchestration: `backend/services/chat_service.py`
- Prompt templates: `backend/models/prompts.py`
- Model client (OpenAI): `backend/models/openai_chat.py`
- Frontend: `frontend/index.html`, `frontend/app.js`

## 3) Data directories and persistence

Configured in `backend/config.py`:

- `data/uploads`: raw uploaded files, stored per document id
- `data/processed`: processed artifacts (for example saved page/slide images)
- `data/db/app.db`: SQLite database (legacy/local fallback)
- `data/index`: reserved index directory (current core index state is rebuilt from DB at startup)

Database backend is selected by env:

- `DB_BACKEND=postgres` (recommended for concurrent ingestion)
- `DB_BACKEND=sqlite` (legacy/single-writer fallback)

### Tables used

`backend/storage.py` creates and uses these tables in both SQLite and Postgres:

- `documents`: document metadata and ingestion status
- `chunks`: chunked retrieval units (text and derived chunk types)
- `embeddings`: one dense vector per chunk (stored as float32 BLOB)
- `diagram_graphs`: structured graph JSON extracted from diagrams

## 4) Upload and ingestion lifecycle

### Upload

- Endpoint: `POST /api/documents`
- A `doc_id` is created immediately.
- File is saved under `data/uploads/<doc_id>/...`.
- Ingestion runs as a background task.

### Document states and progress

Document status transitions are managed in `backend/ingestion/pipeline.py`:

- `queued` -> `processing` -> `ready`
- `failed` if pipeline errors

Progress metadata is written during ingestion:

- `ingest_progress` (0-100)
- `ingest_stage`
- `ingest_message`
- `ingest_updated_at`

The frontend polls `/api/documents` and renders progress bars under each doc.

## 5) How each file type is handled

All file extraction starts in `ingest_file()` and dispatches to extractors by extension.

### PDF (`.pdf`)

Handled by `extract_pdf()`:

- Native text is extracted with PyMuPDF (`fitz`), page by page.
- Each page is also rendered to an image (`PDF_RENDER_DPI`).
- OCR fallback flag is set if native text is too small (`OCR_NATIVE_TEXT_MIN_CHARS`).

Result: PDFs can contribute both native text chunks and image-derived OCR/diagram chunks.

### PPTX (`.pptx`)

Handled by `extract_pptx()`:

- Reads slide shapes with `python-pptx`.
- Extracts text from text boxes.
- Extracts tables.
- Extracts embedded images.
- Builds slide-level graph structure from reading order and connectors.
- Builds document-level slide relationship graph.
- Optionally renders full slides via LibreOffice (`soffice`) to PDF->images when enabled.

Result: PPTX contributes text/table/image plus `slide_graph` graph chunks and optionally full-slide OCR/diagram evidence.

### DOCX (`.docx`)

Handled by `extract_docx()`:

- Primary path: `python-docx`.
- Fallback path: direct OOXML parsing from zip/XML.
- Extracts paragraphs, tables, and embedded images.

### Spreadsheets (`.xlsx`, `.xlsm`, `.xltx`, `.xltm`, `.xls`)

- `.xlsx` family: `openpyxl`
- `.xls`: `xlrd`
- Rows are converted into table text blocks.
- Embedded images in `.xlsx` are also extracted when available.

### Images (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.gif`, `.webp`)

- Loaded as image blocks.
- OCR and diagram parsing run in the ingestion pipeline.

### Text-like files (`.txt`, `.md`, `.csv`, `.tsv`, `.json`, `.yaml`, `.yml`, `.log`, `.ini`, `.cfg`, `.toml`)

- Read directly as text.

### Generic fallback

`extract_generic()` tries:

- HTML/XML stripping
- RTF conversion
- ODF parsing
- Zip-based Office format detection
- PDF parse attempt
- Plain text read as last resort

## 6) OCR handling (including GPU/CPU behavior)

OCR is handled in two layers:

- Parent OCR controller: `backend/ingestion/ocr.py`
- OCR worker subprocess: `backend/ingestion/ocr_worker.py`

### Why a separate worker process

The OCR worker isolates PaddleOCR runtime from the main API process. This avoids many DLL/CUDA conflicts with other libraries.

### OCR flow

1. Ingestion sends an image to OCR controller.
2. Controller encodes image to base64 and sends JSON request to worker over stdin.
3. Worker runs PaddleOCR and returns text/confidence JSON over stdout.
4. Controller merges OCR text into chunk content if available.

### OCR controls

Environment variables include:

- `ENABLE_PADDLE_OCR`
- `PADDLE_OCR_USE_GPU`
- `PADDLE_OCR_LANG`
- `PADDLE_OCR_MIN_CONFIDENCE`
- `PADDLE_OCR_MAX_RETRIES`
- `OCR_WORKER_TIMEOUT_SEC`
- `OCR_WORKER_STARTUP_TIMEOUT_SEC`

### OCR output fields tracked per image block

- `ocr_status`
- `ocr_engine`
- `ocr_confidence`
- `ocr_line_count`
- `ocr_error` (if any)

## 7) Diagram and graph handling

`parse_image_diagram()` in `backend/ingestion/diagram_parser.py` performs structure extraction from images.

### Node detection

Two strategies:

- YOLO detector (`ultralytics`) when enabled and available
- OpenCV contour-based fallback

### Text in nodes

For each detected node region, OCR is run on the crop to get labels.

### Edge detection

- OpenCV Canny + Hough line detection
- Lines are attached to nearest nodes
- Direction hints inferred (left-to-right or top-to-bottom)

### Graph metrics

If NetworkX is available:

- connected components
- largest component
- density

### Persisted outputs

- Structured graph JSON into `diagram_graphs` table
- Retrieval chunks for:
  - `diagram_graph`
  - `diagram_node`
  - `diagram_edge`

These chunks are later retrievable like normal text chunks.

## 8) Chunking and tokenization

There are multiple "tokenization" layers in this app.

### Layer A: ingestion chunking (character-based)

`chunk_text()` (`backend/ingestion/text_chunker.py`) splits text by character count with overlap.

- Default in pipeline: about 900 chars with 120-char overlap
- It tries to break on spaces near chunk boundaries

This is not model-token based chunking. It is a lightweight character-window chunker.

### Layer B: sparse retrieval tokenization

`SparseIndex` tokenizes with regex:

- pattern: `[A-Za-z0-9_]{2,}`
- lowercased tokens
- BM25 scoring

This tokenization is for lexical retrieval only.

### Layer C: model tokenizer (internal)

When generating embeddings or answers, tokenization is handled by the model provider internally:

- Embeddings: OpenAI embedding endpoint (or local SentenceTransformer if configured)
- Chat model: OpenAI chat model tokenizer internally

The app does not manually tokenize into model tokens.

## 9) How the vector database is built (and what type it is)

### What it is

This project uses a hybrid storage/index design:

- Primary persisted store: PostgreSQL (recommended) or SQLite fallback
- Dense vectors persisted in `embeddings`:
  - Postgres mode: `pgvector` column (`vector(PGVECTOR_DIM)`)
  - SQLite mode: float32 BLOB
- Dense retrieval:
  - Postgres mode: SQL vector similarity search with pgvector
  - SQLite mode: in-memory dense matrix search
- Sparse BM25 index rebuilt from chunks at startup (`SparseIndex`)

So this is not FAISS or a managed vector DB service. In Postgres mode it uses `pgvector`.

### Build process during ingestion

For each chunk:

1. Chunk is inserted into `chunks` table.
2. Embedding vector is computed in batch.
3. Embedding is written to `embeddings` table.
4. Dense index/search path updates:
   - Postgres mode: query-time pgvector search (no in-memory preload needed)
   - SQLite mode: in-memory matrix append
5. In-memory `SparseIndex` adds BM25 terms.

### Dense similarity math

- Embeddings are normalized.
- Postgres mode uses pgvector cosine distance (`<=>`) converted to score.
- SQLite mode uses dot product of normalized vectors.

## 10) Retrieval pipeline for a user question

Main path: `ChatService.answer()` + `RetrievalService`

### Step 1: Route planning

`OpenAIRouterService` classifies question into route info:

- `task_type` (`qa`, `compare`, etc.)
- `needs_cross_doc`
- `needs_image_reasoning`
- retrieval plan:
  - `strategy` (`semantic`, `balanced`, `image_first`)
  - `top_k`
  - `per_doc_limit`

If router fails or confidence is low, heuristic fallback is used.

### Step 2: Candidate retrieval

`RetrievalService` supports:

- semantic: dense only
- sparse: BM25 only
- hybrid: RRF fusion of dense+sparse

For hybrid mode, it oversamples candidates, fuses scores with Reciprocal Rank Fusion, hydrates chunks from SQLite, then optional reranking.

### Step 3: Optional reranking

`Reranker` (cross-encoder) can reorder top chunks.
If unavailable, a lexical overlap fallback score is used.

### Step 4: Diagram-aware evidence mixing

For diagram/relationship/image intents, `ChatService` enforces mixed evidence and ordering so context includes types like:

- `diagram_graph`
- `ocr`
- `diagram_node`
- `slide_graph`
- `diagram_edge` (lower priority)

### Step 5: Context assembly

`_build_context_blocks()` builds source-tagged blocks up to `MAX_CONTEXT_CHARS`.

Source tags include doc label, doc_id, page, chunk id, and source type.

## 11) Prompt handling and generation

### Prompt templates

Prompt text is built in `backend/models/prompts.py`:

- `build_prompt()` for standard QA
- `build_compare_prompt()` for comparison/timeline style tasks

Prompt rules enforce:

- Use provided context only
- Include source citations
- Short paragraph formatting
- For open-ended questions: provide best evidence-based explanation and clearly state missing details
- Use refusal sentence only if no relevant evidence exists

### Important clarification

The prompt does not make retrieval happen. Retrieval already happened in backend code before prompt creation.

The prompt controls generation behavior on top of the retrieved context.

### Generation call

`OpenAIChatModel.generate_text()` sends:

- Prompt text
- Optional up to 5 context images (base64)

This allows multimodal response generation for visual questions.

## 12) How context scope is controlled

The frontend lets users select:

- all ready docs
- specific ready docs only

Selected `doc_ids` are sent with chat request.

Backend validates selected docs:

- doc exists
- doc is `ready`

Only validated selected docs are considered during retrieval.

## 13) Why answers can still fail even with OCR/graphs

Common causes:

- OCR text quality is noisy for complex diagrams
- Retrieval may return structural chunks that lack clear explanatory text
- Context budget may truncate useful lower-ranked chunks
- Strict source-grounding rules may force conservative answers

## 14) Practical tuning knobs

High-impact variables:

- Retrieval and context:
  - `TOP_K`
  - `MAX_CONTEXT_CHARS`
  - `RETRIEVAL_MODE`
  - `RERANK_TOP_N`
- OCR:
  - `PADDLE_OCR_USE_GPU`
  - `PADDLE_OCR_MIN_CONFIDENCE`
  - `PDF_RENDER_DPI`
- Diagram parsing:
  - `ENABLE_YOLO_DIAGRAM_DETECTOR`
  - `YOLO_CONF_THRESHOLD`
  - `YOLO_IMAGE_SIZE`
  - `DIAGRAM_*` thresholds and minimum evidence settings

After changing ingestion-related settings, re-ingest documents so new chunks/metadata are built.

## 15) End-to-end request sequence (concise)

### Upload path

1. UI calls `POST /api/documents`.
2. Backend saves file and queues ingestion.
3. Ingestion extracts blocks, OCR, diagrams, graphs.
4. Text is chunked and embedded.
5. Chunks, embeddings, and graph records are stored.
6. Doc becomes `ready`.

### Chat path

1. UI sends `POST /api/chat` with `message`, selected `doc_ids`, and summary flag.
2. Router decides intent/retrieval plan.
3. Retriever fetches and reranks chunks.
4. Diagram-aware mixing/ordering adjusts context.
5. Prompt is assembled with source tags.
6. Model generates answer from context (+ optional images).
7. Backend returns answer, route, and source list.

## 16) Current architecture summary in one sentence

`doc_chatbot` is a Postgres/SQLite-backed, hybrid-retrieval RAG system with OCR and diagram-graph enrichment that builds evidence chunks at ingestion time, then uses router-guided retrieval plus strict source-grounded prompting at answer time.
