# AGENTS.md

## Codex operating rules for this repo

- Work as a **single implementation agent**
- Prefer **small, reviewable diffs**
- Do **not** refactor unrelated code
- Do **not** rename files, functions, classes, endpoints, or APIs unless required
- Do **not** introduce new dependencies unless clearly necessary
- Before changing behavior, read the touched module and nearby tests first
- Preserve existing architecture unless the task explicitly requests redesign

---

## Ground truth for this repo

- This repo is the **current active implementation**, not an older PoC
- Prefer the current codebase and README over stale internal notes
- Active stack:
  - FastAPI backend
  - Backend-served vanilla JS frontend
  - OpenAI-based chat, routing, embeddings
  - SQLite or PostgreSQL + pgvector
  - Hybrid retrieval with optional reranking
  - OCR, slide rendering, diagram extraction
  - Folder-level review APIs

- Frontend tabs:
  - Chat
  - Documents
  - Logs

---

## Main architecture to preserve

- Backend is the **source of truth**
- Frontend is a **thin client**
- Preserve separation of concerns:
  - API layer
  - config/env
  - storage
  - ingestion queue + pipeline
  - retrieval service
  - chat service

- Do not move backend logic into frontend

---

## Runtime assumptions

- Single FastAPI process
- Frontend served by backend
- Ingestion uses an **async queue**
- External dependencies may include:
  - Tesseract
  - LibreOffice
  - Postgres + pgvector

- No separate worker service unless explicitly added

---

## Persistence and data rules

- Must support:
  - `DB_BACKEND=postgres`
  - `DB_BACKEND=sqlite`

- `.env.example` defaults to PostgreSQL
- SQLite must remain functional

- Do not break:
  - documents
  - chunks
  - embeddings
  - diagram_graphs
  - chat_sessions
  - chat_messages

- Postgres:
  - preserve pgvector behavior

- SQLite:
  - embeddings handled in-process

- Avoid changes affecting:
  - embedding dimensions
  - ingestion metadata
  - chunk structure

---

## API contract rules

- Do not change endpoints unless explicitly requested
- Preserve:
  - health
  - document CRUD
  - Drive import
  - chat
  - folder review
  - diagram inspection

- If API changes:
  - update backend
  - update frontend
  - update tests
  - update docs

---

## Ingestion pipeline rules

- Preserve background ingestion + progress tracking
- Maintain lifecycle states (queued → processing → ready/failed)
- Do not weaken:
  - OCR
  - slide parsing
  - diagram extraction
  - embedding persistence

- Supported formats include:
  - PDFs
  - Office files
  - spreadsheets
  - images
  - HTML/XML/RTF
  - ODF

- Handle unknown formats via fallback logic

---

## Retrieval and chat rules

- Preserve:
  - dense retrieval
  - sparse retrieval
  - hybrid mode
  - reranking (optional)
  - document scoping
  - chat memory
  - source-grounded answers

- Do not degrade retrieval quality or grounding

---

## OCR and diagram rules

- Default OCR: Tesseract
- Optional: PaddleOCR
- PPTX rendering supported (LibreOffice optional)
- Diagram parsing produces graph outputs

- Do not degrade:
  - OCR accuracy
  - diagram extraction
  - graph persistence

- If modified, note environment dependencies

---

## Document classification and review workflow

- Preserve heuristic classification
- Optional:
  - semantic classification
  - Azure Document Intelligence

- Folder review APIs exist in backend
- Frontend does **not fully wire review inputs**

- Prioritize backend correctness over UI assumptions

---

## Frontend rules

- Keep frontend simple
- Backend-served only
- Do not introduce frameworks unless asked

- Preserve flows:
  - upload
  - Drive import
  - status inspection
  - document scoping
  - deletion
  - chat
  - logs

---

## Environment and config rules

- Config-driven system
- Do not hardcode values

- Config may include:
  - model selection
  - retrieval mode
  - reranker toggle
  - OCR settings
  - diagram pipeline flags
  - DB backend

- If config changes:
  - update code
  - update docs

---

## Test rules

- Always run relevant tests
- Start with smallest scope
- Expand only as needed

- Do not claim success without execution
- Do not rewrite tests to match broken behavior

- Minimum validation:
  - targeted pytest
  - health check
  - relevant API or workflow test

---

## Test command rules

- Use local venv where applicable
- Prefer targeted pytest first
- Use harness for workflow-level validation
- Expand test scope incrementally

---

## Evidence and debugging rules

- Use logs and execution output as ground truth
- Debug using real failures, not speculation
- Preserve or improve observability
- Always verify health endpoint after major changes

---

## Definition of done

- Minimal, coherent diff
- Tests executed and passing
- Appropriate regression coverage completed
- No unrelated file changes
- Blockers clearly stated
- Docs/tests updated if behavior changed

## Task continuity

- Before starting any task:
  - Read docs/agent_handoff.md if it exists
  - Treat it as the source of truth for:
    - current progress
    - completed steps
    - next step to implement

- Do not rely on prior chat context
- Always re-ground yourself in:
  - AGENTS.md
  - docs/agent_handoff.md
  - current codebase