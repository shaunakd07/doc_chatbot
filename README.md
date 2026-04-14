# Doc Chatbot

FastAPI-based document ingestion and chat application with a backend-served vanilla JS frontend.

## Installation

### Prerequisites

- Python 3.12
- An OpenAI API key
- PostgreSQL with `pgvector` if using the default database setup
- Optional: Tesseract for OCR
- Optional: LibreOffice for PPTX slide rendering fallback

### Setup

```powershell
python -m venv .venv312
.\.venv312\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env`.

`.env.example` defaults to PostgreSQL. For a simpler local setup, switch to SQLite:

```env
DB_BACKEND=sqlite
DATABASE_URL=
SQLITE_DB_PATH=./data/db/app.db
```

### Run

```powershell
.\.venv312\Scripts\python.exe -m backend.app
```

Open `http://127.0.0.1:8000`.

## Architecture

- `backend/`: FastAPI API layer, config, storage, ingestion pipeline, retrieval, and chat services
- `frontend/`: thin vanilla JS client served by the backend
- `data/`: uploaded files, processed artifacts, and local SQLite data
- `tests/`: pytest coverage for backend behavior and API workflows
- Storage supports PostgreSQL with `pgvector` or SQLite
- Ingestion runs in-process through an async queue
- Retrieval supports dense, sparse, and hybrid modes with optional reranking
- Chat is OpenAI-backed and grounded on retrieved document sources
