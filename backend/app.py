from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backend import config, storage
    from backend.index.embeddings import Embedder
    from backend.index.reranker import Reranker
    from backend.index.sparse_index import SparseIndex
    from backend.index.vector_index import VectorIndex
    from backend.ingestion.pipeline import create_document_record, ingest_file, safe_path_for_upload
    from backend.models.openai_chat import OpenAIChatModel
    from backend.services.chat_service import ChatService
    from backend.services.openai_router_service import OpenAIRouterService
    from backend.services.router_service import RouterService
    from backend.services.retrieval_service import RetrievalService
else:
    from . import config, storage
    from .index.embeddings import Embedder
    from .index.reranker import Reranker
    from .index.sparse_index import SparseIndex
    from .index.vector_index import VectorIndex
    from .ingestion.pipeline import create_document_record, ingest_file, safe_path_for_upload
    from .models.openai_chat import OpenAIChatModel
    from .services.chat_service import ChatService
    from .services.openai_router_service import OpenAIRouterService
    from .services.router_service import RouterService
    from .services.retrieval_service import RetrievalService


load_dotenv()
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Doc Chatbot", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"


class ChatRequest(BaseModel):
    message: str
    doc_ids: list[str] | None = None
    top_k: int | None = None


class DriveRequest(BaseModel):
    url: str


@app.on_event("startup")
def startup() -> None:
    config.ensure_dirs()
    storage.init_db()

    embed_model = config.OPENAI_EMBED_MODEL
    embedder = Embedder(
        embed_model,
        device="cuda",
        provider=config.EMBED_PROVIDER,
        openai_api_key=config.OPENAI_API_KEY,
    )
    vector_index = VectorIndex()
    vector_index.load()
    sparse_index = SparseIndex()
    sparse_index.load()
    reranker = Reranker(
        model_name=config.RERANK_MODEL_ID,
        device=config.RERANK_DEVICE,
        enabled=config.ENABLE_RERANKER,
    )

    model = OpenAIChatModel(
        model_id=config.OPENAI_CHAT_MODEL,
        api_key=config.OPENAI_API_KEY,
    )
    router = OpenAIRouterService(
        config.OPENAI_ROUTER_MODEL,
        api_key=config.OPENAI_API_KEY,
        max_new_tokens=256,
    )

    retrieval_service = RetrievalService(
        embedder,
        vector_index,
        sparse_index=sparse_index,
        reranker=reranker,
        default_mode=config.RETRIEVAL_MODE,
        rerank_top_n=config.RERANK_TOP_N,
    )
    chat_service = ChatService(
        retrieval_service,
        model,
        True,
        router=router,
        max_context_chars=config.MAX_CONTEXT_CHARS,
    )

    app.state.embedder = embedder
    app.state.vector_index = vector_index
    app.state.sparse_index = sparse_index
    app.state.chat_service = chat_service
    app.state.vlm = model
    app.state.router = router


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/app.js")
def app_js() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "app.js")


@app.get("/styles.css")
def styles() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "styles.css")


@app.get("/api/health")
def health() -> JSONResponse:
    chat_service = getattr(app.state, "chat_service", None)
    return JSONResponse(
        {
            "status": "ok",
            "vlm_enabled": True,
            "model_provider": config.MODEL_PROVIDER,
            "embed_provider": config.EMBED_PROVIDER,
            "router_provider": config.ROUTER_PROVIDER,
            "router_enabled": True,
            "router_model_id": config.OPENAI_ROUTER_MODEL,
            "openai_chat_model": config.OPENAI_CHAT_MODEL,
            "openai_embed_model": config.OPENAI_EMBED_MODEL,
            "openai_router_model": config.OPENAI_ROUTER_MODEL,
            "retrieval_mode": config.RETRIEVAL_MODE,
            "reranker_enabled": config.ENABLE_RERANKER,
            "reranker_model_id": config.RERANK_MODEL_ID if config.ENABLE_RERANKER else None,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "last_generation_error": getattr(chat_service, "last_generation_error", None),
            "last_route": getattr(chat_service, "last_route", None),
        }
    )


@app.get("/api/documents")
def list_documents() -> JSONResponse:
    docs = storage.list_documents()
    for doc in docs:
        if doc.get("metadata"):
            try:
                doc["metadata"] = json.loads(doc["metadata"])
            except Exception:
                pass
    return JSONResponse({"documents": docs})


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str) -> JSONResponse:
    doc = storage.get_document(doc_id)
    if not doc:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(doc)


@app.post("/api/documents")
def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> JSONResponse:
    doc_id = create_document_record(file.filename)
    destination = safe_path_for_upload(file.filename, doc_id)
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(
        ingest_file,
        destination,
        doc_id,
        app.state.embedder,
        app.state.vector_index,
        app.state.sparse_index,
        app.state.vlm,
    )

    return JSONResponse({"doc_id": doc_id, "status": "queued"})


@app.post("/api/documents/drive")
def import_drive_folder(background_tasks: BackgroundTasks, request: DriveRequest) -> JSONResponse:
    import gdown
    import tempfile
    
    url = request.url.strip()
    if not url:
        return JSONResponse({"error": "No URL provided"}, status_code=400)

    temp_dir = Path(tempfile.mkdtemp(prefix="doc_chat_drive_"))
    try:
        if "folder" in url or "folderview" in url:
            gdown.download_folder(url, output=str(temp_dir), quiet=True, use_cookies=False)
        else:
            gdown.download(url, output=str(temp_dir), quiet=True, fuzzy=True)

        count = 0
        for p in temp_dir.rglob("*"):
            if p.is_file() and not p.name.startswith("."):
                doc_id = create_document_record(p.name)
                destination = safe_path_for_upload(p.name, doc_id)
                shutil.copy2(p, destination)
                background_tasks.add_task(
                    ingest_file,
                    destination,
                    doc_id,
                    app.state.embedder,
                    app.state.vector_index,
                    app.state.sparse_index,
                    app.state.vlm,
                )
                count += 1
                
        if count == 0:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return JSONResponse({"error": "No files found or downloaded from the link."}, status_code=400)
            
        shutil.rmtree(temp_dir, ignore_errors=True)
        return JSONResponse({"status": "queued", "count": count})
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return JSONResponse({"error": str(e)}, status_code=400)


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> JSONResponse:
    deleted = storage.delete_document(doc_id)
    if not deleted:
        return JSONResponse({"error": "not found"}, status_code=404)

    upload_dir = config.UPLOAD_DIR / doc_id
    processed_dir = config.PROCESSED_DIR / doc_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)
    if processed_dir.exists():
        shutil.rmtree(processed_dir, ignore_errors=True)

    # Keep in-memory retrieval index in sync with DB state.
    app.state.vector_index.load()
    app.state.sparse_index.load()

    return JSONResponse({"doc_id": doc_id, "status": "deleted"})


@app.delete("/api/documents")
def delete_all_documents() -> JSONResponse:
    deleted_count = storage.delete_all_documents()

    for root in (config.UPLOAD_DIR, config.PROCESSED_DIR):
        if not root.exists():
            continue
        for entry in root.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)

    # Keep in-memory retrieval index in sync with DB state.
    app.state.vector_index.load()
    app.state.sparse_index.load()

    return JSONResponse({"status": "deleted_all", "deleted_documents": deleted_count})


@app.post("/api/chat")
def chat(request: ChatRequest) -> JSONResponse:
    top_k = request.top_k or config.TOP_K
    response = app.state.chat_service.answer(request.message, doc_ids=request.doc_ids, top_k=top_k)
    return JSONResponse(response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
