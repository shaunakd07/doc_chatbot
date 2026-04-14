from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import shutil
import sys
import time
from pathlib import Path
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
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
    from backend.ingestion.pipeline import create_document_record, safe_path_for_upload
    from backend.models.openai_chat import OpenAIChatModel
    from backend.services.chat_service import ChatService
    from backend.services.document_classifier import normalize_doc_type
    from backend.services.ingestion_queue import IngestionQueue
    from backend.services.openai_router_service import OpenAIRouterService
    from backend.services.out_of_place_detection import detectOutOfPlaceDocuments
    from backend.services.review_workflow_service import apply_review_decision, build_folder_review_summary
    from backend.services.router_service import RouterService
    from backend.services.retrieval_service import RetrievalService
else:
    from . import config, storage
    from .index.embeddings import Embedder
    from .index.reranker import Reranker
    from .index.sparse_index import SparseIndex
    from .index.vector_index import VectorIndex
    from .ingestion.pipeline import create_document_record, safe_path_for_upload
    from .models.openai_chat import OpenAIChatModel
    from .services.chat_service import ChatService
    from .services.document_classifier import normalize_doc_type
    from .services.ingestion_queue import IngestionQueue
    from .services.openai_router_service import OpenAIRouterService
    from .services.out_of_place_detection import detectOutOfPlaceDocuments
    from .services.review_workflow_service import apply_review_decision, build_folder_review_summary
    from .services.router_service import RouterService
    from .services.retrieval_service import RetrievalService


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"
ICONS_DIR = BASE_DIR / "icons"


class ChatRequest(BaseModel):
    message: str
    doc_ids: list[str] | None = None
    top_k: int | None = None
    include_document_summaries: bool = True
    conversation_id: str | None = None


class DriveRequest(BaseModel):
    url: str
    expected_type: str | None = None
    folder_id: str | None = None
    review_threshold: float | None = None
    whitelist_types: list[str] | None = None


class FolderReviewDecisionRequest(BaseModel):
    doc_id: str
    decision: str
    note: str | None = None
    whitelist_predicted_type: bool = False


def _validate_doc_scope(doc_ids: list[str]) -> tuple[list[str] | None, str | None]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in doc_ids:
        doc_id = str(raw).strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        cleaned.append(doc_id)

    if not cleaned:
        return None, "Select at least one ready document before sending a chat request."

    documents = {str(doc.get("id")): doc for doc in storage.list_documents()}
    unknown = [doc_id for doc_id in cleaned if doc_id not in documents]
    if unknown:
        return None, f"Unknown document id(s): {', '.join(unknown[:5])}"

    not_ready = [
        doc_id
        for doc_id in cleaned
        if str((documents.get(doc_id) or {}).get("status", "")).lower() != "ready"
    ]
    if not_ready:
        return None, f"Document(s) are not ready: {', '.join(not_ready[:5])}"

    return cleaned, None


def _request_uploader_metadata(request: Request) -> dict:
    headers = request.headers
    raw_email = str(headers.get("x-user-email", "") or "").strip()
    raw_role = str(headers.get("x-user-role", "") or "").strip()
    raw_type = str(headers.get("x-user-type", "") or "").strip().lower()
    collaborator_type = str(headers.get("x-collaborator-type", "") or "").strip().lower()
    if collaborator_type not in {"internal", "external"}:
        collaborator_type = ""
    if raw_type not in {"human", "service"}:
        raw_type = ""
    return {
        "id": str(headers.get("x-user-id", "") or "").strip(),
        "name": str(headers.get("x-user-name", "") or "").strip(),
        "email": raw_email,
        "role": raw_role,
        "type": raw_type,
        "collaborator_type": collaborator_type,
    }


def _request_reviewer_metadata(request: Request) -> dict[str, str]:
    payload = _request_uploader_metadata(request)
    reviewer_type = str(payload.get("type") or "").strip().lower()
    if reviewer_type not in {"human", "service"}:
        reviewer_type = "human"
    return {
        "id": str(payload.get("id") or "").strip(),
        "name": str(payload.get("name") or "").strip(),
        "email": str(payload.get("email") or "").strip(),
        "role": str(payload.get("role") or "").strip(),
        "type": reviewer_type,
    }


def _folder_review_metadata_from_headers(request: Request) -> dict[str, object]:
    folder_id = str(request.headers.get("x-folder-id", "") or "").strip()
    expected = normalize_doc_type(
        str(request.headers.get("x-expected-doc-type", "") or ""),
        use_alias_map=False,
    )
    threshold_raw = str(request.headers.get("x-doc-review-threshold", "") or "").strip()
    whitelist_raw = str(request.headers.get("x-doc-review-whitelist-types", "") or "").strip()

    metadata: dict[str, object] = {}
    if folder_id:
        metadata["folder_id"] = folder_id
    if expected:
        metadata["expected_doc_type"] = expected
    if threshold_raw:
        try:
            threshold = float(threshold_raw)
        except Exception:
            threshold = None
        if threshold is not None:
            metadata["doc_review_threshold"] = max(0.01, min(0.99, threshold))

    whitelist_types: list[str] = []
    if whitelist_raw:
        for item in whitelist_raw.split(","):
            normalized = normalize_doc_type(item, use_alias_map=False)
            if normalized:
                whitelist_types.append(normalized)
    if whitelist_types:
        metadata["doc_review_whitelist_types"] = sorted(set(whitelist_types))

    if metadata:
        metadata["folder_review_enabled"] = bool(expected)
    return metadata


def _initialize_app_state(app: FastAPI) -> None:
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
    app.state.ingestion_queue = IngestionQueue(max_workers=config.INGEST_MAX_WORKERS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _initialize_app_state(app)
    try:
        yield
    finally:
        ingestion_queue = getattr(app.state, "ingestion_queue", None)
        if ingestion_queue is not None:
            ingestion_queue.shutdown(wait=False)


app = FastAPI(title="Doc Chatbot", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/app.js")
def app_js() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "app.js")


@app.get("/styles.css")
def styles() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "styles.css")


@app.get("/icons/{filename}")
def icon_file(filename: str):
    icon_root = ICONS_DIR.resolve()
    icon_path = (icon_root / filename).resolve()
    try:
        icon_path.relative_to(icon_root)
    except ValueError:
        return JSONResponse({"error": "not found"}, status_code=404)
    if not icon_path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(icon_path)


@app.get("/api/health")
def health() -> JSONResponse:
    chat_service = getattr(app.state, "chat_service", None)
    ingestion_queue = getattr(app.state, "ingestion_queue", None)
    return JSONResponse(
        {
            "status": "ok",
            "vlm_enabled": True,
            "model_provider": config.MODEL_PROVIDER,
            "embed_provider": config.EMBED_PROVIDER,
            "router_provider": config.ROUTER_PROVIDER,
            "db_backend": config.DB_BACKEND,
            "router_enabled": True,
            "router_model_id": config.OPENAI_ROUTER_MODEL,
            "openai_chat_model": config.OPENAI_CHAT_MODEL,
            "openai_embed_model": config.OPENAI_EMBED_MODEL,
            "openai_router_model": config.OPENAI_ROUTER_MODEL,
            "retrieval_mode": config.RETRIEVAL_MODE,
            "reranker_enabled": config.ENABLE_RERANKER,
            "reranker_model_id": config.RERANK_MODEL_ID if config.ENABLE_RERANKER else None,
            "diagram_pipeline_enabled": config.ENABLE_DIAGRAM_PIPELINE,
            "yolo_diagram_enabled": config.ENABLE_YOLO_DIAGRAM_DETECTOR,
            "ingest_max_workers": config.INGEST_MAX_WORKERS,
            "ingest_queue_pending": ingestion_queue.pending_count() if ingestion_queue is not None else 0,
            "doc_type_review_enabled": config.DOC_TYPE_REVIEW_ENABLED,
            "doc_type_review_threshold": config.DOC_TYPE_REVIEW_CONFIDENCE_THRESHOLD,
            "doc_type_classifier_provider": config.DOC_TYPE_CLASSIFIER_PROVIDER,
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


@app.get("/api/documents/{doc_id}/diagram-graphs")
def get_document_diagram_graphs(doc_id: str) -> JSONResponse:
    doc = storage.get_document(doc_id)
    if not doc:
        return JSONResponse({"error": "not found"}, status_code=404)
    graphs = storage.list_diagram_graphs(doc_ids=[doc_id])
    return JSONResponse({"doc_id": doc_id, "graphs": graphs})


@app.get("/api/folders/{folder_id}/review-flags")
def list_folder_review_flags(folder_id: str, request: Request) -> JSONResponse:
    tenant_id = str(request.headers.get("x-tenant-id", "") or "").strip() or config.DEFAULT_TENANT_ID
    summary = build_folder_review_summary(folder_id=folder_id, tenant_id=tenant_id)
    return JSONResponse(
        {
            "tenant_id": tenant_id,
            "folder_id": folder_id,
            "review_status": str(summary.get("review_status") or "clear"),
            "flags": summary.get("flags") if isinstance(summary.get("flags"), list) else [],
            "flag_count": int(summary.get("open_flag_count") or 0),
            "counts": summary.get("counts") if isinstance(summary.get("counts"), dict) else {},
        }
    )


@app.get("/api/folders/{folder_id}/review-summary")
def get_folder_review_summary(folder_id: str, request: Request) -> JSONResponse:
    tenant_id = str(request.headers.get("x-tenant-id", "") or "").strip() or config.DEFAULT_TENANT_ID
    summary = build_folder_review_summary(folder_id=folder_id, tenant_id=tenant_id)
    return JSONResponse(summary)


@app.post("/api/folders/{folder_id}/review-decisions")
def submit_folder_review_decision(
    folder_id: str,
    payload: FolderReviewDecisionRequest,
    request: Request,
) -> JSONResponse:
    tenant_id = str(request.headers.get("x-tenant-id", "") or "").strip() or config.DEFAULT_TENANT_ID
    reviewer = _request_reviewer_metadata(request)
    try:
        result = apply_review_decision(
            folder_id=folder_id,
            doc_id=payload.doc_id,
            tenant_id=tenant_id,
            decision=payload.decision,
            reviewer=reviewer,
            note=payload.note,
            whitelist_predicted_type=bool(payload.whitelist_predicted_type),
        )
    except LookupError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    except PermissionError as exc:
        return JSONResponse({"error": str(exc)}, status_code=403)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse(result)


@app.get("/api/folders/{folder_id}/out-of-place")
def detect_folder_out_of_place(
    folder_id: str,
    expected_type: str,
    request: Request,
    threshold: float | None = None,
) -> JSONResponse:
    tenant_id = str(request.headers.get("x-tenant-id", "") or "").strip() or config.DEFAULT_TENANT_ID
    normalized_expected = normalize_doc_type(expected_type, use_alias_map=False)
    if not normalized_expected:
        return JSONResponse({"error": "expected_type is required"}, status_code=400)
    flags = detectOutOfPlaceDocuments(
        folderId=folder_id,
        expectedType=normalized_expected,
        tenantId=tenant_id,
        threshold=threshold,
    )
    return JSONResponse(
        {
            "tenant_id": tenant_id,
            "folder_id": folder_id,
            "expected_type": normalized_expected,
            "flags": flags,
            "flag_count": len(flags),
            "review_status": "needs_review" if flags else "clear",
        }
    )


@app.post("/api/documents")
def upload_document(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    uploader = _request_uploader_metadata(request)
    tenant_id = str(request.headers.get("x-tenant-id", "") or "").strip() or config.DEFAULT_TENANT_ID
    doc_id = create_document_record(
        file.filename,
        source_system="upload",
        source_uri=file.filename,
        tenant_id=tenant_id,
        uploader=uploader,
    )

    folder_review_metadata = _folder_review_metadata_from_headers(request)
    if folder_review_metadata:
        storage.update_document(doc_id, metadata=folder_review_metadata, merge_metadata=True)

    try:
        destination = safe_path_for_upload(file.filename, doc_id)
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        storage.delete_document(doc_id)
        return JSONResponse({"error": f"Failed to save upload: {exc}"}, status_code=400)

    try:
        app.state.ingestion_queue.submit(
            destination,
            doc_id,
            app.state.embedder,
            app.state.vector_index,
            app.state.sparse_index,
            app.state.vlm,
        )
    except Exception as exc:
        storage.update_document(
            doc_id,
            status="failed",
            metadata={
                "ingest_progress": 0,
                "ingest_stage": "failed",
                "ingest_message": "Ingestion queue submission failed",
                "ingest_error": str(exc)[:800],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            merge_metadata=True,
        )
        return JSONResponse({"error": f"Failed to queue ingestion: {exc}"}, status_code=500)

    response_payload: dict[str, object] = {"doc_id": doc_id, "status": "queued"}
    if folder_review_metadata:
        response_payload["folder_id"] = folder_review_metadata.get("folder_id")
        response_payload["expected_type"] = folder_review_metadata.get("expected_doc_type")
        response_payload["review_mode"] = "soft_warning"
    return JSONResponse(response_payload)


@app.post("/api/documents/drive")
def import_drive_folder(request: DriveRequest, http_request: Request) -> JSONResponse:
    import gdown
    import tempfile

    url = request.url.strip()
    if not url:
        return JSONResponse({"error": "No URL provided"}, status_code=400)
    uploader = _request_uploader_metadata(http_request)
    tenant_id = str(http_request.headers.get("x-tenant-id", "") or "").strip() or config.DEFAULT_TENANT_ID
    expected_type = normalize_doc_type(str(request.expected_type or ""), use_alias_map=False)
    folder_id = str(request.folder_id or "").strip()
    if expected_type and not folder_id:
        folder_id = f"drive-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"

    whitelist_types: list[str] = []
    if isinstance(request.whitelist_types, list):
        for item in request.whitelist_types:
            normalized = normalize_doc_type(str(item or ""), use_alias_map=False)
            if normalized:
                whitelist_types.append(normalized)
    whitelist_types = sorted(set(whitelist_types))

    temp_dir = Path(tempfile.mkdtemp(prefix="doc_chat_drive_"))
    try:
        if "folder" in url or "folderview" in url:
            gdown.download_folder(url, output=str(temp_dir), quiet=True, use_cookies=False)
        else:
            gdown.download(url, output=str(temp_dir), quiet=True, fuzzy=True)

        count = 0
        for p in temp_dir.rglob("*"):
            if p.is_file() and not p.name.startswith("."):
                relative = str(p.relative_to(temp_dir)).replace("\\", "/")
                doc_id = create_document_record(
                    relative or p.name,
                    source_system="google_drive",
                    source_uri=f"{url}#{relative}" if relative else url,
                    tenant_id=tenant_id,
                    uploader=uploader,
                )

                metadata_update: dict[str, object] = {}
                if folder_id:
                    metadata_update["folder_id"] = folder_id
                if expected_type:
                    metadata_update["expected_doc_type"] = expected_type
                    metadata_update["folder_review_enabled"] = True
                if request.review_threshold is not None:
                    metadata_update["doc_review_threshold"] = max(0.01, min(0.99, float(request.review_threshold)))
                if whitelist_types:
                    metadata_update["doc_review_whitelist_types"] = whitelist_types
                if relative:
                    metadata_update["source_folder_path"] = str(Path(relative).parent).replace("\\", "/")
                if metadata_update:
                    storage.update_document(doc_id, metadata=metadata_update, merge_metadata=True)

                destination = safe_path_for_upload(p.name, doc_id)
                shutil.copy2(p, destination)
                app.state.ingestion_queue.submit(
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
        return JSONResponse(
            {
                "status": "queued",
                "count": count,
                "folder_id": folder_id or None,
                "expected_type": expected_type or None,
                "whitelist_types": whitelist_types,
                "review_mode": "soft_warning" if expected_type else "disabled",
            }
        )
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

    app.state.vector_index.load()
    app.state.sparse_index.load()

    return JSONResponse({"status": "deleted_all", "deleted_documents": deleted_count})


@app.post("/api/chat")
def chat(request: ChatRequest) -> JSONResponse:
    started_at = time.perf_counter()
    top_k = request.top_k or config.TOP_K
    conversation_id = str(request.conversation_id or "").strip() or str(uuid.uuid4())
    scoped_doc_ids = request.doc_ids
    if request.doc_ids is not None:
        scoped_doc_ids, scope_error = _validate_doc_scope(request.doc_ids)
        if scope_error:
            return JSONResponse({"error": scope_error}, status_code=400)
    logger.info(
        "api.chat.request %s",
        json.dumps(
            {
                "conversation_id": conversation_id,
                "message_preview": " ".join(str(request.message or "").split())[:180],
                "requested_doc_count": len(request.doc_ids or []),
                "requested_doc_ids": list(request.doc_ids or [])[:8],
                "scoped_doc_count": len(scoped_doc_ids or []),
                "scoped_doc_ids": list(scoped_doc_ids or [])[:8],
                "top_k": top_k,
                "include_document_summaries": bool(request.include_document_summaries),
            },
            sort_keys=True,
            default=str,
        ),
    )
    try:
        response = app.state.chat_service.answer(
            request.message,
            doc_ids=scoped_doc_ids,
            top_k=top_k,
            include_document_summaries=bool(request.include_document_summaries),
            conversation_id=conversation_id,
        )
        if "conversation_id" not in response:
            response["conversation_id"] = conversation_id
    except Exception as exc:
        logger.exception("Chat request failed")
        return JSONResponse(
            {
                "error": "Chat request failed.",
                "detail": str(exc),
            },
            status_code=500,
        )
    route = response.get("route") if isinstance(response.get("route"), dict) else {}
    logger.info(
        "api.chat.response %s",
        json.dumps(
            {
                "conversation_id": conversation_id,
                "duration_ms": round((time.perf_counter() - started_at) * 1000.0, 2),
                "intent": str(response.get("intent") or ""),
                "answer_chars": len(str(response.get("answer") or "")),
                "source_count": len(response.get("sources") or []),
                "task_type": str(route.get("task_type") or ""),
                "needs_cross_doc": bool(route.get("needs_cross_doc", False)),
                "needs_numeric_extraction": bool(route.get("needs_numeric_extraction", False)),
                "needs_image_reasoning": bool(route.get("needs_image_reasoning", False)),
                "retrieval_plan": route.get("retrieval_plan") if isinstance(route.get("retrieval_plan"), dict) else {},
                "exact_lookup": route.get("exact_lookup") if isinstance(route.get("exact_lookup"), dict) else {},
            },
            sort_keys=True,
            default=str,
        ),
    )
    return JSONResponse(response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
