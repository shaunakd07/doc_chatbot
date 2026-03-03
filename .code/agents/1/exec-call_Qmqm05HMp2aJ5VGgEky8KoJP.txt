from __future__ import annotations

import json
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List

from .extractors import (
    extract_docx,
    extract_generic,
    extract_image,
    extract_pdf,
    extract_pptx,
    extract_text,
    extract_xls,
    extract_xlsx,
)
from .diagram_parser import parse_image_diagram
from .doc_tags import build_document_auto_tags
from .ocr import extract_text_from_image
from .text_chunker import chunk_text
from .. import config, storage

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".log",
    ".ini",
    ".cfg",
    ".toml",
}
SPREADSHEET_EXTENSIONS = {".xlsx", ".xlsm", ".xltx", ".xltm", ".xls"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"}
DIAGRAM_HINT_MARKERS = {
    "diagram",
    "flow",
    "workflow",
    "pipeline",
    "architecture",
    "component",
    "process",
    "decision",
    "sequence",
    "connector",
    "arrow",
    "->",
}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _merge_text_without_duplication(base_text: str, extra_text: str) -> str:
    base = str(base_text or "").strip()
    extra = str(extra_text or "").strip()
    if not extra:
        return base
    if not base:
        return extra
    base_norm = _normalize_text(base)
    extra_norm = _normalize_text(extra)
    if not extra_norm:
        return base
    if extra_norm in base_norm:
        return base
    return f"{base}\n\n{extra}".strip()


def _is_likely_diagram_payload(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return any(marker in normalized for marker in DIAGRAM_HINT_MARKERS)


def _dedupe_chunks(chunks: List[dict]) -> List[dict]:
    if not chunks or not config.CHUNK_DEDUP_ENABLED:
        return chunks
    min_chars = max(8, int(config.CHUNK_DEDUP_MIN_CHARS))
    seen_keys: set[tuple[str, str]] = set()
    deduped: List[dict] = []
    for chunk in chunks:
        content = str(chunk.get("content") or "").strip()
        if not content:
            continue
        normalized = _normalize_text(content)
        if len(normalized) >= min_chars:
            source = str(chunk.get("source_type") or "").strip().lower()
            dedupe_key = (source, normalized[:320])
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
        deduped.append(chunk)
    return deduped

def ingest_file(
    file_path: Path,
    doc_id: str,
    embedder,
    vector_index,
    sparse_index=None,
    vlm=None,
    index_lock=None,
) -> None:
    progress_pct = 0
    last_progress_emit_at = 0.0
    last_progress_value = -1
    last_progress_stage = ""
    last_progress_status = ""

    def _set_progress(
        value: int,
        stage: str,
        message: str,
        *,
        status: str | None = None,
        num_pages: int | None = None,
        extra_metadata: Dict[str, Any] | None = None,
        force: bool = False,
    ) -> None:
        nonlocal progress_pct, last_progress_emit_at, last_progress_value, last_progress_stage, last_progress_status
        next_value = max(0, min(100, int(value)))
        next_stage = str(stage or "").strip() or "processing"
        next_status = str(status or "").strip()
        now = time.monotonic()
        value_delta = abs(next_value - int(last_progress_value))
        min_delta = max(1, int(config.INGEST_PROGRESS_MIN_DELTA))
        min_interval = max(0.0, float(config.INGEST_PROGRESS_MIN_INTERVAL_SEC))
        if next_status in {"ready", "failed"}:
            force = True
        should_emit = force
        if not should_emit and next_stage != last_progress_stage:
            should_emit = True
        if not should_emit and next_status != last_progress_status:
            should_emit = True
        if not should_emit and value_delta >= min_delta:
            should_emit = True
        if not should_emit and (now - last_progress_emit_at) >= min_interval:
            should_emit = True
        if not should_emit:
            return

        progress_pct = next_value
        payload: Dict[str, Any] = {
            "ingest_progress": progress_pct,
            "ingest_stage": next_stage,
            "ingest_message": str(message or "").strip(),
            "ingest_updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if isinstance(extra_metadata, dict):
            payload.update(extra_metadata)
        storage.update_document(
            doc_id,
            status=status,
            num_pages=num_pages,
            metadata=payload,
            merge_metadata=False,
        )
        last_progress_emit_at = now
        last_progress_value = next_value
        last_progress_stage = next_stage
        last_progress_status = next_status

    _set_progress(2, "processing", "Preparing document ingestion", status="processing")

    try:
        document = storage.get_document(doc_id) or {}
        doc_filename = str(document.get("filename") or file_path.name)
        processed_image_dir = config.PROCESSED_DIR / doc_id / "images"
        processed_image_dir.mkdir(parents=True, exist_ok=True)

        _set_progress(8, "processing", "Extracting document content")
        suffix = file_path.suffix.lower()
        blocks = []
        if suffix in {".pdf"}:
            blocks = extract_pdf(file_path)
        elif suffix in {".pptx"}:
            blocks = extract_pptx(file_path)
        elif suffix in {".docx"}:
            blocks = extract_docx(file_path)
        elif suffix in SPREADSHEET_EXTENSIONS:
            if suffix == ".xls":
                blocks = extract_xls(file_path)
            else:
                blocks = extract_xlsx(file_path)
        elif suffix in IMAGE_EXTENSIONS:
            blocks = extract_image(file_path)
        elif suffix in TEXT_EXTENSIONS:
            blocks = extract_text(file_path)
        else:
            blocks = extract_generic(file_path)
        total_blocks = len(blocks)
        _set_progress(14, "processing", f"Extracted {total_blocks} content block(s)")

        chunks: List[dict] = []
        page_image_paths: dict[int, str] = {}
        graph_records: List[Dict[str, Any]] = []
        diagram_aux_chunks: List[dict] = []
        image_counter = 0
        diagram_parse_attempts = 0
        diagram_aux_seen: set[tuple[int, str, str]] = set()
        diagram_aux_per_page_source: dict[tuple[int, str], int] = defaultdict(int)
        aux_chunk_global_limit = max(1, int(config.DIAGRAM_AUX_CHUNK_GLOBAL_LIMIT))
        aux_chunk_page_limit = max(1, int(config.DIAGRAM_AUX_CHUNK_LIMIT_PER_PAGE))

        def _should_parse_diagram(metadata: Dict[str, Any], text_value: str, source_kind: str) -> bool:
            policy = str(config.DIAGRAM_PARSE_POLICY).strip().lower()
            max_images = max(0, int(config.DIAGRAM_MAX_IMAGES_PER_DOC))
            if max_images and diagram_parse_attempts >= max_images:
                return False
            if policy == "never":
                return False
            if policy == "always":
                return True

            image_kind = str(metadata.get("image_kind") or "").strip().lower()
            if image_kind == "embedded_picture":
                return True

            native_text_chars = int(metadata.get("native_text_chars") or 0)
            if native_text_chars <= max(8, int(config.OCR_NATIVE_TEXT_MIN_CHARS)):
                return True

            if str(source_kind).strip().lower() == "ocr":
                return _is_likely_diagram_payload(text_value)

            return _is_likely_diagram_payload(text_value)

        def _append_aux_chunks(
            content: str,
            source: str,
            page: int,
            base_metadata: Dict[str, Any],
            extra_metadata: Dict[str, Any] | None = None,
        ) -> None:
            payload = str(content or "").strip()
            if not payload:
                return
            merged_metadata = dict(base_metadata)
            if isinstance(extra_metadata, dict):
                merged_metadata.update(extra_metadata)
            for idx, piece in enumerate(chunk_text(payload, max_chars=900, overlap=120)):
                if len(diagram_aux_chunks) >= aux_chunk_global_limit:
                    break
                normalized_piece = _normalize_text(piece)
                if not normalized_piece:
                    continue
                page_num = int(page or 1)
                source_key = str(source or "").strip().lower() or "diagram"
                dedupe_key = (page_num, source_key, normalized_piece[:280])
                if dedupe_key in diagram_aux_seen:
                    continue
                bucket_key = (page_num, source_key)
                if diagram_aux_per_page_source[bucket_key] >= aux_chunk_page_limit:
                    continue
                diagram_aux_seen.add(dedupe_key)
                diagram_aux_per_page_source[bucket_key] += 1
                diagram_aux_chunks.append(
                    {
                        "id": str(uuid.uuid4()),
                        "doc_id": doc_id,
                        "page": page_num,
                        "chunk_index": idx,
                        "content": piece,
                        "source_type": source,
                        "metadata": merged_metadata,
                    }
                )

        progress_step = max(1, total_blocks // 20) if total_blocks else 1
        for block_index, block in enumerate(blocks):
            if block_index == 0 or (block_index + 1) % progress_step == 0 or (block_index + 1) == total_blocks:
                loop_progress = 15 + int(((block_index + 1) / max(1, total_blocks)) * 55)
                _set_progress(
                    loop_progress,
                    "processing",
                    f"Processing content block {block_index + 1}/{max(1, total_blocks)}",
                )

            metadata = {"doc_filename": doc_filename}
            block_metadata = block.get("metadata")
            if isinstance(block_metadata, dict):
                metadata.update(block_metadata)

            text = str(block.get("text", ""))
            source_type = str(block.get("type", "text"))
            graph_payload = block.get("graph")
            parser_version = str(block.get("parser_version") or "").strip()
            if isinstance(graph_payload, dict):
                page_num = int(block.get("page") or 1)
                graph_metadata = {
                    "doc_filename": doc_filename,
                    "graph_scope": metadata.get("graph_scope"),
                    "graph_kind": metadata.get("graph_kind"),
                    "slide_index": metadata.get("slide_index") or page_num,
                }
                graph_records.append(
                    {
                        "id": str(uuid.uuid4()),
                        "doc_id": doc_id,
                        "page": page_num,
                        "image_path": "",
                        "graph": graph_payload,
                        "parser_version": parser_version or "pptx-graph-v1",
                        "confidence": 1.0,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "metadata": graph_metadata,
                    }
                )

            if block.get("type") == "image":
                image = block.get("image")
                image_path = None
                if image is not None:
                    image_counter += 1
                    page_num = block.get("page") or 1
                    image_name = f"p{page_num}_img{image_counter}.png"
                    target = processed_image_dir / image_name
                    try:
                        image.save(target, format="PNG")
                        image_path = str(target)
                        metadata["image_path"] = image_path
                        is_slide_render = str(metadata.get("image_kind") or "").strip().lower() == "slide_render"
                        if is_slide_render or int(page_num) not in page_image_paths:
                            page_image_paths[int(page_num)] = image_path
                        should_run_ocr = bool(metadata.get("ocr_fallback_required", True))
                        if should_run_ocr:
                            ocr_result = extract_text_from_image(image)
                            metadata["ocr_status"] = ocr_result.get("status")
                            metadata["ocr_engine"] = ocr_result.get("engine")
                            ocr_error = str(ocr_result.get("error") or "").strip()
                            if ocr_error:
                                metadata["ocr_error"] = ocr_error
                            ocr_text = str(ocr_result.get("text") or "").strip()
                            if ocr_text:
                                metadata["ocr_confidence"] = ocr_result.get("avg_confidence", 0.0)
                                metadata["ocr_line_count"] = ocr_result.get("line_count", 0)
                                merged_text = _merge_text_without_duplication(text, ocr_text)
                                if merged_text != text:
                                    source_type = "ocr"
                                text = merged_text
                        if config.ENABLE_AI_INGEST_SUMMARIES and vlm is not None and hasattr(vlm, "answer_image_question"):
                            prompt = (
                                "Describe this page in high detail. Extract all textual content, describe any "
                                "tables row by row, and explain any flowcharts, diagrams or logic presented."
                            )
                            try:
                                desc = vlm.answer_image_question(image, prompt)
                                summary_text = str(desc or "").strip()
                                if summary_text:
                                    text = f"{text}\n\n[AI Visual Summary]\n{summary_text}".strip() if text.strip() else summary_text
                            except Exception as e:
                                print(f"[Ingestion] Failed to get vision summary for {image_name}: {e}")

                        if _should_parse_diagram(metadata, text, source_type):
                            diagram_parse_attempts += 1
                            diagram_result = parse_image_diagram(
                                image,
                                page=int(page_num),
                                image_path=image_path or "",
                            )
                            diagram_status = str(diagram_result.get("status") or "").strip()
                            if diagram_status:
                                metadata["diagram_status"] = diagram_status
                            diagram_error = str(diagram_result.get("error") or "").strip()
                            if diagram_error:
                                metadata["diagram_error"] = diagram_error

                            if diagram_status == "ok":
                                parser_version_value = str(diagram_result.get("parser_version") or "diagram-v1")
                                confidence_value = float(diagram_result.get("confidence") or 0.0)
                                graph_payload = diagram_result.get("graph")
                                if isinstance(graph_payload, dict) and graph_payload:
                                    graph_records.append(
                                        {
                                            "id": str(uuid.uuid4()),
                                            "doc_id": doc_id,
                                            "page": int(page_num),
                                            "image_path": image_path or "",
                                            "graph": graph_payload,
                                            "parser_version": parser_version_value,
                                            "confidence": confidence_value,
                                            "created_at": datetime.now(timezone.utc).isoformat(),
                                            "metadata": {
                                                "doc_filename": doc_filename,
                                                "graph_scope": "image",
                                                "graph_kind": "image_diagram_graph",
                                                "slide_index": int(page_num),
                                                "parser": parser_version_value,
                                            },
                                        }
                                    )
                                _append_aux_chunks(
                                    str(diagram_result.get("summary_text") or ""),
                                    "diagram_graph",
                                    int(page_num),
                                    metadata,
                                    {
                                        "diagram_parser_version": parser_version_value,
                                        "diagram_confidence": confidence_value,
                                    },
                                )
                                node_chunks = diagram_result.get("node_chunks")
                                if isinstance(node_chunks, list):
                                    for node_text in node_chunks[: max(0, int(config.DIAGRAM_MAX_NODE_CHUNKS))]:
                                        _append_aux_chunks(
                                            str(node_text or ""),
                                            "diagram_node",
                                            int(page_num),
                                            metadata,
                                            {
                                                "diagram_parser_version": parser_version_value,
                                                "diagram_confidence": confidence_value,
                                            },
                                        )
                                edge_chunks = diagram_result.get("edge_chunks")
                                if isinstance(edge_chunks, list):
                                    for edge_text in edge_chunks[: max(0, int(config.DIAGRAM_MAX_EDGE_CHUNKS))]:
                                        _append_aux_chunks(
                                            str(edge_text or ""),
                                            "diagram_edge",
                                            int(page_num),
                                            metadata,
                                            {
                                                "diagram_parser_version": parser_version_value,
                                                "diagram_confidence": confidence_value,
                                            },
                                        )
                        else:
                            metadata["diagram_status"] = "skipped"
                    except Exception as e:
                        print(f"Error saving image: {e}")
                if not text.strip():
                    page_num = block.get("page") or 1
                    native_text_chars = int(metadata.get("native_text_chars") or 0)
                    if native_text_chars <= 0:
                        text = f"[Image page {page_num}]"
                        metadata["image_only"] = True

            # Basic text embedding of whatever text/tables/summaries we generated.
            if text.strip():
                for idx, chunk in enumerate(chunk_text(text, max_chars=900, overlap=120)):
                    chunks.append(
                        {
                            "id": str(uuid.uuid4()),
                            "doc_id": doc_id,
                            "page": block.get("page"),
                            "chunk_index": idx,
                            "content": chunk,
                            "source_type": source_type,
                            "metadata": metadata,
                        }
                    )

        _set_progress(72, "processing", "Finalizing extracted chunks")
        if diagram_aux_chunks:
            chunks.extend(diagram_aux_chunks)
        chunks = _dedupe_chunks(chunks)

        tag_samples = [
            str(chunk.get("content") or "")
            for chunk in chunks
            if str(chunk.get("content") or "").strip()
        ][:56]
        auto_tags = build_document_auto_tags(doc_filename, tag_samples, limit=32)

        for chunk in chunks:
            page = chunk.get("page")
            try:
                page_num = int(page) if page is not None else 0
            except Exception:
                page_num = 0
            if page_num <= 0:
                continue
            chunk_metadata = chunk.get("metadata")
            if not isinstance(chunk_metadata, dict):
                continue
            if chunk_metadata.get("image_path"):
                continue
            page_image_path = page_image_paths.get(page_num)
            if page_image_path:
                chunk_metadata["image_path"] = page_image_path

        if graph_records:
            _set_progress(78, "processing", "Persisting diagram relationships")
            seen_graph_hashes: set[str] = set()
            rows_to_insert: List[Dict[str, Any]] = []
            for record in graph_records:
                page_num = int(record.get("page") or 1)
                graph_payload = record.get("graph")
                if not isinstance(graph_payload, dict):
                    continue
                parser_version = str(record.get("parser_version") or "pptx-graph-v1")
                try:
                    graph_fingerprint = json.dumps(graph_payload, sort_keys=True, ensure_ascii=True)
                except Exception:
                    graph_fingerprint = str(graph_payload)
                dedupe_key = f"{page_num}:{parser_version}:{graph_fingerprint[:4000]}"
                if dedupe_key in seen_graph_hashes:
                    continue
                seen_graph_hashes.add(dedupe_key)
                image_path = str(record.get("image_path") or page_image_paths.get(page_num) or "")
                metadata = record.get("metadata")
                safe_metadata = metadata if isinstance(metadata, dict) else {}
                rows_to_insert.append(
                    {
                        "id": str(record.get("id") or uuid.uuid4()),
                        "doc_id": str(record.get("doc_id") or doc_id),
                        "page": page_num,
                        "image_path": image_path,
                        "parser_version": parser_version,
                        "graph": graph_payload,
                        "confidence": float(record.get("confidence") or 1.0),
                        "created_at": str(record.get("created_at") or datetime.now(timezone.utc).isoformat()),
                        "metadata": safe_metadata,
                    }
                )
            if rows_to_insert:
                storage.add_diagram_graphs(rows_to_insert)

        if chunks:
            _set_progress(84, "processing", f"Embedding {len(chunks)} chunk(s)")
            storage.add_chunks(chunks)
            texts = [chunk["content"] for chunk in chunks]
            vectors = embedder.embed_texts(texts)
            dim = vectors.shape[1]
            storage.add_embeddings(
                (chunk["id"], vec.astype("float32").tobytes(), dim)
                for chunk, vec in zip(chunks, vectors)
            )
            _set_progress(95, "processing", "Updating retrieval indexes")
            if index_lock is None:
                vector_index.add(vectors, [chunk["id"] for chunk in chunks])
                if sparse_index is not None:
                    sparse_index.add_chunks(chunks)
            else:
                with index_lock:
                    vector_index.add(vectors, [chunk["id"] for chunk in chunks])
                    if sparse_index is not None:
                        sparse_index.add_chunks(chunks)

        max_page = max((block.get("page") or 0 for block in blocks), default=0)
        _set_progress(
            100,
            "ready",
            "Ingestion complete",
            status="ready",
            num_pages=max_page,
            extra_metadata={
                "ingest_block_count": total_blocks,
                "ingest_chunk_count": len(chunks),
                "auto_tags": auto_tags,
                "auto_tag_version": "v1",
                "ingest_error": None,
            },
            force=True,
        )
    except Exception as exc:
        _set_progress(
            progress_pct,
            "failed",
            "Ingestion failed",
            status="failed",
            extra_metadata={"ingest_error": str(exc)[:800]},
            force=True,
        )
        raise


def create_document_record(filename: str) -> str:
    doc_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    storage.add_document(
        doc_id=doc_id,
        filename=filename,
        status="queued",
        created_at=created_at,
        metadata={
            "ingest_progress": 0,
            "ingest_stage": "queued",
            "ingest_message": "Queued for ingestion",
            "ingest_updated_at": created_at,
        },
    )
    return doc_id


def _safe_relative_upload_path(filename: str) -> Path:
    raw = str(filename or "").replace("\\", "/")
    parts = [part for part in PurePosixPath(raw).parts if part not in {"", ".", ".."}]
    if not parts:
        parts = ["upload.bin"]
    return Path(*parts)


def safe_path_for_upload(filename: str, doc_id: str) -> Path:
    upload_dir = config.UPLOAD_DIR / doc_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    relative_path = _safe_relative_upload_path(filename)
    destination = upload_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination
