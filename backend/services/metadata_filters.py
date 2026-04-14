from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable


def filter_documents_by_metadata(
    docs: list[dict[str, Any]],
    *,
    operation: str,
    filters: dict[str, Any],
    parse_datetime: Callable[[Any], datetime | None],
    doc_created_at: Callable[[dict[str, Any]], datetime | None],
    doc_updated_at: Callable[[dict[str, Any]], datetime | None],
    doc_matches_doc_type_filter: Callable[[dict[str, Any], str], bool],
    doc_matches_target_document: Callable[[dict[str, Any], str], bool],
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    reference_now = now or datetime.now(timezone.utc)
    filtered: list[dict[str, Any]] = []
    date_from = parse_datetime(filters.get("date_from"))
    date_to = parse_datetime(filters.get("date_to"))
    if operation == "created_before" and date_to is None and date_from is not None:
        date_to = date_from
        date_from = None
    relative_days = filters.get("relative_days")
    try:
        relative_days_int = int(relative_days) if relative_days is not None else None
    except Exception:
        relative_days_int = None
    cutoff_dt = reference_now - timedelta(days=max(0, relative_days_int or 0)) if relative_days_int is not None else None

    author_filter = str(filters.get("author") or "").strip().lower()
    editor_filter = str(filters.get("last_modified_by") or "").strip().lower()
    uploader_role_filter = str(filters.get("uploader_role") or "").strip().lower()
    collaborator_filter = str(filters.get("collaborator_type") or "").strip().lower()
    doc_type_filter = str(filters.get("doc_type") or "").strip().lower()
    target_document_filter = str(filters.get("target_document") or "").strip().lower()

    date_baseline = "created" if operation in {"created_after", "created_before", "created_between"} else "updated"
    if operation in {"list", "count"} and (date_from is not None or date_to is not None):
        date_baseline = "created"

    for doc in docs:
        metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        created_at = doc_created_at(doc)
        updated_at = doc_updated_at(doc)
        if doc_type_filter and not doc_matches_doc_type_filter(doc, doc_type_filter):
            continue
        if author_filter:
            author = str(metadata.get("author") or "").strip().lower()
            if author_filter not in author:
                continue
        if editor_filter:
            editor = str(metadata.get("last_modified_by") or "").strip().lower()
            if editor_filter not in editor:
                continue
        if uploader_role_filter:
            role = str(metadata.get("uploaded_by_role") or "").strip().lower()
            if uploader_role_filter not in role:
                continue
        if collaborator_filter:
            collaborator_type = str(metadata.get("collaborator_type") or "").strip().lower()
            if collaborator_filter in {"internal", "external", "unknown"}:
                if collaborator_filter != collaborator_type:
                    continue
            elif collaborator_filter not in collaborator_type:
                continue
        if target_document_filter and not doc_matches_target_document(doc, target_document_filter):
            continue
        if date_from is not None:
            baseline = created_at if date_baseline == "created" else updated_at
            if baseline is None or baseline < date_from:
                continue
        if date_to is not None:
            baseline = created_at if date_baseline == "created" else updated_at
            if baseline is None or baseline > date_to:
                continue
        if cutoff_dt is not None:
            if operation in {"modified_within_days", "edited_by", "last_modified_by"}:
                if updated_at is None or updated_at < cutoff_dt:
                    continue
            elif operation in {"created_after", "created_before", "created_between", "list", "count"}:
                if created_at is None or created_at < cutoff_dt:
                    continue
        filtered.append(doc)
    return filtered
