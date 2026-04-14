# Reviewer workflow utilities for folder out-of-place warning management.
# Provides decision handling and folder-level review summaries for enterprise operations.

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .. import config, storage
from .document_classifier import normalize_doc_type


_ALLOWED_DECISIONS = {"dismiss", "accept", "whitelist", "reopen"}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_type_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    values: list[str] = []
    seen: set[str] = set()
    for item in value:
        normalized = normalize_doc_type(str(item or ""), use_alias_map=False)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    return values


def _normalize_reviewer(reviewer: dict[str, Any] | None) -> dict[str, str]:
    reviewer = reviewer if isinstance(reviewer, dict) else {}
    reviewer_type = str(reviewer.get("type") or "").strip().lower()
    if reviewer_type not in {"human", "service"}:
        reviewer_type = "human"
    return {
        "id": str(reviewer.get("id") or "").strip(),
        "name": str(reviewer.get("name") or "").strip(),
        "email": str(reviewer.get("email") or "").strip(),
        "role": str(reviewer.get("role") or "").strip(),
        "type": reviewer_type,
    }


def _build_open_flag(document: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "fileId": str(document.get("id") or ""),
        "filename": str(document.get("filename") or ""),
        "expectedType": str(metadata.get("out_of_place_review_expected_type") or metadata.get("expected_doc_type") or ""),
        "predictedType": str(metadata.get("out_of_place_review_predicted_type") or metadata.get("doc_type") or ""),
        "confidence": float(metadata.get("out_of_place_review_confidence") or metadata.get("doc_type_confidence") or 0.0),
        "reason": str(metadata.get("out_of_place_review_reason") or ""),
        "checkedAt": str(metadata.get("out_of_place_review_last_checked_at") or ""),
    }


def build_folder_review_summary(
    *,
    folder_id: str,
    tenant_id: str,
) -> dict[str, Any]:
    target_folder = str(folder_id or "").strip()
    target_tenant = str(tenant_id or config.DEFAULT_TENANT_ID).strip() or config.DEFAULT_TENANT_ID

    counts = {
        "total": 0,
        "needs_review": 0,
        "clear": 0,
        "accepted_mismatch": 0,
        "pending": 0,
        "whitelisted": 0,
    }
    open_flags: list[dict[str, Any]] = []

    for document in storage.list_documents():
        metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
        if str(metadata.get("tenant_id") or "").strip() != target_tenant:
            continue
        if str(metadata.get("folder_id") or "").strip() != target_folder:
            continue

        counts["total"] += 1
        status = str(document.get("status") or "").strip().lower()
        if status in {"queued", "processing"}:
            counts["pending"] += 1

        state = str(metadata.get("out_of_place_review_state") or "").strip().lower()
        if state not in {"needs_review", "clear", "accepted_mismatch"}:
            state = "clear"
        counts[state] += 1

        if bool(metadata.get("doc_review_whitelisted")):
            counts["whitelisted"] += 1

        if state == "needs_review":
            open_flags.append(_build_open_flag(document, metadata))

    open_flags.sort(key=lambda item: float(item.get("confidence") or 0.0), reverse=True)

    review_status = "clear"
    if counts["needs_review"] > 0:
        review_status = "needs_review"
    elif counts["pending"] > 0:
        review_status = "processing"

    return {
        "tenant_id": target_tenant,
        "folder_id": target_folder,
        "review_status": review_status,
        "open_flag_count": int(counts["needs_review"]),
        "flags": open_flags,
        "counts": counts,
    }


def apply_review_decision(
    *,
    folder_id: str,
    doc_id: str,
    tenant_id: str,
    decision: str,
    reviewer: dict[str, Any] | None = None,
    note: str | None = None,
    whitelist_predicted_type: bool = False,
) -> dict[str, Any]:
    normalized_decision = str(decision or "").strip().lower()
    if normalized_decision not in _ALLOWED_DECISIONS:
        raise ValueError(f"Unsupported decision '{decision}'. Allowed: {', '.join(sorted(_ALLOWED_DECISIONS))}")

    target_doc_id = str(doc_id or "").strip()
    if not target_doc_id:
        raise ValueError("doc_id is required")

    target_folder = str(folder_id or "").strip()
    if not target_folder:
        raise ValueError("folder_id is required")

    target_tenant = str(tenant_id or config.DEFAULT_TENANT_ID).strip() or config.DEFAULT_TENANT_ID
    document = storage.get_document(target_doc_id)
    if not document:
        raise LookupError(f"Document not found: {target_doc_id}")

    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    doc_tenant = str(metadata.get("tenant_id") or "").strip() or config.DEFAULT_TENANT_ID
    if doc_tenant != target_tenant:
        raise PermissionError("Document tenant does not match requested tenant")

    doc_folder = str(metadata.get("folder_id") or "").strip()
    if doc_folder != target_folder:
        raise ValueError("Document is not part of the specified folder")

    previous_state = str(metadata.get("out_of_place_review_state") or "").strip().lower() or "clear"
    expected_type = normalize_doc_type(
        str(metadata.get("out_of_place_review_expected_type") or metadata.get("expected_doc_type") or ""),
        use_alias_map=False,
    )
    predicted_type = normalize_doc_type(
        str(metadata.get("out_of_place_review_predicted_type") or metadata.get("doc_type") or ""),
        use_alias_map=False,
    )
    reviewer_payload = _normalize_reviewer(reviewer)
    decision_note = str(note or "").strip()[:1000]
    decided_at = _iso_now()

    next_state = previous_state
    manual_override = bool(metadata.get("out_of_place_review_manual_override"))
    action_required = previous_state == "needs_review"
    reason = str(metadata.get("out_of_place_review_reason") or "")
    update_payload: dict[str, Any] = {
        "out_of_place_review_decision": normalized_decision,
        "out_of_place_review_decided_at": decided_at,
        "out_of_place_review_decided_by_id": reviewer_payload["id"],
        "out_of_place_review_decided_by_name": reviewer_payload["name"],
        "out_of_place_review_decided_by_email": reviewer_payload["email"],
        "out_of_place_review_decided_by_role": reviewer_payload["role"],
        "out_of_place_review_decided_by_type": reviewer_payload["type"],
        "out_of_place_review_decision_note": decision_note,
        "updated_at": decided_at,
    }

    if normalized_decision == "dismiss":
        next_state = "clear"
        manual_override = True
        action_required = False
        reason = decision_note or "Flag dismissed by reviewer"
    elif normalized_decision == "accept":
        next_state = "accepted_mismatch"
        manual_override = True
        action_required = False
        reason = decision_note or reason or "Mismatch accepted by reviewer"
    elif normalized_decision == "whitelist":
        next_state = "clear"
        manual_override = True
        action_required = False
        reason = decision_note or "Document whitelisted by reviewer"
        update_payload["doc_review_whitelisted"] = True
        whitelist_types = _normalize_type_list(metadata.get("doc_review_whitelist_types"))
        if whitelist_predicted_type and predicted_type and predicted_type not in whitelist_types:
            whitelist_types.append(predicted_type)
        update_payload["doc_review_whitelist_types"] = whitelist_types
    elif normalized_decision == "reopen":
        next_state = "clear"
        manual_override = False
        action_required = True
        reason = decision_note or "Review reopened"
        update_payload["doc_review_whitelisted"] = False

    update_payload["out_of_place_review_state"] = next_state
    update_payload["out_of_place_review_manual_override"] = manual_override
    update_payload["out_of_place_review_action_required"] = action_required
    update_payload["out_of_place_review_reason"] = reason

    history = metadata.get("out_of_place_review_decisions") if isinstance(metadata.get("out_of_place_review_decisions"), list) else []
    history.append(
        {
            "decided_at": decided_at,
            "decision": normalized_decision,
            "previous_state": previous_state,
            "new_state": next_state,
            "expected_type": expected_type,
            "predicted_type": predicted_type,
            "note": decision_note,
            "reviewer": reviewer_payload,
        }
    )
    update_payload["out_of_place_review_decisions"] = history[-50:]

    storage.update_document(target_doc_id, metadata=update_payload, merge_metadata=True)

    summary = build_folder_review_summary(folder_id=target_folder, tenant_id=target_tenant)
    return {
        "tenant_id": target_tenant,
        "folder_id": target_folder,
        "doc_id": target_doc_id,
        "decision": normalized_decision,
        "previous_state": previous_state,
        "new_state": next_state,
        "manual_override": manual_override,
        "action_required": action_required,
        "predicted_type": predicted_type,
        "expected_type": expected_type,
        "decided_at": decided_at,
        "summary": summary,
    }
