# Out-of-place document detection for folder-labeled enterprise uploads.
# Applies high-precision mismatch rules and writes auditable review metadata.

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import config, storage
from .document_classifier import classify_document, normalize_doc_type


LOGGER = logging.getLogger(__name__)


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_type_list(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    values: set[str] = set()
    for item in value:
        normalized = normalize_doc_type(str(item or ""), use_alias_map=False)
        if normalized:
            values.add(normalized)
    return values


def _find_source_file(doc_id: str) -> Path | None:
    doc_dir = config.UPLOAD_DIR / str(doc_id or "").strip()
    if not doc_dir.exists() or not doc_dir.is_dir():
        return None
    for candidate in doc_dir.rglob("*"):
        if candidate.is_file():
            return candidate
    return None


def _collect_folder_documents(folder_id: str, tenant_id: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    expected_folder = str(folder_id or "").strip()
    expected_tenant = str(tenant_id or config.DEFAULT_TENANT_ID).strip() or config.DEFAULT_TENANT_ID
    for document in storage.list_documents():
        metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
        if str(metadata.get("folder_id") or "").strip() != expected_folder:
            continue
        if str(metadata.get("tenant_id") or "").strip() != expected_tenant:
            continue
        status = str(document.get("status") or "").strip().lower()
        if status not in {"ready", "processing"}:
            continue
        matches.append(document)
    return matches


def _resolve_threshold(
    *,
    tenant_id: str,
    folder_documents: list[dict[str, Any]],
    explicit_threshold: float | None,
) -> float:
    if explicit_threshold is not None:
        return max(0.01, min(0.99, float(explicit_threshold)))

    for document in folder_documents:
        metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
        threshold = metadata.get("doc_review_threshold")
        if threshold is not None:
            value = _to_float(threshold, config.DOC_TYPE_REVIEW_CONFIDENCE_THRESHOLD)
            return max(0.01, min(0.99, value))

    tenant_threshold = config.DOC_TYPE_REVIEW_TENANT_THRESHOLDS.get(str(tenant_id or "").strip())
    if tenant_threshold is not None:
        return max(0.01, min(0.99, float(tenant_threshold)))

    return max(0.01, min(0.99, float(config.DOC_TYPE_REVIEW_CONFIDENCE_THRESHOLD)))


def _equivalent_type_set(expected_type: str) -> set[str]:
    normalized = normalize_doc_type(expected_type, use_alias_map=False)
    allowed = {normalized}
    for value in config.DOC_TYPE_REVIEW_EQUIVALENT_TYPES.get(normalized, set()):
        clean = normalize_doc_type(value, use_alias_map=False)
        if clean:
            allowed.add(clean)
    return allowed


def _normalize_score_map(raw: Any, *, use_alias_map: bool) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    scores: dict[str, float] = {}
    for key, value in raw.items():
        normalized = normalize_doc_type(str(key or ""), use_alias_map=use_alias_map)
        if not normalized:
            continue
        try:
            scores[normalized] = float(value)
        except Exception:
            continue
    return scores


def _backfill_missing_prediction(document: dict[str, Any]) -> dict[str, Any]:
    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    doc_id = str(document.get("id") or "").strip()
    filename = str(document.get("filename") or metadata.get("original_filename") or "document")
    auto_tags = metadata.get("auto_tags") if isinstance(metadata.get("auto_tags"), list) else []

    text_samples: list[str] = []
    if doc_id:
        for chunk in storage.get_chunks_by_doc(doc_id)[:10]:
            value = str(chunk.get("content") or "").strip()
            if value:
                text_samples.append(value[:360])

    source_file = _find_source_file(doc_id)
    result = classify_document(
        file_path=source_file,
        filename=filename,
        auto_tags=[str(item) for item in auto_tags],
        text_samples=text_samples,
    )

    payload = {
        "doc_type": result.doc_type,
        "doc_type_confidence": float(result.confidence),
        "doc_type_scores": result.scores,
        "doc_type_classifier_provider": result.provider,
        "doc_type_classifier_model": result.model,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if doc_id:
        storage.update_document(doc_id, metadata=payload, merge_metadata=True)
        updated = storage.get_document(doc_id) or {}
        return updated.get("metadata") if isinstance(updated.get("metadata"), dict) else {**metadata, **payload}
    merged = dict(metadata)
    merged.update(payload)
    return merged


def _evaluate_document(
    *,
    folder_id: str,
    expected_type: str,
    allowed_expected: set[str],
    threshold: float,
    min_score_ratio: float,
    document: dict[str, Any],
    run_id: str,
    checked_at: str,
) -> dict[str, Any] | None:
    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    doc_id = str(document.get("id") or "").strip()
    classifier_provider = str(metadata.get("doc_type_classifier_provider") or "").strip().lower()
    use_alias_map = classifier_provider not in {"semantic_openai"}

    predicted = normalize_doc_type(str(metadata.get("doc_type") or ""), use_alias_map=use_alias_map)
    confidence = _to_float(metadata.get("doc_type_confidence"), 0.0)
    if not predicted or predicted == "unknown":
        metadata = _backfill_missing_prediction(document)
        classifier_provider = str(metadata.get("doc_type_classifier_provider") or "").strip().lower()
        use_alias_map = classifier_provider not in {"semantic_openai"}
        predicted = normalize_doc_type(str(metadata.get("doc_type") or ""), use_alias_map=use_alias_map)
        confidence = _to_float(metadata.get("doc_type_confidence"), 0.0)

    scores = _normalize_score_map(metadata.get("doc_type_scores"), use_alias_map=use_alias_map)
    expected_score = float(scores.get(expected_type, 0.0))
    predicted_score = float(scores.get(predicted, 0.0))
    score_ratio = predicted_score / max(0.01, expected_score) if expected_score > 0 else 99.0

    ignored = set(normalize_doc_type(value) for value in config.DOC_TYPE_REVIEW_IGNORED_PREDICTED_TYPES)
    ignored.discard("")
    whitelist_types = _normalize_type_list(metadata.get("doc_review_whitelist_types"))
    whitelisted_document = bool(metadata.get("doc_review_whitelisted"))
    manual_override = bool(metadata.get("out_of_place_review_manual_override"))

    flagged = bool(
        predicted
        and predicted not in allowed_expected
        and predicted not in ignored
        and predicted not in whitelist_types
        and not whitelisted_document
        and not manual_override
        and confidence >= threshold
        and score_ratio >= min_score_ratio
    )

    reason = (
        f"Predicted '{predicted}' ({confidence:.2f}) but folder expects '{expected_type}'."
        if flagged
        else ""
    )

    previous_state = str(metadata.get("out_of_place_review_state") or "").strip().lower() or "clear"
    next_state = "needs_review" if flagged else "clear"
    if not flagged and manual_override and previous_state in {"accepted_mismatch", "clear"}:
        next_state = previous_state

    action_required = bool(next_state == "needs_review")

    audit_entry = {
        "checked_at": checked_at,
        "run_id": run_id,
        "folder_id": folder_id,
        "expected_type": expected_type,
        "predicted_type": predicted or "unknown",
        "confidence": round(float(confidence), 4),
        "threshold": round(float(threshold), 4),
        "min_score_ratio": round(float(min_score_ratio), 4),
        "flagged": bool(flagged),
        "manual_override": manual_override,
        "reason": reason,
    }

    audit_log = metadata.get("out_of_place_review_audit") if isinstance(metadata.get("out_of_place_review_audit"), list) else []
    audit_log.append(audit_entry)
    audit_log = audit_log[-100:]

    update_payload = {
        "out_of_place_review_state": next_state,
        "out_of_place_review_action_required": action_required,
        "out_of_place_review_last_checked_at": checked_at,
        "out_of_place_review_run_id": run_id,
        "out_of_place_review_folder_id": folder_id,
        "out_of_place_review_expected_type": expected_type,
        "out_of_place_review_predicted_type": predicted or "unknown",
        "out_of_place_review_confidence": round(float(confidence), 4),
        "out_of_place_review_reason": reason,
        "out_of_place_review_threshold": round(float(threshold), 4),
        "out_of_place_review_min_score_ratio": round(float(min_score_ratio), 4),
        "out_of_place_review_audit": audit_log,
        "updated_at": checked_at,
    }
    if previous_state != next_state:
        update_payload["out_of_place_review_last_transition"] = {
            "changed_at": checked_at,
            "from": previous_state,
            "to": next_state,
            "run_id": run_id,
        }

    if doc_id:
        storage.update_document(doc_id, metadata=update_payload, merge_metadata=True)

    if not flagged:
        return None

    filename = str(document.get("filename") or metadata.get("original_filename") or "")
    return {
        "fileId": doc_id,
        "filename": filename,
        "folderId": folder_id,
        "expectedType": expected_type,
        "predictedType": predicted,
        "confidence": round(float(confidence), 4),
        "reason": reason,
        "scoreRatio": round(float(score_ratio), 4),
        "reviewRunId": run_id,
        "checkedAt": checked_at,
    }


def detect_out_of_place_documents(
    folder_id: str,
    expected_type: str,
    *,
    tenant_id: str | None = None,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    if not config.DOC_TYPE_REVIEW_ENABLED:
        return []

    folder_id = str(folder_id or "").strip()
    if not folder_id:
        return []

    normalized_expected = normalize_doc_type(expected_type, use_alias_map=False)
    if not normalized_expected:
        return []

    resolved_tenant = str(tenant_id or config.DEFAULT_TENANT_ID).strip() or config.DEFAULT_TENANT_ID
    folder_documents = _collect_folder_documents(folder_id, resolved_tenant)
    if not folder_documents:
        return []

    resolved_threshold = _resolve_threshold(
        tenant_id=resolved_tenant,
        folder_documents=folder_documents,
        explicit_threshold=threshold,
    )
    min_score_ratio = max(1.0, float(config.DOC_TYPE_REVIEW_MIN_SCORE_RATIO))
    allowed_expected = _equivalent_type_set(normalized_expected)
    checked_at = datetime.now(timezone.utc).isoformat()
    run_id = f"review-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"

    flagged: list[dict[str, Any]] = []
    for document in folder_documents:
        try:
            result = _evaluate_document(
                folder_id=folder_id,
                expected_type=normalized_expected,
                allowed_expected=allowed_expected,
                threshold=resolved_threshold,
                min_score_ratio=min_score_ratio,
                document=document,
                run_id=run_id,
                checked_at=checked_at,
            )
            if result:
                flagged.append(result)
        except Exception as exc:
            LOGGER.warning("Out-of-place evaluation failed for doc=%s: %s", document.get("id"), exc)

    flagged.sort(key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
    return flagged


def detectOutOfPlaceDocuments(
    folderId: str,
    expectedType: str,
    tenantId: str | None = None,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    return detect_out_of_place_documents(
        folder_id=folderId,
        expected_type=expectedType,
        tenant_id=tenantId,
        threshold=threshold,
    )
