from __future__ import annotations

import hashlib
import mimetypes
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from xml.etree import ElementTree as ET

import fitz

from .. import config


CORE_NS = {
    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
}


def _iso_from_timestamp(timestamp: float | None) -> str:
    if timestamp is None:
        return ""
    try:
        return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat()
    except Exception:
        return ""


def _sanitize_string(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return text[:400]


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _guess_mime(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return str(mime_type or "application/octet-stream")


def _parse_ooxml_core_properties(path: Path) -> dict[str, Any]:
    if path.suffix.lower() not in {".docx", ".pptx", ".xlsx", ".xlsm", ".xltx", ".xltm"}:
        return {}
    try:
        with zipfile.ZipFile(path) as archive:
            if "docProps/core.xml" not in set(archive.namelist()):
                return {}
            raw = archive.read("docProps/core.xml")
    except Exception:
        return {}
    try:
        root = ET.fromstring(raw)
    except Exception:
        return {}
    values: dict[str, Any] = {}
    title = root.findtext(".//dc:title", default="", namespaces=CORE_NS)
    subject = root.findtext(".//dc:subject", default="", namespaces=CORE_NS)
    author = root.findtext(".//dc:creator", default="", namespaces=CORE_NS)
    last_modified_by = root.findtext(".//cp:lastModifiedBy", default="", namespaces=CORE_NS)
    created = root.findtext(".//dcterms:created", default="", namespaces=CORE_NS)
    modified = root.findtext(".//dcterms:modified", default="", namespaces=CORE_NS)
    revision = root.findtext(".//cp:revision", default="", namespaces=CORE_NS)

    if title:
        values["title"] = _sanitize_string(title)
    if subject:
        values["subject"] = _sanitize_string(subject)
    if author:
        values["author"] = _sanitize_string(author)
    if last_modified_by:
        values["last_modified_by"] = _sanitize_string(last_modified_by)
    if created:
        values["content_created_at"] = _sanitize_string(created)
    if modified:
        values["content_modified_at"] = _sanitize_string(modified)
    if revision:
        values["source_revision"] = _sanitize_string(revision)
    return values


def _parse_pdf_properties(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".pdf":
        return {}
    try:
        with fitz.open(str(path)) as doc:
            metadata = doc.metadata or {}
    except Exception:
        return {}
    values: dict[str, Any] = {}
    for key, target in (
        ("title", "title"),
        ("subject", "subject"),
        ("author", "author"),
        ("creator", "creator"),
        ("producer", "producer"),
    ):
        value = _sanitize_string(metadata.get(key))
        if value:
            values[target] = value
    creation_date = _sanitize_string(metadata.get("creationDate"))
    mod_date = _sanitize_string(metadata.get("modDate"))
    if creation_date:
        values["content_created_at"] = creation_date
    if mod_date:
        values["content_modified_at"] = mod_date
    return values


def _normalize_logical_document_key(filename: str) -> str:
    base = str(filename or "").replace("\\", "/").strip().lower()
    if not base:
        return "unknown"
    stem, _, ext = base.rpartition(".")
    if not stem:
        stem = base
        ext = ""
    normalized = re.sub(r"[_\-]+", " ", stem)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"(?:\bversion\b|\brev\b|\bv\b)\s*\d+$", "", normalized).strip()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    suffix = f".{ext}" if ext else ""
    return f"{normalized}{suffix}"


def _infer_collaborator_type(
    *,
    uploader_email: str,
    uploader_role: str,
    existing_value: str,
) -> str:
    prior = str(existing_value or "").strip().lower()
    if prior in {"internal", "external"}:
        return prior
    role = str(uploader_role or "").strip().lower()
    if role == "intern":
        return "internal"
    email = str(uploader_email or "").strip().lower()
    if not email or "@" not in email or not config.INTERNAL_EMAIL_DOMAINS:
        return "unknown"
    domain = email.split("@", 1)[-1]
    if domain in set(config.INTERNAL_EMAIL_DOMAINS):
        return "internal"
    return "external"


def collect_document_metadata(
    *,
    path: Path,
    doc_id: str,
    filename: str,
    existing_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    existing_metadata = existing_metadata if isinstance(existing_metadata, dict) else {}
    now_iso = datetime.now(timezone.utc).isoformat()
    stat = path.stat()
    uploader_email = _sanitize_string(existing_metadata.get("uploaded_by_email"))
    uploader_role = _sanitize_string(existing_metadata.get("uploaded_by_role"))
    payload: Dict[str, Any] = {
        "document_id": doc_id,
        "tenant_id": _sanitize_string(existing_metadata.get("tenant_id") or config.DEFAULT_TENANT_ID),
        "source_system": _sanitize_string(existing_metadata.get("source_system") or "upload"),
        "source_uri": _sanitize_string(existing_metadata.get("source_uri") or filename),
        "checksum": _sha256_file(path),
        "checksum_algorithm": "sha256",
        "mime_type": _guess_mime(path),
        "size_bytes": int(stat.st_size),
        "file_extension": path.suffix.lower(),
        "created_at": _sanitize_string(existing_metadata.get("created_at") or existing_metadata.get("uploaded_at") or now_iso),
        "uploaded_at": _sanitize_string(existing_metadata.get("uploaded_at") or now_iso),
        "updated_at": now_iso,
        "fs_created_at": _iso_from_timestamp(getattr(stat, "st_ctime", None)),
        "fs_modified_at": _iso_from_timestamp(getattr(stat, "st_mtime", None)),
        "logical_document_key": _normalize_logical_document_key(filename),
        "uploaded_by_id": _sanitize_string(existing_metadata.get("uploaded_by_id")),
        "uploaded_by_name": _sanitize_string(existing_metadata.get("uploaded_by_name")),
        "uploaded_by_email": uploader_email,
        "uploaded_by_role": uploader_role,
        "uploaded_by_type": _sanitize_string(existing_metadata.get("uploaded_by_type")),
    }
    payload["collaborator_type"] = _infer_collaborator_type(
        uploader_email=uploader_email,
        uploader_role=uploader_role,
        existing_value=_sanitize_string(existing_metadata.get("collaborator_type")),
    )

    for parser_payload in (_parse_ooxml_core_properties(path), _parse_pdf_properties(path)):
        for key, value in parser_payload.items():
            if value:
                payload[key] = value
    if not payload.get("author"):
        payload["author"] = payload.get("uploaded_by_name") or payload.get("uploaded_by_email") or ""
    if not payload.get("last_modified_by"):
        payload["last_modified_by"] = payload.get("uploaded_by_name") or payload.get("uploaded_by_email") or ""
    return payload


def compute_version_metadata(
    *,
    doc_id: str,
    doc_metadata: Dict[str, Any],
    documents: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    tenant_id = str(doc_metadata.get("tenant_id") or "")
    logical_key = str(doc_metadata.get("logical_document_key") or "")
    current_checksum = str(doc_metadata.get("checksum") or "")
    candidates: list[Dict[str, Any]] = []
    for candidate in documents:
        candidate_id = str(candidate.get("id") or "")
        if not candidate_id or candidate_id == doc_id:
            continue
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        if str(metadata.get("tenant_id") or "") != tenant_id:
            continue
        if str(metadata.get("logical_document_key") or "") != logical_key:
            continue
        candidates.append(candidate)
    candidates.sort(key=lambda item: str(item.get("created_at") or ""))
    latest = candidates[-1] if candidates else None
    latest_md = latest.get("metadata") if isinstance((latest or {}).get("metadata"), dict) else {}
    try:
        previous_version = int(latest_md.get("version_index") or 0)
    except Exception:
        previous_version = len(candidates)
    version_index = max(1, previous_version + 1)
    previous_doc_id = str((latest or {}).get("id") or "")
    previous_checksum = str(latest_md.get("checksum") or "")
    change_type = "new_document"
    if latest:
        change_type = "content_changed" if current_checksum and current_checksum != previous_checksum else "metadata_only_change"
    return {
        "version_index": version_index,
        "version_label": f"v{version_index}",
        "version_previous_doc_id": previous_doc_id or None,
        "version_group_size": len(candidates) + 1,
        "update_count": max(0, version_index - 1),
        "change_type": change_type,
    }
