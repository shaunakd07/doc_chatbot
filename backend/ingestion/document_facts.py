from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
import re
import uuid
from typing import Any, Iterable, List


MONTH_NAME_TO_NUM = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

CURRENCY_CODE_MAP = {
    "$": "USD",
    "usd": "USD",
    "sgd": "SGD",
    "eur": "EUR",
    "gbp": "GBP",
    "inr": "INR",
    "idr": "IDR",
    "myr": "MYR",
    "rm": "MYR",
    "aed": "AED",
    "aud": "AUD",
    "cad": "CAD",
    "chf": "CHF",
    "cny": "CNY",
    "hkd": "HKD",
    "jpy": "JPY",
}

ENTITY_SUFFIX_PATTERN = (
    r"Bank|Bhd\.?|Co\.?|Company|Corp\.?|Corporation|Group|Holdings|Inc\.?|"
    r"LLC|LLP|Limited|Ltd\.?|PLC|Pte\.?\s+Ltd\.?|Services|Solutions|Systems|Technologies|Technology|Labs"
)
ENTITY_RE = re.compile(
    rf"\b([A-Z][A-Za-z0-9&'./-]*(?:\s+[A-Z][A-Za-z0-9&'./-]*){{0,6}}\s+(?:{ENTITY_SUFFIX_PATTERN}))\b"
)
PARTY_LABEL_RE = re.compile(
    r"(?i)\b(customer|client|vendor|supplier|buyer|seller|party|counterparty|bill to|ship to)\s*[:\-]\s*(?P<body>[^\n.;]{2,180})"
)
PARTY_BETWEEN_RE = re.compile(r"(?i)\b(?:by and between|between|among)\s+(?P<body>[^.;\n]{4,240})")

ISO_DATE_RE = re.compile(r"\b((?:19|20)\d{2})[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b")
DMY_SLASH_DATE_RE = re.compile(r"\b(0?[1-9]|[12]\d|3[01])[-/](0?[1-9]|1[0-2])[-/](20\d{2}|19\d{2})\b")
DMY_MONTH_RE = re.compile(
    r"\b([0-3]?\d)\s+"
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+((?:19|20)\d{2})\b",
    flags=re.IGNORECASE,
)
MONTH_DY_RE = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+([0-3]?\d),\s*((?:19|20)\d{2})\b",
    flags=re.IGNORECASE,
)
AMOUNT_RE = re.compile(
    r"(?ix)"
    r"(?<![A-Za-z0-9])"
    r"(?:(?P<currency_before>USD|SGD|EUR|GBP|INR|IDR|MYR|RM|AED|AUD|CAD|CHF|CNY|HKD|JPY)\s*|(?P<symbol>[$]))"
    r"(?P<number>\d{1,3}(?:,\d{3})+|\d+)"
    r"(?P<decimal>\.\d{1,2})?"
    r"(?:\s*(?P<magnitude>k|m|b|thousand|million|billion))?"
    r"(?:\s*(?P<currency_after>USD|SGD|EUR|GBP|INR|IDR|MYR|RM|AED|AUD|CAD|CHF|CNY|HKD|JPY))?"
    r"(?![A-Za-z0-9])"
)

EXTRACTOR_VERSION = "document-facts-v1"


def _normalize_whitespace(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalize_party_value(value: str) -> str:
    cleaned = _normalize_whitespace(value).strip(" ,;:.()[]{}")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left_markers = [
        text.rfind(".", 0, start),
        text.rfind("?", 0, start),
        text.rfind("!", 0, start),
        text.rfind(";", 0, start),
        text.rfind("\n", 0, start),
    ]
    left = max(left_markers) + 1
    right_candidates = [
        idx for idx in (
            text.find(".", end),
            text.find("?", end),
            text.find("!", end),
            text.find(";", end),
            text.find("\n", end),
        )
        if idx >= 0
    ]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    return max(0, left), min(len(text), right)


def _extract_evidence_text(text: str, start: int, end: int) -> str:
    left, right = _sentence_bounds(text, start, end)
    snippet = _normalize_whitespace(text[left:right])
    if snippet:
        return snippet[:500]
    window_start = max(0, start - 120)
    window_end = min(len(text), end + 120)
    return _normalize_whitespace(text[window_start:window_end])[:500]


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _canonical_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _resolve_currency(match: re.Match[str]) -> str:
    for raw in (
        match.group("currency_before"),
        match.group("currency_after"),
        match.group("symbol"),
    ):
        key = str(raw or "").strip().lower()
        if key in CURRENCY_CODE_MAP:
            return CURRENCY_CODE_MAP[key]
    return ""


def _extract_entities(value: str) -> list[str]:
    body = _normalize_whitespace(value)
    if not body:
        return []
    entities: list[str] = []
    seen: set[str] = set()
    for match in ENTITY_RE.finditer(body):
        candidate = _normalize_party_value(match.group(1))
        key = candidate.lower()
        if not candidate or key in seen:
            continue
        seen.add(key)
        entities.append(candidate)
    if entities:
        return entities

    cleaned = _normalize_party_value(body)
    if re.match(r"^[A-Z][A-Za-z0-9&'./-]*(?:\s+[A-Z][A-Za-z0-9&'./-]*){0,6}$", cleaned):
        return [cleaned]
    return []


def _build_fact(
    *,
    doc_id: str,
    fact_type: str,
    canonical_value: str,
    raw_value: str,
    page: int | None,
    chunk_id: str | None,
    evidence_text: str,
    confidence: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "doc_id": doc_id,
        "fact_type": fact_type,
        "canonical_value": _normalize_whitespace(canonical_value)[:400],
        "raw_value": _normalize_whitespace(raw_value)[:400],
        "page": page,
        "chunk_id": chunk_id,
        "evidence_text": _normalize_whitespace(evidence_text)[:1000],
        "confidence": float(confidence),
        "metadata": metadata,
    }


def _extract_date_facts(doc_id: str, chunk: dict[str, Any]) -> list[dict[str, Any]]:
    text = str(chunk.get("content") or "")
    page = _to_int(chunk.get("page"))
    chunk_id = str(chunk.get("id") or "").strip() or None
    out: list[dict[str, Any]] = []

    def add_fact(raw_value: str, canonical_value: str, start: int, end: int, pattern: str, confidence: float) -> None:
        evidence_text = _extract_evidence_text(text, start, end)
        out.append(
            _build_fact(
                doc_id=doc_id,
                fact_type="date",
                canonical_value=canonical_value,
                raw_value=raw_value,
                page=page,
                chunk_id=chunk_id,
                evidence_text=evidence_text,
                confidence=confidence,
                metadata={
                    "extractor": EXTRACTOR_VERSION,
                    "pattern": pattern,
                    "source_type": str(chunk.get("source_type") or ""),
                },
            )
        )

    for match in ISO_DATE_RE.finditer(text):
        year, month, day = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        try:
            canonical = datetime(year, month, day).date().isoformat()
        except ValueError:
            continue
        add_fact(match.group(0), canonical, match.start(), match.end(), "iso_date", 0.99)

    for match in DMY_SLASH_DATE_RE.finditer(text):
        day, month, year = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        try:
            canonical = datetime(year, month, day).date().isoformat()
        except ValueError:
            continue
        add_fact(match.group(0), canonical, match.start(), match.end(), "dmy_slash", 0.97)

    for match in DMY_MONTH_RE.finditer(text):
        day = int(match.group(1))
        month = MONTH_NAME_TO_NUM.get(str(match.group(2) or "").strip().lower())
        year = int(match.group(3))
        if month is None:
            continue
        try:
            canonical = datetime(year, month, day).date().isoformat()
        except ValueError:
            continue
        add_fact(match.group(0), canonical, match.start(), match.end(), "dmy_month_name", 0.97)

    for match in MONTH_DY_RE.finditer(text):
        month = MONTH_NAME_TO_NUM.get(str(match.group(1) or "").strip().lower())
        day = int(match.group(2))
        year = int(match.group(3))
        if month is None:
            continue
        try:
            canonical = datetime(year, month, day).date().isoformat()
        except ValueError:
            continue
        add_fact(match.group(0), canonical, match.start(), match.end(), "month_day_year", 0.97)

    return out


def _extract_amount_facts(doc_id: str, chunk: dict[str, Any]) -> list[dict[str, Any]]:
    text = str(chunk.get("content") or "")
    page = _to_int(chunk.get("page"))
    chunk_id = str(chunk.get("id") or "").strip() or None
    out: list[dict[str, Any]] = []

    multiplier_map = {
        "k": Decimal("1000"),
        "thousand": Decimal("1000"),
        "m": Decimal("1000000"),
        "million": Decimal("1000000"),
        "b": Decimal("1000000000"),
        "billion": Decimal("1000000000"),
    }

    for match in AMOUNT_RE.finditer(text):
        currency = _resolve_currency(match)
        if not currency:
            continue
        raw_number = f"{match.group('number') or ''}{match.group('decimal') or ''}"
        raw_number = raw_number.replace(",", "")
        try:
            amount = Decimal(raw_number)
        except (InvalidOperation, ValueError):
            continue
        magnitude = str(match.group("magnitude") or "").strip().lower()
        if magnitude:
            amount *= multiplier_map.get(magnitude, Decimal("1"))
        canonical_amount = _canonical_decimal(amount)
        evidence_text = _extract_evidence_text(text, match.start(), match.end())
        out.append(
            _build_fact(
                doc_id=doc_id,
                fact_type="amount",
                canonical_value=f"{currency} {canonical_amount}",
                raw_value=match.group(0),
                page=page,
                chunk_id=chunk_id,
                evidence_text=evidence_text,
                confidence=0.96,
                metadata={
                    "extractor": EXTRACTOR_VERSION,
                    "currency": currency,
                    "numeric_value": canonical_amount,
                    "magnitude": magnitude or "",
                    "source_type": str(chunk.get("source_type") or ""),
                },
            )
        )
    return out


def _extract_party_facts(doc_id: str, chunk: dict[str, Any]) -> list[dict[str, Any]]:
    text = str(chunk.get("content") or "")
    page = _to_int(chunk.get("page"))
    chunk_id = str(chunk.get("id") or "").strip() or None
    out: list[dict[str, Any]] = []

    def add_entities(entities: list[str], start: int, end: int, pattern: str, confidence: float) -> None:
        if not entities:
            return
        evidence_text = _extract_evidence_text(text, start, end)
        for entity in entities:
            canonical = _normalize_party_value(entity)
            if not canonical:
                continue
            out.append(
                _build_fact(
                    doc_id=doc_id,
                    fact_type="party",
                    canonical_value=canonical,
                    raw_value=entity,
                    page=page,
                    chunk_id=chunk_id,
                    evidence_text=evidence_text,
                    confidence=confidence,
                    metadata={
                        "extractor": EXTRACTOR_VERSION,
                        "pattern": pattern,
                        "source_type": str(chunk.get("source_type") or ""),
                    },
                )
            )

    for match in PARTY_BETWEEN_RE.finditer(text):
        entities = _extract_entities(match.group("body"))
        add_entities(entities[:4], match.start(), match.end(), "between_clause", 0.92)

    for match in PARTY_LABEL_RE.finditer(text):
        entities = _extract_entities(match.group("body"))
        add_entities(entities[:2], match.start(), match.end(), "labeled_party", 0.88)

    return out


def extract_document_facts(doc_id: str, chunks: Iterable[dict[str, Any]]) -> List[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int | None, str]] = set()
    normalized_doc_id = str(doc_id or "").strip()
    if not normalized_doc_id:
        return []

    for chunk in chunks:
        chunk_doc_id = str(chunk.get("doc_id") or "").strip()
        if chunk_doc_id and chunk_doc_id != normalized_doc_id:
            continue
        text = str(chunk.get("content") or "").strip()
        if not text:
            continue
        extracted = []
        extracted.extend(_extract_date_facts(normalized_doc_id, chunk))
        extracted.extend(_extract_amount_facts(normalized_doc_id, chunk))
        extracted.extend(_extract_party_facts(normalized_doc_id, chunk))
        for fact in extracted:
            dedupe_key = (
                str(fact.get("fact_type") or ""),
                str(fact.get("canonical_value") or "").lower(),
                _to_int(fact.get("page")),
                _normalize_whitespace(fact.get("evidence_text") or "").lower(),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            facts.append(fact)
    return facts
