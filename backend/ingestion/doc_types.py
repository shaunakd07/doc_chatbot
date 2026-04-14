from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_&.\-]{1,}")

DOC_TYPE_HINTS: dict[str, list[str]] = {
    "nda": [
        "nda",
        "non disclosure agreement",
        "nondisclosure agreement",
        "confidentiality agreement",
        "mutual nda",
    ],
    "invoice": ["invoice", "tax invoice", "billing invoice", "bill"],
    "purchase_order": ["purchase order", "po", "p.o", "order form"],
    "quote": ["quote", "quotation", "vendor quote", "price quote"],
    "proposal": ["proposal", "business proposal", "technical proposal"],
    "contract": ["contract", "agreement", "master service agreement", "msa", "service agreement"],
    "statement_of_work": ["statement of work", "sow", "scope of work"],
    "receipt": ["receipt", "payment receipt", "cash receipt", "sales receipt"],
    "hr_form": [
        "hr form",
        "human resources form",
        "employee form",
        "onboarding form",
        "timesheet",
        "leave request",
        "w-4",
        "i-9",
    ],
    "presentation": ["presentation", "slide deck", "slides", "ppt", "pptx"],
    "spreadsheet": ["spreadsheet", "worksheet", "excel", "xlsx", "xls"],
    "report": ["report", "analysis report", "summary report"],
    "policy": ["policy", "procedure", "guideline"],
}

DOC_TYPE_LABELS: dict[str, str] = {
    "nda": "NDA",
    "invoice": "invoice",
    "purchase_order": "purchase order",
    "quote": "quote",
    "proposal": "proposal",
    "contract": "contract",
    "statement_of_work": "statement of work",
    "receipt": "receipt",
    "hr_form": "HR form",
    "presentation": "presentation",
    "spreadsheet": "spreadsheet",
    "report": "report",
    "policy": "policy",
    "unknown": "document",
}

_EXTRA_ALIASES: dict[str, str] = {
    "ndas": "nda",
    "invoices": "invoice",
    "purchase orders": "purchase_order",
    "purchaseorder": "purchase_order",
    "quotes": "quote",
    "quotations": "quote",
    "proposals": "proposal",
    "contracts": "contract",
    "agreements": "contract",
    "sows": "statement_of_work",
    "receipts": "receipt",
    "hr forms": "hr_form",
    "human resources": "hr_form",
    "human resources form": "hr_form",
    "employee forms": "hr_form",
    "onboarding": "hr_form",
    "w4": "hr_form",
    "i9": "hr_form",
    "slides": "presentation",
    "decks": "presentation",
    "presentations": "presentation",
    "spreadsheets": "spreadsheet",
    "worksheets": "spreadsheet",
    "reports": "report",
    "policies": "policy",
    "procedures": "policy",
}


def _normalize_text(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _singularize(token: str) -> str:
    raw = str(token or "").strip().lower()
    if len(raw) >= 5 and raw.endswith("ies"):
        return f"{raw[:-3]}y"
    if len(raw) >= 4 and raw.endswith("s") and not raw.endswith("ss"):
        return raw[:-1]
    return raw


def _tokenize(value: str) -> list[str]:
    tokens: list[str] = []
    for token in TOKEN_RE.findall(str(value or "").lower()):
        normalized = _singularize(token)
        if normalized:
            tokens.append(normalized)
    return tokens


def _alias_map() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for doc_type, hints in DOC_TYPE_HINTS.items():
        aliases[_normalize_text(doc_type.replace("_", " "))] = doc_type
        aliases[doc_type] = doc_type
        for hint in hints:
            normalized = _normalize_text(hint)
            if normalized:
                aliases[normalized] = doc_type
            hint_tokens = _tokenize(normalized)
            if hint_tokens:
                aliases[" ".join(hint_tokens)] = doc_type
    for alias, doc_type in _EXTRA_ALIASES.items():
        aliases[_normalize_text(alias)] = doc_type
        tokens = _tokenize(alias)
        if tokens:
            aliases[" ".join(tokens)] = doc_type
    return aliases


DOC_TYPE_ALIASES = _alias_map()


def infer_doc_type(
    filename: str,
    auto_tags: Iterable[str],
    text_samples: Iterable[str] | None = None,
) -> Tuple[str, float, Dict[str, float]]:
    text_samples = text_samples or []
    parts = [str(filename or "")]
    parts.extend(str(item or "") for item in auto_tags)
    for sample in text_samples:
        value = str(sample or "").strip()
        if value:
            parts.append(value[:420])
    corpus_text = _normalize_text(" ".join(parts))
    tokens = set(_tokenize(corpus_text))
    if not corpus_text and not tokens:
        return "unknown", 0.0, {}

    scores: dict[str, float] = {}
    for doc_type, hints in DOC_TYPE_HINTS.items():
        score = 0.0
        for hint in hints:
            normalized_hint = _normalize_text(hint)
            if not normalized_hint:
                continue
            hint_tokens = _tokenize(normalized_hint)
            if normalized_hint in corpus_text:
                score += 2.0 if len(hint_tokens) > 1 else 1.2
            if hint_tokens and all(token in tokens for token in hint_tokens):
                score += 1.2 if len(hint_tokens) > 1 else 0.65
        for token in _tokenize(doc_type.replace("_", " ")):
            if token in tokens:
                score += 0.6
        scores[doc_type] = score

    filename_lower = str(filename or "").lower()
    ext_bonus = {
        ".pptx": "presentation",
        ".ppt": "presentation",
        ".xlsx": "spreadsheet",
        ".xls": "spreadsheet",
    }
    for ext, doc_type in ext_bonus.items():
        if filename_lower.endswith(ext):
            scores[doc_type] = float(scores.get(doc_type, 0.0)) + 0.5
            break

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return "unknown", 0.0, {}
    best_type, best_score = ranked[0]
    second_score = float(ranked[1][1]) if len(ranked) > 1 else 0.0
    if best_score <= 0.35:
        return "unknown", 0.0, {}

    confidence = best_score / float(best_score + second_score + 0.75)
    confidence = max(0.05, min(0.99, confidence))
    top_scores = {doc_type: round(float(score), 3) for doc_type, score in ranked[:3] if float(score) > 0.0}
    return best_type, round(float(confidence), 4), top_scores


def extract_query_doc_type_candidates(question: str) -> List[str]:
    normalized = _normalize_text(question)
    if not normalized:
        return []
    tokens = set(_tokenize(normalized))
    found: set[str] = set()
    for alias, doc_type in DOC_TYPE_ALIASES.items():
        normalized_alias = _normalize_text(alias)
        if not normalized_alias:
            continue
        alias_tokens = _tokenize(normalized_alias)
        if normalized_alias in normalized:
            found.add(doc_type)
            continue
        if alias_tokens and all(token in tokens for token in alias_tokens):
            found.add(doc_type)
    return sorted(found)
