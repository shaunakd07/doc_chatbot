from __future__ import annotations

import re
from collections import Counter
from pathlib import PurePosixPath
from typing import Iterable

TAG_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]{1,}")
TAG_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "pdf",
    "doc",
    "docx",
    "ppt",
    "pptx",
    "xlsx",
    "xls",
    "txt",
    "copy",
    "final",
    "version",
}


def _normalize_tag(raw: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", str(raw or "").lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) < 2:
        return ""
    return cleaned


def normalize_tags(raw_tags: Iterable[object], limit: int = 32) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_tags:
        value = _normalize_tag(str(item or ""))
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
        if len(normalized) >= max(1, int(limit)):
            break
    return normalized


def _tokenize(value: str) -> list[str]:
    return [token.lower() for token in TAG_TOKEN_RE.findall(str(value or ""))]


def _append_tag(tags: list[str], seen: set[str], raw: str, limit: int) -> None:
    if len(tags) >= limit:
        return
    tag = _normalize_tag(raw)
    if not tag or tag in seen:
        return
    seen.add(tag)
    tags.append(tag)


def extract_filename_tags(filename: str, limit: int = 14) -> list[str]:
    limit = max(1, int(limit))
    raw = str(filename or "").replace("\\", "/")
    path = PurePosixPath(raw)
    parts = [part for part in path.parts if part not in {"", ".", ".."}]
    tags: list[str] = []
    seen: set[str] = set()

    for idx, part in enumerate(parts):
        candidate = part
        if idx == len(parts) - 1:
            candidate = re.sub(r"\.[A-Za-z0-9]+$", "", candidate)
        normalized_part = _normalize_tag(candidate.replace("_", " ").replace("-", " "))
        if not normalized_part:
            continue
        words = [word for word in normalized_part.split() if len(word) >= 2]
        if not words:
            continue
        if 1 < len(words) <= 5:
            _append_tag(tags, seen, " ".join(words), limit)
        for word in words:
            if word in TAG_STOPWORDS:
                continue
            _append_tag(tags, seen, word, limit)
        if len(tags) >= limit:
            break
    return tags[:limit]


def extract_text_tags(text_samples: Iterable[str], limit: int = 20) -> list[str]:
    limit = max(1, int(limit))
    unigram_counts: Counter[str] = Counter()
    bigram_counts: Counter[str] = Counter()

    for raw in text_samples:
        text = str(raw or "").strip()
        if not text:
            continue
        tokens = [token for token in _tokenize(text) if len(token) >= 3 and token not in TAG_STOPWORDS]
        if not tokens:
            continue
        unigram_counts.update(tokens[:320])
        for idx in range(len(tokens) - 1):
            first = tokens[idx]
            second = tokens[idx + 1]
            if first == second:
                continue
            bigram_counts[f"{first} {second}"] += 1

    tags: list[str] = []
    seen: set[str] = set()
    for token, _ in unigram_counts.most_common(limit):
        _append_tag(tags, seen, token, limit)
        if len(tags) >= limit:
            return tags
    for phrase, count in bigram_counts.most_common(limit):
        if count < 2:
            continue
        _append_tag(tags, seen, phrase, limit)
        if len(tags) >= limit:
            return tags
    return tags


def build_document_auto_tags(
    filename: str,
    text_samples: Iterable[str],
    limit: int = 28,
) -> list[str]:
    limit = max(1, int(limit))
    tags: list[str] = []
    seen: set[str] = set()

    for tag in extract_filename_tags(filename, limit=max(6, limit // 2)):
        _append_tag(tags, seen, tag, limit)
    for tag in extract_text_tags(text_samples, limit=limit):
        _append_tag(tags, seen, tag, limit)
    return tags[:limit]
