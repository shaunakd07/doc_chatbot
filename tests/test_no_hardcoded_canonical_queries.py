from __future__ import annotations

import re
from pathlib import Path


CANONICAL_PHRASES = (
    "which ndas are expired",
    "ndas were signed in 2018",
    "written in 2015",
    "what services does lakerunner provide to airtel",
)


def _normalize_text(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _canonical_ngrams(phrase: str, n: int = 4) -> set[str]:
    tokens = _normalize_text(phrase).split()
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[idx : idx + n]) for idx in range(0, len(tokens) - n + 1)}


def test_non_test_sources_do_not_embed_canonical_question_literals() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    backend_root = repo_root / "backend"
    python_files = sorted(path for path in backend_root.rglob("*.py") if path.is_file())
    forbidden_ngrams: set[str] = set()
    for phrase in CANONICAL_PHRASES:
        forbidden_ngrams.update(_canonical_ngrams(phrase, n=4))
        forbidden_ngrams.update(_canonical_ngrams(phrase, n=5))
    for file_path in python_files:
        normalized = _normalize_text(file_path.read_text(encoding="utf-8", errors="ignore"))
        for phrase in CANONICAL_PHRASES:
            normalized_phrase = _normalize_text(phrase)
            assert normalized_phrase not in normalized, f"forbidden canonical phrase in {file_path}"
        for ngram in forbidden_ngrams:
            if not ngram or len(ngram.split()) < 4:
                continue
            assert ngram not in normalized, f"forbidden canonical n-gram in {file_path}: '{ngram}'"
