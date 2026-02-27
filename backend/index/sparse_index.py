from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from .. import storage

TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")


class SparseIndex:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: dict[str, dict] = {}
        self.postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.avg_doc_len = 0.0

    def _tokenize(self, text: str) -> List[str]:
        return [token.lower() for token in TOKEN_RE.findall(text or "")]

    def _recompute_avg_len(self) -> None:
        if not self.docs:
            self.avg_doc_len = 0.0
            return
        total_len = sum(int(doc["len"]) for doc in self.docs.values())
        self.avg_doc_len = total_len / float(len(self.docs))

    def load(self) -> None:
        self.docs = {}
        self.postings = defaultdict(list)
        self.doc_freq = defaultdict(int)
        for chunk in storage.list_chunks():
            self.add_chunk(chunk, recompute_avg_len=False)
        self._recompute_avg_len()

    def add_chunk(self, chunk: dict, recompute_avg_len: bool = True) -> bool:
        chunk_id = str(chunk.get("id") or "")
        if not chunk_id:
            return False
        tokens = self._tokenize(str(chunk.get("content") or ""))
        if not tokens:
            return False
        tf = Counter(tokens)
        doc_len = len(tokens)
        self.docs[chunk_id] = {
            "doc_id": chunk.get("doc_id"),
            "len": doc_len,
            "tf": tf,
        }
        for term, count in tf.items():
            self.postings[term].append((chunk_id, int(count)))
            self.doc_freq[term] += 1
        if recompute_avg_len:
            self._recompute_avg_len()
        return True

    def add_chunks(self, chunks: List[dict]) -> None:
        if not chunks:
            return
        changed = False
        for chunk in chunks:
            changed = self.add_chunk(chunk, recompute_avg_len=False) or changed
        if changed:
            self._recompute_avg_len()

    def _idf(self, term: str) -> float:
        n_docs = len(self.docs)
        if n_docs == 0:
            return 0.0
        df = int(self.doc_freq.get(term, 0))
        return math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5)))

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        if not query.strip() or not self.docs:
            return []
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
        avg_len = self.avg_doc_len if self.avg_doc_len > 0 else 1.0
        scores: Dict[str, float] = defaultdict(float)
        for term in query_terms:
            idf = self._idf(term)
            if idf <= 0:
                continue
            for chunk_id, term_tf in self.postings.get(term, []):
                doc = self.docs.get(chunk_id)
                if not doc:
                    continue
                doc_len = float(doc["len"]) or 1.0
                numer = term_tf * (self.k1 + 1.0)
                denom = term_tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_len))
                scores[chunk_id] += idf * (numer / denom)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[: max(1, top_k)]
