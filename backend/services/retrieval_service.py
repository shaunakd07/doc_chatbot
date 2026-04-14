from __future__ import annotations

from collections import defaultdict
import logging
from typing import List, Optional

from .. import storage

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
        self,
        embedder,
        vector_index,
        sparse_index=None,
        reranker=None,
        default_mode: str = "hybrid",
        rerank_top_n: int = 20,
    ) -> None:
        self.embedder = embedder
        self.vector_index = vector_index
        self.sparse_index = sparse_index
        self.reranker = reranker
        self.default_mode = default_mode
        self.rerank_top_n = max(1, int(rerank_top_n))

    def _attach_document_fields(self, chunk: dict, doc_cache: dict[str, dict]) -> dict:
        doc_id = chunk.get("doc_id")
        doc = doc_cache.get(doc_id)
        if doc is None and doc_id:
            doc = storage.get_document(doc_id) or {}
            doc_cache[doc_id] = doc
        chunk["doc_filename"] = (doc or {}).get("filename")
        chunk["doc_created_at"] = (doc or {}).get("created_at")
        return chunk

    def _rrf_fuse(
        self,
        dense_ranked: list[tuple[str, float]],
        sparse_ranked: list[tuple[str, float]],
        rrf_k: int = 60,
    ) -> list[tuple[str, float]]:
        scores: dict[str, float] = defaultdict(float)
        for rank, (chunk_id, _) in enumerate(dense_ranked, start=1):
            scores[chunk_id] += 1.0 / float(rrf_k + rank)
        for rank, (chunk_id, _) in enumerate(sparse_ranked, start=1):
            scores[chunk_id] += 1.0 / float(rrf_k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _raw_candidates(self, query: str, top_k: int, mode: str) -> list[tuple[str, float]]:
        effective_mode = (mode or self.default_mode or "hybrid").strip().lower()
        if effective_mode == "balanced":
            effective_mode = "hybrid"
        if effective_mode not in {"semantic", "sparse", "hybrid", "image_first"}:
            fallback_mode = (self.default_mode or "hybrid").strip().lower()
            effective_mode = fallback_mode if fallback_mode in {"semantic", "sparse", "hybrid", "image_first"} else "hybrid"
        oversample = max(top_k * 6, 50)
        dense_raw: list[tuple[str, float]] = []
        sparse_raw: list[tuple[str, float]] = []
        dense_error: Exception | None = None

        if effective_mode in {"semantic", "hybrid", "image_first"} and self.embedder is not None:
            try:
                query_vec = self.embedder.embed_query(query)
                dense_raw = self.vector_index.search(query_vec, top_k=oversample)
            except Exception as exc:
                dense_error = exc
                logger.warning("Dense retrieval failed; continuing with fallback path: %s", exc)
        if effective_mode in {"sparse", "hybrid", "image_first"} and self.sparse_index is not None:
            sparse_raw = self.sparse_index.search(query, top_k=oversample)

        if effective_mode == "semantic":
            if not dense_raw and sparse_raw:
                return sparse_raw
            if dense_error is not None and not dense_raw:
                raise dense_error
            return dense_raw
        if effective_mode == "sparse":
            return sparse_raw
        if dense_raw and not sparse_raw:
            return dense_raw
        if sparse_raw and not dense_raw:
            return sparse_raw
        if dense_error is not None and not dense_raw and not sparse_raw:
            raise dense_error
        return self._rrf_fuse(dense_raw, sparse_raw)

    def _hydrate_candidates(
        self,
        ranked: list[tuple[str, float]],
        top_k: int,
        doc_ids: Optional[List[str]] = None,
    ) -> List[dict]:
        results: List[dict] = []
        doc_cache: dict[str, dict] = {}
        for chunk_id, score in ranked:
            chunk = storage.get_chunk(chunk_id)
            if not chunk:
                continue
            if doc_ids and chunk.get("doc_id") not in doc_ids:
                continue
            chunk["score"] = float(score)
            results.append(self._attach_document_fields(chunk, doc_cache))
            if len(results) >= top_k:
                break
        return results

    def _maybe_rerank(self, query: str, chunks: List[dict], top_k: int) -> List[dict]:
        if not chunks:
            return chunks
        if self.reranker is None:
            return chunks[:top_k]
        rerank_pool = chunks[: max(top_k, self.rerank_top_n)]
        reranked = self.reranker.rerank(query, rerank_pool)
        return reranked[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[List[str]] = None,
        mode: Optional[str] = None,
        use_rerank: bool = True,
    ) -> List[dict]:
        if not query.strip():
            return []
        raw = self._raw_candidates(query, top_k=top_k, mode=mode or self.default_mode)
        hydrated = self._hydrate_candidates(raw, top_k=max(top_k, self.rerank_top_n), doc_ids=doc_ids)
        if use_rerank:
            return self._maybe_rerank(query, hydrated, top_k=top_k)
        return hydrated[:top_k]

    def search_balanced(
        self,
        query: str,
        top_k: int = 8,
        doc_ids: Optional[List[str]] = None,
        per_doc_limit: int = 4,
        mode: Optional[str] = None,
        use_rerank: bool = True,
    ) -> List[dict]:
        if not query.strip():
            return []
        oversample = max(top_k * 10, 80)
        raw = self._raw_candidates(query, top_k=oversample, mode=mode or self.default_mode)
        target_count = max(top_k, self.rerank_top_n) if use_rerank else top_k
        by_doc: dict[str, list[dict]] = defaultdict(list)
        doc_cache: dict[str, dict] = {}
        for chunk_id, score in raw:
            chunk = storage.get_chunk(chunk_id)
            if not chunk:
                continue
            doc_id = chunk.get("doc_id")
            if not doc_id:
                continue
            if doc_ids and doc_id not in doc_ids:
                continue
            if len(by_doc[doc_id]) >= per_doc_limit:
                continue
            chunk["score"] = float(score)
            by_doc[doc_id].append(self._attach_document_fields(chunk, doc_cache))

        if not by_doc:
            return []

        ranked_docs = sorted(
            by_doc.keys(),
            key=lambda doc_id: by_doc[doc_id][0].get("score", 0.0),
            reverse=True,
        )
        results: List[dict] = []
        idx = 0
        while len(results) < target_count:
            added = False
            for doc_id in ranked_docs:
                chunks = by_doc[doc_id]
                if idx < len(chunks):
                    results.append(chunks[idx])
                    added = True
                    if len(results) >= target_count:
                        break
            if not added:
                break
            idx += 1
        if not use_rerank:
            return results[:top_k]
        reranked = self._maybe_rerank(query, results, top_k=top_k)
        return reranked[:top_k]
