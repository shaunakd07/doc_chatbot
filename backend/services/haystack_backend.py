from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .. import config, storage

logger = logging.getLogger(__name__)

try:
    from haystack import Document as HaystackDocument
    from haystack.components.builders import PromptBuilder
    from haystack.components.joiners import DocumentJoiner
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    HAYSTACK_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency path
    HaystackDocument = None
    PromptBuilder = None
    DocumentJoiner = None
    InMemoryBM25Retriever = None
    InMemoryEmbeddingRetriever = None
    InMemoryDocumentStore = None
    HAYSTACK_IMPORT_ERROR = exc


class HaystackSearchBackend:
    def __init__(
        self,
        retrieval_service,
        *,
        expansion_model=None,
    ) -> None:
        self.retrieval = retrieval_service
        self.expansion_model = expansion_model
        self.enabled = bool(config.ENABLE_HAYSTACK_RETRIEVAL)
        self.enable_query_expansion = bool(config.ENABLE_HAYSTACK_QUERY_EXPANSION)
        self._prompt_builder = None
        if PromptBuilder is not None:
            self._prompt_builder = PromptBuilder(
                template=(
                    "You expand document retrieval queries for legal and commercial documents.\n"
                    "Return only one rewritten search query.\n"
                    "Rules:\n"
                    "- Keep the original intent.\n"
                    "- Keep named parties, dates, and document type hints.\n"
                    "- Prefer terms that improve retrieval recall.\n"
                    "- Do not explain your answer.\n\n"
                    "Original question: {{ question }}\n"
                    "Semantic hints: {{ semantic_hint_text }}\n"
                    "Expanded query:"
                )
            )

    @property
    def available(self) -> bool:
        return bool(
            self.enabled
            and HaystackDocument is not None
            and InMemoryDocumentStore is not None
            and DocumentJoiner is not None
            and self.retrieval is not None
        )

    def _scoped_chunks(self, doc_ids: Optional[list[str]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        all_chunks = storage.list_chunks()
        if doc_ids:
            scope = {str(doc_id or "").strip() for doc_id in doc_ids if str(doc_id or "").strip()}
            chunks = [chunk for chunk in all_chunks if str(chunk.get("doc_id") or "").strip() in scope]
        else:
            chunks = list(all_chunks)
        doc_ids_for_chunks = {str(chunk.get("doc_id") or "").strip() for chunk in chunks if str(chunk.get("doc_id") or "").strip()}
        docs = {
            str(doc.get("id") or "").strip(): doc
            for doc in storage.list_documents()
            if str(doc.get("id") or "").strip() in doc_ids_for_chunks
        }
        return chunks, docs

    def _embedding_map(self) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for chunk_id, blob, _dim in storage.load_embeddings():
            if not blob:
                continue
            vector = np.frombuffer(blob, dtype="float32")
            if vector.size == 0:
                continue
            out[str(chunk_id)] = vector.astype("float32").tolist()
        return out

    def _build_document_store(self, doc_ids: Optional[list[str]]) -> tuple[Any, list[str]]:
        if not self.available:
            raise RuntimeError("Haystack retrieval is not available")
        chunks, docs = self._scoped_chunks(doc_ids)
        if not chunks:
            store = InMemoryDocumentStore()
            return store, []
        embeddings = self._embedding_map()
        haystack_docs: list[Any] = []
        chunk_ids: list[str] = []
        for chunk in chunks:
            chunk_id = str(chunk.get("id") or "").strip()
            if not chunk_id:
                continue
            doc_id = str(chunk.get("doc_id") or "").strip()
            doc = docs.get(doc_id) or {}
            metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
            haystack_docs.append(
                HaystackDocument(
                    id=chunk_id,
                    content=str(chunk.get("content") or ""),
                    embedding=embeddings.get(chunk_id),
                    meta={
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "page": chunk.get("page"),
                        "chunk_index": chunk.get("chunk_index"),
                        "source_type": chunk.get("source_type"),
                        "chunk_metadata": metadata,
                        "doc_filename": doc.get("filename"),
                        "doc_created_at": doc.get("created_at"),
                    },
                )
            )
            chunk_ids.append(chunk_id)
        store = InMemoryDocumentStore()
        if haystack_docs:
            store.write_documents(haystack_docs)
        return store, chunk_ids

    def _to_chunk(self, item: Any) -> dict[str, Any]:
        meta = getattr(item, "meta", None) or {}
        return {
            "id": str(meta.get("chunk_id") or getattr(item, "id", "")),
            "doc_id": str(meta.get("doc_id") or ""),
            "doc_filename": meta.get("doc_filename"),
            "doc_created_at": meta.get("doc_created_at"),
            "page": meta.get("page"),
            "chunk_index": meta.get("chunk_index"),
            "content": str(getattr(item, "content", "") or ""),
            "score": float(getattr(item, "score", 0.0) or 0.0),
            "source_type": str(meta.get("source_type") or "text"),
            "metadata": meta.get("chunk_metadata") if isinstance(meta.get("chunk_metadata"), dict) else {},
        }

    def _balance_chunks(self, items: list[Any], *, per_doc_limit: int, top_k: int) -> list[dict[str, Any]]:
        by_doc: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            chunk = self._to_chunk(item)
            doc_id = str(chunk.get("doc_id") or "").strip()
            if not doc_id:
                continue
            bucket = by_doc.setdefault(doc_id, [])
            if len(bucket) >= per_doc_limit:
                continue
            bucket.append(chunk)
        if not by_doc:
            return []
        ranked_doc_ids = sorted(
            by_doc.keys(),
            key=lambda doc_id: float(by_doc[doc_id][0].get("score", 0.0) or 0.0),
            reverse=True,
        )
        target_count = max(top_k, int(getattr(self.retrieval, "rerank_top_n", top_k) or top_k))
        results: list[dict[str, Any]] = []
        cursor = 0
        while len(results) < target_count:
            added = False
            for doc_id in ranked_doc_ids:
                bucket = by_doc.get(doc_id) or []
                if cursor < len(bucket):
                    results.append(bucket[cursor])
                    added = True
                    if len(results) >= target_count:
                        break
            if not added:
                break
            cursor += 1
        return results[:top_k]

    def search_chunks(
        self,
        *,
        query: str,
        doc_ids: Optional[list[str]],
        top_k: int,
        per_doc_limit: int,
        mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        if not self.available or not str(query or "").strip():
            return []
        store, chunk_ids = self._build_document_store(doc_ids)
        if not chunk_ids:
            return []
        effective_mode = (mode or "hybrid").strip().lower()
        if effective_mode == "balanced":
            effective_mode = "hybrid"
        if effective_mode == "image_first":
            effective_mode = "hybrid"
        oversample = max(int(top_k) * 6, 40)
        documents_by_source: list[list[Any]] = []
        if effective_mode in {"hybrid", "sparse"}:
            bm25 = InMemoryBM25Retriever(document_store=store)
            bm25_result = bm25.run(query=str(query), top_k=oversample)
            documents_by_source.append(list(bm25_result.get("documents") or []))
        if effective_mode in {"hybrid", "semantic"} and getattr(self.retrieval, "embedder", None) is not None:
            try:
                query_embedding = np.asarray(self.retrieval.embedder.embed_query(str(query)), dtype="float32")
                dense = InMemoryEmbeddingRetriever(document_store=store)
                dense_result = dense.run(query_embedding=query_embedding.astype("float32").tolist(), top_k=oversample)
                documents_by_source.append(list(dense_result.get("documents") or []))
            except Exception as exc:
                logger.warning("Haystack dense retrieval failed; continuing with available paths: %s", exc)
        if not documents_by_source:
            return []
        if len(documents_by_source) == 1:
            ranked = documents_by_source[0]
        else:
            joined = DocumentJoiner(join_mode="reciprocal_rank_fusion")
            ranked = list(joined.run(documents=documents_by_source).get("documents") or [])
        balanced = self._balance_chunks(
            ranked,
            per_doc_limit=max(1, int(per_doc_limit)),
            top_k=max(1, int(top_k)),
        )
        reranker = getattr(self.retrieval, "_maybe_rerank", None)
        if callable(reranker) and balanced:
            return reranker(str(query), balanced, top_k=max(1, int(top_k)))
        return balanced[: max(1, int(top_k))]

    def expand_query(self, question: str, semantic_terms: list[str]) -> str:
        if not (
            self.available
            and self.enable_query_expansion
            and self.expansion_model is not None
            and self._prompt_builder is not None
        ):
            return ""
        semantic_hint_text = ", ".join([term for term in semantic_terms[:10] if str(term).strip()]) or "none"
        try:
            prompt_payload = self._prompt_builder.run(question=str(question or "").strip(), semantic_hint_text=semantic_hint_text)
            prompt = str(prompt_payload.get("prompt") or "").strip()
            if not prompt:
                return ""
            rewritten = str(self.expansion_model.generate_text(prompt, max_new_tokens=160) or "").strip()
            rewritten = rewritten.splitlines()[0].strip(" \t\r\n\"'`")
            return rewritten if len(rewritten) >= 8 else ""
        except Exception as exc:
            logger.warning("Haystack query expansion failed; using fallback query expansion: %s", exc)
            return ""
