from __future__ import annotations

from typing import Any, Callable, Optional

from .haystack_backend import HaystackSearchBackend
from .metadata_filters import filter_documents_by_metadata
from .. import storage


class MetadataSemanticAdapter:
    def __init__(self, retrieval_service, *, expansion_model=None) -> None:
        self.retrieval = retrieval_service
        self.haystack = HaystackSearchBackend(retrieval_service, expansion_model=expansion_model)

    def list_documents(
        self,
        *,
        doc_ids: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        docs = storage.list_documents()
        if not doc_ids:
            return list(docs)
        scoped = {str(doc_id or "").strip() for doc_id in doc_ids if str(doc_id or "").strip()}
        if not scoped:
            return list(docs)
        return [doc for doc in docs if str(doc.get("id") or "").strip() in scoped]

    def filter_documents(
        self,
        docs: list[dict[str, Any]],
        *,
        operation: str,
        filters: dict[str, Any],
        parse_datetime: Callable[[Any], Any],
        doc_created_at: Callable[[dict[str, Any]], Any],
        doc_updated_at: Callable[[dict[str, Any]], Any],
        doc_matches_doc_type_filter: Callable[[dict[str, Any], str], bool],
        doc_matches_target_document: Callable[[dict[str, Any], str], bool],
        now=None,
    ) -> list[dict[str, Any]]:
        return filter_documents_by_metadata(
            docs,
            operation=operation,
            filters=filters,
            parse_datetime=parse_datetime,
            doc_created_at=doc_created_at,
            doc_updated_at=doc_updated_at,
            doc_matches_doc_type_filter=doc_matches_doc_type_filter,
            doc_matches_target_document=doc_matches_target_document,
            now=now,
        )

    def search_chunks(
        self,
        *,
        query: str,
        doc_ids: Optional[list[str]],
        top_k: int,
        per_doc_limit: int,
        mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        if not str(query or "").strip():
            return []
        if self.haystack.available:
            try:
                chunks = self.haystack.search_chunks(
                    query=query,
                    doc_ids=doc_ids,
                    top_k=max(1, int(top_k)),
                    per_doc_limit=max(1, int(per_doc_limit)),
                    mode=mode,
                )
                if chunks:
                    return chunks
            except Exception:
                pass
        try:
            return self.retrieval.search_balanced(
                query,
                top_k=max(1, int(top_k)),
                doc_ids=doc_ids,
                per_doc_limit=max(1, int(per_doc_limit)),
                mode=mode,
            )
        except Exception:
            return self.retrieval.search(
                query,
                top_k=max(1, int(top_k)),
                doc_ids=doc_ids,
                mode=mode,
            )

    def search_chunks_with_expansion(
        self,
        *,
        query: str,
        doc_ids: Optional[list[str]],
        top_k: int,
        per_doc_limit: int,
        mode: str,
        semantic_terms: Optional[list[str]],
        min_evidence_score: float,
        evidence_scorer: Callable[[str, list[dict[str, Any]]], float],
        merge_chunks: Callable[[list[dict[str, Any]], int], list[dict[str, Any]]],
        merged_limit: int,
        fallback_expansion_terms: Optional[list[str]] = None,
    ) -> tuple[list[dict[str, Any]], float, bool]:
        base_chunks = self.search_chunks(
            query=query,
            doc_ids=doc_ids,
            top_k=top_k,
            per_doc_limit=per_doc_limit,
            mode=mode,
        )
        base_score = float(evidence_scorer(query, base_chunks))
        if base_score >= float(min_evidence_score):
            return base_chunks, base_score, False

        expanded_query = self.haystack.expand_query(query, list(semantic_terms or []))
        if not expanded_query:
            expansion_seed = " ".join([str(term).strip() for term in (semantic_terms or [])[:8] if str(term).strip()]).strip()
            extra_terms = " ".join([str(term).strip() for term in (fallback_expansion_terms or [])[:2] if str(term).strip()]).strip()
            expanded_query = " ".join(part for part in (query, expansion_seed, extra_terms) if str(part).strip()).strip()
        if not expanded_query or expanded_query.strip().lower() == str(query or "").strip().lower():
            return base_chunks, base_score, False

        retry_chunks = self.search_chunks(
            query=expanded_query,
            doc_ids=doc_ids,
            top_k=max(int(top_k), 16),
            per_doc_limit=max(int(per_doc_limit), 2),
            mode=mode,
        )
        merged = merge_chunks(base_chunks + retry_chunks, limit=max(1, int(merged_limit)))
        merged_score = float(evidence_scorer(query, merged))
        return merged, merged_score, True
