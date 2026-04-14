from __future__ import annotations

from datetime import datetime, timezone
import unittest

from backend.services.metadata_semantic_adapter import MetadataSemanticAdapter


def _doc(
    doc_id: str,
    *,
    filename: str,
    created_at: str,
    metadata: dict | None = None,
) -> dict:
    return {
        "id": doc_id,
        "filename": filename,
        "status": "ready",
        "created_at": created_at,
        "metadata": metadata or {},
    }


class _RetrievalStub:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search_balanced(
        self,
        query: str,
        top_k: int = 8,
        doc_ids=None,
        per_doc_limit: int = 4,
        mode: str | None = None,
        use_rerank: bool = True,
    ):
        self.queries.append(query)
        if "termination date" in query:
            return [
                {
                    "id": "expanded-1",
                    "doc_id": "doc-1",
                    "doc_filename": "NDA_1.pdf",
                    "doc_created_at": "2026-01-01",
                    "page": 1,
                    "chunk_index": 0,
                    "content": "Agreement termination date is 31 Dec 2028.",
                    "score": 0.91,
                    "source_type": "text",
                    "metadata": {},
                }
            ]
        return [
            {
                "id": "base-1",
                "doc_id": "doc-1",
                "doc_filename": "NDA_1.pdf",
                "doc_created_at": "2026-01-01",
                "page": 1,
                "chunk_index": 0,
                "content": "Agreement overview.",
                "score": 0.12,
                "source_type": "text",
                "metadata": {},
            }
        ]

    def search(self, query: str, top_k: int = 5, doc_ids=None, mode: str | None = None, use_rerank: bool = True):
        self.queries.append(query)
        return []


class MetadataSemanticAdapterTests(unittest.TestCase):
    def test_filter_documents_moves_metadata_logic_out_of_chat_service(self) -> None:
        adapter = MetadataSemanticAdapter(_RetrievalStub())
        docs = [
            _doc(
                "nda-1",
                filename="NDA_1.pdf",
                created_at="2026-01-01T00:00:00+00:00",
                metadata={"doc_type": "nda", "author": "Alice Johnson"},
            ),
            _doc(
                "msa-1",
                filename="MSA_1.pdf",
                created_at="2025-01-01T00:00:00+00:00",
                metadata={"doc_type": "contract", "author": "Bob Smith"},
            ),
        ]

        filtered = adapter.filter_documents(
            docs,
            operation="count",
            filters={"doc_type": "nda", "author": "alice", "date_from": "2026-01-01", "date_to": "2026-12-31"},
            parse_datetime=lambda value: (
                datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=timezone.utc)
                if value and len(str(value)) <= 10
                else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                if value
                else None
            ),
            doc_created_at=lambda doc: datetime.fromisoformat(str(doc.get("created_at")).replace("Z", "+00:00")),
            doc_updated_at=lambda doc: datetime.fromisoformat(str(doc.get("created_at")).replace("Z", "+00:00")),
            doc_matches_doc_type_filter=lambda doc, raw_filter: raw_filter == str(
                (doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}).get("doc_type") or ""
            ).lower(),
            doc_matches_target_document=lambda doc, raw_filter: raw_filter in str(doc.get("filename") or "").lower(),
            now=datetime(2026, 3, 9, tzinfo=timezone.utc),
        )

        self.assertEqual(["nda-1"], [doc["id"] for doc in filtered])

    def test_search_chunks_with_expansion_uses_adapter_fallback_query(self) -> None:
        retrieval = _RetrievalStub()
        adapter = MetadataSemanticAdapter(retrieval)

        chunks, score, did_retry = adapter.search_chunks_with_expansion(
            query="Which nda agreements are expired?",
            doc_ids=["doc-1"],
            top_k=6,
            per_doc_limit=2,
            mode="hybrid",
            semantic_terms=["expired", "termination"],
            min_evidence_score=0.6,
            evidence_scorer=lambda _question, rows: max(float(row.get("score", 0.0) or 0.0) for row in rows) if rows else 0.0,
            merge_chunks=lambda rows, limit: sorted(rows, key=lambda row: float(row.get("score", 0.0) or 0.0), reverse=True)[:limit],
            merged_limit=12,
            fallback_expansion_terms=["agreement effective date execution date termination date expiry date"],
        )

        self.assertTrue(did_retry)
        self.assertGreater(score, 0.6)
        self.assertTrue(any("termination date" in query for query in retrieval.queries[1:]))
        self.assertEqual("expanded-1", chunks[0]["id"])


if __name__ == "__main__":
    unittest.main()
