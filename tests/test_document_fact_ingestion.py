from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from backend import config, storage
from backend.ingestion.document_facts import extract_document_facts
from backend.ingestion.pipeline import ingest_file
from backend.services.chat_service import ChatService


class _DummyEmbedder:
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype="float32")
        base = np.asarray([[0.5, 0.5, 0.70710677]], dtype="float32")
        return np.repeat(base, len(texts), axis=0)


class _DummyVectorIndex:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], list[str]]] = []

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        self.calls.append((tuple(vectors.shape), list(ids)))


class _DummySparseIndex:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def add_chunks(self, chunks: list[dict]) -> None:
        self.calls.append(len(chunks))


class _RecordingRetrieval:
    def __init__(self, *, search_chunks: list[dict] | None = None, balanced_chunks: list[dict] | None = None) -> None:
        self.search_chunks = list(search_chunks or [])
        self.balanced_chunks = list(balanced_chunks or [])
        self.calls: list[tuple] = []

    def search(self, query: str, top_k: int = 5, doc_ids=None, mode: str | None = None, use_rerank: bool = True):
        self.calls.append(("search", query, top_k, tuple(doc_ids or []), mode, use_rerank))
        return [dict(item) for item in self.search_chunks[:top_k]]

    def search_balanced(
        self,
        query: str,
        top_k: int = 8,
        doc_ids=None,
        per_doc_limit: int = 4,
        mode: str | None = None,
        use_rerank: bool = True,
    ):
        self.calls.append(("search_balanced", query, top_k, tuple(doc_ids or []), per_doc_limit, mode, use_rerank))
        return [dict(item) for item in self.balanced_chunks[:top_k]]


class _SQLiteStorageTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.temp_path = Path(self.temp_dir.name)
        self.stack = ExitStack()

        self.data_dir = self.temp_path / "data"
        self.upload_dir = self.data_dir / "uploads"
        self.processed_dir = self.data_dir / "processed"
        self.index_dir = self.data_dir / "index"
        self.db_dir = self.data_dir / "db"
        self.db_path = self.db_dir / "app.db"

        self.stack.enter_context(patch.object(config, "DB_BACKEND", "sqlite"))
        self.stack.enter_context(patch.object(config, "DATA_DIR", self.data_dir))
        self.stack.enter_context(patch.object(config, "UPLOAD_DIR", self.upload_dir))
        self.stack.enter_context(patch.object(config, "PROCESSED_DIR", self.processed_dir))
        self.stack.enter_context(patch.object(config, "INDEX_DIR", self.index_dir))
        self.stack.enter_context(patch.object(config, "DB_DIR", self.db_dir))
        self.stack.enter_context(patch.object(config, "SQLITE_DB_PATH", self.db_path))
        self.stack.enter_context(patch.object(storage, "DB_PATH", self.db_path))

        storage.init_db()

    def tearDown(self) -> None:
        self.stack.close()
        self.temp_dir.cleanup()
        super().tearDown()


class DocumentFactExtractionTests(unittest.TestCase):
    def test_extract_document_facts_extracts_date_amount_and_parties(self) -> None:
        chunks = [
            {
                "id": "chunk-1",
                "doc_id": "doc-1",
                "page": 2,
                "chunk_index": 0,
                "content": (
                    "This confidentiality agreement is made between Neptune Systems Pte Ltd "
                    "and Orion Data Labs Ltd on 14 March 2015 for SGD 12,000."
                ),
                "source_type": "text",
                "metadata": {},
            }
        ]

        facts = extract_document_facts("doc-1", chunks)

        self.assertTrue(any(fact["fact_type"] == "date" and fact["canonical_value"] == "2015-03-14" for fact in facts))
        self.assertTrue(any(fact["fact_type"] == "amount" and fact["canonical_value"] == "SGD 12000" for fact in facts))
        parties = {fact["canonical_value"] for fact in facts if fact["fact_type"] == "party"}
        self.assertIn("Neptune Systems Pte Ltd", parties)
        self.assertIn("Orion Data Labs Ltd", parties)
        self.assertTrue(all(fact["page"] == 2 for fact in facts))
        self.assertTrue(all(fact["chunk_id"] == "chunk-1" for fact in facts))
        self.assertTrue(
            all("Neptune Systems Pte Ltd" in fact["evidence_text"] or "14 March 2015" in fact["evidence_text"] for fact in facts)
        )


class DocumentFactStorageTests(_SQLiteStorageTestCase):
    def test_document_facts_persist_and_round_trip(self) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        storage.add_document("doc-1", "sample.txt", "ready", created_at, metadata={})
        storage.add_chunks(
            [
                {
                    "id": "chunk-1",
                    "doc_id": "doc-1",
                    "page": 1,
                    "chunk_index": 0,
                    "content": "Agreement dated 2024-01-31 for USD 9,500 between Alpha Corp and Beta LLC.",
                    "source_type": "text",
                    "metadata": {},
                }
            ]
        )
        storage.add_document_facts(
            [
                {
                    "id": "fact-1",
                    "doc_id": "doc-1",
                    "fact_type": "date",
                    "canonical_value": "2024-01-31",
                    "raw_value": "2024-01-31",
                    "page": 1,
                    "chunk_id": "chunk-1",
                    "evidence_text": "Agreement dated 2024-01-31 for USD 9,500.",
                    "confidence": 0.99,
                    "metadata": {"extractor": "unit-test"},
                }
            ]
        )

        facts = storage.list_document_facts(doc_ids=["doc-1"])

        self.assertEqual(1, len(facts))
        self.assertEqual("doc-1", facts[0]["doc_id"])
        self.assertEqual("date", facts[0]["fact_type"])
        self.assertEqual("chunk-1", facts[0]["chunk_id"])
        self.assertEqual(1, facts[0]["page"])
        self.assertEqual("unit-test", facts[0]["metadata"]["extractor"])


class DocumentFactIngestionWiringTests(_SQLiteStorageTestCase):
    def test_ingest_file_persists_document_facts_with_provenance(self) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        source_file = self.temp_path / "agreement.txt"
        source_file.write_text(
            (
                "This confidentiality agreement is made between Alpha Corp and Beta LLC on 14 March 2015. "
                "Total contract value is SGD 12,000."
            ),
            encoding="utf-8",
        )
        storage.add_document(
            "doc-1",
            "agreement.txt",
            "queued",
            created_at,
            metadata={
                "document_id": "doc-1",
                "created_at": created_at,
                "uploaded_at": created_at,
                "updated_at": created_at,
            },
        )

        vector_index = _DummyVectorIndex()
        sparse_index = _DummySparseIndex()
        ingest_file(
            source_file,
            "doc-1",
            _DummyEmbedder(),
            vector_index,
            sparse_index=sparse_index,
        )

        chunks = storage.get_chunks_by_doc("doc-1")
        chunk_ids = {str(chunk["id"]) for chunk in chunks}
        facts = storage.list_document_facts(doc_ids=["doc-1"])
        fact_types = {fact["fact_type"] for fact in facts}

        self.assertTrue({"date", "amount", "party"}.issubset(fact_types))
        self.assertTrue(all(fact["doc_id"] == "doc-1" for fact in facts))
        self.assertTrue(all(fact["page"] == 1 for fact in facts))
        self.assertTrue(all(str(fact["chunk_id"]) in chunk_ids for fact in facts))
        self.assertTrue(all(str(fact["evidence_text"]).strip() for fact in facts))

        amount_facts = [fact for fact in facts if fact["fact_type"] == "amount"]
        self.assertEqual("SGD 12000", amount_facts[0]["canonical_value"])

        party_values = {fact["canonical_value"] for fact in facts if fact["fact_type"] == "party"}
        self.assertEqual({"Alpha Corp", "Beta LLC"}, party_values)

        updated_doc = storage.get_document("doc-1")
        metadata = updated_doc.get("metadata") if isinstance(updated_doc, dict) else {}
        self.assertEqual(len(facts), int((metadata or {}).get("ingest_fact_count") or 0))
        self.assertEqual("ready", updated_doc["status"])
        self.assertTrue(vector_index.calls)
        self.assertTrue(sparse_index.calls)


class DocumentFactExactLookupTests(_SQLiteStorageTestCase):
    def _add_document_with_fact(
        self,
        *,
        doc_id: str,
        filename: str,
        chunk_id: str,
        chunk_content: str,
        fact_id: str,
        fact_type: str,
        canonical_value: str,
        raw_value: str,
        evidence_text: str,
    ) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        storage.add_document(doc_id, filename, "ready", created_at, metadata={})
        storage.add_chunks(
            [
                {
                    "id": chunk_id,
                    "doc_id": doc_id,
                    "page": 1,
                    "chunk_index": 0,
                    "content": chunk_content,
                    "source_type": "text",
                    "metadata": {},
                }
            ]
        )
        storage.add_document_facts(
            [
                {
                    "id": fact_id,
                    "doc_id": doc_id,
                    "fact_type": fact_type,
                    "canonical_value": canonical_value,
                    "raw_value": raw_value,
                    "page": 1,
                    "chunk_id": chunk_id,
                    "evidence_text": evidence_text,
                    "confidence": 0.99,
                    "metadata": {"extractor": "unit-test"},
                }
            ]
        )

    def test_direct_fact_question_uses_document_facts_before_retrieval(self) -> None:
        self._add_document_with_fact(
            doc_id="doc-1",
            filename="agreement.txt",
            chunk_id="chunk-1",
            chunk_content="Agreement date is 14 March 2015.",
            fact_id="fact-1",
            fact_type="date",
            canonical_value="2015-03-14",
            raw_value="14 March 2015",
            evidence_text="Agreement date is 14 March 2015.",
        )
        retrieval = _RecordingRetrieval()
        service = ChatService(retrieval, model=None, enable_vlm=False, router=None)

        response = service.answer("What date is this agreement dated?", doc_ids=["doc-1"], top_k=5)

        self.assertTrue(response["route"]["exact_lookup"]["applicable"])
        self.assertEqual("strong", response["route"]["exact_lookup"]["strength"])
        self.assertFalse(response["route"]["exact_lookup"]["used_hybrid_fallback"])
        self.assertFalse(retrieval.calls)
        self.assertIn("2015", response["answer"])
        self.assertTrue(response["sources"])
        self.assertEqual("document_fact", response["sources"][0]["source_type"])
        self.assertEqual("doc-1", response["sources"][0]["doc_id"])

    def test_exact_amount_question_returns_grounded_fact_source(self) -> None:
        self._add_document_with_fact(
            doc_id="doc-amount",
            filename="pricing_schedule.txt",
            chunk_id="chunk-amount",
            chunk_content="Agreement amount is USD 9,500.",
            fact_id="fact-amount",
            fact_type="amount",
            canonical_value="USD 9500",
            raw_value="USD 9,500",
            evidence_text="Agreement amount is USD 9,500.",
        )
        retrieval = _RecordingRetrieval()
        service = ChatService(retrieval, model=None, enable_vlm=False, router=None)

        response = service.answer("What is the contract value?", doc_ids=["doc-amount"], top_k=5)

        self.assertTrue(response["route"]["exact_lookup"]["applicable"])
        self.assertFalse(retrieval.calls)
        self.assertTrue(response["sources"])
        self.assertEqual("document_fact", response["sources"][0]["source_type"])
        self.assertIn("USD 9,500", response["sources"][0]["content"])
        self.assertIn("Agreement amount is USD 9,500.", response["sources"][0]["content"])

    def test_document_contains_question_prefers_fact_index(self) -> None:
        self._add_document_with_fact(
            doc_id="doc-a",
            filename="nda_alpha.txt",
            chunk_id="chunk-a",
            chunk_content="Agreement between Neptune Systems Pte Ltd and Orion Data Labs Ltd.",
            fact_id="fact-a",
            fact_type="party",
            canonical_value="Orion Data Labs Ltd",
            raw_value="Orion Data Labs Ltd",
            evidence_text="Agreement between Neptune Systems Pte Ltd and Orion Data Labs Ltd.",
        )
        self._add_document_with_fact(
            doc_id="doc-b",
            filename="nda_beta.txt",
            chunk_id="chunk-b",
            chunk_content="Agreement between Alpha Corp and Beta LLC.",
            fact_id="fact-b",
            fact_type="party",
            canonical_value="Beta LLC",
            raw_value="Beta LLC",
            evidence_text="Agreement between Alpha Corp and Beta LLC.",
        )
        retrieval = _RecordingRetrieval()
        service = ChatService(retrieval, model=None, enable_vlm=False, router=None)

        response = service.answer("Which document contains Orion Data Labs Ltd?", doc_ids=None, top_k=5)

        self.assertTrue(response["route"]["exact_lookup"]["applicable"])
        self.assertEqual("document_contains", response["route"]["exact_lookup"]["mode"])
        self.assertEqual("strong", response["route"]["exact_lookup"]["strength"])
        self.assertFalse(response["route"]["exact_lookup"]["used_hybrid_fallback"])
        self.assertFalse(retrieval.calls)
        source_doc_ids = {str(source.get("doc_id") or "") for source in response["sources"]}
        self.assertEqual({"doc-a"}, source_doc_ids)
        self.assertIn("Orion Data Labs Ltd", response["answer"])

    def test_document_contains_answer_names_only_matching_document(self) -> None:
        self._add_document_with_fact(
            doc_id="doc-a",
            filename="nda_alpha.txt",
            chunk_id="chunk-a",
            chunk_content="Agreement between Neptune Systems Pte Ltd and Orion Data Labs Ltd.",
            fact_id="fact-a",
            fact_type="party",
            canonical_value="Orion Data Labs Ltd",
            raw_value="Orion Data Labs Ltd",
            evidence_text="Agreement between Neptune Systems Pte Ltd and Orion Data Labs Ltd.",
        )
        self._add_document_with_fact(
            doc_id="doc-b",
            filename="nda_beta.txt",
            chunk_id="chunk-b",
            chunk_content="Agreement between Alpha Corp and Beta LLC.",
            fact_id="fact-b",
            fact_type="party",
            canonical_value="Beta LLC",
            raw_value="Beta LLC",
            evidence_text="Agreement between Alpha Corp and Beta LLC.",
        )
        retrieval = _RecordingRetrieval()
        service = ChatService(retrieval, model=None, enable_vlm=False, router=None)

        response = service.answer("Which document contains Orion Data Labs Ltd?", doc_ids=None, top_k=5)

        self.assertFalse(retrieval.calls)
        self.assertIn("nda_alpha", response["answer"])
        self.assertNotIn("nda_beta", response["answer"])
        self.assertEqual(1, len(response["sources"]))
        self.assertEqual("doc-a", response["sources"][0]["doc_id"])
        self.assertIn("Matched party: Orion Data Labs Ltd", response["sources"][0]["content"])

    def test_exact_lookup_falls_back_to_hybrid_when_fact_index_has_no_match(self) -> None:
        self._add_document_with_fact(
            doc_id="doc-a",
            filename="nda_alpha.txt",
            chunk_id="chunk-a",
            chunk_content="Agreement between Alpha Corp and Beta LLC.",
            fact_id="fact-a",
            fact_type="party",
            canonical_value="Beta LLC",
            raw_value="Beta LLC",
            evidence_text="Agreement between Alpha Corp and Beta LLC.",
        )
        retrieval = _RecordingRetrieval(
            balanced_chunks=[
                {
                    "id": "search-1",
                    "doc_id": "doc-a",
                    "doc_filename": "nda_alpha.txt",
                    "doc_created_at": "2026-01-01",
                    "page": 1,
                    "content": "Fallback evidence mentions Gamma LLC in a free-text paragraph.",
                    "score": 0.83,
                    "source_type": "text",
                    "metadata": {},
                }
            ]
        )
        service = ChatService(retrieval, model=None, enable_vlm=False, router=None)

        response = service.answer("Which document contains Gamma LLC?", doc_ids=None, top_k=5)

        self.assertTrue(response["route"]["exact_lookup"]["applicable"])
        self.assertEqual("empty", response["route"]["exact_lookup"]["strength"])
        self.assertTrue(response["route"]["exact_lookup"]["used_hybrid_fallback"])
        self.assertTrue(any(call[0] == "search_balanced" for call in retrieval.calls))
        self.assertTrue(response["sources"])
        self.assertEqual("text", response["sources"][0]["source_type"])
        self.assertIn("Gamma LLC", response["answer"])

    def test_compare_query_combines_exact_fact_lookup_with_comparison_retrieval(self) -> None:
        self._add_document_with_fact(
            doc_id="doc-a",
            filename="nda_alpha.txt",
            chunk_id="chunk-a",
            chunk_content="Agreement effective date is 14 March 2015.",
            fact_id="fact-a",
            fact_type="date",
            canonical_value="2015-03-14",
            raw_value="14 March 2015",
            evidence_text="Agreement effective date is 14 March 2015.",
        )
        self._add_document_with_fact(
            doc_id="doc-b",
            filename="nda_beta.txt",
            chunk_id="chunk-b",
            chunk_content="Agreement effective date is 20 April 2018.",
            fact_id="fact-b",
            fact_type="date",
            canonical_value="2018-04-20",
            raw_value="20 April 2018",
            evidence_text="Agreement effective date is 20 April 2018.",
        )
        retrieval = _RecordingRetrieval(
            balanced_chunks=[
                {
                    "id": "cmp-a",
                    "doc_id": "doc-a",
                    "doc_filename": "nda_alpha.txt",
                    "doc_created_at": "2026-01-01",
                    "page": 1,
                    "content": "Alpha agreement became effective on 14 March 2015.",
                    "score": 0.62,
                    "source_type": "text",
                    "metadata": {},
                },
                {
                    "id": "cmp-b",
                    "doc_id": "doc-b",
                    "doc_filename": "nda_beta.txt",
                    "doc_created_at": "2026-01-01",
                    "page": 1,
                    "content": "Beta agreement became effective on 20 April 2018.",
                    "score": 0.61,
                    "source_type": "text",
                    "metadata": {},
                },
            ]
        )
        service = ChatService(retrieval, model=None, enable_vlm=False, router=None)

        response = service.answer(
            "Compare the effective dates in nda_alpha and nda_beta",
            doc_ids=["doc-a", "doc-b"],
            top_k=5,
        )

        self.assertEqual("compare", response["intent"])
        self.assertTrue(response["route"]["exact_lookup"]["applicable"])
        self.assertEqual("strong", response["route"]["exact_lookup"]["strength"])
        self.assertEqual(2, response["route"]["exact_lookup"]["matched_doc_count"])
        self.assertEqual(2, response["route"]["exact_lookup"]["required_doc_count"])
        self.assertTrue(any(call[0] == "search_balanced" for call in retrieval.calls))
        source_doc_ids = {str(source.get("doc_id") or "") for source in response["sources"]}
        self.assertEqual({"doc-a", "doc-b"}, source_doc_ids)
        source_types = {str(source.get("source_type") or "") for source in response["sources"]}
        self.assertIn("document_fact", source_types)


if __name__ == "__main__":
    unittest.main()
