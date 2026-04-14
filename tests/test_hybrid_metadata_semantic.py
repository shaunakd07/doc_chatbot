import unittest
from datetime import datetime, timezone
from itertools import product
from unittest.mock import patch

from backend.services.chat_service import ChatService


def _chunk(chunk_id: str, doc_id: str, content: str, score: float = 0.9) -> dict:
    return {
        "id": chunk_id,
        "doc_id": doc_id,
        "doc_filename": f"{doc_id}.pdf",
        "doc_created_at": "2026-01-01",
        "page": 1,
        "content": content,
        "score": score,
        "source_type": "text",
    }


class StubRouter:
    def __init__(self, route: dict) -> None:
        self.route_payload = dict(route)

    def route(self, question: str, doc_ids=None, available_docs=None):  # noqa: ARG002
        return dict(self.route_payload)


class ScopedRetrieval:
    def __init__(self, chunks_by_doc: dict[str, list[dict]]) -> None:
        self.chunks_by_doc = chunks_by_doc
        self.calls: list[tuple] = []

    def _scoped_chunks(self, doc_ids, top_k: int) -> list[dict]:
        scoped_doc_ids = [str(doc_id) for doc_id in (doc_ids or self.chunks_by_doc.keys())]
        results: list[dict] = []
        for doc_id in scoped_doc_ids:
            for chunk in self.chunks_by_doc.get(doc_id, []):
                results.append(dict(chunk))
        return results[:top_k]

    def search(self, query: str, top_k: int = 5, doc_ids=None, mode: str | None = None, use_rerank: bool = True):
        self.calls.append(("search", query, top_k, tuple(doc_ids or []), mode, use_rerank))
        return self._scoped_chunks(doc_ids, top_k)

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
        return self._scoped_chunks(doc_ids, top_k)


def _status_query_variants() -> list[str]:
    doc_forms = ("nda agreements", "nda documents")
    status_forms = ("expired", "inactive", "lapsed", "no longer valid")
    templates = (
        "Which {doc_form} are {status_form}?",
        "List {doc_form} that are {status_form}.",
        "Show {doc_form} considered {status_form}!",
    )
    variants: list[str] = []
    for template, doc_form, status_form in product(templates, doc_forms, status_forms):
        variants.append(template.format(doc_form=doc_form, status_form=status_form))
    return variants


def _year_query_variants(year: int) -> list[str]:
    doc_forms = ("nda agreements", "nda records")
    verbs = ("executed", "signed", "dated", "written")
    templates = (
        "Which {doc_form} were {verb} in {year}?",
        "List {doc_form} {verb} during {year}.",
        "Show {doc_form} that were {verb} around {year}.",
    )
    variants: list[str] = []
    for template, doc_form, verb in product(templates, doc_forms, verbs):
        variants.append(template.format(doc_form=doc_form, verb=verb, year=year))
    return variants


def _party_query_variants() -> list[str]:
    party_terms = ("parties", "counterparties", "customers", "clients")
    doc_forms = ("nda agreements", "nda documents")
    templates = (
        "List {party_term} in {doc_form}.",
        "Who are the {party_term} named in {doc_form}?",
        "Show {party_term} captured by {doc_form}.",
    )
    variants: list[str] = []
    for template, party_term, doc_form in product(templates, party_terms, doc_forms):
        variants.append(template.format(party_term=party_term, doc_form=doc_form))
    return variants


def _metadata_only_query_variants() -> list[str]:
    doc_forms = ("nda documents", "nda files")
    templates = (
        "How many {doc_form} are there?",
        "Count the {doc_form}.",
    )
    variants: list[str] = []
    for template, doc_form in product(templates, doc_forms):
        variants.append(template.format(doc_form=doc_form))
    return variants


class HybridMetadataSemanticTests(unittest.TestCase):
    def test_hybrid_status_scopes_by_metadata_and_derives_state(self) -> None:
        route = {
            "task_type": "metadata_query",
            "needs_cross_doc": True,
            "needs_image_reasoning": False,
            "retrieval_plan": {"strategy": "balanced", "top_k": 8, "per_doc_limit": 2},
            "analysis_plan": {"metadata_operation": "count", "metadata_filters": {"doc_type": "nda"}},
            "expected_answer_type": "count",
            "confidence": 0.72,
        }
        retrieval = ScopedRetrieval(
            {
                "nda-old": [
                    _chunk(
                        "c-old",
                        "nda-old",
                        "Mutual agreement effective date 01 Jan 2020 with termination date 31 Dec 2021 between Alpha Corp and Beta LLC.",
                    )
                ],
                "nda-new": [
                    _chunk(
                        "c-new",
                        "nda-new",
                        "Mutual agreement effective date 01 Jan 2024 with termination date 31 Dec 2028 between Gamma Inc and Delta LLC.",
                    )
                ],
                "msa-a": [
                    _chunk("c-msa", "msa-a", "Master service agreement signed in 2022.")
                ],
            }
        )
        service = ChatService(retrieval, model=None, enable_vlm=False, router=StubRouter(route))
        docs = [
            {"id": "nda-old", "filename": "NDA_old.pdf", "status": "ready", "created_at": "2026-01-01", "metadata": {"doc_type": "nda"}},
            {"id": "nda-new", "filename": "NDA_new.pdf", "status": "ready", "created_at": "2026-01-01", "metadata": {"doc_type": "nda"}},
            {"id": "msa-a", "filename": "MSA_A.pdf", "status": "ready", "created_at": "2026-01-01", "metadata": {"doc_type": "contract"}},
        ]
        with patch("backend.services.chat_service.storage.list_documents", return_value=docs):
            with patch("backend.services.chat_service.config.HYBRID_METADATA_SEMANTIC", True):
                with patch.object(ChatService, "_metadata_reference_now", return_value=datetime(2026, 3, 5, tzinfo=timezone.utc)):
                    for question in _status_query_variants()[:6]:
                        response = service.answer(question, doc_ids=None, top_k=6)
                        self.assertEqual("HYBRID", response["route"]["query_intent"]["answer_mode"])
                        self.assertIn("semantic evidence", response["answer"].lower())
                        source_doc_ids = {str(source.get("doc_id") or "") for source in response["sources"]}
                        self.assertIn("nda-old", source_doc_ids)
                        self.assertNotIn("msa-a", source_doc_ids)
                        scoped_calls = [call for call in retrieval.calls if call[0] == "search_balanced"]
                        self.assertTrue(scoped_calls)
                        self.assertTrue(any(set(call[3]) == {"nda-old", "nda-new"} for call in scoped_calls))
                        retrieval.calls.clear()

    def test_hybrid_year_resolution_prefers_textual_execution_dates(self) -> None:
        route = {
            "task_type": "metadata_query",
            "needs_cross_doc": True,
            "needs_image_reasoning": False,
            "retrieval_plan": {"strategy": "balanced", "top_k": 8, "per_doc_limit": 2},
            "analysis_plan": {"metadata_operation": "list", "metadata_filters": {"doc_type": "nda"}},
            "expected_answer_type": "list",
            "confidence": 0.69,
        }
        retrieval = ScopedRetrieval(
            {
                "nda-a": [
                    _chunk("y-a", "nda-a", "Agreement executed on 14 March 2015 and entered by and between Alpha Corp and Beta LLC.")
                ],
                "nda-b": [
                    _chunk("y-b", "nda-b", "Agreement executed on 09 June 2014 and entered by and between Gamma Inc and Delta LLC.")
                ],
            }
        )
        service = ChatService(retrieval, model=None, enable_vlm=False, router=StubRouter(route))
        docs = [
            {"id": "nda-a", "filename": "NDA_A.pdf", "status": "ready", "created_at": "2026-01-01", "metadata": {"doc_type": "nda"}},
            {"id": "nda-b", "filename": "NDA_B.pdf", "status": "ready", "created_at": "2015-01-03", "metadata": {"doc_type": "nda"}},
        ]
        with patch("backend.services.chat_service.storage.list_documents", return_value=docs):
            with patch("backend.services.chat_service.config.HYBRID_METADATA_SEMANTIC", True):
                with patch.object(ChatService, "_metadata_reference_now", return_value=datetime(2026, 3, 5, tzinfo=timezone.utc)):
                    for question in _year_query_variants(2015)[:6]:
                        response = service.answer(question, doc_ids=None, top_k=6)
                        self.assertEqual("HYBRID", response["route"]["query_intent"]["answer_mode"])
                        self.assertIn("NDA_A.pdf", response["answer"])
                        source_doc_ids = {str(source.get("doc_id") or "") for source in response["sources"]}
                        self.assertIn("nda-a", source_doc_ids)
                        self.assertNotIn("nda-b", source_doc_ids)
                        retrieval.calls.clear()

    def test_hybrid_party_extraction_is_cited(self) -> None:
        route = {
            "task_type": "metadata_query",
            "needs_cross_doc": True,
            "needs_image_reasoning": False,
            "retrieval_plan": {"strategy": "balanced", "top_k": 8, "per_doc_limit": 2},
            "analysis_plan": {"metadata_operation": "list", "metadata_filters": {"doc_type": "nda"}},
            "expected_answer_type": "list",
            "confidence": 0.64,
        }
        retrieval = ScopedRetrieval(
            {
                "nda-party": [
                    _chunk(
                        "p-a",
                        "nda-party",
                        "This confidentiality agreement is made between Neptune Systems Pte Ltd and Orion Data Labs Ltd.",
                    )
                ]
            }
        )
        service = ChatService(retrieval, model=None, enable_vlm=False, router=StubRouter(route))
        docs = [
            {"id": "nda-party", "filename": "NDA_party.pdf", "status": "ready", "created_at": "2026-01-01", "metadata": {"doc_type": "nda"}},
        ]
        with patch("backend.services.chat_service.storage.list_documents", return_value=docs):
            with patch("backend.services.chat_service.config.HYBRID_METADATA_SEMANTIC", True):
                for question in _party_query_variants()[:6]:
                    response = service.answer(question, doc_ids=None, top_k=6)
                    self.assertEqual("HYBRID", response["route"]["query_intent"]["answer_mode"])
                    self.assertIn("parties=", response["answer"].lower())
                    source_doc_ids = {str(source.get("doc_id") or "") for source in response["sources"]}
                    self.assertIn("nda-party", source_doc_ids)
                    retrieval.calls.clear()

    def test_metadata_fast_path_is_unchanged_for_metadata_only_queries(self) -> None:
        route = {
            "task_type": "metadata_query",
            "needs_cross_doc": True,
            "needs_image_reasoning": False,
            "retrieval_plan": {"strategy": "balanced", "top_k": 8, "per_doc_limit": 2},
            "analysis_plan": {"metadata_operation": "count", "metadata_filters": {"doc_type": "nda"}},
            "expected_answer_type": "count",
            "confidence": 0.82,
        }
        retrieval = ScopedRetrieval(
            {
                "nda-1": [_chunk("m1", "nda-1", "NDA reference text.")],
                "nda-2": [_chunk("m2", "nda-2", "NDA reference text.")],
            }
        )
        service = ChatService(retrieval, model=None, enable_vlm=False, router=StubRouter(route))
        docs = [
            {"id": "nda-1", "filename": "NDA_1.pdf", "status": "ready", "created_at": "2026-01-01", "metadata": {"doc_type": "nda"}},
            {"id": "nda-2", "filename": "NDA_2.pdf", "status": "ready", "created_at": "2026-01-01", "metadata": {"doc_type": "nda"}},
        ]
        with patch("backend.services.chat_service.storage.list_documents", return_value=docs):
            with patch("backend.services.chat_service.config.HYBRID_METADATA_SEMANTIC", False):
                response_off = service.answer(_metadata_only_query_variants()[0], doc_ids=None, top_k=5)
            with patch("backend.services.chat_service.config.HYBRID_METADATA_SEMANTIC", True):
                response_on = service.answer(_metadata_only_query_variants()[0], doc_ids=None, top_k=5)
        self.assertEqual(response_off["answer"], response_on["answer"])
        self.assertEqual("METADATA_ONLY", response_on["route"]["query_intent"]["answer_mode"])
        self.assertFalse(retrieval.calls)
