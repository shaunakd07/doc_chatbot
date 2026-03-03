import unittest
from unittest.mock import patch

from backend.services.chat_service import ChatService


def _chunk(chunk_id: str, doc_id: str, content: str, score: float = 0.9, filename: str | None = None) -> dict:
    return {
        "id": chunk_id,
        "doc_id": doc_id,
        "doc_filename": filename or f"{doc_id}.pdf",
        "doc_created_at": "2026-01-01",
        "page": 1,
        "content": content,
        "score": score,
        "source_type": "text",
    }


class StubRetrieval:
    def __init__(self, balanced_chunks: list[dict], search_chunks: list[dict]) -> None:
        self.balanced_chunks = balanced_chunks
        self.search_chunks = search_chunks
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


class StubModel:
    def __init__(self) -> None:
        self.last_prompt: str | None = None

    def generate_text(self, prompt: str, max_new_tokens: int = 1500, images=None) -> str:
        self.last_prompt = prompt
        return "stub answer"


class StubRouter:
    def __init__(self, route: dict) -> None:
        self.route_payload = route

    def route(self, question: str, doc_ids=None, available_docs=None):  # noqa: ARG002
        return dict(self.route_payload)


class ChatServiceCrossDocTests(unittest.TestCase):
    def test_cross_doc_qa_uses_qa_prompt_and_balanced_retrieval(self) -> None:
        route = {
            "task_type": "qa",
            "needs_cross_doc": True,
            "needs_image_reasoning": False,
            "retrieval_plan": {"strategy": "semantic", "top_k": 6, "per_doc_limit": 3},
            "confidence": 0.95,
        }
        retrieval = StubRetrieval(
            balanced_chunks=[
                _chunk("c1", "doc-a", "Sysdig Inc is based in the United States.", 0.93),
                _chunk("c2", "doc-b", "NCS Pte Ltd is based in Singapore.", 0.91),
            ],
            search_chunks=[
                _chunk("s1", "doc-a", "Fallback chunk A", 0.8),
                _chunk("s2", "doc-b", "Fallback chunk B", 0.79),
            ],
        )
        model = StubModel()
        service = ChatService(retrieval, model, enable_vlm=True, router=StubRouter(route))

        response = service.answer(
            "List all customers and where they are based",
            doc_ids=["doc-a", "doc-b"],
            top_k=5,
        )

        self.assertEqual("qa", response["intent"])
        self.assertTrue(any(call[0] == "search_balanced" for call in retrieval.calls))
        self.assertFalse(any(call[0] == "search" and "differences changes over time" in call[1] for call in retrieval.calls))
        self.assertIsNotNone(model.last_prompt)
        self.assertNotIn("Required output order", model.last_prompt or "")
        self.assertNotIn("Key changes over time", model.last_prompt or "")

    def test_compare_still_uses_compare_prompt(self) -> None:
        route = {
            "task_type": "compare",
            "needs_cross_doc": True,
            "needs_image_reasoning": False,
            "retrieval_plan": {"strategy": "semantic", "top_k": 6, "per_doc_limit": 3},
            "confidence": 0.95,
        }
        retrieval = StubRetrieval(
            balanced_chunks=[
                _chunk("c10", "doc-a", "Version 1 includes module A.", 0.93),
                _chunk("c11", "doc-b", "Version 2 includes module A and B.", 0.91),
            ],
            search_chunks=[_chunk("s10", "doc-a", "Fallback chunk", 0.8)],
        )
        model = StubModel()
        service = ChatService(retrieval, model, enable_vlm=True, router=StubRouter(route))

        response = service.answer("Compare the architecture changes between documents", doc_ids=["doc-a", "doc-b"])

        self.assertEqual("compare", response["intent"])
        self.assertIsNotNone(model.last_prompt)
        self.assertIn("document comparison assistant", (model.last_prompt or "").lower())

    def test_heuristic_fallback_keeps_cross_doc_qa(self) -> None:
        retrieval = StubRetrieval(
            balanced_chunks=[_chunk("h1", "doc-a", "customer A", 0.8)],
            search_chunks=[_chunk("h2", "doc-a", "customer B", 0.79)],
        )
        service = ChatService(retrieval, model=None, enable_vlm=False, router=None)

        route = service._route_question(
            "List all the customers and where they are based",
            doc_ids=["doc-a", "doc-b"],
            default_top_k=5,
        )

        self.assertEqual("qa", route["task_type"])
        self.assertTrue(route["needs_cross_doc"])

    def test_cross_doc_route_plan_is_normalized_for_balanced_and_floor_values(self) -> None:
        route = {
            "task_type": "qa",
            "needs_cross_doc": True,
            "needs_image_reasoning": False,
            "retrieval_plan": {"strategy": "semantic", "top_k": 4, "per_doc_limit": 1},
            "confidence": 0.95,
        }
        retrieval = StubRetrieval(
            balanced_chunks=[
                _chunk("n1", "doc-a", "Airtel uses integration services.", 0.9),
                _chunk("n2", "doc-b", "Lakerunner provides integration support.", 0.88),
            ],
            search_chunks=[],
        )
        model = StubModel()
        service = ChatService(retrieval, model, enable_vlm=True, router=StubRouter(route))

        response = service.answer("List services for Airtel across docs", doc_ids=["doc-a", "doc-b"], top_k=5)

        plan = response["route"]["retrieval_plan"]
        self.assertEqual("balanced", plan["strategy"])
        self.assertGreaterEqual(int(plan["top_k"]), 10)
        self.assertGreaterEqual(int(plan["per_doc_limit"]), 2)

    def test_hard_no_answer_is_replaced_when_evidence_exists(self) -> None:
        route = {
            "task_type": "qa",
            "needs_cross_doc": False,
            "needs_image_reasoning": False,
            "retrieval_plan": {"strategy": "hybrid", "top_k": 6, "per_doc_limit": 2},
            "confidence": 0.95,
        }
        retrieval = StubRetrieval(
            balanced_chunks=[],
            search_chunks=[
                _chunk("x1", "doc-a", "Lakerunner provides ServiceNow CMDB integration support to Airtel.", 0.91)
            ],
        )

        class HardFailModel(StubModel):
            def generate_text(self, prompt: str, max_new_tokens: int = 1500, images=None) -> str:  # noqa: ARG002
                self.last_prompt = prompt
                return "I cannot find the answer in the provided documents."

        model = HardFailModel()
        service = ChatService(retrieval, model, enable_vlm=True, router=StubRouter(route))

        response = service.answer("What services does Lakerunner provide to Airtel?", doc_ids=["doc-a"])

        self.assertIn("partial evidence", response["answer"].lower())
        self.assertNotEqual("I cannot find the answer in the provided documents.", response["answer"].strip())

    def test_invoice_fallback_prefers_invoice_chunks_over_proposal_chunks(self) -> None:
        retrieval = StubRetrieval(balanced_chunks=[], search_chunks=[])
        service = ChatService(retrieval, model=None, enable_vlm=False, router=None)

        chunks = [
            _chunk(
                "p1",
                "doc-proposal",
                "Ashnik proposal for managed services and architecture support.",
                score=0.95,
                filename="Ashnik_AI_docs/Proposals/Ashnik Proposal.docx",
            ),
            _chunk(
                "i1",
                "doc-invoice",
                "Invoice INV_19-20_223 lists managed support services for the billed period.",
                score=0.72,
                filename="Ashnik_AI_docs/Ashnik Invoice/Invoice INV_19-20_223.pdf",
            ),
        ]

        answer = service._fallback_answer(
            "What services did Ashnik provide shown in the invoices?",
            chunks,
            intent="qa",
        )

        self.assertIn("INV_19-20_223", answer)
        self.assertNotIn("doc-proposal", answer)

    def test_auto_tag_scoping_prefers_matching_document_type(self) -> None:
        retrieval = StubRetrieval(balanced_chunks=[], search_chunks=[])
        service = ChatService(retrieval, model=None, enable_vlm=False, router=None)

        fake_docs = [
            {
                "id": "doc-invoice",
                "filename": "client_docs/invoice_2024_01.pdf",
                "status": "ready",
                "created_at": "2026-01-01",
                "metadata": {"auto_tags": ["invoice", "tax invoice", "billing services"]},
            },
            {
                "id": "doc-proposal",
                "filename": "client_docs/proposal_alpha.docx",
                "status": "ready",
                "created_at": "2026-01-01",
                "metadata": {"auto_tags": ["proposal", "managed services", "scope"]},
            },
            {
                "id": "doc-contract",
                "filename": "client_docs/master_agreement.pdf",
                "status": "ready",
                "created_at": "2026-01-01",
                "metadata": {"auto_tags": ["agreement", "terms", "conditions"]},
            },
            {
                "id": "doc-slide",
                "filename": "client_docs/overview_slides.pptx",
                "status": "ready",
                "created_at": "2026-01-01",
                "metadata": {"auto_tags": ["presentation", "overview"]},
            },
            {
                "id": "doc-po",
                "filename": "client_docs/po_7744.pdf",
                "status": "ready",
                "created_at": "2026-01-01",
                "metadata": {"auto_tags": ["purchase order", "line items"]},
            },
        ]

        with patch("backend.services.chat_service.storage.list_documents", return_value=fake_docs):
            scope = service._select_candidate_docs_for_query(
                "What services are listed in the invoices?",
                doc_ids=None,
                require_multi_doc=False,
            )

        self.assertIsNotNone(scope)
        scoped_ids = (scope or {}).get("doc_ids") or []
        self.assertTrue(scoped_ids)
        self.assertEqual("doc-invoice", scoped_ids[0])

    def test_count_intent_uses_document_metadata_without_retrieval(self) -> None:
        route = {
            "task_type": "metadata_query",
            "needs_cross_doc": True,
            "needs_image_reasoning": False,
            "retrieval_plan": {"strategy": "hybrid", "top_k": 6, "per_doc_limit": 2},
            "analysis_plan": {"metadata_operation": "count", "metadata_filters": {"doc_type": "nda"}},
            "confidence": 0.40,
        }
        retrieval = StubRetrieval(
            balanced_chunks=[_chunk("b1", "doc-a", "unused", 0.2)],
            search_chunks=[_chunk("s1", "doc-a", "unused", 0.2)],
        )
        service = ChatService(retrieval, model=None, enable_vlm=False, router=StubRouter(route))
        fake_docs = [
            {
                "id": "doc-nda-1",
                "filename": "contracts/NDA_Alpha.pdf",
                "status": "ready",
                "created_at": "2026-01-01",
                "metadata": {"doc_type": "nda", "doc_type_confidence": 0.91, "auto_tags": ["nda", "agreement"]},
            },
            {
                "id": "doc-nda-2",
                "filename": "contracts/NDA_Beta.docx",
                "status": "ready",
                "created_at": "2026-01-01",
                "metadata": {"doc_type": "nda", "doc_type_confidence": 0.88, "auto_tags": ["nda"]},
            },
            {
                "id": "doc-po-1",
                "filename": "purchasing/PO_2024_01.pdf",
                "status": "ready",
                "created_at": "2026-01-01",
                "metadata": {"doc_type": "purchase_order", "doc_type_confidence": 0.86, "auto_tags": ["purchase order"]},
            },
        ]

        with patch("backend.services.chat_service.storage.list_documents", return_value=fake_docs):
            response = service.answer("How many NDA documents are there?", doc_ids=None, top_k=5)

        self.assertEqual("count", response["intent"])
        self.assertIn("2", response["answer"])
        self.assertFalse(retrieval.calls)
        self.assertTrue(response["sources"])

    def test_count_intent_can_match_filename_focus_terms(self) -> None:
        route = {
            "task_type": "metadata_query",
            "needs_cross_doc": True,
            "needs_image_reasoning": False,
            "retrieval_plan": {"strategy": "hybrid", "top_k": 6, "per_doc_limit": 2},
            "analysis_plan": {"metadata_operation": "count", "metadata_filters": {"target_document": "lakerunner"}},
            "confidence": 0.31,
        }
        retrieval = StubRetrieval(balanced_chunks=[], search_chunks=[])
        service = ChatService(retrieval, model=None, enable_vlm=False, router=StubRouter(route))
        fake_docs = [
            {
                "id": "d1",
                "filename": "LakeRunner/Lakerunner_updated_with_licensing_multitenant 1.pptx",
                "status": "ready",
                "created_at": "2026-01-01",
                "metadata": {"doc_type": "presentation", "doc_type_confidence": 0.72, "auto_tags": ["lakerunner"]},
            },
            {
                "id": "d2",
                "filename": "LakeRunner/LakeRunner.pptx",
                "status": "ready",
                "created_at": "2026-01-01",
                "metadata": {"doc_type": "presentation", "doc_type_confidence": 0.70, "auto_tags": ["lakerunner"]},
            },
            {
                "id": "d3",
                "filename": "LakeRunner/LakeRunner.pdf",
                "status": "ready",
                "created_at": "2026-01-01",
                "metadata": {"doc_type": "report", "doc_type_confidence": 0.44, "auto_tags": ["lakerunner"]},
            },
            {
                "id": "d4",
                "filename": "LakeRunner/Lakerunner license.xlsx",
                "status": "ready",
                "created_at": "2026-01-01",
                "metadata": {"doc_type": "spreadsheet", "doc_type_confidence": 0.66, "auto_tags": ["lakerunner", "license"]},
            },
            {
                "id": "d5",
                "filename": "LakeRunner/Airtel Cloud Prep.pptx",
                "status": "ready",
                "created_at": "2026-01-01",
                "metadata": {"doc_type": "presentation", "doc_type_confidence": 0.63, "auto_tags": ["airtel", "lakerunner"]},
            },
            {
                "id": "d6",
                "filename": "LakeRunner/Airtel Cloud PPT 1.pptx",
                "status": "ready",
                "created_at": "2026-01-01",
                "metadata": {"doc_type": "presentation", "doc_type_confidence": 0.62, "auto_tags": ["airtel", "lakerunner"]},
            },
        ]

        with patch("backend.services.chat_service.storage.list_documents", return_value=fake_docs):
            response = service.answer("How many Lakerunner documents are there?", doc_ids=None, top_k=5)

        self.assertEqual("count", response["intent"])
        self.assertIn("6", response["answer"])
        self.assertFalse(retrieval.calls)

    def test_metadata_grounding_infers_excel_filter_without_router_filter(self) -> None:
        route = {
            "task_type": "metadata_query",
            "needs_cross_doc": True,
            "needs_image_reasoning": False,
            "retrieval_plan": {"strategy": "hybrid", "top_k": 6, "per_doc_limit": 2},
            "analysis_plan": {"metadata_operation": "count", "metadata_filters": {}},
            "expected_answer_type": "count",
            "confidence": 0.42,
        }
        retrieval = StubRetrieval(balanced_chunks=[], search_chunks=[])
        service = ChatService(retrieval, model=None, enable_vlm=False, router=StubRouter(route))
        fake_docs = [
            {"id": "d1", "filename": "LakeRunner/LakeRunner.pptx", "status": "ready", "created_at": "2026-01-01", "metadata": {"doc_type": "presentation"}},
            {"id": "d2", "filename": "LakeRunner/LakeRunner.pdf", "status": "ready", "created_at": "2026-01-01", "metadata": {"doc_type": "report"}},
            {"id": "d3", "filename": "LakeRunner/Lakerunner license.xlsx", "status": "ready", "created_at": "2026-01-01", "metadata": {"doc_type": "spreadsheet"}},
        ]
        with patch("backend.services.chat_service.storage.list_documents", return_value=fake_docs):
            response = service.answer("How many excel files are there?", doc_ids=None, top_k=5)

        self.assertEqual("count", response["intent"])
        self.assertIn("1", response["answer"])
        self.assertIn("doc_type=excel", response["answer"].lower())
        self.assertFalse(retrieval.calls)

    def test_person_answer_type_rewrites_list_to_last_modified_query(self) -> None:
        route = {
            "task_type": "metadata_query",
            "needs_cross_doc": True,
            "needs_image_reasoning": False,
            "retrieval_plan": {"strategy": "hybrid", "top_k": 6, "per_doc_limit": 2},
            "analysis_plan": {"metadata_operation": "list", "metadata_filters": {}},
            "expected_answer_type": "person",
            "confidence": 0.47,
        }
        retrieval = StubRetrieval(balanced_chunks=[], search_chunks=[])
        service = ChatService(retrieval, model=None, enable_vlm=False, router=StubRouter(route))
        fake_docs = [
            {
                "id": "a1",
                "filename": "LakeRunner/Airtel Cloud Prep.pptx",
                "status": "ready",
                "created_at": "2026-01-01",
                "metadata": {"last_modified_by": "Sachin Dabir", "doc_type": "presentation"},
            },
            {
                "id": "a2",
                "filename": "LakeRunner/Airtel Cloud PPT 1.pptx",
                "status": "ready",
                "created_at": "2026-01-01",
                "metadata": {"last_modified_by": "Sachin Dabir", "doc_type": "presentation"},
            },
        ]
        with patch("backend.services.chat_service.storage.list_documents", return_value=fake_docs):
            response = service.answer("Who last modified the Airtel pptx", doc_ids=None, top_k=5)

        self.assertEqual("metadata_query", response["intent"])
        self.assertIn("multiple documents matching", response["answer"].lower())
        self.assertIn("Airtel Cloud Prep.pptx", response["answer"])
        self.assertFalse(retrieval.calls)

    def test_plain_qa_with_list_answer_type_does_not_switch_to_metadata_query(self) -> None:
        route = {
            "task_type": "qa",
            "needs_cross_doc": False,
            "needs_image_reasoning": False,
            "retrieval_plan": {"strategy": "hybrid", "top_k": 6, "per_doc_limit": 2},
            "analysis_plan": {},
            "expected_answer_type": "list",
            "confidence": 0.82,
        }
        retrieval = StubRetrieval(
            balanced_chunks=[],
            search_chunks=[_chunk("q1", "doc-a", "Lakerunner provides cloud migration and managed support services to Airtel.", 0.91)],
        )
        service = ChatService(retrieval, model=None, enable_vlm=False, router=StubRouter(route))
        fake_docs = [
            {"id": "doc-a", "filename": "LakeRunner/Airtel Cloud Prep.pptx", "status": "ready", "created_at": "2026-01-01", "metadata": {}},
        ]
        with patch("backend.services.chat_service.storage.list_documents", return_value=fake_docs):
            response = service.answer("What services does Lakerunner provide to Airtel", doc_ids=None, top_k=5)

        self.assertEqual("qa", response["intent"])
        self.assertTrue(any(call[0] == "search" for call in retrieval.calls))
        self.assertNotIn("document(s) in the current metadata query scope", response["answer"].lower())

    def test_metadata_zero_matches_falls_back_to_semantic_evidence(self) -> None:
        route = {
            "task_type": "metadata_query",
            "needs_cross_doc": True,
            "needs_image_reasoning": False,
            "retrieval_plan": {"strategy": "hybrid", "top_k": 6, "per_doc_limit": 2},
            "analysis_plan": {"metadata_operation": "count", "metadata_filters": {"author": "unknown-user"}},
            "expected_answer_type": "count",
            "confidence": 0.65,
        }
        retrieval = StubRetrieval(
            balanced_chunks=[
                _chunk("m1", "doc-a", "NDA executed on 4 Jan 2018 between Prudential and Ashnik.", 0.91),
            ],
            search_chunks=[],
        )
        service = ChatService(retrieval, model=None, enable_vlm=False, router=StubRouter(route))
        fake_docs = [
            {
                "id": "doc-a",
                "filename": "contracts/nda_2018_prudential_ashnik.pdf",
                "status": "ready",
                "created_at": "2026-01-01",
                "metadata": {"doc_type": "nda", "author": "Ashnik"},
            }
        ]
        with patch("backend.services.chat_service.storage.list_documents", return_value=fake_docs):
            response = service.answer("Which NDAs were signed in 2018?", doc_ids=None, top_k=5)

        self.assertIn("partial evidence", response["answer"].lower())
        self.assertTrue(any(call[0] == "search_balanced" for call in retrieval.calls))

    def test_count_task_without_count_signal_is_treated_as_list_metadata_query(self) -> None:
        route = {
            "task_type": "count",
            "needs_cross_doc": True,
            "needs_image_reasoning": False,
            "retrieval_plan": {"strategy": "balanced", "top_k": 8, "per_doc_limit": 2},
            "analysis_plan": {},
            "expected_answer_type": "unknown",
            "confidence": 0.62,
        }
        retrieval = StubRetrieval(
            balanced_chunks=[_chunk("m2", "doc-a", "NDA executed on 4 Jan 2018 between Prudential and Ashnik.", 0.89)],
            search_chunks=[],
        )
        service = ChatService(retrieval, model=None, enable_vlm=False, router=StubRouter(route))
        fake_docs = [
            {
                "id": "doc-a",
                "filename": "contracts/nda_2018_prudential_ashnik.pdf",
                "status": "ready",
                "created_at": "2026-01-01",
                "metadata": {"doc_type": "nda"},
            }
        ]
        with patch("backend.services.chat_service.storage.list_documents", return_value=fake_docs):
            response = service.answer("Which NDAs were signed in 2018?", doc_ids=None, top_k=5)

        self.assertEqual("metadata_query", response["intent"])
        self.assertNotIn("there are", response["answer"].lower())

    def test_trend_analysis_builds_tag_based_classes_and_enforces_numeric_plan(self) -> None:
        route = {
            "task_type": "trend_analysis",
            "needs_cross_doc": True,
            "needs_numeric_extraction": True,
            "needs_image_reasoning": False,
            "retrieval_plan": {"strategy": "semantic", "top_k": 6, "per_doc_limit": 1},
            "analysis_plan": {},
            "confidence": 0.95,
        }
        retrieval = StubRetrieval(
            balanced_chunks=[
                _chunk("t1", "doc-invoice", "Invoice total SGD 12,000 for 2022 managed services.", 0.95),
                _chunk("t2", "doc-invoice", "Invoice total SGD 18,500 for 2023 support services.", 0.93),
                _chunk("t3", "doc-proposal", "Proposal amount SGD 20,000 for 2024 subscription scope.", 0.91),
                _chunk("t4", "doc-proposal", "Proposal amount SGD 24,000 for 2025 renewal scope.", 0.90),
            ],
            search_chunks=[],
        )
        service = ChatService(retrieval, model=None, enable_vlm=False, router=StubRouter(route))
        fake_docs = [
            {
                "id": "doc-invoice",
                "filename": "Ashnik_AI_docs/Ashnik Invoice/INV_2023_100.pdf",
                "status": "ready",
                "created_at": "2026-01-01",
                "metadata": {"auto_tags": ["invoice", "billing", "services"]},
            },
            {
                "id": "doc-proposal",
                "filename": "Ashnik_AI_docs/Proposals/Elastic_Proposal.docx",
                "status": "ready",
                "created_at": "2026-01-01",
                "metadata": {"auto_tags": ["proposal", "pricing", "services"]},
            },
        ]

        with patch("backend.services.chat_service.storage.list_documents", return_value=fake_docs):
            response = service.answer(
                "Based on the invoices and proposals, analyze the trend of Ashnik revenue.",
                doc_ids=["doc-invoice", "doc-proposal"],
                top_k=5,
            )

        self.assertEqual("trend_analysis", response["intent"])
        plan = response["route"]["retrieval_plan"]
        self.assertEqual("balanced", plan["strategy"])
        self.assertGreaterEqual(int(plan["top_k"]), 12)
        self.assertGreaterEqual(int(plan["per_doc_limit"]), 2)
        evidence_classes = response["route"].get("analysis_plan", {}).get("evidence_classes", [])
        labels = " ".join(str(item.get("label", "")).lower() for item in evidence_classes if isinstance(item, dict))
        self.assertIn("invoice", labels)
        self.assertIn("proposal", labels)

    def test_trend_analysis_runs_targeted_followup_passes(self) -> None:
        route = {
            "task_type": "trend_analysis",
            "needs_cross_doc": True,
            "needs_numeric_extraction": True,
            "needs_image_reasoning": False,
            "retrieval_plan": {"strategy": "balanced", "top_k": 6, "per_doc_limit": 2},
            "analysis_plan": {},
            "confidence": 0.95,
        }

        class TrendRetrievalStub:
            def __init__(self) -> None:
                self.calls: list[tuple] = []

            def search(self, query: str, top_k: int = 5, doc_ids=None, mode: str | None = None, use_rerank: bool = True):
                self.calls.append(("search", query, top_k, tuple(doc_ids or []), mode, use_rerank))
                return []

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
                lower_query = query.lower()
                if "proposal" in lower_query:
                    return [
                        _chunk("p-a", "doc-proposal", "Proposal value SGD 30,000 in 2024.", 0.91),
                        _chunk("p-b", "doc-proposal", "Proposal value SGD 40,000 in 2025.", 0.90),
                    ]
                if "invoice" in lower_query:
                    return [
                        _chunk("i-a", "doc-invoice", "Invoice amount SGD 10,000 in 2022.", 0.93),
                        _chunk("i-b", "doc-invoice", "Invoice amount SGD 16,000 in 2023.", 0.92),
                    ]
                return [_chunk("seed-a", "doc-invoice", "Invoice amount SGD 10,000 in 2022.", 0.88)]

        retrieval = TrendRetrievalStub()
        service = ChatService(retrieval, model=None, enable_vlm=False, router=StubRouter(route))
        fake_docs = [
            {
                "id": "doc-invoice",
                "filename": "Ashnik_AI_docs/Ashnik Invoice/INV_2023_100.pdf",
                "status": "ready",
                "created_at": "2026-01-01",
                "metadata": {"auto_tags": ["invoice", "billing", "services"]},
            },
            {
                "id": "doc-proposal",
                "filename": "Ashnik_AI_docs/Proposals/Elastic_Proposal.docx",
                "status": "ready",
                "created_at": "2026-01-01",
                "metadata": {"auto_tags": ["proposal", "pricing", "services"]},
            },
        ]

        with patch("backend.services.chat_service.storage.list_documents", return_value=fake_docs):
            response = service.answer(
                "Analyze revenue trend from invoices and proposals.",
                doc_ids=["doc-invoice", "doc-proposal"],
                top_k=5,
            )

        balanced_queries = [call[1].lower() for call in retrieval.calls if call[0] == "search_balanced"]
        self.assertTrue(any("proposal" in query for query in balanced_queries))
        self.assertGreaterEqual(len(balanced_queries), 3)
        coverage = response["route"].get("analysis_plan", {}).get("coverage", {})
        self.assertIn("needs_follow_up", coverage)


if __name__ == "__main__":
    unittest.main()
