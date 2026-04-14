import unittest

from backend.services.openai_router_service import OpenAIRouterService
from backend.services.router_service import RouterService


class RouterSchemaTests(unittest.TestCase):
    def test_openai_router_parses_trend_analysis_schema(self) -> None:
        router = OpenAIRouterService(model_id="gpt-4o-mini", api_key="test-key")
        parsed = router._parse_route_json(
            """
            {
              "task_type": "trend_analysis",
              "needs_cross_doc": false,
              "needs_numeric_extraction": false,
              "needs_image_reasoning": false,
              "retrieval_plan": {"strategy": "semantic", "top_k": 4, "per_doc_limit": 1},
              "analysis_plan": {"query_entities": ["invoices", "proposals"], "evidence_classes": ["invoice", "proposal"]},
              "confidence": 0.91,
              "rationale": "trend query"
            }
            """
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual("trend_analysis", parsed["task_type"])
        self.assertTrue(parsed["needs_cross_doc"])
        self.assertTrue(parsed["needs_numeric_extraction"])
        self.assertEqual("balanced", parsed["retrieval_plan"]["strategy"])
        self.assertGreaterEqual(int(parsed["retrieval_plan"]["top_k"]), 12)
        self.assertIn("invoices", [item.lower() for item in parsed["analysis_plan"].get("query_entities", [])])

    def test_local_router_parser_keeps_analysis_plan(self) -> None:
        router = RouterService(model_id="dummy-router")
        parsed = router._parse_route_json(
            """
            {
              "task_type": "qa",
              "needs_cross_doc": true,
              "needs_numeric_extraction": true,
              "needs_image_reasoning": false,
              "retrieval_plan": {"strategy": "balanced", "top_k": 12, "per_doc_limit": 3},
              "analysis_plan": {"query_entities": ["revenue"], "evidence_classes": [{"label": "invoice"}]},
              "confidence": 0.76,
              "rationale": "cross-doc numeric synthesis"
            }
            """
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertTrue(parsed["needs_cross_doc"])
        self.assertTrue(parsed["needs_numeric_extraction"])
        classes = parsed["analysis_plan"].get("evidence_classes", [])
        self.assertTrue(classes)
        self.assertEqual("invoice", classes[0].lower())

    def test_openai_router_parses_exact_lookup_hints(self) -> None:
        router = OpenAIRouterService(model_id="gpt-4o-mini", api_key="test-key")
        parsed = router._parse_route_json(
            """
            {
              "task_type": "compare",
              "needs_cross_doc": true,
              "needs_numeric_extraction": false,
              "needs_image_reasoning": false,
              "retrieval_plan": {"strategy": "balanced", "top_k": 10, "per_doc_limit": 2},
              "analysis_plan": {
                "query_entities": ["nda alpha", "nda beta"],
                "target_documents": ["nda_alpha.txt", "nda_beta.txt"],
                "exact_lookup_requested": true,
                "fact_types": ["date", "amount", "ignored"]
              },
              "confidence": 0.84,
              "rationale": "compare exact facts"
            }
            """
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        plan = parsed["analysis_plan"]
        self.assertTrue(plan.get("exact_lookup_requested"))
        self.assertEqual(["date", "amount"], plan.get("fact_types"))
        self.assertEqual(["nda_alpha.txt", "nda_beta.txt"], plan.get("target_documents"))

    def test_openai_router_accepts_count_task_type(self) -> None:
        router = OpenAIRouterService(model_id="gpt-4o-mini", api_key="test-key")
        parsed = router._parse_route_json(
            """
            {
              "task_type": "count",
              "needs_cross_doc": false,
              "needs_numeric_extraction": true,
              "needs_image_reasoning": false,
              "retrieval_plan": {"strategy": "semantic", "top_k": 4, "per_doc_limit": 1},
              "analysis_plan": {},
              "confidence": 0.88,
              "rationale": "counting documents"
            }
            """
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual("count", parsed["task_type"])
        self.assertTrue(parsed["needs_cross_doc"])
        self.assertFalse(parsed["needs_numeric_extraction"])
        self.assertEqual("balanced", parsed["retrieval_plan"]["strategy"])
        self.assertEqual("unknown", parsed["expected_answer_type"])

    def test_openai_router_uses_count_expected_type_when_operation_is_count(self) -> None:
        router = OpenAIRouterService(model_id="gpt-4o-mini", api_key="test-key")
        parsed = router._parse_route_json(
            """
            {
              "task_type": "count",
              "needs_cross_doc": false,
              "needs_numeric_extraction": false,
              "needs_image_reasoning": false,
              "retrieval_plan": {"strategy": "semantic", "top_k": 6, "per_doc_limit": 1},
              "analysis_plan": {"metadata_operation": "count"},
              "confidence": 0.88,
              "rationale": "count operation explicit"
            }
            """
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual("count", parsed["expected_answer_type"])

    def test_openai_router_infers_expected_answer_type_for_person_metadata_query(self) -> None:
        router = OpenAIRouterService(model_id="gpt-4o-mini", api_key="test-key")
        parsed = router._parse_route_json(
            """
            {
              "task_type": "metadata_query",
              "needs_cross_doc": true,
              "needs_numeric_extraction": false,
              "needs_image_reasoning": false,
              "retrieval_plan": {"strategy": "balanced", "top_k": 10, "per_doc_limit": 2},
              "analysis_plan": {"metadata_operation": "last_modified_by", "metadata_filters": {"target_document": "Airtel pptx"}},
              "confidence": 0.84,
              "rationale": "metadata person lookup"
            }
            """
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual("metadata_query", parsed["task_type"])
        self.assertEqual("person", parsed["expected_answer_type"])


if __name__ == "__main__":
    unittest.main()
