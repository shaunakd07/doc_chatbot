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


if __name__ == "__main__":
    unittest.main()

