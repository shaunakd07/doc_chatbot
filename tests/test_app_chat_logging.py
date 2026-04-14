from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import backend.app as app_mod


class _StubChatService:
    def answer(
        self,
        question: str,
        doc_ids=None,
        top_k: int = 5,
        include_document_summaries: bool = True,
        conversation_id: str | None = None,
    ) -> dict:
        return {
            "answer": f"stub answer for {question}",
            "sources": [{"doc_id": "doc-a", "source_type": "text"}],
            "intent": "qa",
            "route": {
                "task_type": "qa",
                "needs_cross_doc": False,
                "needs_numeric_extraction": False,
                "needs_image_reasoning": False,
                "retrieval_plan": {"strategy": "semantic", "top_k": top_k, "per_doc_limit": 1},
                "exact_lookup": {"applicable": False, "used_fact_lookup": False, "used_hybrid_fallback": False},
            },
            "conversation_id": conversation_id or "conv-test",
            "include_document_summaries": include_document_summaries,
        }


class AppChatLoggingTests(unittest.TestCase):
    def test_api_chat_logs_request_and_response_details(self) -> None:
        stub_service = _StubChatService()

        def _init_stub(app) -> None:
            app.state.chat_service = stub_service

        with patch.object(app_mod, "_initialize_app_state", _init_stub):
            with TestClient(app_mod.app) as client:
                with self.assertLogs("backend.app", level="INFO") as captured:
                    response = client.post(
                        "/api/chat",
                        json={
                            "message": "summarize the crunchy data agreement",
                            "top_k": 4,
                            "include_document_summaries": True,
                        },
                    )

        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("qa", payload["intent"])
        log_text = "\n".join(captured.output)
        self.assertIn("api.chat.request", log_text)
        self.assertIn("api.chat.response", log_text)
        self.assertIn('"task_type": "qa"', log_text)
        self.assertIn('"needs_cross_doc": false', log_text)
        self.assertIn('"needs_image_reasoning": false', log_text)
        self.assertIn('"retrieval_plan"', log_text)


if __name__ == "__main__":
    unittest.main()
