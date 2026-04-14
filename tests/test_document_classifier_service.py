import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from backend.services.document_classifier import AzureDocumentIntelligenceClassifier, SemanticOpenAIClassifier


class _MockAzureClient:
    def __init__(self) -> None:
        self._poll_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def post(self, url, params=None, headers=None, content=None):
        request = httpx.Request("POST", url, params=params, headers=headers)
        return httpx.Response(
            status_code=202,
            headers={"Operation-Location": "https://mock.azure/poll"},
            request=request,
        )

    def get(self, url, headers=None):
        request = httpx.Request("GET", url, headers=headers)
        self._poll_count += 1
        if self._poll_count == 1:
            return httpx.Response(status_code=200, json={"status": "running"}, request=request)
        return httpx.Response(
            status_code=200,
            json={
                "status": "succeeded",
                "result": {
                    "documents": [
                        {"docType": "contract", "confidence": 0.93},
                        {"docType": "invoice", "confidence": 0.07},
                    ],
                    "classes": {
                        "contract": {"confidence": 0.93},
                        "invoice": {"confidence": 0.07},
                    },
                },
            },
            request=request,
        )


class DocumentClassifierServiceTests(unittest.TestCase):
    def test_azure_classifier_parses_mocked_response(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "sample.pdf"
            source.write_bytes(b"%PDF-1.4\n%mock")

            classifier = AzureDocumentIntelligenceClassifier(
                endpoint="https://mock.azure",
                api_key="test-key",
                classifier_id="enterprise-doc-classifier",
                api_version="2024-11-30",
                timeout_sec=5,
                poll_interval_sec=0.01,
            )

            with patch("backend.services.document_classifier.httpx.Client", return_value=_MockAzureClient()):
                result = classifier.classify(
                    file_path=source,
                    filename="sample.pdf",
                    auto_tags=[],
                    text_samples=[],
                )

        self.assertEqual("contract", result.doc_type)
        self.assertAlmostEqual(0.93, float(result.confidence), places=2)
        self.assertEqual("azure_document_intelligence", result.provider)
        self.assertEqual("enterprise-doc-classifier", result.model)

    def test_semantic_classifier_parses_dynamic_label(self) -> None:
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"label":"technical proposal","confidence":0.91,'
                            '"alternatives":[{"label":"invoice","confidence":0.07}]}'
                        )
                    )
                )
            ]
        )
        with patch("backend.services.document_classifier.OpenAI") as openai_cls:
            openai_cls.return_value.chat.completions.create.return_value = mock_response
            classifier = SemanticOpenAIClassifier(
                model_id="gpt-4o-mini",
                api_key="test-key",
                timeout_sec=12,
            )
            result = classifier.classify(
                file_path=None,
                filename="Technical Proposal for Customer.docx",
                auto_tags=[],
                text_samples=["Proposal Terms", "Scope of Work", "Commercial proposal"],
            )

        self.assertEqual("technical_proposal", result.doc_type)
        self.assertAlmostEqual(0.91, float(result.confidence), places=2)
        self.assertEqual("semantic_openai", result.provider)
        self.assertEqual("gpt-4o-mini", result.model)
        self.assertIn("invoice", result.scores)

    def test_semantic_classifier_falls_back_to_heuristic_on_invalid_json(self) -> None:
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not-json"))]
        )
        with patch("backend.services.document_classifier.OpenAI") as openai_cls:
            openai_cls.return_value.chat.completions.create.return_value = mock_response
            classifier = SemanticOpenAIClassifier(
                model_id="gpt-4o-mini",
                api_key="test-key",
                timeout_sec=12,
            )
            result = classifier.classify(
                file_path=None,
                filename="Invoice INV_42.pdf",
                auto_tags=[],
                text_samples=["Invoice Number INV_42", "Total Amount Due", "Tax Invoice"],
            )

        self.assertEqual("invoice", result.doc_type)
        self.assertEqual("heuristic", result.provider)


if __name__ == "__main__":
    unittest.main()
