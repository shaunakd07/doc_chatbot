import unittest
from unittest.mock import patch

from backend import config
from backend.services.out_of_place_detection import detectOutOfPlaceDocuments


def _doc(
    doc_id: str,
    *,
    folder_id: str,
    tenant_id: str,
    doc_type: str,
    confidence: float,
    scores: dict[str, float],
    extra_metadata: dict | None = None,
) -> dict:
    metadata = {
        "tenant_id": tenant_id,
        "folder_id": folder_id,
        "doc_type": doc_type,
        "doc_type_confidence": confidence,
        "doc_type_scores": scores,
    }
    if isinstance(extra_metadata, dict):
        metadata.update(extra_metadata)
    return {
        "id": doc_id,
        "filename": f"{doc_id}.pdf",
        "status": "ready",
        "metadata": metadata,
    }


class OutOfPlaceDetectionTests(unittest.TestCase):
    @patch("backend.services.out_of_place_detection.storage.update_document")
    @patch("backend.services.out_of_place_detection.storage.list_documents")
    def test_flags_high_confidence_mismatch(self, list_documents, _update_document) -> None:
        list_documents.return_value = [
            _doc(
                "invoice-doc",
                folder_id="folder-invoices",
                tenant_id="tenant-a",
                doc_type="invoice",
                confidence=0.96,
                scores={"invoice": 2.0, "contract": 0.1},
            ),
            _doc(
                "contract-doc",
                folder_id="folder-invoices",
                tenant_id="tenant-a",
                doc_type="contract",
                confidence=0.95,
                scores={"contract": 2.1, "invoice": 0.2},
            ),
        ]

        with patch.object(config, "DOC_TYPE_REVIEW_ENABLED", True), patch.object(
            config,
            "DOC_TYPE_REVIEW_TENANT_THRESHOLDS",
            {},
        ), patch.object(config, "DOC_TYPE_REVIEW_CONFIDENCE_THRESHOLD", 0.9), patch.object(
            config,
            "DOC_TYPE_REVIEW_MIN_SCORE_RATIO",
            1.2,
        ), patch.object(config, "DOC_TYPE_REVIEW_EQUIVALENT_TYPES", {}), patch.object(
            config,
            "DOC_TYPE_REVIEW_IGNORED_PREDICTED_TYPES",
            {"unknown"},
        ):
            flags = detectOutOfPlaceDocuments(
                folderId="folder-invoices",
                expectedType="invoice",
                tenantId="tenant-a",
            )

        self.assertEqual(1, len(flags))
        self.assertEqual("contract-doc", flags[0]["fileId"])
        self.assertEqual("contract", flags[0]["predictedType"])
        self.assertGreaterEqual(float(flags[0]["confidence"]), 0.95)

    @patch("backend.services.out_of_place_detection.storage.update_document")
    @patch("backend.services.out_of_place_detection.storage.list_documents")
    def test_does_not_flag_below_threshold(self, list_documents, _update_document) -> None:
        list_documents.return_value = [
            _doc(
                "contract-doc",
                folder_id="folder-invoices",
                tenant_id="tenant-a",
                doc_type="contract",
                confidence=0.72,
                scores={"contract": 1.2, "invoice": 0.9},
            )
        ]

        with patch.object(config, "DOC_TYPE_REVIEW_ENABLED", True), patch.object(
            config,
            "DOC_TYPE_REVIEW_TENANT_THRESHOLDS",
            {},
        ), patch.object(config, "DOC_TYPE_REVIEW_CONFIDENCE_THRESHOLD", 0.9), patch.object(
            config,
            "DOC_TYPE_REVIEW_MIN_SCORE_RATIO",
            1.3,
        ), patch.object(config, "DOC_TYPE_REVIEW_EQUIVALENT_TYPES", {}), patch.object(
            config,
            "DOC_TYPE_REVIEW_IGNORED_PREDICTED_TYPES",
            {"unknown"},
        ):
            flags = detectOutOfPlaceDocuments(
                folderId="folder-invoices",
                expectedType="invoice",
                tenantId="tenant-a",
            )

        self.assertEqual([], flags)

    @patch("backend.services.out_of_place_detection.storage.update_document")
    @patch("backend.services.out_of_place_detection.storage.list_documents")
    def test_equivalent_types_do_not_flag(self, list_documents, _update_document) -> None:
        list_documents.return_value = [
            _doc(
                "nda-doc",
                folder_id="folder-contracts",
                tenant_id="tenant-a",
                doc_type="nda",
                confidence=0.97,
                scores={"nda": 2.1, "contract": 1.9},
            )
        ]

        with patch.object(config, "DOC_TYPE_REVIEW_ENABLED", True), patch.object(
            config,
            "DOC_TYPE_REVIEW_TENANT_THRESHOLDS",
            {},
        ), patch.object(config, "DOC_TYPE_REVIEW_CONFIDENCE_THRESHOLD", 0.9), patch.object(
            config,
            "DOC_TYPE_REVIEW_MIN_SCORE_RATIO",
            1.1,
        ), patch.object(
            config,
            "DOC_TYPE_REVIEW_EQUIVALENT_TYPES",
            {"contract": {"nda"}, "nda": {"contract"}},
        ), patch.object(config, "DOC_TYPE_REVIEW_IGNORED_PREDICTED_TYPES", {"unknown"}):
            flags = detectOutOfPlaceDocuments(
                folderId="folder-contracts",
                expectedType="contract",
                tenantId="tenant-a",
            )

        self.assertEqual([], flags)

    @patch("backend.services.out_of_place_detection.storage.update_document")
    @patch("backend.services.out_of_place_detection.storage.list_documents")
    def test_manual_override_suppresses_repeat_flag(self, list_documents, _update_document) -> None:
        list_documents.return_value = [
            _doc(
                "contract-doc",
                folder_id="folder-invoices",
                tenant_id="tenant-a",
                doc_type="contract",
                confidence=0.97,
                scores={"contract": 2.3, "invoice": 0.2},
                extra_metadata={
                    "out_of_place_review_manual_override": True,
                    "out_of_place_review_state": "accepted_mismatch",
                },
            )
        ]

        with patch.object(config, "DOC_TYPE_REVIEW_ENABLED", True), patch.object(
            config,
            "DOC_TYPE_REVIEW_TENANT_THRESHOLDS",
            {},
        ), patch.object(config, "DOC_TYPE_REVIEW_CONFIDENCE_THRESHOLD", 0.9), patch.object(
            config,
            "DOC_TYPE_REVIEW_MIN_SCORE_RATIO",
            1.2,
        ), patch.object(config, "DOC_TYPE_REVIEW_EQUIVALENT_TYPES", {}), patch.object(
            config,
            "DOC_TYPE_REVIEW_IGNORED_PREDICTED_TYPES",
            {"unknown"},
        ):
            flags = detectOutOfPlaceDocuments(
                folderId="folder-invoices",
                expectedType="invoice",
                tenantId="tenant-a",
            )

        self.assertEqual([], flags)

    @patch("backend.services.out_of_place_detection.storage.update_document")
    @patch("backend.services.out_of_place_detection.storage.list_documents")
    def test_dynamic_expected_label_normalizes_free_text(self, list_documents, _update_document) -> None:
        list_documents.return_value = [
            _doc(
                "proposal-doc",
                folder_id="folder-proposals",
                tenant_id="tenant-a",
                doc_type="technical_proposal",
                confidence=0.93,
                scores={"technical_proposal": 2.3, "invoice": 0.1},
                extra_metadata={"doc_type_classifier_provider": "semantic_openai"},
            )
        ]

        with patch.object(config, "DOC_TYPE_REVIEW_ENABLED", True), patch.object(
            config,
            "DOC_TYPE_REVIEW_TENANT_THRESHOLDS",
            {},
        ), patch.object(config, "DOC_TYPE_REVIEW_CONFIDENCE_THRESHOLD", 0.9), patch.object(
            config,
            "DOC_TYPE_REVIEW_MIN_SCORE_RATIO",
            1.1,
        ), patch.object(config, "DOC_TYPE_REVIEW_EQUIVALENT_TYPES", {}), patch.object(
            config,
            "DOC_TYPE_REVIEW_IGNORED_PREDICTED_TYPES",
            {"unknown"},
        ):
            flags = detectOutOfPlaceDocuments(
                folderId="folder-proposals",
                expectedType="technical proposal",
                tenantId="tenant-a",
            )

        self.assertEqual([], flags)


if __name__ == "__main__":
    unittest.main()
