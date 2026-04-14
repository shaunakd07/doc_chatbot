import unittest
from unittest.mock import patch

from backend.services.review_workflow_service import apply_review_decision, build_folder_review_summary


def _doc(
    doc_id: str,
    *,
    folder_id: str,
    tenant_id: str,
    state: str,
    confidence: float,
    predicted: str,
    expected: str = "invoice",
    extra_metadata: dict | None = None,
) -> dict:
    metadata = {
        "tenant_id": tenant_id,
        "folder_id": folder_id,
        "out_of_place_review_state": state,
        "out_of_place_review_confidence": confidence,
        "out_of_place_review_predicted_type": predicted,
        "out_of_place_review_expected_type": expected,
        "out_of_place_review_reason": f"Predicted '{predicted}' but folder expects '{expected}'.",
    }
    if isinstance(extra_metadata, dict):
        metadata.update(extra_metadata)
    return {
        "id": doc_id,
        "filename": f"{doc_id}.pdf",
        "status": "ready",
        "metadata": metadata,
    }


class ReviewWorkflowServiceTests(unittest.TestCase):
    @patch("backend.services.review_workflow_service.storage.list_documents")
    def test_build_folder_review_summary_counts_states(self, list_documents) -> None:
        list_documents.return_value = [
            _doc("doc-a", folder_id="invoices-q1", tenant_id="tenant-a", state="needs_review", confidence=0.94, predicted="contract"),
            _doc("doc-b", folder_id="invoices-q1", tenant_id="tenant-a", state="clear", confidence=0.21, predicted="invoice"),
            _doc("doc-c", folder_id="invoices-q1", tenant_id="tenant-a", state="accepted_mismatch", confidence=0.96, predicted="nda"),
            _doc("doc-x", folder_id="contracts-q1", tenant_id="tenant-a", state="needs_review", confidence=0.95, predicted="invoice"),
            _doc("doc-y", folder_id="invoices-q1", tenant_id="tenant-b", state="needs_review", confidence=0.90, predicted="contract"),
        ]

        summary = build_folder_review_summary(folder_id="invoices-q1", tenant_id="tenant-a")

        self.assertEqual("tenant-a", summary["tenant_id"])
        self.assertEqual("invoices-q1", summary["folder_id"])
        self.assertEqual("needs_review", summary["review_status"])
        self.assertEqual(3, int(summary["counts"]["total"]))
        self.assertEqual(1, int(summary["counts"]["needs_review"]))
        self.assertEqual(1, int(summary["counts"]["clear"]))
        self.assertEqual(1, int(summary["counts"]["accepted_mismatch"]))
        self.assertEqual(1, int(summary["open_flag_count"]))
        self.assertEqual("doc-a", summary["flags"][0]["fileId"])

    @patch("backend.services.review_workflow_service.build_folder_review_summary")
    @patch("backend.services.review_workflow_service.storage.update_document")
    @patch("backend.services.review_workflow_service.storage.get_document")
    def test_apply_dismiss_decision_clears_flag(self, get_document, update_document, build_summary) -> None:
        get_document.return_value = _doc(
            "doc-a",
            folder_id="invoices-q1",
            tenant_id="tenant-a",
            state="needs_review",
            confidence=0.94,
            predicted="contract",
        )
        build_summary.return_value = {"review_status": "clear", "open_flag_count": 0}

        result = apply_review_decision(
            folder_id="invoices-q1",
            doc_id="doc-a",
            tenant_id="tenant-a",
            decision="dismiss",
            reviewer={"id": "u-1", "name": "A Reviewer", "email": "a@example.com", "type": "human"},
            note="Known exception",
        )

        payload = update_document.call_args.kwargs.get("metadata")
        self.assertEqual("clear", payload["out_of_place_review_state"])
        self.assertEqual(True, payload["out_of_place_review_manual_override"])
        self.assertEqual(False, payload["out_of_place_review_action_required"])
        self.assertEqual("dismiss", payload["out_of_place_review_decision"])
        self.assertEqual("Known exception", payload["out_of_place_review_decision_note"])
        self.assertEqual("clear", result["new_state"])
        self.assertEqual("dismiss", result["decision"])

    @patch("backend.services.review_workflow_service.build_folder_review_summary")
    @patch("backend.services.review_workflow_service.storage.update_document")
    @patch("backend.services.review_workflow_service.storage.get_document")
    def test_apply_whitelist_decision_adds_predicted_type(self, get_document, update_document, build_summary) -> None:
        get_document.return_value = _doc(
            "doc-a",
            folder_id="invoices-q1",
            tenant_id="tenant-a",
            state="needs_review",
            confidence=0.94,
            predicted="contract",
            extra_metadata={"doc_review_whitelist_types": ["invoice"]},
        )
        build_summary.return_value = {"review_status": "clear", "open_flag_count": 0}

        result = apply_review_decision(
            folder_id="invoices-q1",
            doc_id="doc-a",
            tenant_id="tenant-a",
            decision="whitelist",
            reviewer={"id": "u-2", "name": "B Reviewer", "email": "b@example.com", "type": "human"},
            whitelist_predicted_type=True,
        )

        payload = update_document.call_args.kwargs.get("metadata")
        self.assertEqual(True, payload["doc_review_whitelisted"])
        self.assertIn("invoice", payload["doc_review_whitelist_types"])
        self.assertIn("contract", payload["doc_review_whitelist_types"])
        self.assertEqual("whitelist", result["decision"])

    @patch("backend.services.review_workflow_service.storage.get_document")
    def test_invalid_decision_raises_value_error(self, get_document) -> None:
        get_document.return_value = _doc(
            "doc-a",
            folder_id="invoices-q1",
            tenant_id="tenant-a",
            state="needs_review",
            confidence=0.94,
            predicted="contract",
        )

        with self.assertRaises(ValueError):
            apply_review_decision(
                folder_id="invoices-q1",
                doc_id="doc-a",
                tenant_id="tenant-a",
                decision="banish",
                reviewer={"id": "u-3", "name": "C Reviewer", "email": "c@example.com", "type": "human"},
            )


if __name__ == "__main__":
    unittest.main()
