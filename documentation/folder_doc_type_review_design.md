# Folder Document-Type Review Workflow

This document describes the current backend implementation and the remaining gaps in the user-facing workflow.

## 1. Current implementation status

Implemented in the repo today:

- document-type classification metadata during ingestion
- pluggable classifier providers
- folder-level out-of-place detection after ingestion
- review summary and review decision APIs
- reviewer audit history stored in document metadata
- Drive import support for expected type, folder id, threshold, and whitelist settings

Partially implemented:

- single-file upload review metadata is supported by backend request headers
- the frontend currently shows review-related inputs, but `frontend/app.js` does not yet send those headers during upload

Not implemented:

- hard blocking of uploads based on predicted document type

The workflow is advisory. Uploads still succeed and flagged items are marked for review.

## 2. Provider status

Current classifier providers in `backend/services/document_classifier.py`:

- `heuristic`
- `semantic_openai`
- `azure_document_intelligence`

Current default from `.env.example`:

- `DOC_TYPE_CLASSIFIER_PROVIDER=heuristic`

That means Azure is supported but not the default runtime path.

## 3. Architecture overview

### 3.1 Upload and import layer

Relevant handlers in `backend/app.py`:

- `POST /api/documents`
- `POST /api/documents/drive`

Supported review inputs:

- `folder_id`
- `expected_doc_type`
- optional `doc_review_threshold`
- optional `doc_review_whitelist_types`

Single-file upload path:

- review metadata is accepted from headers such as `x-folder-id` and `x-expected-doc-type`

Drive import path:

- review metadata is accepted in the JSON body

### 3.2 Ingestion pipeline

`backend/ingestion/pipeline.py` writes baseline classification-related metadata such as:

- `doc_type`
- `doc_type_confidence`
- `doc_type_scores`

### 3.3 Post-ingestion detection

`backend/services/ingestion_queue.py` triggers `detectOutOfPlaceDocuments()` after a document finishes ingesting if the document metadata contains:

- `folder_id`
- `expected_doc_type`

### 3.4 Detection service

`backend/services/out_of_place_detection.py` applies the current mismatch rules:

- predicted type differs from expected type after equivalence handling
- predicted type is not ignored
- predicted type is not whitelisted
- the document is not manually overridden
- confidence is above threshold
- score ratio is above minimum

### 3.5 Review workflow service

`backend/services/review_workflow_service.py` aggregates folder review state and applies reviewer decisions.

Supported decisions:

- `dismiss`
- `accept`
- `whitelist`
- `reopen`

## 4. Current data model

Review and classification data is stored in `documents.metadata`.

Classification fields:

- `doc_type`
- `doc_type_confidence`
- `doc_type_scores`
- `doc_type_classifier_provider`
- `doc_type_classifier_model`

Folder intent fields:

- `folder_id`
- `expected_doc_type`
- `doc_review_threshold`
- `doc_review_whitelist_types`

Review state and audit fields:

- `out_of_place_review_state`
- `out_of_place_review_action_required`
- `out_of_place_review_reason`
- `out_of_place_review_confidence`
- `out_of_place_review_expected_type`
- `out_of_place_review_predicted_type`
- `out_of_place_review_manual_override`
- `out_of_place_review_last_checked_at`
- `out_of_place_review_run_id`
- `out_of_place_review_last_transition`
- `out_of_place_review_audit`
- `out_of_place_review_decisions`
- `out_of_place_review_last_flag_count`

## 5. API contracts

### 5.1 Single file upload with review metadata

Endpoint:

- `POST /api/documents`

Headers:

- `x-tenant-id`
- `x-folder-id`
- `x-expected-doc-type`
- `x-doc-review-threshold`
- `x-doc-review-whitelist-types`

Response example:

```json
{
  "doc_id": "b44d...",
  "status": "queued",
  "folder_id": "invoices-2026-q1",
  "expected_type": "invoice",
  "review_mode": "soft_warning"
}
```

### 5.2 Drive import with review metadata

Endpoint:

- `POST /api/documents/drive`

Body example:

```json
{
  "url": "https://drive.google.com/...",
  "folder_id": "invoices-2026-q1",
  "expected_type": "invoice",
  "review_threshold": 0.92,
  "whitelist_types": ["receipt"]
}
```

### 5.3 Review status endpoints

- `GET /api/folders/{folder_id}/review-flags`
- `GET /api/folders/{folder_id}/review-summary`
- `GET /api/folders/{folder_id}/out-of-place?expected_type=invoice&threshold=0.92`

### 5.4 Review decision endpoint

- `POST /api/folders/{folder_id}/review-decisions`

Body example:

```json
{
  "doc_id": "doc-123",
  "decision": "dismiss",
  "note": "Known exception",
  "whitelist_predicted_type": false
}
```

Reviewer metadata is captured from headers such as:

- `x-user-id`
- `x-user-name`
- `x-user-email`
- `x-user-role`
- `x-user-type`

## 6. Current rule behavior

The current implementation is intentionally high-precision and advisory.

Important behaviors:

- folder review only runs when folder metadata is present
- non-ready documents are not treated as final review outcomes
- equivalent-type mappings can suppress false positives
- tenant-level thresholds and per-folder overrides are supported
- reviewer overrides are respected on future runs

This is designed to reduce noisy flagging for enterprise document sets.

## 7. Configuration

Key environment variables:

- `DOC_TYPE_REVIEW_ENABLED`
- `DOC_TYPE_REVIEW_CONFIDENCE_THRESHOLD`
- `DOC_TYPE_REVIEW_MIN_SCORE_RATIO`
- `DOC_TYPE_REVIEW_TENANT_THRESHOLDS`
- `DOC_TYPE_REVIEW_EQUIVALENT_TYPES`
- `DOC_TYPE_REVIEW_IGNORED_PREDICTED_TYPES`
- `DOC_TYPE_CLASSIFIER_PROVIDER`
- `DOC_TYPE_SEMANTIC_MODEL`
- `AZURE_DOC_INTELLIGENCE_ENDPOINT`
- `AZURE_DOC_INTELLIGENCE_API_KEY`
- `AZURE_DOC_INTELLIGENCE_CLASSIFIER_ID`

Per-folder runtime overrides:

- `doc_review_threshold`
- `doc_review_whitelist_types`

## 8. Security and audit considerations

Current design choices:

- tenant filtering is applied during review lookup and detection
- uploads are not blocked; the workflow produces review signals instead
- reviewer identity is captured in decision metadata
- automated detection runs append audit entries
- document text is not intentionally written into review audit logs

## 9. Current gaps and next logical work

The main remaining product gap is frontend wiring.

Practical next steps if this workflow is being productized:

1. wire the review inputs in `frontend/app.js` into upload headers for `POST /api/documents`
2. add frontend summary and decision views for folder review endpoints
3. add end-to-end browser-level tests once the UI path exists
4. validate threshold tuning with tenant-specific real corpora

The backend foundation already exists; the UI path is the missing piece.
