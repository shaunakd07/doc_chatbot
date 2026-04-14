# Current Test Strategy and Coverage

This document describes the current testing surface in the repository rather than a purely aspirational plan.

## 1. What is automated today

The repo currently has two complementary testing layers.

### 1.1 Direct pytest coverage

There are focused pytest modules under `tests/` for core behaviors such as:

- router schema behavior
- cross-document chat behavior
- document classifier service
- hybrid metadata-semantic retrieval behavior
- metadata semantic adapter behavior
- out-of-place detection
- review workflow service
- bug tournament flow, scoring, and subprocess orchestration

Representative files:

- `tests/test_chat_service_cross_doc.py`
- `tests/test_document_classifier_service.py`
- `tests/test_hybrid_metadata_semantic.py`
- `tests/test_metadata_semantic_adapter.py`
- `tests/test_no_hardcoded_canonical_queries.py`
- `tests/test_out_of_place_detection.py`
- `tests/test_review_workflow_service.py`
- `tests/test_router_schema.py`
- `tests/test_bug_tournament_flow.py`
- `tests/test_bug_tournament_scoring.py`
- `tests/test_bug_tournament_subprocess_flow.py`

### 1.2 API/DB harness coverage

The higher-level harness under `tests/harness/` drives:

- document uploads
- readiness polling
- chat probes
- DB inspections
- evidence capture
- multi-case regression runs using YAML-defined scenarios

It is launched through `scripts/run_test_harness.py`.

## 2. Corpus and dataset selectors

The harness does not depend on hardcoded file-count assumptions in documentation. The authoritative selectors live in:

- `tests/harness/scenarios/datasets.yaml`

Current test data sources in the repo include:

- canonical LakeRunner documents
- invoice PDFs and one DOCX invoice document
- a spreadsheet
- diagram-heavy image files
- large image subsets from `test_docs/spdocvqa_images/`

For repeatable test selection, use the dataset selectors rather than manually counting files from prose docs.

## 3. Environment prerequisites

Recommended local test environment:

1. Python 3.12 environment such as `.venv312`
2. Installed dependencies from `requirements.txt`
3. `OPENAI_API_KEY` configured
4. Tesseract installed if OCR-backed paths are being exercised
5. LibreOffice installed if PPTX slide rendering behavior needs to be exercised
6. Either:
   - PostgreSQL + pgvector for parity with `.env.example`, or
   - SQLite for simpler local runs

Before execute-mode harness runs, confirm:

- `GET /api/health` returns `status=ok`
- the configured DB is reachable
- the target corpus files exist under `test_docs/`

## 4. Recommended commands

### 4.1 Direct pytest

```powershell
.\.venv312\Scripts\python.exe -m pytest tests
```

Run a narrower slice when iterating:

```powershell
.\.venv312\Scripts\python.exe -m pytest tests/test_review_workflow_service.py
.\.venv312\Scripts\python.exe -m pytest tests/test_hybrid_metadata_semantic.py
```

### 4.2 Harness dry-run

```powershell
.\.venv312\Scripts\python.exe scripts/run_test_harness.py --mode dry-run --profile smoke
```

### 4.3 Harness execute against a running backend

```powershell
.\.venv312\Scripts\python.exe scripts/run_test_harness.py --mode execute --profile small_regression --api-base-url http://127.0.0.1:8000
```

### 4.4 Harness execute with managed backend process

```powershell
.\.venv312\Scripts\python.exe scripts/run_test_harness.py --mode execute --profile smoke --manage-app --app-cmd ".venv312\\Scripts\\python.exe -m backend.app"
```

## 5. Harness suites and current intent

The scenario manifest in `tests/harness/scenarios/cases.yaml` is organized into suites `A` through `H`.

### Suite A: Ingestion lifecycle and API contract

Covers:

- upload lifecycle
- readiness transitions
- basic chunk and embedding creation
- mixed upload safety

### Suite B: Extractor correctness

Covers:

- PDF text/image behavior
- PPTX text, table, and image extraction
- LibreOffice-dependent slide rendering path
- DOCX and spreadsheet extraction

### Suite C: OCR validation

Covers:

- OCR on canonical diagram inputs
- OCR resilience on transparent PNGs
- OCR behavior on invoice PDFs
- OCR-heavy batch behavior

### Suite D: Diagram pipeline validation

Covers:

- graph extraction on diagram images
- false-positive control
- node and edge chunk presence
- PPTX slide graph generation
- diagram-heavy load behavior

### Suite E: Persistence integrity

Covers:

- chunk non-emptiness
- embedding cardinality
- embedding dimension consistency
- source-type diversity
- delete and delete-all cleanup

### Suite F: Retrieval and chat grounding

Covers:

- semantic QA
- invoice retrieval
- diagram-oriented questions
- relationship questions
- cross-document compare
- source scoping validation
- invalid scope rejection

### Suite G: Scale and durability

Covers:

- medium-scale image ingestion
- full-corpus long runs
- mixed-format stress
- restart durability

### Suite H: Regression baselines

Covers:

- source-type distribution baselines
- diagram metric baselines
- OCR baselines
- retrieval/citation baselines

## 6. Harness profiles and their current use

The current profiles in `tests/harness/scenarios/profiles.yaml` are:

- `smoke`
- `small_regression`
- `medium_scale`
- `full_automated`
- `full_longrun`
- `manual_resilience`

Practical usage:

- use `smoke` while changing app startup, API contracts, or basic ingestion
- use `small_regression` for day-to-day backend changes
- use `medium_scale` for OCR/diagram throughput work
- use `full_automated` for broad validation
- use `full_longrun` only when you explicitly want the largest corpus run
- use `manual_resilience` when validating restart behavior

## 7. Harness assertion coverage

The implemented condition kinds in `tests/harness/assertions.py` currently include:

- `doc_count_equals`
- `all_docs_ready`
- `each_doc_num_pages_gt`
- `each_doc_min_chunks`
- `each_doc_embeddings_match_chunks`
- `any_doc_has_source_types`
- `each_doc_has_source_types`
- `each_doc_min_diagram_graphs`
- `chat_non_empty`
- `chat_sources_scoped`
- `chat_any_error`
- `chat_expected_errors`
- `db_table_non_empty`
- `db_table_empty`
- `single_embedding_dim`

Unknown condition kinds are marked as `skip` rather than crashing the harness.

## 8. What is still partial or manual

Not all scenario coverage is fully headless.

Current partial or manual areas include:

- LibreOffice-dependent PPTX rendering checks
- OCR-heavy and diagram-heavy scale runs
- restart/resilience scenarios
- extremely large corpus runs

Those cases are already labeled in `cases.yaml` with `automation_level: partial` or `automation_level: manual`.

## 9. Evidence and outputs

Harness runs write evidence under:

- `harness_runs/<run_id>/`

Typical outputs:

- `summary.json`
- `summary.md`
- `events/timeline.jsonl`
- `snapshots/run_context.json`
- `snapshots/cases/<case_id>.json`
- `api/health.json`
- `logs/managed_app.log` when managed-app mode is used

These artifacts are the main source of truth for debugging failed harness runs.

## 10. Current gaps and operational notes

- There is no repository-level CI definition in this repo today, so test execution remains a local or ad hoc process.
- The frontend review inputs are not fully wired to upload headers, so review workflow validation is more reliable through direct API or service-level tests than through the browser UI.
- OCR, PPTX rendering, and diagram extraction are environment-sensitive. A passing run on one machine does not guarantee identical results on another without matching system dependencies.
- `.env.example` defaults to PostgreSQL, but many targeted local tests are still easier to iterate in SQLite mode.

## 11. Practical release bar

For a meaningful local validation pass on backend changes, the current minimum bar should usually include:

1. targeted direct pytest for the touched area
2. `smoke` or `small_regression` harness run
3. manual `GET /api/health` verification
4. at least one end-to-end upload plus chat sanity check against the local backend

For OCR, diagram, or retrieval changes, add the relevant partial or scale profiles before treating the change as stable.
