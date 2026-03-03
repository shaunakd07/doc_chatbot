# Test Harness Stack

## Purpose

This harness operationalizes the test strategy in [tests.md](D:/shaun/Projects/doc_chatbot/docs/tests.md) without hardcoding one-off scripts.

It supports:

1. API-driven ingestion/retrieval test execution.
2. DB integrity inspection (SQLite or Postgres).
3. Deterministic test corpus selection from `test_docs/`.
4. Per-case evidence capture in machine-readable artifacts.
5. Dry-run planning mode (no API calls) and execute mode.

## Prerequisites

1. Python environment with `requirements.txt` installed (harness uses `httpx` and `PyYAML` directly).
2. `test_docs/` populated with the corpus referenced by `tests/harness/scenarios/datasets.yaml`.
3. For `execute` mode: running backend API (`/api/health`, `/api/documents`, `/api/chat`) and configured DB access for inspector checks.

## Layout

- `tests/harness/types.py`: dataclasses for selectors, probes, conditions, cases, profiles.
- `tests/harness/config.py`: YAML loading/parsing and case/profile selection.
- `tests/harness/catalog.py`: deterministic dataset selector resolution.
- `tests/harness/api_client.py`: API client wrapper for upload/poll/chat/delete.
- `tests/harness/db_inspector.py`: DB table/doc/chunk/embedding/graph inspection.
- `tests/harness/assertions.py`: condition evaluation engine.
- `tests/harness/evidence.py`: timeline + artifact writer.
- `tests/harness/log_capture.py`: optional managed backend process logging.
- `tests/harness/runner.py`: core case orchestration.
- `tests/harness/cli.py`: CLI entrypoint.
- `scripts/run_test_harness.py`: convenience launcher.
- `tests/harness/scenarios/datasets.yaml`: file selectors.
- `tests/harness/scenarios/cases.yaml`: A-H case manifest.
- `tests/harness/scenarios/profiles.yaml`: runnable profile definitions.

## Core Modes

### `dry-run`

- Resolves dataset selectors.
- Selects cases by profile.
- Writes run plan and case snapshots.
- Does **not** call API or DB mutation endpoints.

### `execute`

- Calls API endpoints for upload/poll/chat/delete.
- Collects DB evidence and assertions.
- Writes summary + per-case outputs.

## Commands

Run from repo root.

### Dry-run smoke

```bash
python scripts/run_test_harness.py --mode dry-run --profile smoke
```

### Execute small regression against running backend

```bash
python scripts/run_test_harness.py --mode execute --profile small_regression --api-base-url http://127.0.0.1:8000
```

### Execute with managed app process

```bash
python scripts/run_test_harness.py --mode execute --profile smoke --manage-app
```

Optional managed app command override:

```bash
python scripts/run_test_harness.py --mode execute --profile smoke --manage-app --app-cmd ".venv312\\Scripts\\python.exe -m backend.app"
```

## Evidence Artifacts

By default, each run writes to:

- `harness_runs/<run_id>/`

Key files:

1. `summary.json`: machine-readable run summary.
2. `summary.md`: human-readable summary.
3. `events/timeline.jsonl`: timestamped event stream.
4. `snapshots/run_context.json`: run metadata and selected case IDs.
5. `snapshots/cases/<case_id>.json`: full per-case evidence.
6. `api/health.json`: health snapshot (or dry-run note).
7. `logs/managed_app.log`: backend output (managed app mode).

## Dataset Selectors

Supported selector kinds:

1. `file`: single file path.
2. `paths`: explicit file list.
3. `dir_all`: all files matching pattern in dir.
4. `dir_first_n`: deterministic first N files matching pattern.
5. `glob`: pattern search (recursive configurable).

Selectors are deterministic and sorted naturally to keep repeated runs comparable.

## Assertion Engine Coverage

Implemented condition kinds:

1. `doc_count_equals`
2. `all_docs_ready`
3. `each_doc_num_pages_gt`
4. `each_doc_min_chunks`
5. `each_doc_embeddings_match_chunks`
6. `any_doc_has_source_types`
7. `each_doc_has_source_types`
8. `each_doc_min_diagram_graphs`
9. `chat_non_empty`
10. `chat_sources_scoped`
11. `chat_any_error`
12. `chat_expected_errors`
13. `db_table_non_empty`
14. `db_table_empty`
15. `single_embedding_dim`

Unknown condition kinds are recorded as `skip` rather than crashing the run.

## Case Automation Levels

From `cases.yaml`:

1. `full`: expected to run headlessly.
2. `partial`: mostly automated, but interpretation/environment may require review.
3. `manual`: operator-required (e.g., restart-resilience scenarios).

Profiles control whether manual cases are included.

## Safety and Reproducibility

1. Default behavior is case-level reset (`DELETE /api/documents`) before each case.
2. Scale cases are isolated via profile selection.
3. Dataset selection is deterministic.
4. Case artifacts include enough evidence for post-run triage without rerun.

## Extension Guide

To add a new test:

1. Add/extend selectors in `datasets.yaml`.
2. Add case in `cases.yaml`.
3. Add/adjust profile entries in `profiles.yaml`.
4. If needed, implement new assertion kinds in `assertions.py`.

To add a new evidence type:

1. Extend runner collection logic.
2. Write to `snapshots/cases/<case_id>.json` and/or structured subfolder.
