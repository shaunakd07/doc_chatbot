# Test Harness Stack

This document describes the current harness implementation under `tests/harness/`.

## 1. Purpose

The harness operationalizes higher-level ingestion, retrieval, chat, and persistence checks without hardcoding one-off scripts.

It currently supports:

1. API-driven upload, poll, chat, and delete flows
2. SQLite or PostgreSQL inspection
3. deterministic corpus selection from YAML selectors
4. run artifacts and per-case evidence capture
5. dry-run planning and execute mode
6. optional managed backend process startup

Related doc:

- [tests.md](tests.md)

## 2. Current layout

- `tests/harness/types.py`
  Dataclasses for selectors, cases, probes, conditions, and summaries.

- `tests/harness/config.py`
  Loads selectors, cases, and profiles from YAML.

- `tests/harness/catalog.py`
  Resolves dataset selectors to deterministic file lists.

- `tests/harness/api_client.py`
  Wrapper for `/api/health`, `/api/documents`, `/api/chat`, and delete flows.

- `tests/harness/db_inspector.py`
  DB inspection utilities for SQLite and PostgreSQL.

- `tests/harness/assertions.py`
  Implements the assertion engine used by case checks.

- `tests/harness/evidence.py`
  Writes event streams, snapshots, and summaries.

- `tests/harness/log_capture.py`
  Optional managed backend process logging.

- `tests/harness/runner.py`
  Core orchestration for case execution.

- `tests/harness/cli.py`
  Command-line entrypoint.

- `scripts/run_test_harness.py`
  Thin launcher that forwards to the CLI.

- `tests/harness/scenarios/datasets.yaml`
  Canonical dataset selectors.

- `tests/harness/scenarios/cases.yaml`
  Full case manifest.

- `tests/harness/scenarios/profiles.yaml`
  Runnable profile definitions.

## 3. Modes

### `dry-run`

Behavior:

- loads selectors, cases, and profiles
- resolves datasets
- selects cases for the chosen profile
- writes run metadata and planned case snapshots
- does not call the backend or mutate the DB

Use this when:

- verifying selector paths
- reviewing which cases a profile will run
- checking evidence output structure without touching the app

### `execute`

Behavior:

- calls the backend API
- waits for documents to reach terminal states when configured
- runs chat probes
- captures DB evidence
- evaluates assertions
- writes full run artifacts

Use this when you need a real validation pass against the running app.

## 4. Current CLI surface

The harness CLI in `tests/harness/cli.py` currently supports:

- `--mode dry-run|execute`
- `--profile`
- `--api-base-url`
- `--api-timeout-sec`
- `--datasets-file`
- `--cases-file`
- `--profiles-file`
- `--evidence-root`
- `--run-id`
- `--manage-app`
- `--app-cmd`
- `--startup-timeout-sec`
- `--db-backend`
- `--sqlite-db-path`
- `--database-url`

This is the authoritative flag set; older docs that mention different knobs are stale.

## 5. Commands

Run from repo root.

### Dry-run smoke

```powershell
.\.venv312\Scripts\python.exe scripts/run_test_harness.py --mode dry-run --profile smoke
```

### Execute against an already running backend

```powershell
.\.venv312\Scripts\python.exe scripts/run_test_harness.py --mode execute --profile small_regression --api-base-url http://127.0.0.1:8000
```

### Execute with managed backend startup

```powershell
.\.venv312\Scripts\python.exe scripts/run_test_harness.py --mode execute --profile smoke --manage-app --app-cmd ".venv312\\Scripts\\python.exe -m backend.app"
```

If `--app-cmd` is omitted in managed-app mode, the harness defaults to the current Python interpreter with `-m backend.app`.

## 6. Current profiles

The current profiles in `tests/harness/scenarios/profiles.yaml` are:

- `smoke`
- `small_regression`
- `medium_scale`
- `full_automated`
- `full_longrun`
- `manual_resilience`

These profiles are the supported entrypoints for routine harness use.

## 7. Current assertion coverage

Implemented condition kinds in `tests/harness/assertions.py`:

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

If a case references an unknown condition kind, the harness records a `skip` instead of crashing.

## 8. Evidence artifacts

By default each run writes to:

- `harness_runs/<run_id>/`

Common artifacts:

1. `summary.json`
2. `summary.md`
3. `events/timeline.jsonl`
4. `snapshots/run_context.json`
5. `snapshots/cases/<case_id>.json`
6. `api/health.json`
7. `logs/managed_app.log` when managed-app mode is used

These outputs are intended to support post-run triage without an immediate rerun.

## 9. Dataset selectors

Current selector kinds supported by the harness configuration layer:

1. `file`
2. `paths`
3. `dir_all`
4. `dir_first_n`
5. `glob`

Selectors are resolved deterministically so that repeated runs remain comparable.

## 10. Safety defaults

Current harness defaults from `tests/harness/scenarios/cases.yaml` include:

- `reset_before: true`
- `delete_after: false`
- `delete_all_after: false`
- `wait_ready: true`
- `ready_timeout_sec: 2400`
- `poll_interval_sec: 3.0`

Operational effect:

- cases usually start from a clean document store
- long-running OCR/diagram cases are given substantial ready-timeout headroom

## 11. Current limitations

- Environment-dependent behavior such as Tesseract availability, LibreOffice rendering, and GPU configuration still affects outcomes.
- Large corpus cases can take a long time; use the smaller profiles for routine iteration.
- The harness validates backend behavior, not rich frontend interaction flows.
- Restart/resilience scenarios are still manual by design.

## 12. How to extend it

To add a new automated scenario:

1. add or update dataset selectors in `datasets.yaml`
2. add a case in `cases.yaml`
3. add the case to an existing profile or create a new profile in `profiles.yaml`
4. implement any new assertion kind in `assertions.py` if needed

To add a new evidence type:

1. extend `runner.py`
2. write the new payload into the case snapshot or a dedicated artifact path

When updating the harness docs, prefer the YAML files and CLI implementation as the source of truth.
