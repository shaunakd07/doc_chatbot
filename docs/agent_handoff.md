# Agent Handoff

## Project State
- Current milestone: Step 6 completed
- Status: complete
- Last updated: 2026-03-22
- Updated by: Codex

## Global Rules
- Follow `AGENTS.md`
- Read this file before starting work if prompted
- Do not implement beyond `Next Step`
- Do not rely on prior chat context
- Re-ground in current code and tests before making changes
- Treat `Next Step` as binding
- Treat later planned steps as roadmap only until they become `Next Step`

## Completed Milestones

### Post-roadmap Task: Named-document summary routing and compare exact-fact composition
- Status: complete
- Objective:
  - Keep named-document summary requests on single-document retrieval by default
  - Allow compare queries to combine exact fact lookup with comparison retrieval instead of forcing one or the other

- What was implemented:
  - Tightened both router prompts so named-document summary requests prefer `qa` with `needs_cross_doc=false`, while compare-plus-fact questions can emit `exact_lookup_requested`, `fact_types`, and `target_documents`
  - Extended both router parsers to preserve `analysis_plan.exact_lookup_requested`, `analysis_plan.fact_types`, and `analysis_plan.target_documents`
  - Added a named-document summary route repair in `backend/services/chat_service.py` so a strong single target document can override an overly broad cross-doc summary route
  - Scoped single-document QA to route-resolved target documents when the router identifies a single named document
  - Composed compare intent with exact lookup by allowing compare queries through the exact-lookup classifier, running exact fact lookup alongside compare retrieval, and merging both evidence sets
  - Added compare-side exact lookup coverage tracking so compare fact hits are only treated as strong when evidence spans both compared documents

- Files modified:
  - `backend/services/openai_router_service.py`
  - `backend/services/router_service.py`
  - `backend/services/chat_service.py`
  - `tests/test_router_schema.py`
  - `tests/test_chat_service_cross_doc.py`
  - `tests/test_document_fact_ingestion.py`
  - `docs/agent_handoff.md`

- Key decisions:
  - Kept `task_type` and exact lookup as orthogonal concerns so compare answers can still use compare prompting while exact fact evidence is added underneath
  - Added a deterministic chat-service repair for named-document summaries rather than relying on router prompt wording alone
  - Reused the existing exact-lookup classifier and fact store instead of adding a separate compare-only fact pipeline
  - Scoped route-resolved target documents through existing document-match helpers to preserve current document metadata contracts

- Invariants to preserve:
  - Single-document summary requests should not broaden to cross-doc retrieval when a strong named-document match exists
  - Compare queries must still use compare-style retrieval and output behavior
  - Exact fact lookup must remain additive over existing compare retrieval rather than replacing it
  - Cross-document compare fact evidence should be considered strong only when both compared sides are represented
  - Existing router schema compatibility for prior keys must remain intact

- Tests run:
  - `.\.venv312\Scripts\python.exe -m py_compile backend\services\openai_router_service.py backend\services\router_service.py backend\services\chat_service.py tests\test_chat_service_cross_doc.py tests\test_document_fact_ingestion.py tests\test_router_schema.py`
  - `.\.venv312\Scripts\python.exe -m pytest tests\test_router_schema.py tests\test_chat_service_cross_doc.py tests\test_document_fact_ingestion.py`
  - `.\.venv312\Scripts\python.exe -m pytest tests\test_router_schema.py tests\test_chat_service_cross_doc.py tests\test_document_fact_ingestion.py tests\test_hybrid_metadata_semantic.py`
  - `.\.venv312\Scripts\python.exe -c "from unittest.mock import patch; from fastapi.testclient import TestClient; import backend.app as app_mod; p = patch.object(app_mod, '_initialize_app_state', lambda app: None); p.start(); client = TestClient(app_mod.app); response = client.get('/api/health'); print(response.status_code); print(response.json()['status']); client.close(); p.stop()"`

- Test results:
  - syntax check: passed
  - targeted routing/chat/fact slice: 40 passed, 1 warning
  - broader regression slice with hybrid metadata: 44 passed, 1 warning
  - health verification: HTTP 200, status `ok`

- Known limitations:
  - Router-originated `target_documents` still depend on model quality; the deterministic summary repair currently covers only named-document summary-like phrasings
  - Compare-side exact lookup currently uses the existing fact types (`date`, `amount`, `party`) and does not add new fact categories

- Blockers:
  - None

### Post-roadmap Task: Chat routing and retrieval diagnostics logging
- Status: complete
- Objective:
  - Add much richer backend logs for chat requests so routing, retrieval, and response decisions are visible during debugging

- What was implemented:
  - Added structured `INFO` logs in `backend/services/chat_service.py` for `chat.route` and `chat.answer`
  - Logged route fields including `task_type`, `needs_cross_doc`, `needs_numeric_extraction`, `needs_image_reasoning`, retrieval mode/plan, route source/confidence, analysis plan, exact-lookup hints, and resolved document scopes
  - Logged response/evidence summaries including exact-lookup outcome, chunk/source counts, source type distribution, top chunk metadata, context block count, image count, and answer length
  - Added structured API-layer `INFO` logs in `backend/app.py` for `api.chat.request` and `api.chat.response`
  - Logged API request envelope fields including conversation id, message preview, requested/scoped doc ids, `top_k`, and summary toggle
  - Logged API response envelope fields including duration, intent, answer length, source count, route task fields, retrieval plan, and exact-lookup summary
  - Added focused regression tests for both service-level and API-level logging output

- Files modified:
  - `backend/app.py`
  - `backend/services/chat_service.py`
  - `tests/test_app_chat_logging.py`
  - `tests/test_chat_service_cross_doc.py`
  - `docs/agent_handoff.md`

- Key decisions:
  - Kept logs structured and JSON-encoded so route and retrieval fields are easy to grep and parse
  - Logged summaries rather than full chunk contents or full answers to preserve signal while avoiding noisy, oversized log lines
  - Added logs at both the API boundary and the chat-service decision layer so request context and internal routing state are both visible

- Invariants to preserve:
  - Successful chat requests should now emit informative `INFO` logs without changing API responses
  - Logging must not require frontend changes or alter retrieval behavior
  - Log payloads should remain serializable and bounded in size

- Tests run:
  - `.\.venv312\Scripts\python.exe -m py_compile backend\app.py backend\services\chat_service.py tests\test_chat_service_cross_doc.py tests\test_app_chat_logging.py`
  - `.\.venv312\Scripts\python.exe -m pytest tests\test_app_chat_logging.py tests\test_chat_service_cross_doc.py -k "log"`
  - `.\.venv312\Scripts\python.exe -m pytest tests\test_app_chat_logging.py tests\test_chat_service_cross_doc.py tests\test_document_fact_ingestion.py tests\test_router_schema.py tests\test_hybrid_metadata_semantic.py`
  - `.\.venv312\Scripts\python.exe -c "from unittest.mock import patch; from fastapi.testclient import TestClient; import backend.app as app_mod; p = patch.object(app_mod, '_initialize_app_state', lambda app: None); p.start(); client = TestClient(app_mod.app); response = client.get('/api/health'); print(response.status_code); print(response.json()['status']); client.close(); p.stop()"`

- Test results:
  - syntax check: passed
  - focused logging slice: 2 passed, 25 deselected, 1 warning
  - broader regression slice: 46 passed, 1 warning
  - health verification: HTTP 200, status `ok`

- Known limitations:
  - Logs summarize top chunks and source distributions rather than printing full retrieved evidence bodies
  - Frontend logs remain client-side activity logs; this change improves backend/server logs only

- Blockers:
  - None

### Step 1: Document fact index and ingestion extraction
- Status: complete
- Goal:
  - Add a portable `document_facts` index for SQLite and Postgres
  - Extract and persist dates, amounts, and parties during ingestion with provenance

- What was implemented:
  - Added a portable `document_facts` table in both SQLite and Postgres with JSON metadata handling aligned to the existing storage layer
  - Added storage helpers to insert and list document facts without changing existing retrieval behavior
  - Added deterministic ingestion-time extraction for `date`, `amount`, and `party` facts from final chunk text
  - Preserved provenance on each fact via `doc_id`, `page`, `chunk_id`, and `evidence_text`
  - Wired fact persistence into normal ingestion after chunk persistence and before embeddings/index updates
  - Added an `ingest_fact_count` metadata field on the document record
  - Added focused tests for extraction, persistence, and ingestion wiring under SQLite

- Files modified:
  - `backend/ingestion/document_facts.py`
  - `backend/ingestion/pipeline.py`
  - `backend/storage.py`
  - `docs/agent_handoff.md`
  - `tests/test_document_fact_ingestion.py`

- Key decisions:
  - Kept extraction deterministic and rule-based only; no ML models or retrieval changes were introduced
  - Limited amount extraction to values with explicit currency indicators to reduce noise in mixed corpora
  - Used normalized sentence-level evidence text for provenance, but did not implement any retrieval-time span selector
  - Stored fact metadata in the existing cross-DB JSON/TEXT style already used elsewhere in storage
  - Kept `document_facts` separate from `documents` and `chunks` metadata to preserve the current architecture

- Invariants to preserve:
  - Must remain compatible with SQLite and Postgres
  - Do not break the existing ingestion pipeline
  - Do not change retrieval behavior yet
  - Do not change API contracts yet
  - Do not move exact lookup into retrieval until Step 2
  - Keep fact provenance aligned to current chunk ids and page numbering

- Tests run:
  - `.\.venv312\Scripts\python.exe -m pytest tests\test_document_fact_ingestion.py`
  - `.\.venv312\Scripts\python.exe -m pytest tests\test_document_fact_ingestion.py tests\test_chat_service_cross_doc.py tests\test_hybrid_metadata_semantic.py tests\test_router_schema.py`
  - `.\.venv312\Scripts\python.exe -c "from unittest.mock import patch; from fastapi.testclient import TestClient; import backend.app as app_mod; p = patch.object(app_mod, '_initialize_app_state', lambda app: None); p.start(); client = TestClient(app_mod.app); response = client.get('/api/health'); print(response.status_code); print(response.json()['status']); client.close(); p.stop()"`

- Test results:
  - `tests/test_document_fact_ingestion.py`: 3 passed
  - combined regression slice: 30 passed, 1 warning
  - health verification: HTTP 200, status `ok`

- Known limitations:
  - Party extraction is intentionally conservative and currently depends on deterministic clause/label patterns
  - Amount extraction currently requires an explicit currency code or `$` symbol
  - Date extraction currently targets parseable full dates, not broader temporal inference
  - No exact-lookup retrieval path uses `document_facts` yet
  - No sentence/span selector has been added yet

- Blockers:
  - None

### Step 2: Exact-lookup retrieval path
- Status: complete
- Objective:
  - Query the fact index first for exact-value, date, party, and "which document contains X" questions, then fall back to hybrid retrieval when the fact index is weak or empty.

- What was implemented:
  - Added exact-lookup classification for direct fact questions and "which document contains X" requests
  - Queried `document_facts` before normal hybrid retrieval for matching date, amount, and party lookups
  - Preserved fallback to the existing chunk retrieval paths when fact evidence was weak or empty
  - Surfaced fact evidence as `document_fact` sources while preserving document, page, and chunk provenance
  - Added focused exact-lookup tests under `tests/test_document_fact_ingestion.py`

- Invariants to preserve:
  - Fact lookup remains additive over the existing retrieval stack
  - Weak or empty fact hits must still fall back to current chunk retrieval
  - Exact-lookup evidence remains grounded to stored fact provenance

- Validation status:
  - Revalidated in the current regression slice on 2026-03-22

### Step 3: Sentence/span evidence selector
- Status: complete
- Objective:
  - Keep chunk retrieval for recall, but rank sentences or spans inside the best chunks and pass only the smallest relevant evidence spans forward.

- What was implemented:
  - Added a post-retrieval span selector in `backend/services/chat_service.py` after chunk recall and dedupe, but before prompt/context and source assembly
  - Ranked sentence, clause, and short window candidates inside retrieved text chunks using query overlap, entity matching, and upstream chunk score
  - Replaced full chunk content with the best tighter span for prompt construction, fallback answers, and returned sources when a smaller span scored better
  - Kept `document_fact` and other non-textual sources untouched so exact lookup and diagram/image evidence paths were not broadened or rewritten
  - Added focused chat-service tests proving the prompt, fallback answer, and returned sources use the selected span rather than the full chunk

- Files modified:
  - `backend/services/chat_service.py`
  - `tests/test_chat_service_cross_doc.py`
  - `docs/agent_handoff.md`

- Key decisions:
  - Kept chunk retrieval as the unchanged upstream recall layer and made span selection an additive post-processing step only
  - Limited span shrinking to `text`, `ocr`, and `table` chunks so already-compact fact sources were preserved
  - Kept original chunk ids, doc ids, and source tags stable to avoid changing downstream API contracts

- Invariants to preserve:
  - Do not change retrieval routing, API shapes, or storage contracts
  - Keep exact lookup and hybrid fallback behavior intact
  - Keep sources grounded to the same document/page/chunk provenance even when the visible content is narrowed to a smaller span

- Tests run:
  - `.\.venv312\Scripts\python.exe -m pytest tests\test_chat_service_cross_doc.py -k "selected_evidence_span or fallback_answer_uses_selected_evidence_span"`
  - `.\.venv312\Scripts\python.exe -m pytest tests\test_chat_service_cross_doc.py tests\test_document_fact_ingestion.py tests\test_hybrid_metadata_semantic.py tests\test_router_schema.py`
  - `.\.venv312\Scripts\python.exe -c "from unittest.mock import patch; from fastapi.testclient import TestClient; import backend.app as app_mod; p = patch.object(app_mod, '_initialize_app_state', lambda app: None); p.start(); client = TestClient(app_mod.app); response = client.get('/api/health'); print(response.status_code); print(response.json()['status']); client.close(); p.stop()"`

- Test results:
  - targeted span-grounding slice: 2 passed
  - broader regression slice: 35 passed, 1 warning
  - health verification: HTTP 200, status `ok`

- Known limitations:
  - Span ranking is still heuristic and does not emit character offsets or multi-span stitched evidence
  - Very short compare/trend chunks are left as-is to avoid dropping useful synthesis context
  - Sentence splitting is punctuation-based and may still be imperfect on noisy OCR text

- Blockers:
  - None

### Step 4: Weak-evidence output policy
- Status: complete
- Objective:
  - Replace the current weak-evidence fallback wording with the exact required prefix.

- What was implemented:
  - Replaced the QA weak-evidence fallback prefix with exactly `I have weak evidence, my best guess is:`
  - Kept the change scoped to the intended weak-evidence QA fallback path in `backend/services/chat_service.py`
  - Updated the metadata semantic fallback wrapper to recognize the new weak-evidence prefix so it does not prepend unrelated wording on that path
  - Added focused chat-service coverage for the exact weak-evidence prefix and updated the affected assertions that previously looked for the old partial-evidence wording

- Files modified:
  - `backend/services/chat_service.py`
  - `tests/test_chat_service_cross_doc.py`
  - `docs/agent_handoff.md`

- Key decisions:
  - Left compare/trend fallback wording unchanged because Step 4 was scoped only to the intended weak-evidence QA path
  - Used a single shared prefix constant and a narrow helper to keep the detection logic aligned with the exact required text
  - Preserved existing fallback evidence formatting after the prefix so the change stayed reviewable and localized

- Invariants to preserve:
  - All intended weak-evidence QA responses must continue to start with the exact required prefix
  - Non-weak-evidence answers and compare-style fallback wording must remain unchanged unless a later step explicitly changes them
  - The metadata strict-filter fallback must not add unrelated wording when the QA weak-evidence prefix is already present

- Tests run:
  - `.\.venv312\Scripts\python.exe -m pytest tests\test_chat_service_cross_doc.py -k "hard_no_answer_is_replaced_when_evidence_exists or weak_evidence_fallback_uses_exact_required_prefix or metadata_zero_matches_falls_back_to_semantic_evidence"`
  - `.\.venv312\Scripts\python.exe -m pytest tests\test_chat_service_cross_doc.py`
  - `.\.venv312\Scripts\python.exe -c "from unittest.mock import patch; from fastapi.testclient import TestClient; import backend.app as app_mod; p = patch.object(app_mod, '_initialize_app_state', lambda app: None); p.start(); client = TestClient(app_mod.app); response = client.get('/api/health'); print(response.status_code); print(response.json()['status']); client.close(); p.stop()"`

- Test results:
  - targeted weak-evidence slice: 3 passed, 18 deselected, 1 warning
  - `tests/test_chat_service_cross_doc.py`: 21 passed, 1 warning
  - health verification: HTTP 200, status `ok`

- Known limitations:
  - Compare/trend fallback wording still uses the existing partial-evidence phrasing by design because Step 4 did not broaden that path

- Blockers:
  - None

### Step 5: Goal-focused test expansion
- Status: complete
- Objective:
  - Add direct regression coverage for the product behaviors that matter most.

- What was implemented:
  - Added exact fact retrieval coverage for amount lookups that asserts fact-grounded sources are returned without falling back to hybrid retrieval
  - Added a "which document contains X" regression that checks the answer and sources stay scoped to the matching document only
  - Added compare-flow regression coverage that preserves conflicting evidence from two documents in the fallback comparison output and sources
  - Added span-grounding regression coverage that verifies selected spans keep the original chunk provenance in returned sources
  - Added an exact weak-evidence first-line assertion so the required prefix is validated as returned text, not only as a prefix match

- Files modified:
  - `tests/test_document_fact_ingestion.py`
  - `tests/test_chat_service_cross_doc.py`
  - `docs/agent_handoff.md`

- Key decisions:
  - Kept Step 5 backend-centered by expanding existing focused test modules instead of introducing wider workflow or frontend coverage
  - Targeted answer text and returned sources rather than only route internals so the tests reflect product-visible behavior
  - Reused the existing SQLite-backed fact test fixture and chat-service stubs to keep the added coverage deterministic and fast

- Invariants to preserve:
  - Exact fact and document-contains queries must remain grounded to `document_fact` provenance when the fact index has strong matches
  - Compare flows must continue surfacing conflicting evidence from both documents instead of collapsing to one side
  - Span selection must keep original source provenance stable while narrowing visible content
  - Weak-evidence responses must keep the exact required first line

- Tests run:
  - `.\.venv312\Scripts\python.exe -m pytest tests\test_document_fact_ingestion.py -k "exact_amount_question_returns_grounded_fact_source or document_contains_answer_names_only_matching_document"`
  - `.\.venv312\Scripts\python.exe -m pytest tests\test_chat_service_cross_doc.py -k "compare_flow_preserves_conflicting_evidence_from_both_documents or selected_evidence_span_keeps_original_source_provenance or weak_evidence_prefix_is_exact_first_line"`
  - `.\.venv312\Scripts\python.exe -m pytest tests\test_document_fact_ingestion.py tests\test_chat_service_cross_doc.py`
  - `.\.venv312\Scripts\python.exe -c "from unittest.mock import patch; from fastapi.testclient import TestClient; import backend.app as app_mod; p = patch.object(app_mod, '_initialize_app_state', lambda app: None); p.start(); client = TestClient(app_mod.app); response = client.get('/api/health'); print(response.status_code); print(response.json()['status']); client.close(); p.stop()"`

- Test results:
  - targeted exact/fact slice: 2 passed, 6 deselected, 1 warning
  - targeted compare/span/prefix slice: 3 passed, 21 deselected, 1 warning
  - combined targeted regression slice: 32 passed, 1 warning
  - health verification: HTTP 200, status `ok`

- Known limitations:
  - The new compare contradiction coverage validates the current fallback comparison path; it does not validate model-generated synthesis behavior

- Blockers:
  - None

### Step 6: Integration hardening and regression pass
- Status: complete
- Objective:
  - Validate that Steps 2-5 work together cleanly and do not degrade current chat, retrieval, or metadata behavior.

- What was implemented:
  - Re-ran the integrated backend regression slice covering exact lookup, hybrid fallback, span grounding, contradiction handling, weak-evidence policy, hybrid metadata behavior, and router schema expectations together
  - Revalidated the health endpoint after the integration pass
  - Confirmed the current Step 2-5 behavior works together cleanly without requiring runtime code changes

- Files modified:
  - `docs/agent_handoff.md`

- Key decisions:
  - Kept Step 6 as an integration validation pass only because the combined regression slice passed cleanly without exposing integration defects
  - Preserved the current architecture and API behavior because no failures required hardening changes

- Invariants confirmed:
  - Exact lookup and hybrid fallback still coexist cleanly
  - Span grounding still preserves source grounding while narrowing evidence content
  - Weak-evidence policy remains in effect on the intended QA path
  - Cross-document and hybrid-metadata behavior remain intact
  - Router schema expectations remain intact

- Tests run:
  - `.\.venv312\Scripts\python.exe -m pytest tests\test_document_fact_ingestion.py tests\test_chat_service_cross_doc.py tests\test_hybrid_metadata_semantic.py tests\test_router_schema.py`
  - `.\.venv312\Scripts\python.exe -c "from unittest.mock import patch; from fastapi.testclient import TestClient; import backend.app as app_mod; p = patch.object(app_mod, '_initialize_app_state', lambda app: None); p.start(); client = TestClient(app_mod.app); response = client.get('/api/health'); print(response.status_code); print(response.json()['status']); client.close(); p.stop()"`

- Test results:
  - integrated regression slice: 41 passed, 1 warning
  - health verification: HTTP 200, status `ok`

- Known limitations:
  - Residual risks remain the same as previously noted; this step validated the current integrated state but did not add new product behavior

- Blockers:
  - None

## Next Step

### Roadmap Status
- Status: complete
- No further planned milestones remain in this handoff. Start a new task only with a fresh user request.

## Remaining Planned Milestones

### Step 4: Weak-evidence output policy
- Status: complete
- Objective:
  - Replace the current weak-evidence fallback wording with the exact required prefix.

- Requirements:
  - Replace the current fallback prefix with exactly:
    - `I have weak evidence, my best guess is:`
  - Apply this only to the intended weak-evidence path
  - Do not silently change unrelated answer-style wording

- Out of scope:
  - broad prompt/style rewrites
  - retrieval-path redesign

- Definition of done:
  - all intended weak-evidence responses use the exact required prefix
  - existing non-weak-evidence answers remain unchanged unless necessary
  - tests covering the prefix pass
  - this handoff file is updated again

### Step 5: Goal-focused test expansion
- Status: complete
- Objective:
  - Add direct regression coverage for the product behaviors that matter most.

- Requirements:
  - Add tests for exact fact retrieval
  - Add tests for "which document contains X"
  - Add tests for contradiction handling via compare flows
  - Add tests for span-level grounding
  - Add tests for the weak-evidence prefix
  - Keep tests targeted and backend-centered

- Out of scope:
  - large unrelated test-suite refactors
  - flaky end-to-end expansion without clear value

- Definition of done:
  - the new goal-focused tests exist
  - they run cleanly in the intended local test flow
  - this handoff file is updated again

### Step 6: Integration hardening and regression pass
- Status: complete
- Objective:
  - Validate that Steps 2–5 work together cleanly and do not degrade current chat, retrieval, or metadata behavior.

- Requirements:
  - Run the focused regression slice plus any new tests introduced in prior steps
  - Verify exact lookup, hybrid fallback, span grounding, contradiction handling, and weak-evidence policy together
  - Preserve current cross-document and hybrid-metadata behavior
  - Update docs or internal notes if behavior/contracts changed

- Out of scope:
  - new feature expansion beyond the existing roadmap
  - unrelated architecture rewrites

- Definition of done:
  - all relevant targeted and regression tests pass
  - no known integration regressions remain unaddressed
  - this handoff file is updated again with the final integrated state

## Current Code Facts
- Fact index exists: yes
- Exact lookup path exists: yes
- Span grounding exists: yes
- Weak-evidence prefix updated: yes
- Goal-focused regression expansion exists: yes
- Full post-Step-1 integration hardening completed: yes


## Open Risks
- Cross-DB JSON/metadata handling may differ between SQLite and Postgres
- Deterministic party extraction may be noisy
- Provenance fields must align with the current chunk/page model
- Exact-lookup routing may over-trigger on mixed semantic/factual questions if classification is too coarse
- Span selection may accidentally discard useful surrounding context if ranking is too aggressive

## Resume Instructions for Codex
1. Read `AGENTS.md`
2. Read this file
3. Inspect storage, ingestion, retrieval, chat, and relevant test modules
4. Implement only the step asked for
5. Run targeted tests first, then broader tests if needed
6. Fix failures before stopping
7. Update this file when done
8. Advance only one step at a time by writing a `Next Step` section in this file after each completed milestone, the section describes the Next Step to be taken. 
9. Do not delete any steps in this file
