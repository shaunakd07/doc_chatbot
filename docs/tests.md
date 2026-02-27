# Document Ingestion and Processing Test Plan

## 1) Scope and Constraints

This test plan is designed for the current `doc_chatbot` codebase and test corpus under `test_docs/`.


Primary goals:

1. Validate ingestion correctness across supported document types.
2. Validate OCR + diagram pipeline behavior on real mixed-quality documents.
3. Validate chunking, embedding, and persistence integrity.
4. Validate retrieval and chat grounding quality after ingestion.
5. Validate stability under scale using the large image corpus.

## 2) Test Corpus Inventory (`test_docs/`)

Observed corpus summary:

- `12847` files total
- `12828` PNG
- `11` PDF
- `5` PPTX
- `1` XLSX
- `1` DOCX
- `1` JPG

Key canonical files (use these explicitly in test execution):

1. `test_docs/LakeRunner/LakeRunner.pdf`
2. `test_docs/LakeRunner/LakeRunner.pptx`
3. `test_docs/LakeRunner/Airtel Cloud PPT 1.pptx`
4. `test_docs/LakeRunner/Airtel Cloud Prep.pptx`
5. `test_docs/LakeRunner/Lakerunner_updated_with_licensing_multitenant 1.pptx`
6. `test_docs/LakeRunner/Lakerunner license.xlsx`
7. `test_docs/Ashnik Invoice-20260223T051635Z-1-001/Ashnik Invoice/Invoice INV-ID-21-0015.docx`
8. `test_docs/Ashnik Invoice-20260223T051635Z-1-001/Ashnik Invoice/Invoice INV-ID-21-0003.pdf`
9. `test_docs/Ashnik Invoice-20260223T051635Z-1-001/Ashnik Invoice/Invoice INV-ID-21-0015.pdf`
10. `test_docs/Ashnik Invoice-20260223T051635Z-1-001/Ashnik Invoice/Invoice INV-ID-21-0024.pdf`
11. `test_docs/Ashnik Invoice-20260223T051635Z-1-001/Ashnik Invoice/Invoice INV_19-20_051.pdf`
12. `test_docs/Ashnik Invoice-20260223T051635Z-1-001/Ashnik Invoice/Invoice INV_19-20_070.pdf`
13. `test_docs/Ashnik Invoice-20260223T051635Z-1-001/Ashnik Invoice/Invoice INV_19-20_084.pdf`
14. `test_docs/Ashnik Invoice-20260223T051635Z-1-001/Ashnik Invoice/Invoice INV_19-20_086.pdf`
15. `test_docs/Ashnik Invoice-20260223T051635Z-1-001/Ashnik Invoice/Invoice INV_19-20_169.pdf`
16. `test_docs/Ashnik Invoice-20260223T051635Z-1-001/Ashnik Invoice/Invoice INV_19-20_198.pdf`
17. `test_docs/Ashnik Invoice-20260223T051635Z-1-001/Ashnik Invoice/Invoice INV_19-20_223.pdf`
18. `test_docs/3-ML-Lifecycle-Detail.jpg`
19. `test_docs/Intelligent Retrieval System.drawio (2).png`
20. `test_docs/PNG_transparency_demonstration_1.png`
21. A small image batch from `test_docs/test images/` (pick first 50 PNG)
22. Full large corpus `test_docs/spdocvqa_images/` (12,828 PNG)

## 3) Test Environment and Pre-Checks

Run these before execution of any suite:

1. Use Python 3.12 environment (`.venv312`) for runtime parity.
2. Confirm `.env` settings used for OCR/diagram/retrieval:
   - `ENABLE_OCR=true`
   - `OCR_ENGINE=tesseract`
   - `ENABLE_DIAGRAM_PIPELINE=true`
   - `ENABLE_YOLO_DIAGRAM_DETECTOR=true`
   - `ENABLE_PPTX_SLIDE_RENDER=true`
   - `DB_BACKEND=sqlite` for local baseline
3. Confirm API starts cleanly (`/api/health` returns `status=ok`).
4. Empty document store (`DELETE /api/documents`) before baseline run.
5. Capture baseline DB row counts for:
   - `documents`
   - `chunks`
   - `embeddings`
   - `diagram_graphs`

## 4) Common Observability and Evidence to Capture

For every test case capture:

1. API request/response payloads.
2. Final document status (`queued` -> `processing` -> `ready` or `failed`).
3. Ingestion progress metadata trajectory.
4. Document-level row counts in DB tables.
5. Sample chunk rows including `source_type`, `page`, and metadata fields.
6. If relevant, diagram graph JSON samples and metrics.
7. For retrieval tests, returned chunk IDs, source tags, and answer text.

## 5) Execution Utilities (Reference)

Use API-based execution for deterministic logs.

Suggested flow for each upload test:

1. `POST /api/documents` with multipart file.
2. Poll `GET /api/documents/{doc_id}` every 2–5 seconds until terminal status.
3. Query DB for chunk/embedding/graph evidence.
4. Run one or more `POST /api/chat` probes tied to that `doc_id`.

## 6) Test Suites

### Suite A: Ingestion Lifecycle and API Contract

#### A-001 Single upload lifecycle (PDF)
- Input: `test_docs/LakeRunner/LakeRunner.pdf`
- Steps:
1. Upload file.
2. Poll status until terminal.
3. Verify progress fields evolve and end at `ingest_progress=100`.
- Expected:
1. Final status `ready`.
2. `num_pages > 0`.
3. Non-zero chunks and embeddings created.

#### A-002 Single upload lifecycle (PPTX)
- Input: `test_docs/test.pptx`
- Steps: same as A-001.
- Expected:
1. Status `ready`.
2. Chunks include `slide_graph` and/or OCR-derived content.

#### A-003 Single upload lifecycle (XLSX)
- Input: `test_docs/LakeRunner/Lakerunner license.xlsx`
- Steps: same as A-001.
- Expected:
1. Status `ready`.
2. Table-derived chunks exist.

#### A-004 Single upload lifecycle (DOCX)
- Input: `test_docs/Ashnik Invoice-20260223T051635Z-1-001/Ashnik Invoice/Invoice INV-ID-21-0015.docx`
- Expected:
1. Text/table chunks extracted from DOCX.
2. Embeddings present for all chunks.

#### A-005 Single upload lifecycle (JPG image)
- Input: `test_docs/3-ML-Lifecycle-Detail.jpg`
- Expected:
1. Status `ready`.
2. OCR metadata populated on image-derived chunks.
3. Diagram metadata present (status either `ok` or `no_structure`, no crash).

#### A-006 Single upload lifecycle (diagram PNG)
- Input: `test_docs/Intelligent Retrieval System.drawio (2).png`
- Expected:
1. Status `ready`.
2. Diagram graph records created if structure detected.

#### A-007 Filename handling with spaces and punctuation
- Inputs:
1. `test_docs/Intelligent Retrieval System.drawio (2).png`
2. `test_docs/LakeRunner/Lakerunner_updated_with_licensing_multitenant 1.pptx`
- Expected:
1. Upload/save path handling succeeds.
2. No filename normalization collision.

#### A-008 Sequential mixed uploads (small regression pack)
- Inputs: one each of PDF, PPTX, XLSX, DOCX, PNG, JPG.
- Expected:
1. All documents reach `ready`.
2. No cross-document contamination in metadata or chunks.

### Suite B: Extractor Correctness by File Type

#### B-001 PDF native text extraction
- Input: `test_docs/LakeRunner/LakeRunner.pdf`
- Steps:
1. Ingest document.
2. Query chunks by `doc_id`.
- Expected:
1. Presence of `source_type=text` chunks from native PDF text.
2. Page mapping is coherent (`page` values within `num_pages`).

#### B-002 PDF image generation for each page
- Input: `test_docs/LakeRunner/LakeRunner.pdf`
- Expected:
1. Processed page images saved under `data/processed/<doc_id>/images`.
2. Chunk metadata includes `image_path` for page-related chunks.

#### B-003 Invoice PDF extraction robustness (10-doc batch)
- Inputs: all 10 invoice PDFs in `test_docs/Ashnik Invoice.../Ashnik Invoice/`.
- Expected:
1. All ingest successfully.
2. OCR-backed text appears where native text is weak.
3. No parser crashes on financial layouts.

#### B-004 PPTX text and table extraction
- Inputs:
1. `test_docs/test.pptx`
2. `test_docs/LakeRunner/LakeRunner.pptx`
- Expected:
1. Text chunks from slide text shapes.
2. Table chunks when tables exist.

#### B-005 PPTX embedded image extraction
- Input: `test_docs/LakeRunner/Airtel Cloud Prep.pptx`
- Expected:
1. `image`/`ocr` source chunks created from embedded pictures.
2. OCR fields populated in metadata.

#### B-006 PPTX slide rendering path (LibreOffice)
- Input: `test_docs/LakeRunner/Airtel Cloud PPT 1.pptx`
- Expected:
1. Slide render blocks exist when `ENABLE_PPTX_SLIDE_RENDER=true`.
2. No hard failure if rendering backend unavailable; ingestion still completes.

#### B-007 DOCX extraction structure
- Input: `Invoice INV-ID-21-0015.docx`
- Expected:
1. Paragraph text extracted.
2. Table text extracted if tables present.
3. Embedded image extraction does not fail ingestion.

#### B-008 XLSX extraction structure
- Input: `Lakerunner license.xlsx`
- Expected:
1. Sheet name appears in table chunk text.
2. Tabular rows serialized consistently.

### Suite C: OCR Path Validation

#### C-001 OCR baseline on clean diagram image
- Input: `Intelligent Retrieval System.drawio (2).png`
- Expected:
1. `ocr_status` set.
2. Non-empty OCR text for labeled regions.

#### C-002 OCR on transparent image edge case
- Input: `PNG_transparency_demonstration_1.png`
- Expected:
1. Ingestion does not fail on alpha/transparency.
2. OCR may be empty, but status/error fields are coherent.

#### C-003 OCR on invoice pages
- Inputs: 3 representative invoice PDFs.
- Expected:
1. OCR text captures invoice fields (invoice numbers/dates/amount-like tokens).
2. `avg_confidence` and `line_count` are non-zero for text-rich pages.

#### C-004 OCR timeout behavior
- Inputs: 200 images sampled from `spdocvqa_images`.
- Expected:
1. No global ingestion crash if any OCR operation times out.
2. Failures are recorded per-image in metadata, document can still complete.

#### C-005 OCR worker restart resilience
- Inputs: 50-image batch from `test_docs/test images`.
- Steps:
1. Start ingestion.
2. Restart API process once mid-run (or simulate worker reset before next doc).
- Expected:
1. Worker health recovers.
2. Subsequent docs still ingest correctly.

### Suite D: Diagram Pipeline Validation

#### D-001 Diagram detection on canonical architecture diagram
- Input: `Intelligent Retrieval System.drawio (2).png`
- Expected:
1. `diagram_status=ok` or meaningful alternative status.
2. `diagram_graphs` table entry exists when status is `ok`.
3. Graph includes `nodes`, `edges`, and `metrics` fields.

#### D-002 Diagram false-positive control on non-diagram image
- Input: `PNG_transparency_demonstration_1.png`
- Expected:
1. Status should typically be `no_structure` or low-confidence output.
2. No malformed graph JSON persisted.

#### D-003 Node OCR labeling quality
- Input: `3-ML-Lifecycle-Detail.jpg`
- Expected:
1. Node labels extracted for at least a subset of nodes.
2. Node chunk text (`diagram_node`) references labels and bbox fields.

#### D-004 Edge extraction consistency
- Input: `Intelligent Retrieval System.drawio (2).png`
- Expected:
1. `diagram_edge` chunks reference existing node IDs.
2. Direction hints are populated (`left_to_right` or `top_to_bottom`).

#### D-005 PPTX slide graph generation
- Inputs:
1. `LakeRunner.pptx`
2. `Lakerunner_updated_with_licensing_multitenant 1.pptx`
- Expected:
1. Slide-level graph chunks (`slide_graph`) created.
2. Document-level relationship graph created.

#### D-006 Diagram pipeline under load
- Inputs: 500-image sample from `spdocvqa_images`.
- Expected:
1. No memory-related crash.
2. Stable completion with mixed `ok`/`no_structure` statuses.

### Suite E: Chunking, Embedding, and Persistence Integrity

#### E-001 Chunk generation non-emptiness
- Inputs: one file per type.
- Expected:
1. Every `ready` document has at least one chunk.

#### E-002 Embedding cardinality match
- Inputs: same docs as E-001.
- Expected:
1. Number of embeddings equals number of chunks for each doc.
2. No orphan embeddings without chunks.

#### E-003 Embedding dimension consistency
- Inputs: all docs in small regression pack.
- Expected:
1. All embedding rows share one `dim` value.
2. Dimension matches configured model expectation.

#### E-004 Source type distribution sanity
- Input: `LakeRunner.pptx` and one invoice PDF.
- Expected:
1. Mixed `source_type` values observed (text/table/image/ocr/slide_graph/diagram_* where applicable).

#### E-005 Metadata completeness
- Input: any image-heavy document.
- Expected:
1. Metadata includes `doc_filename`.
2. Image-related chunks include `image_path`.
3. OCR and diagram metadata fields present when those paths execute.

#### E-006 Delete document cleanup
- Input: upload one PDF then delete it.
- Expected:
1. Rows removed from `documents`, `chunks`, `embeddings`, `diagram_graphs` for that doc.
2. `data/uploads/<doc_id>` and `data/processed/<doc_id>` removed.

#### E-007 Delete-all cleanup
- Input: ingest multiple docs then call delete-all.
- Expected:
1. Core tables are empty.
2. In-memory indexes are reloaded and queries return empty results.

### Suite F: Retrieval and Chat Grounding

#### F-001 Basic semantic retrieval
- Input doc: `LakeRunner.pdf`
- Query examples:
1. "Summarize LakeRunner architecture"
2. "What capabilities are listed in LakeRunner?"
- Expected:
1. Non-empty answer.
2. Sources point only to selected `doc_id`.

#### F-002 Invoice retrieval
- Input docs: 10 invoice PDFs.
- Query examples:
1. "List invoice IDs found"
2. "Which invoices mention INV-ID-21-0015?"
- Expected:
1. Relevant invoice chunks returned.
2. Citations include invoice documents/pages.

#### F-003 Diagram intent routing
- Input doc: `Intelligent Retrieval System.drawio (2).png`
- Query: "Explain this diagram step by step"
- Expected:
1. Router selects diagram/image-oriented strategy.
2. Retrieved chunks include `diagram_graph`/`diagram_node`/`ocr` evidence mix.

#### F-004 Relationship query behavior
- Input doc: `LakeRunner.pptx`
- Query: "How are the components connected across slides?"
- Expected:
1. `slide_graph` chunks contribute to answer.
2. Response references relationship evidence.

#### F-005 Cross-document compare
- Input docs: `LakeRunner.pdf` + `LakeRunner.pptx`
- Query: "Compare architecture described in PDF vs PPT"
- Expected:
1. Retrieval includes both doc IDs.
2. Response surfaces differences with citations.

#### F-006 Doc scope enforcement
- Input docs: at least two ready docs.
- Steps:
1. Ask query with one selected `doc_id`.
2. Repeat with both selected.
- Expected:
1. First answer cites only scoped document.
2. Second answer may cite both.

#### F-007 Empty/invalid scope rejection
- Steps:
1. Submit chat request with unknown `doc_id`.
2. Submit chat request with non-ready `doc_id`.
- Expected:
1. API returns validation error messages.

### Suite G: Scale, Throughput, and Stability

#### G-001 Medium image batch ingestion
- Input: first 500 files from `test_docs/spdocvqa_images`.
- Expected:
1. All complete without process crash.
2. No stuck documents in perpetual `processing`.

#### G-002 Large image batch ingestion
- Input: full `test_docs/spdocvqa_images` corpus.
- Expected:
1. System remains responsive (health endpoint alive).
2. Throughput degrades gracefully, not catastrophically.

#### G-003 Mixed-format batch ingestion stress
- Input set:
1. All LakeRunner docs
2. All invoice PDFs
3. 200 images from `spdocvqa_images`
- Expected:
1. No extractor-specific deadlocks.
2. OCR worker remains stable across mixed inputs.

#### G-004 Restart durability test
- Steps:
1. Ingest medium batch.
2. Restart service.
3. Verify documents, chunks, embeddings reload correctly.
- Expected:
1. Existing ready docs remain queryable after restart.

### Suite H: Data Quality and Regression Checks

#### H-001 Regression snapshot for chunk/source distribution
- Inputs: canonical small suite (LakeRunner.pdf, LakeRunner.pptx, invoice PDF, diagram PNG, xlsx, docx)
- Expected:
1. Persist a snapshot of counts by `source_type` per document.
2. Future code changes should not regress unexpectedly.

#### H-002 Regression snapshot for diagram metrics
- Inputs: diagram PNG + one diagram-heavy slide deck.
- Expected:
1. Capture node/edge counts and graph density metrics.
2. Track drift after model/config updates.

#### H-003 Regression snapshot for OCR confidence
- Inputs: 3 invoice PDFs + 20 random `spdocvqa_images`.
- Expected:
1. Capture median/mean OCR confidence and line counts.
2. Investigate major drops release-to-release.

#### H-004 Regression snapshot for retrieval quality
- Inputs: fixed query set mapped to expected doc IDs.
- Expected:
1. Track top-k source doc hit rate.
2. Track citation completeness.

## 7) Suggested Test Execution Order

1. Suite A (contract/lifecycle)
2. Suite B (extractors)
3. Suite C + D (OCR + diagram)
4. Suite E (storage integrity)
5. Suite F (retrieval/chat)
6. Suite G (scale)
7. Suite H (regression baselines)

## 8) Minimal Acceptance Criteria for Release

1. No ingestion crashes for canonical files listed in Section 2.
2. All canonical files reach `ready`.
3. Every ready document has non-zero chunks and matching embeddings.
4. Diagram tests on canonical diagram input produce structurally valid outputs.
5. Retrieval returns grounded citations for PDF/PPTX/invoice queries.
6. Delete and delete-all operations fully clean DB and processed artifacts.
7. Medium scale image batch (500 files) completes without service failure.

## 9) Optional Automation Layer (Future)

To operationalize this plan:

1. Add API-driven pytest suite for Suites A, E, and F.
2. Add nightly long-run job for Suites G and H.
3. Export structured test evidence as JSON for trend dashboards.

## 10) Notes

- This plan intentionally uses only files currently present in `test_docs/`.
- Tests involving the full `spdocvqa_images` corpus are intended for stress/perf windows, not every commit.
- If environment flags change (`ENABLE_OCR`, `OCR_ENGINE`, `ENABLE_PPTX_SLIDE_RENDER`, `ENABLE_DIAGRAM_PIPELINE`), rerun impacted suites and compare against prior snapshots.
