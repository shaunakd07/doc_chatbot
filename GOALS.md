Build this chatbot as a document-grounded reasoning system, not a generic chat assistant.

Primary goals:
- Support high-quality semantic reasoning across multiple documents
- Support precise retrieval of document-specific facts, metadata, and exact details
- Ground all substantive answers in retrieved evidence
- Distinguish explicit facts from inferred conclusions
- Adapt retrieval and answer strategy based on query type

The system should handle at least these query types:
- exact fact lookup
- metadata lookup
- document-specific Q&A
- cross-document synthesis
- comparison
- contradiction detection
- timeline reconstruction
- summarization

Retrieval requirements:
- Use hybrid retrieval, not dense-only
- Combine semantic retrieval, keyword/sparse retrieval, metadata filtering, and structured lookup where available
- Support corpus-level, document-level, section-level, and chunk-level retrieval
- Prefer the smallest relevant evidence spans for precise answers
- Aggregate evidence across documents for synthesis questions

Structured understanding requirements:
- Extract and persist useful document metadata and section-aware information when possible
- Make structured fields queryable independently of embeddings
- Support questions about which document contains a fact, topic, entity, date, value, or contradiction

Reasoning requirements:
- Separate explicit document facts from inferred conclusions
- Surface conflicts when documents disagree
- Be explicit when evidence is weak or incomplete
- Do not produce confident unsupported answers

Architecture priorities:
- backend-centered logic
- inspectable retrieval pipeline
- query classification / route planning
- source-grounded answer generation
- easy extension for OCR, diagrams, tables, and classification

Definition of success:
- The chatbot can answer both broad semantic questions and highly specific detail questions accurately
- It can reason across multiple documents without losing source traceability
- It can use metadata and structured extraction in addition to embeddings
- It fails conservatively when evidence is insufficient

Implementation guidance:
- Add or improve query classification so the system can distinguish lookup vs synthesis vs comparison vs metadata queries
- Route different query types through different retrieval strategies
- Preserve and expand structured document metadata storage where useful
- Improve evidence assembly so answers cite the most relevant supporting spans
- Favor backend implementations over frontend heuristics
- Add tests that cover:
  - exact fact retrieval
  - metadata queries
  - cross-document reasoning
  - contradiction handling
  - weak-evidence behavior