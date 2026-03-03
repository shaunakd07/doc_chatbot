Data and Persistence

File system state:
Raw uploads: data/uploads/<doc_id>/... (pipeline.py)
Processed images: data/processed/<doc_id>/images/... (pipeline.py)
SQLite file: app.db (config.py)

DB tables:
documents (status/progress metadata)
chunks (retrieval text units)
embeddings (one vector per chunk)
diagram_graphs (structured graph JSON + image path)
Schema setup in storage.py
SQLite mode stores embedding as BLOB; Postgres mode stores vector(PGVECTOR_DIM) with pgvector index (storage.py, storage.py).
In Postgres mode, dense search is SQL (embedding <=> query) (storage.py).

Ingestion and Query Flows

Upload flow:
POST /api/documents saves file, creates document row, starts background ingestion task (app.py).

Ingestion flow:
Extract blocks by type.
For images/pages: save image artifact, OCR, optional visual summary, diagram parse.
Chunk resulting text.
Insert chunks, compute embeddings, insert embeddings, update sparse index (pipeline.py).

Query flow:
Router classifies intent and retrieval strategy (openai_router_service.py).
Retrieval service runs dense/sparse/hybrid and reranker (retrieval_service.py).
Chat service assembles context and attaches up to 5 images from metadata.image_path (chat_service.py).
Generation is OpenAI Chat Completions, optionally multimodal (openai_chat.py).
External Dependencies / Infra-Relevant Runtime

OpenAI API for:
embeddings
router
answer generation
OpenAI usage in embeddings.py, openai_router_service.py, openai_chat.py

Local compute libs:
PaddleOCR - OCR engine used to extract text from scanned/image-based pages and diagram crops (runs in isolated worker process).
Ultralytics YOLO - object detection for diagram node/region proposals before graph construction.
OpenCV - image preprocessing and classical vision operations (thresholding/contours, Canny edges, Hough line detection) and fallback for node detection.
NetworkX - computes graph structure metrics from extracted nodes/edges (components, largest component, density).
NumPy - array math for image tensors and vector operations in diagram/OCR/embedding pipelines.
Pillow (PIL) - image loading/conversion/saving for ingestion artifacts and OCR payload preparation.
sentence-transformers - local embedding and reranker model runtime when local providers are used.
PyTorch - runtime backend for local ML models (SentenceTransformer/CrossEncoder/optional local router paths).
System dependency for PPTX full-slide rendering: soffice (LibreOffice) - headless PPTX->PDF conversion for slide rasterization/OCR fallback (extractors.py).

Document-reading modules:
PyMuPDF (fitz) - PDF text extraction and page rendering
python-pptx - PPTX shape/text/table/image extraction
python-docx - DOCX paragraph/table/image extraction
openpyxl - XLSX/XLSM/XLTX/XLTM spreadsheet parsing
xlrd - XLS spreadsheet parsing
beautifulsoup4 - HTML/XML text extraction fallback
striprtf - RTF text extraction fallback
Pillow (PIL) - image file loading/normalization
zipfile + ElementTree (stdlib) - OOXML/ODF fallback parsing

{
  "kind": "image_diagram_graph",
  "page": 2,
  "image_path": "data/processed/8f1d3c2a/images/p2_img1.png",
  "node_count": 3,
  "edge_count": 2,
  "nodes": [
    {
      "id": "node_001",
      "label": "Start",
      "has_ocr_label": true,
      "score": 0.94,
      "detector": "yolo",
      "node_kind": "process",
      "bbox": {
        "x": 0.12,
        "y": 0.18,
        "w": 0.16,
        "h": 0.09,
        "x1": 145,
        "y1": 210,
        "x2": 338,
        "y2": 315
      },
      "center_x": 241.5,
      "center_y": 262.5
    },
    {
      "id": "node_002",
      "label": "Validate Input",
      "has_ocr_label": true,
      "score": 0.91,
      "detector": "yolo",
      "node_kind": "decision",
      "bbox": {
        "x": 0.39,
        "y": 0.18,
        "w": 0.21,
        "h": 0.11,
        "x1": 470,
        "y1": 206,
        "x2": 724,
        "y2": 334
      },
      "center_x": 597.0,
      "center_y": 270.0
    },
    {
      "id": "node_003",
      "label": "End",
      "has_ocr_label": true,
      "score": 0.89,
      "detector": "opencv",
      "node_kind": "terminator",
      "bbox": {
        "x": 0.71,
        "y": 0.18,
        "w": 0.14,
        "h": 0.09,
        "x1": 858,
        "y1": 212,
        "x2": 1028,
        "y2": 317
      },
      "center_x": 943.0,
      "center_y": 264.5
    }
  ],
  "edges": [
    {
      "id": "edge_001",
      "from": "node_001",
      "to": "node_002",
      "line": {
        "x1": 338,
        "y1": 262,
        "x2": 470,
        "y2": 268
      },
      "length": 132,
      "direction_hint": "left_to_right",
      "type": "line_connection"
    },
    {
      "id": "edge_002",
      "from": "node_002",
      "to": "node_003",
      "line": {
        "x1": 724,
        "y1": 269,
        "x2": 858,
        "y2": 265
      },
      "length": 134,
      "direction_hint": "left_to_right",
      "type": "line_connection"
    }
  ],
  "metrics": {
    "connected_components": 1,
    "largest_component": 3,
    "density": 0.3333,
    "library": "networkx"
  }
}
