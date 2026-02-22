from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import List
import base64

from PIL import Image

from .extractors import (
    extract_docx,
    extract_generic,
    extract_image,
    extract_pdf,
    extract_pptx,
    extract_text,
    extract_xls,
    extract_xlsx,
)
from .text_chunker import chunk_text
from .. import config, storage

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".log",
    ".ini",
    ".cfg",
    ".toml",
}
SPREADSHEET_EXTENSIONS = {".xlsx", ".xlsm", ".xltx", ".xltm", ".xls"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"}

def _encode_image_file(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def ingest_file(
    file_path: Path,
    doc_id: str,
    embedder,
    vector_index,
    sparse_index=None,
    vlm=None,
) -> None:
    storage.update_document(doc_id, status="processing")
    document = storage.get_document(doc_id) or {}
    doc_filename = str(document.get("filename") or file_path.name)
    processed_image_dir = config.PROCESSED_DIR / doc_id / "images"
    processed_image_dir.mkdir(parents=True, exist_ok=True)

    suffix = file_path.suffix.lower()
    blocks = []
    try:
        if suffix in {".pdf"}:
            blocks = extract_pdf(file_path)
        elif suffix in {".pptx"}:
            blocks = extract_pptx(file_path)
        elif suffix in {".docx"}:
            blocks = extract_docx(file_path)
        elif suffix in SPREADSHEET_EXTENSIONS:
            if suffix == ".xls":
                blocks = extract_xls(file_path)
            else:
                blocks = extract_xlsx(file_path)
        elif suffix in IMAGE_EXTENSIONS:
            blocks = extract_image(file_path)
        elif suffix in TEXT_EXTENSIONS:
            blocks = extract_text(file_path)
        else:
            blocks = extract_generic(file_path)
    except Exception:
        storage.update_document(doc_id, status="failed")
        raise

    chunks: List[dict] = []
    image_counter = 0
    for block in blocks:
        metadata = {"doc_filename": doc_filename}
        block_metadata = block.get("metadata")
        if isinstance(block_metadata, dict):
            metadata.update(block_metadata)
        
        text = str(block.get("text", ""))

        if block.get("type") == "image":
            image = block.get("image")
            image_path = None
            if image is not None:
                image_counter += 1
                page_num = block.get("page") or 1
                image_name = f"p{page_num}_img{image_counter}.png"
                target = processed_image_dir / image_name
                try:
                    image.save(target, format="PNG")
                    image_path = str(target)
                    metadata["image_path"] = image_path
                    
                    # Instead of local OCR or complex diagram graphs, we use gpt-4o-mini to get a 
                    # detailed text description to embed so we can hybrid-search it later.
                    if vlm is not None and hasattr(vlm, "answer_image_question"):
                        prompt = "Describe this page in high detail. Extract all textual content, describe any tables row by row, and explain any flowcharts, diagrams or logic presented."
                        try:
                            with Image.open(target) as saved_img:
                                desc = vlm.answer_image_question(saved_img, prompt)
                                text += f"\n[AI Visual Summary]: {desc}\n"
                        except Exception as e:
                            # Log and fall back to minimal text if vision proxy fails.
                            print(f"[Ingestion] Failed to get vision summary for {image_name}: {e}")
                except Exception as e:
                    print(f"Error saving image: {e}")

        # Basic text embedding of whatever text/tables/summaries we generated.
        if text.strip():
            for idx, chunk in enumerate(chunk_text(text, max_chars=900, overlap=120)):
                chunks.append(
                    {
                        "id": str(uuid.uuid4()),
                        "doc_id": doc_id,
                        "page": block.get("page"),
                        "chunk_index": idx,
                        "content": chunk,
                        "source_type": str(block.get("type", "text")),
                        "metadata": metadata,
                    }
                )

    if chunks:
        storage.add_chunks(chunks)
        texts = [chunk["content"] for chunk in chunks]
        vectors = embedder.embed_texts(texts)
        dim = vectors.shape[1]
        storage.add_embeddings(
            (chunk["id"], vec.astype("float32").tobytes(), dim)
            for chunk, vec in zip(chunks, vectors)
        )
        vector_index.add(vectors, [chunk["id"] for chunk in chunks])
        if sparse_index is not None:
            sparse_index.add_chunks(chunks)

    max_page = max((block.get("page") or 0 for block in blocks), default=0)
    storage.update_document(doc_id, status="ready", num_pages=max_page)


def create_document_record(filename: str) -> str:
    doc_id = str(uuid.uuid4())
    storage.add_document(
        doc_id=doc_id,
        filename=filename,
        status="queued",
        created_at=datetime.utcnow().isoformat(),
        metadata={},
    )
    return doc_id


def safe_path_for_upload(filename: str, doc_id: str) -> Path:
    upload_dir = config.UPLOAD_DIR / doc_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir / filename
