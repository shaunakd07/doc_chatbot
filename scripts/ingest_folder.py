import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import config, storage
from backend.index.embeddings import Embedder
from backend.index.sparse_index import SparseIndex
from backend.index.vector_index import VectorIndex
from backend.ingestion.pipeline import create_document_record, ingest_file
from backend.models.internvl import InternVLModel


def main() -> None:
    load_dotenv()
    config.ensure_dirs()
    storage.init_db()

    embed_model = config.OPENAI_EMBED_MODEL if config.EMBED_PROVIDER == "openai" else config.EMBED_MODEL
    embedder = Embedder(
        embed_model,
        device="cuda",
        provider=config.EMBED_PROVIDER,
        openai_api_key=config.OPENAI_API_KEY,
    )
    vector_index = VectorIndex()
    vector_index.load()
    sparse_index = SparseIndex()
    sparse_index.load()
    vlm = None
    if config.ENABLE_VLM:
        vlm = InternVLModel(config.MODEL_ID, device=config.DEVICE, hf_token=config.HF_TOKEN)

    upload_dir = config.UPLOAD_DIR
    files = [p for p in upload_dir.rglob("*") if p.is_file()]
    if not files:
        print("No files found in uploads.")
        return

    for file_path in files:
        doc_id = create_document_record(file_path.name)
        ingest_file(file_path, doc_id, embedder, vector_index, sparse_index, vlm)

        processed_dir = config.PROCESSED_DIR / doc_id
        processed_dir.mkdir(parents=True, exist_ok=True)
        file_path.replace(processed_dir / file_path.name)

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
