import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from . import config


try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover - optional dependency at import time
    psycopg = None
    dict_row = None


LOGGER = logging.getLogger(__name__)
DB_PATH = config.SQLITE_DB_PATH


def _is_postgres() -> bool:
    return str(config.DB_BACKEND).strip().lower() == "postgres"


def _connect_sqlite() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _connect_postgres():
    if psycopg is None or dict_row is None:
        raise RuntimeError(
            "DB_BACKEND=postgres requires psycopg. Install dependencies with: pip install -r requirements.txt"
        )
    if not str(config.DATABASE_URL).strip():
        raise RuntimeError("DB_BACKEND=postgres requires DATABASE_URL to be set.")
    timeout_sec = max(1, int(float(config.PG_CONNECT_TIMEOUT_SEC)))
    return psycopg.connect(
        config.DATABASE_URL,
        row_factory=dict_row,
        autocommit=False,
        connect_timeout=timeout_sec,
    )


def _connect():
    if _is_postgres():
        return _connect_postgres()
    return _connect_sqlite()


def _decode_json_field(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _row_to_document(row: Any) -> Dict[str, Any]:
    doc = dict(row)
    doc["metadata"] = _decode_json_field(doc.get("metadata")) or {}
    return doc


def _row_to_chunk(row: Any) -> Dict[str, Any]:
    chunk = dict(row)
    chunk["metadata"] = _decode_json_field(chunk.get("metadata")) or {}
    return chunk


def _row_to_diagram_graph(row: Any) -> Dict[str, Any]:
    record = dict(row)
    record["graph"] = _decode_json_field(record.get("graph_json")) or {}
    record["metadata"] = _decode_json_field(record.get("metadata")) or {}
    record.pop("graph_json", None)
    return record


def _vector_literal_from_array(vector: np.ndarray) -> str:
    values = [f"{float(v):.8f}" for v in vector.astype("float32").tolist()]
    return "[" + ",".join(values) + "]"


def _vector_literal_from_blob(blob: bytes) -> Tuple[str, int]:
    vector = np.frombuffer(blob, dtype="float32")
    return _vector_literal_from_array(vector), int(vector.shape[0])


def _parse_vector_text(value: str) -> np.ndarray:
    text = str(value or "").strip()
    if not text:
        return np.zeros((0,), dtype="float32")
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text.strip():
        return np.zeros((0,), dtype="float32")
    return np.fromstring(text, sep=",", dtype="float32")


def init_db() -> None:
    config.ensure_dirs()
    if _is_postgres():
        dim = max(1, int(config.PGVECTOR_DIM))
        with _connect_postgres() as conn:
            cur = conn.cursor()
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    num_pages INTEGER DEFAULT 0,
                    metadata JSONB DEFAULT '{}'::jsonb
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    page INTEGER,
                    chunk_index INTEGER,
                    content TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    embedding vector({dim}) NOT NULL,
                    dim INTEGER NOT NULL,
                    FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS diagram_graphs (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    page INTEGER,
                    image_path TEXT,
                    parser_version TEXT NOT NULL,
                    graph_json JSONB NOT NULL,
                    confidence REAL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_diagram_graph_doc ON diagram_graphs(doc_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_diagram_graph_image ON diagram_graphs(image_path)")
            try:
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_embeddings_ivfflat_cosine
                    ON embeddings USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                    """
                )
            except Exception as exc:  # pragma: no cover - best effort index creation
                LOGGER.warning("Could not create ivfflat index for embeddings: %s", exc)
            conn.commit()
        return

    with _connect_sqlite() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                num_pages INTEGER DEFAULT 0,
                metadata TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                page INTEGER,
                chunk_index INTEGER,
                content TEXT NOT NULL,
                source_type TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY(doc_id) REFERENCES documents(id)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                dim INTEGER NOT NULL,
                FOREIGN KEY(chunk_id) REFERENCES chunks(id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS diagram_graphs (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                page INTEGER,
                image_path TEXT,
                parser_version TEXT NOT NULL,
                graph_json TEXT NOT NULL,
                confidence REAL DEFAULT 0,
                created_at TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY(doc_id) REFERENCES documents(id)
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_diagram_graph_doc ON diagram_graphs(doc_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_diagram_graph_image ON diagram_graphs(image_path)")
        conn.commit()


def add_document(doc_id: str, filename: str, status: str, created_at: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    payload = json.dumps(metadata or {})
    with _connect() as conn:
        if _is_postgres():
            conn.execute(
                "INSERT INTO documents (id, filename, status, created_at, metadata) VALUES (%s, %s, %s, %s, %s::jsonb)",
                (doc_id, filename, status, created_at, payload),
            )
        else:
            conn.execute(
                "INSERT INTO documents (id, filename, status, created_at, metadata) VALUES (?, ?, ?, ?, ?)",
                (doc_id, filename, status, created_at, payload),
            )
        conn.commit()


def update_document(
    doc_id: str,
    status: Optional[str] = None,
    num_pages: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    merge_metadata: bool = True,
) -> None:
    fields = []
    values: List[Any] = []
    if status is not None:
        fields.append("status = %s" if _is_postgres() else "status = ?")
        values.append(status)
    if num_pages is not None:
        fields.append("num_pages = %s" if _is_postgres() else "num_pages = ?")
        values.append(num_pages)
    if metadata is not None:
        merged_metadata: Dict[str, Any]
        if merge_metadata:
            existing = get_document(doc_id) or {}
            existing_metadata = existing.get("metadata")
            merged_metadata = existing_metadata if isinstance(existing_metadata, dict) else {}
            merged_metadata = dict(merged_metadata)
            merged_metadata.update(metadata)
        else:
            merged_metadata = metadata if isinstance(metadata, dict) else {}
        if _is_postgres():
            fields.append("metadata = %s::jsonb")
        else:
            fields.append("metadata = ?")
        values.append(json.dumps(merged_metadata))
    if not fields:
        return
    values.append(doc_id)
    with _connect() as conn:
        if _is_postgres():
            conn.execute(f"UPDATE documents SET {', '.join(fields)} WHERE id = %s", values)
        else:
            conn.execute(f"UPDATE documents SET {', '.join(fields)} WHERE id = ?", values)
        conn.commit()


def list_documents() -> List[Dict[str, Any]]:
    with _connect() as conn:
        if _is_postgres():
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        else:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
    return [_row_to_document(row) for row in rows]


def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        if _is_postgres():
            row = conn.execute("SELECT * FROM documents WHERE id = %s", (doc_id,)).fetchone()
        else:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    return _row_to_document(row) if row else None


def delete_document(doc_id: str) -> bool:
    with _connect() as conn:
        if _is_postgres():
            row = conn.execute("SELECT 1 FROM documents WHERE id = %s", (doc_id,)).fetchone()
        else:
            row = conn.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not row:
            return False
        if _is_postgres():
            conn.execute(
                "DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = %s)",
                (doc_id,),
            )
            conn.execute("DELETE FROM diagram_graphs WHERE doc_id = %s", (doc_id,))
            conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
            conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        else:
            conn.execute(
                "DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?)",
                (doc_id,),
            )
            conn.execute("DELETE FROM diagram_graphs WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
    return True


def delete_all_documents() -> int:
    with _connect() as conn:
        if _is_postgres():
            row = conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()
            count = int(row["c"]) if row else 0
            conn.execute("DELETE FROM diagram_graphs")
            conn.execute("DELETE FROM embeddings")
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents")
        else:
            row = conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()
            count = int(row["c"]) if row else 0
            conn.execute("DELETE FROM diagram_graphs")
            conn.execute("DELETE FROM embeddings")
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents")
        conn.commit()
    return count


def add_chunks(chunks: Iterable[Dict[str, Any]]) -> None:
    rows = [
        (
            chunk["id"],
            chunk["doc_id"],
            chunk.get("page"),
            chunk.get("chunk_index"),
            chunk["content"],
            chunk["source_type"],
            json.dumps(chunk.get("metadata", {})),
        )
        for chunk in chunks
    ]
    if not rows:
        return

    with _connect() as conn:
        if _is_postgres():
            conn.executemany(
                """
                INSERT INTO chunks (id, doc_id, page, chunk_index, content, source_type, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                rows,
            )
        else:
            conn.executemany(
                """
                INSERT INTO chunks (id, doc_id, page, chunk_index, content, source_type, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        conn.commit()


def get_chunks_by_doc(doc_id: str) -> List[Dict[str, Any]]:
    with _connect() as conn:
        if _is_postgres():
            rows = conn.execute("SELECT * FROM chunks WHERE doc_id = %s ORDER BY page, chunk_index", (doc_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM chunks WHERE doc_id = ? ORDER BY page, chunk_index", (doc_id,)).fetchall()
    return [_row_to_chunk(row) for row in rows]


def list_chunks() -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM chunks").fetchall()
    return [_row_to_chunk(row) for row in rows]


def add_embeddings(vectors: Iterable[Tuple[str, bytes, int]]) -> None:
    records = list(vectors)
    if not records:
        return

    if _is_postgres():
        expected_dim = max(1, int(config.PGVECTOR_DIM))
        rows: List[Tuple[str, str, int]] = []
        for chunk_id, blob, dim in records:
            vector_literal, actual_dim = _vector_literal_from_blob(blob)
            if actual_dim != expected_dim:
                raise RuntimeError(
                    f"Embedding dimension mismatch for pgvector: got {actual_dim}, expected {expected_dim}. "
                    "Set PGVECTOR_DIM to match your embedding model output and recreate schema."
                )
            rows.append((chunk_id, vector_literal, int(dim)))
        with _connect_postgres() as conn:
            conn.executemany(
                """
                INSERT INTO embeddings (chunk_id, embedding, dim)
                VALUES (%s, %s::vector, %s)
                ON CONFLICT (chunk_id)
                DO UPDATE SET embedding = EXCLUDED.embedding, dim = EXCLUDED.dim
                """,
                rows,
            )
            conn.commit()
        return

    with _connect_sqlite() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings (chunk_id, vector, dim) VALUES (?, ?, ?)",
            records,
        )
        conn.commit()


def load_embeddings() -> List[Tuple[str, bytes, int]]:
    if _is_postgres():
        with _connect_postgres() as conn:
            rows = conn.execute("SELECT chunk_id, embedding::text AS embedding_text, dim FROM embeddings").fetchall()
        out: List[Tuple[str, bytes, int]] = []
        for row in rows:
            vector = _parse_vector_text(str(row.get("embedding_text") or ""))
            out.append((str(row["chunk_id"]), vector.astype("float32").tobytes(), int(row["dim"])))
        return out

    with _connect_sqlite() as conn:
        rows = conn.execute("SELECT chunk_id, vector, dim FROM embeddings").fetchall()
    return [(row["chunk_id"], row["vector"], row["dim"]) for row in rows]


def search_embeddings(query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
    if query_vector.size == 0:
        return []

    if _is_postgres():
        query = np.asarray(query_vector, dtype="float32").reshape(-1)
        expected_dim = max(1, int(config.PGVECTOR_DIM))
        if int(query.shape[0]) != expected_dim:
            raise RuntimeError(
                f"Query embedding dimension mismatch for pgvector: got {int(query.shape[0])}, expected {expected_dim}."
            )
        literal = _vector_literal_from_array(query)
        with _connect_postgres() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, 1 - (embedding <=> %s::vector) AS score
                FROM embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (literal, literal, max(1, int(top_k))),
            ).fetchall()
        return [(str(row["chunk_id"]), float(row["score"])) for row in rows]

    records = load_embeddings()
    if not records:
        return []
    vectors = []
    chunk_ids: List[str] = []
    for chunk_id, blob, _ in records:
        vectors.append(np.frombuffer(blob, dtype="float32"))
        chunk_ids.append(chunk_id)
    matrix = np.vstack(vectors).astype("float32")
    query = np.asarray(query_vector, dtype="float32")
    scores = matrix @ query
    k = min(max(1, int(top_k)), int(scores.shape[0]))
    indices = np.argpartition(-scores, k - 1)[:k]
    ranked = sorted(((idx, scores[idx]) for idx in indices), key=lambda item: item[1], reverse=True)
    return [(chunk_ids[idx], float(score)) for idx, score in ranked]


def get_chunk(chunk_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        if _is_postgres():
            row = conn.execute("SELECT * FROM chunks WHERE id = %s", (chunk_id,)).fetchone()
        else:
            row = conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
    return _row_to_chunk(row) if row else None


def add_diagram_graph(
    graph_id: str,
    doc_id: str,
    page: int,
    image_path: str,
    parser_version: str,
    graph: Dict[str, Any],
    confidence: float,
    created_at: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    graph_payload = json.dumps(graph)
    metadata_payload = json.dumps(metadata or {})
    with _connect() as conn:
        if _is_postgres():
            conn.execute(
                """
                INSERT INTO diagram_graphs (id, doc_id, page, image_path, parser_version, graph_json, confidence, created_at, metadata)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s::jsonb)
                """,
                (
                    graph_id,
                    doc_id,
                    page,
                    image_path,
                    parser_version,
                    graph_payload,
                    float(confidence),
                    created_at,
                    metadata_payload,
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO diagram_graphs (id, doc_id, page, image_path, parser_version, graph_json, confidence, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    graph_id,
                    doc_id,
                    page,
                    image_path,
                    parser_version,
                    graph_payload,
                    float(confidence),
                    created_at,
                    metadata_payload,
                ),
            )
        conn.commit()


def list_diagram_graphs(
    doc_ids: Optional[List[str]] = None,
    image_paths: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    query = "SELECT * FROM diagram_graphs"
    clauses: List[str] = []
    params: List[Any] = []

    if doc_ids:
        if _is_postgres():
            placeholders = ",".join("%s" for _ in doc_ids)
        else:
            placeholders = ",".join("?" for _ in doc_ids)
        clauses.append(f"doc_id IN ({placeholders})")
        params.extend(doc_ids)

    if image_paths:
        if _is_postgres():
            placeholders = ",".join("%s" for _ in image_paths)
        else:
            placeholders = ",".join("?" for _ in image_paths)
        clauses.append(f"image_path IN ({placeholders})")
        params.extend(image_paths)

    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC"

    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [_row_to_diagram_graph(row) for row in rows]


def get_diagram_graph(graph_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        if _is_postgres():
            row = conn.execute("SELECT * FROM diagram_graphs WHERE id = %s", (graph_id,)).fetchone()
        else:
            row = conn.execute("SELECT * FROM diagram_graphs WHERE id = ?", (graph_id,)).fetchone()
    return _row_to_diagram_graph(row) if row else None
