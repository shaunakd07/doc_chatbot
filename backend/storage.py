import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import config


DB_PATH = config.DB_DIR / "app.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


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


def _row_to_document(row: sqlite3.Row) -> Dict[str, Any]:
    doc = dict(row)
    doc["metadata"] = _decode_json_field(doc.get("metadata")) or {}
    return doc


def _row_to_chunk(row: sqlite3.Row) -> Dict[str, Any]:
    chunk = dict(row)
    chunk["metadata"] = _decode_json_field(chunk.get("metadata")) or {}
    return chunk


def _row_to_diagram_graph(row: sqlite3.Row) -> Dict[str, Any]:
    record = dict(row)
    record["graph"] = _decode_json_field(record.get("graph_json")) or {}
    record["metadata"] = _decode_json_field(record.get("metadata")) or {}
    record.pop("graph_json", None)
    return record


def init_db() -> None:
    config.ensure_dirs()
    with _connect() as conn:
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
        conn.execute(
            "INSERT INTO documents (id, filename, status, created_at, metadata) VALUES (?, ?, ?, ?, ?)",
            (doc_id, filename, status, created_at, payload),
        )
        conn.commit()


def update_document(doc_id: str, status: Optional[str] = None, num_pages: Optional[int] = None) -> None:
    fields = []
    values: List[Any] = []
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if num_pages is not None:
        fields.append("num_pages = ?")
        values.append(num_pages)
    if not fields:
        return
    values.append(doc_id)
    with _connect() as conn:
        conn.execute(f"UPDATE documents SET {', '.join(fields)} WHERE id = ?", values)
        conn.commit()


def list_documents() -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
    return [_row_to_document(row) for row in rows]


def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    return _row_to_document(row) if row else None


def delete_document(doc_id: str) -> bool:
    with _connect() as conn:
        row = conn.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not row:
            return False
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
        row = conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()
        count = int(row["c"]) if row else 0
        conn.execute("DELETE FROM diagram_graphs")
        conn.execute("DELETE FROM embeddings")
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM documents")
        conn.commit()
    return count


def add_chunks(chunks: Iterable[Dict[str, Any]]) -> None:
    with _connect() as conn:
        conn.executemany(
            """
            INSERT INTO chunks (id, doc_id, page, chunk_index, content, source_type, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
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
            ],
        )
        conn.commit()


def get_chunks_by_doc(doc_id: str) -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM chunks WHERE doc_id = ? ORDER BY page, chunk_index", (doc_id,)).fetchall()
    return [_row_to_chunk(row) for row in rows]


def list_chunks() -> List[Dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM chunks").fetchall()
    return [_row_to_chunk(row) for row in rows]


def add_embeddings(vectors: Iterable[Tuple[str, bytes, int]]) -> None:
    with _connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings (chunk_id, vector, dim) VALUES (?, ?, ?)",
            list(vectors),
        )
        conn.commit()


def load_embeddings() -> List[Tuple[str, bytes, int]]:
    with _connect() as conn:
        rows = conn.execute("SELECT chunk_id, vector, dim FROM embeddings").fetchall()
    return [(row["chunk_id"], row["vector"], row["dim"]) for row in rows]


def get_chunk(chunk_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
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
    with _connect() as conn:
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
                json.dumps(graph),
                float(confidence),
                created_at,
                json.dumps(metadata or {}),
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
        placeholders = ",".join("?" for _ in doc_ids)
        clauses.append(f"doc_id IN ({placeholders})")
        params.extend(doc_ids)
    if image_paths:
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
        row = conn.execute("SELECT * FROM diagram_graphs WHERE id = ?", (graph_id,)).fetchone()
    return _row_to_diagram_graph(row) if row else None
