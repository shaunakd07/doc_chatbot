from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, Iterable, List, Optional


def _decode_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


class DBInspector:
    def __init__(
        self,
        *,
        db_backend: Optional[str] = None,
        sqlite_db_path: Optional[str] = None,
        database_url: Optional[str] = None,
        pg_connect_timeout_sec: float = 5.0,
    ) -> None:
        if db_backend is None:
            db_backend = os.getenv("DB_BACKEND", "sqlite")
        self.db_backend = str(db_backend or "sqlite").strip().lower()
        self.sqlite_db_path = sqlite_db_path or os.getenv("SQLITE_DB_PATH", "data/db/app.db")
        self.database_url = database_url or os.getenv("DATABASE_URL", "")
        self.pg_connect_timeout_sec = max(1, int(float(pg_connect_timeout_sec)))

        self._psycopg = None
        self._dict_row = None
        if self.db_backend == "postgres":
            try:
                import psycopg
                from psycopg.rows import dict_row
            except Exception as exc:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    "Postgres inspection requires psycopg. Install project dependencies first."
                ) from exc
            if not self.database_url:
                raise RuntimeError("DB_BACKEND=postgres but DATABASE_URL is empty.")
            self._psycopg = psycopg
            self._dict_row = dict_row

    def _connect_sqlite(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_db_path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _connect_postgres(self):
        assert self._psycopg is not None and self._dict_row is not None
        return self._psycopg.connect(
            self.database_url,
            row_factory=self._dict_row,
            autocommit=False,
            connect_timeout=self.pg_connect_timeout_sec,
        )

    def _connect(self):
        if self.db_backend == "postgres":
            return self._connect_postgres()
        return self._connect_sqlite()

    def _scalar(self, query_sqlite: str, query_pg: str, params: Iterable[Any]) -> int:
        with self._connect() as conn:
            if self.db_backend == "postgres":
                row = conn.execute(query_pg, tuple(params)).fetchone()
            else:
                row = conn.execute(query_sqlite, tuple(params)).fetchone()
        if not row:
            return 0
        value = row["c"] if isinstance(row, dict) else row["c"]
        return int(value or 0)

    def table_count(self, table: str) -> int:
        table_name = str(table).strip().lower()
        allowed = {"documents", "chunks", "embeddings", "diagram_graphs"}
        if table_name not in allowed:
            raise RuntimeError(f"Unsupported table for counting: {table}")
        query = f"SELECT COUNT(*) AS c FROM {table_name}"
        return self._scalar(query, query, [])

    def doc_chunk_count(self, doc_id: str) -> int:
        return self._scalar(
            "SELECT COUNT(*) AS c FROM chunks WHERE doc_id = ?",
            "SELECT COUNT(*) AS c FROM chunks WHERE doc_id = %s",
            [doc_id],
        )

    def doc_embedding_count(self, doc_id: str) -> int:
        return self._scalar(
            "SELECT COUNT(*) AS c FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?)",
            "SELECT COUNT(*) AS c FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = %s)",
            [doc_id],
        )

    def doc_diagram_graph_count(self, doc_id: str) -> int:
        return self._scalar(
            "SELECT COUNT(*) AS c FROM diagram_graphs WHERE doc_id = ?",
            "SELECT COUNT(*) AS c FROM diagram_graphs WHERE doc_id = %s",
            [doc_id],
        )

    def doc_source_type_counts(self, doc_id: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        with self._connect() as conn:
            if self.db_backend == "postgres":
                rows = conn.execute(
                    """
                    SELECT source_type, COUNT(*) AS c
                    FROM chunks
                    WHERE doc_id = %s
                    GROUP BY source_type
                    ORDER BY source_type
                    """,
                    (doc_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT source_type, COUNT(*) AS c
                    FROM chunks
                    WHERE doc_id = ?
                    GROUP BY source_type
                    ORDER BY source_type
                    """,
                    (doc_id,),
                ).fetchall()
        for row in rows:
            source_type = str(row["source_type"] or "")
            out[source_type] = int(row["c"] or 0)
        return out

    def sample_chunks(self, doc_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        n = max(1, int(limit))
        with self._connect() as conn:
            if self.db_backend == "postgres":
                rows = conn.execute(
                    """
                    SELECT id, doc_id, page, chunk_index, source_type, content, metadata
                    FROM chunks
                    WHERE doc_id = %s
                    ORDER BY page NULLS LAST, chunk_index NULLS LAST
                    LIMIT %s
                    """,
                    (doc_id, n),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, doc_id, page, chunk_index, source_type, content, metadata
                    FROM chunks
                    WHERE doc_id = ?
                    ORDER BY page, chunk_index
                    LIMIT ?
                    """,
                    (doc_id, n),
                ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["metadata"] = _decode_json(item.get("metadata")) or {}
            out.append(item)
        return out

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            if self.db_backend == "postgres":
                row = conn.execute("SELECT * FROM documents WHERE id = %s", (doc_id,)).fetchone()
            else:
                row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not row:
            return None
        out = dict(row)
        out["metadata"] = _decode_json(out.get("metadata")) or {}
        return out

    def embedding_dim_distribution(self, doc_id: Optional[str] = None) -> Dict[int, int]:
        out: Dict[int, int] = {}
        with self._connect() as conn:
            if doc_id:
                if self.db_backend == "postgres":
                    rows = conn.execute(
                        """
                        SELECT e.dim AS dim, COUNT(*) AS c
                        FROM embeddings e
                        JOIN chunks c ON c.id = e.chunk_id
                        WHERE c.doc_id = %s
                        GROUP BY e.dim
                        ORDER BY e.dim
                        """,
                        (doc_id,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT e.dim AS dim, COUNT(*) AS c
                        FROM embeddings e
                        JOIN chunks c ON c.id = e.chunk_id
                        WHERE c.doc_id = ?
                        GROUP BY e.dim
                        ORDER BY e.dim
                        """,
                        (doc_id,),
                    ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT dim, COUNT(*) AS c FROM embeddings GROUP BY dim ORDER BY dim"
                ).fetchall()
        for row in rows:
            out[int(row["dim"])] = int(row["c"] or 0)
        return out

