import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

try:
    import psycopg
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        f"psycopg is required for migration: {exc}\nInstall with: pip install -r requirements.txt"
    )


def _json_payload(value: Any) -> str:
    if value is None:
        return "{}"
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "{}"
        try:
            json.loads(text)
            return text
        except Exception:
            return json.dumps({"raw": text})
    return json.dumps(value)


def _vector_literal_from_blob(blob: bytes) -> tuple[str, int]:
    vector = np.frombuffer(blob, dtype="float32")
    literal = "[" + ",".join(f"{float(v):.8f}" for v in vector.tolist()) + "]"
    return literal, int(vector.shape[0])


def _ensure_schema(conn, dim: int) -> None:
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
    except Exception:
        # Best-effort index creation; continue migration even if this fails.
        pass
    conn.commit()


def migrate(sqlite_path: Path, database_url: str, pgvector_dim: int, truncate_target: bool) -> None:
    sqlite_conn = sqlite3.connect(str(sqlite_path))
    sqlite_conn.row_factory = sqlite3.Row

    pg_conn = psycopg.connect(database_url, autocommit=False)
    try:
        _ensure_schema(pg_conn, pgvector_dim)

        if truncate_target:
            cur = pg_conn.cursor()
            cur.execute("DELETE FROM diagram_graphs")
            cur.execute("DELETE FROM embeddings")
            cur.execute("DELETE FROM chunks")
            cur.execute("DELETE FROM documents")
            pg_conn.commit()

        s_cur = sqlite_conn.cursor()
        p_cur = pg_conn.cursor()

        docs = s_cur.execute(
            "SELECT id, filename, status, created_at, num_pages, metadata FROM documents"
        ).fetchall()
        p_cur.executemany(
            """
            INSERT INTO documents (id, filename, status, created_at, num_pages, metadata)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                filename = EXCLUDED.filename,
                status = EXCLUDED.status,
                created_at = EXCLUDED.created_at,
                num_pages = EXCLUDED.num_pages,
                metadata = EXCLUDED.metadata
            """,
            [
                (
                    row["id"],
                    row["filename"],
                    row["status"],
                    row["created_at"],
                    int(row["num_pages"] or 0),
                    _json_payload(row["metadata"]),
                )
                for row in docs
            ],
        )

        chunks = s_cur.execute(
            "SELECT id, doc_id, page, chunk_index, content, source_type, metadata FROM chunks"
        ).fetchall()
        p_cur.executemany(
            """
            INSERT INTO chunks (id, doc_id, page, chunk_index, content, source_type, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                page = EXCLUDED.page,
                chunk_index = EXCLUDED.chunk_index,
                content = EXCLUDED.content,
                source_type = EXCLUDED.source_type,
                metadata = EXCLUDED.metadata
            """,
            [
                (
                    row["id"],
                    row["doc_id"],
                    row["page"],
                    row["chunk_index"],
                    row["content"],
                    row["source_type"],
                    _json_payload(row["metadata"]),
                )
                for row in chunks
            ],
        )

        embeddings = s_cur.execute("SELECT chunk_id, vector, dim FROM embeddings").fetchall()
        embedding_rows = []
        for row in embeddings:
            vector_literal, actual_dim = _vector_literal_from_blob(row["vector"])
            if actual_dim != pgvector_dim:
                raise RuntimeError(
                    f"Embedding dim mismatch for chunk {row['chunk_id']}: sqlite has {actual_dim}, "
                    f"target schema expects {pgvector_dim}. Use --pgvector-dim {actual_dim}."
                )
            embedding_rows.append((row["chunk_id"], vector_literal, int(row["dim"])))
        if embedding_rows:
            p_cur.executemany(
                """
                INSERT INTO embeddings (chunk_id, embedding, dim)
                VALUES (%s, %s::vector, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    dim = EXCLUDED.dim
                """,
                embedding_rows,
            )

        graphs = s_cur.execute(
            """
            SELECT id, doc_id, page, image_path, parser_version, graph_json, confidence, created_at, metadata
            FROM diagram_graphs
            """
        ).fetchall()
        p_cur.executemany(
            """
            INSERT INTO diagram_graphs (id, doc_id, page, image_path, parser_version, graph_json, confidence, created_at, metadata)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                doc_id = EXCLUDED.doc_id,
                page = EXCLUDED.page,
                image_path = EXCLUDED.image_path,
                parser_version = EXCLUDED.parser_version,
                graph_json = EXCLUDED.graph_json,
                confidence = EXCLUDED.confidence,
                created_at = EXCLUDED.created_at,
                metadata = EXCLUDED.metadata
            """,
            [
                (
                    row["id"],
                    row["doc_id"],
                    row["page"],
                    row["image_path"],
                    row["parser_version"],
                    _json_payload(row["graph_json"]),
                    float(row["confidence"] or 0.0),
                    row["created_at"],
                    _json_payload(row["metadata"]),
                )
                for row in graphs
            ],
        )

        pg_conn.commit()

        print(f"Migrated documents: {len(docs)}")
        print(f"Migrated chunks: {len(chunks)}")
        print(f"Migrated embeddings: {len(embeddings)}")
        print(f"Migrated diagram_graphs: {len(graphs)}")
        print("Done.")
    finally:
        sqlite_conn.close()
        pg_conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate doc_chatbot data from SQLite to PostgreSQL+pgvector.")
    parser.add_argument(
        "--sqlite-path",
        default="data/db/app.db",
        help="Path to SQLite file (default: data/db/app.db)",
    )
    parser.add_argument(
        "--database-url",
        required=True,
        help="PostgreSQL connection URL, e.g. postgresql://postgres:postgres@localhost:5432/doc_chatbot",
    )
    parser.add_argument(
        "--pgvector-dim",
        type=int,
        default=1536,
        help="Vector dimension for pgvector schema (default: 1536)",
    )
    parser.add_argument(
        "--truncate-target",
        action="store_true",
        help="Delete existing rows in target tables before migrating.",
    )
    args = parser.parse_args()

    sqlite_path = Path(args.sqlite_path).resolve()
    if not sqlite_path.exists():
        raise SystemExit(f"SQLite file not found: {sqlite_path}")
    migrate(
        sqlite_path=sqlite_path,
        database_url=args.database_url.strip(),
        pgvector_dim=max(1, int(args.pgvector_dim)),
        truncate_target=bool(args.truncate_target),
    )


if __name__ == "__main__":
    main()
