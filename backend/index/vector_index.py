from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .. import config, storage


class VectorIndex:
    def __init__(self) -> None:
        self.chunk_ids: List[str] = []
        self.embeddings = np.zeros((0, 1), dtype="float32")
        self._use_db_search = str(config.DB_BACKEND).strip().lower() == "postgres"

    def load(self) -> None:
        if self._use_db_search:
            # pgvector search runs directly against PostgreSQL; no in-memory matrix preload needed.
            self.chunk_ids = []
            self.embeddings = np.zeros((0, 1), dtype="float32")
            return
        records = storage.load_embeddings()
        if not records:
            self.chunk_ids = []
            self.embeddings = np.zeros((0, 1), dtype="float32")
            return
        chunk_ids: List[str] = []
        vectors = []
        dim = None
        for chunk_id, blob, d in records:
            vec = np.frombuffer(blob, dtype="float32")
            if dim is None:
                dim = d
            vectors.append(vec)
            chunk_ids.append(chunk_id)
        self.chunk_ids = chunk_ids
        self.embeddings = np.vstack(vectors).astype("float32")

    def add(self, vectors: np.ndarray, chunk_ids: List[str]) -> None:
        if self._use_db_search:
            # Vectors are already persisted to PostgreSQL in storage.add_embeddings().
            return
        if vectors.size == 0:
            return
        if self.embeddings.size == 0:
            self.embeddings = vectors.astype("float32")
        else:
            self.embeddings = np.vstack([self.embeddings, vectors.astype("float32")])
        self.chunk_ids.extend(chunk_ids)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if self._use_db_search:
            return storage.search_embeddings(query_vector, top_k=top_k)
        if self.embeddings.size == 0:
            return []
        query = query_vector.astype("float32")
        if self.embeddings.ndim != 2 or query.ndim != 1:
            raise ValueError("Invalid embedding tensor shape for dense search.")
        if self.embeddings.shape[1] != query.shape[0]:
            raise ValueError(
                "Embedding dimension mismatch between index and query "
                f"(index_dim={self.embeddings.shape[1]}, query_dim={query.shape[0]}). "
                "Rebuild embeddings or align the configured embedding model."
            )
        scores = self.embeddings @ query
        top_k = min(top_k, scores.shape[0])
        indices = np.argpartition(-scores, top_k - 1)[:top_k]
        ranked = sorted(((idx, scores[idx]) for idx in indices), key=lambda x: x[1], reverse=True)
        return [(self.chunk_ids[idx], float(score)) for idx, score in ranked]
