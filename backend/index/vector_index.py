from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .. import storage


class VectorIndex:
    def __init__(self) -> None:
        self.chunk_ids: List[str] = []
        self.embeddings = np.zeros((0, 1), dtype="float32")

    def load(self) -> None:
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
        if vectors.size == 0:
            return
        if self.embeddings.size == 0:
            self.embeddings = vectors.astype("float32")
        else:
            self.embeddings = np.vstack([self.embeddings, vectors.astype("float32")])
        self.chunk_ids.extend(chunk_ids)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.embeddings.size == 0:
            return []
        query = query_vector.astype("float32")
        scores = self.embeddings @ query
        top_k = min(top_k, scores.shape[0])
        indices = np.argpartition(-scores, top_k - 1)[:top_k]
        ranked = sorted(((idx, scores[idx]) for idx in indices), key=lambda x: x[1], reverse=True)
        return [(self.chunk_ids[idx], float(score)) for idx, score in ranked]
