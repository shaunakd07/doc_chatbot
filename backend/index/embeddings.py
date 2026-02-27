from __future__ import annotations

import os

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import torch


class Embedder:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        provider: str = "local",
        openai_api_key: str | None = None,
    ) -> None:
        self.device = device
        self.provider = (provider or "local").strip().lower()
        self.model_name = model_name
        self.model = None
        self.client = None
        self.openai_batch_size = max(1, int(os.getenv("OPENAI_EMBED_BATCH_SIZE", "96")))
        self.local_batch_size = max(1, int(os.getenv("LOCAL_EMBED_BATCH_SIZE", "64")))
        if self.provider == "openai":
            if not openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is required when EMBED_PROVIDER=openai.")
            self.client = OpenAI(api_key=openai_api_key)
        else:
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested for embeddings but no CUDA device is available.")
            self.model = SentenceTransformer(model_name, device=device)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        if vectors.size == 0:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return vectors / norms

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype="float32")

        unique_texts: list[str] = []
        unique_lookup: dict[str, int] = {}
        remap: list[int] = []
        for raw in texts:
            text = str(raw or "")
            existing = unique_lookup.get(text)
            if existing is None:
                existing = len(unique_texts)
                unique_lookup[text] = existing
                unique_texts.append(text)
            remap.append(existing)

        if self.provider == "openai":
            assert self.client is not None
            unique_vectors: list[list[float]] = []
            for start in range(0, len(unique_texts), self.openai_batch_size):
                batch = unique_texts[start : start + self.openai_batch_size]
                response = self.client.embeddings.create(model=self.model_name, input=batch)
                unique_vectors.extend(row.embedding for row in response.data)
            vectors_unique = self._normalize(np.asarray(unique_vectors, dtype="float32")).astype("float32")
            return vectors_unique[np.asarray(remap, dtype=np.int64)]

        assert self.model is not None
        vectors_unique = self.model.encode(
            unique_texts,
            normalize_embeddings=True,
            device=self.device,
            batch_size=self.local_batch_size,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors_unique, dtype="float32")
        return vectors[np.asarray(remap, dtype=np.int64)]

    def embed_query(self, text: str) -> np.ndarray:
        if self.provider == "openai":
            assert self.client is not None
            response = self.client.embeddings.create(model=self.model_name, input=[text])
            vector = np.asarray(response.data[0].embedding, dtype="float32")
            norm = max(float(np.linalg.norm(vector)), 1e-12)
            return (vector / norm).astype("float32")
        assert self.model is not None
        vectors = self.model.encode([text], normalize_embeddings=True, device=self.device)
        return np.asarray(vectors[0], dtype="float32")
