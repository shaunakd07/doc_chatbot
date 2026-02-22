from __future__ import annotations

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
        if self.provider == "openai":
            assert self.client is not None
            response = self.client.embeddings.create(model=self.model_name, input=texts)
            vectors = np.asarray([row.embedding for row in response.data], dtype="float32")
            return self._normalize(vectors).astype("float32")
        assert self.model is not None
        vectors = self.model.encode(texts, normalize_embeddings=True, device=self.device)
        return np.asarray(vectors, dtype="float32")

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
