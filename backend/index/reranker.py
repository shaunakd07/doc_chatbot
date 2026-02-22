from __future__ import annotations

import re
from typing import List, Optional

import torch
from sentence_transformers import CrossEncoder

TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")


class Reranker:
    def __init__(
        self,
        model_name: Optional[str],
        device: str = "auto",
        enabled: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.enabled = enabled and bool(model_name)
        self.model: CrossEncoder | None = None
        self.load_error: str | None = None

    def _resolve_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def load(self) -> None:
        if not self.enabled or self.model is not None:
            return
        try:
            self.model = CrossEncoder(self.model_name, device=self._resolve_device())
        except Exception as exc:
            self.load_error = str(exc)
            self.enabled = False
            self.model = None

    def _lexical_fallback_score(self, query: str, text: str) -> float:
        q_tokens = {token.lower() for token in TOKEN_RE.findall(query)}
        if not q_tokens:
            return 0.0
        t_tokens = {token.lower() for token in TOKEN_RE.findall(text)}
        if not t_tokens:
            return 0.0
        overlap = len(q_tokens.intersection(t_tokens))
        return overlap / float(len(q_tokens))

    def rerank(self, query: str, chunks: List[dict]) -> List[dict]:
        if not chunks:
            return chunks
        self.load()
        if self.enabled and self.model is not None:
            pairs = [(query, str(chunk.get("content") or "")) for chunk in chunks]
            scores = self.model.predict(pairs)
            for chunk, score in zip(chunks, scores):
                chunk["rerank_score"] = float(score)
        else:
            for chunk in chunks:
                chunk["rerank_score"] = self._lexical_fallback_score(query, str(chunk.get("content") or ""))
        return sorted(chunks, key=lambda c: float(c.get("rerank_score", 0.0)), reverse=True)
