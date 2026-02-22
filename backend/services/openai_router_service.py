from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

ALLOWED_TASK_TYPES = {"qa", "compare", "summarize", "image_qa", "timeline", "out_of_scope"}
ALLOWED_STRATEGIES = {"semantic", "balanced", "image_first"}


class OpenAIRouterService:
    def __init__(self, model_id: str, api_key: str, max_new_tokens: int = 220) -> None:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when ROUTER_PROVIDER=openai.")
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.client = OpenAI(api_key=api_key)

    def route(
        self,
        question: str,
        doc_ids: Optional[List[str]] = None,
        available_docs: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        doc_count = len(doc_ids or [])
        available_count = len(available_docs or [])
        system_prompt = (
            "You are a routing classifier for a document chatbot. "
            "Return ONLY JSON with keys: task_type, needs_cross_doc, needs_image_reasoning, "
            "retrieval_plan, confidence, rationale. "
            "task_type must be one of qa, compare, summarize, image_qa, timeline, out_of_scope. "
            "retrieval_plan must contain strategy (semantic|balanced|image_first), top_k, per_doc_limit."
        )
        user_prompt = (
            f"Question: {question}\n"
            f"SelectedDocCount: {doc_count}\n"
            f"AvailableDocCount: {available_count}\n"
            "JSON:"
        )
        response = self.client.chat.completions.create(
            model=self.model_id,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=max(140, int(self.max_new_tokens)),
        )
        text = (response.choices[0].message.content or "").strip()
        parsed = self._parse_route_json(text)
        if parsed is None:
            raise ValueError(f"Router returned invalid JSON: {text[:400]}")
        return parsed

    def _parse_route_json(self, raw: str) -> Dict[str, Any] | None:
        candidate = raw.strip()
        if not candidate:
            return None
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if match:
            candidate = match.group(0)
        try:
            data = json.loads(candidate)
        except Exception:
            return None
        task_type = str(data.get("task_type", "")).strip().lower()
        if task_type not in ALLOWED_TASK_TYPES:
            return None
        retrieval = data.get("retrieval_plan") or {}
        strategy = str(retrieval.get("strategy", "semantic")).strip().lower()
        if strategy not in ALLOWED_STRATEGIES:
            strategy = "semantic"
        try:
            top_k = int(retrieval.get("top_k", 8))
        except Exception:
            top_k = 8
        try:
            per_doc_limit = int(retrieval.get("per_doc_limit", 4))
        except Exception:
            per_doc_limit = 4
        try:
            confidence = float(data.get("confidence", 0.5))
        except Exception:
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))
        return {
            "task_type": task_type,
            "needs_cross_doc": bool(data.get("needs_cross_doc", False)),
            "needs_image_reasoning": bool(data.get("needs_image_reasoning", False)),
            "retrieval_plan": {
                "strategy": strategy,
                "top_k": max(1, min(32, top_k)),
                "per_doc_limit": max(1, min(12, per_doc_limit)),
            },
            "confidence": confidence,
            "rationale": str(data.get("rationale", "")).strip(),
        }
