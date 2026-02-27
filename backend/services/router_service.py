from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)

ALLOWED_TASK_TYPES = {
    "qa",
    "compare",
    "summarize",
    "image_qa",
    "timeline",
    "trend_analysis",
    "out_of_scope",
}
ALLOWED_STRATEGIES = {"semantic", "balanced", "image_first"}


class RouterService:
    def __init__(self, model_id: str, device: str = "auto", max_new_tokens: int = 196) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _model_device(self) -> torch.device:
        if self.model is None:
            return torch.device(self._resolve_device())
        try:
            return next(self.model.parameters()).device
        except Exception:
            return torch.device(self._resolve_device())

    def load(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return
        kwargs: Dict[str, Any] = {}
        runtime_device = self._resolve_device()
        dtype = torch.float16 if runtime_device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if runtime_device.startswith("cuda"):
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map="auto",
                **kwargs,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                **kwargs,
            )
            self.model.to(runtime_device)
        self.model.eval()

    def route(
        self,
        question: str,
        doc_ids: Optional[List[str]] = None,
        available_docs: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        self.load()
        doc_count = len(doc_ids or [])
        available_count = len(available_docs or [])
        system_prompt = (
            "You are a routing classifier for a document chatbot. "
            "Return ONLY JSON with keys: task_type, needs_cross_doc, needs_numeric_extraction, "
            "needs_image_reasoning, retrieval_plan, analysis_plan, confidence, rationale. "
            "task_type must be one of qa, compare, summarize, image_qa, timeline, trend_analysis, out_of_scope. "
            "retrieval_plan must contain strategy (semantic|balanced|image_first), top_k, per_doc_limit. "
            "Set task_type=compare ONLY when the user explicitly asks to compare or asks for differences, changes, "
            "before/after, or timeline evolution. For multi-document synthesis or aggregation questions that do not "
            "ask for differences, set task_type=qa and needs_cross_doc=true. "
            "Set task_type=trend_analysis for analytical questions requiring trend/pattern analysis over evidence. "
            "For trend_analysis set needs_cross_doc=true and needs_numeric_extraction=true. "
            "analysis_plan should include query_entities (list[str]) and evidence_classes (list[str]) inferred "
            "from the query content, not hardcoded labels. "
            "When needs_cross_doc=true, prefer retrieval_plan.strategy=balanced with top_k>=10 and per_doc_limit>=2. "
            "needs_cross_doc indicates retrieval coverage across multiple docs, not answer format."
        )
        user_prompt = (
            f"Question: {question}\n"
            f"SelectedDocCount: {doc_count}\n"
            f"AvailableDocCount: {available_count}\n"
            "JSON:"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        assert self.tokenizer is not None
        assert self.model is not None
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt")
        device = self._model_device()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = int(inputs["input_ids"].shape[-1])
        text = self.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
        parsed = self._parse_route_json(text)
        if parsed is None:
            raise ValueError(f"Router returned non-JSON response: {text[:400]}")
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
        needs_cross_doc = bool(data.get("needs_cross_doc", False))
        needs_numeric_extraction = bool(data.get("needs_numeric_extraction", False))
        if task_type == "trend_analysis":
            needs_cross_doc = True
            needs_numeric_extraction = True
        if needs_cross_doc:
            if strategy == "semantic":
                strategy = "balanced"
            top_k = max(10, top_k)
            per_doc_limit = max(2, per_doc_limit)
        if needs_numeric_extraction:
            if strategy == "semantic":
                strategy = "balanced"
            top_k = max(12, top_k)
            per_doc_limit = max(2, per_doc_limit)

        analysis_plan: Dict[str, Any] = {}
        raw_plan = data.get("analysis_plan")
        if isinstance(raw_plan, dict):
            raw_entities = raw_plan.get("query_entities")
            entities: List[str] = []
            if isinstance(raw_entities, list):
                for value in raw_entities[:12]:
                    text = str(value or "").strip()
                    if text and text not in entities:
                        entities.append(text)
            raw_classes = raw_plan.get("evidence_classes")
            evidence_classes: List[str] = []
            if isinstance(raw_classes, list):
                for value in raw_classes[:8]:
                    if isinstance(value, dict):
                        text = str(value.get("label") or value.get("name") or "").strip()
                    else:
                        text = str(value or "").strip()
                    if text and text not in evidence_classes:
                        evidence_classes.append(text)
            if entities:
                analysis_plan["query_entities"] = entities
            if evidence_classes:
                analysis_plan["evidence_classes"] = evidence_classes
        return {
            "task_type": task_type,
            "needs_cross_doc": needs_cross_doc,
            "needs_numeric_extraction": needs_numeric_extraction,
            "needs_image_reasoning": bool(data.get("needs_image_reasoning", False)),
            "retrieval_plan": {
                "strategy": strategy,
                "top_k": max(1, min(32, top_k)),
                "per_doc_limit": max(1, min(12, per_doc_limit)),
            },
            "analysis_plan": analysis_plan,
            "confidence": confidence,
            "rationale": str(data.get("rationale", "")).strip(),
        }
