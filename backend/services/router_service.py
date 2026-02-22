from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)

ALLOWED_TASK_TYPES = {"qa", "compare", "summarize", "image_qa", "timeline", "out_of_scope"}
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
