from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

ALLOWED_TASK_TYPES = {
    "qa",
    "count",
    "metadata_query",
    "compare",
    "summarize",
    "image_qa",
    "timeline",
    "trend_analysis",
    "out_of_scope",
}
ALLOWED_STRATEGIES = {"semantic", "balanced", "image_first"}
ALLOWED_METADATA_OPERATIONS = {
    "count",
    "latest_uploaded",
    "earliest_uploaded",
    "created_after",
    "created_before",
    "created_between",
    "modified_within_days",
    "list",
    "most_frequently_updated",
    "changed_between_versions",
    "authored_by",
    "edited_by",
    "last_modified_by",
    "external_collaborators",
    "uploaded_by_role",
}
ALLOWED_EXPECTED_ANSWER_TYPES = {
    "count",
    "person",
    "document",
    "list",
    "comparison",
    "timeline",
    "boolean",
    "unknown",
}
ALLOWED_FACT_TYPES = {"date", "amount", "party"}


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
            "Return ONLY JSON with keys: task_type, needs_cross_doc, needs_numeric_extraction, "
            "needs_image_reasoning, retrieval_plan, analysis_plan, expected_answer_type, confidence, rationale. "
            "task_type must be one of qa, count, metadata_query, compare, summarize, image_qa, timeline, trend_analysis, out_of_scope. "
            "retrieval_plan must contain strategy (semantic|balanced|image_first), top_k, per_doc_limit. "
            "Set task_type=count or metadata_query for metadata-driven questions (counts, latest/earliest upload, date filters, "
            "authors/editors, collaborator role, versions/changes). "
            "If the user asks to summarize, explain, or say what one named document/agreement/NDA says and is not asking for "
            "differences between documents, set task_type=qa and needs_cross_doc=false. "
            "Set task_type=compare ONLY when the user explicitly asks to compare or asks for differences, changes, "
            "before/after, or timeline evolution. For multi-document synthesis or aggregation questions that do not "
            "ask for differences, set task_type=qa and needs_cross_doc=true. "
            "Set task_type=trend_analysis for analytical questions requiring trend/pattern analysis over evidence. "
            "For trend_analysis set needs_cross_doc=true and needs_numeric_extraction=true. "
            "analysis_plan should include query_entities (list[str]) and evidence_classes (list[str]) inferred "
            "from the query content, not hardcoded labels. "
            "When the user explicitly names documents, analysis_plan may include target_documents (list[str]). "
            "When the question asks for exact factual fields such as dates, amounts, values, parties, counterparties, "
            "or 'which document contains X', analysis_plan may include exact_lookup_requested (bool) and fact_types "
            "(subset of date|amount|party). "
            "For compare questions that also ask for exact facts across two or more named documents, set task_type=compare, "
            "needs_cross_doc=true, analysis_plan.exact_lookup_requested=true, include fact_types, and include "
            "target_documents when identifiable. "
            "expected_answer_type must be one of count, person, document, list, comparison, timeline, boolean, unknown. "
            "For count/metadata_query, analysis_plan should include metadata_operation and metadata_filters. "
            "metadata_operation must be one of count, latest_uploaded, earliest_uploaded, created_after, created_before, "
            "created_between, modified_within_days, list, most_frequently_updated, changed_between_versions, authored_by, "
            "edited_by, last_modified_by, external_collaborators, uploaded_by_role. "
            "metadata_filters may include doc_type, author, last_modified_by, uploader_role, collaborator_type, date_from, date_to, "
            "relative_days, target_document, version_a, version_b. "
            "When needs_cross_doc=true, prefer retrieval_plan.strategy=balanced with top_k>=10 and per_doc_limit>=2. "
            "needs_cross_doc indicates retrieval coverage across multiple docs, not answer format."
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
        needs_cross_doc = bool(data.get("needs_cross_doc", False))
        needs_numeric_extraction = bool(data.get("needs_numeric_extraction", False))
        if task_type in {"count", "metadata_query"}:
            needs_cross_doc = True
            needs_numeric_extraction = False
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
            analysis_plan["exact_lookup_requested"] = bool(raw_plan.get("exact_lookup_requested", False))
            raw_fact_types = raw_plan.get("fact_types")
            fact_types: List[str] = []
            if isinstance(raw_fact_types, list):
                for value in raw_fact_types[:4]:
                    fact_type = str(value or "").strip().lower()
                    if fact_type in ALLOWED_FACT_TYPES and fact_type not in fact_types:
                        fact_types.append(fact_type)
            if fact_types:
                analysis_plan["fact_types"] = fact_types
            raw_target_documents = raw_plan.get("target_documents")
            target_documents: List[str] = []
            if isinstance(raw_target_documents, list):
                for value in raw_target_documents[:6]:
                    target = str(value or "").strip()
                    if target and target not in target_documents:
                        target_documents.append(target)
            if target_documents:
                analysis_plan["target_documents"] = target_documents
            metadata_operation = str(raw_plan.get("metadata_operation", "")).strip().lower()
            if metadata_operation in ALLOWED_METADATA_OPERATIONS:
                analysis_plan["metadata_operation"] = metadata_operation
            raw_filters = raw_plan.get("metadata_filters")
            metadata_filters: Dict[str, Any] = {}
            if isinstance(raw_filters, dict):
                for key in (
                    "doc_type",
                    "author",
                    "last_modified_by",
                    "uploader_role",
                    "collaborator_type",
                    "date_from",
                    "date_to",
                    "target_document",
                ):
                    value = str(raw_filters.get(key, "")).strip()
                    if value:
                        metadata_filters[key] = value
                for key in ("relative_days", "version_a", "version_b"):
                    try:
                        value = int(raw_filters.get(key))
                    except Exception:
                        value = None
                    if value is not None:
                        metadata_filters[key] = value
            if metadata_filters:
                analysis_plan["metadata_filters"] = metadata_filters
        expected_answer_type = str(data.get("expected_answer_type", "")).strip().lower()
        if expected_answer_type not in ALLOWED_EXPECTED_ANSWER_TYPES:
            if task_type == "count":
                operation_hint = str(analysis_plan.get("metadata_operation") or "").strip().lower()
                expected_answer_type = "count" if operation_hint == "count" else "unknown"
            elif task_type in {"compare", "trend_analysis"}:
                expected_answer_type = "comparison"
            elif task_type == "timeline":
                expected_answer_type = "timeline"
            elif task_type == "metadata_query":
                operation = str(analysis_plan.get("metadata_operation") or "").strip().lower()
                if operation == "count":
                    expected_answer_type = "count"
                elif operation in {"last_modified_by", "authored_by", "edited_by"}:
                    expected_answer_type = "person"
                elif operation in {"latest_uploaded", "earliest_uploaded", "most_frequently_updated", "changed_between_versions"}:
                    expected_answer_type = "document"
                else:
                    expected_answer_type = "list"
            else:
                expected_answer_type = "unknown"
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
            "expected_answer_type": expected_answer_type,
            "confidence": confidence,
            "rationale": str(data.get("rationale", "")).strip(),
        }
