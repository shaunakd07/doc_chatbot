from __future__ import annotations

import json
import logging
import re
import traceback
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, List, Optional

import numpy as np

from .. import config, storage
from ..ingestion.doc_tags import build_document_auto_tags, normalize_tags
from ..ingestion.doc_types import DOC_TYPE_HINTS, DOC_TYPE_LABELS, extract_query_doc_type_candidates
from ..models.prompts import build_compare_prompt, build_prompt
from .metadata_semantic_adapter import MetadataSemanticAdapter
from PIL import Image

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_&.\-]{1,}")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
COUNT_SIGNAL_RE = re.compile(r"\b(?:how many|number of|count|total number|total count)\b", flags=re.IGNORECASE)
NUMERIC_SIGNAL_RE = re.compile(
    r"(?i)(?:[$€£¥]\s?\d[\d,]*(?:\.\d+)?|\b(?:usd|sgd|inr|idr|myr|rm|eur|gbp)\b|\b\d[\d,]*(?:\.\d+)?\s?(?:%|k|m|b|million|billion|crore|lakh)\b)"
)
QUERY_STOPWORDS = {
    "a",
    "about",
    "all",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "give",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "list",
    "me",
    "of",
    "on",
    "or",
    "our",
    "please",
    "provide",
    "show",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "to",
    "us",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}
LOW_SIGNAL_TERMS = {
    "service",
    "provide",
    "provided",
    "shown",
    "show",
    "detail",
    "details",
    "information",
    "document",
    "documents",
    "file",
    "files",
}
METADATA_OPERATIONS = {
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
EXPECTED_ANSWER_TYPES = {
    "count",
    "person",
    "document",
    "list",
    "comparison",
    "timeline",
    "boolean",
    "unknown",
}
PERSON_METADATA_OPERATIONS = {"last_modified_by", "authored_by", "edited_by"}
GENERIC_METADATA_OPERATIONS = {"list", "count"}
QUERY_INTENT_METADATA_FIELDS = {
    "doc_type",
    "author",
    "last_modified_by",
    "uploader_role",
    "collaborator_type",
    "date_from",
    "date_to",
    "relative_days",
    "target_document",
    "version_a",
    "version_b",
}
QUERY_INTENT_NON_SEMANTIC_TERMS = {
    "count",
    "list",
    "number",
    "many",
    "total",
    "document",
    "documents",
    "file",
    "files",
    "show",
    "give",
    "find",
}
SUMMARY_ROUTE_PREFIX_RE = re.compile(
    r"(?is)^\s*(?:please\s+)?(?:summari[sz]e|describe|explain|tell me about)\s+(?:the\s+)?(?P<body>.+?)\s*$"
)
SUMMARY_ROUTE_WHAT_SAY_RE = re.compile(
    r"(?is)^\s*what does\s+(?:the\s+)?(?P<body>.+?)\s+say\s*[?.!]*\s*$"
)
SEMANTIC_TEMPORAL_TERMS = {
    "effective",
    "execution",
    "executed",
    "signed",
    "dated",
    "term",
    "termination",
    "expiry",
    "expiration",
    "valid",
    "validity",
    "active",
    "inactive",
    "lapse",
    "lapsed",
}
SEMANTIC_PARTY_TERMS = {
    "party",
    "parties",
    "counterparty",
    "counterparties",
    "customer",
    "client",
    "vendor",
    "supplier",
    "recipient",
    "buyer",
    "seller",
}
STATUS_EXPIRED_TERMS = {"expired", "inactive", "lapsed", "terminated", "ended"}
STATUS_ACTIVE_TERMS = {"active", "valid", "effective", "enforceable", "current"}
SEMANTIC_EXPANSION_TERMS = (
    "agreement effective date execution date signature date termination date expiry date party counterparty validity period",
    "contract commencement date end date renewal clause signatory named party legal entity",
)
DATE_ROLE_TERMS = {
    "termination": {"termination", "terminate", "terminated", "expiry", "expiration", "expires", "end"},
    "execution": {"executed", "execution", "signed", "dated", "signature"},
    "effective": {"effective", "commencement", "start", "in_force", "valid_from"},
}
MONTH_NAME_TO_NUM = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
ISO_DATE_RE = re.compile(r"\b((?:19|20)\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b")
DMY_MONTH_RE = re.compile(
    r"\b([0-3]?\d)\s+"
    r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+((?:19|20)\d{2})\b",
    flags=re.IGNORECASE,
)
MONTH_DY_RE = re.compile(
    r"\b("
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+([0-3]?\d),\s*((?:19|20)\d{2})\b",
    flags=re.IGNORECASE,
)
QUERY_AMOUNT_RE = re.compile(
    r"(?ix)"
    r"(?<![A-Za-z0-9])"
    r"(?:(?P<currency_before>USD|SGD|EUR|GBP|INR|IDR|MYR|RM|AED|AUD|CAD|CHF|CNY|HKD|JPY)\s*|(?P<symbol>[$]))"
    r"(?P<number>\d{1,3}(?:,\d{3})+|\d+)"
    r"(?P<decimal>\.\d{1,2})?"
    r"(?:\s*(?P<magnitude>k|m|b|thousand|million|billion))?"
    r"(?:\s*(?P<currency_after>USD|SGD|EUR|GBP|INR|IDR|MYR|RM|AED|AUD|CAD|CHF|CNY|HKD|JPY))?"
    r"(?![A-Za-z0-9])"
)
EXACT_DOCUMENT_LOOKUP_MARKERS = (
    "which document",
    "which documents",
    "what document",
    "what documents",
    "which file",
    "which files",
)
EXACT_CONTAINS_MARKERS = (
    "contains",
    "contain",
    "mentions",
    "mention",
    "mentioned",
    "includes",
    "include",
    "included",
    "has",
    "have",
    "named",
)
FACT_DATE_HINTS = {
    "date",
    "dated",
    "signed",
    "executed",
    "effective",
    "execution",
    "expiry",
    "expiration",
    "terminate",
    "termination",
}
FACT_AMOUNT_HINTS = {
    "amount",
    "value",
    "price",
    "pricing",
    "fee",
    "fees",
    "payment",
    "payments",
    "total",
    "consideration",
    "contract value",
}
FACT_PARTY_HINTS = {
    "party",
    "parties",
    "counterparty",
    "counterparties",
    "customer",
    "customers",
    "client",
    "clients",
    "vendor",
    "vendors",
    "supplier",
    "suppliers",
    "buyer",
    "buyers",
    "seller",
    "sellers",
    "recipient",
    "recipients",
}
ENTITY_SUFFIX_TOKENS = {
    "bank",
    "bhd",
    "co",
    "company",
    "corp",
    "corporation",
    "group",
    "holdings",
    "inc",
    "llc",
    "llp",
    "limited",
    "ltd",
    "plc",
    "pte",
    "services",
    "solutions",
    "systems",
    "technologies",
    "technology",
    "labs",
}
QUERY_CURRENCY_CODE_MAP = {
    "$": "USD",
    "usd": "USD",
    "sgd": "SGD",
    "eur": "EUR",
    "gbp": "GBP",
    "inr": "INR",
    "idr": "IDR",
    "myr": "MYR",
    "rm": "MYR",
    "aed": "AED",
    "aud": "AUD",
    "cad": "CAD",
    "chf": "CHF",
    "cny": "CNY",
    "hkd": "HKD",
    "jpy": "JPY",
}
EVIDENCE_SPAN_SOURCE_TYPES = {"text", "ocr", "table"}
EVIDENCE_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
EVIDENCE_CLAUSE_SPLIT_RE = re.compile(r"\s*(?:[;:]\s+|\|\s+)\s*")
WEAK_EVIDENCE_PREFIX = "I have weak evidence, my best guess is:"


class ChatService:
    def __init__(
        self,
        retrieval_service,
        model,
        enable_vlm: bool,
        router=None,
        metadata_semantic_adapter=None,
        max_context_chars: int = 12000,
    ) -> None:
        self.retrieval = retrieval_service
        self.model = model
        self.enable_vlm = enable_vlm
        self.router = router
        self.metadata_semantic_adapter = metadata_semantic_adapter or MetadataSemanticAdapter(
            retrieval_service,
            expansion_model=model,
        )
        self.max_context_chars = max_context_chars
        self.last_generation_error: str | None = None
        self.last_route: dict[str, Any] | None = None
        self._doc_tag_cache: dict[str, list[str]] = {}
        self._tag_embedding_cache: dict[str, np.ndarray] = {}

    def _log_preview(self, value: Any, limit: int = 180) -> str:
        text = re.sub(r"\s+", " ", str(value or "").strip())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)] + "..."

    def _log_json_event(self, event: str, payload: dict[str, Any]) -> None:
        try:
            logger.info("%s %s", event, json.dumps(payload, sort_keys=True, default=str))
        except Exception:
            logger.info("%s %s", event, str(payload))

    def _chunk_log_summary(self, chunks: List[dict]) -> dict[str, Any]:
        source_counts: Counter[str] = Counter()
        doc_ids: list[str] = []
        seen_doc_ids: set[str] = set()
        top_chunks: list[dict[str, Any]] = []
        for chunk in chunks[:5]:
            top_chunks.append(
                {
                    "doc_id": str(chunk.get("doc_id") or ""),
                    "doc_filename": self._log_preview(chunk.get("doc_filename"), limit=120),
                    "page": chunk.get("page"),
                    "source_type": str(chunk.get("source_type") or ""),
                    "score": round(float(chunk.get("rerank_score", chunk.get("score", 0.0)) or 0.0), 4),
                }
            )
        for chunk in chunks:
            source_type = str(chunk.get("source_type") or "unknown").strip().lower() or "unknown"
            source_counts[source_type] += 1
            doc_id = str(chunk.get("doc_id") or "").strip()
            if doc_id and doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                doc_ids.append(doc_id)
        return {
            "chunk_count": len(chunks),
            "doc_count": len(doc_ids),
            "doc_ids": doc_ids[:8],
            "source_type_counts": dict(sorted(source_counts.items())),
            "top_chunks": top_chunks,
        }

    def answer(
        self,
        question: str,
        doc_ids: Optional[List[str]] = None,
        top_k: int = 5,
        include_document_summaries: bool = True,
        conversation_id: Optional[str] = None,
    ) -> dict:
        original_question = str(question or "").strip()
        retrieval_question = original_question
        session_id = self._normalize_conversation_id(conversation_id)
        conversation_state: dict[str, Any] = {}
        if session_id and bool(config.CHAT_MEMORY_ENABLED):
            conversation_state = self._load_conversation_state(session_id)
            retrieval_question = self._rewrite_question_with_conversation(original_question, conversation_state)

        route = self._route_question(retrieval_question, doc_ids, default_top_k=top_k)
        route = self._repair_metadata_route_if_needed(retrieval_question, route, doc_ids=doc_ids)
        route = self._repair_named_document_summary_route_if_needed(retrieval_question, route, doc_ids=doc_ids)
        self.last_route = route
        intent = str(route.get("task_type", "qa")).strip().lower()
        if intent not in {"compare", "qa", "trend_analysis", "count", "metadata_query"}:
            intent = "qa"
        metadata_signal = self._metadata_query_signal(route)
        if metadata_signal.get("is_metadata_query", False):
            intent = str(metadata_signal.get("intent") or intent or "metadata_query")
            route["task_type"] = intent
            route["metadata_query"] = metadata_signal
            self._log_json_event(
                "chat.route",
                {
                    "conversation_id": session_id or "",
                    "question_preview": self._log_preview(original_question),
                    "resolved_question_preview": self._log_preview(retrieval_question),
                    "selected_doc_count": len(doc_ids or []),
                    "selected_doc_ids": list(doc_ids or [])[:8],
                    "task_type": intent,
                    "needs_cross_doc": bool(route.get("needs_cross_doc", False)),
                    "needs_numeric_extraction": bool(route.get("needs_numeric_extraction", False)),
                    "needs_image_reasoning": bool(route.get("needs_image_reasoning", False)),
                    "retrieval_plan": route.get("retrieval_plan") if isinstance(route.get("retrieval_plan"), dict) else {},
                    "route_source": str(route.get("source") or ""),
                    "route_confidence": route.get("confidence"),
                    "expected_answer_type": str(route.get("expected_answer_type") or ""),
                    "analysis_plan": route.get("analysis_plan") if isinstance(route.get("analysis_plan"), dict) else {},
                    "metadata_signal": metadata_signal,
                },
            )
            answer, sources, query_intent = self._answer_metadata_or_hybrid_query(
                retrieval_question,
                doc_ids=doc_ids,
                metadata_signal=metadata_signal,
            )
            route["query_intent"] = query_intent
            response = {
                "answer": answer,
                "sources": sources,
                "intent": intent,
                "route": route,
                "include_document_summaries": False,
            }
            if session_id:
                response["conversation_id"] = session_id
                self._persist_conversation_turn(
                    session_id,
                    question=original_question,
                    answer=answer,
                    route=route,
                    intent=intent,
                    resolved_question=retrieval_question,
                )
            self._log_json_event(
                "chat.answer",
                {
                    "conversation_id": session_id or "",
                    "intent": intent,
                    "task_type": str(route.get("task_type") or intent),
                    "needs_cross_doc": bool(route.get("needs_cross_doc", False)),
                    "needs_numeric_extraction": bool(route.get("needs_numeric_extraction", False)),
                    "needs_image_reasoning": bool(route.get("needs_image_reasoning", False)),
                    "retrieval_plan": route.get("retrieval_plan") if isinstance(route.get("retrieval_plan"), dict) else {},
                    "query_intent": query_intent,
                    "answer_chars": len(str(answer or "")),
                    "source_count": len(sources),
                    "source_doc_count": len({str(source.get("doc_id") or "").strip() for source in sources if str(source.get("doc_id") or "").strip()}),
                    "source_type_counts": dict(sorted(Counter(str(source.get("source_type") or "unknown") for source in sources).items())),
                    "generation_error": bool(self.last_generation_error),
                },
            )
            return response
        prefer_exhaustive_list = self._is_exhaustive_list_query(
            retrieval_question,
            route=route,
            intent=intent,
            needs_cross_doc=bool(route.get("needs_cross_doc", False)),
        )
        needs_cross_doc = bool(route.get("needs_cross_doc", False))
        needs_numeric_extraction = bool(route.get("needs_numeric_extraction", False))
        if intent == "trend_analysis":
            needs_cross_doc = True
            needs_numeric_extraction = True
        image_intent = bool(route.get("needs_image_reasoning", False)) or route.get("task_type") == "image_qa"
        relationship_intent = self._is_relationship_intent(retrieval_question)
        diagram_intent = image_intent or relationship_intent
        retrieval_mode, route_top_k, per_doc_limit = self._normalize_retrieval_plan(
            route,
            default_top_k=top_k,
            needs_cross_doc=needs_cross_doc,
            needs_numeric_extraction=needs_numeric_extraction,
            diagram_intent=diagram_intent,
        )
        analysis_plan: dict[str, Any] = {}
        if intent == "trend_analysis":
            analysis_plan = self._build_analysis_plan(
                retrieval_question,
                doc_ids=doc_ids,
                router_analysis_plan=route.get("analysis_plan"),
                require_multi_doc=True,
            )
            route["analysis_plan"] = analysis_plan

        exact_lookup_chunks: List[dict] | None = None
        qa_doc_scope: Optional[List[str]] = None
        if intent == "qa" and not needs_cross_doc:
            scoped_docs = self._documents_in_scope(doc_ids)
            resolved_qa_scope = self._resolve_route_target_doc_ids(route, scoped_docs, min_resolved=1)
            if len(resolved_qa_scope) == 1:
                qa_doc_scope = resolved_qa_scope
        compare_doc_scope: Optional[List[str]] = None
        if intent == "compare":
            scoped_docs = self._documents_in_scope(doc_ids)
            compare_doc_scope = self._resolve_route_target_doc_ids(
                route,
                scoped_docs,
                min_resolved=2,
            )
            if not compare_doc_scope:
                compare_scope = self._select_candidate_docs_for_query(
                    retrieval_question,
                    doc_ids=doc_ids,
                    require_multi_doc=True,
                )
                compare_doc_scope = compare_scope["doc_ids"] if compare_scope else doc_ids

        exact_lookup_hint = self._route_exact_lookup_hint(route)
        self._log_json_event(
            "chat.route",
            {
                "conversation_id": session_id or "",
                "question_preview": self._log_preview(original_question),
                "resolved_question_preview": self._log_preview(retrieval_question),
                "selected_doc_count": len(doc_ids or []),
                "selected_doc_ids": list(doc_ids or [])[:8],
                "task_type": intent,
                "needs_cross_doc": needs_cross_doc,
                "needs_numeric_extraction": needs_numeric_extraction,
                "needs_image_reasoning": bool(route.get("needs_image_reasoning", False)),
                "relationship_intent": relationship_intent,
                "diagram_intent": diagram_intent,
                "retrieval_mode": retrieval_mode,
                "retrieval_plan": route.get("retrieval_plan") if isinstance(route.get("retrieval_plan"), dict) else {},
                "route_top_k": route_top_k,
                "per_doc_limit": per_doc_limit,
                "prefer_exhaustive_list": prefer_exhaustive_list,
                "route_source": str(route.get("source") or ""),
                "route_confidence": route.get("confidence"),
                "expected_answer_type": str(route.get("expected_answer_type") or ""),
                "analysis_plan": route.get("analysis_plan") if isinstance(route.get("analysis_plan"), dict) else {},
                "qa_doc_scope": list(qa_doc_scope or [])[:8],
                "compare_doc_scope": list(compare_doc_scope or [])[:8],
                "exact_lookup_hint": exact_lookup_hint,
            },
        )
        if intent in {"qa", "compare"}:
            exact_lookup_chunks, exact_lookup = self._retrieve_for_exact_lookup(
                retrieval_question,
                doc_ids=compare_doc_scope if intent == "compare" else (qa_doc_scope or doc_ids),
                top_k=route_top_k,
                per_doc_limit=per_doc_limit,
                mode=retrieval_mode,
                needs_cross_doc=needs_cross_doc,
                prefer_exhaustive=prefer_exhaustive_list,
                allow_compare=intent == "compare",
                include_fallback=intent == "qa",
                fact_types_hint=exact_lookup_hint.get("fact_types"),
                prefer_fact_lookup=bool(exact_lookup_hint.get("requested")),
            )
            route["exact_lookup"] = exact_lookup
        else:
            route["exact_lookup"] = {
                "applicable": False,
                "used_fact_lookup": False,
                "used_hybrid_fallback": False,
                "reason": f"intent_{intent}",
            }

        if intent == "qa" and exact_lookup_chunks is not None:
            chunks = exact_lookup_chunks
        elif intent == "compare":
            compare_chunks = self._retrieve_for_comparison(
                retrieval_question,
                doc_ids=compare_doc_scope or doc_ids,
                top_k=route_top_k,
                per_doc_limit=per_doc_limit,
                mode=retrieval_mode,
            )
            if exact_lookup_chunks:
                chunks = self._merge_chunks(
                    list(exact_lookup_chunks) + compare_chunks,
                    limit=max(16, route_top_k * 4),
                )
            else:
                chunks = compare_chunks
        elif intent == "trend_analysis":
            chunks, coverage = self._retrieve_for_trend_analysis(
                retrieval_question,
                doc_ids=doc_ids,
                top_k=route_top_k,
                per_doc_limit=per_doc_limit,
                mode=retrieval_mode,
                analysis_plan=analysis_plan,
                needs_numeric_extraction=needs_numeric_extraction,
            )
            if isinstance(route.get("analysis_plan"), dict):
                route["analysis_plan"]["coverage"] = coverage
        elif needs_cross_doc:
            chunks = self._retrieve_for_cross_doc_qa(
                retrieval_question,
                doc_ids=doc_ids,
                top_k=route_top_k,
                per_doc_limit=per_doc_limit,
                mode=retrieval_mode,
                prefer_exhaustive=prefer_exhaustive_list,
            )
        else:
            chunks = self._retrieve_for_single_doc_qa(
                retrieval_question,
                doc_ids=qa_doc_scope or doc_ids,
                top_k=route_top_k,
                mode=retrieval_mode,
                prefer_exhaustive=prefer_exhaustive_list,
            )
        if image_intent:
            chunks = self._augment_for_image_queries(retrieval_question, chunks, doc_ids=doc_ids, mode=retrieval_mode)
        if relationship_intent:
            chunks = self._augment_for_relationship_queries(
                retrieval_question,
                chunks,
                doc_ids=doc_ids,
                mode=retrieval_mode,
            )

        if diagram_intent:
            chunks = self._ensure_diagram_evidence_mix(
                retrieval_question,
                chunks,
                doc_ids=doc_ids,
                mode=retrieval_mode,
                target_k=route_top_k,
            )

        chunks = self._dedupe_redundant_chunks(chunks, limit=max(24, route_top_k * 4))
        if diagram_intent:
            chunks = self._prioritize_diagram_chunk_order(chunks)
        chunks = self._select_evidence_spans(retrieval_question, chunks, intent=intent)
        context_blocks = self._build_context_blocks(chunks)
        
        image_paths: List[str] = []
        seen = set()
        for chunk in chunks:
            metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
            image_path = str(metadata.get("image_path") or "").strip()
            if image_path and image_path not in seen:
                seen.add(image_path)
                image_paths.append(image_path)
        
        b64_images = []
        for path in image_paths[:5]:  # Limit to 5 images per request
            import base64
            try:
                with open(path, "rb") as f:
                    b64_images.append(base64.b64encode(f.read()).decode("utf-8"))
            except Exception as e:
                logger.warning(f"Could not load image {path}: {e}")

        prompt_question = retrieval_question or original_question
        if intent in {"compare", "trend_analysis"}:
            prompt = build_compare_prompt(
                prompt_question,
                context_blocks,
                self._document_briefs(chunks),
                include_document_summaries=include_document_summaries,
            )
        else:
            prompt = build_prompt(prompt_question, context_blocks)
            
        self.last_generation_error = None

        if self.enable_vlm and self.model is not None:
            try:
                answer = self.model.generate_text(prompt, max_new_tokens=1500, images=b64_images)
            except Exception as exc:
                logger.exception("Model text generation failed: %s", exc)
                self.last_generation_error = traceback.format_exc(limit=12)
                answer = self._fallback_answer(
                    original_question,
                    chunks,
                    intent=intent,
                    include_document_summaries=include_document_summaries,
                )
        else:
            answer = self._fallback_answer(
                original_question,
                chunks,
                intent=intent,
                include_document_summaries=include_document_summaries,
            )

        if self._looks_like_hard_no_answer(answer) and self._has_relevant_evidence(retrieval_question, chunks):
            answer = self._fallback_answer(
                original_question,
                chunks,
                intent=intent,
                include_document_summaries=include_document_summaries,
            )

        if intent == "compare" and not include_document_summaries:
            answer = self._strip_document_summaries(answer)

        response = {
            "answer": answer,
            "sources": self._prepare_response_sources(chunks, intent=intent),
            "intent": intent,
            "route": route,
            "include_document_summaries": include_document_summaries,
        }
        if session_id:
            response["conversation_id"] = session_id
            self._persist_conversation_turn(
                session_id,
                question=original_question,
                answer=answer,
                route=route,
                intent=intent,
                resolved_question=retrieval_question,
            )
            if retrieval_question and retrieval_question != original_question:
                response["resolved_question"] = retrieval_question
        if self.last_generation_error:
            response["generation_error"] = self.last_generation_error
        self._log_json_event(
            "chat.answer",
            {
                "conversation_id": session_id or "",
                "intent": intent,
                "task_type": str(route.get("task_type") or intent),
                "needs_cross_doc": needs_cross_doc,
                "needs_numeric_extraction": needs_numeric_extraction,
                "needs_image_reasoning": bool(route.get("needs_image_reasoning", False)),
                "retrieval_mode": retrieval_mode,
                "retrieval_plan": route.get("retrieval_plan") if isinstance(route.get("retrieval_plan"), dict) else {},
                "include_document_summaries": bool(include_document_summaries),
                "exact_lookup": route.get("exact_lookup") if isinstance(route.get("exact_lookup"), dict) else {},
                "chunk_summary": self._chunk_log_summary(chunks),
                "context_block_count": len(context_blocks),
                "image_path_count": len(image_paths),
                "answer_chars": len(str(answer or "")),
                "source_count": len(response["sources"]),
                "source_doc_count": len({str(source.get("doc_id") or "").strip() for source in response["sources"] if str(source.get("doc_id") or "").strip()}),
                "source_type_counts": dict(sorted(Counter(str(source.get("source_type") or "unknown") for source in response["sources"]).items())),
                "generation_error": bool(self.last_generation_error),
            },
        )
        return response

    def _normalize_conversation_id(self, value: Optional[str]) -> str | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        normalized = re.sub(r"[^A-Za-z0-9_\-:.]", "", raw)
        if not normalized:
            return None
        return normalized[:96]

    def _load_conversation_state(self, session_id: str) -> dict[str, Any]:
        state: dict[str, Any] = {
            "session_id": session_id,
            "summary": "",
            "messages": [],
        }
        try:
            session = storage.get_chat_session(session_id)
            if session is None:
                session = storage.upsert_chat_session(session_id, summary="")
            state["summary"] = str(session.get("summary") or "").strip()
            state["messages"] = storage.list_chat_messages(
                session_id,
                limit=max(2, int(config.CHAT_MEMORY_MAX_MESSAGES)),
                ascending=True,
            )
        except Exception as exc:
            logger.warning("Failed to load conversation state for %s: %s", session_id, exc)
        return state

    def _is_followup_question(self, question: str) -> bool:
        text = str(question or "").strip().lower()
        if not text:
            return False
        tokens = re.findall(r"[a-z0-9]+", text)
        if not tokens:
            return False
        pronoun_markers = {
            "it",
            "its",
            "they",
            "them",
            "their",
            "those",
            "these",
            "that",
            "this",
            "he",
            "she",
            "his",
            "her",
            "former",
            "latter",
            "same",
        }
        bridge_prefixes = (
            "and ",
            "also ",
            "what about",
            "how about",
            "why ",
            "when ",
            "where ",
            "who ",
            "which ",
            "based on",
            "on what basis",
        )
        if any(marker in tokens for marker in pronoun_markers):
            return True
        if any(text.startswith(prefix) for prefix in bridge_prefixes):
            return True
        return len(tokens) <= 8

    def _compact_text(self, value: Any, max_chars: int) -> str:
        cleaned = re.sub(r"\s+", " ", str(value or "").strip())
        if max_chars > 0 and len(cleaned) > max_chars:
            return cleaned[:max_chars].rstrip()
        return cleaned

    def _conversation_transcript(self, messages: list[dict], max_chars: int) -> str:
        if not messages:
            return ""
        max_turns = max(2, int(config.CHAT_MEMORY_RECENT_TURNS) * 2)
        lines: list[str] = []
        budget = max(200, int(max_chars))
        for msg in messages[-max_turns:]:
            role = str(msg.get("role") or "user").strip().lower()
            role_label = "assistant" if role == "assistant" else "user"
            content = self._compact_text(msg.get("content"), 360)
            if not content:
                continue
            line = f"{role_label}: {content}"
            if lines and sum(len(item) for item in lines) + len(line) > budget:
                break
            lines.append(line)
        return "\n".join(lines)

    def _parse_json_object(self, raw: str) -> dict[str, Any] | None:
        text = str(raw or "").strip()
        if not text:
            return None
        candidate = text
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            candidate = match.group(0)
        try:
            payload = json.loads(candidate)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _heuristic_followup_rewrite(self, question: str, messages: list[dict]) -> str:
        if not self._is_followup_question(question):
            return question
        prior_user_questions = [
            self._compact_text(msg.get("content"), 260)
            for msg in messages
            if str(msg.get("role") or "").strip().lower() == "user"
        ]
        prior_user_questions = [item for item in prior_user_questions if item]
        if not prior_user_questions:
            return question
        last_question = prior_user_questions[-1]
        if last_question.lower() == str(question or "").strip().lower():
            if len(prior_user_questions) >= 2:
                last_question = prior_user_questions[-2]
            else:
                return question
        return f"{question}. Context: {last_question}"

    def _rewrite_question_with_conversation(self, question: str, state: dict[str, Any]) -> str:
        text = str(question or "").strip()
        if not text:
            return text
        messages = state.get("messages") if isinstance(state.get("messages"), list) else []
        summary = str(state.get("summary") or "").strip()
        if not messages:
            return text
        if not self._is_followup_question(text):
            return text

        transcript = self._conversation_transcript(messages, int(config.CHAT_MEMORY_REWRITE_MAX_CHARS))
        if not transcript and not summary:
            return text
        if not (self.enable_vlm and self.model is not None):
            return self._heuristic_followup_rewrite(text, messages)

        prompt = (
            "You rewrite follow-up questions for retrieval.\n"
            "Return ONLY JSON with keys: standalone_question, confidence.\n"
            "Rules:\n"
            "- Keep intent unchanged.\n"
            "- Resolve pronouns and omitted entities from conversation context.\n"
            "- Keep the standalone question concise and specific.\n"
            "- If already standalone, return it unchanged.\n\n"
            f"Conversation summary:\n{summary or '(none)'}\n\n"
            f"Recent conversation turns:\n{transcript or '(none)'}\n\n"
            f"Current user question: {text}\n"
            "JSON:"
        )
        try:
            raw = self.model.generate_text(prompt, max_new_tokens=220)
            parsed = self._parse_json_object(raw)
            candidate = ""
            if parsed:
                candidate = self._compact_text(parsed.get("standalone_question"), 420)
            if candidate:
                return candidate
        except Exception as exc:
            logger.warning("Conversation rewrite failed; using heuristic rewrite. %s", exc)
        return self._heuristic_followup_rewrite(text, messages)

    def _fallback_summary_update(self, previous_summary: str, question: str, answer: str) -> str:
        target = max(240, int(config.CHAT_MEMORY_SUMMARY_TARGET_CHARS))
        cap = max(target, int(config.CHAT_MEMORY_SUMMARY_MAX_CHARS))
        prior = self._compact_text(previous_summary, cap)
        q = self._compact_text(question, 240)
        a = self._compact_text(self._strip_document_summaries(answer), 360)
        update = f"Q: {q} A: {a}"
        merged = f"{prior} {update}".strip() if prior else update
        if len(merged) <= cap:
            return merged
        return merged[-cap:].lstrip()

    def _update_conversation_summary(
        self,
        previous_summary: str,
        question: str,
        answer: str,
    ) -> str:
        prior = self._compact_text(previous_summary, int(config.CHAT_MEMORY_SUMMARY_MAX_CHARS))
        if not (self.enable_vlm and self.model is not None):
            return self._fallback_summary_update(prior, question, answer)

        target = max(240, int(config.CHAT_MEMORY_SUMMARY_TARGET_CHARS))
        cap = max(target, int(config.CHAT_MEMORY_SUMMARY_MAX_CHARS))
        prompt = (
            "You maintain compact conversation memory for a document QA assistant.\n"
            "Return plain text only.\n"
            "Focus on user intent, named entities, constraints, open references, and unresolved follow-ups.\n"
            "Exclude low-value filler and avoid repeating full answers.\n"
            f"Target length: <= {target} characters.\n\n"
            f"Previous memory summary:\n{prior or '(none)'}\n\n"
            f"Latest user question:\n{self._compact_text(question, 300)}\n\n"
            f"Latest assistant answer:\n{self._compact_text(self._strip_document_summaries(answer), 520)}\n\n"
            "Updated memory summary:"
        )
        try:
            summary = self._compact_text(self.model.generate_text(prompt, max_new_tokens=260), cap)
            if summary:
                return summary
        except Exception as exc:
            logger.warning("Conversation summary update failed; using fallback summary. %s", exc)
        return self._fallback_summary_update(prior, question, answer)

    def _persist_conversation_turn(
        self,
        session_id: str,
        *,
        question: str,
        answer: str,
        route: dict[str, Any],
        intent: str,
        resolved_question: str,
    ) -> None:
        if not session_id or not bool(config.CHAT_MEMORY_ENABLED):
            return
        try:
            session = storage.upsert_chat_session(session_id, summary="")
            now_iso = datetime.now(timezone.utc).isoformat()
            question_text = self._compact_text(question, 2000)
            answer_text = self._compact_text(answer, 12000)
            if question_text:
                storage.add_chat_message(
                    message_id=str(uuid.uuid4()),
                    session_id=session_id,
                    role="user",
                    content=question_text,
                    created_at=now_iso,
                    metadata={
                        "intent": intent,
                        "route_task_type": str(route.get("task_type") or "").strip().lower(),
                        "resolved_question": resolved_question if resolved_question != question_text else "",
                    },
                )
            if answer_text:
                storage.add_chat_message(
                    message_id=str(uuid.uuid4()),
                    session_id=session_id,
                    role="assistant",
                    content=answer_text,
                    created_at=now_iso,
                    metadata={
                        "intent": intent,
                        "route_task_type": str(route.get("task_type") or "").strip().lower(),
                    },
                )
            previous_summary = str(session.get("summary") or "").strip()
            updated_summary = self._update_conversation_summary(previous_summary, question_text, answer_text)
            storage.upsert_chat_session(
                session_id,
                summary=updated_summary,
                metadata={
                    "last_intent": intent,
                    "last_route_task_type": str(route.get("task_type") or "").strip().lower(),
                    "last_updated_at": now_iso,
                },
                merge_metadata=True,
            )
        except Exception as exc:
            logger.warning("Failed to persist conversation turn for %s: %s", session_id, exc)

    def _build_context_blocks(self, chunks: List[dict]) -> List[str]:
        blocks: List[str] = []
        total = 0
        for chunk in chunks:
            content = str(chunk.get("content", "")).strip()
            if not content:
                continue
            block = f"{self._build_source_tag(chunk)} {content}"
            if blocks and total + len(block) > self.max_context_chars:
                break
            blocks.append(block)
            total += len(block)
        return blocks

    def _select_evidence_spans(
        self,
        question: str,
        chunks: List[dict],
        *,
        intent: str,
    ) -> List[dict]:
        if not chunks:
            return []
        entities = self._extract_query_entities(question)
        selected: List[dict] = []
        for chunk in chunks:
            selected.append(
                self._select_evidence_span_for_chunk(
                    question,
                    chunk,
                    entities=entities,
                    intent=intent,
                )
            )
        return selected

    def _candidate_evidence_spans(self, content: str) -> list[dict[str, Any]]:
        normalized = re.sub(r"\s+", " ", str(content or "")).strip()
        if not normalized:
            return []

        units = [
            part.strip(" \t\r\n-")
            for part in EVIDENCE_SENTENCE_SPLIT_RE.split(normalized)
            if part and part.strip(" \t\r\n-")
        ]
        if not units:
            units = [normalized]

        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add_candidate(text: str, kind: str) -> None:
            cleaned = re.sub(r"\s+", " ", str(text or "")).strip(" \t\r\n-")
            if len(cleaned) < 18:
                return
            key = cleaned.lower()
            if key in seen:
                return
            seen.add(key)
            candidates.append({"text": cleaned, "kind": kind})

        for unit in units:
            add_candidate(unit, "sentence")
            clauses = [
                clause.strip(" \t\r\n-")
                for clause in EVIDENCE_CLAUSE_SPLIT_RE.split(unit)
                if clause and clause.strip(" \t\r\n-")
            ]
            if len(clauses) > 1:
                for clause in clauses:
                    add_candidate(clause, "clause")

        for idx in range(len(units) - 1):
            add_candidate(f"{units[idx]} {units[idx + 1]}", "window")

        add_candidate(normalized, "full")
        return candidates[:24]

    def _score_evidence_span(
        self,
        question: str,
        *,
        span_text: str,
        span_kind: str,
        chunk_text: str,
        chunk_score: float,
        entities: list[str],
    ) -> float:
        span_overlap = self._chunk_query_overlap(question, span_text, entities=entities)
        chunk_overlap = self._chunk_query_overlap(question, chunk_text, entities=entities)
        query_terms = self._tokenize_for_match(question)
        span_terms = self._tokenize_for_match(span_text)
        exact_match_bonus = 0.0
        compact_span = self._compact_lookup_value(span_text)
        for entity in entities[:8]:
            compact_entity = self._compact_lookup_value(entity)
            if compact_entity and compact_entity in compact_span:
                exact_match_bonus = max(exact_match_bonus, 0.12)
                break
        term_hit_ratio = 0.0
        if query_terms and span_terms:
            term_hit_ratio = len(query_terms.intersection(span_terms)) / float(max(1, len(span_terms)))

        kind_penalty = {
            "clause": 0.0,
            "sentence": 0.01,
            "window": 0.03,
            "full": 0.08,
        }.get(span_kind, 0.04)
        length_penalty = min(0.12, max(0, len(span_text) - 190) / 900.0)

        return max(
            0.0,
            (span_overlap * 0.64)
            + (chunk_overlap * 0.14)
            + (term_hit_ratio * 0.12)
            + min(0.10, max(0.0, chunk_score) * 0.10)
            + exact_match_bonus
            - kind_penalty
            - length_penalty,
        )

    def _select_evidence_span_for_chunk(
        self,
        question: str,
        chunk: dict,
        *,
        entities: list[str],
        intent: str,
    ) -> dict:
        content = re.sub(r"\s+", " ", str(chunk.get("content", ""))).strip()
        if not content:
            return chunk
        source_type = str(chunk.get("source_type") or "").strip().lower()
        if source_type not in EVIDENCE_SPAN_SOURCE_TYPES:
            return chunk
        if intent in {"compare", "trend_analysis"} and len(content) <= 220:
            return chunk
        if len(content) <= 180:
            return chunk

        candidates = self._candidate_evidence_spans(content)
        if len(candidates) <= 1:
            return chunk

        chunk_score = float(chunk.get("rerank_score", chunk.get("score", 0.0)) or 0.0)
        best = max(
            candidates,
            key=lambda candidate: self._score_evidence_span(
                question,
                span_text=str(candidate.get("text") or ""),
                span_kind=str(candidate.get("kind") or "full"),
                chunk_text=content,
                chunk_score=chunk_score,
                entities=entities,
            ),
        )
        best_text = str(best.get("text") or "").strip()
        best_kind = str(best.get("kind") or "full").strip().lower()
        if not best_text or best_kind == "full" or best_text == content:
            return chunk

        updated = dict(chunk)
        updated["content"] = best_text
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        updated["metadata"] = {
            **metadata,
            "selected_span": True,
            "selected_span_type": best_kind,
            "selected_span_text": best_text,
        }
        return updated

    def _normalize_retrieval_plan(
        self,
        route: dict[str, Any],
        *,
        default_top_k: int,
        needs_cross_doc: bool,
        needs_numeric_extraction: bool,
        diagram_intent: bool,
    ) -> tuple[str, int, int]:
        retrieval_plan = route.get("retrieval_plan") if isinstance(route.get("retrieval_plan"), dict) else {}
        try:
            route_top_k = int(retrieval_plan.get("top_k", max(default_top_k, 8)))
        except Exception:
            route_top_k = max(default_top_k, 8)
        try:
            per_doc_limit = int(retrieval_plan.get("per_doc_limit", 4))
        except Exception:
            per_doc_limit = 4

        retrieval_mode = str(retrieval_plan.get("strategy", "semantic")).strip().lower() or "semantic"
        if retrieval_mode not in {"semantic", "balanced", "hybrid", "sparse", "image_first"}:
            retrieval_mode = "semantic"

        if needs_cross_doc:
            route_top_k = max(route_top_k, 10)
            per_doc_limit = max(per_doc_limit, 2)
            if retrieval_mode == "semantic":
                retrieval_mode = "balanced"

        if needs_numeric_extraction:
            route_top_k = max(route_top_k, 12)
            per_doc_limit = max(per_doc_limit, 2)
            if retrieval_mode == "semantic":
                retrieval_mode = "balanced"

        if diagram_intent:
            route_top_k = max(route_top_k, int(config.DIAGRAM_TOP_K_FLOOR))
            per_doc_limit = max(per_doc_limit, int(config.DIAGRAM_PER_DOC_LIMIT_FLOOR))

        if isinstance(route.get("retrieval_plan"), dict):
            route["retrieval_plan"]["strategy"] = retrieval_mode
            route["retrieval_plan"]["top_k"] = route_top_k
            route["retrieval_plan"]["per_doc_limit"] = per_doc_limit

        return retrieval_mode, route_top_k, per_doc_limit

    def _retrieve_for_single_doc_qa(
        self,
        question: str,
        *,
        doc_ids: Optional[List[str]],
        top_k: int,
        mode: str,
        prefer_exhaustive: bool = False,
    ) -> List[dict]:
        scoped = self._select_candidate_docs_for_query(
            question,
            doc_ids=doc_ids,
            require_multi_doc=False,
            prefer_recall=prefer_exhaustive,
        )
        scoped_doc_ids = scoped["doc_ids"] if scoped else doc_ids
        chunks = self.retrieval.search(
            question,
            top_k=top_k,
            doc_ids=scoped_doc_ids,
            mode=mode,
        )
        if scoped and (
            not bool(scoped.get("confident"))
            or self._retrieval_confidence_low(question, chunks, min_docs=1)
        ):
            fallback_doc_ids = scoped["doc_ids"] if bool(scoped.get("confident")) else (doc_ids or [])
            global_chunks = self.retrieval.search(
                question,
                top_k=top_k,
                doc_ids=fallback_doc_ids or doc_ids,
                mode=mode,
            )
            chunks = self._merge_chunks(chunks + global_chunks, limit=max(12, top_k * 3))
        return chunks

    def _compact_lookup_value(self, value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())

    def _extract_query_date_targets(self, question: str) -> list[dict[str, str]]:
        text = str(question or "")
        targets: list[dict[str, str]] = []
        seen: set[str] = set()

        def add_target(raw_value: str, canonical_value: str) -> None:
            compact = self._compact_lookup_value(canonical_value)
            if not raw_value or not canonical_value or compact in seen:
                return
            seen.add(compact)
            targets.append(
                {
                    "kind": "date",
                    "raw": raw_value,
                    "canonical": canonical_value,
                    "compact": compact,
                }
            )

        for match in ISO_DATE_RE.finditer(text):
            year, month, day = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
            try:
                canonical = datetime(year, month, day).date().isoformat()
            except Exception:
                continue
            add_target(match.group(0), canonical)

        for match in DMY_MONTH_RE.finditer(text):
            month = MONTH_NAME_TO_NUM.get(str(match.group(2) or "").strip().lower())
            if month is None:
                continue
            try:
                canonical = datetime(int(match.group(3)), month, int(match.group(1))).date().isoformat()
            except Exception:
                continue
            add_target(match.group(0), canonical)

        for match in MONTH_DY_RE.finditer(text):
            month = MONTH_NAME_TO_NUM.get(str(match.group(1) or "").strip().lower())
            if month is None:
                continue
            try:
                canonical = datetime(int(match.group(3)), month, int(match.group(2))).date().isoformat()
            except Exception:
                continue
            add_target(match.group(0), canonical)

        return targets

    def _extract_query_amount_targets(self, question: str) -> list[dict[str, str]]:
        text = str(question or "")
        targets: list[dict[str, str]] = []
        seen: set[str] = set()

        def add_target(raw_value: str, canonical_value: str) -> None:
            compact = self._compact_lookup_value(canonical_value or raw_value)
            if not raw_value or compact in seen:
                return
            seen.add(compact)
            targets.append(
                {
                    "kind": "amount",
                    "raw": raw_value,
                    "canonical": canonical_value or raw_value,
                    "compact": compact,
                }
            )

        for match in QUERY_AMOUNT_RE.finditer(text):
            raw_value = str(match.group(0) or "").strip()
            if not raw_value:
                continue
            currency_key = ""
            for raw_currency in (
                match.group("currency_before"),
                match.group("currency_after"),
                match.group("symbol"),
            ):
                value = str(raw_currency or "").strip().lower()
                if value:
                    currency_key = value
                    break
            currency = QUERY_CURRENCY_CODE_MAP.get(currency_key, "")
            number_text = str(match.group("number") or "").replace(",", "")
            decimal_text = str(match.group("decimal") or "")
            magnitude = str(match.group("magnitude") or "").strip().lower()
            multiplier = Decimal("1")
            if magnitude in {"k", "thousand"}:
                multiplier = Decimal("1000")
            elif magnitude in {"m", "million"}:
                multiplier = Decimal("1000000")
            elif magnitude in {"b", "billion"}:
                multiplier = Decimal("1000000000")
            try:
                amount = (Decimal(number_text + decimal_text) if decimal_text else Decimal(number_text)) * multiplier
            except (InvalidOperation, ValueError):
                add_target(raw_value, raw_value)
                continue
            canonical_number = format(amount.normalize(), "f")
            if "." in canonical_number:
                canonical_number = canonical_number.rstrip("0").rstrip(".")
            canonical_value = f"{currency} {canonical_number}".strip()
            add_target(raw_value, canonical_value)
        return targets

    def _extract_contains_target_text(self, question: str) -> str:
        text = str(question or "").strip()
        if not text:
            return ""
        pattern = re.compile(
            r"(?i)\b(?:contains?|mentions?|includes?|has|have|named)\b\s+(?P<body>[^?]+?)(?:\?|$)"
        )
        match = pattern.search(text)
        if match is None:
            return ""
        body = str(match.group("body") or "").strip(" \t\r\n\"'`.,:;!?()[]{}")
        if not body:
            return ""
        body = re.sub(
            r"(?i)^(?:the\s+)?(?:exact\s+)?(?:date|amount|value|party|parties|counterparty|counterparties)\s+",
            "",
            body,
        ).strip()
        return body

    def _looks_like_party_target(self, value: str) -> bool:
        text = str(value or "").strip()
        if not text:
            return False
        compact = self._compact_lookup_value(text)
        if len(compact) < 4:
            return False
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9&./'-]*", text)
        if not tokens:
            return False
        lowered_tokens = [token.lower().rstrip(".") for token in tokens]
        if any(token in ENTITY_SUFFIX_TOKENS for token in lowered_tokens):
            return True
        alpha_tokens = [token for token in tokens if re.search(r"[A-Za-z]", token)]
        if len(alpha_tokens) >= 2 and not any(token.isdigit() for token in alpha_tokens):
            return True
        return any(token[:1].isupper() for token in alpha_tokens if token)

    def _classify_exact_lookup_query(
        self,
        question: str,
        *,
        doc_ids: Optional[List[str]],
        allow_compare: bool = False,
        prefer_fact_lookup: bool = False,
        fact_types_hint: Optional[List[str]] = None,
    ) -> dict[str, Any]:
        text = str(question or "").strip()
        lowered = text.lower()
        if not text:
            return {
                "applicable": False,
                "reason": "empty_question",
            }
        if self._has_count_signal(question):
            return {
                "applicable": False,
                "reason": "count_query",
            }
        if not allow_compare and self._detect_intent(question, doc_ids) == "compare":
            return {
                "applicable": False,
                "reason": "compare_query",
            }

        wants_documents = any(marker in lowered for marker in EXACT_DOCUMENT_LOOKUP_MARKERS)
        contains_signal = any(marker in lowered for marker in EXACT_CONTAINS_MARKERS)
        date_targets = self._extract_query_date_targets(text)
        amount_targets = self._extract_query_amount_targets(text)
        target_values: list[dict[str, str]] = []
        target_values.extend(date_targets)
        target_values.extend(amount_targets)

        raw_target_text = self._extract_contains_target_text(text) if (wants_documents or contains_signal) else ""
        if raw_target_text and not date_targets and not amount_targets and self._looks_like_party_target(raw_target_text):
            target_values.append(
                {
                    "kind": "party",
                    "raw": raw_target_text,
                    "canonical": raw_target_text,
                    "compact": self._compact_lookup_value(raw_target_text),
                }
            )

        fact_types: list[str] = []
        if date_targets or any(hint in lowered for hint in FACT_DATE_HINTS):
            fact_types.append("date")
        if amount_targets or any(hint in lowered for hint in FACT_AMOUNT_HINTS):
            fact_types.append("amount")
        if any(item.get("kind") == "party" for item in target_values) or any(hint in lowered for hint in FACT_PARTY_HINTS):
            fact_types.append("party")
        for fact_type in fact_types_hint or []:
            cleaned = str(fact_type or "").strip().lower()
            if cleaned in {"date", "amount", "party"} and cleaned not in fact_types:
                fact_types.append(cleaned)

        if wants_documents and target_values:
            return {
                "applicable": True,
                "mode": "document_contains",
                "wants_documents": True,
                "fact_types": fact_types or sorted({str(item.get("kind") or "").strip() for item in target_values if item.get("kind")}),
                "targets": target_values,
            }

        direct_fact_markers = (
            "what is",
            "what are",
            "who is",
            "who are",
            "when is",
            "when was",
            "what date",
            "which date",
            "what amount",
            "what value",
            "how much",
        )
        broad_semantic_markers = (
            " where ",
            " based ",
            " location ",
            " country ",
            " countries ",
            " address ",
        )
        non_exact_markers = (
            "basis",
            "considered",
            "because",
            "reason",
            "why ",
            "how ",
        )
        single_doc_scope = len([str(doc_id).strip() for doc_id in (doc_ids or []) if str(doc_id).strip()]) == 1
        direct_fact_prompt = any(marker in lowered for marker in direct_fact_markers)
        is_non_exact_prompt = any(marker in lowered for marker in non_exact_markers)
        if fact_types and not is_non_exact_prompt and (
            target_values
            or direct_fact_prompt
            or prefer_fact_lookup
            or (allow_compare and bool(fact_types))
            or (single_doc_scope and not any(marker in lowered for marker in broad_semantic_markers))
        ):
            return {
                "applicable": True,
                "mode": "fact_lookup",
                "wants_documents": False,
                "fact_types": sorted(set(fact_types)),
                "targets": target_values,
            }

        return {
            "applicable": False,
            "reason": "no_exact_fact_signal",
        }

    def _score_exact_lookup_fact(
        self,
        question: str,
        *,
        fact: dict[str, Any],
        lookup: dict[str, Any],
    ) -> float:
        fact_type = str(fact.get("fact_type") or "").strip().lower()
        if lookup.get("fact_types") and fact_type not in set(lookup.get("fact_types") or []):
            return 0.0
        raw_value = str(fact.get("raw_value") or "").strip()
        canonical_value = str(fact.get("canonical_value") or "").strip()
        evidence_text = str(fact.get("evidence_text") or "").strip()
        compact_candidates = {
            self._compact_lookup_value(raw_value),
            self._compact_lookup_value(canonical_value),
            self._compact_lookup_value(evidence_text),
        }
        compact_candidates.discard("")
        target_compacts = {
            self._compact_lookup_value(str(target.get("canonical") or target.get("raw") or ""))
            for target in (lookup.get("targets") or [])
            if self._compact_lookup_value(str(target.get("canonical") or target.get("raw") or ""))
        }
        question_terms = self._tokenize_for_match(question)
        fact_terms = self._tokenize_for_match(" ".join([raw_value, canonical_value, evidence_text]))
        overlap = 0.0
        if question_terms and fact_terms:
            overlap = len(question_terms.intersection(fact_terms)) / float(max(1, len(question_terms)))

        score = 0.22 + min(0.26, overlap * 0.52)
        try:
            score += min(0.12, max(0.0, float(fact.get("confidence") or 0.0)) * 0.12)
        except Exception:
            score += 0.0

        if target_compacts:
            if any(target == candidate for target in target_compacts for candidate in compact_candidates):
                score += 0.48
            elif any(target and target in candidate for target in target_compacts for candidate in compact_candidates):
                score += 0.36
            else:
                return 0.0
        elif fact_type in set(lookup.get("fact_types") or []):
            score += 0.18

        return float(max(0.0, min(1.0, score)))

    def _search_exact_lookup_facts(
        self,
        question: str,
        *,
        doc_ids: Optional[List[str]],
        lookup: dict[str, Any],
        top_k: int,
        per_doc_limit: int,
    ) -> list[dict[str, Any]]:
        fact_types = [str(value).strip().lower() for value in (lookup.get("fact_types") or []) if str(value).strip()]
        try:
            facts = storage.list_document_facts(
                doc_ids=doc_ids,
                fact_types=fact_types or None,
            )
        except Exception as exc:
            logger.debug("Exact lookup skipped because document_facts is unavailable: %s", exc)
            return []
        matches: list[dict[str, Any]] = []
        for fact in facts:
            score = self._score_exact_lookup_fact(question, fact=fact, lookup=lookup)
            if score <= 0.0:
                continue
            matches.append({"fact": fact, "score": score})
        matches.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)

        if not bool(lookup.get("wants_documents")):
            deduped: list[dict[str, Any]] = []
            seen: set[tuple[str, str, str]] = set()
            for item in matches:
                fact = item["fact"]
                key = (
                    str(fact.get("doc_id") or "").strip(),
                    str(fact.get("fact_type") or "").strip().lower(),
                    self._compact_lookup_value(str(fact.get("canonical_value") or fact.get("raw_value") or "")),
                )
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(item)
                if len(deduped) >= max(top_k, per_doc_limit * 3):
                    break
            return deduped

        best_by_doc: dict[str, dict[str, Any]] = {}
        for item in matches:
            fact = item["fact"]
            doc_id = str(fact.get("doc_id") or "").strip()
            if not doc_id:
                continue
            previous = best_by_doc.get(doc_id)
            if previous is None or float(item.get("score") or 0.0) > float(previous.get("score") or 0.0):
                best_by_doc[doc_id] = item
        ranked = sorted(best_by_doc.values(), key=lambda item: float(item.get("score") or 0.0), reverse=True)
        return ranked[: max(top_k, per_doc_limit)]

    def _fact_match_to_chunk(
        self,
        match: dict[str, Any],
        *,
        doc_cache: dict[str, dict[str, Any]],
        wants_documents: bool,
    ) -> dict[str, Any]:
        fact = match["fact"]
        doc_id = str(fact.get("doc_id") or "").strip()
        doc = doc_cache.get(doc_id)
        if doc is None and doc_id:
            doc = storage.get_document(doc_id) or {}
            doc_cache[doc_id] = doc
        fact_type = str(fact.get("fact_type") or "").strip().lower() or "fact"
        raw_value = str(fact.get("raw_value") or "").strip()
        canonical_value = str(fact.get("canonical_value") or "").strip()
        value_text = raw_value or canonical_value
        if canonical_value and self._compact_lookup_value(canonical_value) != self._compact_lookup_value(raw_value):
            value_text = f"{value_text} (canonical: {canonical_value})"
        evidence_text = str(fact.get("evidence_text") or "").strip()
        prefix = "Matched" if wants_documents else "Extracted"
        return {
            "id": f"fact:{fact.get('id')}",
            "doc_id": doc_id,
            "doc_filename": str((doc or {}).get("filename") or ""),
            "doc_created_at": str((doc or {}).get("created_at") or ""),
            "page": fact.get("page"),
            "content": f"{prefix} {fact_type}: {value_text}. Evidence: {evidence_text}",
            "score": float(match.get("score") or 0.0),
            "source_type": "document_fact",
            "metadata": {
                "fact_id": str(fact.get("id") or ""),
                "fact_type": fact_type,
                "canonical_value": canonical_value,
                "raw_value": raw_value,
                "chunk_id": str(fact.get("chunk_id") or ""),
                "fact_confidence": float(fact.get("confidence") or 0.0),
            },
        }

    def _exact_lookup_fallback_chunks(
        self,
        question: str,
        *,
        doc_ids: Optional[List[str]],
        top_k: int,
        per_doc_limit: int,
        mode: str,
        needs_cross_doc: bool,
        prefer_exhaustive: bool,
        wants_documents: bool,
    ) -> list[dict]:
        if wants_documents or needs_cross_doc or len(doc_ids or []) > 1:
            return self._retrieve_for_cross_doc_qa(
                question,
                doc_ids=doc_ids,
                top_k=top_k,
                per_doc_limit=per_doc_limit,
                mode=mode,
                prefer_exhaustive=prefer_exhaustive,
            )
        return self._retrieve_for_single_doc_qa(
            question,
            doc_ids=doc_ids,
            top_k=top_k,
            mode=mode,
            prefer_exhaustive=prefer_exhaustive,
        )

    def _retrieve_for_exact_lookup(
        self,
        question: str,
        *,
        doc_ids: Optional[List[str]],
        top_k: int,
        per_doc_limit: int,
        mode: str,
        needs_cross_doc: bool,
        prefer_exhaustive: bool,
        allow_compare: bool = False,
        include_fallback: bool = True,
        fact_types_hint: Optional[List[str]] = None,
        prefer_fact_lookup: bool = False,
    ) -> tuple[list[dict] | None, dict[str, Any]]:
        lookup = self._classify_exact_lookup_query(
            question,
            doc_ids=doc_ids,
            allow_compare=allow_compare,
            prefer_fact_lookup=prefer_fact_lookup,
            fact_types_hint=fact_types_hint,
        )
        if not bool(lookup.get("applicable")):
            info = dict(lookup)
            info.setdefault("applicable", False)
            info["used_fact_lookup"] = False
            info["used_hybrid_fallback"] = False
            return None, info

        matches = self._search_exact_lookup_facts(
            question,
            doc_ids=doc_ids,
            lookup=lookup,
            top_k=top_k,
            per_doc_limit=per_doc_limit,
        )
        doc_cache: dict[str, dict[str, Any]] = {}
        fact_chunks = [
            self._fact_match_to_chunk(
                match,
                doc_cache=doc_cache,
                wants_documents=bool(lookup.get("wants_documents")),
            )
            for match in matches
        ]

        info = {
            "applicable": True,
            "mode": str(lookup.get("mode") or "fact_lookup"),
            "fact_types": list(lookup.get("fact_types") or []),
            "targets": [str(item.get("raw") or item.get("canonical") or "").strip() for item in (lookup.get("targets") or [])],
            "fact_match_count": len(matches),
            "used_fact_lookup": True,
            "used_hybrid_fallback": False,
            "reason": "",
        }
        matched_doc_ids = {
            str((match.get("fact") or {}).get("doc_id") or "").strip()
            for match in matches
            if str((match.get("fact") or {}).get("doc_id") or "").strip()
        }
        required_doc_count = 2 if allow_compare else 1
        info["matched_doc_count"] = len(matched_doc_ids)
        info["required_doc_count"] = required_doc_count
        if not matches:
            info["strength"] = "empty"
            info["reason"] = "no_fact_matches"
            if not include_fallback:
                return [], info
            info["used_hybrid_fallback"] = True
            return (
                self._exact_lookup_fallback_chunks(
                    question,
                    doc_ids=doc_ids,
                    top_k=top_k,
                    per_doc_limit=per_doc_limit,
                    mode=mode,
                    needs_cross_doc=needs_cross_doc,
                    prefer_exhaustive=prefer_exhaustive,
                    wants_documents=bool(lookup.get("wants_documents")),
                ),
                info,
            )

        best_score = max(float(match.get("score") or 0.0) for match in matches)
        if lookup.get("targets"):
            strong = best_score >= 0.72
        else:
            strong = best_score >= 0.42
        if allow_compare and len(matched_doc_ids) < required_doc_count:
            strong = False
            info["reason"] = "insufficient_doc_coverage"
        info["strength"] = "strong" if strong else "weak"
        info["best_score"] = round(best_score, 4)

        limited_fact_chunks = fact_chunks[: max(top_k, per_doc_limit * 2)]
        if strong:
            return limited_fact_chunks, info

        if not include_fallback:
            return limited_fact_chunks, info
        info["used_hybrid_fallback"] = True
        if not info["reason"]:
            info["reason"] = "weak_fact_signal"
        fallback_chunks = self._exact_lookup_fallback_chunks(
            question,
            doc_ids=doc_ids,
            top_k=top_k,
            per_doc_limit=per_doc_limit,
            mode=mode,
            needs_cross_doc=needs_cross_doc,
            prefer_exhaustive=prefer_exhaustive,
            wants_documents=bool(lookup.get("wants_documents")),
        )
        merged = self._merge_chunks(
            limited_fact_chunks + fallback_chunks,
            limit=max(14, top_k * 3),
        )
        return merged, info

    def _metadata_answer_type_for_operation(self, operation: str) -> str:
        op = str(operation or "").strip().lower()
        if op == "count":
            return "count"
        if op in PERSON_METADATA_OPERATIONS:
            return "person"
        if op in {"latest_uploaded", "earliest_uploaded", "most_frequently_updated", "changed_between_versions"}:
            return "document"
        if op in METADATA_OPERATIONS:
            return "list"
        return "unknown"

    def _route_expected_answer_type(self, route: dict[str, Any]) -> str:
        raw = str(route.get("expected_answer_type") or "").strip().lower()
        if raw in EXPECTED_ANSWER_TYPES:
            return raw
        analysis_plan = route.get("analysis_plan") if isinstance(route.get("analysis_plan"), dict) else {}
        nested = str(analysis_plan.get("expected_answer_type") or "").strip().lower()
        if nested in EXPECTED_ANSWER_TYPES:
            return nested
        operation = str(analysis_plan.get("metadata_operation") or "").strip().lower()
        inferred = self._metadata_answer_type_for_operation(operation)
        if inferred != "unknown":
            return inferred
        task_type = str(route.get("task_type") or "").strip().lower()
        if task_type == "count":
            return "unknown"
        if task_type in {"compare", "trend_analysis"}:
            return "comparison"
        if task_type == "timeline":
            return "timeline"
        return "unknown"

    def _metadata_operation_for_answer_type(self, answer_type: str, filters: dict[str, Any]) -> str:
        normalized = str(answer_type or "").strip().lower()
        if normalized == "count":
            return "count"
        if normalized == "person":
            if str(filters.get("author") or "").strip():
                return "authored_by"
            if str(filters.get("last_modified_by") or "").strip():
                return "edited_by"
            return "last_modified_by"
        if normalized == "document":
            return "list"
        return "list"

    def _has_count_signal(self, question: str) -> bool:
        text = str(question or "").strip().lower()
        if not text:
            return False
        return bool(COUNT_SIGNAL_RE.search(text))

    def _documents_in_scope(self, doc_ids: Optional[List[str]]) -> list[dict]:
        docs = storage.list_documents()
        if not doc_ids:
            return docs
        selected = {str(doc_id or "").strip() for doc_id in doc_ids if str(doc_id or "").strip()}
        if not selected:
            return docs
        return [doc for doc in docs if str(doc.get("id") or "").strip() in selected]

    def _metadata_catalog_entries(self, docs: list[dict]) -> list[dict[str, str]]:
        if not docs:
            return []
        entries: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        per_field_counts: defaultdict[str, int] = defaultdict(int)
        field_limits = {
            "doc_type": 160,
            "target_document": 800,
            "author": 240,
            "last_modified_by": 240,
            "uploader_role": 120,
            "collaborator_type": 32,
            "mime_type": 180,
        }

        def add_entry(field: str, value: Any, text: str, *, doc_id: str = "") -> None:
            raw_value = str(value or "").strip()
            raw_text = str(text or "").strip()
            if not raw_value or not raw_text:
                return
            normalized = raw_value.lower()
            key = (field, normalized, str(doc_id or "").strip())
            if key in seen:
                return
            if per_field_counts[field] >= int(field_limits.get(field, 200)):
                return
            seen.add(key)
            per_field_counts[field] += 1
            entries.append(
                {
                    "field": field,
                    "value": raw_value,
                    "text": raw_text,
                    "doc_id": str(doc_id or "").strip(),
                }
            )

        for doc in docs:
            doc_id = str(doc.get("id") or "").strip()
            filename = str(doc.get("filename") or "").strip()
            metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}

            doc_type = str(metadata.get("doc_type") or "").strip().lower()
            if doc_type:
                label = DOC_TYPE_LABELS.get(doc_type, doc_type.replace("_", " "))
                add_entry(
                    "doc_type",
                    doc_type,
                    f"document type {label} ({doc_type}) for file {filename}",
                    doc_id=doc_id,
                )
            file_ext = str(metadata.get("file_extension") or "").strip().lower()
            if not file_ext and "." in filename:
                file_ext = "." + filename.rsplit(".", 1)[-1].lower()
            if file_ext:
                ext = file_ext.lstrip(".")
                add_entry("doc_type", ext, f"file extension {ext} for {filename}", doc_id=doc_id)
                if ext in {"xls", "xlsx", "xlsm", "xltx", "xltm"}:
                    add_entry(
                        "doc_type",
                        "excel",
                        f"excel spreadsheet file ({ext}) for {filename}",
                        doc_id=doc_id,
                    )
                if ext in {"ppt", "pptx"}:
                    add_entry("doc_type", "pptx", f"powerpoint presentation ({ext}) for {filename}", doc_id=doc_id)
                if ext in {"doc", "docx"}:
                    add_entry("doc_type", "docx", f"word document ({ext}) for {filename}", doc_id=doc_id)

            logical_key = str(metadata.get("logical_document_key") or "").strip()
            title = str(metadata.get("title") or "").strip()
            source_uri = str(metadata.get("source_uri") or "").strip()
            for value in (filename, logical_key, title, source_uri):
                if value:
                    add_entry(
                        "target_document",
                        value,
                        f"document name {value}",
                        doc_id=doc_id,
                    )

            for field in ("author", "last_modified_by", "uploaded_by_role", "collaborator_type", "mime_type"):
                raw = str(metadata.get(field) or "").strip()
                if not raw:
                    continue
                mapped = "uploader_role" if field == "uploaded_by_role" else field
                add_entry(mapped, raw, f"{mapped} value {raw} for {filename}", doc_id=doc_id)
        return entries

    def _semantic_ground_metadata_filters(
        self,
        question: str,
        route: dict[str, Any],
        docs: list[dict],
        existing_filters: dict[str, Any],
    ) -> dict[str, Any]:
        entries = self._metadata_catalog_entries(docs)
        if not entries:
            return {"filters": dict(existing_filters), "matches": []}

        analysis_plan = route.get("analysis_plan") if isinstance(route.get("analysis_plan"), dict) else {}
        operation = str(analysis_plan.get("metadata_operation") or "").strip().lower()
        expected_answer_type = self._route_expected_answer_type(route)
        query_entities = [
            str(value or "").strip()
            for value in (analysis_plan.get("query_entities") or [])
            if str(value or "").strip()
        ]
        if not query_entities:
            query_entities = self._extract_query_entities(question)
        query_text = " ".join([question] + query_entities[:8]).strip()
        query_terms = self._tokenize_for_match(query_text)

        lexical_scores: list[float] = []
        sem_scores: list[float] = []
        entry_terms: list[set[str]] = []
        for entry in entries:
            terms = self._tokenize_for_match(str(entry.get("text") or ""))
            entry_terms.append(terms)
            if query_terms and terms:
                lexical = len(query_terms.intersection(terms)) / float(max(1, len(query_terms)))
            else:
                lexical = 0.0
            lexical_scores.append(lexical)

        embedder = getattr(self.retrieval, "embedder", None)
        if embedder is not None:
            try:
                query_vec = embedder.embed_query(query_text or question)
                text_vecs = embedder.embed_texts([str(entry.get("text") or "") for entry in entries])
                for vec in text_vecs:
                    sem_scores.append(float(np.dot(query_vec, vec)))
            except Exception as exc:
                logger.debug("Metadata grounding embedding failed: %s", exc)
                sem_scores = [0.0 for _ in entries]
        else:
            sem_scores = [0.0 for _ in entries]
        if len(sem_scores) != len(entries):
            sem_scores = [0.0 for _ in entries]

        scored: list[dict[str, Any]] = []
        lowered_question = question.lower()
        for idx, entry in enumerate(entries):
            lexical = max(0.0, float(lexical_scores[idx]))
            semantic = max(0.0, float(sem_scores[idx]))
            value = str(entry.get("value") or "")
            substring_boost = 0.22 if value and value.lower() in lowered_question else 0.0
            score = (0.72 * semantic) + (0.38 * lexical) + substring_boost
            scored.append(
                {
                    "field": str(entry.get("field") or ""),
                    "value": value,
                    "doc_id": str(entry.get("doc_id") or ""),
                    "lexical": round(float(lexical), 6),
                    "semantic": round(float(semantic), 6),
                    "score": round(float(score), 6),
                }
            )
        scored.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)

        best_by_field: dict[str, dict[str, Any]] = {}
        second_by_field: dict[str, dict[str, Any]] = {}
        for item in scored:
            field = str(item.get("field") or "")
            if not field:
                continue
            if field not in best_by_field:
                best_by_field[field] = item
                continue
            if field not in second_by_field:
                second_by_field[field] = item

        merged_filters = dict(existing_filters)
        threshold_by_field = {
            "doc_type": 0.30,
            "target_document": 0.26,
            "uploader_role": 0.34,
            "collaborator_type": 0.28,
        }
        allowed_auto_fields = {"doc_type", "target_document", "uploader_role", "collaborator_type"}
        if expected_answer_type == "person" or operation in PERSON_METADATA_OPERATIONS:
            allowed_auto_fields.add("target_document")
        for field, best in best_by_field.items():
            if field in merged_filters and str(merged_filters.get(field) or "").strip():
                continue
            if field not in allowed_auto_fields:
                continue
            if field not in threshold_by_field:
                continue
            best_score = float(best.get("score") or 0.0)
            second_score = float((second_by_field.get(field) or {}).get("score") or 0.0)
            min_margin = 0.03 if field in {"doc_type", "target_document"} else 0.02
            if best_score < float(threshold_by_field[field]):
                continue
            if second_score > 0.0 and (best_score - second_score) < min_margin:
                continue
            value = str(best.get("value") or "").strip()
            if not value:
                continue
            if field == "target_document":
                value_terms = self._tokenize_for_match(value)
                if len(value_terms) > 6 and value.lower() not in lowered_question:
                    continue
                matched_docs = [doc for doc in docs if self._doc_matches_target_document(doc, value)]
                if not matched_docs:
                    continue
                coverage = float(len(matched_docs)) / float(max(1, len(docs)))
                if coverage > 0.85 and expected_answer_type != "person":
                    continue
            if field in {"uploader_role", "collaborator_type"} and float(best.get("lexical") or 0.0) < 0.12:
                continue
            merged_filters[field] = value

        top_matches = [
            item
            for item in scored[:20]
            if float(item.get("score") or 0.0) > 0.0
        ]
        return {"filters": merged_filters, "matches": top_matches}

    def _infer_date_filters_from_question(
        self,
        question: str,
        existing_filters: dict[str, Any],
    ) -> dict[str, Any]:
        filters = dict(existing_filters)
        if str(filters.get("date_from") or "").strip() or str(filters.get("date_to") or "").strip():
            return filters
        text = str(question or "").strip()
        if not text:
            return filters
        iso_dates = re.findall(
            r"\b((?:19|20)\d{2})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b",
            text,
        )
        parsed_dates: list[datetime] = []
        for year, month, day in iso_dates[:8]:
            try:
                parsed_dates.append(datetime(int(year), int(month), int(day), tzinfo=timezone.utc))
            except Exception:
                continue
        if parsed_dates:
            filters["date_from"] = min(parsed_dates).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            filters["date_to"] = max(parsed_dates).replace(hour=23, minute=59, second=59, microsecond=0).isoformat()
            return filters

        years = sorted({int(year) for year in YEAR_RE.findall(text)})
        if len(years) == 1:
            year = years[0]
            filters["date_from"] = datetime(year, 1, 1, tzinfo=timezone.utc).isoformat()
            filters["date_to"] = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc).isoformat()
            return filters
        if len(years) >= 2:
            start = years[0]
            end = years[-1]
            filters["date_from"] = datetime(start, 1, 1, tzinfo=timezone.utc).isoformat()
            filters["date_to"] = datetime(end, 12, 31, 23, 59, 59, tzinfo=timezone.utc).isoformat()
        return filters

    def _metadata_route_structural_score(self, route: dict[str, Any]) -> int:
        task_type = str(route.get("task_type") or "").strip().lower()
        analysis_plan = route.get("analysis_plan") if isinstance(route.get("analysis_plan"), dict) else {}
        operation = str(analysis_plan.get("metadata_operation") or "").strip().lower()
        filters = analysis_plan.get("metadata_filters") if isinstance(analysis_plan.get("metadata_filters"), dict) else {}
        expected_answer_type = self._route_expected_answer_type(route)
        score = 0
        if task_type in {"count", "metadata_query"}:
            score += 2
        if operation and operation in METADATA_OPERATIONS and operation not in GENERIC_METADATA_OPERATIONS:
            score += 3
        elif operation:
            score += 1
        if filters:
            score += min(3, len([k for k, v in filters.items() if str(v).strip()]))
        if expected_answer_type in {"count", "person", "document", "list"}:
            score += 1
        return score

    def _infer_target_document_hint(self, question: str, docs: list[dict]) -> str:
        entities = self._extract_query_entities(question)
        if not entities:
            entities = sorted(self._tokenize_for_match(question), key=len, reverse=True)[:6]
        best_value = ""
        best_score = -1.0
        total_docs = max(1, len(docs))
        for raw in entities[:12]:
            candidate = str(raw or "").strip()
            if len(candidate) < 3:
                continue
            matched = [doc for doc in docs if self._doc_matches_target_document(doc, candidate)]
            if not matched:
                continue
            coverage = float(len(matched)) / float(total_docs)
            if coverage >= 0.95 and total_docs > 4:
                continue
            specificity = (1.0 - coverage) * min(len(candidate), 24) / 24.0
            score = (0.65 * coverage) + specificity
            if score > best_score:
                best_score = score
                best_value = candidate
        return best_value

    def _run_metadata_repair_router_pass(
        self,
        question: str,
        route: dict[str, Any],
        *,
        doc_ids: Optional[List[str]],
        docs: list[dict],
        grounded_filters: dict[str, Any],
        matches: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if self.router is None:
            return None
        top_hints: list[str] = []
        for item in matches[:10]:
            field = str(item.get("field") or "").strip()
            value = str(item.get("value") or "").strip()
            score = float(item.get("score") or 0.0)
            if not field or not value:
                continue
            top_hints.append(f"- {field}: {value} (score={score:.3f})")
        if not top_hints and not grounded_filters:
            return None
        current = {
            "task_type": route.get("task_type"),
            "analysis_plan": route.get("analysis_plan"),
            "expected_answer_type": route.get("expected_answer_type"),
            "confidence": route.get("confidence"),
        }
        augmented_question = (
            "Metadata planning repair task.\n"
            f"Original question: {question}\n"
            f"Current route guess: {json.dumps(current, ensure_ascii=True)}\n"
            f"Grounded filters: {json.dumps(grounded_filters, ensure_ascii=True)}\n"
            "Candidate metadata values:\n"
            + ("\n".join(top_hints) if top_hints else "- none")
            + "\nReturn corrected routing JSON for the ORIGINAL question."
        )
        try:
            repaired = self.router.route(augmented_question, doc_ids=doc_ids, available_docs=docs)
        except Exception as exc:
            logger.debug("Metadata router repair pass failed: %s", exc)
            return None
        if not isinstance(repaired, dict):
            return None
        repaired.setdefault("source", "llm_router_repair")
        return repaired

    def _repair_metadata_route_if_needed(
        self,
        question: str,
        route: dict[str, Any],
        *,
        doc_ids: Optional[List[str]],
    ) -> dict[str, Any]:
        if not isinstance(route, dict):
            return route
        analysis_plan = route.get("analysis_plan") if isinstance(route.get("analysis_plan"), dict) else {}
        task_type = str(route.get("task_type") or "").strip().lower()
        operation = str(analysis_plan.get("metadata_operation") or "").strip().lower()
        metadata_filters = analysis_plan.get("metadata_filters") if isinstance(analysis_plan.get("metadata_filters"), dict) else {}
        expected_answer_type = self._route_expected_answer_type(route)
        has_count_signal = self._has_count_signal(question)

        strong_metadata_signal = bool(
            task_type in {"count", "metadata_query"}
            or operation in METADATA_OPERATIONS
            or expected_answer_type in {"count", "person"}
        )
        if not strong_metadata_signal:
            return route

        scoped_docs = self._documents_in_scope(doc_ids)
        grounded = self._semantic_ground_metadata_filters(
            question,
            route,
            scoped_docs,
            existing_filters=dict(metadata_filters),
        )
        merged_filters = grounded.get("filters") if isinstance(grounded.get("filters"), dict) else dict(metadata_filters)
        matches = grounded.get("matches") if isinstance(grounded.get("matches"), list) else []
        merged_filters = self._infer_date_filters_from_question(question, merged_filters)

        if expected_answer_type == "person" and not str(merged_filters.get("target_document") or "").strip():
            target_hint = self._infer_target_document_hint(question, scoped_docs)
            if target_hint:
                merged_filters["target_document"] = target_hint

        corrected_operation = operation
        if corrected_operation not in METADATA_OPERATIONS:
            corrected_operation = ""
        if task_type == "count" and not corrected_operation:
            corrected_operation = "count" if has_count_signal else "list"
        if not corrected_operation:
            corrected_operation = self._metadata_operation_for_answer_type(expected_answer_type, merged_filters)
        if expected_answer_type == "count":
            corrected_operation = "count"
        if expected_answer_type == "person" and corrected_operation not in PERSON_METADATA_OPERATIONS:
            corrected_operation = self._metadata_operation_for_answer_type(expected_answer_type, merged_filters)
        if corrected_operation == "count" and not has_count_signal and expected_answer_type != "count":
            corrected_operation = "list"

        needs_repair_pass = bool(
            corrected_operation in GENERIC_METADATA_OPERATIONS
            and not merged_filters
            and len(matches) > 0
        ) or bool(expected_answer_type == "person" and corrected_operation not in PERSON_METADATA_OPERATIONS)

        candidate_route = dict(route)
        candidate_plan = dict(analysis_plan)
        candidate_plan["metadata_operation"] = corrected_operation or "list"
        candidate_plan["metadata_filters"] = dict(merged_filters)
        candidate_route["analysis_plan"] = candidate_plan
        candidate_route["expected_answer_type"] = expected_answer_type
        candidate_route["task_type"] = "count" if candidate_plan["metadata_operation"] == "count" else "metadata_query"

        repaired_route = None
        if needs_repair_pass:
            repaired_route = self._run_metadata_repair_router_pass(
                question,
                candidate_route,
                doc_ids=doc_ids,
                docs=scoped_docs,
                grounded_filters=dict(merged_filters),
                matches=matches,
            )

        best = candidate_route
        if isinstance(repaired_route, dict):
            repaired_plan = repaired_route.get("analysis_plan") if isinstance(repaired_route.get("analysis_plan"), dict) else {}
            repaired_filters = repaired_plan.get("metadata_filters") if isinstance(repaired_plan.get("metadata_filters"), dict) else {}
            if repaired_filters and not repaired_plan.get("metadata_operation"):
                repaired_plan["metadata_operation"] = "list"
            repaired_route["analysis_plan"] = repaired_plan
            repaired_score = self._metadata_route_structural_score(repaired_route)
            baseline_score = self._metadata_route_structural_score(candidate_route)
            if repaired_score >= baseline_score:
                best = repaired_route
                plan = best.get("analysis_plan") if isinstance(best.get("analysis_plan"), dict) else {}
                merged = plan.get("metadata_filters") if isinstance(plan.get("metadata_filters"), dict) else {}
                for key, value in merged_filters.items():
                    if key not in merged and str(value).strip():
                        merged[key] = value
                if merged:
                    plan["metadata_filters"] = merged
                if not str(plan.get("metadata_operation") or "").strip():
                    plan["metadata_operation"] = corrected_operation or "list"
                best["analysis_plan"] = plan

        best_plan = best.get("analysis_plan") if isinstance(best.get("analysis_plan"), dict) else {}
        best_operation = str(best_plan.get("metadata_operation") or "").strip().lower()
        if best_operation not in METADATA_OPERATIONS:
            best_operation = corrected_operation or "list"
            best_plan["metadata_operation"] = best_operation
        best_filters = best_plan.get("metadata_filters") if isinstance(best_plan.get("metadata_filters"), dict) else {}
        if not best_filters:
            best_plan["metadata_filters"] = dict(merged_filters)
        best["analysis_plan"] = best_plan
        best["expected_answer_type"] = self._route_expected_answer_type(best)
        best["task_type"] = "count" if best_operation == "count" else "metadata_query"
        return best

    def _metadata_query_signal(self, route: dict[str, Any]) -> dict[str, Any]:
        task_type = str(route.get("task_type", "")).strip().lower()
        analysis_plan = route.get("analysis_plan") if isinstance(route.get("analysis_plan"), dict) else {}
        operation = str(analysis_plan.get("metadata_operation", "")).strip().lower()
        if operation and operation not in METADATA_OPERATIONS:
            operation = ""
        filters = analysis_plan.get("metadata_filters") if isinstance(analysis_plan.get("metadata_filters"), dict) else {}
        expected_answer_type = self._route_expected_answer_type(route)
        try:
            confidence = float(route.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        is_metadata_query = bool(task_type in {"count", "metadata_query"} or operation)
        if not is_metadata_query:
            return {"is_metadata_query": False}
        if not operation:
            operation = "count" if task_type == "count" else "list"
        intent = "count" if operation == "count" or task_type == "count" else "metadata_query"
        return {
            "is_metadata_query": True,
            "intent": intent,
            "operation": operation,
            "filters": dict(filters),
            "expected_answer_type": expected_answer_type,
            "confidence": confidence,
        }

    def _parse_datetime(self, value: Any) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00").replace("/", "-")
        parsed: datetime | None = None
        for candidate in (normalized, text):
            if not candidate:
                continue
            try:
                parsed = datetime.fromisoformat(candidate)
                break
            except Exception:
                continue
        if parsed is None:
            year_month = re.fullmatch(r"((?:19|20)\d{2})-(0[1-9]|1[0-2])", normalized)
            if year_month:
                parsed = datetime(
                    int(year_month.group(1)),
                    int(year_month.group(2)),
                    1,
                    tzinfo=timezone.utc,
                )
        if parsed is None:
            year_only = re.fullmatch(r"(?:19|20)\d{2}", normalized)
            if year_only:
                parsed = datetime(int(normalized), 1, 1, tzinfo=timezone.utc)
        if parsed is None:
            compact = re.search(r"((?:19|20)\d{2})(0[1-9]|1[0-2])([0-2]\d|3[01])", normalized)
            if compact:
                try:
                    parsed = datetime(
                        int(compact.group(1)),
                        int(compact.group(2)),
                        int(compact.group(3)),
                        tzinfo=timezone.utc,
                    )
                except Exception:
                    parsed = None
        if parsed is None:
            year_match = re.search(r"(?:19|20)\d{2}", normalized)
            if year_match:
                try:
                    parsed = datetime(int(year_match.group(0)), 1, 1, tzinfo=timezone.utc)
                except Exception:
                    parsed = None
        if parsed is None:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _doc_created_at(self, doc: dict[str, Any]) -> datetime | None:
        metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        for raw in (
            metadata.get("content_created_at"),
            doc.get("created_at"),
            metadata.get("uploaded_at"),
            metadata.get("fs_created_at"),
        ):
            parsed = self._parse_datetime(raw)
            if parsed is not None:
                return parsed
        return None

    def _doc_updated_at(self, doc: dict[str, Any]) -> datetime | None:
        metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        for raw in (
            metadata.get("updated_at"),
            metadata.get("content_modified_at"),
            metadata.get("fs_modified_at"),
            doc.get("created_at"),
        ):
            parsed = self._parse_datetime(raw)
            if parsed is not None:
                return parsed
        return None

    def _metadata_sources_from_documents(self, docs: list[dict]) -> list[dict]:
        sources: list[dict] = []
        for doc in docs[:24]:
            metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
            doc_type = str(metadata.get("doc_type") or "unknown").strip().lower()
            label = DOC_TYPE_LABELS.get(doc_type, doc_type.replace("_", " ") or "document")
            try:
                confidence = float(metadata.get("doc_type_confidence") or 0.0)
            except Exception:
                confidence = 0.0
            summary_parts = [
                f"doc_type={label}",
                f"doc_type_confidence={confidence:.2f}",
                f"created_at={str(doc.get('created_at') or '')}",
                f"updated_at={str(metadata.get('updated_at') or '')}",
            ]
            author = str(metadata.get("author") or "").strip()
            editor = str(metadata.get("last_modified_by") or "").strip()
            if author:
                summary_parts.append(f"author={author}")
            if editor:
                summary_parts.append(f"last_modified_by={editor}")
            sources.append(
                {
                    "id": f"doc-meta:{doc.get('id')}",
                    "doc_id": str(doc.get("id") or ""),
                    "doc_filename": str(doc.get("filename") or ""),
                    "page": 1,
                    "content": "; ".join(summary_parts),
                    "source_type": "document_metadata",
                }
            )
        return sources

    def _doc_matches_doc_type_filter(self, doc: dict[str, Any], raw_filter: str) -> bool:
        value = str(raw_filter or "").strip().lower()
        if not value:
            return True
        metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        filename = str(doc.get("filename") or "").strip().lower()
        file_ext = str(metadata.get("file_extension") or "").strip().lower()
        if not file_ext and "." in filename:
            file_ext = "." + filename.rsplit(".", 1)[-1]
        normalized_value = value.lstrip(".")
        if normalized_value in {"pdf", "docx", "pptx", "xlsx", "xls", "txt", "csv", "json"}:
            return file_ext == f".{normalized_value}"
        doc_type = str(metadata.get("doc_type") or "").strip().lower()
        if value == doc_type:
            return True
        candidate_types = extract_query_doc_type_candidates(value)
        if candidate_types and doc_type in set(candidate_types):
            return True
        tags = metadata.get("auto_tags") if isinstance(metadata.get("auto_tags"), list) else []
        doc_text = " ".join([filename, doc_type, " ".join(str(tag or "") for tag in tags)]).lower()
        return value in doc_text

    def _doc_matches_target_document(self, doc: dict[str, Any], raw_filter: str) -> bool:
        value = str(raw_filter or "").strip().lower()
        if not value:
            return True
        metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        candidates = [
            str(doc.get("filename") or ""),
            str(metadata.get("title") or ""),
            str(metadata.get("subject") or ""),
            str(metadata.get("logical_document_key") or ""),
            str(metadata.get("source_uri") or ""),
        ]
        lowered = [entry.lower() for entry in candidates if str(entry).strip()]
        if any(value in entry for entry in lowered):
            return True
        target_terms = self._tokenize_for_match(value)
        if not target_terms:
            return False
        haystack_terms: set[str] = set()
        for entry in lowered:
            haystack_terms.update(self._tokenize_for_match(entry))
        if not haystack_terms:
            return False
        overlap = len(target_terms.intersection(haystack_terms))
        return overlap >= max(1, int(round(0.6 * len(target_terms))))

    def _route_exact_lookup_hint(self, route: dict[str, Any]) -> dict[str, Any]:
        analysis_plan = route.get("analysis_plan") if isinstance(route.get("analysis_plan"), dict) else {}
        fact_types = [
            str(value).strip().lower()
            for value in (analysis_plan.get("fact_types") or [])
            if str(value).strip().lower() in {"date", "amount", "party"}
        ]
        return {
            "requested": bool(analysis_plan.get("exact_lookup_requested", False) or fact_types),
            "fact_types": fact_types,
        }

    def _resolve_route_target_doc_ids(
        self,
        route: dict[str, Any],
        docs: list[dict],
        *,
        min_resolved: int = 1,
    ) -> list[str]:
        if not docs or not isinstance(route, dict):
            return []
        analysis_plan = route.get("analysis_plan") if isinstance(route.get("analysis_plan"), dict) else {}
        task_type = str(route.get("task_type") or "").strip().lower()
        metadata_filters = analysis_plan.get("metadata_filters") if isinstance(analysis_plan.get("metadata_filters"), dict) else {}
        raw_targets: list[str] = []
        seen_targets: set[str] = set()

        def add_target(value: Any) -> None:
            cleaned = re.sub(r"\s+", " ", str(value or "").strip())
            key = cleaned.lower()
            if len(cleaned) < 3 or key in seen_targets:
                return
            seen_targets.add(key)
            raw_targets.append(cleaned)

        for value in analysis_plan.get("target_documents") or []:
            add_target(value)
        add_target(metadata_filters.get("target_document"))
        if task_type == "compare":
            for value in analysis_plan.get("query_entities") or []:
                add_target(value)

        resolved: list[str] = []
        seen_doc_ids: set[str] = set()
        for target in raw_targets[:6]:
            matched = [doc for doc in docs if self._doc_matches_target_document(doc, target)]
            if not matched or len(matched) > max(4, min(6, len(docs))):
                continue
            for doc in matched:
                doc_id = str(doc.get("id") or "").strip()
                if not doc_id or doc_id in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc_id)
                resolved.append(doc_id)
        return resolved if len(resolved) >= min_resolved else []

    def _is_summary_like_query(self, question: str) -> bool:
        text = str(question or "").strip().lower()
        if not text:
            return False
        return bool(SUMMARY_ROUTE_PREFIX_RE.match(text) or SUMMARY_ROUTE_WHAT_SAY_RE.match(text))

    def _extract_summary_target_text(self, question: str) -> str:
        text = str(question or "").strip()
        if not text:
            return ""
        match = SUMMARY_ROUTE_PREFIX_RE.match(text) or SUMMARY_ROUTE_WHAT_SAY_RE.match(text)
        if match is None:
            return ""
        body = re.sub(r"\s+", " ", str(match.group("body") or "").strip(" \t\r\n\"'`.,:;!?()[]{}"))
        if not body:
            return ""
        generic = {
            "document",
            "doc",
            "file",
            "agreement",
            "nda",
            "this document",
            "that document",
            "this agreement",
            "that agreement",
        }
        return "" if body.lower() in generic else body

    def _repair_named_document_summary_route_if_needed(
        self,
        question: str,
        route: dict[str, Any],
        *,
        doc_ids: Optional[List[str]],
    ) -> dict[str, Any]:
        if not isinstance(route, dict):
            return route
        task_type = str(route.get("task_type") or "").strip().lower()
        if task_type not in {"qa", "summarize"} or not bool(route.get("needs_cross_doc", False)):
            return route
        if self._detect_intent(question, doc_ids) == "compare":
            return route
        selected_doc_ids = [str(doc_id).strip() for doc_id in (doc_ids or []) if str(doc_id).strip()]
        if len(selected_doc_ids) >= 2 or not self._is_summary_like_query(question):
            return route

        scoped_docs = self._documents_in_scope(doc_ids)
        target_doc_ids = self._resolve_route_target_doc_ids(route, scoped_docs, min_resolved=1)
        if len(target_doc_ids) != 1:
            target_text = self._extract_summary_target_text(question)
            if target_text:
                matched = [doc for doc in scoped_docs if self._doc_matches_target_document(doc, target_text)]
                if len(matched) == 1:
                    target_doc_ids = [str(matched[0].get("id") or "").strip()]
        if len(target_doc_ids) != 1:
            return route

        repaired = dict(route)
        repaired["task_type"] = "qa"
        repaired["needs_cross_doc"] = False
        repaired["source"] = "llm_router_named_doc_summary_repair"
        repaired["rationale"] = "Named-document summary request repaired to single-document retrieval."

        retrieval_plan = repaired.get("retrieval_plan") if isinstance(repaired.get("retrieval_plan"), dict) else {}
        updated_plan = dict(retrieval_plan)
        updated_plan["strategy"] = "semantic"
        try:
            updated_plan["top_k"] = max(6, int(updated_plan.get("top_k", 8) or 8))
        except Exception:
            updated_plan["top_k"] = 8
        try:
            updated_plan["per_doc_limit"] = max(1, int(updated_plan.get("per_doc_limit", 1) or 1))
        except Exception:
            updated_plan["per_doc_limit"] = 1
        repaired["retrieval_plan"] = updated_plan

        analysis_plan = repaired.get("analysis_plan") if isinstance(repaired.get("analysis_plan"), dict) else {}
        updated_analysis = dict(analysis_plan)
        if not updated_analysis.get("target_documents"):
            target_doc = next((doc for doc in scoped_docs if str(doc.get("id") or "").strip() == target_doc_ids[0]), None)
            if target_doc is not None:
                updated_analysis["target_documents"] = [
                    str(target_doc.get("filename") or (target_doc.get("metadata") or {}).get("title") or target_doc_ids[0])
                ]
        repaired["analysis_plan"] = updated_analysis
        return repaired

    def _rank_documents_for_target(self, target: str, docs: list[dict]) -> list[dict]:
        query = str(target or "").strip().lower()
        if not query:
            return list(docs)
        query_terms = self._tokenize_for_match(query)
        embedder = getattr(self.retrieval, "embedder", None)
        query_vec = None
        if embedder is not None:
            try:
                query_vec = embedder.embed_query(query)
            except Exception:
                query_vec = None
        scored: list[tuple[float, dict]] = []
        for doc in docs:
            metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
            text = " ".join(
                [
                    str(doc.get("filename") or ""),
                    str(metadata.get("logical_document_key") or ""),
                    str(metadata.get("title") or ""),
                    str(metadata.get("source_uri") or ""),
                ]
            ).strip()
            terms = self._tokenize_for_match(text)
            lexical = 0.0
            if query_terms and terms:
                lexical = len(query_terms.intersection(terms)) / float(max(1, len(query_terms)))
            substring_boost = 0.25 if query in text.lower() else 0.0
            semantic = 0.0
            if embedder is not None and query_vec is not None:
                try:
                    vec = embedder.embed_texts([text])[0]
                    semantic = float(np.dot(query_vec, vec))
                except Exception:
                    semantic = 0.0
            score = (0.70 * semantic) + (0.42 * lexical) + substring_boost
            scored.append((score, doc))
        return [item[1] for item in sorted(scored, key=lambda current: current[0], reverse=True)]

    def _metadata_reference_now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _answer_metadata_or_hybrid_query(
        self,
        question: str,
        *,
        doc_ids: Optional[List[str]],
        metadata_signal: dict[str, Any],
    ) -> tuple[str, list[dict], dict[str, Any]]:
        query_intent = self._build_query_intent(question, metadata_signal)
        if not bool(config.HYBRID_METADATA_SEMANTIC):
            answer, sources = self._answer_metadata_query(
                question,
                doc_ids=doc_ids,
                metadata_signal=metadata_signal,
            )
            return answer, sources, query_intent
        if str(query_intent.get("answer_mode") or "METADATA_ONLY") == "METADATA_ONLY":
            answer, sources = self._answer_metadata_query(
                question,
                doc_ids=doc_ids,
                metadata_signal=metadata_signal,
            )
            return answer, sources, query_intent
        answer, sources = self._answer_hybrid_metadata_semantic_query(
            question,
            doc_ids=doc_ids,
            metadata_signal=metadata_signal,
            query_intent=query_intent,
        )
        return answer, sources, query_intent

    def _metadata_terms_for_intent(
        self,
        operation: str,
        filters: dict[str, Any],
    ) -> set[str]:
        terms = set(self._tokenize_for_match(operation.replace("_", " ")))
        for field in QUERY_INTENT_METADATA_FIELDS:
            terms.update(self._tokenize_for_match(field.replace("_", " ")))
        for value in filters.values():
            terms.update(self._tokenize_for_match(str(value or "")))
            candidate_types = extract_query_doc_type_candidates(str(value or ""))
            for doc_type in candidate_types:
                terms.update(self._tokenize_for_match(doc_type.replace("_", " ")))
                for hint in DOC_TYPE_HINTS.get(doc_type, []):
                    terms.update(self._tokenize_for_match(hint))
        return terms

    def _build_query_intent(
        self,
        question: str,
        metadata_signal: dict[str, Any],
    ) -> dict[str, Any]:
        filters = dict(metadata_signal.get("filters") or {}) if isinstance(metadata_signal.get("filters"), dict) else {}
        hard_filters = {key: value for key, value in filters.items() if str(value).strip()}
        soft_filters: dict[str, Any] = {}
        operation = str(metadata_signal.get("operation") or "list").strip().lower()
        if operation not in METADATA_OPERATIONS:
            operation = "list"
        expected_answer_type = str(metadata_signal.get("expected_answer_type") or "").strip().lower()
        query_terms = self._tokenize_for_match(question)
        metadata_terms = self._metadata_terms_for_intent(operation, hard_filters)
        semantic_terms = sorted(
            {
                term
                for term in query_terms
                if term not in metadata_terms and term not in QUERY_INTENT_NON_SEMANTIC_TERMS
            }
        )
        query_years = sorted({year for year in YEAR_RE.findall(question)})
        semantic_classes: set[str] = set()
        if set(semantic_terms).intersection(SEMANTIC_TEMPORAL_TERMS):
            semantic_classes.add("temporal")
        if set(semantic_terms).intersection(SEMANTIC_PARTY_TERMS):
            semantic_classes.add("party")
        if set(semantic_terms).intersection(STATUS_EXPIRED_TERMS.union(STATUS_ACTIVE_TERMS)):
            semantic_classes.add("status")
        if query_years:
            semantic_classes.add("year")
        if semantic_terms and not semantic_classes:
            semantic_classes.add("content")

        semantic_targets: list[str] = []
        if "status" in semantic_classes:
            semantic_targets.append("derived_status")
        if "temporal" in semantic_classes or "year" in semantic_classes:
            semantic_targets.append("temporal_facts")
        if "party" in semantic_classes:
            semantic_targets.append("party_entities")
        if "content" in semantic_classes:
            semantic_targets.append("content_facts")
        if query_years and not {"date_from", "date_to", "relative_days"}.intersection(hard_filters):
            if "year_inference" not in semantic_targets:
                semantic_targets.append("year_inference")
        if query_years and {"date_from", "date_to"}.intersection(hard_filters):
            for key in ("date_from", "date_to"):
                value = hard_filters.pop(key, None)
                if value is not None:
                    soft_filters[key] = value

        requires_semantic = bool(semantic_targets)
        if operation not in GENERIC_METADATA_OPERATIONS and not requires_semantic:
            answer_mode = "METADATA_ONLY"
        elif not requires_semantic:
            answer_mode = "METADATA_ONLY"
        elif hard_filters:
            answer_mode = "HYBRID"
        else:
            answer_mode = "SEMANTIC_ONLY"
        if expected_answer_type == "person" and operation in PERSON_METADATA_OPERATIONS:
            answer_mode = "METADATA_ONLY"
            requires_semantic = False
            semantic_targets = []
        status_target = ""
        lowered_question = str(question or "").lower()
        negative_active_pattern = re.search(
            r"\b(?:no|not)\s+(?:longer\s+)?(?:valid|active|effective|enforceable)\b",
            lowered_question,
        )
        expired_hits = len(set(semantic_terms).intersection(STATUS_EXPIRED_TERMS))
        active_hits = len(set(semantic_terms).intersection(STATUS_ACTIVE_TERMS))
        if negative_active_pattern is not None:
            status_target = "expired"
        elif expired_hits > active_hits and expired_hits > 0:
            status_target = "expired"
        elif active_hits > expired_hits and active_hits > 0:
            status_target = "active"
        covered = len(set(query_terms).intersection(metadata_terms))
        query_count = max(1, len(query_terms))
        metadata_coverage = covered / float(query_count)
        confidence = max(0.05, min(0.99, 0.35 + (0.45 * metadata_coverage) + (0.15 if not requires_semantic else 0.0)))
        return {
            "hard_filters": hard_filters,
            "soft_filters": soft_filters,
            "semantic_targets": semantic_targets,
            "requires_semantic": requires_semantic,
            "answer_mode": answer_mode,
            "confidence": round(float(confidence), 4),
            "semantic_terms": semantic_terms,
            "query_years": query_years,
            "semantic_classes": sorted(semantic_classes),
            "status_target": status_target,
            "operation": operation,
        }

    def _answer_hybrid_metadata_semantic_query(
        self,
        question: str,
        *,
        doc_ids: Optional[List[str]],
        metadata_signal: dict[str, Any],
        query_intent: dict[str, Any],
    ) -> tuple[str, list[dict]]:
        operation = str(metadata_signal.get("operation") or "list").strip().lower()
        if operation not in METADATA_OPERATIONS:
            operation = "list"
        filters = dict(metadata_signal.get("filters") or {}) if isinstance(metadata_signal.get("filters"), dict) else {}
        if operation == "external_collaborators":
            filters["collaborator_type"] = "external"
        if operation == "authored_by" and not filters.get("author") and filters.get("target_document"):
            filters["author"] = str(filters.get("target_document") or "").strip()
        if operation == "edited_by" and not filters.get("last_modified_by") and filters.get("target_document"):
            filters["last_modified_by"] = str(filters.get("target_document") or "").strip()

        all_docs = self.metadata_semantic_adapter.list_documents(doc_ids=doc_ids)
        if not all_docs:
            return "I cannot find documents in the current scope.", []

        hard_filters = dict(query_intent.get("hard_filters") or {})
        candidate_docs = (
            self._filter_documents_by_metadata(all_docs, operation=operation, filters=hard_filters)
            if hard_filters
            else list(all_docs)
        )
        scope_relaxed = False
        if not candidate_docs:
            scope_relaxed = True
            candidate_docs = list(all_docs)
        ranked_candidates = sorted(
            candidate_docs,
            key=lambda item: self._doc_updated_at(item) or self._doc_created_at(item) or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        candidate_docs = ranked_candidates[: int(config.HYBRID_METADATA_MAX_CANDIDATES)]
        candidate_doc_ids = [str(doc.get("id") or "").strip() for doc in candidate_docs if str(doc.get("id") or "").strip()]
        if not candidate_doc_ids:
            return "I cannot find documents in the current scope.", []

        chunks, evidence_score, did_retry = self._retrieve_hybrid_semantic_chunks(
            question,
            candidate_doc_ids=candidate_doc_ids,
            semantic_terms=list(query_intent.get("semantic_terms") or []),
        )
        if not chunks:
            fallback_answer, fallback_sources = self._answer_metadata_query(
                question,
                doc_ids=doc_ids,
                metadata_signal=metadata_signal,
            )
            return fallback_answer, fallback_sources

        matched_docs, evidence_chunks, facts_by_doc = self._match_documents_from_hybrid_evidence(
            question,
            candidate_docs=candidate_docs,
            chunks=chunks,
            query_intent=query_intent,
        )
        if not matched_docs:
            answer = self._fallback_answer(
                question,
                chunks,
                intent="qa",
                include_document_summaries=False,
            )
            if scope_relaxed:
                answer = (
                    "I expanded scope beyond strict metadata filters because no candidate documents matched those filters.\n"
                    + answer
                )
            return answer, self._prepare_response_sources(chunks, intent="qa")

        answer = self._build_hybrid_metadata_answer(
            matched_docs=matched_docs,
            facts_by_doc=facts_by_doc,
            query_intent=query_intent,
            evidence_score=evidence_score,
            did_retry=did_retry,
            scope_relaxed=scope_relaxed,
        )
        return answer, self._prepare_response_sources(evidence_chunks, intent="qa")

    def _retrieve_hybrid_semantic_chunks(
        self,
        question: str,
        *,
        candidate_doc_ids: list[str],
        semantic_terms: list[str],
    ) -> tuple[list[dict], float, bool]:
        return self.metadata_semantic_adapter.search_chunks_with_expansion(
            query=question,
            doc_ids=candidate_doc_ids,
            top_k=int(config.HYBRID_METADATA_TOP_K),
            per_doc_limit=int(config.HYBRID_METADATA_PER_DOC_LIMIT),
            mode="hybrid",
            semantic_terms=semantic_terms,
            min_evidence_score=float(config.HYBRID_METADATA_MIN_EVIDENCE_SCORE),
            evidence_scorer=self._hybrid_evidence_score,
            merge_chunks=self._merge_chunks,
            merged_limit=max(20, int(config.HYBRID_METADATA_TOP_K) * 2),
            fallback_expansion_terms=list(SEMANTIC_EXPANSION_TERMS),
        )

    def _hybrid_evidence_score(self, question: str, chunks: list[dict]) -> float:
        if not chunks:
            return 0.0
        scored = sorted(
            chunks,
            key=lambda item: float(item.get("rerank_score", item.get("score", 0.0)) or 0.0),
            reverse=True,
        )[:12]
        support = 0.0
        for chunk in scored:
            overlap = self._chunk_query_overlap(question, str(chunk.get("content") or ""))
            score = float(chunk.get("rerank_score", chunk.get("score", 0.0)) or 0.0)
            support += max(overlap, score)
        unique_docs = {
            str(chunk.get("doc_id") or "").strip()
            for chunk in scored
            if str(chunk.get("doc_id") or "").strip()
        }
        diversity_bonus = min(0.35, float(len(unique_docs)) * 0.08)
        normalized_support = support / float(max(1, len(scored)))
        return float(max(0.0, min(1.0, normalized_support + diversity_bonus)))

    def _extract_dates_with_roles(self, text: str) -> list[dict[str, Any]]:
        normalized = str(text or "")
        if not normalized:
            return []
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for match in ISO_DATE_RE.finditer(normalized):
            year, month, day = match.groups()
            try:
                dt = datetime(int(year), int(month), int(day), tzinfo=timezone.utc)
            except Exception:
                continue
            key = dt.isoformat()
            if key in seen:
                continue
            seen.add(key)
            context = normalized[max(0, match.start() - 96) : match.start()].lower()
            role = "unspecified"
            context_terms = self._tokenize_for_match(context.replace("-", "_"))
            for candidate_role, role_terms in DATE_ROLE_TERMS.items():
                if context_terms.intersection(role_terms):
                    role = candidate_role
                    break
            results.append({"datetime": dt, "role": role, "value": match.group(0)})
        for regex in (DMY_MONTH_RE, MONTH_DY_RE):
            for match in regex.finditer(normalized):
                if regex is DMY_MONTH_RE:
                    day_raw, month_raw, year_raw = match.groups()
                else:
                    month_raw, day_raw, year_raw = match.groups()
                month = MONTH_NAME_TO_NUM.get(str(month_raw).strip().lower(), 0)
                if month <= 0:
                    continue
                try:
                    dt = datetime(int(year_raw), int(month), int(day_raw), tzinfo=timezone.utc)
                except Exception:
                    continue
                key = dt.isoformat()
                if key in seen:
                    continue
                seen.add(key)
                context = normalized[max(0, match.start() - 96) : match.start()].lower()
                role = "unspecified"
                context_terms = self._tokenize_for_match(context.replace("-", "_"))
                for candidate_role, role_terms in DATE_ROLE_TERMS.items():
                    if context_terms.intersection(role_terms):
                        role = candidate_role
                        break
                results.append({"datetime": dt, "role": role, "value": match.group(0)})
        return sorted(results, key=lambda item: item["datetime"])

    def _extract_party_mentions(self, text: str) -> list[str]:
        normalized = str(text or "")
        if not normalized:
            return []
        parties: list[str] = []
        seen: set[str] = set()
        patterns = (
            re.compile(
                r"\b(?:between|by and between)\s+([A-Z][A-Za-z0-9&.,'()\- ]{2,80}?)\s+and\s+([A-Z][A-Za-z0-9&.,'()\- ]{2,80}?)(?:[\n\r.,;]|$)"
            ),
            re.compile(r"\b(?:party|counterparty|customer|client|vendor|supplier)\s*[:\-]\s*([A-Z][A-Za-z0-9&.,'()\- ]{2,90})"),
        )
        for pattern in patterns:
            for match in pattern.finditer(normalized):
                for group_idx in range(1, len(match.groups()) + 1):
                    value = re.sub(r"\s+", " ", str(match.group(group_idx) or "").strip(" \t\r\n,.;:"))
                    if len(value) < 3:
                        continue
                    lowered = value.lower()
                    if lowered in seen:
                        continue
                    seen.add(lowered)
                    parties.append(value)
        return parties[:12]

    def _extract_semantic_facts_for_doc(
        self,
        doc: dict[str, Any],
        doc_chunks: list[dict],
    ) -> dict[str, Any]:
        combined_text = " ".join(str(chunk.get("content") or "") for chunk in doc_chunks[:12])
        dates = self._extract_dates_with_roles(combined_text)
        parties = self._extract_party_mentions(combined_text)
        now = self._metadata_reference_now()
        effective_dates = [item["datetime"] for item in dates if str(item.get("role")) == "effective"]
        execution_dates = [item["datetime"] for item in dates if str(item.get("role")) == "execution"]
        termination_dates = [item["datetime"] for item in dates if str(item.get("role")) == "termination"]
        text_years = {str(item["datetime"].year) for item in dates}
        metadata_years: set[str] = set()
        metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        created_dt = self._doc_created_at(doc)
        updated_dt = self._doc_updated_at(doc)
        if created_dt is not None:
            metadata_years.add(str(created_dt.year))
        if updated_dt is not None:
            metadata_years.add(str(updated_dt.year))
        years = set(text_years).union(metadata_years)
        status = "unknown"
        if termination_dates:
            latest_termination = sorted(termination_dates)[-1]
            status = "expired" if latest_termination < now else "active"
        elif effective_dates:
            earliest_effective = sorted(effective_dates)[0]
            status = "active" if earliest_effective <= now else "unknown"
        combined_terms = self._tokenize_for_match(combined_text)
        if status == "unknown":
            if combined_terms.intersection(STATUS_EXPIRED_TERMS):
                status = "expired"
            elif combined_terms.intersection(STATUS_ACTIVE_TERMS):
                status = "active"
        status_terms = {status} if status != "unknown" else set()
        party_terms: set[str] = set()
        for party in parties:
            party_terms.update(self._tokenize_for_match(party))
        fact_terms = set(combined_terms)
        fact_terms.update({year for year in years if year})
        fact_terms.update(status_terms)
        fact_terms.update(party_terms)
        return {
            "status": status,
            "years": sorted(years),
            "text_years": sorted(text_years),
            "metadata_years": sorted(metadata_years),
            "parties": parties,
            "effective_dates": effective_dates,
            "execution_dates": execution_dates,
            "termination_dates": termination_dates,
            "fact_terms": fact_terms,
            "author": str(metadata.get("author") or "").strip(),
        }

    def _semantic_doc_match_score(
        self,
        question: str,
        *,
        query_intent: dict[str, Any],
        doc_chunks: list[dict],
        facts: dict[str, Any],
    ) -> float:
        semantic_terms = set(str(term) for term in (query_intent.get("semantic_terms") or []) if str(term).strip())
        query_years = set(str(year) for year in (query_intent.get("query_years") or []) if str(year).strip())
        status_target = str(query_intent.get("status_target") or "").strip().lower()
        doc_terms = set(facts.get("fact_terms") or set())
        overlap_score = 0.0
        if semantic_terms:
            overlap_score = len(semantic_terms.intersection(doc_terms)) / float(max(1, len(semantic_terms)))
        top_chunk_score = 0.0
        for chunk in sorted(
            doc_chunks,
            key=lambda item: float(item.get("rerank_score", item.get("score", 0.0)) or 0.0),
            reverse=True,
        )[:4]:
            chunk_score = max(
                self._chunk_query_overlap(question, str(chunk.get("content") or "")),
                float(chunk.get("rerank_score", chunk.get("score", 0.0)) or 0.0),
            )
            if chunk_score > top_chunk_score:
                top_chunk_score = chunk_score
        score = (0.58 * overlap_score) + (0.42 * top_chunk_score)
        if query_years:
            text_years = set(str(year) for year in (facts.get("text_years") or []))
            metadata_years = set(str(year) for year in (facts.get("metadata_years") or []))
            if text_years and query_years.intersection(text_years):
                score += 0.24
            elif text_years and not query_years.intersection(text_years):
                score -= 0.35
            elif query_years.intersection(metadata_years):
                score += 0.08
            else:
                score -= 0.25
        if status_target:
            status = str(facts.get("status") or "").strip().lower()
            if status == status_target:
                score += 0.22
            elif status and status != "unknown":
                score -= 0.18
        if "party_entities" in set(query_intent.get("semantic_targets") or []) and (facts.get("parties") or []):
            score += 0.16
        return float(score)

    def _match_documents_from_hybrid_evidence(
        self,
        question: str,
        *,
        candidate_docs: list[dict[str, Any]],
        chunks: list[dict],
        query_intent: dict[str, Any],
    ) -> tuple[list[dict], list[dict], dict[str, dict[str, Any]]]:
        by_doc: dict[str, list[dict]] = defaultdict(list)
        for chunk in chunks:
            doc_id = str(chunk.get("doc_id") or "").strip()
            if doc_id:
                by_doc[doc_id].append(chunk)
        scored: list[tuple[float, dict, dict[str, Any], list[dict]]] = []
        facts_by_doc: dict[str, dict[str, Any]] = {}
        semantic_classes = set(str(value) for value in (query_intent.get("semantic_classes") or []))
        status_target = str(query_intent.get("status_target") or "").strip().lower()
        query_years = {str(year) for year in (query_intent.get("query_years") or []) if str(year).strip()}
        for doc in candidate_docs:
            doc_id = str(doc.get("id") or "").strip()
            doc_chunks = by_doc.get(doc_id, [])
            if not doc_chunks:
                continue
            facts = self._extract_semantic_facts_for_doc(doc, doc_chunks)
            facts_by_doc[doc_id] = facts
            if status_target:
                status = str(facts.get("status") or "").strip().lower()
                if status and status != "unknown" and status != status_target:
                    continue
            if query_years:
                text_years = {str(year) for year in (facts.get("text_years") or []) if str(year).strip()}
                if text_years and not query_years.intersection(text_years):
                    continue
                doc_years = {str(year) for year in (facts.get("years") or []) if str(year).strip()}
                if not text_years and doc_years and not query_years.intersection(doc_years):
                    continue
            score = self._semantic_doc_match_score(
                question,
                query_intent=query_intent,
                doc_chunks=doc_chunks,
                facts=facts,
            )
            if "party" in semantic_classes and not (facts.get("parties") or []):
                score -= 0.1
            scored.append((score, doc, facts, doc_chunks))
        scored.sort(key=lambda item: item[0], reverse=True)
        threshold = 0.22 if query_intent.get("semantic_terms") else 0.12
        matched_docs: list[dict] = []
        evidence_chunks: list[dict] = []
        for score, doc, _facts, doc_chunks in scored:
            if score < threshold and matched_docs:
                continue
            if score < 0.0:
                continue
            matched_docs.append(doc)
            top_chunk = sorted(
                doc_chunks,
                key=lambda item: float(item.get("rerank_score", item.get("score", 0.0)) or 0.0),
                reverse=True,
            )[:2]
            evidence_chunks.extend(top_chunk)
            if len(matched_docs) >= 8:
                break
        if not matched_docs and scored:
            score, doc, _facts, doc_chunks = scored[0]
            if score >= 0.0:
                matched_docs = [doc]
                evidence_chunks = sorted(
                    doc_chunks,
                    key=lambda item: float(item.get("rerank_score", item.get("score", 0.0)) or 0.0),
                    reverse=True,
                )[:2]
        merged_chunks = self._merge_chunks(evidence_chunks, limit=max(6, len(evidence_chunks)))
        return matched_docs, merged_chunks, facts_by_doc

    def _build_hybrid_metadata_answer(
        self,
        *,
        matched_docs: list[dict],
        facts_by_doc: dict[str, dict[str, Any]],
        query_intent: dict[str, Any],
        evidence_score: float,
        did_retry: bool,
        scope_relaxed: bool,
    ) -> str:
        operation = str(query_intent.get("operation") or "list").strip().lower()
        count = len(matched_docs)
        header = (
            f"There are {count} document(s) matching metadata scope and semantic evidence."
            if operation == "count"
            else f"I found {count} document(s) matching metadata scope and semantic evidence."
        )
        lines = [header]
        for doc in matched_docs[:8]:
            doc_id = str(doc.get("id") or "").strip()
            filename = str(doc.get("filename") or doc_id or "unknown")
            facts = facts_by_doc.get(doc_id, {})
            parts: list[str] = []
            status = str(facts.get("status") or "").strip().lower()
            if status and status != "unknown":
                parts.append(f"status={status}")
            years = [str(value) for value in (facts.get("years") or [])][:3]
            if years:
                parts.append(f"years={', '.join(years)}")
            parties = [str(value) for value in (facts.get("parties") or [])][:2]
            if parties:
                parts.append(f"parties={', '.join(parties)}")
            detail = "; ".join(parts) if parts else "semantic evidence available in cited excerpts"
            lines.append(f"- {filename}: {detail}.")
        lines.append(f"Evidence score: {evidence_score:.2f}.")
        if did_retry:
            lines.append("A single bounded semantic retry was used to improve evidence coverage.")
        if scope_relaxed:
            lines.append("Metadata scope was relaxed because strict candidates were empty.")
        return "\n".join(lines)

    def _filter_documents_by_metadata(
        self,
        docs: list[dict],
        *,
        operation: str,
        filters: dict[str, Any],
    ) -> list[dict]:
        return self.metadata_semantic_adapter.filter_documents(
            docs,
            operation=operation,
            filters=filters,
            parse_datetime=self._parse_datetime,
            doc_created_at=self._doc_created_at,
            doc_updated_at=self._doc_updated_at,
            doc_matches_doc_type_filter=self._doc_matches_doc_type_filter,
            doc_matches_target_document=self._doc_matches_target_document,
            now=datetime.now(timezone.utc),
        )

    def _semantic_fallback_for_empty_metadata_result(
        self,
        question: str,
        *,
        doc_ids: Optional[List[str]],
    ) -> tuple[str, list[dict]]:
        chunks: list[dict] = []
        try:
            scoped = self._select_candidate_docs_for_query(
                question,
                doc_ids=doc_ids,
                require_multi_doc=False,
            )
            scoped_doc_ids = scoped["doc_ids"] if scoped else doc_ids
            chunks = self.retrieval.search_balanced(
                question,
                top_k=12,
                doc_ids=scoped_doc_ids,
                per_doc_limit=3,
                mode="hybrid",
            )
            if self._retrieval_confidence_low(question, chunks, min_docs=1):
                global_chunks = self.retrieval.search(
                    question,
                    top_k=12,
                    doc_ids=doc_ids,
                    mode="hybrid",
                )
                chunks = self._merge_chunks(chunks + global_chunks, limit=16)
        except Exception as exc:
            logger.debug("Semantic fallback for empty metadata result failed: %s", exc)
            chunks = []
        if not chunks:
            return "", []
        answer = self._fallback_answer(
            question,
            chunks,
            intent="qa",
            include_document_summaries=False,
        )
        if not self._has_weak_evidence_prefix(answer):
            answer = (
                "I could not satisfy the strict metadata filters, but I found evidence in document content:\n"
                + answer
            )
        return answer, self._prepare_response_sources(chunks, intent="qa")

    def _answer_metadata_query(
        self,
        question: str,
        *,
        doc_ids: Optional[List[str]],
        metadata_signal: dict[str, Any],
    ) -> tuple[str, list[dict]]:
        operation = str(metadata_signal.get("operation") or "list").strip().lower()
        if operation not in METADATA_OPERATIONS:
            operation = "list"
        expected_answer_type = str(metadata_signal.get("expected_answer_type") or "").strip().lower()
        if expected_answer_type not in EXPECTED_ANSWER_TYPES:
            expected_answer_type = self._metadata_answer_type_for_operation(operation)
        filters = dict(metadata_signal.get("filters") or {}) if isinstance(metadata_signal.get("filters"), dict) else {}
        all_docs = storage.list_documents()
        if doc_ids:
            scoped_ids = {str(doc_id or "").strip() for doc_id in doc_ids if str(doc_id or "").strip()}
            docs = [doc for doc in all_docs if str(doc.get("id") or "").strip() in scoped_ids]
        else:
            docs = list(all_docs)
        if not docs:
            return "I cannot find documents in the current scope.", []

        if expected_answer_type == "count":
            operation = "count"
        elif expected_answer_type == "person" and operation not in PERSON_METADATA_OPERATIONS:
            operation = self._metadata_operation_for_answer_type(expected_answer_type, filters)

        if operation == "external_collaborators":
            filters["collaborator_type"] = "external"

        if operation == "authored_by" and not filters.get("author") and filters.get("target_document"):
            filters["author"] = str(filters.get("target_document") or "").strip()
        if operation == "edited_by" and not filters.get("last_modified_by") and filters.get("target_document"):
            filters["last_modified_by"] = str(filters.get("target_document") or "").strip()

        filtered_docs = self._filter_documents_by_metadata(docs, operation=operation, filters=filters)
        has_filters = bool(filters)
        scoped_docs = filtered_docs if has_filters else docs
        if has_filters and not filtered_docs:
            fallback_answer, fallback_sources = self._semantic_fallback_for_empty_metadata_result(
                question,
                doc_ids=doc_ids,
            )
            if fallback_answer and fallback_sources:
                return fallback_answer, fallback_sources

        if operation == "latest_uploaded":
            ranked = sorted(
                scoped_docs,
                key=lambda item: self._doc_created_at(item) or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True,
            )
            target = ranked[0] if ranked else None
            if not target:
                return "I cannot determine the most recently uploaded document.", []
            created = self._doc_created_at(target)
            answer = (
                f"The most recently uploaded document is '{target.get('filename')}' "
                f"(uploaded at {created.isoformat() if created else 'unknown'})."
            )
            return answer, self._metadata_sources_from_documents([target])

        if operation == "earliest_uploaded":
            ranked = sorted(
                scoped_docs,
                key=lambda item: self._doc_created_at(item) or datetime.max.replace(tzinfo=timezone.utc),
            )
            target = ranked[0] if ranked else None
            if not target:
                return "I cannot determine the first uploaded document.", []
            created = self._doc_created_at(target)
            answer = (
                f"The first uploaded document is '{target.get('filename')}' "
                f"(uploaded at {created.isoformat() if created else 'unknown'})."
            )
            return answer, self._metadata_sources_from_documents([target])

        if operation == "most_frequently_updated":
            grouped: dict[str, list[dict]] = defaultdict(list)
            for doc in scoped_docs:
                metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
                key = str(metadata.get("logical_document_key") or doc.get("filename") or "unknown").strip().lower()
                grouped[key].append(doc)
            if not grouped:
                return "I cannot determine update frequency from available metadata.", []
            ranked_groups = sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True)
            best_key, best_docs = ranked_groups[0]
            best_docs = sorted(best_docs, key=lambda item: self._doc_created_at(item) or datetime.min.replace(tzinfo=timezone.utc))
            answer = (
                f"The most frequently updated document lineage is '{best_key}' with "
                f"{len(best_docs)} version(s). Latest: {best_docs[-1].get('filename')}."
            )
            return answer, self._metadata_sources_from_documents(best_docs[-8:])

        if operation == "changed_between_versions":
            target_name = str(filters.get("target_document") or "").strip().lower()
            version_a = filters.get("version_a")
            version_b = filters.get("version_b")
            groups: dict[str, list[dict]] = defaultdict(list)
            for doc in scoped_docs:
                metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
                key = str(metadata.get("logical_document_key") or doc.get("filename") or "unknown").strip().lower()
                groups[key].append(doc)
            if target_name:
                groups = {
                    key: value
                    for key, value in groups.items()
                    if target_name in key or any(target_name in str(doc.get("filename") or "").lower() for doc in value)
                }
            if not groups:
                return "I cannot find a document lineage matching the requested versions.", []
            key, candidates = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)[0]
            ordered = sorted(
                candidates,
                key=lambda doc: int(
                    (doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}).get("version_index") or 0
                ),
            )
            if len(ordered) < 2:
                return f"I found '{key}', but it has only one version.", self._metadata_sources_from_documents(ordered)
            doc_a = ordered[0]
            doc_b = ordered[-1]
            if version_a is not None:
                for candidate in ordered:
                    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
                    try:
                        if int(metadata.get("version_index") or 0) == int(version_a):
                            doc_a = candidate
                            break
                    except Exception:
                        continue
            if version_b is not None:
                for candidate in ordered:
                    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
                    try:
                        if int(metadata.get("version_index") or 0) == int(version_b):
                            doc_b = candidate
                            break
                    except Exception:
                        continue
            md_a = doc_a.get("metadata") if isinstance(doc_a.get("metadata"), dict) else {}
            md_b = doc_b.get("metadata") if isinstance(doc_b.get("metadata"), dict) else {}
            differences: list[str] = []
            if str(md_a.get("checksum") or "") != str(md_b.get("checksum") or ""):
                differences.append("content checksum changed")
            if int(md_a.get("size_bytes") or 0) != int(md_b.get("size_bytes") or 0):
                differences.append(
                    f"size changed ({int(md_a.get('size_bytes') or 0)} -> {int(md_b.get('size_bytes') or 0)} bytes)"
                )
            if str(md_a.get("last_modified_by") or "") != str(md_b.get("last_modified_by") or ""):
                differences.append(
                    f"last_modified_by changed ({md_a.get('last_modified_by') or 'unknown'} -> {md_b.get('last_modified_by') or 'unknown'})"
                )
            if str(md_a.get("author") or "") != str(md_b.get("author") or ""):
                differences.append(f"author changed ({md_a.get('author') or 'unknown'} -> {md_b.get('author') or 'unknown'})")
            if not differences:
                differences.append("no tracked metadata differences detected")
            version_a_label = str(md_a.get("version_label") or f"v{md_a.get('version_index') or '?'}")
            version_b_label = str(md_b.get("version_label") or f"v{md_b.get('version_index') or '?'}")
            answer = (
                f"Changes between {version_a_label} and {version_b_label} for '{key}': "
                + "; ".join(differences)
                + "."
            )
            return answer, self._metadata_sources_from_documents([doc_a, doc_b])

        if operation == "last_modified_by":
            question_target_hint = self._infer_target_document_hint(question, docs)
            if question_target_hint:
                hinted_docs = [doc for doc in docs if self._doc_matches_target_document(doc, question_target_hint)]
                if len(hinted_docs) > 1:
                    ranked_hinted = self._rank_documents_for_target(question_target_hint, hinted_docs)
                    top_candidates = ranked_hinted[:4]
                    names = [str(doc.get("filename") or doc.get("id") or "unknown") for doc in top_candidates]
                    answer = (
                        f"I found multiple documents matching '{question_target_hint}'. "
                        f"Please specify one: {', '.join(names)}."
                    )
                    return answer, self._metadata_sources_from_documents(top_candidates)
            target_document = str(filters.get("target_document") or "").strip()
            if target_document:
                ranked_matches = self._rank_documents_for_target(target_document, scoped_docs)
                if len(ranked_matches) > 1:
                    top_candidates = ranked_matches[:4]
                    names = [str(doc.get("filename") or doc.get("id") or "unknown") for doc in top_candidates]
                    answer = (
                        f"I found multiple documents matching '{target_document}'. "
                        f"Please specify one: {', '.join(names)}."
                    )
                    return answer, self._metadata_sources_from_documents(top_candidates)
                if ranked_matches:
                    scoped_docs = ranked_matches[:1]
            ranked = sorted(
                scoped_docs,
                key=lambda item: self._doc_updated_at(item) or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True,
            )
            target = ranked[0] if ranked else None
            if not target:
                return "I cannot determine the latest modifier for the requested document.", []
            metadata = target.get("metadata") if isinstance(target.get("metadata"), dict) else {}
            editor = str(metadata.get("last_modified_by") or metadata.get("uploaded_by_name") or metadata.get("uploaded_by_email") or "unknown")
            answer = f"The latest recorded modifier for '{target.get('filename')}' is {editor}."
            return answer, self._metadata_sources_from_documents([target])

        if operation == "count":
            count = len(scoped_docs)
            qualifier_parts = [
                f"{key}={value}"
                for key, value in filters.items()
                if key in {"doc_type", "author", "last_modified_by", "uploader_role", "collaborator_type", "target_document"}
                and str(value).strip()
            ][:3]
            qualifier = f" matching {'; '.join(qualifier_parts)}" if qualifier_parts else ""
            answer = f"There are {count} document(s){qualifier} in scope."
            return answer, self._metadata_sources_from_documents(scoped_docs)

        listed = scoped_docs
        if operation in {"created_after", "created_before", "created_between", "modified_within_days", "uploaded_by_role"}:
            answer = f"I found {len(listed)} document(s) matching the requested metadata filter."
            return answer, self._metadata_sources_from_documents(listed)
        if operation in {"authored_by", "edited_by", "external_collaborators"}:
            answer = f"I found {len(listed)} document(s) matching the requested collaborator/author filter."
            return answer, self._metadata_sources_from_documents(listed)
        answer = f"I found {len(listed)} document(s) in the current metadata query scope."
        return answer, self._metadata_sources_from_documents(listed)

    def _tokenize_for_match(self, text: str) -> set[str]:
        if not text:
            return set()
        normalized: set[str] = set()
        for raw in TOKEN_RE.findall(str(text)):
            token = raw.lower().strip("._-")
            if len(token) < 3 or token in QUERY_STOPWORDS:
                continue
            if token.endswith("ies") and len(token) >= 5:
                token = f"{token[:-3]}y"
            elif token.endswith("s") and len(token) >= 5 and not token.endswith("ss"):
                token = token[:-1]
            if len(token) >= 3 and token not in QUERY_STOPWORDS:
                normalized.add(token)
        return normalized

    def _extract_query_entities(self, question: str) -> list[str]:
        text = str(question or "").strip()
        if not text:
            return []

        entities: list[str] = []
        seen: set[str] = set()

        def add_entity(raw: str) -> None:
            cleaned = re.sub(r"\s+", " ", str(raw or "").strip(" \t\r\n\"'`.,:;!?()[]{}"))
            if len(cleaned) < 2:
                return
            lowered = cleaned.lower()
            if lowered in seen or lowered in QUERY_STOPWORDS:
                return
            seen.add(lowered)
            entities.append(cleaned)

        for phrase in re.findall(r"[\"'`]([^\"'`]{2,80})[\"'`]", text):
            add_entity(phrase)

        for match in re.finditer(r"\b(?:[A-Z][A-Za-z0-9&.\-]*\s+){0,3}[A-Z][A-Za-z0-9&.\-]*\b", text):
            value = re.sub(r"\s+", " ", match.group(0)).strip()
            if value and value.lower() not in QUERY_STOPWORDS:
                add_entity(value)

        for token in re.findall(r"[A-Za-z][A-Za-z0-9&.\-]{2,}", text):
            if token[0].isupper() or token.isupper() or len(token) >= 6:
                add_entity(token)

        return entities[:12]

    def _document_match_text(self, doc: dict[str, Any]) -> str:
        parts = [str(doc.get("filename") or "")]
        metadata = doc.get("metadata")
        if isinstance(metadata, dict):
            for key in ("title", "name", "subject", "customer", "client", "vendor", "project", "summary"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
            extras = [value for value in metadata.values() if isinstance(value, str) and value.strip()]
            parts.extend(extras[:4])
        return " ".join(parts).lower()

    def _get_doc_auto_tags(
        self,
        doc_id: str,
        doc_record: dict[str, Any],
        *,
        include_content_fallback: bool = False,
    ) -> list[str]:
        if include_content_fallback:
            cached = self._doc_tag_cache.get(doc_id)
            if cached:
                return list(cached)

        metadata = doc_record.get("metadata")
        raw_tags = []
        if isinstance(metadata, dict):
            raw_tags = metadata.get("auto_tags") or metadata.get("doc_auto_tags") or []
        tags = normalize_tags(raw_tags if isinstance(raw_tags, list) else [], limit=32)
        if tags:
            if len(self._doc_tag_cache) < 6000:
                self._doc_tag_cache[doc_id] = list(tags)
            return tags

        filename = str(doc_record.get("filename") or doc_record.get("doc_filename") or "")
        if include_content_fallback:
            content_samples = [
                str(chunk.get("content") or "")
                for chunk in storage.get_chunks_by_doc(doc_id)[:20]
                if str(chunk.get("content") or "").strip()
            ]
            tags = build_document_auto_tags(filename, content_samples, limit=24)
        else:
            tags = build_document_auto_tags(filename, [], limit=16)
        if include_content_fallback and tags and len(self._doc_tag_cache) < 6000:
            self._doc_tag_cache[doc_id] = list(tags)
        return tags

    def _query_terms_for_tag_scoring(self, question: str, entities: list[str]) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()

        def add(raw: str) -> None:
            value = re.sub(r"\s+", " ", str(raw or "").strip().lower())
            if len(value) < 2 or value in seen:
                return
            seen.add(value)
            terms.append(value)

        for entity in entities[:8]:
            add(entity)
        for token in sorted(self._tokenize_for_match(question), key=len, reverse=True)[:14]:
            add(token)
        return terms

    def _semantic_tag_scores(
        self,
        query_text: str,
        doc_tags_by_id: dict[str, list[str]],
    ) -> dict[str, float]:
        embedder = getattr(self.retrieval, "embedder", None)
        if embedder is None:
            return {}

        unique_tags: list[str] = []
        seen: set[str] = set()
        for tags in doc_tags_by_id.values():
            for tag in tags:
                key = str(tag or "").strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    unique_tags.append(key)
        if not unique_tags:
            return {}

        if len(self._tag_embedding_cache) > 6000:
            self._tag_embedding_cache.clear()

        missing = [tag for tag in unique_tags if tag not in self._tag_embedding_cache]
        try:
            if missing:
                vectors = embedder.embed_texts(missing)
                for idx, tag in enumerate(missing):
                    self._tag_embedding_cache[tag] = vectors[idx]
            query_vec = embedder.embed_query(query_text)
        except Exception as exc:
            logger.debug("Tag embedding scoring failed: %s", exc)
            return {}

        semantic_scores: dict[str, float] = {}
        for doc_id, tags in doc_tags_by_id.items():
            best = 0.0
            for tag in tags:
                vec = self._tag_embedding_cache.get(str(tag or "").strip().lower())
                if vec is None:
                    continue
                similarity = float(np.dot(query_vec, vec))
                if similarity > best:
                    best = similarity
            if best > 0.0:
                semantic_scores[doc_id] = best
        return semantic_scores

    def _score_docs_by_query_tags(
        self,
        question: str,
        entities: list[str],
        doc_tags_by_id: dict[str, list[str]],
    ) -> dict[str, Any]:
        if not doc_tags_by_id:
            return {"scores": {}, "ranked_doc_ids": [], "high_confidence": False, "top_tags": []}

        query_terms = self._tokenize_for_match(question)
        for entity in entities:
            query_terms.update(self._tokenize_for_match(entity))
        doc_term_sets: dict[str, set[str]] = {}
        lexical_scores: dict[str, float] = {}
        for doc_id, tags in doc_tags_by_id.items():
            doc_terms: set[str] = set()
            for tag in tags:
                doc_terms.update(self._tokenize_for_match(tag))
            doc_term_sets[doc_id] = doc_terms

        if query_terms and doc_term_sets:
            term_doc_freq: Counter[str] = Counter()
            for term in query_terms:
                for doc_terms in doc_term_sets.values():
                    if term in doc_terms:
                        term_doc_freq[term] += 1
            term_weights: dict[str, float] = {}
            total_weight = 0.0
            for term in query_terms:
                doc_freq = int(term_doc_freq.get(term, 0))
                if doc_freq <= 0:
                    continue
                weight = 1.0 / float(1.0 + doc_freq)
                if term in LOW_SIGNAL_TERMS:
                    weight *= 0.4
                term_weights[term] = weight
                total_weight += weight
            total_weight = max(total_weight, 1e-8)
            for doc_id, doc_terms in doc_term_sets.items():
                if not doc_terms:
                    lexical_scores[doc_id] = 0.0
                    continue
                hit_weight = sum(term_weights.get(term, 0.0) for term in query_terms if term in doc_terms)
                lexical_scores[doc_id] = hit_weight / total_weight
        else:
            lexical_scores = {doc_id: 0.0 for doc_id in doc_tags_by_id}

        query_tag_terms = self._query_terms_for_tag_scoring(question, entities)
        semantic_scores = self._semantic_tag_scores(" ".join(query_tag_terms) or question, doc_tags_by_id)

        scores: dict[str, float] = {}
        for doc_id in doc_tags_by_id:
            lex = max(0.0, float(lexical_scores.get(doc_id, 0.0)))
            sem = max(0.0, float(semantic_scores.get(doc_id, 0.0)))
            scores[doc_id] = (0.55 * lex) + (0.75 * sem)

        ranked_doc_ids = sorted(doc_tags_by_id.keys(), key=lambda doc_id: scores.get(doc_id, 0.0), reverse=True)
        best = float(scores.get(ranked_doc_ids[0], 0.0)) if ranked_doc_ids else 0.0
        second = float(scores.get(ranked_doc_ids[1], 0.0)) if len(ranked_doc_ids) > 1 else 0.0
        best_doc = ranked_doc_ids[0] if ranked_doc_ids else ""
        second_doc = ranked_doc_ids[1] if len(ranked_doc_ids) > 1 else ""
        best_lex = float(lexical_scores.get(best_doc, 0.0)) if best_doc else 0.0
        second_lex = float(lexical_scores.get(second_doc, 0.0)) if second_doc else 0.0
        best_sem = float(semantic_scores.get(best_doc, 0.0)) if best_doc else 0.0
        high_confidence = bool(
            (best >= 0.72 and (best_lex >= 0.15 or best_sem >= 0.60))
            or (best >= 0.52 and (best - second) >= 0.07 and (best_lex >= 0.12 or best_sem >= 0.56))
            or (best_lex >= 0.20 and (best_lex - second_lex) >= 0.08)
        )

        top_tags: list[str] = []
        if ranked_doc_ids:
            tag_counter: Counter[str] = Counter()
            for doc_id in ranked_doc_ids[:3]:
                for tag in doc_tags_by_id.get(doc_id, [])[:12]:
                    overlap = len(self._tokenize_for_match(tag).intersection(query_terms))
                    if overlap > 0:
                        tag_counter[tag] += overlap
            top_tags = [tag for tag, _ in tag_counter.most_common(8)]

        return {
            "scores": scores,
            "ranked_doc_ids": ranked_doc_ids,
            "high_confidence": high_confidence,
            "top_tags": top_tags,
        }

    def _chunk_query_overlap(
        self,
        question: str,
        content: str,
        entities: Optional[list[str]] = None,
    ) -> float:
        query_terms = self._tokenize_for_match(question)
        for entity in entities or []:
            query_terms.update(self._tokenize_for_match(entity))
        if not query_terms:
            return 0.0
        content_terms = self._tokenize_for_match(content)
        if not content_terms:
            return 0.0
        overlap = len(query_terms.intersection(content_terms))
        return overlap / float(max(1, len(query_terms)))

    def _selected_or_ready_doc_ids(self, doc_ids: Optional[List[str]]) -> list[str]:
        if doc_ids:
            ordered: list[str] = []
            seen: set[str] = set()
            for doc_id in doc_ids:
                raw = str(doc_id or "").strip()
                if raw and raw not in seen:
                    seen.add(raw)
                    ordered.append(raw)
            return ordered
        ready = [
            str(doc.get("id") or "").strip()
            for doc in storage.list_documents()
            if str(doc.get("status") or "").strip().lower() == "ready"
        ]
        return [doc_id for doc_id in ready if doc_id]

    def _select_candidate_docs_for_query(
        self,
        question: str,
        doc_ids: Optional[List[str]],
        require_multi_doc: bool,
        prefer_recall: bool = False,
    ) -> Optional[dict[str, Any]]:
        available_doc_ids = self._selected_or_ready_doc_ids(doc_ids)
        doc_type_candidates = extract_query_doc_type_candidates(question)
        if len(available_doc_ids) <= 4 and not doc_type_candidates:
            return None

        entities = self._extract_query_entities(question)
        query_terms = self._tokenize_for_match(question)
        if not entities and not query_terms:
            return None

        doc_records = {str(doc.get("id") or "").strip(): doc for doc in storage.list_documents()}
        doc_type_scoped = False
        if doc_type_candidates:
            typed_doc_ids: list[str] = []
            for doc_id in available_doc_ids:
                doc_record = doc_records.get(doc_id) or {"id": doc_id, "filename": ""}
                if any(self._doc_matches_doc_type_filter(doc_record, candidate) for candidate in doc_type_candidates):
                    typed_doc_ids.append(doc_id)
            min_type_docs = 2 if require_multi_doc else 1
            if len(typed_doc_ids) >= min_type_docs and len(typed_doc_ids) < len(available_doc_ids):
                available_doc_ids = typed_doc_ids
                doc_type_scoped = True

        doc_text_by_id: dict[str, str] = {}
        doc_tags_by_id: dict[str, list[str]] = {}
        filename_terms_by_id: dict[str, set[str]] = {}
        for doc_id in available_doc_ids:
            doc_record = doc_records.get(doc_id) or {"id": doc_id, "filename": ""}
            doc_text_by_id[doc_id] = self._document_match_text(doc_record)
            filename_terms_by_id[doc_id] = self._tokenize_for_match(str(doc_record.get("filename") or ""))
            doc_tags_by_id[doc_id] = self._get_doc_auto_tags(
                doc_id,
                doc_record,
                include_content_fallback=False,
            )

        tag_signal = self._score_docs_by_query_tags(question, entities, doc_tags_by_id)
        tag_scores = tag_signal.get("scores", {})
        if not bool(tag_signal.get("high_confidence")):
            ranked_seed = list(tag_signal.get("ranked_doc_ids", [])) or list(available_doc_ids)
            enrich_candidates = ranked_seed[: min(len(ranked_seed), 18)]
            enriched = False
            for doc_id in enrich_candidates:
                doc_record = doc_records.get(doc_id) or {}
                metadata = doc_record.get("metadata")
                has_stored_tags = bool(
                    isinstance(metadata, dict)
                    and isinstance(metadata.get("auto_tags"), list)
                    and metadata.get("auto_tags")
                )
                if has_stored_tags:
                    continue
                rich_tags = self._get_doc_auto_tags(
                    doc_id,
                    doc_record,
                    include_content_fallback=True,
                )
                if rich_tags and rich_tags != doc_tags_by_id.get(doc_id, []):
                    doc_tags_by_id[doc_id] = rich_tags
                    enriched = True
            if enriched:
                tag_signal = self._score_docs_by_query_tags(question, entities, doc_tags_by_id)
                tag_scores = tag_signal.get("scores", {})

        candidate_pool = list(available_doc_ids)
        if bool(tag_signal.get("high_confidence")):
            ranked_by_tag = [
                doc_id
                for doc_id in tag_signal.get("ranked_doc_ids", [])
                if float(tag_scores.get(doc_id, 0.0)) > 0.0
            ]
            pool_cap = min(
                len(available_doc_ids),
                max(8 if require_multi_doc else 6, min(18, max(6, len(available_doc_ids) // 2))),
            )
            narrowed = ranked_by_tag[:pool_cap]
            if len(narrowed) >= (2 if require_multi_doc else 1):
                candidate_pool = narrowed

        filename_anchor_confident = False
        filename_anchor_docs: list[str] = []
        anchor_score_best = 0.0
        for term in query_terms:
            if term in LOW_SIGNAL_TERMS:
                continue
            matched_docs = [doc_id for doc_id in available_doc_ids if term in filename_terms_by_id.get(doc_id, set())]
            freq = len(matched_docs)
            if freq <= 0 or freq >= len(available_doc_ids):
                continue
            specificity = (1.0 - (float(freq) / float(len(available_doc_ids)))) * (min(len(term), 12) / 12.0)
            if specificity > anchor_score_best:
                anchor_score_best = specificity
                filename_anchor_docs = matched_docs
        min_anchor_docs = 2 if require_multi_doc else 1
        if filename_anchor_docs and anchor_score_best >= 0.32 and len(filename_anchor_docs) >= min_anchor_docs:
            narrowed = [doc_id for doc_id in candidate_pool if doc_id in set(filename_anchor_docs)]
            if len(narrowed) >= min_anchor_docs:
                candidate_pool = narrowed
                filename_anchor_confident = True

        doc_scores: defaultdict[str, float] = defaultdict(float)
        for doc_id in available_doc_ids:
            doc_scores[doc_id] += 2.4 * float(tag_scores.get(doc_id, 0.0))

        filename_term_freq: Counter[str] = Counter()
        for term in query_terms:
            for doc_id in available_doc_ids:
                if term in filename_terms_by_id.get(doc_id, set()):
                    filename_term_freq[term] += 1
        for doc_id in available_doc_ids:
            for term in query_terms:
                if term not in filename_terms_by_id.get(doc_id, set()):
                    continue
                doc_freq = int(filename_term_freq.get(term, 0))
                if doc_freq <= 0:
                    continue
                doc_scores[doc_id] += 2.0 * (1.0 / float(1.0 + doc_freq))
            if filename_anchor_confident and doc_id in filename_anchor_docs:
                doc_scores[doc_id] += 2.2

        for doc_id in available_doc_ids:
            doc_text = doc_text_by_id.get(doc_id, "")
            for term in entities:
                lowered = term.lower()
                if lowered and lowered in doc_text:
                    doc_scores[doc_id] += 1.4 if " " in lowered else 1.1

        expansion_terms = [str(tag) for tag in tag_signal.get("top_tags", [])][:6]
        seed_terms = entities[:6] + expansion_terms
        if not seed_terms:
            seed_terms = sorted(query_terms, key=len, reverse=True)[:6]
        seed_query = " ".join(seed_terms).strip() or question
        expanded_question = f"{question} {' '.join(expansion_terms)}".strip()
        candidate_k = max(24, min(96, len(candidate_pool) * 4))
        sparse_hits = self.retrieval.search(
            seed_query,
            top_k=candidate_k,
            doc_ids=candidate_pool,
            mode="sparse",
            use_rerank=False,
        )
        hybrid_hits = self.retrieval.search(
            expanded_question,
            top_k=candidate_k,
            doc_ids=candidate_pool,
            mode="hybrid",
            use_rerank=False,
        )

        for rank, chunk in enumerate(sparse_hits):
            doc_id = str(chunk.get("doc_id") or "").strip()
            if not doc_id:
                continue
            weight = 1.0 if not bool(tag_signal.get("high_confidence")) else max(
                0.25,
                min(1.0, float(tag_scores.get(doc_id, 0.0)) + 0.15),
            )
            doc_scores[doc_id] += weight * (2.0 / float(1.0 + (0.35 * rank)))
            doc_scores[doc_id] += weight * 1.1 * self._chunk_query_overlap(
                question,
                str(chunk.get("content") or ""),
                entities=entities,
            )

        for rank, chunk in enumerate(hybrid_hits):
            doc_id = str(chunk.get("doc_id") or "").strip()
            if not doc_id:
                continue
            weight = 1.0 if not bool(tag_signal.get("high_confidence")) else max(
                0.25,
                min(1.0, float(tag_scores.get(doc_id, 0.0)) + 0.15),
            )
            doc_scores[doc_id] += weight * (1.25 / float(1.0 + (0.4 * rank)))
            doc_scores[doc_id] += weight * 0.7 * self._chunk_query_overlap(
                question,
                str(chunk.get("content") or ""),
                entities=entities,
            )

        ranked_doc_ids = sorted(
            candidate_pool,
            key=lambda current_doc_id: doc_scores.get(current_doc_id, 0.0),
            reverse=True,
        )
        if not ranked_doc_ids:
            return None

        top_score = float(doc_scores.get(ranked_doc_ids[0], 0.0))
        if top_score <= 0.0:
            return None

        if bool(tag_signal.get("high_confidence")):
            max_docs = min(
                len(candidate_pool),
                max(4 if require_multi_doc else 3, len(entities) + 2),
            )
            cutoff = top_score * (0.45 if require_multi_doc else 0.40)
        else:
            max_docs = min(
                len(candidate_pool),
                max(6 if require_multi_doc else 5, len(entities) + 3),
            )
            cutoff = top_score * (0.35 if require_multi_doc else 0.25)
        scoped_doc_ids = [
            current_doc_id
            for current_doc_id in ranked_doc_ids
            if doc_scores.get(current_doc_id, 0.0) >= cutoff
        ][:max_docs]

        min_docs = 2 if require_multi_doc else 1
        if len(scoped_doc_ids) < min_docs:
            scoped_doc_ids = ranked_doc_ids[: max(min_docs, min(max_docs, 4))]

        if prefer_recall and len(scoped_doc_ids) < len(ranked_doc_ids):
            recall_cap = min(
                len(ranked_doc_ids),
                max(
                    len(scoped_doc_ids),
                    max(10 if require_multi_doc else 8, len(entities) + 4),
                ),
            )
            scoped_doc_ids = ranked_doc_ids[:recall_cap]

        if (
            not bool(tag_signal.get("high_confidence"))
            and not filename_anchor_confident
            and len(scoped_doc_ids) >= len(available_doc_ids)
            and not doc_type_scoped
        ):
            return None

        confident = bool(
            (bool(tag_signal.get("high_confidence")) or filename_anchor_confident)
            and len(scoped_doc_ids) <= max(6, len(available_doc_ids) // 2)
        )
        return {
            "doc_ids": scoped_doc_ids,
            "entities": entities,
            "top_tags": expansion_terms,
            "tag_confident": bool(tag_signal.get("high_confidence")),
            "confident": confident,
            "doc_type_candidates": doc_type_candidates,
            "doc_type_scoped": doc_type_scoped,
        }

    def _build_analysis_plan(
        self,
        question: str,
        doc_ids: Optional[List[str]],
        router_analysis_plan: Any = None,
        require_multi_doc: bool = True,
    ) -> dict[str, Any]:
        available_doc_ids = self._selected_or_ready_doc_ids(doc_ids)
        min_docs = 2 if require_multi_doc else 1
        if not available_doc_ids:
            return {
                "query_entities": self._extract_query_entities(question),
                "candidate_doc_ids": [],
                "evidence_classes": [],
                "tag_confident": False,
            }

        entities: list[str] = []
        seen_entities: set[str] = set()

        def add_entity(value: Any) -> None:
            cleaned = re.sub(r"\s+", " ", str(value or "").strip())
            key = cleaned.lower()
            if len(cleaned) < 2 or key in seen_entities or key in QUERY_STOPWORDS:
                return
            seen_entities.add(key)
            entities.append(cleaned)

        if isinstance(router_analysis_plan, dict):
            for value in (router_analysis_plan.get("query_entities") or [])[:12]:
                add_entity(value)
            for value in (router_analysis_plan.get("evidence_classes") or [])[:8]:
                if isinstance(value, dict):
                    add_entity(value.get("label") or value.get("name"))
                else:
                    add_entity(value)
        for value in self._extract_query_entities(question):
            add_entity(value)

        doc_records = {str(doc.get("id") or "").strip(): doc for doc in storage.list_documents()}
        doc_tags_by_id: dict[str, list[str]] = {}
        for doc_id in available_doc_ids:
            doc_record = doc_records.get(doc_id) or {"id": doc_id, "filename": ""}
            tags = self._get_doc_auto_tags(
                doc_id,
                doc_record,
                include_content_fallback=False,
            )
            if not tags:
                tags = self._get_doc_auto_tags(
                    doc_id,
                    doc_record,
                    include_content_fallback=True,
                )
            doc_tags_by_id[doc_id] = tags

        tag_signal = self._score_docs_by_query_tags(question, entities, doc_tags_by_id)
        ranked_doc_ids = [doc_id for doc_id in tag_signal.get("ranked_doc_ids", []) if doc_id in set(available_doc_ids)]
        score_map = tag_signal.get("scores", {})
        candidate_doc_ids: list[str] = []
        if ranked_doc_ids:
            best_score = max(0.0, float(score_map.get(ranked_doc_ids[0], 0.0)))
            if bool(tag_signal.get("high_confidence")):
                cutoff = best_score * 0.42
                max_docs = min(len(ranked_doc_ids), max(8 if require_multi_doc else 6, len(entities) + 3))
            else:
                cutoff = best_score * 0.25
                max_docs = min(len(ranked_doc_ids), max(12 if require_multi_doc else 8, len(entities) + 4))
            candidate_doc_ids = [
                current_doc_id
                for current_doc_id in ranked_doc_ids
                if float(score_map.get(current_doc_id, 0.0)) >= cutoff
            ][:max_docs]
        if len(candidate_doc_ids) < min_docs:
            fallback_limit = max(min_docs, min(10 if require_multi_doc else 6, len(available_doc_ids)))
            candidate_doc_ids = ranked_doc_ids[:fallback_limit] if ranked_doc_ids else available_doc_ids[:fallback_limit]

        evidence_classes = self._derive_evidence_classes_from_tags(
            question,
            entities,
            candidate_doc_ids,
            doc_tags_by_id,
            max_classes=4,
        )
        if not evidence_classes and candidate_doc_ids:
            fallback_label = " ".join(sorted(self._tokenize_for_match(question), key=len, reverse=True)[:3]) or "analysis"
            evidence_classes = [{"label": fallback_label, "doc_ids": candidate_doc_ids[:8], "score": 0.0}]

        return {
            "query_entities": entities[:12],
            "candidate_doc_ids": candidate_doc_ids,
            "evidence_classes": evidence_classes,
            "tag_confident": bool(tag_signal.get("high_confidence")),
        }

    def _derive_evidence_classes_from_tags(
        self,
        question: str,
        entities: list[str],
        candidate_doc_ids: list[str],
        doc_tags_by_id: dict[str, list[str]],
        max_classes: int = 4,
    ) -> list[dict[str, Any]]:
        if not candidate_doc_ids:
            return []

        tag_docs: dict[str, set[str]] = defaultdict(set)
        display_label: dict[str, str] = {}
        tag_terms_by_norm: dict[str, set[str]] = {}
        for doc_id in candidate_doc_ids:
            for raw_tag in doc_tags_by_id.get(doc_id, []):
                tag_text = re.sub(r"\s+", " ", str(raw_tag or "").strip())
                tag_norm = tag_text.lower()
                if len(tag_norm) < 2:
                    continue
                tag_docs[tag_norm].add(doc_id)
                display_label.setdefault(tag_norm, tag_text)
                if tag_norm not in tag_terms_by_norm:
                    tag_terms_by_norm[tag_norm] = self._tokenize_for_match(tag_norm)

        if not tag_docs:
            return []

        query_terms = self._tokenize_for_match(question)
        for entity in entities[:12]:
            query_terms.update(self._tokenize_for_match(entity))
        if not query_terms:
            return []

        term_doc_freq: Counter[str] = Counter()
        for term in query_terms:
            for tag_norm, tag_terms in tag_terms_by_norm.items():
                if term in tag_terms:
                    term_doc_freq[term] += 1

        term_weights: dict[str, float] = {}
        total_weight = 0.0
        for term in query_terms:
            doc_freq = int(term_doc_freq.get(term, 0))
            if doc_freq <= 0:
                continue
            weight = 1.0 / float(1.0 + doc_freq)
            if term in LOW_SIGNAL_TERMS:
                weight *= 0.45
            term_weights[term] = weight
            total_weight += weight
        total_weight = max(total_weight, 1e-8)

        tag_list = list(tag_docs.keys())
        semantic_input = {f"tag::{idx}": [tag] for idx, tag in enumerate(tag_list)}
        semantic_scores = self._semantic_tag_scores(" ".join(entities[:8]) + " " + question, semantic_input)

        scored_tags: list[tuple[str, float]] = []
        denom_docs = float(max(1, len(candidate_doc_ids)))
        for idx, tag_norm in enumerate(tag_list):
            tag_terms = tag_terms_by_norm.get(tag_norm, set())
            lexical = 0.0
            if tag_terms and term_weights:
                lexical = sum(weight for term, weight in term_weights.items() if term in tag_terms) / total_weight
            semantic = max(0.0, float(semantic_scores.get(f"tag::{idx}", 0.0)))
            support = float(len(tag_docs.get(tag_norm, set()))) / denom_docs
            score = (0.62 * lexical) + (0.64 * semantic) + (0.18 * support)
            if tag_terms and all(term in LOW_SIGNAL_TERMS for term in tag_terms):
                score *= 0.55
            scored_tags.append((tag_norm, score))

        scored_tags.sort(key=lambda item: item[1], reverse=True)
        selected: list[dict[str, Any]] = []
        selected_term_sets: list[set[str]] = []
        for tag_norm, score in scored_tags:
            if score <= 0.0:
                continue
            tag_terms = tag_terms_by_norm.get(tag_norm, set())
            if not tag_terms:
                continue
            if selected and score < 0.12:
                continue
            too_similar = False
            for previous_terms in selected_term_sets:
                overlap = len(tag_terms.intersection(previous_terms))
                ratio = overlap / float(max(1, min(len(tag_terms), len(previous_terms))))
                if ratio >= 0.80:
                    too_similar = True
                    break
            if too_similar:
                continue
            label = display_label.get(tag_norm, tag_norm)
            class_doc_ids = [doc_id for doc_id in candidate_doc_ids if doc_id in tag_docs.get(tag_norm, set())]
            if not class_doc_ids:
                continue
            selected.append(
                {
                    "label": label,
                    "doc_ids": class_doc_ids[:10],
                    "score": round(float(score), 4),
                }
            )
            selected_term_sets.append(tag_terms)
            if len(selected) >= max(1, int(max_classes)):
                break
        return selected

    def _chunk_year_signals(self, chunk: dict) -> set[str]:
        years: set[str] = set()
        content = str(chunk.get("content") or "")
        years.update(YEAR_RE.findall(content))
        created_at = str(chunk.get("doc_created_at") or "")
        years.update(YEAR_RE.findall(created_at))
        return years

    def _chunk_has_numeric_signal(self, chunk: dict) -> bool:
        content = str(chunk.get("content") or "")
        if NUMERIC_SIGNAL_RE.search(content):
            return True
        values = re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", content)
        return len(values) >= 2

    def _chunk_matches_evidence_class(
        self,
        chunk: dict,
        label: str,
        class_doc_ids: set[str],
    ) -> bool:
        doc_id = str(chunk.get("doc_id") or "").strip()
        if class_doc_ids and doc_id and doc_id in class_doc_ids:
            return True
        label_terms = self._tokenize_for_match(label)
        if not label_terms:
            return False
        content = str(chunk.get("content") or "")
        filename = str(chunk.get("doc_filename") or "")
        haystack_terms = self._tokenize_for_match(f"{filename} {content}")
        if not haystack_terms:
            return False
        overlap = len(label_terms.intersection(haystack_terms))
        return (overlap / float(max(1, len(label_terms)))) >= 0.45

    def _evaluate_trend_coverage(
        self,
        chunks: List[dict],
        evidence_classes: list[dict[str, Any]],
        needs_numeric_extraction: bool,
    ) -> dict[str, Any]:
        min_chunks_per_class = 2 if needs_numeric_extraction else 1
        min_dates_global = 2
        min_numeric_chunks = max(2, len(evidence_classes)) if needs_numeric_extraction else 0

        global_years: set[str] = set()
        numeric_chunk_count = 0
        for chunk in chunks:
            global_years.update(self._chunk_year_signals(chunk))
            if self._chunk_has_numeric_signal(chunk):
                numeric_chunk_count += 1

        class_stats: dict[str, dict[str, Any]] = {}
        for class_info in evidence_classes:
            label = str(class_info.get("label") or "").strip()
            if not label:
                continue
            class_doc_ids = {
                str(doc_id or "").strip()
                for doc_id in (class_info.get("doc_ids") or [])
                if str(doc_id or "").strip()
            }
            matched = [
                chunk
                for chunk in chunks
                if self._chunk_matches_evidence_class(chunk, label, class_doc_ids)
            ]
            class_years: set[str] = set()
            class_numeric = 0
            class_docs: set[str] = set()
            for chunk in matched:
                chunk_doc_id = str(chunk.get("doc_id") or "").strip()
                if chunk_doc_id:
                    class_docs.add(chunk_doc_id)
                class_years.update(self._chunk_year_signals(chunk))
                if self._chunk_has_numeric_signal(chunk):
                    class_numeric += 1
            class_ok = len(matched) >= min_chunks_per_class and len(class_docs) >= 1
            if needs_numeric_extraction:
                class_ok = class_ok and class_numeric >= 1
            class_stats[label] = {
                "chunks": len(matched),
                "docs": len(class_docs),
                "dates": sorted(class_years),
                "numeric_chunks": class_numeric,
                "meets_minimum": class_ok,
            }

        if class_stats:
            class_coverage_ok = all(stat.get("meets_minimum", False) for stat in class_stats.values())
        else:
            class_coverage_ok = len(
                {
                    str(chunk.get("doc_id") or "").strip()
                    for chunk in chunks
                    if str(chunk.get("doc_id") or "").strip()
                }
            ) >= 2
        date_coverage_ok = len(global_years) >= min_dates_global
        numeric_coverage_ok = (not needs_numeric_extraction) or (numeric_chunk_count >= min_numeric_chunks)
        needs_follow_up = not (class_coverage_ok and date_coverage_ok and numeric_coverage_ok)
        return {
            "needs_follow_up": needs_follow_up,
            "class_stats": class_stats,
            "global_dates": sorted(global_years),
            "numeric_chunk_count": numeric_chunk_count,
            "requirements": {
                "min_chunks_per_class": min_chunks_per_class,
                "min_global_dates": min_dates_global,
                "min_numeric_chunks": min_numeric_chunks,
            },
        }

    def _run_trend_followup_pass(
        self,
        question: str,
        *,
        candidate_doc_ids: list[str],
        evidence_classes: list[dict[str, Any]],
        query_entities: list[str],
        coverage: dict[str, Any],
        top_k: int,
        per_doc_limit: int,
        mode: str,
        needs_numeric_extraction: bool,
    ) -> List[dict]:
        extras: List[dict] = []
        class_stats = coverage.get("class_stats") if isinstance(coverage.get("class_stats"), dict) else {}
        missing_classes = [
            class_info
            for class_info in evidence_classes
            if not bool(class_stats.get(str(class_info.get("label") or ""), {}).get("meets_minimum", False))
        ]
        if not missing_classes and bool(coverage.get("needs_follow_up", False)):
            missing_classes = evidence_classes[:2]

        for class_info in missing_classes[:4]:
            label = str(class_info.get("label") or "").strip()
            class_doc_ids = [
                str(doc_id or "").strip()
                for doc_id in (class_info.get("doc_ids") or [])
                if str(doc_id or "").strip()
            ]
            scoped_ids = class_doc_ids or candidate_doc_ids
            if not label or not scoped_ids:
                continue
            query_parts = [question, label, " ".join(query_entities[:4]).strip()]
            if needs_numeric_extraction:
                query_parts.append("amount total value pricing date year")
            follow_query = " ".join(part for part in query_parts if part).strip()
            extras.extend(
                self.retrieval.search_balanced(
                    follow_query,
                    top_k=max(8, top_k * 2),
                    doc_ids=scoped_ids,
                    per_doc_limit=max(2, per_doc_limit + 1),
                    mode=mode,
                )
            )

        global_dates = coverage.get("global_dates") if isinstance(coverage.get("global_dates"), list) else []
        if len(global_dates) < 2 and candidate_doc_ids:
            label_seed = " ".join(
                str(class_info.get("label") or "").strip()
                for class_info in evidence_classes[:3]
                if str(class_info.get("label") or "").strip()
            )
            date_query_parts = [question, label_seed, "year month quarter date total amount"]
            date_query = " ".join(part for part in date_query_parts if part).strip()
            extras.extend(
                self.retrieval.search_balanced(
                    date_query,
                    top_k=max(8, top_k * 2),
                    doc_ids=candidate_doc_ids,
                    per_doc_limit=max(2, per_doc_limit),
                    mode=mode,
                )
            )

        if not extras and candidate_doc_ids:
            broad_query = f"{question} {' '.join(query_entities[:4])}".strip()
            extras.extend(
                self.retrieval.search(
                    broad_query,
                    top_k=max(8, top_k * 2),
                    doc_ids=candidate_doc_ids,
                    mode=mode,
                )
            )

        return extras

    def _retrieve_for_trend_analysis(
        self,
        question: str,
        doc_ids: Optional[List[str]],
        top_k: int,
        per_doc_limit: int = 4,
        mode: str = "balanced",
        analysis_plan: Optional[dict[str, Any]] = None,
        needs_numeric_extraction: bool = True,
    ) -> tuple[List[dict], dict[str, Any]]:
        plan = analysis_plan if isinstance(analysis_plan, dict) else {}
        candidate_doc_ids = [
            str(doc_id or "").strip()
            for doc_id in (plan.get("candidate_doc_ids") or [])
            if str(doc_id or "").strip()
        ]
        if len(candidate_doc_ids) < 2:
            candidate_doc_ids = self._selected_or_ready_doc_ids(doc_ids)

        evidence_classes = [
            item
            for item in (plan.get("evidence_classes") or [])
            if isinstance(item, dict) and str(item.get("label") or "").strip()
        ]
        query_entities = [
            str(value or "").strip()
            for value in (plan.get("query_entities") or [])
            if str(value or "").strip()
        ]

        class_seed = " ".join(
            str(class_info.get("label") or "").strip()
            for class_info in evidence_classes[:4]
            if str(class_info.get("label") or "").strip()
        )
        expanded_query = " ".join(part for part in [question, class_seed, " ".join(query_entities[:4])] if part).strip()
        initial = self.retrieval.search_balanced(
            expanded_query or question,
            top_k=max(12, top_k * 2),
            doc_ids=candidate_doc_ids or doc_ids,
            per_doc_limit=max(2, per_doc_limit),
            mode=mode,
        )

        targeted: List[dict] = []
        for class_info in evidence_classes[:4]:
            label = str(class_info.get("label") or "").strip()
            if not label:
                continue
            class_doc_ids = [
                str(doc_id or "").strip()
                for doc_id in (class_info.get("doc_ids") or [])
                if str(doc_id or "").strip()
            ]
            scoped_ids = class_doc_ids or candidate_doc_ids or (doc_ids or [])
            query_parts = [question, label]
            if needs_numeric_extraction:
                query_parts.append("amount value total")
            targeted_query = " ".join(query_parts)
            targeted.extend(
                self.retrieval.search_balanced(
                    targeted_query,
                    top_k=max(8, top_k * 2),
                    doc_ids=scoped_ids,
                    per_doc_limit=max(2, per_doc_limit),
                    mode=mode,
                )
            )

        merged = self._merge_chunks(initial + targeted, limit=max(18, top_k * 5))
        coverage = self._evaluate_trend_coverage(
            merged,
            evidence_classes=evidence_classes,
            needs_numeric_extraction=needs_numeric_extraction,
        )

        for _ in range(2):
            if not coverage.get("needs_follow_up", False):
                break
            follow_up = self._run_trend_followup_pass(
                question,
                candidate_doc_ids=candidate_doc_ids or self._selected_or_ready_doc_ids(doc_ids),
                evidence_classes=evidence_classes,
                query_entities=query_entities,
                coverage=coverage,
                top_k=top_k,
                per_doc_limit=per_doc_limit,
                mode=mode,
                needs_numeric_extraction=needs_numeric_extraction,
            )
            if not follow_up:
                break
            merged = self._merge_chunks(merged + follow_up, limit=max(24, top_k * 6))
            coverage = self._evaluate_trend_coverage(
                merged,
                evidence_classes=evidence_classes,
                needs_numeric_extraction=needs_numeric_extraction,
            )

        if coverage.get("needs_follow_up", False) and self._retrieval_confidence_low(question, merged, min_docs=2):
            fallback = self.retrieval.search_balanced(
                expanded_query or question,
                top_k=max(12, top_k * 2),
                doc_ids=doc_ids,
                per_doc_limit=max(2, per_doc_limit),
                mode=mode,
            )
            merged = self._merge_chunks(merged + fallback, limit=max(26, top_k * 6))
            coverage = self._evaluate_trend_coverage(
                merged,
                evidence_classes=evidence_classes,
                needs_numeric_extraction=needs_numeric_extraction,
            )

        return merged, coverage

    def _retrieval_confidence_low(self, question: str, chunks: List[dict], min_docs: int) -> bool:
        if not chunks:
            return True
        unique_doc_ids = {str(chunk.get("doc_id") or "").strip() for chunk in chunks if chunk.get("doc_id")}
        if len(unique_doc_ids) < max(1, min_docs):
            return True
        support_required = 2 if min_docs >= 2 else 1
        support_count = 0
        for chunk in chunks[:10]:
            overlap = self._chunk_query_overlap(question, str(chunk.get("content") or ""))
            score = float(chunk.get("rerank_score", chunk.get("score", 0.0)) or 0.0)
            if overlap >= 0.16 or score >= 0.05:
                support_count += 1
            if support_count >= support_required:
                return False
        return True

    def _has_relevant_evidence(self, question: str, chunks: List[dict]) -> bool:
        return not self._retrieval_confidence_low(question, chunks, min_docs=1)

    def _looks_like_hard_no_answer(self, answer: str) -> bool:
        text = str(answer or "").strip().lower()
        if not text:
            return False
        markers = (
            "i cannot find the answer in the provided documents",
            "cannot find the answer in the provided documents",
            "cannot find the answer",
            "i cannot find the answer",
        )
        return any(marker in text for marker in markers)

    def _has_weak_evidence_prefix(self, answer: str) -> bool:
        text = str(answer or "").strip()
        return text.startswith(WEAK_EVIDENCE_PREFIX)

    def _fallback_answer(
        self,
        question: str,
        chunks: List[dict],
        intent: str = "qa",
        include_document_summaries: bool = True,
    ) -> str:
        if not chunks:
            return "I cannot find the answer in the provided documents."

        if intent in {"compare", "trend_analysis"}:
            return self._fallback_compare_answer(
                question,
                chunks,
                include_document_summaries=include_document_summaries,
            )

        candidate_chunks = list(chunks)
        candidate_doc_ids = sorted(
            {
                str(chunk.get("doc_id") or "").strip()
                for chunk in candidate_chunks
                if str(chunk.get("doc_id") or "").strip()
            }
        )
        if candidate_doc_ids:
            doc_records = {str(doc.get("id") or "").strip(): doc for doc in storage.list_documents()}
            for chunk in candidate_chunks:
                doc_id = str(chunk.get("doc_id") or "").strip()
                if not doc_id or doc_id in doc_records:
                    continue
                doc_records[doc_id] = {
                    "id": doc_id,
                    "filename": str(chunk.get("doc_filename") or ""),
                    "metadata": {},
                }
            entities = self._extract_query_entities(question)
            doc_tags_by_id = {
                doc_id: self._get_doc_auto_tags(doc_id, doc_records.get(doc_id) or {}, include_content_fallback=False)
                for doc_id in candidate_doc_ids
            }
            tag_signal = self._score_docs_by_query_tags(question, entities, doc_tags_by_id)
            if bool(tag_signal.get("high_confidence")):
                ranked_doc_ids = list(tag_signal.get("ranked_doc_ids", []))
                allowed_limit = 1
                if len(ranked_doc_ids) > 1:
                    scores = tag_signal.get("scores", {})
                    first = float(scores.get(ranked_doc_ids[0], 0.0))
                    second = float(scores.get(ranked_doc_ids[1], 0.0))
                    if (first - second) < 0.08:
                        allowed_limit = 2
                allowed_doc_ids = set(ranked_doc_ids[: max(1, min(allowed_limit, len(ranked_doc_ids)))])
                filtered = [
                    chunk
                    for chunk in candidate_chunks
                    if str(chunk.get("doc_id") or "").strip() in allowed_doc_ids
                ]
                if filtered:
                    candidate_chunks = filtered

        ranked_chunks = sorted(
            candidate_chunks,
            key=lambda chunk: (
                float(chunk.get("rerank_score", chunk.get("score", 0.0)) or 0.0),
                self._chunk_query_overlap(question, str(chunk.get("content") or "")),
            ),
            reverse=True,
        )
        excerpts: List[str] = []
        for chunk in ranked_chunks:
            content = " ".join(str(chunk.get("content", "")).split())
            if not content:
                continue
            overlap = self._chunk_query_overlap(question, content)
            score = float(chunk.get("rerank_score", chunk.get("score", 0.0)) or 0.0)
            if overlap < 0.11 and score < 0.05:
                continue
            snippet = content[:260].rstrip()
            source = self._build_source_tag(chunk)
            excerpts.append(f"- {snippet}... {source}")
            if len(excerpts) >= 3:
                break

        if not excerpts:
            return "I cannot find the answer in the provided documents."

        return WEAK_EVIDENCE_PREFIX + "\n" + "\n".join(excerpts)

    def _fallback_compare_answer(
        self,
        question: str,
        chunks: List[dict],
        include_document_summaries: bool = True,
    ) -> str:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for chunk in chunks:
            grouped[str(chunk.get("doc_id"))].append(chunk)
        if len(grouped) < 2:
            return (
                "I found evidence from only one document for this comparison question. "
                "Please select at least two documents or ask a single-document question."
            )

        if not include_document_summaries:
            lines = [
                f"I found partial comparison evidence for '{question}':"
            ]
            top_chunks = sorted(chunks, key=lambda c: float(c.get("score", 0.0)), reverse=True)[:6]
            for chunk in top_chunks:
                text = " ".join(str(chunk.get("content", "")).split())
                if not text:
                    continue
                source = self._build_source_tag(chunk)
                lines.append(f"- {text[:210].rstrip()}... {source}")
            return "\n".join(lines)

        lines = [
            f"I found partial evidence for '{question}'. Key points by document:"
        ]
        for doc_id, doc_chunks in grouped.items():
            top = sorted(doc_chunks, key=lambda c: float(c.get("score", 0.0)), reverse=True)[:2]
            filename = top[0].get("doc_filename") or doc_id
            created_at = top[0].get("doc_created_at") or "unknown date"
            snippets = []
            for chunk in top:
                text = " ".join(str(chunk.get("content", "")).split())
                if text:
                    snippets.append(text[:180].rstrip())
            if snippets:
                lines.append(f"- {filename} ({created_at}): {' | '.join(snippets)}...")
        return "\n".join(lines)

    def _detect_intent(self, question: str, doc_ids: Optional[List[str]]) -> str:
        text = question.lower().strip()
        compare_markers = [
            "compare",
            "difference",
            "different",
            "changed",
            "change",
            "vs",
            "versus",
            "between",
            "before",
            "after",
            "over time",
            "timeline",
            "evolution",
        ]
        if any(marker in text for marker in compare_markers):
            return "compare"
        year_hits = re.findall(r"\b(?:19|20)\d{2}\b", text)
        if len(year_hits) >= 2:
            return "compare"
        return "qa"

    def _needs_cross_doc(
        self,
        question: str,
        doc_ids: Optional[List[str]],
        task_type: str,
    ) -> bool:
        if task_type in {"compare", "trend_analysis", "count"}:
            return True
        selected_docs = [str(doc_id).strip() for doc_id in (doc_ids or []) if str(doc_id).strip()]
        if len(selected_docs) < 2:
            return False
        text = question.lower().strip()
        cross_doc_markers = [
            "across documents",
            "across docs",
            "across all",
            "all documents",
            "all docs",
            "from all",
            "each document",
            "per document",
            "document-wise",
            "combined",
            "consolidated",
            "list all",
        ]
        if any(marker in text for marker in cross_doc_markers):
            return True
        broad_list_markers = ["list", "what are", "which", "show me", "give me all"]
        list_entities = ["customers", "clients", "vendors", "suppliers", "locations", "countries"]
        if any(marker in text for marker in broad_list_markers) and any(entity in text for entity in list_entities):
            return True
        return False

    def _is_exhaustive_list_query(
        self,
        question: str,
        *,
        route: dict[str, Any],
        intent: str,
        needs_cross_doc: bool,
    ) -> bool:
        if intent != "qa" or not needs_cross_doc:
            return False
        expected_answer_type = self._route_expected_answer_type(route)
        text = str(question or "").strip().lower()
        if not text:
            return False

        list_markers = (
            "list",
            "which",
            "what are",
            "show me",
            "give me all",
            "who are",
            "name all",
        )
        explicit_exhaustive_markers = ("all", "each", "every")
        has_list_marker = any(marker in text for marker in list_markers)
        has_exhaustive_marker = any(marker in text for marker in explicit_exhaustive_markers)
        has_plural_signal = bool(re.search(r"\b[a-z0-9]{3,}s\b", text))

        if expected_answer_type == "list" and (has_list_marker or has_exhaustive_marker):
            return True
        return has_list_marker and has_exhaustive_marker and has_plural_signal

    def _route_question(
        self, question: str, doc_ids: Optional[List[str]], default_top_k: int
    ) -> dict[str, Any]:
        detected_intent = self._detect_intent(question, doc_ids)
        fallback_needs_cross_doc = self._needs_cross_doc(question, doc_ids, detected_intent)
        fallback = {
            "task_type": detected_intent,
            "needs_cross_doc": fallback_needs_cross_doc,
            "needs_numeric_extraction": False,
            "needs_image_reasoning": self._is_image_intent(question),
            "retrieval_plan": {
                "strategy": "balanced" if fallback_needs_cross_doc else "semantic",
                "top_k": max(default_top_k, 10 if fallback_needs_cross_doc else 8),
                "per_doc_limit": 4,
            },
            "analysis_plan": {},
            "confidence": 0.0,
            "source": "heuristic_fallback",
        }
        if self.router is None:
            return fallback
        try:
            available_docs = storage.list_documents()
            route = self.router.route(question, doc_ids=doc_ids, available_docs=available_docs)
            route["source"] = "llm_router"
            route.setdefault("needs_numeric_extraction", False)
            route.setdefault("analysis_plan", {})
            confidence = float(route.get("confidence", 0.0))
            route_task_type = str(route.get("task_type") or "").strip().lower()
            analysis_plan = route.get("analysis_plan") if isinstance(route.get("analysis_plan"), dict) else {}
            metadata_operation = str(analysis_plan.get("metadata_operation") or "").strip().lower()
            if (
                confidence < 0.35
                and route_task_type not in {"trend_analysis", "count", "metadata_query"}
                and not metadata_operation
            ):
                return fallback
            return route
        except Exception as exc:
            logger.warning("Router failed, using heuristic fallback: %s", exc)
            return fallback

    def _is_image_intent(self, question: str) -> bool:
        text = question.lower()
        markers = {
            "image",
            "diagram",
            "flowchart",
            "workflow",
            "figure",
            "chart",
            "what does this image show",
            "what does this diagram show",
            "in this picture",
            "in this graph",
        }
        return any(marker in text for marker in markers)

    def _is_relationship_intent(self, question: str) -> bool:
        text = question.lower()
        markers = {
            "relationship",
            "related",
            "connected",
            "connection",
            "flow",
            "sequence",
            "dependency",
            "dependencies",
            "across slides",
            "between slides",
            "slide progression",
            "transition",
            "diagram",
            "workflow",
        }
        return any(marker in text for marker in markers)

    def _augment_for_image_queries(
        self, question: str, chunks: List[dict], doc_ids: Optional[List[str]], mode: str = "hybrid"
    ) -> List[dict]:
        if not self._is_image_intent(question):
            return chunks
        if any(
            isinstance(chunk.get("metadata"), dict) and chunk.get("metadata", {}).get("image_path")
            for chunk in chunks
        ):
            return chunks
        supplemental = self.retrieval.search(
            f"{question} diagram image workflow figure chart",
            top_k=12,
            doc_ids=doc_ids,
            mode=mode,
        )
        return self._merge_chunks(chunks + supplemental, limit=16)

    def _augment_for_relationship_queries(
        self, question: str, chunks: List[dict], doc_ids: Optional[List[str]], mode: str = "hybrid"
    ) -> List[dict]:
        if not self._is_relationship_intent(question):
            return chunks
        relationship_sources = {"slide_graph", "diagram_graph", "diagram_node", "diagram_edge"}
        if any(str(chunk.get("source_type") or "").strip().lower() in relationship_sources for chunk in chunks):
            return chunks
        supplemental = self.retrieval.search(
            f"{question} relationship connector flow sequence dependency slide graph",
            top_k=14,
            doc_ids=doc_ids,
            mode=mode,
        )
        return self._merge_chunks(chunks + supplemental, limit=20)

    def _source_type_counts(self, chunks: List[dict]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for chunk in chunks:
            source_type = str(chunk.get("source_type") or "").strip().lower()
            if not source_type:
                continue
            counts[source_type] += 1
        return counts

    def _collect_companion_chunks(
        self,
        chunks: List[dict],
        doc_ids: Optional[List[str]],
        source_types: set[str],
        limit: int,
    ) -> List[dict]:
        if not source_types or limit <= 0:
            return []
        source_types = {str(item).strip().lower() for item in source_types if str(item).strip()}
        if not source_types:
            return []

        candidate_doc_ids: list[str] = []
        seen_doc_ids: set[str] = set()
        for doc_id in (doc_ids or []):
            raw = str(doc_id or "").strip()
            if raw and raw not in seen_doc_ids:
                seen_doc_ids.add(raw)
                candidate_doc_ids.append(raw)
        if not candidate_doc_ids:
            for chunk in chunks:
                raw = str(chunk.get("doc_id") or "").strip()
                if raw and raw not in seen_doc_ids:
                    seen_doc_ids.add(raw)
                    candidate_doc_ids.append(raw)
                if len(candidate_doc_ids) >= 6:
                    break
        if not candidate_doc_ids:
            return []

        page_scope: set[tuple[str, int]] = set()
        scope_sources = {"diagram_graph", "diagram_node", "diagram_edge", "slide_graph", "ocr", "image"}
        for chunk in chunks:
            source = str(chunk.get("source_type") or "").strip().lower()
            if source not in scope_sources:
                continue
            raw_doc_id = str(chunk.get("doc_id") or "").strip()
            if not raw_doc_id:
                continue
            try:
                page_num = int(chunk.get("page") or 0)
            except Exception:
                page_num = 0
            if page_num > 0:
                page_scope.add((raw_doc_id, page_num))

        highest_score = max((float(chunk.get("score") or 0.0) for chunk in chunks), default=0.0)
        fallback_score = highest_score - 0.005
        collected: List[dict] = []
        for doc_id in candidate_doc_ids:
            document = storage.get_document(doc_id) or {}
            doc_chunks = storage.get_chunks_by_doc(doc_id)
            for original in doc_chunks:
                source = str(original.get("source_type") or "").strip().lower()
                if source not in source_types:
                    continue
                try:
                    page_num = int(original.get("page") or 0)
                except Exception:
                    page_num = 0
                if page_scope and source != "slide_graph" and (doc_id, page_num) not in page_scope:
                    continue
                chunk = dict(original)
                chunk["doc_filename"] = document.get("filename")
                chunk["doc_created_at"] = document.get("created_at")
                chunk["score"] = float(chunk.get("score") or fallback_score - (0.0001 * len(collected)))
                collected.append(chunk)
                if len(collected) >= limit:
                    return collected
        return collected

    def _ensure_diagram_evidence_mix(
        self,
        question: str,
        chunks: List[dict],
        doc_ids: Optional[List[str]],
        mode: str,
        target_k: int,
    ) -> List[dict]:
        required_by_type = {
            "diagram_graph": max(0, int(config.DIAGRAM_MIN_GRAPH_CHUNKS)),
            "ocr": max(0, int(config.DIAGRAM_MIN_OCR_CHUNKS)),
            "diagram_node": max(0, int(config.DIAGRAM_MIN_NODE_CHUNKS)),
            "diagram_edge": max(0, int(config.DIAGRAM_MIN_EDGE_CHUNKS)),
            "slide_graph": max(0, int(config.DIAGRAM_MIN_SLIDE_GRAPH_CHUNKS)),
        }
        merge_limit = max(
            int(config.DIAGRAM_MIXED_EVIDENCE_LIMIT),
            max(8, int(target_k)) * 4,
        )
        working = list(chunks)
        working = self._merge_chunks(working, limit=merge_limit)
        counts = self._source_type_counts(working)

        missing_types = [source for source, required in required_by_type.items() if counts.get(source, 0) < required]
        if missing_types:
            companions = self._collect_companion_chunks(
                working,
                doc_ids=doc_ids,
                source_types=set(missing_types),
                limit=max(24, merge_limit // 2),
            )
            if companions:
                working = self._merge_chunks(working + companions, limit=merge_limit)
                counts = self._source_type_counts(working)
                missing_types = [
                    source for source, required in required_by_type.items() if counts.get(source, 0) < required
                ]

        if missing_types:
            source_queries = {
                "diagram_graph": f"{question} diagram summary key nodes relationships",
                "ocr": f"{question} exact text labels boxes shapes written words",
                "diagram_node": f"{question} diagram nodes labels regions boxes",
                "diagram_edge": f"{question} diagram edges arrows flow direction connections",
                "slide_graph": f"{question} slide relationship graph connectors",
            }
            extra: List[dict] = []
            for source in missing_types:
                required = required_by_type.get(source, 0)
                if required <= 0:
                    continue
                candidates = self.retrieval.search(
                    source_queries.get(source, question),
                    top_k=max(8, required * 8),
                    doc_ids=doc_ids,
                    mode=mode,
                )
                filtered = [
                    chunk
                    for chunk in candidates
                    if str(chunk.get("source_type") or "").strip().lower() == source
                ]
                if filtered:
                    extra.extend(filtered[: max(required * 2, required + 1)])
            if extra:
                working = self._merge_chunks(working + extra, limit=merge_limit)
                counts = self._source_type_counts(working)

        essential_sources = {"diagram_graph", "ocr", "diagram_node", "diagram_edge", "slide_graph"}
        present_sources = {
            source
            for source, count in counts.items()
            if source in essential_sources and int(count) > 0
        }
        min_types = max(1, int(config.DIAGRAM_MIN_EVIDENCE_SOURCE_TYPES))
        if len(present_sources) < min_types:
            broad = self.retrieval.search(
                f"{question} diagram ocr node edge relationship labels text",
                top_k=max(int(config.DIAGRAM_TOP_K_FLOOR) * 2, 18),
                doc_ids=doc_ids,
                mode=mode,
            )
            prioritized = [
                chunk
                for chunk in broad
                if str(chunk.get("source_type") or "").strip().lower() in essential_sources
            ]
            if prioritized:
                working = self._merge_chunks(working + prioritized, limit=merge_limit)

        final_target = max(int(target_k), int(config.DIAGRAM_TOP_K_FLOOR))
        ordered = self._diversify_diagram_chunks(
            working,
            required_by_type=required_by_type,
            limit=max(final_target, merge_limit),
        )
        return ordered

    def _diversify_diagram_chunks(
        self,
        chunks: List[dict],
        required_by_type: dict[str, int],
        limit: int,
    ) -> List[dict]:
        if not chunks:
            return []
        limit = max(1, int(limit))
        priority_order = ["diagram_graph", "ocr", "diagram_node", "slide_graph", "diagram_edge"]
        prioritized: dict[str, List[dict]] = {source: [] for source in priority_order}
        remainder: List[dict] = []

        ranked = sorted(chunks, key=lambda c: float(c.get("score", 0.0)), reverse=True)
        for chunk in ranked:
            source = str(chunk.get("source_type") or "").strip().lower()
            if source in prioritized:
                prioritized[source].append(chunk)
            else:
                remainder.append(chunk)

        selected: List[dict] = []
        seen_ids: set[str] = set()

        def add_chunk(chunk: dict) -> bool:
            chunk_id = str(chunk.get("id") or "").strip()
            if not chunk_id or chunk_id in seen_ids:
                return False
            seen_ids.add(chunk_id)
            selected.append(chunk)
            return True

        for source in priority_order:
            required = max(0, int(required_by_type.get(source, 0)))
            if required <= 0:
                continue
            for chunk in prioritized.get(source, [])[:required]:
                if len(selected) >= limit:
                    return selected
                add_chunk(chunk)

        cursors = {source: 0 for source in priority_order}
        while len(selected) < limit:
            added_in_round = False
            for source in priority_order:
                bucket = prioritized.get(source, [])
                cursor = int(cursors.get(source, 0))
                while cursor < len(bucket):
                    candidate = bucket[cursor]
                    cursor += 1
                    if add_chunk(candidate):
                        added_in_round = True
                        break
                cursors[source] = cursor
                if len(selected) >= limit:
                    return selected
            if not added_in_round:
                break

        for chunk in remainder:
            if len(selected) >= limit:
                break
            add_chunk(chunk)

        return selected

    def _prioritize_diagram_chunk_order(self, chunks: List[dict]) -> List[dict]:
        if not chunks:
            return []
        priority = {
            "diagram_graph": 0,
            "ocr": 1,
            "diagram_node": 2,
            "slide_graph": 3,
            "image": 4,
            "text": 5,
            "table": 5,
            "diagram_edge": 6,
        }
        return sorted(
            chunks,
            key=lambda chunk: (
                priority.get(str(chunk.get("source_type") or "").strip().lower(), 5),
                -float(chunk.get("score", 0.0)),
            ),
        )

    def _retrieve_for_comparison(
        self,
        question: str,
        doc_ids: Optional[List[str]],
        top_k: int,
        per_doc_limit: int = 4,
        mode: str = "hybrid",
    ) -> List[dict]:
        scoped = self._select_candidate_docs_for_query(
            question,
            doc_ids=doc_ids,
            require_multi_doc=True,
        )
        scoped_doc_ids = scoped["doc_ids"] if scoped else doc_ids
        global_doc_scope = scoped_doc_ids if scoped and bool(scoped.get("confident")) else (doc_ids or [])
        primary = self.retrieval.search_balanced(
            question,
            top_k=max(10, top_k * 2),
            doc_ids=scoped_doc_ids,
            per_doc_limit=max(2, per_doc_limit),
            mode=mode,
        )
        secondary = self.retrieval.search_balanced(
            f"{question} differences changes over time",
            top_k=max(8, top_k * 2),
            doc_ids=scoped_doc_ids,
            per_doc_limit=max(2, per_doc_limit - 1),
            mode=mode,
        )
        merged = self._merge_chunks(primary + secondary, limit=max(12, top_k * 3))
        if scoped and (
            not bool(scoped.get("confident"))
            or self._retrieval_confidence_low(question, merged, min_docs=2)
        ):
            global_primary = self.retrieval.search_balanced(
                question,
                top_k=max(10, top_k * 2),
                doc_ids=global_doc_scope or doc_ids,
                per_doc_limit=max(2, per_doc_limit),
                mode=mode,
            )
            global_secondary = self.retrieval.search_balanced(
                f"{question} differences changes over time",
                top_k=max(8, top_k * 2),
                doc_ids=global_doc_scope or doc_ids,
                per_doc_limit=max(2, per_doc_limit - 1),
                mode=mode,
            )
            merged = self._merge_chunks(merged + global_primary + global_secondary, limit=max(14, top_k * 3))
        if len({chunk.get("doc_id") for chunk in merged if chunk.get("doc_id")}) >= 2:
            return merged

        # If compare intent is clear but evidence came mostly from one doc, force coverage.
        candidate_doc_ids = scoped_doc_ids or global_doc_scope or doc_ids or [
            doc["id"] for doc in storage.list_documents() if str(doc.get("status", "")).lower() == "ready"
        ]
        forced: List[dict] = []
        for doc_id in candidate_doc_ids[:4]:
            forced.extend(self.retrieval.search(question, top_k=3, doc_ids=[doc_id], mode=mode))
        merged = self._merge_chunks(merged + forced, limit=max(14, top_k * 3))
        return merged

    def _retrieve_for_cross_doc_qa(
        self,
        question: str,
        doc_ids: Optional[List[str]],
        top_k: int,
        per_doc_limit: int = 4,
        mode: str = "hybrid",
        prefer_exhaustive: bool = False,
    ) -> List[dict]:
        scoped = self._select_candidate_docs_for_query(
            question,
            doc_ids=doc_ids,
            require_multi_doc=True,
            prefer_recall=prefer_exhaustive,
        )
        scoped_doc_ids = scoped["doc_ids"] if scoped else doc_ids
        global_doc_scope = doc_ids
        primary_top_k = max(10, top_k * (3 if prefer_exhaustive else 2))
        primary_per_doc = max(3 if prefer_exhaustive else 2, per_doc_limit)
        primary_use_rerank = not prefer_exhaustive
        primary = self.retrieval.search_balanced(
            question,
            top_k=primary_top_k,
            doc_ids=scoped_doc_ids,
            per_doc_limit=primary_per_doc,
            mode=mode,
            use_rerank=primary_use_rerank,
        )
        merged = self._merge_chunks(primary, limit=max(16 if prefer_exhaustive else 12, top_k * (4 if prefer_exhaustive else 3)))
        if scoped and (
            not bool(scoped.get("confident"))
            or self._retrieval_confidence_low(question, merged, min_docs=2)
        ):
            global_primary = self.retrieval.search_balanced(
                question,
                top_k=primary_top_k,
                doc_ids=global_doc_scope or doc_ids,
                per_doc_limit=primary_per_doc,
                mode=mode,
                use_rerank=primary_use_rerank,
            )
            merged = self._merge_chunks(
                merged + global_primary,
                limit=max(18 if prefer_exhaustive else 14, top_k * (4 if prefer_exhaustive else 3)),
            )

        if prefer_exhaustive:
            scoped_pool = scoped_doc_ids or global_doc_scope or doc_ids or [
                doc["id"] for doc in storage.list_documents() if str(doc.get("status", "")).lower() == "ready"
            ]
            covered_ids = {str(chunk.get("doc_id") or "").strip() for chunk in merged if str(chunk.get("doc_id") or "").strip()}
            missing_ids = [candidate for candidate in scoped_pool if str(candidate or "").strip() and str(candidate or "").strip() not in covered_ids]
            if missing_ids:
                forced_coverage: List[dict] = []
                for candidate_doc_id in missing_ids[: max(8, top_k * 2)]:
                    forced_coverage.extend(
                        self.retrieval.search(
                            question,
                            top_k=3,
                            doc_ids=[candidate_doc_id],
                            mode=mode,
                            use_rerank=False,
                        )
                    )
                merged = self._merge_chunks(
                    merged + forced_coverage,
                    limit=max(20, top_k * 5),
                )

        if len({chunk.get("doc_id") for chunk in merged if chunk.get("doc_id")}) >= 2:
            return merged

        # For multi-document QA, gently enforce document coverage without compare-style query expansion.
        candidate_doc_ids = scoped_doc_ids or global_doc_scope or doc_ids or [
            doc["id"] for doc in storage.list_documents() if str(doc.get("status", "")).lower() == "ready"
        ]
        forced: List[dict] = []
        forced_doc_cap = 10 if prefer_exhaustive else 6
        forced_top_k = 3 if prefer_exhaustive else 2
        forced_use_rerank = not prefer_exhaustive
        for doc_id in candidate_doc_ids[:forced_doc_cap]:
            forced.extend(
                self.retrieval.search(
                    question,
                    top_k=forced_top_k,
                    doc_ids=[doc_id],
                    mode=mode,
                    use_rerank=forced_use_rerank,
                )
            )
        return self._merge_chunks(merged + forced, limit=max(20 if prefer_exhaustive else 14, top_k * (5 if prefer_exhaustive else 3)))

    def _merge_chunks(self, chunks: List[dict], limit: int) -> List[dict]:
        def effective_score(chunk: dict) -> float:
            return float(chunk.get("rerank_score", chunk.get("score", 0.0)) or 0.0)

        best_by_id: dict[str, dict] = {}
        for chunk in chunks:
            chunk_id = str(chunk.get("id"))
            prev = best_by_id.get(chunk_id)
            if prev is None or effective_score(chunk) > effective_score(prev):
                best_by_id[chunk_id] = chunk
        return sorted(best_by_id.values(), key=effective_score, reverse=True)[:limit]

    def _dedupe_redundant_chunks(self, chunks: List[dict], limit: int = 0) -> List[dict]:
        deduped: List[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for chunk in chunks:
            filename = str(chunk.get("doc_filename") or "").strip().lower()
            page = str(chunk.get("page") or "?").strip()
            content = re.sub(r"\s+", " ", str(chunk.get("content", "")).strip().lower())
            if not content:
                continue
            key = (filename, page, content[:240])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(chunk)
            if limit and len(deduped) >= limit:
                break
        return deduped

    def _prepare_response_sources(self, chunks: List[dict], intent: str) -> List[dict]:
        max_total = 16 if intent == "compare" else 12
        per_doc_limit = 1 if intent == "compare" else 2
        diagram_sources = {"diagram_graph", "ocr", "diagram_node", "diagram_edge", "slide_graph"}
        has_diagram_sources = any(
            str(chunk.get("source_type") or "").strip().lower() in diagram_sources for chunk in chunks
        )
        if intent != "compare" and has_diagram_sources:
            max_total = max(max_total, 20)
            per_doc_limit = max(per_doc_limit, 4)
        by_doc_count: dict[str, int] = defaultdict(int)
        seen_keys: set[tuple[str, str]] = set()
        compact: List[dict] = []
        for chunk in chunks:
            doc_id = str(chunk.get("doc_id") or "").strip()
            if not doc_id:
                continue
            content = re.sub(r"\s+", " ", str(chunk.get("content", "")).strip().lower())
            if not content:
                continue
            if by_doc_count[doc_id] >= per_doc_limit:
                continue
            key = (doc_id, content[:220])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            by_doc_count[doc_id] += 1
            compact.append(chunk)
            if len(compact) >= max_total:
                break
        return compact

    def _document_briefs(self, chunks: List[dict]) -> List[str]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for chunk in chunks:
            doc_id = str(chunk.get("doc_id"))
            if doc_id:
                grouped[doc_id].append(chunk)
        briefs: List[str] = []
        seen_brief_keys: set[tuple[str, str]] = set()
        ranked_docs = sorted(
            grouped.items(),
            key=lambda item: max((float(c.get("score", 0.0)) for c in item[1]), default=0.0),
            reverse=True,
        )
        for doc_id, doc_chunks in ranked_docs:
            top = sorted(doc_chunks, key=lambda c: float(c.get("score", 0.0)), reverse=True)[:3]
            filename = top[0].get("doc_filename") or doc_id
            created_at = top[0].get("doc_created_at") or "unknown date"
            summary_lines = []
            for chunk in top:
                content = " ".join(str(chunk.get("content", "")).split())
                if content:
                    summary_lines.append(content[:180].rstrip())
            if summary_lines:
                filename_key = re.sub(r"\s+", " ", str(filename).strip().lower())
                evidence_key = re.sub(r"\s+", " ", " ".join(summary_lines[:2]).strip().lower())[:220]
                brief_key = (filename_key, evidence_key)
                if brief_key in seen_brief_keys:
                    continue
                seen_brief_keys.add(brief_key)
                briefs.append(
                    f"Document: {filename} (doc_id={doc_id}, created_at={created_at})\n"
                    f"Key evidence: {' | '.join(summary_lines)}"
                )
        return briefs

    def _strip_document_summaries(self, text: str) -> str:
        raw = str(text or "").replace("\r\n", "\n").strip()
        if not raw:
            return raw
        patterns = [
            r"\n+\s*document summaries\s*\n",
            r"\n+\s*document summary\s*\n",
            r"\n+\s*document summaries\s*/\s*sources\s*\n",
            r"\n+\s*summaries\s*\n",
        ]
        lower_raw = raw.lower()
        cut_index = -1
        for pattern in patterns:
            match = re.search(pattern, lower_raw, flags=re.IGNORECASE)
            if match:
                idx = match.start()
                if cut_index == -1 or idx < cut_index:
                    cut_index = idx
        if cut_index > 0:
            raw = raw[:cut_index].rstrip()
        return raw

    def _doc_label(self, filename: object, doc_id: object) -> str:
        raw = str(filename or "").strip().replace("\\", "/")
        base = raw.split("/")[-1] if raw else ""
        if base:
            stem = base.rsplit(".", 1)[0]
            if stem:
                return stem
            return base
        fallback = str(doc_id or "").strip()
        return fallback or "unknown"

    def _build_source_tag(self, chunk: dict) -> str:
        doc_id = str(chunk.get("doc_id") or "").strip() or "unknown"
        page = str(chunk.get("page") or "?").strip()
        chunk_id = str(chunk.get("id") or "").strip() or "unknown"
        doc_name = self._doc_label(chunk.get("doc_filename"), doc_id)
        source_type = str(chunk.get("source_type") or "").strip()
        source_suffix = f"|type:{source_type}" if source_type else ""
        return f"[source:doc:{doc_name}|doc_id:{doc_id}|page:{page}|chunk:{chunk_id}{source_suffix}]"
