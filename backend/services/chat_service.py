from __future__ import annotations

import logging
import re
import traceback
from collections import Counter, defaultdict
from typing import Any, List, Optional

import numpy as np

from .. import config, storage
from ..ingestion.doc_tags import build_document_auto_tags, normalize_tags
from ..models.prompts import build_compare_prompt, build_prompt
from PIL import Image

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_&.\-]{1,}")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
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


class ChatService:
    def __init__(
        self,
        retrieval_service,
        model,
        enable_vlm: bool,
        router=None,
        max_context_chars: int = 12000,
    ) -> None:
        self.retrieval = retrieval_service
        self.model = model
        self.enable_vlm = enable_vlm
        self.router = router
        self.max_context_chars = max_context_chars
        self.last_generation_error: str | None = None
        self.last_route: dict[str, Any] | None = None
        self._doc_tag_cache: dict[str, list[str]] = {}
        self._tag_embedding_cache: dict[str, np.ndarray] = {}

    def answer(
        self,
        question: str,
        doc_ids: Optional[List[str]] = None,
        top_k: int = 5,
        include_document_summaries: bool = True,
    ) -> dict:
        route = self._route_question(question, doc_ids, default_top_k=top_k)
        self.last_route = route
        intent = str(route.get("task_type", "qa")).strip().lower()
        if intent not in {"compare", "qa", "trend_analysis"}:
            intent = "qa"
        needs_cross_doc = bool(route.get("needs_cross_doc", False))
        needs_numeric_extraction = bool(route.get("needs_numeric_extraction", False))
        if intent == "trend_analysis":
            needs_cross_doc = True
            needs_numeric_extraction = True
        image_intent = bool(route.get("needs_image_reasoning", False)) or route.get("task_type") == "image_qa"
        relationship_intent = self._is_relationship_intent(question)
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
                question,
                doc_ids=doc_ids,
                router_analysis_plan=route.get("analysis_plan"),
                require_multi_doc=True,
            )
            route["analysis_plan"] = analysis_plan

        if intent == "compare":
            chunks = self._retrieve_for_comparison(
                question,
                doc_ids=doc_ids,
                top_k=route_top_k,
                per_doc_limit=per_doc_limit,
                mode=retrieval_mode,
            )
        elif intent == "trend_analysis":
            chunks, coverage = self._retrieve_for_trend_analysis(
                question,
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
                question,
                doc_ids=doc_ids,
                top_k=route_top_k,
                per_doc_limit=per_doc_limit,
                mode=retrieval_mode,
            )
        else:
            scoped = self._select_candidate_docs_for_query(
                question,
                doc_ids=doc_ids,
                require_multi_doc=False,
            )
            scoped_doc_ids = scoped["doc_ids"] if scoped else doc_ids
            chunks = self.retrieval.search(
                question,
                top_k=route_top_k,
                doc_ids=scoped_doc_ids,
                mode=retrieval_mode,
            )
            if scoped and (
                not bool(scoped.get("confident"))
                or self._retrieval_confidence_low(question, chunks, min_docs=1)
            ):
                fallback_doc_ids = scoped["doc_ids"] if bool(scoped.get("confident")) else (doc_ids or [])
                global_chunks = self.retrieval.search(
                    question,
                    top_k=route_top_k,
                    doc_ids=fallback_doc_ids or doc_ids,
                    mode=retrieval_mode,
                )
                chunks = self._merge_chunks(chunks + global_chunks, limit=max(12, route_top_k * 3))
        if image_intent:
            chunks = self._augment_for_image_queries(question, chunks, doc_ids=doc_ids, mode=retrieval_mode)
        if relationship_intent:
            chunks = self._augment_for_relationship_queries(
                question,
                chunks,
                doc_ids=doc_ids,
                mode=retrieval_mode,
            )

        if diagram_intent:
            chunks = self._ensure_diagram_evidence_mix(
                question,
                chunks,
                doc_ids=doc_ids,
                mode=retrieval_mode,
                target_k=route_top_k,
            )

        chunks = self._dedupe_redundant_chunks(chunks, limit=max(24, route_top_k * 4))
        if diagram_intent:
            chunks = self._prioritize_diagram_chunk_order(chunks)
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

        if intent in {"compare", "trend_analysis"}:
            prompt = build_compare_prompt(
                question,
                context_blocks,
                self._document_briefs(chunks),
                include_document_summaries=include_document_summaries,
            )
        else:
            prompt = build_prompt(question, context_blocks)
            
        self.last_generation_error = None

        if self.enable_vlm and self.model is not None:
            try:
                answer = self.model.generate_text(prompt, max_new_tokens=1500, images=b64_images)
            except Exception as exc:
                logger.exception("Model text generation failed: %s", exc)
                self.last_generation_error = traceback.format_exc(limit=12)
                answer = self._fallback_answer(
                    question,
                    chunks,
                    intent=intent,
                    include_document_summaries=include_document_summaries,
                )
        else:
            answer = self._fallback_answer(
                question,
                chunks,
                intent=intent,
                include_document_summaries=include_document_summaries,
            )

        if self._looks_like_hard_no_answer(answer) and self._has_relevant_evidence(question, chunks):
            answer = self._fallback_answer(
                question,
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
        if self.last_generation_error:
            response["generation_error"] = self.last_generation_error
        return response

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
    ) -> Optional[dict[str, Any]]:
        available_doc_ids = self._selected_or_ready_doc_ids(doc_ids)
        if len(available_doc_ids) <= 4:
            return None

        entities = self._extract_query_entities(question)
        query_terms = self._tokenize_for_match(question)
        if not entities and not query_terms:
            return None

        doc_records = {str(doc.get("id") or "").strip(): doc for doc in storage.list_documents()}
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

        if not bool(tag_signal.get("high_confidence")) and not filename_anchor_confident and len(scoped_doc_ids) >= len(available_doc_ids):
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

        return (
            f"I found partial evidence for '{question}'. The answer may be incomplete:\n"
            + "\n".join(excerpts)
        )

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
        if task_type in {"compare", "trend_analysis"}:
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
            if confidence < 0.35 and str(route.get("task_type") or "").strip().lower() != "trend_analysis":
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
        merged = self._merge_chunks(primary, limit=max(12, top_k * 3))
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
            merged = self._merge_chunks(merged + global_primary, limit=max(14, top_k * 3))
        if len({chunk.get("doc_id") for chunk in merged if chunk.get("doc_id")}) >= 2:
            return merged

        # For multi-document QA, gently enforce document coverage without compare-style query expansion.
        candidate_doc_ids = scoped_doc_ids or global_doc_scope or doc_ids or [
            doc["id"] for doc in storage.list_documents() if str(doc.get("status", "")).lower() == "ready"
        ]
        forced: List[dict] = []
        for doc_id in candidate_doc_ids[:6]:
            forced.extend(self.retrieval.search(question, top_k=2, doc_ids=[doc_id], mode=mode))
        return self._merge_chunks(merged + forced, limit=max(14, top_k * 3))

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
