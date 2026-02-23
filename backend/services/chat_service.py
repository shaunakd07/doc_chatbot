from __future__ import annotations

import logging
import re
import traceback
from collections import defaultdict
from typing import Any, List, Optional

from .. import config, storage

from ..models.prompts import build_compare_prompt, build_prompt
from PIL import Image

logger = logging.getLogger(__name__)


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

    def answer(
        self,
        question: str,
        doc_ids: Optional[List[str]] = None,
        top_k: int = 5,
        include_document_summaries: bool = True,
    ) -> dict:
        route = self._route_question(question, doc_ids, default_top_k=top_k)
        self.last_route = route
        intent = "compare" if route.get("needs_cross_doc") else str(route.get("task_type", "qa"))
        if intent not in {"compare", "qa"}:
            intent = "qa"
        image_intent = bool(route.get("needs_image_reasoning", False)) or route.get("task_type") == "image_qa"
        relationship_intent = self._is_relationship_intent(question)
        diagram_intent = image_intent or relationship_intent
        retrieval_plan = route.get("retrieval_plan") if isinstance(route.get("retrieval_plan"), dict) else {}
        route_top_k = int(retrieval_plan.get("top_k", max(top_k, 8)))
        per_doc_limit = int(retrieval_plan.get("per_doc_limit", 4))
        if diagram_intent:
            route_top_k = max(route_top_k, int(config.DIAGRAM_TOP_K_FLOOR))
            per_doc_limit = max(per_doc_limit, int(config.DIAGRAM_PER_DOC_LIMIT_FLOOR))
        retrieval_mode = str(retrieval_plan.get("strategy", "semantic")).strip().lower() or "semantic"
        if isinstance(route.get("retrieval_plan"), dict):
            route["retrieval_plan"]["top_k"] = route_top_k
            route["retrieval_plan"]["per_doc_limit"] = per_doc_limit

        if intent == "compare":
            chunks = self._retrieve_for_comparison(
                question,
                doc_ids=doc_ids,
                top_k=route_top_k,
                per_doc_limit=per_doc_limit,
                mode=retrieval_mode,
            )
            if image_intent:
                chunks = self._augment_for_image_queries(
                    question,
                    chunks,
                    doc_ids=doc_ids,
                    mode=retrieval_mode,
                )
            if relationship_intent:
                chunks = self._augment_for_relationship_queries(
                    question,
                    chunks,
                    doc_ids=doc_ids,
                    mode=retrieval_mode,
                )
        else:
            chunks = self.retrieval.search(question, top_k=route_top_k, doc_ids=doc_ids, mode=retrieval_mode)
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

        if intent == "compare":
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

    def _fallback_answer(
        self,
        question: str,
        chunks: List[dict],
        intent: str = "qa",
        include_document_summaries: bool = True,
    ) -> str:
        if not chunks:
            return "I could not find relevant content in your uploaded documents."

        if intent == "compare":
            return self._fallback_compare_answer(
                question,
                chunks,
                include_document_summaries=include_document_summaries,
            )

        excerpts: List[str] = []
        for chunk in chunks[:3]:
            content = " ".join(str(chunk.get("content", "")).split())
            if not content:
                continue
            snippet = content[:260].rstrip()
            source = self._build_source_tag(chunk)
            excerpts.append(f"- {snippet}... {source}")

        if not excerpts:
            return "I found sources but could not extract readable text."

        return (
            f"I could not run full model generation, but here is what your documents say about '{question}':\n"
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
                f"I could not run full model generation, but here are key comparison excerpts for '{question}':"
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
            f"I could not run full model generation, but here are key points per document for '{question}':"
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
        if doc_ids and len(doc_ids) > 1 and ("summarize all" in text or "across" in text):
            return "compare"
        return "qa"

    def _route_question(
        self, question: str, doc_ids: Optional[List[str]], default_top_k: int
    ) -> dict[str, Any]:
        fallback = {
            "task_type": self._detect_intent(question, doc_ids),
            "needs_cross_doc": self._detect_intent(question, doc_ids) == "compare",
            "needs_image_reasoning": self._is_image_intent(question),
            "retrieval_plan": {"strategy": "semantic", "top_k": max(default_top_k, 8), "per_doc_limit": 4},
            "confidence": 0.0,
            "source": "heuristic_fallback",
        }
        if self.router is None:
            return fallback
        try:
            available_docs = storage.list_documents()
            route = self.router.route(question, doc_ids=doc_ids, available_docs=available_docs)
            route["source"] = "llm_router"
            confidence = float(route.get("confidence", 0.0))
            if confidence < 0.35:
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
        primary = self.retrieval.search_balanced(
            question,
            top_k=max(10, top_k * 2),
            doc_ids=doc_ids,
            per_doc_limit=per_doc_limit,
            mode=mode,
        )
        secondary = self.retrieval.search_balanced(
            f"{question} differences changes over time",
            top_k=max(8, top_k * 2),
            doc_ids=doc_ids,
            per_doc_limit=max(2, per_doc_limit - 1),
            mode=mode,
        )
        merged = self._merge_chunks(primary + secondary, limit=max(12, top_k * 3))
        if len({chunk.get("doc_id") for chunk in merged if chunk.get("doc_id")}) >= 2:
            return merged

        # If compare intent is clear but evidence came mostly from one doc, force coverage.
        candidate_doc_ids = doc_ids or [
            doc["id"] for doc in storage.list_documents() if str(doc.get("status", "")).lower() == "ready"
        ]
        forced: List[dict] = []
        for doc_id in candidate_doc_ids[:4]:
            forced.extend(self.retrieval.search(question, top_k=3, doc_ids=[doc_id], mode=mode))
        merged = self._merge_chunks(merged + forced, limit=max(14, top_k * 3))
        return merged

    def _merge_chunks(self, chunks: List[dict], limit: int) -> List[dict]:
        best_by_id: dict[str, dict] = {}
        for chunk in chunks:
            chunk_id = str(chunk.get("id"))
            prev = best_by_id.get(chunk_id)
            if prev is None or float(chunk.get("score", 0.0)) > float(prev.get("score", 0.0)):
                best_by_id[chunk_id] = chunk
        return sorted(best_by_id.values(), key=lambda c: float(c.get("score", 0.0)), reverse=True)[:limit]

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
