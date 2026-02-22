from __future__ import annotations

import logging
import re
import traceback
from collections import defaultdict
from pathlib import Path
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

    def answer(self, question: str, doc_ids: Optional[List[str]] = None, top_k: int = 5) -> dict:
        route = self._route_question(question, doc_ids, default_top_k=top_k)
        self.last_route = route
        intent = "compare" if route.get("needs_cross_doc") else str(route.get("task_type", "qa"))
        if intent not in {"compare", "qa"}:
            intent = "qa"
        image_intent = bool(route.get("needs_image_reasoning", False)) or route.get("task_type") == "image_qa"
        retrieval_plan = route.get("retrieval_plan") if isinstance(route.get("retrieval_plan"), dict) else {}
        route_top_k = int(retrieval_plan.get("top_k", max(top_k, 8)))
        per_doc_limit = int(retrieval_plan.get("per_doc_limit", 4))
        retrieval_mode = str(retrieval_plan.get("strategy", "semantic")).strip().lower() or "semantic"

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
        else:
            chunks = self.retrieval.search(question, top_k=route_top_k, doc_ids=doc_ids, mode=retrieval_mode)
            if image_intent:
                chunks = self._augment_for_image_queries(question, chunks, doc_ids=doc_ids, mode=retrieval_mode)

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
            prompt = build_compare_prompt(question, context_blocks, self._document_briefs(chunks))
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
                )
        else:
            answer = self._fallback_answer(
                question,
                chunks,
                intent=intent,
            )

        response = {
            "answer": answer,
            "sources": chunks,
            "intent": intent,
            "route": route,
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
            filename = chunk.get("doc_filename") or "unknown"
            created_at = chunk.get("doc_created_at") or "unknown"
            block = (
                f"[source:{chunk['doc_id']}:{chunk.get('page','?')}:{chunk['id']}"
                f"|file:{filename}|created:{created_at}] {content}"
            )
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
    ) -> str:
        if not chunks:
            return "I could not find relevant content in your uploaded documents."

        if intent == "compare":
            return self._fallback_compare_answer(question, chunks)

        excerpts: List[str] = []
        for chunk in chunks[:3]:
            content = " ".join(str(chunk.get("content", "")).split())
            if not content:
                continue
            snippet = content[:260].rstrip()
            source = f"[source:{chunk['doc_id']}:{chunk.get('page','?')}:{chunk['id']}]"
            excerpts.append(f"- {snippet}... {source}")

        if not excerpts:
            return "I found sources but could not extract readable text."

        return (
            f"I could not run full model generation, but here is what your documents say about '{question}':\n"
            + "\n".join(excerpts)
        )

    def _fallback_compare_answer(self, question: str, chunks: List[dict]) -> str:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for chunk in chunks:
            grouped[str(chunk.get("doc_id"))].append(chunk)
        if len(grouped) < 2:
            return (
                "I found evidence from only one document for this comparison question. "
                "Please select at least two documents or ask a single-document question."
            )

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

    def _document_briefs(self, chunks: List[dict]) -> List[str]:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for chunk in chunks:
            doc_id = str(chunk.get("doc_id"))
            if doc_id:
                grouped[doc_id].append(chunk)
        briefs: List[str] = []
        for doc_id, doc_chunks in grouped.items():
            top = sorted(doc_chunks, key=lambda c: float(c.get("score", 0.0)), reverse=True)[:3]
            filename = top[0].get("doc_filename") or doc_id
            created_at = top[0].get("doc_created_at") or "unknown date"
            summary_lines = []
            for chunk in top:
                content = " ".join(str(chunk.get("content", "")).split())
                if content:
                    summary_lines.append(content[:180].rstrip())
            if summary_lines:
                briefs.append(
                    f"Document: {filename} (doc_id={doc_id}, created_at={created_at})\n"
                    f"Key evidence: {' | '.join(summary_lines)}"
                )
        return briefs
