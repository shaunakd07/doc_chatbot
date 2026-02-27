from __future__ import annotations

from typing import Any, Dict, List

from .types import Condition, TestCase


def _status(value: bool) -> str:
    return "pass" if value else "fail"


class AssertionEngine:
    def evaluate(self, case: TestCase, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for cond in case.checks:
            try:
                out.append(self._evaluate_condition(cond, context))
            except Exception as exc:
                out.append(
                    {
                        "condition": cond.kind,
                        "status": "error",
                        "message": f"Assertion evaluation error: {exc}",
                        "raw": cond.raw,
                    }
                )
        return out

    def _evaluate_condition(self, cond: Condition, context: Dict[str, Any]) -> Dict[str, Any]:
        kind = cond.kind
        doc_ids: List[str] = list(context.get("doc_ids") or [])
        docs: Dict[str, Dict[str, Any]] = dict(context.get("documents") or {})
        stats: Dict[str, Dict[str, Any]] = dict(context.get("doc_stats") or {})
        chat_results: List[Dict[str, Any]] = list(context.get("chat_results") or [])
        db_global: Dict[str, Any] = dict(context.get("db_global") or {})

        if kind == "doc_count_equals":
            expected = int(cond.value or 0)
            actual = len(doc_ids)
            ok = actual == expected
            return {
                "condition": kind,
                "status": _status(ok),
                "message": f"Expected {expected} docs, got {actual}.",
            }

        if kind == "all_docs_ready":
            bad = []
            for doc_id in doc_ids:
                status = str((docs.get(doc_id) or {}).get("status") or "").strip().lower()
                if status != "ready":
                    bad.append((doc_id, status))
            ok = len(bad) == 0
            return {
                "condition": kind,
                "status": _status(ok),
                "message": "All docs ready." if ok else f"Non-ready docs: {bad}",
            }

        if kind == "each_doc_num_pages_gt":
            threshold = int(cond.value or 0)
            bad = []
            for doc_id in doc_ids:
                num_pages = int((docs.get(doc_id) or {}).get("num_pages") or 0)
                if num_pages <= threshold:
                    bad.append((doc_id, num_pages))
            ok = len(bad) == 0
            return {
                "condition": kind,
                "status": _status(ok),
                "message": f"All docs have num_pages > {threshold}." if ok else f"Violations: {bad}",
            }

        if kind == "each_doc_min_chunks":
            minimum = int(cond.value or 1)
            bad = []
            for doc_id in doc_ids:
                count = int((stats.get(doc_id) or {}).get("chunk_count") or 0)
                if count < minimum:
                    bad.append((doc_id, count))
            ok = len(bad) == 0
            return {
                "condition": kind,
                "status": _status(ok),
                "message": f"All docs have >= {minimum} chunks." if ok else f"Violations: {bad}",
            }

        if kind == "each_doc_embeddings_match_chunks":
            bad = []
            for doc_id in doc_ids:
                stat = stats.get(doc_id) or {}
                chunks = int(stat.get("chunk_count") or 0)
                embeddings = int(stat.get("embedding_count") or 0)
                if chunks != embeddings:
                    bad.append((doc_id, chunks, embeddings))
            ok = len(bad) == 0
            return {
                "condition": kind,
                "status": _status(ok),
                "message": "Embeddings match chunks for all docs." if ok else f"Mismatches: {bad}",
            }

        if kind == "any_doc_has_source_types":
            required = {v.strip().lower() for v in cond.source_types if v.strip()}
            present_any = False
            present_map: Dict[str, List[str]] = {}
            for doc_id in doc_ids:
                counts = (stats.get(doc_id) or {}).get("source_type_counts") or {}
                sources = [str(k).strip().lower() for k in counts.keys()]
                present_map[doc_id] = sorted(sources)
                if required.intersection(sources):
                    present_any = True
            return {
                "condition": kind,
                "status": _status(present_any),
                "message": (
                    f"At least one required source type found: {sorted(required)}."
                    if present_any
                    else f"None of required source types found. Present: {present_map}"
                ),
            }

        if kind == "each_doc_has_source_types":
            required = {v.strip().lower() for v in cond.source_types if v.strip()}
            bad = []
            for doc_id in doc_ids:
                counts = (stats.get(doc_id) or {}).get("source_type_counts") or {}
                sources = {str(k).strip().lower() for k in counts.keys()}
                if not required.intersection(sources):
                    bad.append({"doc_id": doc_id, "sources": sorted(sources)})
            ok = len(bad) == 0
            return {
                "condition": kind,
                "status": _status(ok),
                "message": "Each doc contains at least one required source type." if ok else f"Missing required source types: {bad}",
            }

        if kind == "each_doc_min_diagram_graphs":
            minimum = int(cond.value or 1)
            bad = []
            for doc_id in doc_ids:
                count = int((stats.get(doc_id) or {}).get("diagram_graph_count") or 0)
                if count < minimum:
                    bad.append((doc_id, count))
            ok = len(bad) == 0
            return {
                "condition": kind,
                "status": _status(ok),
                "message": f"All docs have >= {minimum} diagram_graphs." if ok else f"Violations: {bad}",
            }

        if kind == "chat_non_empty":
            bad = []
            for idx, result in enumerate(chat_results, start=1):
                probe = result.get("probe") if isinstance(result.get("probe"), dict) else {}
                if str((probe or {}).get("expect_error_contains") or "").strip():
                    continue
                answer = str((result.get("response") or {}).get("answer") or "").strip()
                if not answer:
                    bad.append(idx)
            ok = len(bad) == 0
            return {
                "condition": kind,
                "status": _status(ok),
                "message": "All chat responses have non-empty answers." if ok else f"Empty answers on probes: {bad}",
            }

        if kind == "chat_sources_scoped":
            bad: List[int] = []
            scoped_set = set(doc_ids)
            for idx, result in enumerate(chat_results, start=1):
                if str(result.get("error") or "").strip():
                    continue
                sources = (result.get("response") or {}).get("sources") or []
                if not isinstance(sources, list):
                    continue
                out_of_scope = []
                for source in sources:
                    if not isinstance(source, dict):
                        continue
                    doc_id = str(source.get("doc_id") or "").strip()
                    if doc_id and doc_id not in scoped_set:
                        out_of_scope.append(doc_id)
                if out_of_scope:
                    bad.append(idx)
            ok = len(bad) == 0
            return {
                "condition": kind,
                "status": _status(ok),
                "message": "All source docs are within requested scope." if ok else f"Out-of-scope sources in probes: {bad}",
            }

        if kind == "chat_any_error":
            any_error = any(str(result.get("error") or "").strip() for result in chat_results)
            return {
                "condition": kind,
                "status": _status(any_error),
                "message": "At least one chat probe returned an error." if any_error else "No chat probe errors observed.",
            }

        if kind == "chat_expected_errors":
            bad = []
            checked = 0
            for idx, result in enumerate(chat_results, start=1):
                probe = result.get("probe") if isinstance(result.get("probe"), dict) else {}
                expected = str((probe or {}).get("expect_error_contains") or "").strip().lower()
                if not expected:
                    continue
                checked += 1
                actual = str(result.get("error") or "").strip().lower()
                if not actual or expected not in actual:
                    bad.append({"probe_index": idx, "expected_substring": expected, "actual_error": actual})
            ok = checked > 0 and len(bad) == 0
            return {
                "condition": kind,
                "status": _status(ok),
                "message": "All expected chat errors observed." if ok else f"Expected chat error mismatches: {bad}",
            }

        if kind == "db_table_non_empty":
            table = str(cond.table or "").strip().lower()
            count = int((db_global.get("table_counts") or {}).get(table) or 0)
            ok = count > 0
            return {
                "condition": kind,
                "status": _status(ok),
                "message": f"Table `{table}` count={count}.",
            }

        if kind == "db_table_empty":
            table = str(cond.table or "").strip().lower()
            count = int((db_global.get("table_counts") or {}).get(table) or 0)
            ok = count == 0
            return {
                "condition": kind,
                "status": _status(ok),
                "message": f"Table `{table}` count={count}.",
            }

        if kind == "single_embedding_dim":
            dim_dist: Dict[int, int] = dict(db_global.get("embedding_dim_distribution") or {})
            ok = len(dim_dist) <= 1
            return {
                "condition": kind,
                "status": _status(ok),
                "message": f"Embedding dim distribution: {dim_dist}",
            }

        return {
            "condition": kind,
            "status": "skip",
            "message": "Condition kind is not implemented in harness engine.",
            "raw": cond.raw,
        }
