from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

from .api_client import ApiClient
from .assertions import AssertionEngine
from .catalog import resolve_dataset
from .db_inspector import DBInspector
from .evidence import EvidenceWriter, utc_now_iso
from .types import FileSelector, Profile, RunSummary, TestCase


TERMINAL_DOC_STATUSES = {"ready", "failed"}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _elapsed_sec(start: float) -> float:
    return round(time.monotonic() - start, 3)


class HarnessRunner:
    def __init__(
        self,
        *,
        repo_root: Path,
        mode: str,
        profile: Profile,
        selectors: Dict[str, FileSelector],
        cases: List[TestCase],
        evidence: EvidenceWriter,
        api_base_url: str,
        api_timeout_sec: float = 120.0,
        db_backend: Optional[str] = None,
        sqlite_db_path: Optional[str] = None,
        database_url: Optional[str] = None,
    ) -> None:
        self.repo_root = repo_root
        self.mode = mode.strip().lower()
        self.profile = profile
        self.selectors = selectors
        self.cases = cases
        self.evidence = evidence
        self.api_base_url = api_base_url
        self.api_timeout_sec = max(2.0, float(api_timeout_sec))
        self.assertions = AssertionEngine()
        self.db = DBInspector(
            db_backend=db_backend,
            sqlite_db_path=sqlite_db_path,
            database_url=database_url,
        )

    def run(self) -> RunSummary:
        started_at = _now_utc()
        run_timer = time.monotonic()

        self.evidence.write_json(
            "snapshots/run_context.json",
            {
                "run_id": self.evidence.run_id,
                "mode": self.mode,
                "profile": self.profile.name,
                "profile_description": self.profile.description,
                "started_at_utc": started_at.isoformat(),
                "selected_case_ids": [case.case_id for case in self.cases],
            },
        )

        if self.mode == "execute":
            with ApiClient(self.api_base_url, timeout_sec=self.api_timeout_sec) as api:
                health = api.health()
                self.evidence.write_json("api/health.json", health)
        else:
            self.evidence.write_json(
                "api/health.json",
                {"mode": "dry-run", "note": "health endpoint not called"},
            )

        executed_cases = 0
        passed_cases = 0
        failed_cases = 0
        skipped_cases = 0

        for case in self.cases:
            case_start = time.monotonic()
            self.evidence.event("case_start", {"case_id": case.case_id, "title": case.title})
            try:
                result = self._run_case(case)
            except Exception as exc:
                result = {
                    "case_id": case.case_id,
                    "suite": case.suite,
                    "title": case.title,
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "duration_sec": _elapsed_sec(case_start),
                }

            self.evidence.write_json(f"snapshots/cases/{case.case_id}.json", result)
            self.evidence.event(
                "case_end",
                {
                    "case_id": case.case_id,
                    "status": result.get("status"),
                    "duration_sec": result.get("duration_sec"),
                },
            )

            status = str(result.get("status") or "").strip().lower()
            if status in {"pass", "fail", "error", "skip"}:
                executed_cases += 1
            if status == "pass":
                passed_cases += 1
            elif status in {"fail", "error"}:
                failed_cases += 1
            elif status == "skip":
                skipped_cases += 1

            if failed_cases > 0 and self.profile.stop_on_failure:
                break

        summary = RunSummary(
            run_id=self.evidence.run_id,
            profile=self.profile.name,
            mode=self.mode,
            selected_cases=len(self.cases),
            executed_cases=executed_cases,
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            skipped_cases=skipped_cases,
            started_at=started_at.isoformat(),
            finished_at=_now_utc().isoformat(),
            duration_sec=_elapsed_sec(run_timer),
        )
        self.evidence.write_json("summary.json", summary)
        return summary

    def _resolve_case_files(self, case: TestCase) -> Tuple[List[Path], List[Dict[str, Any]]]:
        files: List[Path] = []
        details: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for dataset_name in case.datasets:
            dataset_files = resolve_dataset(dataset_name, self.selectors, self.repo_root)
            details.append(
                {
                    "dataset": dataset_name,
                    "count": len(dataset_files),
                    "sample": [path.as_posix() for path in dataset_files[:5]],
                }
            )
            for path in dataset_files:
                key = path.as_posix().lower()
                if key in seen:
                    continue
                seen.add(key)
                files.append(path)
        files = sorted(files, key=lambda p: p.as_posix().lower())
        if self.profile.max_files_per_case is not None:
            files = files[: max(0, int(self.profile.max_files_per_case))]
        return files, details

    def _wait_for_documents(
        self,
        api: ApiClient,
        doc_ids: List[str],
        *,
        timeout_sec: int,
        poll_interval_sec: float,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + max(1, int(timeout_sec))
        history: List[Dict[str, Any]] = []
        last_docs: Dict[str, Dict[str, Any]] = {}
        while time.monotonic() < deadline:
            all_terminal = True
            snapshot: Dict[str, Any] = {"time_utc": utc_now_iso(), "docs": []}
            for doc_id in doc_ids:
                doc = api.get_document(doc_id)
                last_docs[doc_id] = doc
                status = str(doc.get("status") or "").strip().lower()
                snapshot["docs"].append(
                    {
                        "doc_id": doc_id,
                        "status": status,
                        "progress": (doc.get("metadata") or {}).get("ingest_progress")
                        if isinstance(doc.get("metadata"), dict)
                        else None,
                    }
                )
                if status not in TERMINAL_DOC_STATUSES:
                    all_terminal = False
            history.append(snapshot)
            if all_terminal:
                break
            time.sleep(max(0.2, float(poll_interval_sec)))

        timed_out = any(
            str((last_docs.get(doc_id) or {}).get("status") or "").strip().lower()
            not in TERMINAL_DOC_STATUSES
            for doc_id in doc_ids
        )
        return {"history": history, "last_docs": last_docs, "timed_out": timed_out}

    def _collect_doc_stats(self, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for doc_id in doc_ids:
            out[doc_id] = {
                "chunk_count": self.db.doc_chunk_count(doc_id),
                "embedding_count": self.db.doc_embedding_count(doc_id),
                "diagram_graph_count": self.db.doc_diagram_graph_count(doc_id),
                "source_type_counts": self.db.doc_source_type_counts(doc_id),
                "chunk_sample": self.db.sample_chunks(doc_id, limit=8),
                "embedding_dims": self.db.embedding_dim_distribution(doc_id),
            }
        return out

    def _collect_global_db_stats(self) -> Dict[str, Any]:
        table_counts = {
            "documents": self.db.table_count("documents"),
            "chunks": self.db.table_count("chunks"),
            "embeddings": self.db.table_count("embeddings"),
            "diagram_graphs": self.db.table_count("diagram_graphs"),
        }
        return {
            "table_counts": table_counts,
            "embedding_dim_distribution": self.db.embedding_dim_distribution(),
        }

    def _run_case(self, case: TestCase) -> Dict[str, Any]:
        timer = time.monotonic()
        files, dataset_details = self._resolve_case_files(case)

        if self.mode == "dry-run":
            return {
                "case_id": case.case_id,
                "suite": case.suite,
                "title": case.title,
                "status": "skip",
                "reason": "dry-run",
                "mode": self.mode,
                "automation_level": case.automation_level,
                "datasets": dataset_details,
                "resolved_files_count": len(files),
                "resolved_files_sample": [path.as_posix() for path in files[:20]],
                "checks": [cond.raw or {"kind": cond.kind} for cond in case.checks],
                "chat_probe_count": len(case.chat_probes),
                "duration_sec": _elapsed_sec(timer),
            }

        if case.automation_level == "manual":
            return {
                "case_id": case.case_id,
                "suite": case.suite,
                "title": case.title,
                "status": "skip",
                "reason": "manual-case",
                "automation_level": case.automation_level,
                "manual_steps": case.manual_steps,
                "duration_sec": _elapsed_sec(timer),
            }

        with ApiClient(self.api_base_url, timeout_sec=self.api_timeout_sec) as api:
            before_global = self._collect_global_db_stats()
            if case.reset_before:
                reset_result = api.delete_all_documents()
            else:
                reset_result = {"status": "skipped"}

            uploads: List[Dict[str, Any]] = []
            doc_ids: List[str] = []
            for path in files:
                upload_payload = api.upload_document(path)
                doc_id = str(upload_payload.get("doc_id") or "").strip()
                uploads.append(
                    {
                        "file": path.as_posix(),
                        "response": upload_payload,
                        "doc_id": doc_id,
                    }
                )
                if doc_id:
                    doc_ids.append(doc_id)

            wait_payload = {"history": [], "last_docs": {}, "timed_out": False}
            if case.wait_ready and doc_ids:
                wait_payload = self._wait_for_documents(
                    api,
                    doc_ids,
                    timeout_sec=case.ready_timeout_sec,
                    poll_interval_sec=case.poll_interval_sec,
                )

            documents: Dict[str, Dict[str, Any]] = {}
            for doc_id in doc_ids:
                documents[doc_id] = api.get_document(doc_id)

            doc_stats = self._collect_doc_stats(doc_ids)

            chat_results: List[Dict[str, Any]] = []
            for probe in case.chat_probes:
                probe_doc_ids = doc_ids if probe.use_case_doc_ids else self._resolve_probe_doc_ids(probe.doc_ids, doc_ids)
                response: Dict[str, Any] = {}
                error_message = ""
                try:
                    response = api.chat(
                        probe.prompt,
                        doc_ids=probe_doc_ids,
                        top_k=probe.top_k,
                        include_document_summaries=probe.include_document_summaries,
                    )
                except Exception as exc:
                    error_message = str(exc)
                chat_results.append(
                    {
                        "probe": {
                            "prompt": probe.prompt,
                            "top_k": probe.top_k,
                            "include_document_summaries": probe.include_document_summaries,
                            "use_case_doc_ids": probe.use_case_doc_ids,
                            "doc_ids": probe_doc_ids,
                            "expect_error_contains": probe.expect_error_contains,
                            "expected_doc_ids": probe.expected_doc_ids,
                            "require_non_empty_answer": probe.require_non_empty_answer,
                        },
                        "response": response,
                        "error": error_message,
                    }
                )

            if case.delete_after:
                delete_results = [api.delete_document(doc_id) for doc_id in doc_ids]
            else:
                delete_results = []
            if case.delete_all_after:
                delete_all_result = api.delete_all_documents()
            else:
                delete_all_result = {"status": "skipped"}

            after_global = self._collect_global_db_stats()
            context = {
                "doc_ids": doc_ids,
                "documents": documents,
                "doc_stats": doc_stats,
                "chat_results": chat_results,
                "db_global": after_global,
            }
            check_results = self.assertions.evaluate(case, context)

            has_failures = any(result.get("status") in {"fail", "error"} for result in check_results)
            has_timeouts = bool(wait_payload.get("timed_out"))
            status = "pass"
            if has_timeouts:
                status = "fail"
            if has_failures:
                status = "fail"

            return {
                "case_id": case.case_id,
                "suite": case.suite,
                "title": case.title,
                "status": status,
                "objective": case.objective,
                "automation_level": case.automation_level,
                "datasets": dataset_details,
                "resolved_files_count": len(files),
                "uploads": uploads,
                "reset_result": reset_result,
                "wait": {
                    "timed_out": bool(wait_payload.get("timed_out")),
                    "history_size": len(wait_payload.get("history") or []),
                    "history_sample": (wait_payload.get("history") or [])[-10:],
                },
                "doc_ids": doc_ids,
                "documents": documents,
                "doc_stats": doc_stats,
                "chat_results": chat_results,
                "delete_results": delete_results,
                "delete_all_result": delete_all_result,
                "checks": check_results,
                "db_before": before_global,
                "db_after": after_global,
                "duration_sec": _elapsed_sec(timer),
            }

    def _resolve_probe_doc_ids(self, probe_doc_ids: List[str], case_doc_ids: List[str]) -> List[str]:
        resolved: List[str] = []
        for raw in probe_doc_ids:
            token = str(raw or "").strip()
            upper = token.upper()
            if upper == "__ALL_CASE_DOCS__":
                resolved.extend(case_doc_ids)
                continue
            if upper == "__FIRST_DOC__":
                if case_doc_ids:
                    resolved.append(case_doc_ids[0])
                continue
            if upper == "__SECOND_DOC__":
                if len(case_doc_ids) >= 2:
                    resolved.append(case_doc_ids[1])
                continue
            if upper == "__THIRD_DOC__":
                if len(case_doc_ids) >= 3:
                    resolved.append(case_doc_ids[2])
                continue
            resolved.append(token)
        deduped: List[str] = []
        seen: set[str] = set()
        for doc_id in resolved:
            if not doc_id:
                continue
            if doc_id in seen:
                continue
            seen.add(doc_id)
            deduped.append(doc_id)
        return deduped
