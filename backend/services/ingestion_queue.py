from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

from .. import storage
from ..ingestion.pipeline import ingest_file
from .out_of_place_detection import detectOutOfPlaceDocuments


LOGGER = logging.getLogger(__name__)


class IngestionQueue:
    def __init__(self, max_workers: int = 2) -> None:
        self.max_workers = max(1, int(max_workers))
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="ingest-worker",
        )
        self._index_lock = Lock()
        self._futures: dict[str, Future[Any]] = {}
        self._futures_lock = Lock()

    def submit(
        self,
        file_path: Path,
        doc_id: str,
        embedder,
        vector_index,
        sparse_index=None,
        vlm=None,
    ) -> None:
        future = self._executor.submit(
            ingest_file,
            file_path,
            doc_id,
            embedder,
            vector_index,
            sparse_index,
            vlm,
            self._index_lock,
        )
        with self._futures_lock:
            self._futures[doc_id] = future
        future.add_done_callback(lambda completed, did=doc_id: self._on_done(did, completed))

    def _on_done(self, doc_id: str, future: Future[Any]) -> None:
        with self._futures_lock:
            self._futures.pop(doc_id, None)
        exc = future.exception()
        if exc is not None:
            LOGGER.warning("Ingestion task failed for %s: %s", doc_id, exc)
            return

        try:
            document = storage.get_document(doc_id) or {}
            metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
            folder_id = str(metadata.get("folder_id") or "").strip()
            expected_type = str(metadata.get("expected_doc_type") or "").strip()
            if not folder_id or not expected_type:
                return

            threshold = None
            if metadata.get("doc_review_threshold") is not None:
                try:
                    threshold = float(metadata.get("doc_review_threshold"))
                except Exception:
                    threshold = None

            flags = detectOutOfPlaceDocuments(
                folderId=folder_id,
                expectedType=expected_type,
                tenantId=str(metadata.get("tenant_id") or "").strip() or None,
                threshold=threshold,
            )
            flagged_ids = {str(item.get("fileId") or "") for item in flags}
            storage.update_document(
                doc_id,
                metadata={
                    "out_of_place_review_state": "needs_review" if doc_id in flagged_ids else "clear",
                    "out_of_place_review_last_flag_count": len(flags),
                },
                merge_metadata=True,
            )
        except Exception as review_exc:
            LOGGER.warning("Post-ingestion review failed for %s: %s", doc_id, review_exc)

    def pending_count(self) -> int:
        with self._futures_lock:
            return sum(1 for future in self._futures.values() if not future.done())

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)
