from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

from ..ingestion.pipeline import ingest_file


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

    def pending_count(self) -> int:
        with self._futures_lock:
            return sum(1 for future in self._futures.values() if not future.done())

    def shutdown(self, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=False)
