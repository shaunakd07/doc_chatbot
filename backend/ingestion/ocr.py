from __future__ import annotations

import atexit
import base64
import io
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from typing import Any, Dict

from PIL import Image

from .. import config

logger = logging.getLogger(__name__)

_WORKER_LOCK = threading.Lock()
_WORKER_PROCESS: subprocess.Popen | None = None
_WORKER_RESPONSES: queue.Queue[Dict[str, Any]] | None = None
_WORKER_STDOUT_THREAD: threading.Thread | None = None
_WORKER_STDERR_THREAD: threading.Thread | None = None
_WORKER_INIT_ERROR: str | None = None


def _unavailable_response(error: str) -> Dict[str, object]:
    return {
        "text": "",
        "line_count": 0,
        "avg_confidence": 0.0,
        "engine": "paddleocr",
        "status": "unavailable",
        "error": str(error or "PaddleOCR worker unavailable"),
    }


def _error_response(error: str) -> Dict[str, object]:
    return {
        "text": "",
        "line_count": 0,
        "avg_confidence": 0.0,
        "engine": "paddleocr",
        "status": "error",
        "error": str(error or "PaddleOCR worker failed"),
    }


def _worker_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["ENABLE_PADDLE_OCR"] = "true" if config.ENABLE_PADDLE_OCR else "false"
    env["PADDLE_OCR_LANG"] = str(config.PADDLE_OCR_LANG)
    env["PADDLE_OCR_USE_GPU"] = "true" if config.PADDLE_OCR_USE_GPU else "false"
    env["PADDLE_OCR_MIN_CONFIDENCE"] = str(config.PADDLE_OCR_MIN_CONFIDENCE)
    env["PADDLE_OCR_MAX_RETRIES"] = str(config.PADDLE_OCR_MAX_RETRIES)
    return env


def _stdout_reader(proc: subprocess.Popen, response_queue: queue.Queue[Dict[str, Any]]) -> None:
    if proc.stdout is None:
        return
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        try:
            payload = json.loads(line.strip())
            if isinstance(payload, dict):
                response_queue.put(payload)
        except Exception:
            continue


def _stderr_reader(proc: subprocess.Popen) -> None:
    if proc.stderr is None:
        return
    for line in proc.stderr:
        text = line.strip()
        if text:
            logger.warning("OCR worker: %s", text)


def _shutdown_worker_locked() -> None:
    global _WORKER_PROCESS, _WORKER_RESPONSES, _WORKER_STDOUT_THREAD, _WORKER_STDERR_THREAD, _WORKER_INIT_ERROR

    proc = _WORKER_PROCESS
    response_queue = _WORKER_RESPONSES
    _WORKER_PROCESS = None
    _WORKER_RESPONSES = None
    _WORKER_STDOUT_THREAD = None
    _WORKER_STDERR_THREAD = None

    if proc is not None:
        try:
            if proc.stdin is not None:
                request_id = str(uuid.uuid4())
                proc.stdin.write(json.dumps({"request_id": request_id, "type": "shutdown"}) + "\n")
                proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=2.0)
        except Exception:
            pass
        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=2.0)
            except Exception:
                pass
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except Exception:
            pass
        try:
            if proc.stderr is not None:
                proc.stderr.close()
        except Exception:
            pass

    if response_queue is not None:
        while not response_queue.empty():
            try:
                response_queue.get_nowait()
            except Exception:
                break

    _WORKER_INIT_ERROR = None


def _shutdown_worker() -> None:
    with _WORKER_LOCK:
        _shutdown_worker_locked()


def _send_request_locked(payload: Dict[str, Any], timeout_sec: float) -> Dict[str, Any]:
    proc = _WORKER_PROCESS
    response_queue = _WORKER_RESPONSES
    if proc is None or response_queue is None or proc.stdin is None:
        raise RuntimeError("OCR worker is not initialized")

    request_id = str(uuid.uuid4())
    message = dict(payload)
    message["request_id"] = request_id
    proc.stdin.write(json.dumps(message, ensure_ascii=True) + "\n")
    proc.stdin.flush()

    deadline = time.monotonic() + timeout_sec
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise queue.Empty("Timed out waiting for OCR worker response")
        if proc.poll() is not None:
            raise RuntimeError(f"OCR worker exited unexpectedly (code={proc.returncode})")
        try:
            response = response_queue.get(timeout=min(1.0, remaining))
        except queue.Empty:
            continue
        if str(response.get("request_id") or "") != request_id:
            continue
        return response


def _start_worker_locked() -> bool:
    global _WORKER_PROCESS, _WORKER_RESPONSES, _WORKER_STDOUT_THREAD, _WORKER_STDERR_THREAD, _WORKER_INIT_ERROR

    startup_timeout = max(5.0, float(config.OCR_WORKER_STARTUP_TIMEOUT_SEC))
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "backend.ingestion.ocr_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=_worker_env(),
        )
    except Exception as exc:
        _WORKER_INIT_ERROR = f"Failed to start OCR worker process: {exc}"
        logger.exception("Failed to start OCR worker process: %s", exc)
        return False

    responses: queue.Queue[Dict[str, Any]] = queue.Queue()
    stdout_thread = threading.Thread(
        target=_stdout_reader,
        args=(proc, responses),
        daemon=True,
        name="ocr-worker-stdout-reader",
    )
    stderr_thread = threading.Thread(
        target=_stderr_reader,
        args=(proc,),
        daemon=True,
        name="ocr-worker-stderr-reader",
    )
    stdout_thread.start()
    stderr_thread.start()

    _WORKER_PROCESS = proc
    _WORKER_RESPONSES = responses
    _WORKER_STDOUT_THREAD = stdout_thread
    _WORKER_STDERR_THREAD = stderr_thread

    try:
        health = _send_request_locked({"type": "health"}, startup_timeout)
    except Exception as exc:
        _WORKER_INIT_ERROR = f"OCR worker health check failed: {exc}"
        logger.warning("%s", _WORKER_INIT_ERROR)
        _shutdown_worker_locked()
        return False

    if str(health.get("status") or "") != "ok":
        _WORKER_INIT_ERROR = str(health.get("error") or "OCR worker reported unavailable")
        logger.warning("OCR worker unavailable: %s", _WORKER_INIT_ERROR)
        _shutdown_worker_locked()
        return False

    _WORKER_INIT_ERROR = None
    return True


def _ensure_worker_locked() -> bool:
    if not config.ENABLE_PADDLE_OCR:
        return False

    proc = _WORKER_PROCESS
    if proc is not None and proc.poll() is None and _WORKER_RESPONSES is not None:
        return True

    _shutdown_worker_locked()
    return _start_worker_locked()


def extract_text_from_image(image: Image.Image) -> Dict[str, object]:
    """
    Runs PaddleOCR in an isolated subprocess and returns normalized OCR output.
    """

    if not config.ENABLE_PADDLE_OCR:
        return _unavailable_response("PaddleOCR is disabled")

    rgb = image.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    timeout_sec = max(2.0, float(config.OCR_WORKER_TIMEOUT_SEC))

    with _WORKER_LOCK:
        if not _ensure_worker_locked():
            return _unavailable_response(_WORKER_INIT_ERROR or "OCR worker unavailable")
        try:
            response = _send_request_locked(
                {"type": "ocr", "image_b64": image_b64},
                timeout_sec=timeout_sec,
            )
        except queue.Empty:
            logger.warning("OCR worker timed out")
            _shutdown_worker_locked()
            return _error_response("OCR worker timed out")
        except Exception as exc:
            logger.warning("OCR worker request failed: %s", exc)
            _shutdown_worker_locked()
            return _error_response(f"OCR worker request failed: {exc}")

    result = response.get("result")
    if not isinstance(result, dict):
        return _error_response("OCR worker returned invalid response payload")
    return {
        "text": str(result.get("text") or "").strip(),
        "line_count": int(result.get("line_count") or 0),
        "avg_confidence": float(result.get("avg_confidence") or 0.0),
        "engine": str(result.get("engine") or "paddleocr"),
        "status": str(result.get("status") or "error"),
        "error": str(result.get("error") or ""),
    }


atexit.register(_shutdown_worker)
