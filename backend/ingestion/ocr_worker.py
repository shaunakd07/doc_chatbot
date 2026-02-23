from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


logger = logging.getLogger(__name__)


def _configure_windows_dll_search_paths() -> None:
    if os.name != "nt":
        return

    python_path = Path(sys.executable).resolve()
    venv_root = python_path.parent.parent
    candidates = [
        venv_root / "Lib" / "site-packages" / "nvidia" / "cu13" / "bin" / "x86_64",
        venv_root / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin",
        venv_root / "Lib" / "site-packages" / "torch" / "lib",
    ]
    existing: list[str] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        existing.append(str(candidate))
        try:
            os.add_dll_directory(str(candidate))
        except Exception:
            pass
    if existing:
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = ";".join(existing + [current_path])


def _build_ocr_instance(settings: Dict[str, Any]) -> Tuple[object | None, str]:
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        return None, f"PaddleOCR import failed: {exc}"

    try:
        engine = PaddleOCR(
            use_angle_cls=True,
            lang=str(settings.get("lang") or "en"),
            use_gpu=bool(settings.get("use_gpu", False)),
            show_log=False,
        )
        return engine, ""
    except Exception as exc:
        return None, f"PaddleOCR initialization failed: {exc}"


def _normalize_ocr_output(raw: Any, min_confidence: float) -> Dict[str, object]:
    entries = []
    if isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, list):
            entries = first

    lines: list[str] = []
    confidences: list[float] = []
    for entry in entries:
        if not entry or len(entry) < 2:
            continue
        line_data = entry[1]
        if not isinstance(line_data, (list, tuple)) or len(line_data) < 2:
            continue
        text = str(line_data[0] or "").strip()
        try:
            score = float(line_data[1])
        except Exception:
            score = 0.0
        if not text or score < min_confidence:
            continue
        lines.append(text)
        confidences.append(score)

    avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0
    return {
        "text": "\n".join(lines).strip(),
        "line_count": len(lines),
        "avg_confidence": round(avg_conf, 4),
        "engine": "paddleocr",
        "status": "ok",
        "error": "",
    }


def _settings_from_env() -> Dict[str, Any]:
    def _bool(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "y"}

    return {
        "enabled": _bool("ENABLE_PADDLE_OCR", True),
        "lang": (os.getenv("PADDLE_OCR_LANG", "en").strip() or "en"),
        "use_gpu": _bool("PADDLE_OCR_USE_GPU", False),
        "min_confidence": float(os.getenv("PADDLE_OCR_MIN_CONFIDENCE", "0.50")),
        "max_retries": max(0, int(os.getenv("PADDLE_OCR_MAX_RETRIES", "1"))),
    }


def _read_json_line() -> Dict[str, Any] | None:
    line = sys.stdin.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return {}
    try:
        payload = json.loads(line)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _write_json_line(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _unavailable(settings: Dict[str, Any], error: str) -> Dict[str, object]:
    return {
        "text": "",
        "line_count": 0,
        "avg_confidence": 0.0,
        "engine": "paddleocr",
        "status": "unavailable",
        "error": error or str(settings.get("init_error") or "PaddleOCR unavailable"),
    }


def _run_ocr(engine: object | None, image_b64: str, settings: Dict[str, Any]) -> Dict[str, object]:
    if engine is None:
        return _unavailable(settings, str(settings.get("init_error") or "PaddleOCR worker unavailable"))

    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:
        return {
            "text": "",
            "line_count": 0,
            "avg_confidence": 0.0,
            "engine": "paddleocr",
            "status": "error",
            "error": f"OCR worker dependency import failed: {exc}",
        }

    try:
        image_bytes = base64.b64decode(image_b64.encode("ascii"))
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_image = np.array(image)
    except Exception as exc:
        return {
            "text": "",
            "line_count": 0,
            "avg_confidence": 0.0,
            "engine": "paddleocr",
            "status": "error",
            "error": f"Invalid OCR image payload: {exc}",
        }

    min_conf = float(settings.get("min_confidence", 0.5))
    max_retries = max(0, int(settings.get("max_retries", 1)))
    last_error = ""
    raw = None
    local_engine = engine
    for attempt in range(1, max_retries + 2):
        try:
            raw = local_engine.ocr(np_image, cls=True)
            break
        except Exception as exc:
            last_error = str(exc)
            if attempt <= max_retries:
                local_engine, init_error = _build_ocr_instance(settings)
                settings["init_error"] = init_error
                if local_engine is None:
                    break

    if raw is None:
        return {
            "text": "",
            "line_count": 0,
            "avg_confidence": 0.0,
            "engine": "paddleocr",
            "status": "error",
            "error": last_error or "PaddleOCR inference failed",
        }
    return _normalize_ocr_output(raw, min_conf)


def run_worker_loop(settings: Dict[str, Any]) -> int:
    worker_settings = dict(settings or {})
    enabled = bool(worker_settings.get("enabled", True))
    engine = None
    init_error = ""

    if enabled:
        engine, init_error = _build_ocr_instance(worker_settings)
    else:
        init_error = "PaddleOCR is disabled"
    worker_settings["init_error"] = init_error

    while True:
        message = _read_json_line()
        if message is None:
            return 0
        request_id = str(message.get("request_id") or "")
        req_type = str(message.get("type") or "").strip().lower()

        if req_type == "shutdown":
            _write_json_line({"request_id": request_id, "status": "ok"})
            return 0
        if req_type == "health":
            _write_json_line(
                {
                    "request_id": request_id,
                    "status": "ok" if engine is not None else "unavailable",
                    "engine": "paddleocr",
                    "use_gpu": bool(worker_settings.get("use_gpu", False)),
                    "error": init_error,
                }
            )
            continue
        if req_type != "ocr":
            _write_json_line(
                {
                    "request_id": request_id,
                    "result": {
                        "text": "",
                        "line_count": 0,
                        "avg_confidence": 0.0,
                        "engine": "paddleocr",
                        "status": "error",
                        "error": f"Unknown worker request type: {req_type}",
                    },
                }
            )
            continue

        image_b64 = message.get("image_b64")
        if not isinstance(image_b64, str) or not image_b64:
            _write_json_line(
                {
                    "request_id": request_id,
                    "result": {
                        "text": "",
                        "line_count": 0,
                        "avg_confidence": 0.0,
                        "engine": "paddleocr",
                        "status": "error",
                        "error": "OCR worker received empty image payload",
                    },
                }
            )
            continue

        result = _run_ocr(engine, image_b64, worker_settings)
        _write_json_line({"request_id": request_id, "result": result})


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    _configure_windows_dll_search_paths()
    settings = _settings_from_env()
    return run_worker_loop(settings)


if __name__ == "__main__":
    raise SystemExit(main())
