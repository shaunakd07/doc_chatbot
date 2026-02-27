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

_FATAL_INFERENCE_MARKERS = (
    "convertpirattribute2runtimeattribute",
    "onednn_instruction.cc",
    "paddleocr.predict() got an unexpected keyword argument",
)


def _bool_from_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


def _is_fatal_inference_error(message: str) -> bool:
    text = str(message or "").strip().lower()
    if not text:
        return False
    return any(marker in text for marker in _FATAL_INFERENCE_MARKERS)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _engine_name_from_settings(settings: Dict[str, Any]) -> str:
    raw = str(settings.get("ocr_engine") or "tesseract").strip().lower()
    if raw not in {"paddle", "tesseract"}:
        return "tesseract"
    return raw


def _engine_label_from_name(name: str) -> str:
    return "paddleocr" if str(name).strip().lower() == "paddle" else "tesseract"


def _engine_label(settings: Dict[str, Any]) -> str:
    return _engine_label_from_name(_engine_name_from_settings(settings))


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


def _build_paddle_instance(settings: Dict[str, Any]) -> Tuple[object | None, str]:
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        return None, f"PaddleOCR import failed: {exc}"

    try:
        device = "gpu" if bool(settings.get("use_gpu", False)) else "cpu"
        lang = str(settings.get("lang") or "en")
        # PaddleOCR v3 uses `device`; legacy versions used `use_gpu`/`use_angle_cls`.
        try:
            engine = PaddleOCR(lang=lang, device=device)
        except TypeError:
            engine = PaddleOCR(
                lang=lang,
                use_gpu=bool(settings.get("use_gpu", False)),
                use_angle_cls=True,
                show_log=False,
            )
        return {"backend": "paddle", "instance": engine}, ""
    except Exception as exc:
        return None, f"PaddleOCR initialization failed: {exc}"


def _build_tesseract_instance(settings: Dict[str, Any]) -> Tuple[object | None, str]:
    try:
        import pytesseract  # type: ignore
    except Exception as exc:
        return None, f"pytesseract import failed: {exc}"

    tesseract_cmd = str(settings.get("tesseract_cmd") or "").strip()
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        _ = pytesseract.get_tesseract_version()
    except Exception as exc:
        return None, f"Tesseract initialization failed: {exc}"

    return {"backend": "tesseract", "instance": pytesseract}, ""


def _build_ocr_instance(settings: Dict[str, Any]) -> Tuple[object | None, str]:
    engine_name = _engine_name_from_settings(settings)
    if engine_name == "tesseract":
        return _build_tesseract_instance(settings)
    return _build_paddle_instance(settings)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _is_legacy_entry(entry: Any) -> bool:
    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
        return False
    line_data = entry[1]
    if not isinstance(line_data, (list, tuple)) or len(line_data) < 2:
        return False
    if isinstance(line_data[0], (list, tuple, dict)):
        return False
    try:
        float(line_data[1])
    except Exception:
        return False
    return True


def _extract_v3_payload(item: Any) -> Dict[str, Any] | None:
    if isinstance(item, dict):
        if isinstance(item.get("res"), dict):
            return item["res"]
        return item

    json_payload = getattr(item, "json", None)
    if isinstance(json_payload, dict):
        res = json_payload.get("res")
        if isinstance(res, dict):
            return res
    return None


def _normalize_paddle_output(raw: Any, min_confidence: float) -> Dict[str, object]:
    lines: list[str] = []
    confidences: list[float] = []

    # PaddleOCR <=2.x shape:
    #   [[ [box, [text, score]], ... ]]
    legacy_entries: list[Any] = []
    if isinstance(raw, list) and raw:
        if _is_legacy_entry(raw[0]):
            legacy_entries = raw
        elif isinstance(raw[0], list) and raw[0] and _is_legacy_entry(raw[0][0]):
            legacy_entries = raw[0]

    if legacy_entries:
        for entry in legacy_entries:
            line_data = entry[1]
            text = str(line_data[0] or "").strip()
            score = _safe_float(line_data[1], 0.0)
            if text and score >= min_confidence:
                lines.append(text)
                confidences.append(score)
    else:
        # PaddleOCR v3 returns OCRResult objects (or dict-like payloads)
        # with `rec_texts` and `rec_scores`.
        candidates = raw if isinstance(raw, list) else [raw]
        for candidate in candidates:
            payload = _extract_v3_payload(candidate)
            if not isinstance(payload, dict):
                continue

            rec_texts = payload.get("rec_texts")
            rec_scores = payload.get("rec_scores")
            if isinstance(rec_texts, str):
                rec_texts = [rec_texts]
            if not isinstance(rec_texts, list):
                continue
            if not isinstance(rec_scores, list):
                rec_scores = []

            for idx, raw_text in enumerate(rec_texts):
                text = str(raw_text or "").strip()
                if not text:
                    continue
                score = _safe_float(rec_scores[idx], 1.0) if idx < len(rec_scores) else 1.0
                if score < min_confidence:
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


def _normalize_tesseract_output(raw: Dict[str, Any], min_confidence: float) -> Dict[str, object]:
    texts = raw.get("text") if isinstance(raw, dict) else []
    confs = raw.get("conf") if isinstance(raw, dict) else []
    if not isinstance(texts, list):
        texts = []
    if not isinstance(confs, list):
        confs = []

    pages = raw.get("page_num") if isinstance(raw, dict) else []
    blocks = raw.get("block_num") if isinstance(raw, dict) else []
    pars = raw.get("par_num") if isinstance(raw, dict) else []
    lines = raw.get("line_num") if isinstance(raw, dict) else []
    if not isinstance(pages, list):
        pages = []
    if not isinstance(blocks, list):
        blocks = []
    if not isinstance(pars, list):
        pars = []
    if not isinstance(lines, list):
        lines = []

    grouped: Dict[Tuple[int, int, int, int], list[str]] = {}
    confidences: list[float] = []
    for idx, raw_text in enumerate(texts):
        text = str(raw_text or "").strip()
        if not text:
            continue
        conf_value = _safe_float(confs[idx], -1.0) if idx < len(confs) else -1.0
        if conf_value < 0:
            continue
        conf_norm = conf_value / 100.0
        if conf_norm < min_confidence:
            continue
        key = (
            _safe_int(pages[idx], 0) if idx < len(pages) else 0,
            _safe_int(blocks[idx], 0) if idx < len(blocks) else 0,
            _safe_int(pars[idx], 0) if idx < len(pars) else 0,
            _safe_int(lines[idx], 0) if idx < len(lines) else 0,
        )
        grouped.setdefault(key, []).append(text)
        confidences.append(conf_norm)

    line_texts = [
        " ".join(parts).strip()
        for _, parts in sorted(grouped.items(), key=lambda item: item[0])
        if parts
    ]
    line_texts = [line for line in line_texts if line]
    avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0
    return {
        "text": "\n".join(line_texts).strip(),
        "line_count": len(line_texts),
        "avg_confidence": round(avg_conf, 4),
        "engine": "tesseract",
        "status": "ok",
        "error": "",
    }


def _settings_from_env() -> Dict[str, Any]:
    enable_ocr_raw = os.getenv("ENABLE_OCR")
    if enable_ocr_raw is None:
        enabled = _bool_from_env("ENABLE_PADDLE_OCR", True)
    else:
        enabled = _bool_from_env("ENABLE_OCR", True)
    return {
        "ocr_engine": (os.getenv("OCR_ENGINE", "tesseract").strip().lower() or "tesseract"),
        "enabled": enabled,
        "lang": (os.getenv("PADDLE_OCR_LANG", "en").strip() or "en"),
        "use_gpu": _bool_from_env("PADDLE_OCR_USE_GPU", False),
        "min_confidence": float(os.getenv("PADDLE_OCR_MIN_CONFIDENCE", "0.50")),
        "max_retries": max(0, int(os.getenv("PADDLE_OCR_MAX_RETRIES", "1"))),
        "reinit_on_failure": _bool_from_env("PADDLE_OCR_REINIT_ON_FAILURE", False),
        "tesseract_cmd": os.getenv("OCR_TESSERACT_CMD", "").strip(),
        "tesseract_lang": (os.getenv("OCR_TESSERACT_LANG", "eng").strip() or "eng"),
        "tesseract_oem": max(0, _safe_int(os.getenv("OCR_TESSERACT_OEM", "1"), 1)),
        "tesseract_psm": max(0, _safe_int(os.getenv("OCR_TESSERACT_PSM", "3"), 3)),
        "runtime_disabled_error": "",
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
    engine = _engine_label(settings)
    return {
        "text": "",
        "line_count": 0,
        "avg_confidence": 0.0,
        "engine": engine,
        "status": "unavailable",
        "error": error or str(settings.get("init_error") or f"{engine} unavailable"),
    }


def _infer_with_paddle_engine(engine: object, image_array: Any) -> Any:
    predict = getattr(engine, "predict", None)
    if callable(predict):
        return predict(image_array)

    legacy_ocr = getattr(engine, "ocr", None)
    if callable(legacy_ocr):
        try:
            return legacy_ocr(image_array, cls=True)
        except TypeError:
            return legacy_ocr(image_array)

    raise RuntimeError("PaddleOCR engine has no supported inference method")


def _infer_with_tesseract_engine(
    pytesseract_module: Any,
    pil_image: Any,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    output_type = getattr(getattr(pytesseract_module, "Output", None), "DICT", None)
    if output_type is None:
        raise RuntimeError("pytesseract Output.DICT is unavailable")

    lang = str(settings.get("tesseract_lang") or "eng").strip() or "eng"
    oem = max(0, _safe_int(settings.get("tesseract_oem"), 1))
    psm = max(0, _safe_int(settings.get("tesseract_psm"), 3))
    tess_config = f"--oem {oem} --psm {psm}"
    return pytesseract_module.image_to_data(
        pil_image,
        lang=lang,
        config=tess_config,
        output_type=output_type,
    )


def _infer_with_engine(engine: object, image_array: Any, pil_image: Any, settings: Dict[str, Any]) -> Any:
    if isinstance(engine, dict) and str(engine.get("backend") or "").strip().lower() == "tesseract":
        module = engine.get("instance")
        return _infer_with_tesseract_engine(module, pil_image, settings)

    paddle_instance = engine.get("instance") if isinstance(engine, dict) else engine
    return _infer_with_paddle_engine(paddle_instance, image_array)


def _run_ocr(
    engine: object | None, image_b64: str, settings: Dict[str, Any]
) -> Tuple[Dict[str, object], object | None]:
    engine_label = _engine_label(settings)
    engine_backend = str((engine or {}).get("backend") or "").strip().lower() if isinstance(engine, dict) else "paddle"
    runtime_disabled_error = str(settings.get("runtime_disabled_error") or "").strip()
    if runtime_disabled_error:
        return (_unavailable(settings, runtime_disabled_error), engine)

    if engine is None:
        return (
            _unavailable(
                settings,
                str(settings.get("init_error") or f"{engine_label} worker unavailable"),
            ),
            engine,
        )

    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:
        return (
            {
                "text": "",
                "line_count": 0,
                "avg_confidence": 0.0,
                "engine": engine_label,
                "status": "error",
                "error": f"OCR worker dependency import failed: {exc}",
            },
            engine,
        )

    try:
        image_bytes = base64.b64decode(image_b64.encode("ascii"))
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_image = np.array(image)
    except Exception as exc:
        return (
            {
                "text": "",
                "line_count": 0,
                "avg_confidence": 0.0,
                "engine": engine_label,
                "status": "error",
                "error": f"Invalid OCR image payload: {exc}",
            },
            engine,
        )

    min_conf = float(settings.get("min_confidence", 0.5))
    max_retries = max(0, int(settings.get("max_retries", 1)))
    reinit_on_failure = bool(settings.get("reinit_on_failure", False))
    last_error = ""
    raw = None
    local_engine = engine
    for attempt in range(1, max_retries + 2):
        try:
            raw = _infer_with_engine(local_engine, np_image, image, settings)
            break
        except Exception as exc:
            last_error = str(exc)
            if engine_backend == "paddle" and _is_fatal_inference_error(last_error):
                fatal_error = f"Fatal paddle runtime error: {last_error}"
                logger.warning("%s", fatal_error)
                settings["runtime_disabled_error"] = fatal_error
                return (_unavailable(settings, fatal_error), local_engine)
            if engine_backend == "paddle" and attempt <= max_retries and reinit_on_failure:
                local_engine, init_error = _build_ocr_instance(settings)
                settings["init_error"] = init_error
                if local_engine is None:
                    break
            else:
                break

    if raw is None:
        return (
            {
                "text": "",
                "line_count": 0,
                "avg_confidence": 0.0,
                "engine": engine_label,
                "status": "error",
                "error": last_error or f"{engine_label} inference failed",
            },
            local_engine,
        )
    if engine_backend == "tesseract":
        return _normalize_tesseract_output(raw, min_conf), local_engine
    return _normalize_paddle_output(raw, min_conf), local_engine


def run_worker_loop(settings: Dict[str, Any]) -> int:
    worker_settings = dict(settings or {})
    engine_label = _engine_label(worker_settings)
    enabled = bool(worker_settings.get("enabled", True))
    engine = None
    init_error = ""

    if enabled:
        engine, init_error = _build_ocr_instance(worker_settings)
    else:
        init_error = "OCR is disabled"
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
                    "engine": engine_label,
                    "use_gpu": bool(worker_settings.get("use_gpu", False)) if engine_label == "paddleocr" else False,
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
                        "engine": engine_label,
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
                        "engine": engine_label,
                        "status": "error",
                        "error": "OCR worker received empty image payload",
                    },
                }
            )
            continue

        result, engine = _run_ocr(engine, image_b64, worker_settings)
        _write_json_line({"request_id": request_id, "result": result})


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    _configure_windows_dll_search_paths()
    settings = _settings_from_env()
    return run_worker_loop(settings)


if __name__ == "__main__":
    raise SystemExit(main())
