from __future__ import annotations

import logging
import math
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

from .. import config

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import networkx as nx
except Exception:  # pragma: no cover - optional dependency
    nx = None

_YOLO_IMPORT_ERROR: str | None = None
try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - optional dependency
    YOLO = None
    _YOLO_IMPORT_ERROR = str(exc)


LOGGER = logging.getLogger(__name__)
_YOLO_MODEL = None
_YOLO_LOCK = Lock()
_YOLO_INIT_ERROR: str | None = None
_OCR_EXTRACTOR = None
_OCR_IMPORT_ERROR: str | None = None


def _bbox_center(bbox: Dict[str, int]) -> Tuple[float, float]:
    return (
        float((bbox["x1"] + bbox["x2"]) / 2.0),
        float((bbox["y1"] + bbox["y2"]) / 2.0),
    )


def _bbox_area(bbox: Dict[str, int]) -> int:
    return max(0, int(bbox["x2"] - bbox["x1"])) * max(0, int(bbox["y2"] - bbox["y1"]))


def _clamp_bbox(bbox: Dict[str, int], width: int, height: int) -> Dict[str, int]:
    x1 = max(0, min(width - 1, int(bbox["x1"])))
    y1 = max(0, min(height - 1, int(bbox["y1"])))
    x2 = max(0, min(width - 1, int(bbox["x2"])))
    y2 = max(0, min(height - 1, int(bbox["y2"])))
    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _normalized_bbox(bbox: Dict[str, int], width: int, height: int) -> Dict[str, float]:
    w = max(1, int(width))
    h = max(1, int(height))
    bw = max(1, int(bbox["x2"] - bbox["x1"]))
    bh = max(1, int(bbox["y2"] - bbox["y1"]))
    return {
        "x": round(float(bbox["x1"]) / float(w), 5),
        "y": round(float(bbox["y1"]) / float(h), 5),
        "w": round(float(bw) / float(w), 5),
        "h": round(float(bh) / float(h), 5),
        "x1": int(bbox["x1"]),
        "y1": int(bbox["y1"]),
        "x2": int(bbox["x2"]),
        "y2": int(bbox["y2"]),
    }


def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def _detect_nodes_opencv(gray: np.ndarray, width: int, height: int) -> List[Dict[str, Any]]:
    if cv2 is None:
        return []

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5,
    )
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = max(1, int(config.DIAGRAM_MIN_NODE_AREA))
    max_nodes = max(1, int(config.DIAGRAM_MAX_NODES))
    candidates: List[Dict[str, Any]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 16 or h < 12:
            continue
        bbox = _clamp_bbox({"x1": x, "y1": y, "x2": x + w, "y2": y + h}, width, height)
        area = _bbox_area(bbox)
        if area < min_area:
            continue
        approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)
        shape_kind = "box" if len(approx) in {4, 5} else "region"
        candidates.append(
            {
                "bbox": bbox,
                "score": round(min(1.0, 0.45 + (float(area) / float(width * height + 1))), 4),
                "kind": shape_kind,
                "detector": "opencv",
            }
        )
    candidates = sorted(candidates, key=lambda item: (_bbox_area(item["bbox"]), item["score"]), reverse=True)
    return candidates[:max_nodes]


def _yolo_predict_kwargs(max_nodes: int) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "conf": float(config.YOLO_CONF_THRESHOLD),
        "iou": float(config.YOLO_IOU_THRESHOLD),
        "imgsz": int(config.YOLO_IMAGE_SIZE),
        "max_det": int(max_nodes),
        "verbose": False,
    }
    device = str(config.YOLO_DEVICE).strip()
    if device and device.lower() != "auto":
        kwargs["device"] = device
    return kwargs


def _resolve_yolo_model_ref() -> str:
    model_ref = str(config.YOLO_MODEL).strip() or "yolo26n.pt"
    model_path = Path(model_ref)
    if model_path.is_absolute() or model_path.parent != Path("."):
        return str(model_path)
    cache_dir = (config.DATA_DIR / "models" / "yolo").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str((cache_dir / model_path.name).resolve())


def _get_yolo_model():
    global _YOLO_MODEL, _YOLO_INIT_ERROR

    if not config.ENABLE_YOLO_DIAGRAM_DETECTOR:
        return None
    if YOLO is None:
        if _YOLO_IMPORT_ERROR:
            LOGGER.warning("YOLO import unavailable: %s", _YOLO_IMPORT_ERROR)
        return None
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    if _YOLO_INIT_ERROR is not None:
        return None

    with _YOLO_LOCK:
        if _YOLO_MODEL is not None:
            return _YOLO_MODEL
        if _YOLO_INIT_ERROR is not None:
            return None
        try:
            _YOLO_MODEL = YOLO(_resolve_yolo_model_ref())
        except Exception as exc:
            _YOLO_INIT_ERROR = str(exc)
            LOGGER.warning("YOLO model init failed: %s", exc)
            return None
    return _YOLO_MODEL


def _detect_nodes_yolo(image_bgr: np.ndarray, width: int, height: int) -> List[Dict[str, Any]]:
    model = _get_yolo_model()
    if model is None:
        return []

    max_nodes = max(1, int(config.DIAGRAM_MAX_NODES))
    predict_kwargs = _yolo_predict_kwargs(max_nodes)
    with _YOLO_LOCK:
        try:
            predictions = model.predict(source=image_bgr, **predict_kwargs)
        except Exception as exc:
            LOGGER.warning("YOLO inference failed: %s", exc)
            return []

    if not predictions:
        return []
    result = predictions[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) <= 0:
        return []

    names = getattr(result, "names", {}) or {}
    min_area = max(1, int(config.DIAGRAM_MIN_NODE_AREA))
    detections: List[Dict[str, Any]] = []
    for box in boxes:
        try:
            xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
            x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy[:4]]
            score = float(box.conf[0].detach().cpu().item()) if getattr(box, "conf", None) is not None else 1.0
            cls_idx = int(box.cls[0].detach().cpu().item()) if getattr(box, "cls", None) is not None else -1
        except Exception:
            continue
        bbox = _clamp_bbox({"x1": x1, "y1": y1, "x2": x2, "y2": y2}, width, height)
        if _bbox_area(bbox) < min_area:
            continue
        kind = str(names.get(cls_idx, f"class_{cls_idx}"))
        detections.append(
            {
                "bbox": bbox,
                "score": round(score, 4),
                "kind": kind,
                "detector": "yolo",
            }
        )
    detections = sorted(detections, key=lambda item: item.get("score", 0.0), reverse=True)
    return detections[:max_nodes]


def _detect_lines(gray: np.ndarray) -> List[Dict[str, Any]]:
    if cv2 is None:
        return []
    edges = cv2.Canny(gray, 80, 160, apertureSize=3)
    raw = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=60,
        minLineLength=max(12, int(config.DIAGRAM_MIN_EDGE_LENGTH)),
        maxLineGap=12,
    )
    if raw is None:
        return []

    min_length = max(1, int(config.DIAGRAM_MIN_EDGE_LENGTH))
    results: List[Dict[str, Any]] = []
    for row in raw:
        x1, y1, x2, y2 = [int(v) for v in row[0]]
        length = int(math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2)))
        if length < min_length:
            continue
        results.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "length": length})
    max_edges = max(1, int(config.DIAGRAM_MAX_EDGES))
    return sorted(results, key=lambda item: item["length"], reverse=True)[: max_edges * 2]


def _get_ocr_extractor():
    global _OCR_EXTRACTOR, _OCR_IMPORT_ERROR
    if _OCR_EXTRACTOR is not None:
        return _OCR_EXTRACTOR
    if _OCR_IMPORT_ERROR is not None:
        return None
    try:
        from .ocr import extract_text_from_image
    except Exception as exc:
        _OCR_IMPORT_ERROR = str(exc)
        LOGGER.warning("OCR extractor import failed: %s", exc)
        return None
    _OCR_EXTRACTOR = extract_text_from_image
    return _OCR_EXTRACTOR


def _label_for_bbox(image_rgb: np.ndarray, bbox: Dict[str, int]) -> str:
    if not config.ENABLE_PADDLE_OCR:
        return ""
    extractor = _get_ocr_extractor()
    if extractor is None:
        return ""
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    crop = image_rgb[y1:y2, x1:x2]
    if crop.size <= 0:
        return ""
    pil_crop = Image.fromarray(crop)
    result = extractor(pil_crop)
    return str(result.get("text") or "").strip()


def _build_nodes(image_rgb: np.ndarray, candidates: List[Dict[str, Any]], width: int, height: int) -> List[Dict[str, Any]]:
    nodes: List[Dict[str, Any]] = []
    max_nodes = max(1, int(config.DIAGRAM_MAX_NODES))
    for idx, candidate in enumerate(candidates[:max_nodes], start=1):
        bbox = _clamp_bbox(candidate["bbox"], width, height)
        center = _bbox_center(bbox)
        label = _label_for_bbox(image_rgb, bbox)
        node_id = f"node_{idx:03d}"
        nodes.append(
            {
                "id": node_id,
                "label": label or f"Region {idx}",
                "has_ocr_label": bool(label),
                "score": float(candidate.get("score", 0.5)),
                "detector": str(candidate.get("detector") or "opencv"),
                "node_kind": str(candidate.get("kind") or "region"),
                "bbox": _normalized_bbox(bbox, width, height),
                "center_x": round(center[0], 2),
                "center_y": round(center[1], 2),
            }
        )
    return nodes


def _nearest_node(point: Tuple[float, float], nodes: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not nodes:
        return None
    best = None
    best_dist = None
    for node in nodes:
        center = (float(node.get("center_x", 0.0)), float(node.get("center_y", 0.0)))
        dist = _distance(point, center)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = node
    return best


def _assign_edges(nodes: List[Dict[str, Any]], lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not nodes:
        return []
    max_edges = max(1, int(config.DIAGRAM_MAX_EDGES))
    dedupe: set[Tuple[str, str]] = set()
    edges: List[Dict[str, Any]] = []
    for line in lines:
        p1 = (float(line["x1"]), float(line["y1"]))
        p2 = (float(line["x2"]), float(line["y2"]))
        n1 = _nearest_node(p1, nodes)
        n2 = _nearest_node(p2, nodes)
        if n1 is None or n2 is None:
            continue
        if n1["id"] == n2["id"]:
            continue

        dx = float(n2["center_x"]) - float(n1["center_x"])
        dy = float(n2["center_y"]) - float(n1["center_y"])
        if abs(dx) >= abs(dy):
            source, target = (n1, n2) if float(n1["center_x"]) <= float(n2["center_x"]) else (n2, n1)
            direction = "left_to_right"
        else:
            source, target = (n1, n2) if float(n1["center_y"]) <= float(n2["center_y"]) else (n2, n1)
            direction = "top_to_bottom"

        key = (str(source["id"]), str(target["id"]))
        if key in dedupe:
            continue
        dedupe.add(key)
        edges.append(
            {
                "id": f"edge_{len(edges) + 1:03d}",
                "from": source["id"],
                "to": target["id"],
                "line": {
                    "x1": int(line["x1"]),
                    "y1": int(line["y1"]),
                    "x2": int(line["x2"]),
                    "y2": int(line["y2"]),
                },
                "length": int(line["length"]),
                "direction_hint": direction,
                "type": "line_connection",
            }
        )
        if len(edges) >= max_edges:
            break
    return edges


def _graph_metrics(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    if nx is None:
        return {
            "connected_components": 0,
            "largest_component": 0,
            "density": 0.0,
            "library": "none",
        }
    graph = nx.DiGraph()
    for node in nodes:
        graph.add_node(node["id"])
    for edge in edges:
        graph.add_edge(edge["from"], edge["to"])
    if graph.number_of_nodes() == 0:
        return {
            "connected_components": 0,
            "largest_component": 0,
            "density": 0.0,
            "library": "networkx",
        }
    weak_components = list(nx.weakly_connected_components(graph))
    largest = max((len(component) for component in weak_components), default=0)
    density = float(nx.density(graph)) if graph.number_of_nodes() > 1 else 0.0
    return {
        "connected_components": len(weak_components),
        "largest_component": int(largest),
        "density": round(density, 4),
        "library": "networkx",
    }


def _summary_text(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], metrics: Dict[str, Any]) -> str:
    lines = [
        "Detected diagram relationship graph from image.",
        f"Nodes: {len(nodes)}. Edges: {len(edges)}. Connected components: {metrics.get('connected_components', 0)}.",
    ]
    if nodes:
        top_nodes = nodes[:8]
        labels = [str(node.get("label") or node.get("id")) for node in top_nodes]
        lines.append("Key nodes: " + "; ".join(labels))
    if edges:
        top_edges = edges[:12]
        edge_lines = [f"{edge.get('from')} -> {edge.get('to')} ({edge.get('direction_hint')})" for edge in top_edges]
        lines.append("Relationships: " + "; ".join(edge_lines))
    return "\n".join(lines).strip()


def parse_image_diagram(
    image: Image.Image,
    *,
    page: int,
    image_path: str = "",
) -> Dict[str, Any]:
    if not config.ENABLE_DIAGRAM_PIPELINE:
        return {"status": "disabled", "parser_version": "diagram-v1"}
    if cv2 is None:
        return {
            "status": "unavailable",
            "parser_version": "diagram-v1",
            "error": "opencv-python-headless is not installed",
        }

    rgb = np.array(image.convert("RGB"))
    if rgb.size <= 0:
        return {"status": "empty", "parser_version": "diagram-v1"}

    height, width = rgb.shape[0], rgb.shape[1]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    yolo_nodes = _detect_nodes_yolo(bgr, width, height)
    fallback_nodes = _detect_nodes_opencv(gray, width, height)
    candidates = yolo_nodes if yolo_nodes else fallback_nodes
    lines = _detect_lines(gray)

    if not candidates and not lines:
        return {
            "status": "no_structure",
            "parser_version": "diagram-v1",
            "summary_text": "",
            "graph": {},
            "node_chunks": [],
            "edge_chunks": [],
            "confidence": 0.0,
        }

    nodes = _build_nodes(rgb, candidates, width, height)
    edges = _assign_edges(nodes, lines)
    metrics = _graph_metrics(nodes, edges)
    summary = _summary_text(nodes, edges, metrics)
    score = 0.2 + min(0.35, len(nodes) * 0.015) + min(0.35, len(edges) * 0.02)
    if yolo_nodes:
        score += 0.1
    score = max(0.0, min(0.99, score))

    graph = {
        "kind": "image_diagram_graph",
        "page": int(page),
        "image_path": image_path,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
        "metrics": metrics,
    }

    node_chunks = [
        (
            f"Diagram node {node['id']}: label={node['label']}; kind={node['node_kind']}; "
            f"bbox=({node['bbox']['x']:.3f},{node['bbox']['y']:.3f},{node['bbox']['w']:.3f},{node['bbox']['h']:.3f}); "
            f"detector={node['detector']}; score={node['score']:.3f}"
        )
        for node in nodes[: max(0, int(config.DIAGRAM_MAX_NODE_CHUNKS))]
    ]
    edge_chunks = [
        (
            f"Diagram edge {edge['id']}: {edge['from']} -> {edge['to']}; type={edge['type']}; "
            f"direction_hint={edge['direction_hint']}; length={edge['length']}"
        )
        for edge in edges[: max(0, int(config.DIAGRAM_MAX_EDGE_CHUNKS))]
    ]
    return {
        "status": "ok",
        "parser_version": "diagram-v1",
        "summary_text": summary,
        "graph": graph,
        "node_chunks": node_chunks,
        "edge_chunks": edge_chunks,
        "confidence": round(score, 4),
        "metrics": metrics,
    }
