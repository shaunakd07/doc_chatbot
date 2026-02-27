from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _to_jsonable(payload: Any) -> Any:
    if is_dataclass(payload):
        return asdict(payload)
    if isinstance(payload, dict):
        return {str(k): _to_jsonable(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [_to_jsonable(v) for v in payload]
    if isinstance(payload, tuple):
        return [_to_jsonable(v) for v in payload]
    return payload


class EvidenceWriter:
    def __init__(self, root_dir: Path, run_id: str) -> None:
        self.root_dir = root_dir
        self.run_id = run_id
        self.run_dir = (root_dir / run_id).resolve()
        self.events_dir = self.run_dir / "events"
        self.api_dir = self.run_dir / "api"
        self.db_dir = self.run_dir / "db"
        self.logs_dir = self.run_dir / "logs"
        self.snapshots_dir = self.run_dir / "snapshots"
        for directory in (
            self.run_dir,
            self.events_dir,
            self.api_dir,
            self.db_dir,
            self.logs_dir,
            self.snapshots_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def path_for_log(self, name: str) -> Path:
        return self.logs_dir / name

    def write_json(self, relative_path: str, payload: Any) -> Path:
        target = self.run_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(_to_jsonable(payload), ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return target

    def write_text(self, relative_path: str, text: str) -> Path:
        target = self.run_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(text), encoding="utf-8")
        return target

    def append_jsonl(self, relative_path: str, payload: Dict[str, Any]) -> Path:
        target = self.run_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_to_jsonable(payload), ensure_ascii=True) + "\n")
        return target

    def event(self, event_type: str, payload: Dict[str, Any]) -> None:
        record = {
            "time_utc": utc_now_iso(),
            "event_type": str(event_type),
            "payload": _to_jsonable(payload),
        }
        self.append_jsonl("events/timeline.jsonl", record)

