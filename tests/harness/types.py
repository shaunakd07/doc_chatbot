from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FileSelector:
    kind: str
    path: Optional[str] = None
    root: Optional[str] = None
    pattern: Optional[str] = None
    n: Optional[int] = None
    paths: List[str] = field(default_factory=list)
    recursive: bool = True


@dataclass
class ChatProbe:
    prompt: str
    top_k: Optional[int] = None
    include_document_summaries: bool = True
    use_case_doc_ids: bool = True
    doc_ids: List[str] = field(default_factory=list)
    expect_error_contains: Optional[str] = None
    expected_doc_ids: List[str] = field(default_factory=list)
    require_non_empty_answer: bool = True


@dataclass
class Condition:
    kind: str
    value: Optional[float] = None
    values: List[str] = field(default_factory=list)
    table: Optional[str] = None
    source_types: List[str] = field(default_factory=list)
    doc_scope: str = "all"
    min_count: Optional[int] = None
    notes: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestCase:
    case_id: str
    suite: str
    title: str
    objective: str
    tags: List[str] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)
    chat_probes: List[ChatProbe] = field(default_factory=list)
    checks: List[Condition] = field(default_factory=list)
    reset_before: bool = True
    delete_after: bool = False
    delete_all_after: bool = False
    wait_ready: bool = True
    ready_timeout_sec: int = 1800
    poll_interval_sec: float = 3.0
    automation_level: str = "full"
    manual_steps: List[str] = field(default_factory=list)
    expected_notes: List[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class Profile:
    name: str
    description: str
    include_suites: List[str] = field(default_factory=list)
    include_tags: List[str] = field(default_factory=list)
    include_case_ids: List[str] = field(default_factory=list)
    exclude_case_ids: List[str] = field(default_factory=list)
    allow_manual_cases: bool = False
    max_files_per_case: Optional[int] = None
    stop_on_failure: bool = False


@dataclass
class HarnessSettings:
    api_base_url: str = "http://127.0.0.1:8000"
    test_docs_root: str = "test_docs"
    evidence_root: str = "harness_runs"
    startup_timeout_sec: float = 120.0
    request_timeout_sec: float = 120.0


@dataclass
class RunSummary:
    run_id: str
    profile: str
    mode: str
    selected_cases: int
    executed_cases: int
    passed_cases: int
    failed_cases: int
    skipped_cases: int
    started_at: str
    finished_at: str
    duration_sec: float
