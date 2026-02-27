from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .types import ChatProbe, Condition, FileSelector, Profile, TestCase


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise RuntimeError("PyYAML is required to load harness scenario files.") from exc
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not parse YAML file: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"YAML root must be a mapping: {path}")
    return payload


def load_dataset_selectors(path: Path) -> Dict[str, FileSelector]:
    payload = _load_yaml(path)
    records = payload.get("datasets")
    if not isinstance(records, dict):
        raise RuntimeError(f"`datasets` must be a mapping in {path}")

    out: Dict[str, FileSelector] = {}
    for name, raw in records.items():
        if not isinstance(raw, dict):
            raise RuntimeError(f"Dataset selector `{name}` must be a mapping.")
        selector = FileSelector(
            kind=str(raw.get("kind") or "").strip(),
            path=raw.get("path"),
            root=raw.get("root"),
            pattern=raw.get("pattern"),
            n=int(raw["n"]) if raw.get("n") is not None else None,
            paths=[str(p) for p in (raw.get("paths") or [])],
            recursive=bool(raw.get("recursive", True)),
        )
        if not selector.kind:
            raise RuntimeError(f"Dataset selector `{name}` is missing `kind`.")
        out[str(name)] = selector
    return out


def _parse_probe(raw: Dict[str, Any]) -> ChatProbe:
    prompt = str(raw.get("prompt") or "").strip()
    if not prompt:
        raise RuntimeError("Chat probe is missing `prompt`.")
    return ChatProbe(
        prompt=prompt,
        top_k=int(raw["top_k"]) if raw.get("top_k") is not None else None,
        include_document_summaries=bool(raw.get("include_document_summaries", True)),
        use_case_doc_ids=bool(raw.get("use_case_doc_ids", True)),
        doc_ids=[str(v) for v in (raw.get("doc_ids") or [])],
        expect_error_contains=(
            str(raw.get("expect_error_contains") or "").strip() or None
        ),
        expected_doc_ids=[str(v) for v in (raw.get("expected_doc_ids") or [])],
        require_non_empty_answer=bool(raw.get("require_non_empty_answer", True)),
    )


def _parse_condition(raw: Dict[str, Any]) -> Condition:
    kind = str(raw.get("kind") or "").strip()
    if not kind:
        raise RuntimeError("Condition is missing `kind`.")
    return Condition(
        kind=kind,
        value=float(raw["value"]) if raw.get("value") is not None else None,
        values=[str(v) for v in (raw.get("values") or [])],
        table=str(raw.get("table") or "").strip() or None,
        source_types=[str(v) for v in (raw.get("source_types") or [])],
        doc_scope=str(raw.get("doc_scope") or "all"),
        min_count=int(raw["min_count"]) if raw.get("min_count") is not None else None,
        notes=str(raw.get("notes") or ""),
        raw=dict(raw),
    )


def load_cases(path: Path) -> List[TestCase]:
    payload = _load_yaml(path)
    defaults = payload.get("defaults") if isinstance(payload.get("defaults"), dict) else {}
    records = payload.get("cases")
    if not isinstance(records, list):
        raise RuntimeError(f"`cases` must be a list in {path}")

    out: List[TestCase] = []
    for raw in records:
        if not isinstance(raw, dict):
            raise RuntimeError("Every case entry must be a mapping.")
        case_id = str(raw.get("id") or "").strip()
        if not case_id:
            raise RuntimeError("Case is missing `id`.")

        probes = raw.get("chat_probes") or []
        checks = raw.get("checks") or []
        if not isinstance(probes, list) or not isinstance(checks, list):
            raise RuntimeError(f"Case `{case_id}` has invalid `chat_probes` or `checks`.")

        case = TestCase(
            case_id=case_id,
            suite=str(raw.get("suite") or "").strip(),
            title=str(raw.get("title") or "").strip(),
            objective=str(raw.get("objective") or "").strip(),
            tags=[str(v) for v in (raw.get("tags") or [])],
            datasets=[str(v) for v in (raw.get("datasets") or [])],
            chat_probes=[_parse_probe(p) for p in probes],
            checks=[_parse_condition(c) for c in checks],
            reset_before=bool(raw.get("reset_before", defaults.get("reset_before", True))),
            delete_after=bool(raw.get("delete_after", defaults.get("delete_after", False))),
            delete_all_after=bool(raw.get("delete_all_after", defaults.get("delete_all_after", False))),
            wait_ready=bool(raw.get("wait_ready", defaults.get("wait_ready", True))),
            ready_timeout_sec=int(raw.get("ready_timeout_sec", defaults.get("ready_timeout_sec", 1800))),
            poll_interval_sec=float(raw.get("poll_interval_sec", defaults.get("poll_interval_sec", 3.0))),
            automation_level=str(raw.get("automation_level") or defaults.get("automation_level", "full")),
            manual_steps=[str(v) for v in (raw.get("manual_steps") or [])],
            expected_notes=[str(v) for v in (raw.get("expected_notes") or [])],
            enabled=bool(raw.get("enabled", True)),
        )
        out.append(case)
    return out


def load_profiles(path: Path) -> Dict[str, Profile]:
    payload = _load_yaml(path)
    records = payload.get("profiles")
    if not isinstance(records, list):
        raise RuntimeError(f"`profiles` must be a list in {path}")

    out: Dict[str, Profile] = {}
    for raw in records:
        if not isinstance(raw, dict):
            raise RuntimeError("Profile entry must be a mapping.")
        name = str(raw.get("name") or "").strip()
        if not name:
            raise RuntimeError("Profile is missing `name`.")
        profile = Profile(
            name=name,
            description=str(raw.get("description") or "").strip(),
            include_suites=[str(v) for v in (raw.get("include_suites") or [])],
            include_tags=[str(v) for v in (raw.get("include_tags") or [])],
            include_case_ids=[str(v) for v in (raw.get("include_case_ids") or [])],
            exclude_case_ids=[str(v) for v in (raw.get("exclude_case_ids") or [])],
            allow_manual_cases=bool(raw.get("allow_manual_cases", False)),
            max_files_per_case=int(raw["max_files_per_case"]) if raw.get("max_files_per_case") is not None else None,
            stop_on_failure=bool(raw.get("stop_on_failure", False)),
        )
        out[profile.name] = profile
    return out


def select_cases(cases: List[TestCase], profile: Profile) -> List[TestCase]:
    selected: List[TestCase] = []
    include_case_ids = set(profile.include_case_ids)
    include_suites = {v.upper() for v in profile.include_suites}
    include_tags = {v.lower() for v in profile.include_tags}
    exclude_case_ids = set(profile.exclude_case_ids)

    for case in cases:
        if not case.enabled:
            continue
        if case.case_id in exclude_case_ids:
            continue
        if case.automation_level == "manual" and not profile.allow_manual_cases:
            continue

        include = False
        if include_case_ids and case.case_id in include_case_ids:
            include = True
        if include_suites and case.suite.upper() in include_suites:
            include = True
        if include_tags and include_tags.intersection({t.lower() for t in case.tags}):
            include = True
        if not include_case_ids and not include_suites and not include_tags:
            include = True

        if include:
            selected.append(case)
    return selected
