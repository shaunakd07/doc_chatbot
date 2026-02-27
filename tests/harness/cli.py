from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shlex
import sys
from typing import List

from .api_client import ApiClient
from .config import load_cases, load_dataset_selectors, load_profiles, select_cases
from .evidence import EvidenceWriter
from .log_capture import ManagedAppProcess
from .runner import HarnessRunner
from .types import RunSummary


def _default_run_id(profile: str, mode: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{profile}_{mode}"


def _build_markdown_summary(summary: RunSummary) -> str:
    return "\n".join(
        [
            f"# Harness Run Summary: {summary.run_id}",
            "",
            f"- Profile: `{summary.profile}`",
            f"- Mode: `{summary.mode}`",
            f"- Started: `{summary.started_at}`",
            f"- Finished: `{summary.finished_at}`",
            f"- Duration sec: `{summary.duration_sec}`",
            "",
            "## Case Counts",
            "",
            f"- Selected: `{summary.selected_cases}`",
            f"- Executed: `{summary.executed_cases}`",
            f"- Passed: `{summary.passed_cases}`",
            f"- Failed: `{summary.failed_cases}`",
            f"- Skipped: `{summary.skipped_cases}`",
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="doc_chatbot ingestion/retrieval test harness",
    )
    parser.add_argument(
        "--mode",
        choices=["dry-run", "execute"],
        default="dry-run",
        help="dry-run emits a full execution plan without API calls.",
    )
    parser.add_argument(
        "--profile",
        default="smoke",
        help="Profile name from tests/harness/scenarios/profiles.yaml",
    )
    parser.add_argument(
        "--api-base-url",
        default="http://127.0.0.1:8000",
        help="Target API base URL.",
    )
    parser.add_argument(
        "--api-timeout-sec",
        type=float,
        default=120.0,
        help="HTTP timeout for each API call.",
    )
    parser.add_argument(
        "--datasets-file",
        default="tests/harness/scenarios/datasets.yaml",
        help="Dataset selectors YAML path.",
    )
    parser.add_argument(
        "--cases-file",
        default="tests/harness/scenarios/cases.yaml",
        help="Cases YAML path.",
    )
    parser.add_argument(
        "--profiles-file",
        default="tests/harness/scenarios/profiles.yaml",
        help="Profiles YAML path.",
    )
    parser.add_argument(
        "--evidence-root",
        default="harness_runs",
        help="Directory where run artifacts are written.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run id. If omitted, a timestamp-based id is generated.",
    )
    parser.add_argument(
        "--manage-app",
        action="store_true",
        help="Start/stop backend app process around harness execution.",
    )
    parser.add_argument(
        "--app-cmd",
        default="",
        help="Optional command for managed app mode. Default: current python -m backend.app",
    )
    parser.add_argument(
        "--startup-timeout-sec",
        type=float,
        default=120.0,
        help="Startup timeout for managed app mode.",
    )
    parser.add_argument(
        "--db-backend",
        default="",
        help="Override DB backend (sqlite/postgres). Defaults to env value.",
    )
    parser.add_argument(
        "--sqlite-db-path",
        default="",
        help="Override sqlite DB path for inspector.",
    )
    parser.add_argument(
        "--database-url",
        default="",
        help="Override postgres DATABASE_URL for inspector.",
    )
    return parser


def _parse_app_command(app_cmd: str) -> List[str]:
    cmd = str(app_cmd or "").strip()
    if cmd:
        return shlex.split(cmd, posix=False)
    return [sys.executable, "-m", "backend.app"]


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    datasets_path = (repo_root / args.datasets_file).resolve()
    cases_path = (repo_root / args.cases_file).resolve()
    profiles_path = (repo_root / args.profiles_file).resolve()
    evidence_root = (repo_root / args.evidence_root).resolve()

    selectors = load_dataset_selectors(datasets_path)
    cases = load_cases(cases_path)
    profiles = load_profiles(profiles_path)
    profile = profiles.get(args.profile)
    if profile is None:
        raise RuntimeError(f"Unknown profile: {args.profile}. Available: {sorted(profiles.keys())}")

    selected_cases = select_cases(cases, profile)
    run_id = args.run_id.strip() or _default_run_id(profile.name, args.mode)
    evidence = EvidenceWriter(evidence_root, run_id)

    managed_app = None
    if args.manage_app and args.mode == "execute":
        command = _parse_app_command(args.app_cmd)
        log_path = evidence.path_for_log("managed_app.log")
        managed_app = ManagedAppProcess(
            command=command,
            cwd=repo_root,
            log_path=log_path,
            startup_timeout_sec=args.startup_timeout_sec,
        )
        with ApiClient(args.api_base_url, timeout_sec=args.api_timeout_sec) as api:
            managed_app.start(health_check=lambda: bool(api.health().get("status") == "ok"))

    try:
        runner = HarnessRunner(
            repo_root=repo_root,
            mode=args.mode,
            profile=profile,
            selectors=selectors,
            cases=selected_cases,
            evidence=evidence,
            api_base_url=args.api_base_url,
            api_timeout_sec=args.api_timeout_sec,
            db_backend=(args.db_backend.strip() or None),
            sqlite_db_path=(args.sqlite_db_path.strip() or None),
            database_url=(args.database_url.strip() or None),
        )
        summary = runner.run()
    finally:
        if managed_app is not None:
            managed_app.stop()

    summary_md = _build_markdown_summary(summary)
    evidence.write_text("summary.md", summary_md)
    print(summary_md)
    print(f"Evidence directory: {evidence.run_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

