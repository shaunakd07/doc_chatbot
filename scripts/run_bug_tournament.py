from __future__ import annotations

import argparse
from datetime import datetime, timezone
import sys
from pathlib import Path


def _default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_bug_tournament"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-agent bug tournament over harness artifacts.")
    parser.add_argument(
        "--harness-run-dir",
        required=True,
        help="Path to an existing harness run directory containing summary.json and snapshots/cases/*.json",
    )
    parser.add_argument(
        "--evidence-root",
        default="harness_runs",
        help="Directory where tournament artifacts are written.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional tournament run id. Defaults to a timestamp-based id.",
    )
    parser.add_argument(
        "--agent-execution",
        choices=["subprocess", "in-process"],
        default="subprocess",
        help="How to execute role agents. Default uses one subprocess per role.",
    )
    parser.add_argument(
        "--role-runner-path",
        default="",
        help="Optional path to role runner script when using subprocess execution.",
    )
    parser.add_argument(
        "--agent-python-executable",
        default="",
        help="Optional python executable for subprocess role agents.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from tests.harness.bug_tournament.orchestrator import TournamentOrchestrator
    from tests.harness.bug_tournament.subprocess_agents import create_subprocess_agents
    from tests.harness.evidence import EvidenceWriter

    harness_run_dir = (repo_root / args.harness_run_dir).resolve()
    evidence_root = (repo_root / args.evidence_root).resolve()
    run_id = args.run_id.strip() or _default_run_id()

    if not harness_run_dir.exists():
        raise RuntimeError(f"Harness run directory does not exist: {harness_run_dir}")

    evidence = EvidenceWriter(evidence_root, run_id)
    if args.agent_execution == "subprocess":
        role_runner_path = (repo_root / args.role_runner_path).resolve() if args.role_runner_path else None
        agent_python = args.agent_python_executable.strip() or None
        subprocess_agents = create_subprocess_agents(
            python_executable=agent_python,
            role_runner_path=role_runner_path,
        )
        orchestrator = TournamentOrchestrator(
            evidence=evidence,
            code_quality_agent=subprocess_agents["code_quality_agent"],
            bug_finder_agent=subprocess_agents["bug_finder_agent"],
            disprove_bug_agent=subprocess_agents["disprove_bug_agent"],
            referee_agent=subprocess_agents["referee_agent"],
            oracle=subprocess_agents["oracle"],
        )
    else:
        orchestrator = TournamentOrchestrator(evidence=evidence)
    result = orchestrator.run(harness_run_dir=harness_run_dir)

    summary_md = (evidence.run_dir / "summary.md").read_text(encoding="utf-8")
    print(summary_md)
    print(f"Evidence directory: {evidence.run_dir.as_posix()}")
    print(f"Totals by role: {result.scoreboard.totals_by_role}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
