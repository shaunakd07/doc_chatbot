from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from ..evidence import EvidenceWriter
from .contracts import TournamentResult


def build_summary_payload(result: TournamentResult) -> Dict[str, Any]:
    return {
        "round_id": result.round_id,
        "harness_run_dir": result.context.harness_run_dir,
        "harness_case_count": len(result.context.cases),
        "issues_count": len(result.issues),
        "disputes_count": len(result.disputes),
        "verdicts_count": len(result.verdicts),
        "oracle_count": len(result.oracle),
        "totals_by_role": result.scoreboard.totals_by_role,
        "totals_by_agent": result.scoreboard.totals_by_agent,
    }


def build_markdown_summary(result: TournamentResult) -> str:
    lines = [
        f"# Bug Tournament Summary: {result.round_id}",
        "",
        f"- Harness source: `{result.context.harness_run_dir}`",
        f"- Harness cases: `{len(result.context.cases)}`",
        f"- Issues: `{len(result.issues)}`",
        f"- Disputes: `{len(result.disputes)}`",
        f"- Referee verdicts: `{len(result.verdicts)}`",
        f"- Oracle decisions: `{len(result.oracle)}`",
        "",
        "## Totals By Role",
        "",
    ]
    for role, score in sorted(result.scoreboard.totals_by_role.items()):
        lines.append(f"- {role}: `{score}`")

    lines.extend(["", "## Totals By Agent", ""])
    for agent_id, score in sorted(result.scoreboard.totals_by_agent.items()):
        lines.append(f"- {agent_id}: `{score}`")

    return "\n".join(lines) + "\n"


def write_result_artifacts(evidence: EvidenceWriter, result: TournamentResult) -> None:
    evidence.write_json("summary.json", build_summary_payload(result))
    evidence.write_text("summary.md", build_markdown_summary(result))

    evidence.write_json("agents/code_quality.json", asdict(result.code_quality))
    evidence.write_json("agents/bug_finder_issues.json", [asdict(item) for item in result.issues])
    evidence.write_json("agents/disprove_claims.json", [asdict(item) for item in result.disputes])
    evidence.write_json("agents/referee_verdicts.json", [asdict(item) for item in result.verdicts])
    evidence.write_json("agents/oracle_decisions.json", [asdict(item) for item in result.oracle])
    evidence.write_json("agents/scoreboard.json", asdict(result.scoreboard))

