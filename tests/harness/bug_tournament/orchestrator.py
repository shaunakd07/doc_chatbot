from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from ..evidence import EvidenceWriter
from .contracts import (
    BugFinderAgent,
    CodeQualityAgent,
    DisproveBugAgent,
    HarnessCaseSnapshot,
    IssueReport,
    Oracle,
    RefereeAgent,
    TournamentContext,
    TournamentResult,
)
from .reporting import write_result_artifacts
from .roles import (
    DefaultBugFinderAgent,
    DefaultCodeQualityAgent,
    DefaultDisproveBugAgent,
    DefaultRefereeAgent,
    HarnessStatusOracle,
)
from .scoring import score_round


class TournamentOrchestrator:
    def __init__(
        self,
        *,
        evidence: EvidenceWriter,
        code_quality_agent: Optional[CodeQualityAgent] = None,
        bug_finder_agent: Optional[BugFinderAgent] = None,
        disprove_bug_agent: Optional[DisproveBugAgent] = None,
        referee_agent: Optional[RefereeAgent] = None,
        oracle: Optional[Oracle] = None,
    ) -> None:
        self.evidence = evidence
        self.code_quality_agent = code_quality_agent or DefaultCodeQualityAgent()
        self.bug_finder_agent = bug_finder_agent or DefaultBugFinderAgent()
        self.disprove_bug_agent = disprove_bug_agent or DefaultDisproveBugAgent()
        self.referee_agent = referee_agent or DefaultRefereeAgent()
        self.oracle = oracle or HarnessStatusOracle()

    def run(self, *, harness_run_dir: Path) -> TournamentResult:
        context = load_tournament_context(harness_run_dir)

        self.evidence.event(
            "tournament_start",
            {
                "round_id": self.evidence.run_id,
                "harness_run_dir": context.harness_run_dir,
                "case_count": len(context.cases),
            },
        )

        code_quality = self.code_quality_agent.review(context)
        self.evidence.event(
            "code_quality_completed",
            {
                "agent_id": code_quality.agent_id,
                "findings": len(code_quality.findings),
            },
        )

        raw_issues = self.bug_finder_agent.find_bugs(context, code_quality)
        issues = _dedupe_and_validate_issues(raw_issues)
        self.evidence.event(
            "bug_finder_completed",
            {
                "agent_id": self.bug_finder_agent.agent_id,
                "raw_issues": len(raw_issues),
                "accepted_issues": len(issues),
            },
        )

        disputes = self.disprove_bug_agent.challenge(context, issues)
        self.evidence.event(
            "disprove_completed",
            {
                "agent_id": self.disprove_bug_agent.agent_id,
                "claims": len(disputes),
            },
        )

        verdicts = self.referee_agent.adjudicate(context, issues, disputes)
        self.evidence.event(
            "referee_completed",
            {
                "agent_id": self.referee_agent.agent_id,
                "verdicts": len(verdicts),
            },
        )

        oracle_decisions = self.oracle.decide(context, issues, disputes, verdicts)
        self.evidence.event(
            "oracle_completed",
            {
                "decisions": len(oracle_decisions),
            },
        )

        scoreboard = score_round(
            round_id=self.evidence.run_id,
            issues=issues,
            disputes=disputes,
            verdicts=verdicts,
            oracle=oracle_decisions,
        )

        result = TournamentResult(
            round_id=self.evidence.run_id,
            context=context,
            code_quality=code_quality,
            issues=issues,
            disputes=disputes,
            verdicts=verdicts,
            oracle=oracle_decisions,
            scoreboard=scoreboard,
        )
        write_result_artifacts(self.evidence, result)
        self.evidence.event(
            "tournament_end",
            {
                "round_id": result.round_id,
                "totals_by_role": result.scoreboard.totals_by_role,
                "totals_by_agent": result.scoreboard.totals_by_agent,
            },
        )
        return result


def load_tournament_context(harness_run_dir: Path) -> TournamentContext:
    run_dir = harness_run_dir.resolve()
    summary_path = run_dir / "summary.json"
    cases_dir = run_dir / "snapshots" / "cases"

    summary = _load_json(summary_path) if summary_path.exists() else {}
    cases: List[HarnessCaseSnapshot] = []
    if cases_dir.exists() and cases_dir.is_dir():
        for snapshot_path in sorted(cases_dir.glob("*.json")):
            payload = _load_json(snapshot_path)
            case_id = str(payload.get("case_id") or snapshot_path.stem)
            title = str(payload.get("title") or "")
            status = str(payload.get("status") or "")
            cases.append(
                HarnessCaseSnapshot(
                    case_id=case_id,
                    title=title,
                    status=status,
                    source_path=snapshot_path.as_posix(),
                    raw=payload,
                )
            )

    return TournamentContext(
        harness_run_dir=run_dir.as_posix(),
        harness_summary=summary,
        cases=cases,
    )


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dedupe_and_validate_issues(issues: List[IssueReport]) -> List[IssueReport]:
    accepted: List[IssueReport] = []
    seen: set[str] = set()

    for issue in sorted(issues, key=lambda item: item.issue_id):
        if not _is_valid_issue(issue):
            continue
        fingerprint = issue.normalized_fingerprint()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        accepted.append(issue)

    return accepted


def _is_valid_issue(issue: IssueReport) -> bool:
    if issue.severity not in {"low", "some", "critical"}:
        return False
    if not issue.issue_id.strip():
        return False
    if not issue.case_id.strip():
        return False
    if not issue.title.strip():
        return False
    if not issue.evidence:
        return False
    return 0.0 <= float(issue.confidence) <= 1.0

