from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

from .contracts import (
    DisputeClaim,
    IssueReport,
    OracleDecision,
    RefereeVerdict,
    ScoreEvent,
    Scoreboard,
)


SEVERITY_POINTS: Dict[str, int] = {
    "low": 1,
    "some": 5,
    "critical": 10,
}


def severity_points(severity: str) -> int:
    return int(SEVERITY_POINTS.get(str(severity).strip().lower(), 0))


def score_round(
    *,
    round_id: str,
    issues: Iterable[IssueReport],
    disputes: Iterable[DisputeClaim],
    verdicts: Iterable[RefereeVerdict],
    oracle: Iterable[OracleDecision],
) -> Scoreboard:
    issues_list = list(issues)
    disputes_by_issue = {item.issue_id: item for item in disputes}
    verdicts_by_issue = {item.issue_id: item for item in verdicts}
    oracle_by_issue = {item.issue_id: item for item in oracle}

    totals_by_role: Dict[str, int] = defaultdict(int)
    totals_by_agent: Dict[str, int] = defaultdict(int)
    events: List[ScoreEvent] = []

    for issue in issues_list:
        oracle_item = oracle_by_issue.get(issue.issue_id)
        if oracle_item is None:
            continue

        points = severity_points(issue.severity)
        oracle_is_valid = oracle_item.label == "valid"

        bug_delta = points if oracle_is_valid else 0
        _record_score(
            events,
            totals_by_role,
            totals_by_agent,
            round_id=round_id,
            issue_id=issue.issue_id,
            role="bug-finder",
            agent_id=issue.bug_finder_agent_id,
            delta=bug_delta,
            reason="confirmed_bug" if oracle_is_valid else "not_confirmed",
        )

        dispute = disputes_by_issue.get(issue.issue_id)
        if dispute is not None:
            if oracle_is_valid:
                disprove_delta = -2 * points
                disprove_reason = "wrong_disprove"
            else:
                disprove_delta = points
                disprove_reason = "successful_disprove"
            _record_score(
                events,
                totals_by_role,
                totals_by_agent,
                round_id=round_id,
                issue_id=issue.issue_id,
                role="disprove-bug",
                agent_id=dispute.disprove_agent_id,
                delta=disprove_delta,
                reason=disprove_reason,
            )

        verdict = verdicts_by_issue.get(issue.issue_id)
        if verdict is not None:
            referee_correct = verdict.decision == oracle_item.label
            referee_delta = 1 if referee_correct else -1
            _record_score(
                events,
                totals_by_role,
                totals_by_agent,
                round_id=round_id,
                issue_id=issue.issue_id,
                role="referee",
                agent_id=verdict.referee_agent_id,
                delta=referee_delta,
                reason="correct_adjudication" if referee_correct else "incorrect_adjudication",
            )

    return Scoreboard(
        round_id=round_id,
        totals_by_role=dict(sorted(totals_by_role.items())),
        totals_by_agent=dict(sorted(totals_by_agent.items())),
        events=events,
    )


def _record_score(
    events: List[ScoreEvent],
    totals_by_role: Dict[str, int],
    totals_by_agent: Dict[str, int],
    *,
    round_id: str,
    issue_id: str,
    role: str,
    agent_id: str,
    delta: int,
    reason: str,
) -> None:
    totals_by_role[role] += int(delta)
    totals_by_agent[agent_id] += int(delta)
    events.append(
        ScoreEvent(
            round_id=round_id,
            issue_id=issue_id,
            agent_role=role,
            agent_id=agent_id,
            delta=int(delta),
            reason=reason,
        )
    )

