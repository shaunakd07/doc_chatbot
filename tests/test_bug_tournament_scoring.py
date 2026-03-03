import unittest

from tests.harness.bug_tournament.contracts import (
    DisputeClaim,
    EvidenceRef,
    IssueReport,
    OracleDecision,
    RefereeVerdict,
)
from tests.harness.bug_tournament.scoring import score_round, severity_points


class BugTournamentScoringTests(unittest.TestCase):
    def test_severity_points_mapping(self) -> None:
        self.assertEqual(1, severity_points("low"))
        self.assertEqual(5, severity_points("some"))
        self.assertEqual(10, severity_points("critical"))
        self.assertEqual(0, severity_points("unknown"))

    def test_valid_bug_rewards_bug_finder_and_referee(self) -> None:
        issue = IssueReport(
            issue_id="i-1",
            case_id="A-001",
            title="case failed",
            description="assertions failed",
            severity="some",
            confidence=0.9,
            evidence=[EvidenceRef(source="case_snapshot", path="/tmp/A-001.json")],
            bug_finder_agent_id="bf-1",
        )
        verdict = RefereeVerdict(
            verdict_id="v-1",
            issue_id="i-1",
            decision="valid",
            rationale="matches failure evidence",
            referee_agent_id="ref-1",
        )
        oracle = OracleDecision(issue_id="i-1", label="valid", basis="case_status:fail")

        scoreboard = score_round(
            round_id="r-1",
            issues=[issue],
            disputes=[],
            verdicts=[verdict],
            oracle=[oracle],
        )

        self.assertEqual(5, scoreboard.totals_by_role["bug-finder"])
        self.assertEqual(1, scoreboard.totals_by_role["referee"])
        self.assertEqual(5, scoreboard.totals_by_agent["bf-1"])
        self.assertEqual(1, scoreboard.totals_by_agent["ref-1"])

    def test_successful_disprove_gets_bug_score(self) -> None:
        issue = IssueReport(
            issue_id="i-2",
            case_id="A-002",
            title="possible issue",
            description="weak signal",
            severity="critical",
            confidence=0.2,
            evidence=[EvidenceRef(source="case_snapshot", path="/tmp/A-002.json")],
            bug_finder_agent_id="bf-2",
        )
        claim = DisputeClaim(
            claim_id="d-2",
            issue_id="i-2",
            stance="disprove",
            confidence_invalid=0.9,
            rationale="status is pass",
            disprove_agent_id="dp-1",
        )
        verdict = RefereeVerdict(
            verdict_id="v-2",
            issue_id="i-2",
            decision="invalid",
            rationale="claim is stronger",
            referee_agent_id="ref-2",
        )
        oracle = OracleDecision(issue_id="i-2", label="invalid", basis="case_status:pass")

        scoreboard = score_round(
            round_id="r-2",
            issues=[issue],
            disputes=[claim],
            verdicts=[verdict],
            oracle=[oracle],
        )

        self.assertEqual(0, scoreboard.totals_by_role["bug-finder"])
        self.assertEqual(10, scoreboard.totals_by_role["disprove-bug"])
        self.assertEqual(1, scoreboard.totals_by_role["referee"])
        self.assertEqual(10, scoreboard.totals_by_agent["dp-1"])

    def test_wrong_disprove_gets_double_penalty(self) -> None:
        issue = IssueReport(
            issue_id="i-3",
            case_id="A-003",
            title="real bug",
            description="error path",
            severity="low",
            confidence=0.95,
            evidence=[EvidenceRef(source="case_snapshot", path="/tmp/A-003.json")],
            bug_finder_agent_id="bf-3",
        )
        claim = DisputeClaim(
            claim_id="d-3",
            issue_id="i-3",
            stance="disprove",
            confidence_invalid=0.8,
            rationale="attempted disprove",
            disprove_agent_id="dp-2",
        )
        verdict = RefereeVerdict(
            verdict_id="v-3",
            issue_id="i-3",
            decision="invalid",
            rationale="mistaken",
            referee_agent_id="ref-3",
        )
        oracle = OracleDecision(issue_id="i-3", label="valid", basis="case_status:error")

        scoreboard = score_round(
            round_id="r-3",
            issues=[issue],
            disputes=[claim],
            verdicts=[verdict],
            oracle=[oracle],
        )

        self.assertEqual(1, scoreboard.totals_by_role["bug-finder"])
        self.assertEqual(-2, scoreboard.totals_by_role["disprove-bug"])
        self.assertEqual(-1, scoreboard.totals_by_role["referee"])
        self.assertEqual(-2, scoreboard.totals_by_agent["dp-2"])


if __name__ == "__main__":
    unittest.main()

