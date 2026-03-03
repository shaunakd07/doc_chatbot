from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol


Severity = Literal["low", "some", "critical"]
VerdictDecision = Literal["valid", "invalid", "inconclusive"]
OracleLabel = Literal["valid", "invalid"]

SEVERITY_LEVELS: tuple[Severity, ...] = ("low", "some", "critical")


@dataclass(frozen=True)
class EvidenceRef:
    source: str
    detail: str = ""
    path: str = ""
    line: Optional[int] = None


@dataclass
class HarnessCaseSnapshot:
    case_id: str
    title: str
    status: str
    source_path: str
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TournamentContext:
    harness_run_dir: str
    harness_summary: Dict[str, Any] = field(default_factory=dict)
    cases: List[HarnessCaseSnapshot] = field(default_factory=list)


@dataclass
class CodeQualityFinding:
    finding_id: str
    title: str
    severity: Severity
    summary: str
    evidence: List[EvidenceRef] = field(default_factory=list)


@dataclass
class CodeQualityReport:
    agent_id: str
    summary: str
    findings: List[CodeQualityFinding] = field(default_factory=list)


@dataclass
class IssueReport:
    issue_id: str
    case_id: str
    title: str
    description: str
    severity: Severity
    confidence: float
    evidence: List[EvidenceRef] = field(default_factory=list)
    fingerprint: str = ""
    bug_finder_agent_id: str = "bug-finder-agent"

    def normalized_fingerprint(self) -> str:
        if self.fingerprint:
            return self.fingerprint.strip().lower()
        return f"{self.case_id.strip().lower()}::{self.title.strip().lower()}"


@dataclass
class DisputeClaim:
    claim_id: str
    issue_id: str
    stance: Literal["disprove"]
    confidence_invalid: float
    rationale: str
    evidence: List[EvidenceRef] = field(default_factory=list)
    disprove_agent_id: str = "disprove-bug-agent"


@dataclass
class RefereeVerdict:
    verdict_id: str
    issue_id: str
    decision: VerdictDecision
    rationale: str
    evidence: List[EvidenceRef] = field(default_factory=list)
    referee_agent_id: str = "referee-agent"


@dataclass
class OracleDecision:
    issue_id: str
    label: OracleLabel
    basis: str
    confidence: float = 1.0


@dataclass
class ScoreEvent:
    round_id: str
    issue_id: str
    agent_role: str
    agent_id: str
    delta: int
    reason: str


@dataclass
class Scoreboard:
    round_id: str
    totals_by_role: Dict[str, int]
    totals_by_agent: Dict[str, int]
    events: List[ScoreEvent] = field(default_factory=list)


@dataclass
class TournamentResult:
    round_id: str
    context: TournamentContext
    code_quality: CodeQualityReport
    issues: List[IssueReport]
    disputes: List[DisputeClaim]
    verdicts: List[RefereeVerdict]
    oracle: List[OracleDecision]
    scoreboard: Scoreboard


class CodeQualityAgent(Protocol):
    agent_id: str

    def review(self, context: TournamentContext) -> CodeQualityReport:
        ...


class BugFinderAgent(Protocol):
    agent_id: str

    def find_bugs(self, context: TournamentContext, code_quality: CodeQualityReport) -> List[IssueReport]:
        ...


class DisproveBugAgent(Protocol):
    agent_id: str

    def challenge(self, context: TournamentContext, issues: List[IssueReport]) -> List[DisputeClaim]:
        ...


class RefereeAgent(Protocol):
    agent_id: str

    def adjudicate(
        self,
        context: TournamentContext,
        issues: List[IssueReport],
        disputes: List[DisputeClaim],
    ) -> List[RefereeVerdict]:
        ...


class Oracle(Protocol):
    def decide(
        self,
        context: TournamentContext,
        issues: List[IssueReport],
        disputes: List[DisputeClaim],
        verdicts: List[RefereeVerdict],
    ) -> List[OracleDecision]:
        ...

