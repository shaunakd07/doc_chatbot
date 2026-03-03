from __future__ import annotations

from collections import Counter
from typing import Dict, List

from .contracts import (
    CodeQualityFinding,
    CodeQualityReport,
    DisputeClaim,
    EvidenceRef,
    HarnessCaseSnapshot,
    IssueReport,
    OracleDecision,
    RefereeVerdict,
    TournamentContext,
)


def _clamp_confidence(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return round(value, 3)


def _case_status(case: HarnessCaseSnapshot) -> str:
    return str(case.status or "").strip().lower()


class DefaultCodeQualityAgent:
    agent_id = "code-quality-agent"

    def review(self, context: TournamentContext) -> CodeQualityReport:
        counts = Counter(_case_status(case) for case in context.cases)
        findings: List[CodeQualityFinding] = []

        if counts.get("error", 0) > 0:
            findings.append(
                CodeQualityFinding(
                    finding_id="cq-error-cases",
                    title="Harness cases ended in error",
                    severity="critical",
                    summary="At least one harness case ended with an error status.",
                    evidence=[
                        EvidenceRef(
                            source="harness_summary",
                            detail=f"error_cases={counts.get('error', 0)}",
                            path=context.harness_run_dir,
                        )
                    ],
                )
            )

        if counts.get("fail", 0) > 0:
            findings.append(
                CodeQualityFinding(
                    finding_id="cq-failed-cases",
                    title="Harness cases failed assertions",
                    severity="some",
                    summary="One or more harness cases failed checks.",
                    evidence=[
                        EvidenceRef(
                            source="harness_summary",
                            detail=f"failed_cases={counts.get('fail', 0)}",
                            path=context.harness_run_dir,
                        )
                    ],
                )
            )

        summary = (
            f"reviewed_cases={len(context.cases)} "
            f"failed={counts.get('fail', 0)} "
            f"error={counts.get('error', 0)}"
        )
        return CodeQualityReport(agent_id=self.agent_id, summary=summary, findings=findings)


class DefaultBugFinderAgent:
    agent_id = "bug-finder-agent"

    def find_bugs(self, context: TournamentContext, code_quality: CodeQualityReport) -> List[IssueReport]:
        del code_quality
        issues: List[IssueReport] = []

        for case in context.cases:
            status = _case_status(case)
            if status not in {"fail", "error"}:
                continue

            severity = "some"
            confidence = 0.82
            reason = "case failed harness checks"
            if status == "error":
                severity = "critical"
                confidence = 0.96
                reason = "case ended with runtime error"
            elif bool((case.raw.get("wait") or {}).get("timed_out")):
                severity = "critical"
                confidence = 0.92
                reason = "case timed out waiting for terminal document status"

            issue = IssueReport(
                issue_id=f"issue-{case.case_id.lower()}",
                case_id=case.case_id,
                title=f"{case.case_id}: {case.title}",
                description=reason,
                severity=severity,
                confidence=_clamp_confidence(confidence),
                evidence=[
                    EvidenceRef(
                        source="case_snapshot",
                        detail=f"status={status}",
                        path=case.source_path,
                    )
                ],
                fingerprint=f"{case.case_id.lower()}::{status}",
                bug_finder_agent_id=self.agent_id,
            )
            issues.append(issue)

        return sorted(issues, key=lambda item: item.issue_id)


class DefaultDisproveBugAgent:
    agent_id = "disprove-bug-agent"

    _abstain_thresholds: Dict[str, float] = {
        "low": 0.68,
        "some": 0.79,
        "critical": 0.91,
    }

    def challenge(self, context: TournamentContext, issues: List[IssueReport]) -> List[DisputeClaim]:
        del context
        claims: List[DisputeClaim] = []
        for issue in issues:
            threshold = self._abstain_thresholds.get(issue.severity, 0.79)
            confidence_invalid = _clamp_confidence(1.0 - float(issue.confidence))
            if confidence_invalid < threshold:
                continue
            claims.append(
                DisputeClaim(
                    claim_id=f"dispute-{issue.issue_id}",
                    issue_id=issue.issue_id,
                    stance="disprove",
                    confidence_invalid=confidence_invalid,
                    rationale="issue confidence below risk threshold for this severity",
                    evidence=list(issue.evidence),
                    disprove_agent_id=self.agent_id,
                )
            )
        return sorted(claims, key=lambda item: item.claim_id)


class DefaultRefereeAgent:
    agent_id = "referee-agent"

    def adjudicate(
        self,
        context: TournamentContext,
        issues: List[IssueReport],
        disputes: List[DisputeClaim],
    ) -> List[RefereeVerdict]:
        del context
        claims_by_issue = {claim.issue_id: claim for claim in disputes}
        verdicts: List[RefereeVerdict] = []

        for issue in issues:
            claim = claims_by_issue.get(issue.issue_id)
            if claim is None:
                decision = "valid"
                rationale = "no disprove claim; bug-finder evidence stands"
                evidence = list(issue.evidence)
            elif claim.confidence_invalid > float(issue.confidence):
                decision = "invalid"
                rationale = "disprove confidence outweighs bug confidence"
                evidence = list(claim.evidence)
            else:
                decision = "valid"
                rationale = "bug confidence outweighs disprove confidence"
                evidence = list(issue.evidence)

            verdicts.append(
                RefereeVerdict(
                    verdict_id=f"verdict-{issue.issue_id}",
                    issue_id=issue.issue_id,
                    decision=decision,
                    rationale=rationale,
                    evidence=evidence,
                    referee_agent_id=self.agent_id,
                )
            )

        return sorted(verdicts, key=lambda item: item.verdict_id)


class HarnessStatusOracle:
    def decide(
        self,
        context: TournamentContext,
        issues: List[IssueReport],
        disputes: List[DisputeClaim],
        verdicts: List[RefereeVerdict],
    ) -> List[OracleDecision]:
        del disputes
        del verdicts
        case_by_id = {case.case_id: case for case in context.cases}
        decisions: List[OracleDecision] = []
        for issue in issues:
            case = case_by_id.get(issue.case_id)
            if case is None:
                decisions.append(
                    OracleDecision(
                        issue_id=issue.issue_id,
                        label="invalid",
                        basis="missing_case_snapshot",
                        confidence=0.0,
                    )
                )
                continue

            status = _case_status(case)
            if status in {"fail", "error"}:
                label = "valid"
                basis = f"case_status:{status}"
                confidence = 0.99
            else:
                label = "invalid"
                basis = f"case_status:{status or 'unknown'}"
                confidence = 0.8

            decisions.append(
                OracleDecision(
                    issue_id=issue.issue_id,
                    label=label,
                    basis=basis,
                    confidence=confidence,
                )
            )
        return sorted(decisions, key=lambda item: item.issue_id)

