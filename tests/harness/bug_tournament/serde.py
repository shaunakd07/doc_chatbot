from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

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


def context_to_dict(context: TournamentContext) -> Dict[str, Any]:
    return asdict(context)


def context_from_dict(payload: Dict[str, Any]) -> TournamentContext:
    raw_cases = payload.get("cases") or []
    cases = [harness_case_from_dict(item) for item in raw_cases]
    return TournamentContext(
        harness_run_dir=str(payload.get("harness_run_dir") or ""),
        harness_summary=dict(payload.get("harness_summary") or {}),
        cases=cases,
    )


def code_quality_to_dict(report: CodeQualityReport) -> Dict[str, Any]:
    return asdict(report)


def code_quality_from_dict(payload: Dict[str, Any]) -> CodeQualityReport:
    findings = [code_quality_finding_from_dict(item) for item in (payload.get("findings") or [])]
    return CodeQualityReport(
        agent_id=str(payload.get("agent_id") or ""),
        summary=str(payload.get("summary") or ""),
        findings=findings,
    )


def issues_to_dict(issues: List[IssueReport]) -> List[Dict[str, Any]]:
    return [asdict(item) for item in issues]


def issues_from_dict(payload: List[Dict[str, Any]]) -> List[IssueReport]:
    return [issue_from_dict(item) for item in payload]


def disputes_to_dict(disputes: List[DisputeClaim]) -> List[Dict[str, Any]]:
    return [asdict(item) for item in disputes]


def disputes_from_dict(payload: List[Dict[str, Any]]) -> List[DisputeClaim]:
    return [dispute_from_dict(item) for item in payload]


def verdicts_to_dict(verdicts: List[RefereeVerdict]) -> List[Dict[str, Any]]:
    return [asdict(item) for item in verdicts]


def verdicts_from_dict(payload: List[Dict[str, Any]]) -> List[RefereeVerdict]:
    return [verdict_from_dict(item) for item in payload]


def oracle_to_dict(decisions: List[OracleDecision]) -> List[Dict[str, Any]]:
    return [asdict(item) for item in decisions]


def oracle_from_dict(payload: List[Dict[str, Any]]) -> List[OracleDecision]:
    return [oracle_decision_from_dict(item) for item in payload]


def evidence_ref_from_dict(payload: Dict[str, Any]) -> EvidenceRef:
    line = payload.get("line")
    return EvidenceRef(
        source=str(payload.get("source") or ""),
        detail=str(payload.get("detail") or ""),
        path=str(payload.get("path") or ""),
        line=int(line) if line is not None else None,
    )


def code_quality_finding_from_dict(payload: Dict[str, Any]) -> CodeQualityFinding:
    evidence = [evidence_ref_from_dict(item) for item in (payload.get("evidence") or [])]
    return CodeQualityFinding(
        finding_id=str(payload.get("finding_id") or ""),
        title=str(payload.get("title") or ""),
        severity=str(payload.get("severity") or "low"),
        summary=str(payload.get("summary") or ""),
        evidence=evidence,
    )


def issue_from_dict(payload: Dict[str, Any]) -> IssueReport:
    evidence = [evidence_ref_from_dict(item) for item in (payload.get("evidence") or [])]
    return IssueReport(
        issue_id=str(payload.get("issue_id") or ""),
        case_id=str(payload.get("case_id") or ""),
        title=str(payload.get("title") or ""),
        description=str(payload.get("description") or ""),
        severity=str(payload.get("severity") or "low"),
        confidence=float(payload.get("confidence") or 0.0),
        evidence=evidence,
        fingerprint=str(payload.get("fingerprint") or ""),
        bug_finder_agent_id=str(payload.get("bug_finder_agent_id") or "bug-finder-agent"),
    )


def dispute_from_dict(payload: Dict[str, Any]) -> DisputeClaim:
    evidence = [evidence_ref_from_dict(item) for item in (payload.get("evidence") or [])]
    return DisputeClaim(
        claim_id=str(payload.get("claim_id") or ""),
        issue_id=str(payload.get("issue_id") or ""),
        stance="disprove",
        confidence_invalid=float(payload.get("confidence_invalid") or 0.0),
        rationale=str(payload.get("rationale") or ""),
        evidence=evidence,
        disprove_agent_id=str(payload.get("disprove_agent_id") or "disprove-bug-agent"),
    )


def verdict_from_dict(payload: Dict[str, Any]) -> RefereeVerdict:
    evidence = [evidence_ref_from_dict(item) for item in (payload.get("evidence") or [])]
    return RefereeVerdict(
        verdict_id=str(payload.get("verdict_id") or ""),
        issue_id=str(payload.get("issue_id") or ""),
        decision=str(payload.get("decision") or "inconclusive"),
        rationale=str(payload.get("rationale") or ""),
        evidence=evidence,
        referee_agent_id=str(payload.get("referee_agent_id") or "referee-agent"),
    )


def oracle_decision_from_dict(payload: Dict[str, Any]) -> OracleDecision:
    return OracleDecision(
        issue_id=str(payload.get("issue_id") or ""),
        label=str(payload.get("label") or "invalid"),
        basis=str(payload.get("basis") or ""),
        confidence=float(payload.get("confidence") or 0.0),
    )


def harness_case_from_dict(payload: Dict[str, Any]) -> HarnessCaseSnapshot:
    return HarnessCaseSnapshot(
        case_id=str(payload.get("case_id") or ""),
        title=str(payload.get("title") or ""),
        status=str(payload.get("status") or ""),
        source_path=str(payload.get("source_path") or ""),
        raw=dict(payload.get("raw") or {}),
    )

