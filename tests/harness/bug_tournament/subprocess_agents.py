from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from .contracts import (
    CodeQualityAgent,
    CodeQualityReport,
    DisproveBugAgent,
    DisputeClaim,
    IssueReport,
    Oracle,
    OracleDecision,
    RefereeAgent,
    RefereeVerdict,
    TournamentContext,
)
from .serde import (
    code_quality_from_dict,
    code_quality_to_dict,
    context_to_dict,
    disputes_from_dict,
    disputes_to_dict,
    issues_from_dict,
    issues_to_dict,
    oracle_from_dict,
    verdicts_from_dict,
    verdicts_to_dict,
)


def default_role_runner_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "scripts" / "bug_tournament_role_runner.py"


class _SubprocessRoleClient:
    def __init__(
        self,
        *,
        python_executable: Optional[str] = None,
        role_runner_path: Optional[Path] = None,
    ) -> None:
        self.python_executable = python_executable or sys.executable
        self.role_runner_path = (role_runner_path or default_role_runner_path()).resolve()

    def invoke(self, *, role: str, payload: Dict[str, object]) -> Dict[str, object]:
        if not self.role_runner_path.exists():
            raise RuntimeError(f"Role runner script not found: {self.role_runner_path}")

        with tempfile.TemporaryDirectory(prefix="bug-role-") as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.json"
            output_path = temp_path / "output.json"

            input_path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )

            command = [
                self.python_executable,
                str(self.role_runner_path),
                "--role",
                role,
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "Subprocess role invocation failed "
                    f"(role={role}, code={result.returncode}):\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )

            if not output_path.exists():
                raise RuntimeError(f"Subprocess role produced no output file: {output_path}")

            return json.loads(output_path.read_text(encoding="utf-8"))


class SubprocessCodeQualityAgent(CodeQualityAgent):
    agent_id = "code-quality-agent"

    def __init__(self, client: _SubprocessRoleClient) -> None:
        self._client = client

    def review(self, context: TournamentContext) -> CodeQualityReport:
        payload = {"context": context_to_dict(context)}
        response = self._client.invoke(role="code-quality", payload=payload)
        return code_quality_from_dict(dict(response.get("report") or {}))


class SubprocessBugFinderAgent:
    agent_id = "bug-finder-agent"

    def __init__(self, client: _SubprocessRoleClient) -> None:
        self._client = client

    def find_bugs(self, context: TournamentContext, code_quality: CodeQualityReport) -> List[IssueReport]:
        payload = {
            "context": context_to_dict(context),
            "code_quality": code_quality_to_dict(code_quality),
        }
        response = self._client.invoke(role="bug-finder", payload=payload)
        return issues_from_dict(list(response.get("issues") or []))


class SubprocessDisproveBugAgent(DisproveBugAgent):
    agent_id = "disprove-bug-agent"

    def __init__(self, client: _SubprocessRoleClient) -> None:
        self._client = client

    def challenge(self, context: TournamentContext, issues: List[IssueReport]) -> List[DisputeClaim]:
        payload = {
            "context": context_to_dict(context),
            "issues": issues_to_dict(issues),
        }
        response = self._client.invoke(role="disprove-bug", payload=payload)
        return disputes_from_dict(list(response.get("disputes") or []))


class SubprocessRefereeAgent(RefereeAgent):
    agent_id = "referee-agent"

    def __init__(self, client: _SubprocessRoleClient) -> None:
        self._client = client

    def adjudicate(
        self,
        context: TournamentContext,
        issues: List[IssueReport],
        disputes: List[DisputeClaim],
    ) -> List[RefereeVerdict]:
        payload = {
            "context": context_to_dict(context),
            "issues": issues_to_dict(issues),
            "disputes": disputes_to_dict(disputes),
        }
        response = self._client.invoke(role="referee", payload=payload)
        return verdicts_from_dict(list(response.get("verdicts") or []))


class SubprocessOracle(Oracle):
    def __init__(self, client: _SubprocessRoleClient) -> None:
        self._client = client

    def decide(
        self,
        context: TournamentContext,
        issues: List[IssueReport],
        disputes: List[DisputeClaim],
        verdicts: List[RefereeVerdict],
    ) -> List[OracleDecision]:
        payload = {
            "context": context_to_dict(context),
            "issues": issues_to_dict(issues),
            "disputes": disputes_to_dict(disputes),
            "verdicts": verdicts_to_dict(verdicts),
        }
        response = self._client.invoke(role="oracle", payload=payload)
        return oracle_from_dict(list(response.get("oracle") or []))


def create_subprocess_agents(
    *,
    python_executable: Optional[str] = None,
    role_runner_path: Optional[Path] = None,
) -> Dict[str, object]:
    client = _SubprocessRoleClient(
        python_executable=python_executable,
        role_runner_path=role_runner_path,
    )
    return {
        "code_quality_agent": SubprocessCodeQualityAgent(client),
        "bug_finder_agent": SubprocessBugFinderAgent(client),
        "disprove_bug_agent": SubprocessDisproveBugAgent(client),
        "referee_agent": SubprocessRefereeAgent(client),
        "oracle": SubprocessOracle(client),
    }

