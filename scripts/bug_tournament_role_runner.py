from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one bug tournament role in isolation.")
    parser.add_argument(
        "--role",
        choices=["code-quality", "bug-finder", "disprove-bug", "referee", "oracle"],
        required=True,
        help="Role to execute.",
    )
    parser.add_argument("--input", required=True, help="Input JSON path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from tests.harness.bug_tournament.roles import (
        DefaultBugFinderAgent,
        DefaultCodeQualityAgent,
        DefaultDisproveBugAgent,
        DefaultRefereeAgent,
        HarnessStatusOracle,
    )
    from tests.harness.bug_tournament.serde import (
        code_quality_from_dict,
        code_quality_to_dict,
        context_from_dict,
        disputes_from_dict,
        disputes_to_dict,
        issues_from_dict,
        issues_to_dict,
        oracle_to_dict,
        verdicts_from_dict,
        verdicts_to_dict,
    )

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    payload = _load_json(input_path)
    context = context_from_dict(dict(payload.get("context") or {}))

    if args.role == "code-quality":
        role = DefaultCodeQualityAgent()
        report = role.review(context)
        result: Dict[str, Any] = {"report": code_quality_to_dict(report)}
    elif args.role == "bug-finder":
        role = DefaultBugFinderAgent()
        code_quality = code_quality_from_dict(dict(payload.get("code_quality") or {}))
        issues = role.find_bugs(context, code_quality)
        result = {"issues": issues_to_dict(issues)}
    elif args.role == "disprove-bug":
        role = DefaultDisproveBugAgent()
        issues = issues_from_dict(list(payload.get("issues") or []))
        disputes = role.challenge(context, issues)
        result = {"disputes": disputes_to_dict(disputes)}
    elif args.role == "referee":
        role = DefaultRefereeAgent()
        issues = issues_from_dict(list(payload.get("issues") or []))
        disputes = disputes_from_dict(list(payload.get("disputes") or []))
        verdicts = role.adjudicate(context, issues, disputes)
        result = {"verdicts": verdicts_to_dict(verdicts)}
    else:
        role = HarnessStatusOracle()
        issues = issues_from_dict(list(payload.get("issues") or []))
        disputes = disputes_from_dict(list(payload.get("disputes") or []))
        verdicts = verdicts_from_dict(list(payload.get("verdicts") or []))
        decisions = role.decide(context, issues, disputes, verdicts)
        result = {"oracle": oracle_to_dict(decisions)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return 0


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())

