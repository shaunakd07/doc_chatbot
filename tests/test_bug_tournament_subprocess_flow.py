import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class BugTournamentSubprocessFlowTests(unittest.TestCase):
    def test_cli_uses_subprocess_agents_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            harness_run_dir = root / "harness_input"
            cases_dir = harness_run_dir / "snapshots" / "cases"
            cases_dir.mkdir(parents=True, exist_ok=True)

            _write_json(
                harness_run_dir / "summary.json",
                {
                    "run_id": "input-001",
                    "selected_cases": 2,
                    "executed_cases": 2,
                    "failed_cases": 1,
                    "passed_cases": 1,
                    "skipped_cases": 0,
                },
            )
            _write_json(
                cases_dir / "A-010.json",
                {
                    "case_id": "A-010",
                    "title": "failing case",
                    "status": "fail",
                },
            )
            _write_json(
                cases_dir / "A-011.json",
                {
                    "case_id": "A-011",
                    "title": "passing case",
                    "status": "pass",
                },
            )

            evidence_root = root / "tournament_output"
            run_id = "subprocess-round"

            repo_root = Path(__file__).resolve().parents[1]
            script_path = repo_root / "scripts" / "run_bug_tournament.py"
            command = [
                sys.executable,
                str(script_path),
                "--harness-run-dir",
                harness_run_dir.as_posix(),
                "--evidence-root",
                evidence_root.as_posix(),
                "--run-id",
                run_id,
            ]
            completed = subprocess.run(
                command,
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                self.fail(
                    "Subprocess CLI failed:\n"
                    f"stdout:\n{completed.stdout}\n"
                    f"stderr:\n{completed.stderr}"
                )

            run_dir = evidence_root / run_id
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "summary.md").exists())
            self.assertTrue((run_dir / "agents" / "scoreboard.json").exists())
            self.assertTrue((run_dir / "events" / "timeline.jsonl").exists())

            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(1, summary["issues_count"])
            self.assertEqual(0, summary["disputes_count"])

            scoreboard = json.loads((run_dir / "agents" / "scoreboard.json").read_text(encoding="utf-8"))
            self.assertEqual(5, scoreboard["totals_by_role"]["bug-finder"])
            self.assertEqual(1, scoreboard["totals_by_role"]["referee"])


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

