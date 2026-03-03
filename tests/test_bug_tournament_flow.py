import json
import tempfile
import unittest
from pathlib import Path

from tests.harness.bug_tournament.orchestrator import TournamentOrchestrator
from tests.harness.evidence import EvidenceWriter


class BugTournamentFlowTests(unittest.TestCase):
    def test_orchestrator_runs_and_writes_harness_style_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            harness_run_dir = root / "harness_source"
            cases_dir = harness_run_dir / "snapshots" / "cases"
            cases_dir.mkdir(parents=True, exist_ok=True)

            _write_json(
                harness_run_dir / "summary.json",
                {
                    "run_id": "source-run",
                    "selected_cases": 3,
                    "executed_cases": 3,
                    "failed_cases": 1,
                    "passed_cases": 1,
                    "skipped_cases": 0,
                },
            )
            _write_json(
                cases_dir / "A-001.json",
                {
                    "case_id": "A-001",
                    "title": "failure case",
                    "status": "fail",
                    "checks": [{"kind": "chat_non_empty", "status": "fail"}],
                },
            )
            _write_json(
                cases_dir / "A-002.json",
                {
                    "case_id": "A-002",
                    "title": "error case",
                    "status": "error",
                    "error": "traceback",
                },
            )
            _write_json(
                cases_dir / "A-003.json",
                {
                    "case_id": "A-003",
                    "title": "passing case",
                    "status": "pass",
                },
            )

            evidence = EvidenceWriter(root / "tournament_runs", "round-001")
            orchestrator = TournamentOrchestrator(evidence=evidence)
            result = orchestrator.run(harness_run_dir=harness_run_dir)

            self.assertEqual("round-001", result.round_id)
            self.assertEqual(3, len(result.context.cases))
            self.assertEqual(2, len(result.issues))
            self.assertEqual(0, len(result.disputes))
            self.assertEqual(2, len(result.verdicts))
            self.assertEqual(2, len(result.oracle))

            self.assertEqual(15, result.scoreboard.totals_by_role["bug-finder"])
            self.assertEqual(2, result.scoreboard.totals_by_role["referee"])

            summary_path = evidence.run_dir / "summary.json"
            self.assertTrue(summary_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(2, summary["issues_count"])
            self.assertEqual(2, summary["verdicts_count"])

            self.assertTrue((evidence.run_dir / "summary.md").exists())
            self.assertTrue((evidence.run_dir / "agents" / "scoreboard.json").exists())
            timeline_path = evidence.run_dir / "events" / "timeline.jsonl"
            self.assertTrue(timeline_path.exists())
            timeline_lines = timeline_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(timeline_lines), 2)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

