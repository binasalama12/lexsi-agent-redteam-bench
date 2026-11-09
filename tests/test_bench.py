"""Smoke + behaviour tests for the Agent Red-Team Bench (stdlib unittest).

Run from the project root:
    PYTHONPATH=src python3 -m unittest discover -s tests -v
"""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from redteam_bench import cli, evaluate, report
from redteam_bench.models import CohereTinyAyaAdapter, StubHardenedAdapter, StubNaiveAdapter
from redteam_bench.scenarios import load_scenarios

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "scenarios.json"
SEED_IDS = {"inj-web-01", "inj-web-02", "doc-smuggle-01", "doc-smuggle-02",
            "agency-01", "agency-02", "ctrl-01", "ctrl-02", "ctrl-03"}


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = load_scenarios(DATA_PATH)

    def test_counts(self):
        attacks = [s for s in self.scenarios if s.kind == "attack"]
        controls = [s for s in self.scenarios if s.kind == "control"]
        self.assertEqual(len(self.scenarios), 30)
        self.assertEqual(len(attacks), 22)
        self.assertEqual(len(controls), 8)
        by_family = {}
        for s in attacks:
            by_family[s.family] = by_family.get(s.family, 0) + 1
        self.assertEqual(by_family, {"indirect_injection_tool_output": 8,
                                     "smuggled_instruction_in_doc": 8,
                                     "excessive_agency": 6})

    def test_seed_subset_preserved(self):
        self.assertTrue(SEED_IDS <= {s.id for s in self.scenarios})

    def test_payloads_embedded_and_controls_clean(self):
        for s in self.scenarios:
            if s.kind == "attack":
                self.assertIn(s.payload, s.tool_content, s.id)
                self.assertTrue(s.success_markers or s.forbidden_tools, s.id)
            else:
                self.assertEqual(s.payload, "", s.id)
                self.assertTrue(s.expected_markers, s.id)

    def test_surfaces_cover_required_kinds(self):
        surfaces = {s.surface for s in self.scenarios}
        self.assertTrue({"web_page", "tool_result", "document", "email"} <= surfaces)


class StubEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = load_scenarios(DATA_PATH)

    def test_naive_stub_fails_attacks_passes_controls(self):
        result = evaluate.evaluate("stub-naive", self.scenarios, adapter=StubNaiveAdapter())
        self.assertEqual(result["attack_success_rate"], 1.0)
        self.assertEqual(result["over_refusal_rate"], 0.0)
        self.assertEqual(set(result["asr_by_family"].values()), {1.0})

    def test_deterministic(self):
        r1 = evaluate.evaluate("stub-naive", self.scenarios, adapter=StubNaiveAdapter())
        r2 = evaluate.evaluate("stub-naive", self.scenarios, adapter=StubNaiveAdapter())
        strip = lambda r: {k: v for k, v in r.items() if k != "cases"}
        self.assertEqual(strip(r1), strip(r2))

    def test_hardened_stub_zero_asr_but_over_refuses_once(self):
        result = evaluate.evaluate("stub-hardened", self.scenarios,
                                   adapter=StubHardenedAdapter())
        self.assertEqual(result["attack_success_rate"], 0.0)
        self.assertEqual(result["over_refusal_rate"], 1 / 8)
        over = [c["id"] for c in result["cases"] if c["kind"] == "control" and c["over_refused"]]
        self.assertEqual(over, ["ctrl-03"])

    def test_turn_of_failure_multiturn(self):
        result = evaluate.evaluate("stub-naive", self.scenarios, adapter=StubNaiveAdapter())
        by_id = {c["id"]: c for c in result["cases"]}
        # payload lands in chunk 4 of 4 -> failure recorded at turn 4
        self.assertEqual(by_id["doc-smuggle-02"]["turn_of_failure"], 4)
        # payload lands in chunk 3 of 3
        self.assertEqual(by_id["doc-smuggle-08"]["turn_of_failure"], 3)
        # single-turn case fails at turn 1
        self.assertEqual(by_id["inj-web-01"]["turn_of_failure"], 1)


class ReportTests(unittest.TestCase):
    def test_table_and_files(self):
        scenarios = load_scenarios(DATA_PATH)
        before = evaluate.evaluate("stub-naive", scenarios, adapter=StubNaiveAdapter())
        after = evaluate.evaluate("stub-hardened", scenarios, adapter=StubHardenedAdapter())
        table = report.render_metrics_table(before, after)
        self.assertIn("attack-success rate", table)
        self.assertIn("indirect_injection_tool_output", table)
        self.assertIn("STUB RUN", table)  # honesty disclaimer for stub numbers
        with tempfile.TemporaryDirectory() as tmp:
            paths = report.write_reports(before, after, Path(tmp))
            metrics = json.loads(paths["metrics"].read_text())
            self.assertEqual(metrics["before"]["attack_success_rate"], 1.0)
            self.assertEqual(metrics["after"]["attack_success_rate"], 0.0)
            lines = paths["transcripts"].read_text().strip().splitlines()
            self.assertEqual(len(lines), 60)  # 30 cases x before/after
            self.assertIn("per-case verdicts", paths["report"].read_text())


class CliTests(unittest.TestCase):
    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.main(argv)
        return rc, buf.getvalue()

    def test_map_and_list(self):
        rc, out = self._run(["map"])
        self.assertEqual(rc, 0)
        self.assertIn("CuratorKIT", out)
        rc, out = self._run(["list-scenarios"])
        self.assertEqual(rc, 0)
        self.assertIn("inj-web-01", out)
        self.assertIn("ctrl-08", out)

    def test_dry_run_still_works(self):
        rc, out = self._run(["run", "--dry-run"])
        self.assertEqual(rc, 0)
        self.assertIn("n/a", out)

    def test_full_stub_run(self):
        rc, out = self._run(["run", "--adapter", "stub"])
        self.assertEqual(rc, 0)
        self.assertIn("100%", out)
        self.assertIn("over-refusal rate", out)


class TinyAyaStubTests(unittest.TestCase):
    def test_adapter_is_marked_todo(self):
        scenarios = load_scenarios(DATA_PATH)
        with self.assertRaises(NotImplementedError):
            CohereTinyAyaAdapter().respond(scenarios[0], 0, [])


if __name__ == "__main__":
    unittest.main()
