"""Protect scientific denominators and pairing in the dynamic holdout report."""
import json
from pathlib import Path
import tempfile
import unittest

from report_dynamic_evaluation import expected_runs, report


class EvaluationReportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.names = expected_runs(2)
        self.write("pipeline_config.json", {"evaluation_repeats": 2})
        self.write("pipeline_status.json", {"stage": "heldout_evaluation", "jobs": {}})
        self.write("heldout_evaluation_jobs.json", [
            {"name": name, "complete": str(self.root / name / "metrics.json")} for name in self.names])

    def write(self, filename, value):
        path = self.root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def outcome(self, name, success, failure=None, *, success_ever=None):
        self.write(f"{name}/metrics.json", [{
            "final_success": success, "success": success if success_ever is None else success_ever,
            "steps": 450 if failure is None else 13, "failure": failure,
            "elapsed_s": 20., "first_success_step": 140 if success else None}])

    def test_pending_and_invalid_do_not_become_task_failures(self):
        name = "test_body14054_pass_rep0_none"
        self.outcome(name, False, {"kind": "invalid_physics"})
        # A temporary early success is not final sustained completion.
        self.outcome("test_body14054_pass_rep1_none", False, success_ever=True)
        value = report(self.root)
        base = value["policies"]["none"]
        self.assertEqual(base["expected"], 8)
        self.assertEqual(base["valid"], 1)
        self.assertEqual(base["successes"], 0)
        self.assertEqual(base["statuses"], {"invalid_physics": 1, "valid": 1, "pending": 6})
        self.assertEqual(base["task_failures"], {"timeout": 1})
        self.assertIsNone(value["policies"]["history_seed0"]["success_rate"])
        self.assertFalse(value["evaluation_complete"])

    def test_seed_pairs_remain_nested_in_four_task_cells(self):
        for name, design in self.names.items():
            self.outcome(name, design["model"].startswith("history_"))
        self.write("pipeline_status.json", {"stage": "complete", "jobs": {}})
        value = report(self.root)
        self.assertTrue(value["evaluation_complete"])
        self.assertEqual(value["statuses"], {"valid": 72})
        self.assertEqual(len(value["policies"]), 9)
        self.assertEqual(value["expected_task_cells"], 4)
        for result in value["history_current_by_cell"].values():
            self.assertEqual(result["expected_pairs"], 6)
            self.assertEqual(result["valid_pairs"], 6)
            self.assertEqual(result["mean_success_difference"], 1.)

    def test_invalid_pair_is_excluded_without_changing_expected_coverage(self):
        self.outcome("test_body14054_pass_rep0_history_seed0", True)
        self.outcome("test_body14054_pass_rep0_current_seed0", False, {"kind": "invalid_physics"})
        value = report(self.root)["history_current_by_cell"]["body14054_pass"]
        self.assertEqual(value["expected_pairs"], 6)
        self.assertEqual(value["valid_pairs"], 0)
        self.assertIsNone(value["mean_success_difference"])

    def test_changed_grid_and_incomplete_success_are_flagged(self):
        self.write("heldout_evaluation_jobs.json", [{
            "name": "test_body14054_pass_rep0_none",
            "complete": str(self.root / "test_body14054_pass_rep0_none/metrics.json")}])
        self.outcome("test_body14054_pass_rep0_none", True, {"kind": "grasp_lost"})
        value = report(self.root)
        self.assertEqual(len(value["protocol_errors"]), 2)
        self.assertEqual(value["policies"]["none"]["valid"], 0)
        self.assertFalse(value["evaluation_complete"])


if __name__ == "__main__":
    unittest.main()
