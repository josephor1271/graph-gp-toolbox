"""Bounded subprocess accounting and result-screening failure semantics.

Mocked children exercise coordinator errors here. The CLI smoke separately runs
real sparse GPyTorch likelihood, predictions, oracle and residual measurements.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from graphgp.benchmarking.cases import dense_eligible, primary_cases
from graphgp.benchmarking.coordinator import execute, request, scaling_requests, solver_requests
from graphgp.benchmarking.report import compare_accuracy, flatten, summarize, validate_records
from graphgp.benchmarking.worker import peak_rss_mib


class Stage04BenchmarkTests(unittest.TestCase):
    def _spec(self, **kwargs):
        return request(primary_cases()[0], "mll", "sparse_chebyshev", **kwargs)

    def _execute(self, runner):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "workers").mkdir()
            return execute(self._spec(), output, timeout=.01, runner=runner)

    def test_timeout_nonzero_and_signal_are_distinct_and_not_retried(self):
        calls = []

        def timeout(command, **kwargs):
            calls.append(command)
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])

        timed_out = self._execute(timeout)
        self.assertEqual(timed_out["status"], "timeout")
        self.assertEqual(len(calls), 1)
        self.assertIsNone(timed_out["returncode"])
        for code, expected in ((3, "nonzero_exit"), (-9, "signal")):
            with self.subTest(returncode=code):
                row = self._execute(lambda command, **kwargs: subprocess.CompletedProcess(
                    command, code, "", "out of memory"))
                self.assertEqual(row["status"], expected)
                self.assertEqual(row["returncode"], code)
                self.assertTrue(row["oom_indicator"])
                self.assertNotIn("elapsed_seconds", row)

    def test_missing_invalid_and_spawn_failure_outputs_are_bounded(self):
        missing = self._execute(lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "", ""))
        self.assertEqual(missing["status"], "missing_output")

        def invalid(command, **kwargs):
            Path(command[command.index("--output") + 1]).write_text("{invalid JSON")
            return subprocess.CompletedProcess(command, 0, "", "")

        self.assertEqual(self._execute(invalid)["status"], "invalid_output")

        def cannot_spawn(command, **kwargs):
            raise OSError("injected spawn failure")

        self.assertEqual(self._execute(cannot_spawn)["status"], "launch_error")

    def test_unsupported_memory_keeps_success_and_failure_denominators(self):
        with patch("graphgp.benchmarking.worker.platform.system", return_value="unsupported"):
            self.assertIsNone(peak_rss_mib())
        records = []
        for repeat, status in enumerate(("ok", "timeout", "nonzero_exit")):
            spec = self._spec(repeat=repeat)
            row = {**spec, "status": status, "rss_peak_mib": None,
                   "memory_status": "unsupported", "elapsed_seconds": 1.0 if status == "ok" else None}
            records.append(row)
        summary = summarize([flatten(record) for record in records])
        self.assertEqual(len(summary), 1)
        self.assertEqual((summary[0]["attempted"], summary[0]["success"],
                          summary[0]["timeout"], summary[0]["failed"]), (3, 1, 1, 1))
        self.assertNotIn("rss_peak_mib_median", summary[0])
        self.assertEqual(summary[0]["seconds_median"], 1.0)

    def test_composite_duplicates_missing_rows_and_nonfinite_success_are_rejected(self):
        spec = self._spec()
        row = {**spec, "status": "ok", "elapsed_seconds": 1., "rss_peak_mib": None,
               "memory_status": "unsupported", "result": {"nll": .9, "raw_gradients": {"heat_time": .1}}}
        self.assertEqual(validate_records([spec], [row])["actual_rows"], 1)
        duplicate_spec = {**spec, "run_id": "different-name-same-measurement"}
        duplicate_row = {**row, "run_id": duplicate_spec["run_id"]}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_records([spec, duplicate_spec], [row, duplicate_row])
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_records([spec], [])
        for field, invalid in (("nll", float("nan")), ("raw_gradients", {"heat_time": float("inf")}),
                               ("mean", [0., float("nan")])):
            altered = copy.deepcopy(row)
            altered["result"][field] = invalid
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "nonfinite"):
                validate_records([spec], [altered])

    def _accuracy_fixture(self):
        case = primary_cases()[0]
        reference = dict(nll=1., raw_gradients={"heat_time": .1, "lengthscale": -.2},
                         mean=[.3] * 32, latent_variance=[.2] * 32,
                         observation_variance=[.26] * 32,
                         exact_solutions=[[1., 2., 3.]] * case.m,
                         prior_mvp=[[.5] * 8] * case.nT)
        records = [{**request(case, "reference", "dense_exact", campaign="solver"),
                    "status": "ok", "measured_q": 32, "result": reference}]
        for operation in ("mll", "mean", "marginal", "solve"):
            result = {"mll": {"nll": 1., "raw_gradients": reference["raw_gradients"]},
                      "mean": {"mean": reference["mean"]},
                      "marginal": {"mean": reference["mean"], "latent_variance": reference["latent_variance"],
                                   "observation_variance": reference["observation_variance"]},
                      "solve": {"solutions": reference["exact_solutions"], "relative_residual": [1e-5, 2e-5, 3e-5]}}[operation]
            records.append({**request(case, operation, "sparse_chebyshev", campaign="solver"),
                            "status": "ok", "measured_q": 32, "result": copy.deepcopy(result),
                            "settings": {"cg_tolerance": 1e-4}, "numerical_warning_count": 1})
        # A separate 64-target reference must never replace the solver prefix.
        full_reference = copy.deepcopy(reference)
        for key in ("mean", "latent_variance", "observation_variance"):
            full_reference[key] *= 2
        records.append({**request(case, "reference", "dense_exact"), "status": "ok",
                        "measured_q": 64, "result": full_reference})
        return records

    def test_accuracy_failure_is_not_success_and_preserves_independent_solves(self):
        records = self._accuracy_fixture()
        checks, accuracy = compare_accuracy(records)
        screening = [row for row in accuracy if row["screening_scope"] == "profile_seed"]
        self.assertEqual(screening[0]["screen_status"], "pass")
        self.assertEqual(len(checks), 3)
        self.assertEqual(screening[0]["q"], 32)
        mll = next(row for row in records if row["operation"] == "mll")
        mll["result"]["nll"] += .06
        checks, accuracy = compare_accuracy(records)
        screening = [row for row in accuracy if row["screening_scope"] == "profile_seed"]
        self.assertEqual(screening[0]["screen_status"], "fail")
        self.assertIn("nll", screening[0]["screen_failures"])
        self.assertTrue(all(check["status"] == "ok" for check in checks))
        mll["status"] = "timeout"
        checks, accuracy = compare_accuracy(records)
        screening = [row for row in accuracy if row["screening_scope"] == "profile_seed"]
        self.assertIn("mll_not_successful", screening[0]["screen_failures"])
        self.assertEqual([check["relative_residual"] for check in checks], [1e-5, 2e-5, 3e-5])

    def test_fixed_request_matrix_keeps_profiles_and_dense_ceiling(self):
        cases = primary_cases()
        scaling = scaling_requests(cases)
        self.assertEqual(len({case.case_id for case in cases}), 12)
        self.assertEqual(len({row["run_id"] for row in scaling}), len(scaling))
        for case in cases:
            rows = [row for row in scaling if row["case"]["case_id"] == case.case_id]
            sparse = [row for row in rows if row["backend"] == "sparse_chebyshev"]
            self.assertTrue(sparse)
            self.assertTrue(all(row["profile"] == "balanced_probe" for row in sparse))
            if not dense_eligible(case):
                self.assertFalse(any(row["backend"] == "dense_exact" for row in rows))
            self.assertTrue({"mean", "marginal", "mvp"} <= {row["operation"] for row in sparse})
        sweep = solver_requests(cases)
        for n, profile, expected in ((16, "fast_probe", {301, 302, 303}),
                                     (49, "strict_probe", {301, 302, 303}),
                                     (100, "balanced_probe", {301, 302}),
                                     (16, "exhaustive_tiny", {301})):
            actual = {row["probe_seed"] for row in sweep if row["case"]["n"] == n
                      and row["profile"] == profile and row["operation"] == "mll"}
            self.assertEqual(actual, expected)
        self.assertFalse(any(row["profile"] == "exhaustive_tiny" and row["case"]["n"] != 16 for row in sweep))

    def test_parent_coordinator_import_does_not_load_numerical_stack(self):
        command = """import json, sys
import graphgp.benchmarking.coordinator
print(json.dumps(sorted(name for name in ('torch','gpytorch','numpy','scipy') if name in sys.modules)))
"""
        output = subprocess.check_output([sys.executable, "-c", command], text=True)
        self.assertEqual(json.loads(output), [])


if __name__ == "__main__":
    unittest.main()
