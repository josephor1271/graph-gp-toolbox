"""The predeclared measurement matrix and operation budget remain fixed."""

import os
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from graphgp.benchmarking.cases import dense_eligible, primary_cases, smoke_case
from graphgp.benchmarking.coordinator import (
    TIMEOUT_SECONDS, scaling_requests, solver_requests,
)


class Stage04WorkloadTests(unittest.TestCase):
    def test_primary_matrix_covers_all_cases_with_balanced_sparse_only(self):
        cases = primary_cases()
        requests = scaling_requests(cases)
        self.assertEqual({row["case"]["case_id"] for row in requests}, {c.case_id for c in cases})
        self.assertEqual(len({row["run_id"] for row in requests}), len(requests))
        self.assertEqual({row["profile"] for row in requests
                          if row["backend"] == "sparse_chebyshev"}, {"balanced_probe"})
        for case in cases:
            rows = [row for row in requests if row["case"]["case_id"] == case.case_id]
            backends = {row["backend"] for row in rows}
            self.assertEqual(backends, {"sparse_chebyshev", "dense_exact"}
                             if dense_eligible(case) else {"sparse_chebyshev"})
            for backend in backends:
                measured = {row["operation"] for row in rows if row["backend"] == backend}
                expected = {"mean", "marginal", "mvp"}
                if set(case.groups) & {"N", "T", "M"}:
                    expected.add("mll")
                    if dense_eligible(case):
                        expected.add("train3")
                if set(case.groups) & {"N", "T"}:
                    expected.add("construct")
                if backend == "dense_exact":
                    expected.add("reference")
                self.assertEqual(measured, expected, (case.case_id, backend))
                for operation in measured:
                    op_rows = [row for row in rows
                               if row["backend"] == backend and row["operation"] == operation]
                    small = case.n <= 100 and case.num_times < 32
                    repeats = 1 if operation == "reference" or not small else (
                        3 if operation in ("mvp", "construct") else 2)
                    widths = (1, 8) if operation == "mvp" else (0,)
                    self.assertEqual({(row["repeat"], row["rhs_width"]) for row in op_rows},
                                     {(repeat, width) for repeat in range(repeats) for width in widths},
                                     (case.case_id, backend, operation))
                    self.assertEqual(len(op_rows), repeats * len(widths))
            references = [row for row in rows if row["operation"] == "reference"]
            self.assertEqual(len(references), int(dense_eligible(case)))

    def test_solver_matrix_exact_profiles_seeds_and_required_operations(self):
        requests = solver_requests(primary_cases())
        self.assertEqual(len(requests), 95)
        self.assertEqual(len({row["run_id"] for row in requests}), 95)
        reference = [row for row in requests if row["operation"] == "reference"]
        self.assertEqual({row["case"]["n"] for row in reference}, {16, 49, 100})
        self.assertEqual(len(reference), 3)
        for n in (16, 49, 100):
            rows = [row for row in requests if row["case"]["n"] == n
                    and row["operation"] != "reference"]
            profiles = ("fast_probe", "balanced_probe", "strict_probe") if n < 100 else (
                "fast_probe", "balanced_probe")
            expected = {(profile, seed, operation)
                        for profile in profiles
                        for seed in ((301, 302, 303) if n < 100 else (301, 302))
                        for operation in ("mll", "mean", "marginal", "solve")}
            if n == 16:
                expected |= {("exhaustive_tiny", 301, operation)
                             for operation in ("mll", "mean", "marginal", "solve")}
            self.assertEqual({(row["profile"], row["probe_seed"], row["operation"])
                              for row in rows}, expected)
            for row in rows:
                self.assertEqual(row["campaign"], "solver")
                self.assertEqual(row["backend"], "sparse_chebyshev")
                self.assertEqual(row["case"]["num_times"], 8)
        self.assertEqual(TIMEOUT_SECONDS, 180)

    def test_smoke_contains_real_sparse_routes_and_independent_reference(self):
        case = smoke_case()
        solver = solver_requests([case], smoke=True)
        sparse = [row for row in solver if row["backend"] == "sparse_chebyshev"]
        self.assertEqual({row["operation"] for row in sparse}, {"mll", "mean", "marginal", "solve"})
        self.assertEqual(len(solver), 5)
        self.assertTrue(all(row["profile"] == "exhaustive_tiny" for row in sparse))
        self.assertEqual(sum(row["operation"] == "reference" for row in solver), 1)
        scaling = scaling_requests([case], smoke=True)
        self.assertTrue(all(row["repeat"] == 0 for row in scaling))
        self.assertIn("train3", {row["operation"] for row in scaling})

    def test_coordinator_import_is_free_of_numerical_stack(self):
        code = ("import graphgp.benchmarking.coordinator,sys; "
                "assert not any(name in sys.modules for name in "
                "('numpy','scipy','torch','gpytorch'))")
        subprocess.run([sys.executable, "-c", code], check=True,
                       cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)

    def test_dense_reference_enforces_ceiling_before_reading_arrays(self):
        from graphgp.benchmarking.reference import dense_reference
        for case in primary_cases():
            if not dense_eligible(case):
                with self.subTest(case=case.case_id), self.assertRaisesRegex(ValueError, "ceiling"):
                    dense_reference({}, case)

    def test_train3_performs_exactly_three_adam_updates_and_four_loss_checks(self):
        import torch
        from graphgp.benchmarking.coordinator import request
        from graphgp.benchmarking.worker import run

        case = smoke_case()
        spec = request(case, "train3", "dense_exact")
        original = torch.optim.Adam.step
        calls = []

        def counted(optimizer, *args, **kwargs):
            calls.append(optimizer.param_groups[0]["lr"])
            return original(optimizer, *args, **kwargs)

        old_threads = torch.get_num_threads()
        try:
            with patch.dict(os.environ), patch.object(torch.optim.Adam, "step", counted):
                row = run(spec)
        finally:
            torch.set_num_threads(old_threads)
        self.assertEqual(calls, [0.05, 0.05, 0.05])
        self.assertEqual(row["result"]["updates"], 3)
        self.assertEqual([point["step"] for point in row["result"]["trajectory"]], [0, 1, 2, 3])
        self.assertAlmostEqual(row["result"]["trajectory"][0]["heat_time"], 1.2, places=14)
        self.assertAlmostEqual(row["result"]["trajectory"][0]["lengthscale"], 0.24, places=14)
        self.assertEqual(row["solver"]["cg_calls"], 0)
        self.assertGreater(row["result"]["final_gradient_norm"], 0)

    def test_sparse_worker_guards_reject_explicit_dense_reference_routes(self):
        import numpy as np
        import scipy.linalg
        import torch
        from graphgp.benchmarking.worker import guarded_sparse

        numpy_matrix = np.eye(2)
        torch_matrix = torch.eye(2, dtype=torch.float64)
        forbidden = {
            "numpy_kron": lambda: np.kron(numpy_matrix, numpy_matrix),
            "torch_kron": lambda: torch.kron(torch_matrix, torch_matrix),
            "numpy_graph_eigh": lambda: np.linalg.eigh(numpy_matrix),
            "scipy_graph_eigh": lambda: scipy.linalg.eigh(numpy_matrix),
            "torch_graph_eigh": lambda: torch.linalg.eigh(torch_matrix),
            "scipy_graph_expm": lambda: scipy.linalg.expm(numpy_matrix),
            "torch_direct_cholesky": lambda: torch.linalg.cholesky(torch_matrix),
        }
        with guarded_sparse():
            for name, operation in forbidden.items():
                with self.subTest(method=name), self.assertRaises(AssertionError):
                    operation()
        # The scoped wrappers must restore the original numerical functions.
        self.assertEqual(scipy.linalg.expm(numpy_matrix).shape, (2, 2))
        self.assertEqual(torch.linalg.cholesky(torch_matrix).shape, (2, 2))

    def test_all_dense_eligible_cases_report_fixed_point_errors_per_measurement(self):
        from graphgp.benchmarking.coordinator import request
        from graphgp.benchmarking.report import compare_accuracy

        records = []
        requested_ids = set()
        for case in primary_cases():
            if not dense_eligible(case):
                continue
            reference = dict(nll=0.6, raw_gradients={"heat_time": 0.1, "lengthscale": -0.2},
                             mean=[0.0] * case.q, latent_variance=[0.2] * case.q,
                             observation_variance=[0.26] * case.q,
                             prior_mvp=[[float((i+j) % 17) / 17 for j in range(8)]
                                        for i in range(case.nT)])
            records.append(request(case, "reference", "dense_exact")
                           | {"status": "ok", "result": reference})
            for backend in ("dense_exact", "sparse_chebyshev"):
                for operation in ("mll", "mean", "marginal", "mvp"):
                    if operation == "mll" and not set(case.groups) & {"N", "T", "M"}:
                        continue
                    for repeat in (0, 1):
                        row = request(case, operation, backend, repeat=repeat,
                                      width=8 if operation == "mvp" else 0)
                        result = deepcopy(reference)
                        if backend == "sparse_chebyshev":
                            result["nll"] += 1e-4
                            result["raw_gradients"]["heat_time"] += 1e-4
                            result["mean"] = [1e-5] * case.q
                        result["product"] = result.pop("prior_mvp")
                        records.append(row | {"status": "ok", "result": result})
                        requested_ids.add(row["run_id"])
        checks, accuracy = compare_accuracy(records)
        self.assertEqual(checks, [])
        self.assertEqual({row["run_id"] for row in accuracy}, requested_ids)
        self.assertEqual({row["case_id"] for row in accuracy},
                         {case.case_id for case in primary_cases() if dense_eligible(case)})
        self.assertTrue(all(row["screening_scope"] == "case_check" for row in accuracy))
        self.assertTrue(all(row["screen_status"] == "pass" for row in accuracy))
        for row in accuracy:
            error_key = {"mll": "nll_abs_error", "mean": "posterior_mean_max_abs_error",
                         "marginal": "latent_variance_max_abs_error",
                         "mvp": "kernel_mvp_max_abs_error"}[row["operation"]]
            self.assertIn(error_key, row)
        polynomial_rows = [row for row in records if row.get("polynomial_mvp_max_abs_error") is not None]
        self.assertEqual(len(polynomial_rows), 2 * sum(dense_eligible(case) for case in primary_cases()))

    def test_scaling_accuracy_keeps_failures_and_does_not_use_32_target_solver_reference(self):
        from graphgp.benchmarking.coordinator import request
        from graphgp.benchmarking.report import compare_accuracy

        case = primary_cases()[0]
        rows = [request(case, "reference", "dense_exact", campaign="solver")
                | {"status": "ok", "measured_q": 32, "result": {"mean": [1.0]*32}},
                request(case, "reference", "dense_exact")
                | {"status": "ok", "result": {"mean": [0.0]*64}},
                request(case, "mean", "sparse_chebyshev")
                | {"status": "ok", "result": {"mean": [1e-5]*64}},
                request(case, "marginal", "sparse_chebyshev")
                | {"status": "timeout"},
                request(case, "mll", "sparse_chebyshev")
                | {"status": "ok", "result": {"nll": float("nan"), "raw_gradients": {}}}]
        _, accuracy = compare_accuracy(rows)
        by_operation = {row["operation"]: row for row in accuracy}
        self.assertEqual(by_operation["mean"]["posterior_mean_max_abs_error"], 1e-5)
        self.assertEqual(by_operation["mean"]["screen_status"], "pass")
        self.assertEqual(by_operation["marginal"]["operation_status"], "timeout")
        self.assertEqual(by_operation["marginal"]["screen_status"], "fail")
        self.assertIn("operation_not_successful", by_operation["marginal"]["screen_failures"])
        self.assertIn("invalid_accuracy_input", by_operation["mll"]["screen_failures"])


if __name__ == "__main__":
    unittest.main()
