"""Independent selected-noisy residuals and scoped installed-solver observations."""
import json
from types import SimpleNamespace
import unittest
import warnings

import gpytorch
from linear_operator import utils as operator_utils
from linear_operator.operators import DenseLinearOperator
from linear_operator.utils.stochastic_lq import StochasticLQ
import numpy as np
import torch

from graphgp.benchmarking.reference import dense_reference
from graphgp.benchmarking.solver import (
    installed_solver_audit, relative_residual, solver_diagnostics,
)
from graphgp.inference.predict import predict_marginals
from graphgp.inference.solver_settings import IterativeConfig, iterative_context
from graphgp.models.graph_time import make_graph_time_gp
from graphgp.validation_graph_time import graph_time_inference, tiny_validation_problem


class Stage04SolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_residual_uses_selected_noisy_A_and_detects_perturbation(self):
        # Selected latent K has equal rows for duplicate coordinate recordings;
        # independent noise makes the selected covariance A positive definite.
        latent = torch.tensor([[1., .3, 1.], [.3, .9, .3], [1., .3, 1.]],
                              dtype=torch.float64)
        noisy = latent + .06 * torch.eye(3, dtype=torch.float64)
        rhs = torch.tensor([[1., .4, .8], [-.2, .7, -.4], [.5, -.1, .2]],
                           dtype=torch.float64)
        solution = torch.linalg.solve(noisy, rhs)
        self.assertLess(float(relative_residual(noisy, solution, rhs).max()), 1e-13)
        self.assertGreater(float(relative_residual(latent, solution, rhs).max()), .1)
        altered = solution.clone()
        altered[0, 1] += .02
        self.assertGreater(float(relative_residual(noisy, altered, rhs)[1]), .01)
        vector_residual = relative_residual(DenseLinearOperator(noisy), solution[:, 0], rhs[:, 0])
        self.assertEqual(vector_residual.ndim, 0)
        self.assertLess(float(vector_residual), 1e-13)

    def test_diagnostics_record_actual_orders_and_restore_after_exception(self):
        original_cg, original_slq = operator_utils.linear_cg, StochasticLQ.to_dense
        config = IterativeConfig(
            cg_tolerance=1e-4, eval_cg_tolerance=1e-4, max_cg_iterations=50,
            trace_samples=3, lanczos_iterations=8, max_preconditioner_size=0,
            terminate_cg_by_size=False)
        matrix = torch.diag(torch.linspace(.2, 2., 12, dtype=torch.float64))
        matrix.requires_grad_(True)
        rhs = torch.arange(12, dtype=torch.float64) / 10
        with warnings.catch_warnings(record=True), torch.random.fork_rng(), iterative_context(config):
            torch.manual_seed(301)
            with solver_diagnostics() as diagnostic:
                iq, ld = DenseLinearOperator(matrix).inv_quad_logdet(rhs, logdet=True)
                (iq + ld).backward()
                self.assertIsNotNone(matrix.grad)
            self.assertGreaterEqual(diagnostic["cg_calls"], 1)
            self.assertEqual(diagnostic["cg_rhs_shapes"][0], [12, 4])
            self.assertEqual(diagnostic["cg_tridiag_counts"][0], 3)
            self.assertEqual(diagnostic["slq_calls"], 1)
            self.assertEqual(diagnostic["slq_probe_counts"], [3])
            self.assertEqual(diagnostic["cg_actual_tridiagonal_orders"],
                             diagnostic["slq_tridiagonal_orders"])
            self.assertTrue(1 <= diagnostic["slq_tridiagonal_orders"][0] <= 8)
            self.assertIsNone(diagnostic["actual_cg_iterations"])
            json.dumps(diagnostic, allow_nan=False)
        self.assertIs(operator_utils.linear_cg, original_cg)
        self.assertIs(StochasticLQ.to_dense, original_slq)
        with warnings.catch_warnings(record=True), iterative_context(config):
            with solver_diagnostics() as outer:
                outer_wrapper = operator_utils.linear_cg
                with solver_diagnostics() as inner:
                    operator_utils.linear_cg(matrix.detach().matmul, rhs, max_iter=50)
                self.assertIs(operator_utils.linear_cg, outer_wrapper)
                self.assertEqual(inner["cg_calls"], 1)
                self.assertEqual(outer["cg_calls"], 1)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with solver_diagnostics():
                raise RuntimeError("injected")
        self.assertIs(operator_utils.linear_cg, original_cg)
        self.assertIs(StochasticLQ.to_dense, original_slq)

    def test_compact_source_audit_identifies_actual_installed_api(self):
        audit = installed_solver_audit()
        self.assertEqual(audit["installed_cg_eps"], 1e-10)
        self.assertFalse(audit["iteration_count_instrumented"])
        self.assertIn("n_tridiag", audit["sources"]["linear_cg"]["signature"])
        self.assertEqual(len(audit["sources"]["linear_cg"]["source_file_sha256"]), 64)
        json.dumps(audit, allow_nan=False)

    def test_independent_reference_matches_dense_gp_raw_gradients_and_predictions(self):
        problem = tiny_validation_problem()
        rng = np.random.default_rng(19)
        problem["solve_rhs"] = np.column_stack((problem["train_y"], rng.normal(size=(18, 2))))
        problem["mvp_rhs"] = rng.normal(size=(45, 8))
        case = SimpleNamespace(n=9, num_times=5, m=18, q=8)
        reference = dense_reference(problem, case)
        json.dumps(reference, allow_nan=False)
        train = torch.tensor(problem["train_pairs"], dtype=torch.float64)
        values = torch.tensor(problem["train_y"], dtype=torch.float64)
        model = make_graph_time_gp(problem["L"], train, values, num_times=5,
                                   backend="dense_exact", heat_time=1.2, lengthscale=.24)
        model.train()
        with warnings.catch_warnings(record=True), graph_time_inference("dense_exact", 18):
            loss = -gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)(
                model(train), values)
            loss.backward()
            moments = predict_marginals(
                model, torch.tensor(problem["target_pairs"], dtype=torch.float64), chunk_size=3)
        self.assertAlmostEqual(float(loss.detach()), reference["nll"], places=12)
        self.assertAlmostEqual(model.covar_module.graph_kernel.raw_heat_time.grad.item(),
                               reference["raw_gradients"]["heat_time"], places=11)
        self.assertAlmostEqual(model.covar_module.time_kernel.raw_lengthscale.grad.item(),
                               reference["raw_gradients"]["lengthscale"], places=11)
        for name in ("mean", "latent_variance", "observation_variance"):
            np.testing.assert_allclose(getattr(moments, name).numpy(), reference[name],
                                       rtol=1e-11, atol=1e-12)
        prefix_problem = {**problem, "target_pairs": problem["target_pairs"][:3]}
        with self.assertRaisesRegex(ValueError, "dimensions"):
            dense_reference(prefix_problem, case)
        prefix_reference = dense_reference(prefix_problem, case, target_count=3)
        # Different RHS widths may select different BLAS kernels and change
        # final rounding (Linux CI differed by 1.4e-17). Require float64
        # agreement rather than bitwise identity of independently solved blocks.
        np.testing.assert_allclose(prefix_reference["mean"], reference["mean"][:3],
                                   rtol=1e-14, atol=1e-14)
        # The ceiling is checked before trying to access any problem data.
        with self.assertRaisesRegex(ValueError, "ceiling"):
            dense_reference({}, SimpleNamespace(n=225, num_times=8, m=450, q=64))


if __name__ == "__main__":
    unittest.main()
