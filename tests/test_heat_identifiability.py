"""Small deterministic mathematical gates for the dense statistical reference."""

import json
import unittest
from unittest.mock import patch

import gpytorch
import numpy as np
import scipy.linalg
import torch

from graphgp.analysis.heat_identifiability import (
    HEAT_RANGE, NOISE_RANGE, HeatOracle, fit_multistart, sample_pool,
)
from graphgp.models.factory import (
    HEAT_RANGE as MODEL_HEAT_RANGE, NOISE_RANGE as MODEL_NOISE_RANGE, make_heat_gp,
)
from graphgp.recovery import exact_nll, sample_problem
from graphgp.utils.graphs import make_laplacian
from graphgp.validation import inference_context

torch.set_num_threads(1)


class HeatIdentifiabilityTests(unittest.TestCase):
    def setUp(self):
        self.laplacian, _ = make_laplacian("path", 8)
        self.oracle = HeatOracle(self.laplacian)
        self.ids = np.array([7, 1, 5, 2, 0])
        self.Y = sample_pool(self.oracle, 0.6, 0.03, 5, 123)["observations"][self.ids]

    def test_likelihood_matches_existing_oracle_and_gpytorch_at_multiple_pairs(self):
        for family in ("path", "grid", "irregular"):
            problem = sample_problem(family, n=16, seed=23)
            oracle = HeatOracle(problem["L"])
            ids = problem["train"].numpy()
            data = problem["y"][problem["train"]].numpy()
            for tau, noise in ((0.2, 0.02), (1.3, 0.11), (3.4, 0.37)):
                with self.subTest(family=family, tau=tau, noise=noise):
                    actual = oracle.nll(tau, noise, data, ids)
                    expected = exact_nll(problem, tau, noise)
                    model = make_heat_gp(problem["L"], problem["train"],
                                         problem["y"][problem["train"]], "A",
                                         heat_time=tau, noise=noise)
                    # GPyTorch converts a Python scalar through its default
                    # dtype; assign float64 explicitly to compare identical
                    # physical noise, rather than a float32-rounded value.
                    model.likelihood.noise = torch.tensor(noise, dtype=torch.float64)
                    self.assertAlmostEqual(model.likelihood.noise.item(), noise, places=15)
                    model.train()
                    model.likelihood.train()
                    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
                    with inference_context("A", len(ids)):
                        gpytorch_nll = float(-mll(model(problem["train"]),
                                                  problem["y"][problem["train"]]).detach())
                    np.testing.assert_allclose([actual, actual], [expected, gpytorch_nll],
                                               rtol=1e-10, atol=1e-11)

    def test_log_gradient_matches_centered_finite_differences(self):
        eta = np.log([1.1, 0.08])
        value, gradient = self.oracle.nll_and_gradient(eta, self.Y, self.ids)
        self.assertAlmostEqual(value, self.oracle.nll(1.1, 0.08, self.Y, self.ids), places=14)
        step = 1e-5
        finite = []
        for direction in np.eye(2):
            right = self.oracle.nll_and_gradient(eta+step*direction, self.Y, self.ids)[0]
            left = self.oracle.nll_and_gradient(eta-step*direction, self.Y, self.ids)[0]
            finite.append((right-left)/(2*step))
        np.testing.assert_allclose(gradient, finite, rtol=1e-5, atol=1e-9)

    def test_physical_gradient_matches_torch_matrix_exponential_autograd(self):
        tau = torch.tensor(1.1, dtype=torch.float64, requires_grad=True)
        noise = torch.tensor(0.08, dtype=torch.float64, requires_grad=True)
        full_heat = torch.matrix_exp(-tau*torch.from_numpy(self.laplacian.toarray()))
        covariance = full_heat[self.ids][:, self.ids]+noise*torch.eye(len(self.ids), dtype=torch.float64)
        factor = torch.linalg.cholesky(covariance)
        data = torch.from_numpy(self.Y)
        m, r = self.Y.shape
        value = (2*r*factor.diagonal().log().sum()
                 + (data*torch.cholesky_solve(data, factor)).sum()
                 + m*r*np.log(2*np.pi))/(2*m*r)
        gradient = torch.autograd.grad(value, (tau, noise))
        actual, actual_gradient = self.oracle.physical_nll_and_gradient(1.1, 0.08, self.Y, self.ids)
        self.assertAlmostEqual(actual, float(value.detach()), places=13)
        np.testing.assert_allclose(actual_gradient, [x.item() for x in gradient], rtol=1e-10, atol=1e-12)

    def test_replicate_normalization_and_mean_of_individual_objectives(self):
        first = self.Y[:, 0]
        repeated = np.repeat(first[:, None], 7, axis=1)
        single = self.oracle.physical_nll_and_gradient(1.1, 0.08, first, self.ids)
        copies = self.oracle.physical_nll_and_gradient(1.1, 0.08, repeated, self.ids)
        self.assertAlmostEqual(single[0], copies[0], places=14)
        np.testing.assert_allclose(single[1], copies[1], atol=1e-14)
        individual = [self.oracle.nll_and_gradient(np.log([1.1, 0.08]), column, self.ids)
                      for column in self.Y.T]
        pooled = self.oracle.nll_and_gradient(np.log([1.1, 0.08]), self.Y, self.ids)
        self.assertAlmostEqual(pooled[0], np.mean([pair[0] for pair in individual]), places=14)
        np.testing.assert_allclose(pooled[1], np.mean([pair[1] for pair in individual], axis=0), atol=1e-14)

    def test_kl_zero_at_truth_and_nonnegative_without_clamping(self):
        value = self.oracle.expected_excess(0.6, 0.03, 0.6, 0.03, self.ids)
        self.assertAlmostEqual(value, 0, places=14)
        for tau in (0.05, 0.3, 0.6, 2.0, 4.0):
            for noise in (0.005, 0.03, 0.12, 0.5):
                self.assertGreaterEqual(
                    self.oracle.expected_excess(tau, noise, 0.6, 0.03, self.ids), -1e-14)

    def test_fisher_symmetry_psd_chain_rule_and_replicate_scaling(self):
        log_information = self.oracle.fisher(0.6, 0.03, self.ids)
        physical = self.oracle.fisher(0.6, 0.03, self.ids, log_coordinates=False)
        np.testing.assert_allclose(log_information, log_information.T, atol=1e-15)
        self.assertGreaterEqual(np.linalg.eigvalsh(log_information)[0], -1e-14)
        transform = np.diag([0.6, 0.03])
        np.testing.assert_allclose(log_information, transform@physical@transform, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(self.oracle.fisher(0.6, 0.03, self.ids, replicates=5),
                                   5*log_information, rtol=1e-14, atol=1e-14)
        diagnostics = self.oracle.fisher_diagnostics(0.6, 0.03, self.ids)
        self.assertEqual(diagnostics["information_status"], "nonsingular")
        self.assertGreater(diagnostics["condition_number"], 1)
        self.assertLessEqual(abs(diagnostics["log_parameter_correlation"]), 1)

    def test_fisher_matches_expected_log_nll_curvature(self):
        truth = np.array([0.6, 0.03])
        eta = np.log(truth)
        step = 2e-4

        def objective(point):
            return self.oracle.expected_excess(*np.exp(point), *truth, self.ids)

        base = objective(eta)
        directions = np.eye(2)*step
        hessian = np.empty((2, 2))
        for i in range(2):
            hessian[i, i] = (objective(eta+directions[i])-2*base
                              + objective(eta-directions[i]))/step**2
        hessian[0, 1] = hessian[1, 0] = (
            objective(eta+directions[0]+directions[1])
            - objective(eta+directions[0]-directions[1])
            - objective(eta-directions[0]+directions[1])
            + objective(eta-directions[0]-directions[1]))/(4*step**2)
        np.testing.assert_allclose(hessian, self.oracle.fisher(*truth, self.ids)/len(self.ids),
                                   rtol=2e-5, atol=1e-8)

    def test_singular_fisher_has_no_estimate_correlation(self):
        oracle = HeatOracle(np.zeros((3, 3)))
        information = oracle.fisher_diagnostics(0.6, 0.03)
        self.assertEqual(information["information_status"], "numerically_singular")
        self.assertIsNone(information["condition_number"])
        self.assertIsNone(information["log_parameter_correlation"])

    def test_mask_is_selected_after_full_graph_exponential_and_eigen_cache(self):
        with patch("graphgp.analysis.heat_identifiability.eigh", wraps=scipy.linalg.eigh) as eigen:
            oracle = HeatOracle(self.laplacian)
            tau = 1.7
            actual = oracle.covariance(tau, 0, self.ids)
            expected = scipy.linalg.expm(-tau*self.laplacian.toarray())[np.ix_(self.ids, self.ids)]
            wrong = scipy.linalg.expm(-tau*self.laplacian.toarray()[np.ix_(self.ids, self.ids)])
            np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
            self.assertGreater(np.linalg.norm(actual-wrong), 0.05)
            oracle.nll(tau, 0.03, self.Y, self.ids)
            oracle.fisher(tau, 0.03, self.ids)
            self.assertEqual(eigen.call_count, 1)

    def test_sampling_independent_streams_exact_transform_and_nested_prefixes(self):
        key = (8, 0, 1)
        full = sample_pool(self.oracle, 0.6, 0.03, 20, 123, mask_seed=456, stream_key=key)
        short = sample_pool(self.oracle, 0.6, 0.03, 5, 123, mask_seed=456, stream_key=key)
        for name in ("latent", "noise", "observations"):
            self.assertEqual(full[name].shape, (8, 20))
            np.testing.assert_allclose(short[name], full[name][:, :5], rtol=0, atol=1e-15)
            self.assertFalse(np.array_equal(full[name][:, 0], full[name][:, 1]))
        np.testing.assert_array_equal(full["permutation"], short["permutation"])
        factor = scipy.linalg.cholesky(scipy.linalg.expm(-0.6*self.laplacian.toarray()), lower=True)
        latent_white = np.random.default_rng(np.random.SeedSequence([123, *key], spawn_key=(0,)))
        noise_white = np.random.default_rng(np.random.SeedSequence([123, *key], spawn_key=(1,)))
        np.testing.assert_allclose(full["latent"], factor@latent_white.standard_normal((20, 8)).T,
                                   rtol=1e-13, atol=1e-14)
        np.testing.assert_array_equal(full["noise"], np.sqrt(0.03)*noise_white.standard_normal((20, 8)).T)
        other_mask = sample_pool(self.oracle, 0.6, 0.03, 20, 123, mask_seed=789, stream_key=key)
        for name in ("latent", "noise", "observations"):
            np.testing.assert_array_equal(full[name], other_mask[name])
        self.assertFalse(np.array_equal(full["permutation"], other_mask["permutation"]))
        other_pool = sample_pool(self.oracle, 0.6, 0.03, 20, 124, mask_seed=456, stream_key=key)
        self.assertFalse(np.array_equal(full["latent"], other_pool["latent"]))
        self.assertFalse(np.array_equal(full["noise"], other_pool["noise"]))
        np.testing.assert_array_equal(full["permutation"], other_pool["permutation"])

    def test_known_noise_fit_changes_only_heat_time(self):
        known_noise = 0.03
        rows, selected = fit_multistart(self.oracle, self.Y, self.ids, known_noise=known_noise)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertTrue(row["valid"])
            self.assertEqual(row["fit_noise_variance"], known_noise)
            self.assertEqual(len(json.loads(row["log_gradient"])), 1)
        self.assertEqual(selected["fit_noise_variance"], known_noise)
        self.assertNotEqual(selected["fit_tau"], rows[0]["initial_tau"])
        expected = self.oracle.nll(selected["fit_tau"], known_noise, self.Y, self.ids)
        self.assertAlmostEqual(selected["nll_per_observation"], expected, places=14)
        eta = np.log([1.1])
        _, gradient = self.oracle.nll_and_gradient(eta, self.Y, self.ids, known_noise)
        physical = self.oracle.physical_nll_and_gradient(1.1, known_noise, self.Y, self.ids)[1]
        np.testing.assert_allclose(gradient, physical[:1]*1.1, rtol=1e-13, atol=1e-14)

    def test_boundary_solutions_and_nonconverged_starts_are_retained(self):
        rows, selected = fit_multistart(self.oracle, np.zeros_like(self.Y), self.ids)
        self.assertEqual(HEAT_RANGE, MODEL_HEAT_RANGE)
        self.assertEqual(NOISE_RANGE, MODEL_NOISE_RANGE)
        self.assertEqual(len(rows), 2)
        self.assertTrue(selected["valid"])
        self.assertAlmostEqual(selected["fit_tau"], HEAT_RANGE[1], places=12)
        self.assertAlmostEqual(selected["fit_noise_variance"], NOISE_RANGE[0], places=12)
        self.assertIn("tau_upper", json.loads(selected["boundary_flags"]))
        self.assertIn("noise_lower", json.loads(selected["boundary_flags"]))
        self.assertGreater(selected["gradient_norm"], 1e-4)
        self.assertLess(selected["projected_gradient_norm"], 1e-9)
        self.assertTrue(selected["stationary"])
        limited, best = fit_multistart(self.oracle, self.Y, self.ids, maxiter=1)
        self.assertEqual(len(limited), 2)
        self.assertTrue(all(row["valid"] and not row["optimizer_success"] for row in limited))
        self.assertEqual(best["selection_status"], "selected_not_converged")
        self.assertEqual(best["nll_per_observation"], min(row["nll_per_observation"] for row in limited))

    def test_factorization_failures_are_explicit_and_retained_without_jitter(self):
        with patch("graphgp.analysis.heat_identifiability.cholesky",
                   side_effect=np.linalg.LinAlgError("test non-SPD covariance")) as factor:
            with self.assertRaises(np.linalg.LinAlgError):
                self.oracle.nll(0.6, 0.03, self.Y, self.ids)
            rows, selected = fit_multistart(self.oracle, self.Y, self.ids)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(not row["valid"] for row in rows))
            self.assertTrue(all("LinAlgError" in row["optimizer_message"] for row in rows))
            self.assertEqual(selected["selection_status"], "no_valid_start")
            self.assertEqual(factor.call_count, 3)

    def test_invalid_parameters_data_and_masks_fail_clearly(self):
        for ids in ([0, 0], [-1, 2], [0, 9], [0.2, 1.2], []):
            with self.assertRaises(ValueError):
                self.oracle.covariance(0.6, 0.03, ids)
        for tau, noise in ((-1, 0.03), (0.6, -0.03), (np.nan, 0.03)):
            with self.assertRaises(ValueError):
                self.oracle.nll(tau, noise, self.Y, self.ids)
        with self.assertRaises(ValueError):
            self.oracle.nll(0.6, 0.03, np.full_like(self.Y, np.nan), self.ids)
        with self.assertRaises(ValueError):
            HeatOracle([[1, 1], [0, 1]])
        with self.assertRaises(ValueError):
            HeatOracle([[-1, 0], [0, 1]])


if __name__ == "__main__":
    unittest.main()
