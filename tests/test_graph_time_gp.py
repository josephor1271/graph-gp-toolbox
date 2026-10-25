"""Actual ExactGP/MLL/prediction gates against independent dense references."""
import io
import unittest
import warnings

import gpytorch
import numpy as np
import torch

from graphgp.inference.predict import predict_marginals
from graphgp.models.graph_time import make_graph_time_gp
from graphgp.validation_graph_time import (
    graph_time_inference,
    measure_validation,
    tiny_validation_problem,
)


class GraphTimeGPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.measurements = measure_validation()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_normalized_marginal_likelihood(self):
        for case in self.measurements["cases"]:
            for backend, result in case["backends"].items():
                with self.subTest(parameters=case["conditions"], backend=backend):
                    self.assertTrue(np.isfinite(result["nll"]))
                    if backend == "dense_exact":
                        self.assertLessEqual(result["nll_relative_error"], 1e-9)
                    else:
                        self.assertLessEqual(result["nll_abs_error"], 1e-5)

    def test_physical_and_raw_heat_lengthscale_noise_gradients(self):
        for case in self.measurements["cases"]:
            for backend, result in case["backends"].items():
                for parameter, gradient in result["gradients"].items():
                    with self.subTest(backend=backend, parameter=parameter,
                                      parameters=case["conditions"]):
                        for kind in ("physical", "raw"):
                            actual = gradient[kind]
                            self.assertTrue(np.isfinite(actual))
                            self.assertGreater(abs(actual), 1e-12)
                            np.testing.assert_allclose(actual, gradient["reference_" + kind],
                                                       rtol=2e-4, atol=2e-6)

    def test_latent_observation_forecast_marginals_chunking_mean_only(self):
        problem = tiny_validation_problem()
        self.assertTrue((problem["train_pairs"][:, 1] < 4).all())
        self.assertTrue((problem["target_pairs"][:, 1] == 4).any())
        self.assertLess(len(np.unique(problem["target_pairs"], axis=0)),
                        len(problem["target_pairs"]))
        for case in self.measurements["cases"]:
            for backend, result in case["backends"].items():
                tolerance = 1e-8 if backend == "dense_exact" else 2e-4
                for name, error in result["posterior"].items():
                    with self.subTest(backend=backend, measurement=name,
                                      parameters=case["conditions"]):
                        self.assertLessEqual(error, tolerance)

    def test_sparse_run_uses_chebyshev_cg_slq_and_retains_warnings(self):
        for case in self.measurements["cases"]:
            dense = case["backends"]["dense_exact"]
            self.assertEqual(dense["solver"]["cg_calls"], 0)
            self.assertEqual(dense["solver"]["slq_calls"], 0)
            sparse = case["backends"]["sparse_chebyshev"]
            self.assertGreater(sparse["solver"]["cg_calls"], 0)
            self.assertGreater(sparse["solver"]["slq_calls"], 0)
            self.assertEqual(sparse["solver"]["cg_tridiag_counts"][0], 18)
            self.assertLessEqual(sparse["solver"]["cg_tolerance"], 1e-8)
            self.assertEqual(sparse["solver"]["installed_cg_eps"], 1e-10)
            for phase in ("forward", "backward", "prediction"):
                count = sparse["operator_phases"][phase]["operator_matmul_calls"]
                self.assertGreater(count, 0, phase)
                self.assertEqual(sparse["operator_phases"][phase]["estimated_sparse_l_mvps"],
                                 32 * count)
            # A dense observed covariance assembled by selected entry queries
            # would not demonstrate the requested matrix-free training path.
            self.assertEqual(sparse["operator_phases"]["forward"]["get_indices_calls"], 0)
            self.assertTrue(all("category" in message and "message" in message
                                for message in sparse["warnings"]))
            self.assertTrue(self.measurements["guards"]["full_kronecker_to_dense"])

    def _make_model(self, backend, **kwargs):
        problem = tiny_validation_problem()
        model = make_graph_time_gp(
            problem["L"], torch.tensor(problem["train_pairs"], dtype=torch.float64),
            torch.tensor(problem["train_y"], dtype=torch.float64), num_times=5,
            backend=backend, **kwargs)
        return model, torch.tensor(problem["target_pairs"], dtype=torch.float64)

    def test_frozen_noise_precision_and_gpytorch_training(self):
        tau, ell, noise = 0.735012345678901, 0.314159265358979, 0.067891234567891
        for backend in ("dense_exact", "sparse_chebyshev"):
            with self.subTest(backend=backend), warnings.catch_warnings(record=True):
                model, _ = self._make_model(backend, heat_time=tau, lengthscale=ell,
                                            noise_variance=noise)
                self.assertFalse(model.likelihood.raw_noise.requires_grad)
                self.assertEqual(sum(parameter.numel() for parameter in model.parameters()
                                     if parameter.requires_grad), 2)
                self.assertLessEqual(abs(model.covar_module.heat_time.item() - tau), 1e-14)
                self.assertLessEqual(abs(model.covar_module.lengthscale.item() - ell), 1e-14)
                self.assertLessEqual(abs(model.likelihood.noise.item() - noise), 1e-14)
                model.train()
                model.likelihood.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
                losses = []
                with graph_time_inference(backend, 18):
                    for _ in range(3):
                        optimizer.zero_grad()
                        loss = -mll(model(model.train_inputs[0]), model.train_targets)
                        losses.append(loss.item())
                        loss.backward()
                        for parameter in (model.covar_module.graph_kernel.raw_heat_time,
                                          model.covar_module.time_kernel.raw_lengthscale):
                            self.assertIsNotNone(parameter.grad)
                            self.assertTrue(torch.isfinite(parameter.grad).all())
                            self.assertGreater(parameter.grad.abs().max().item(), 1e-10)
                        self.assertIsNone(model.likelihood.raw_noise.grad)
                        optimizer.step()
                self.assertLess(losses[-1], losses[0])
                self.assertLessEqual(abs(model.likelihood.noise.item() - noise), 1e-14)

    def test_state_dict_roundtrip_fresh_gp_preserves_predictions_and_buffers(self):
        for backend in ("dense_exact", "sparse_chebyshev"):
            with self.subTest(backend=backend), warnings.catch_warnings(record=True):
                model, targets = self._make_model(
                    backend, heat_time=1.1, lengthscale=0.27, noise_variance=0.08123456789)
                serialized = io.BytesIO()
                torch.save(model.state_dict(), serialized)
                serialized.seek(0)
                restored, _ = self._make_model(
                    backend, heat_time=2.2, lengthscale=0.65, noise_variance=0.2)
                restored.load_state_dict(torch.load(serialized, weights_only=True))
                self.assertIsNot(model, restored)
                self.assertIsNot(model.covar_module, restored.covar_module)
                torch.testing.assert_close(model.covar_module.time_grid,
                                           restored.covar_module.time_grid, rtol=0, atol=0)
                with graph_time_inference(backend, 18):
                    expected = predict_marginals(model, targets, chunk_size=8)
                    actual = predict_marginals(restored, targets, chunk_size=3)
                for name in ("mean", "latent_variance", "observation_variance"):
                    torch.testing.assert_close(getattr(actual, name), getattr(expected, name),
                                               rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
