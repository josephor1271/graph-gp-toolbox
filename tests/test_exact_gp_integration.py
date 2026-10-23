"""Dense oracles and exhaustive trace probes are ONLY for tiny tests."""
from contextlib import contextmanager
import unittest
import warnings
from unittest.mock import patch

import gpytorch
from linear_operator.operators import LinearOperator
from linear_operator.utils.warnings import NumericalWarning
import torch

from graphgp.demo import GraphGP, iterative_settings, make_problem
from graphgp.kernels.heat import GraphHeatKernel
from graphgp.operators.chebyshev import ChebyshevFunctionLinearOperator
from tests.helpers import DTYPE


class DenseHeatKernel(gpytorch.kernels.Kernel):
    def __init__(self, L, heat_time):
        super().__init__()
        values, vectors = torch.linalg.eigh(torch.tensor(L.toarray(), dtype=DTYPE))
        self.register_buffer("values", values)
        self.register_buffer("vectors", vectors)
        self.register_parameter("raw_heat_time", torch.nn.Parameter(torch.zeros((), dtype=DTYPE)))
        self.register_constraint("raw_heat_time", gpytorch.constraints.Positive())
        self.initialize(raw_heat_time=self.raw_heat_time_constraint.inverse_transform(heat_time))

    def forward(self, x1, x2, diag=False, **kwargs):
        t = self.raw_heat_time_constraint.transform(self.raw_heat_time)
        covariance = (self.vectors * torch.exp(-t * self.values)) @ self.vectors.T
        result = covariance[x1.squeeze(-1).long()][:, x2.squeeze(-1).long()]
        return result.diagonal() if diag else result


@contextmanager
def exhaustive_trace_probes():
    # Orthonormal probes remove stochastic trace error without changing CG,
    # Lanczos, SLQ, or the covariance operator. This O(n_train^2) test oracle
    # is deliberately absent from the production example.
    def probes(operator):
        n = operator.shape[-1]
        return (torch.eye(n, dtype=operator.dtype, device=operator.device),
                torch.full((1, n), n ** 0.5, dtype=operator.dtype, device=operator.device))
    with patch.object(LinearOperator, "_probe_vectors_and_norms", probes):
        yield


@contextmanager
def instrument_matrix_free():
    counts = {"matmul": 0}
    original = ChebyshevFunctionLinearOperator._matmul

    def counted(operator, rhs):
        counts["matmul"] += 1
        return original(operator, rhs)

    with warnings.catch_warnings(), \
         patch.object(ChebyshevFunctionLinearOperator, "_matmul", counted), \
         patch.object(ChebyshevFunctionLinearOperator, "to_dense", side_effect=AssertionError("dense base covariance")), \
         patch.object(LinearOperator, "_cholesky", side_effect=AssertionError("Cholesky used")):
        warnings.simplefilter("error", NumericalWarning)
        yield counts


class ExactGPTests(unittest.TestCase):
    def setUp(self):
        self.L, self.train_ids, self.train_y, self.test_ids = make_problem()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
        self.likelihood.noise = 0.12
        self.kernel = GraphHeatKernel(self.L, lambda_max=4, num_nodes=128, heat_time=1.1)
        self.model = GraphGP(self.train_ids, self.train_y, self.likelihood, self.kernel)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

    def dense_model(self):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
        likelihood.load_state_dict(self.likelihood.state_dict())
        kernel = DenseHeatKernel(self.L, self.kernel.heat_time.detach())
        return GraphGP(self.train_ids, self.train_y, likelihood, kernel), likelihood

    def test_fixed_parameters_mll_and_gradient(self):
        dense_model, dense_likelihood = self.dense_model()
        torch.testing.assert_close(
            self.model(self.train_ids).covariance_matrix,
            dense_model(self.train_ids).covariance_matrix, rtol=1e-10, atol=1e-12)
        dense_mll = gpytorch.mlls.ExactMarginalLogLikelihood(dense_likelihood, dense_model)
        with gpytorch.settings.max_cholesky_size(1000):
            expected = -dense_mll(dense_model(self.train_ids), self.train_y)
            expected.backward()
        with iterative_settings(), exhaustive_trace_probes(), instrument_matrix_free() as counts:
            actual = -self.mll(self.model(self.train_ids), self.train_y)
            forward_count = counts["matmul"]
            self.assertGreater(forward_count, 0)
            actual.backward()
            backward_count = counts["matmul"] - forward_count
            self.assertGreater(backward_count, 0)
        torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-8)
        torch.testing.assert_close(self.kernel.raw_heat_time.grad,
                                   dense_model.covar_module.raw_heat_time.grad, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(self.likelihood.raw_noise.grad, dense_likelihood.raw_noise.grad,
                                   rtol=1e-6, atol=1e-8)
        print(f"\nExactGP: MLL abs error={abs(actual.item()-expected.item()):.3e}; "
              f"MVP calls forward={forward_count}, backward={backward_count}")

    def test_default_stochastic_trace_path(self):
        with iterative_settings(), instrument_matrix_free() as counts, torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            loss = -self.mll(self.model(self.train_ids), self.train_y)
            self.assertTrue(torch.isfinite(loss))
            before = counts["matmul"]
            self.assertGreater(before, 0)
            loss.backward()
            self.assertGreater(counts["matmul"], before)
            self.assertTrue(torch.isfinite(self.kernel.raw_heat_time.grad))
            self.assertGreater(self.kernel.raw_heat_time.grad.abs().item(), 1e-8)

    def test_training_and_prediction(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.08)
        losses = []
        with iterative_settings(), exhaustive_trace_probes(), instrument_matrix_free() as counts:
            for step in range(21):
                optimizer.zero_grad()
                loss = -self.mll(self.model(self.train_ids), self.train_y)
                self.assertTrue(torch.isfinite(loss))
                losses.append(loss.item())
                if step < 20:
                    before = counts["matmul"]
                    loss.backward()
                    self.assertGreater(counts["matmul"], before)
                    for parameter in [self.kernel.raw_heat_time, self.likelihood.raw_noise]:
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        self.assertGreater(parameter.grad.abs().max().item(), 1e-10)
                    optimizer.step()
            self.assertLess(losses[-1], losses[0] - 0.05)
            self.model.eval()
            self.likelihood.eval()
            before = counts["matmul"]
            with torch.no_grad():
                prediction = self.likelihood(self.model(self.test_ids))
                mean, variance = prediction.mean, prediction.variance
            prediction_count = counts["matmul"] - before
            self.assertGreater(prediction_count, 0)
        self.assertTrue(mean.isfinite().all())
        self.assertTrue(variance.isfinite().all())
        self.assertTrue((variance > 0).all())
        dense_model, dense_likelihood = self.dense_model()
        dense_model.eval()
        dense_likelihood.eval()
        with gpytorch.settings.max_cholesky_size(1000), torch.no_grad():
            expected = dense_likelihood(dense_model(self.test_ids))
        torch.testing.assert_close(mean, expected.mean, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(variance, expected.variance, rtol=1e-5, atol=1e-5)
        print(f"\nExactGP training: loss {losses[0]:.6f} -> {losses[-1]:.6f}; "
              f"heat_time={self.kernel.heat_time.item():.6f}, noise={self.likelihood.noise.item():.6f}; "
              f"prediction MVP calls={prediction_count}; "
              f"mean max abs error={(mean-expected.mean).abs().max().item():.3e}; "
              f"variance max abs error={(variance-expected.variance).abs().max().item():.3e}")
