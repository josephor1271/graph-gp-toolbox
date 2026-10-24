"""Separable covariance, coordinates and derivatives against independent algebra."""

from contextlib import ExitStack, contextmanager
import itertools
import unittest
from unittest.mock import patch

import gpytorch
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import torch
from linear_operator.operators import KroneckerProductLinearOperator

from graphgp.inference.predict import predict_marginals
from graphgp.kernels.graph_time import GraphTimeHeatRBFKernel
from graphgp.kernels.heat import GraphHeatKernel
from graphgp.models.graph_time import make_graph_time_gp
from graphgp.operators.chebyshev import ChebyshevFunctionLinearOperator
from graphgp.utils.graphs import make_laplacian
from graphgp.validation import inference_context


DTYPE = torch.float64


def all_pairs(n, num_times):
    return torch.cartesian_prod(torch.arange(n, dtype=DTYPE),
                                torch.arange(num_times, dtype=DTYPE))


def independent_prior(laplacian, num_times, tau, ell):
    graph = scipy.linalg.expm(-tau * laplacian.toarray())
    times = np.arange(num_times, dtype=np.float64) / (num_times - 1)
    temporal = np.exp(-(times[:, None] - times[None, :])**2 / (2 * ell**2))
    return graph, temporal, np.kron(graph, temporal)


@contextmanager
def forbid_sparse_dense_paths():
    """Guard graph/full-prior assembly; small selected blocks remain allowed."""
    with ExitStack() as stack:
        for target, attribute in [
            (ChebyshevFunctionLinearOperator, "to_dense"),
            (KroneckerProductLinearOperator, "to_dense"),
            (torch, "kron"), (torch.linalg, "eigh"),
            (torch.linalg, "eigvalsh"), (scipy.linalg, "expm"),
            (scipy.linalg, "eigh"), (scipy.sparse.linalg, "eigsh"),
        ]:
            stack.enter_context(patch.object(
                target, attribute,
                side_effect=AssertionError(f"forbidden sparse path: {attribute}")))
        yield


class GraphTimeKernelTests(unittest.TestCase):
    def test_dense_all_pairs_and_sparse_mvp_parameter_grid(self):
        generator = torch.Generator().manual_seed(8301)
        for family, n, num_times, tau, ell in itertools.product(
                ("path", "grid"), (9, 16), (4, 7), (0.7, 2.2), (0.24, 0.56)):
            with self.subTest(family=family, n=n, T=num_times, tau=tau, ell=ell):
                L, _ = make_laplacian(family, n=n)
                graph, temporal, full = independent_prior(L, num_times, tau, ell)
                reference = torch.from_numpy(full)
                pairs = all_pairs(n, num_times)
                dense = GraphTimeHeatRBFKernel(
                    L, num_times=num_times, backend="dense_exact", heat_time=tau,
                    lengthscale=ell)
                torch.testing.assert_close(dense(pairs).to_dense(), reference,
                                           rtol=1e-11, atol=1e-12)
                selected_left = pairs[torch.tensor([n * num_times - 1, 3, 3, 8, 1])]
                selected_right = pairs[torch.tensor([4, 0, n * num_times - 2, 3])]
                left_ids = (selected_left[:, 0] * num_times + selected_left[:, 1]).long()
                right_ids = (selected_right[:, 0] * num_times + selected_right[:, 1]).long()
                with forbid_sparse_dense_paths():
                    sparse = GraphTimeHeatRBFKernel(
                        L, num_times=num_times, backend="sparse_chebyshev",
                        heat_time=tau, lengthscale=ell)
                    base = sparse._full_grid_operator()
                    for width in (1, 3):
                        rhs = torch.randn(n * num_times, width, dtype=DTYPE,
                                          generator=generator)
                        actual = base @ rhs
                        torch.testing.assert_close(actual, reference @ rhs,
                                                   rtol=1e-9, atol=1e-10)
                        # Non-separable, non-symmetric RHS catches factor ordering.
                        x = rhs.numpy().reshape(n, num_times, width)
                        ordered = np.stack([graph @ x[:, :, k] @ temporal.T
                                            for k in range(width)], axis=-1)
                        torch.testing.assert_close(actual, torch.from_numpy(
                            ordered.reshape(n * num_times, width)), rtol=1e-9, atol=1e-10)
                    vector = torch.randn(n * num_times, dtype=DTYPE, generator=generator)
                    torch.testing.assert_close(base @ vector, reference @ vector,
                                               rtol=1e-9, atol=1e-10)
                    selected = sparse(selected_left, selected_right).to_dense()
                    torch.testing.assert_close(selected, reference[left_ids][:, right_ids],
                                               rtol=1e-9, atol=1e-10)

    def test_full_graph_helper_and_tau_derivative(self):
        generator = torch.Generator().manual_seed(982)
        for family, n, tau in itertools.product(("path", "grid"), (9, 16), (0.7, 2.2)):
            with self.subTest(family=family, n=n, tau=tau):
                L, _ = make_laplacian(family, n=n)
                exact = scipy.linalg.expm(-tau * L.toarray())
                derivative = -L.toarray() @ exact
                rhs = torch.randn(n, 3, dtype=DTYPE, generator=generator)
                weights = torch.randn(n, 3, dtype=DTYPE, generator=generator)
                with forbid_sparse_dense_paths():
                    kernel = GraphHeatKernel(L, lambda_min=0, lambda_max=2,
                                             num_coefficients=33, num_nodes=33,
                                             heat_time=tau)
                    operator = kernel.full_graph_operator()
                    self.assertIs(operator.L, kernel.L)
                    actual = operator @ rhs
                    raw_gradient, = torch.autograd.grad((actual * weights).sum(),
                                                       kernel.raw_heat_time)
                    jacobian, = torch.autograd.grad(kernel.heat_time, kernel.raw_heat_time)
                torch.testing.assert_close(actual, torch.from_numpy(exact) @ rhs,
                                           rtol=1e-9, atol=1e-10)
                expected = ((torch.from_numpy(derivative) @ rhs) * weights).sum()
                torch.testing.assert_close(raw_gradient / jacobian, expected,
                                           rtol=1e-8, atol=1e-10)

    def test_selections_diagonals_repeats_symmetry_and_psd(self):
        L, _ = make_laplacian("grid", n=9)
        num_times, tau, ell = 7, 1.2, 0.24
        _, _, full = independent_prior(L, num_times, tau, ell)
        left = torch.tensor([[8, 6], [0, 1], [3, 4], [0, 1], [2, 0]], dtype=DTYPE)
        right = torch.tensor([[2, 6], [0, 1], [1, 0], [4, 3], [8, 5]], dtype=DTYPE)
        li = (left[:, 0] * num_times + left[:, 1]).long()
        ri = (right[:, 0] * num_times + right[:, 1]).long()
        ref = torch.from_numpy(full)
        for backend in ("dense_exact", "sparse_chebyshev"):
            with self.subTest(backend=backend):
                kernel = GraphTimeHeatRBFKernel(L, num_times=num_times, backend=backend,
                                               heat_time=tau, lengthscale=ell)
                cov = kernel(left).to_dense()
                self.assertEqual(cov.shape, (5, 5))
                torch.testing.assert_close(cov, cov.T, rtol=1e-12, atol=1e-12)
                self.assertGreaterEqual(torch.linalg.eigvalsh(cov).min().item(), -1e-12)
                torch.testing.assert_close(cov[1], cov[3], rtol=0, atol=0)
                torch.testing.assert_close(kernel(left, diag=True), ref[li, li],
                                           rtol=1e-9, atol=1e-10)
                torch.testing.assert_close(kernel(left, right, diag=True), ref[li, ri],
                                           rtol=1e-9, atol=1e-10)
                cross = kernel(left, right[:3]).to_dense()
                self.assertEqual(cross.shape, (5, 3))
                torch.testing.assert_close(cross, ref[li][:, ri[:3]], rtol=1e-9, atol=1e-10)
                torch.testing.assert_close(kernel(left, right[:3], diag=True),
                                           ref[li[:3], ri[:3]], rtol=1e-9, atol=1e-10)
                with self.assertRaises(AttributeError):
                    kernel.heat_time = 1.0
                with self.assertRaises(AttributeError):
                    kernel.lengthscale = 0.3

    def test_tau_and_ell_gradients_against_dense_torch_and_finite_differences(self):
        L, _ = make_laplacian("grid", n=9)
        times = torch.arange(7, dtype=DTYPE) / 6
        left = torch.tensor([[8, 6], [0, 1], [3, 4], [0, 1], [2, 0]], dtype=DTYPE)
        right = torch.tensor([[2, 6], [0, 1], [1, 0], [4, 3]], dtype=DTYPE)
        li = (left[:, 0] * 7 + left[:, 1]).long()
        ri = (right[:, 0] * 7 + right[:, 1]).long()
        u = torch.tensor([0.2, -0.7, 0.9, 0.6, -0.4], dtype=DTYPE)
        v = torch.tensor([0.6, 0.8, -0.5, 0.3], dtype=DTYPE)
        for tau, ell in ((0.7, 0.24), (2.2, 0.56)):
            physical_tau = torch.tensor(tau, dtype=DTYPE, requires_grad=True)
            physical_ell = torch.tensor(ell, dtype=DTYPE, requires_grad=True)
            graph = torch.matrix_exp(-physical_tau * torch.from_numpy(L.toarray()))
            temporal = torch.exp(-(times[:, None] - times[None, :])**2 / (2 * physical_ell**2))
            full = torch.kron(graph, temporal)
            expected = torch.autograd.grad(u @ full[li][:, ri] @ v,
                                           (physical_tau, physical_ell))
            def scalar_value(a, b):
                _, _, matrix = independent_prior(L, 7, a, b)
                return (u.numpy() @ matrix[li.numpy()][:, ri.numpy()] @ v.numpy()).item()
            epsilon = 1e-5
            finite_difference = (
                (scalar_value(tau + epsilon, ell) - scalar_value(tau - epsilon, ell)) / (2 * epsilon),
                (scalar_value(tau, ell + epsilon) - scalar_value(tau, ell - epsilon)) / (2 * epsilon))
            for backend in ("dense_exact", "sparse_chebyshev"):
                with self.subTest(tau=tau, ell=ell, backend=backend):
                    kernel = GraphTimeHeatRBFKernel(L, num_times=7, backend=backend,
                                                   heat_time=tau, lengthscale=ell)
                    value = u @ (kernel(left, right) @ v)
                    raw_parameters = (kernel.graph_kernel.raw_heat_time,
                                      kernel.time_kernel.raw_lengthscale)
                    actual_raw = torch.autograd.grad(value, raw_parameters)
                    jacobians = (torch.autograd.grad(kernel.heat_time, raw_parameters[0])[0],
                                 torch.autograd.grad(kernel.lengthscale, raw_parameters[1])[0])
                    actual = [a.reshape(()) / b.reshape(()) for a, b in zip(actual_raw, jacobians)]
                    for result, dense_gradient, fd in zip(actual, expected, finite_difference):
                        self.assertTrue(torch.isfinite(result))
                        self.assertGreater(abs(result.item()), 1e-6)
                        torch.testing.assert_close(result, dense_gradient, rtol=1e-8, atol=1e-10)
                        self.assertAlmostEqual(result.item(), fd, delta=2e-8)

    def test_input_validation_and_bounds(self):
        L, _ = make_laplacian("path", n=9)
        valid = torch.tensor([[0, 0], [8, 6]], dtype=DTYPE)
        invalid = [torch.empty(0, 2, dtype=DTYPE), torch.ones(2, dtype=DTYPE),
                   torch.ones(2, 3, dtype=DTYPE), torch.ones(1, 2, 2, dtype=DTYPE),
                   valid.float(), valid.long(), valid.bool(), valid.to(torch.complex128),
                   torch.tensor([[0.5, 1]], dtype=DTYPE),
                   torch.tensor([[0, 1.5]], dtype=DTYPE),
                   torch.tensor([[float("nan"), 0]], dtype=DTYPE),
                   torch.tensor([[0, float("inf")]], dtype=DTYPE),
                   torch.tensor([[-1, 0]], dtype=DTYPE), torch.tensor([[9, 0]], dtype=DTYPE),
                   torch.tensor([[0, -1]], dtype=DTYPE), torch.tensor([[0, 7]], dtype=DTYPE),
                   torch.tensor([[2**53, 0]], dtype=DTYPE)]
        for backend in ("dense_exact", "sparse_chebyshev"):
            kernel = GraphTimeHeatRBFKernel(L, num_times=7, backend=backend)
            for pairs in invalid:
                with self.subTest(backend=backend, shape=pairs.shape, dtype=pairs.dtype):
                    with self.assertRaises(ValueError):
                        kernel.validate_pairs(pairs)
                    with self.assertRaises(ValueError):
                        kernel.forward(valid, pairs)
            with self.assertRaises(ValueError):
                kernel.forward(valid, valid, last_dim_is_batch=True)
            with self.assertRaises(ValueError):
                GraphTimeHeatRBFKernel(L, num_times=7, backend=backend,
                                       batch_shape=torch.Size([2]))
            for name, value in [("heat_time", 0.05), ("heat_time", 4.0),
                                ("lengthscale", 0.08), ("lengthscale", 0.8),
                                ("heat_time", float("nan")), ("lengthscale", float("inf"))]:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    GraphTimeHeatRBFKernel(L, num_times=7, backend=backend, **{name: value})
        for num_times in (0, 1, 4.0, True):
            with self.assertRaises(ValueError):
                GraphTimeHeatRBFKernel(L, num_times=num_times, backend="dense_exact")
        with self.assertRaises(ValueError):
            GraphTimeHeatRBFKernel(L, num_times=7, backend="unknown")
        if torch.cuda.is_available():
            with self.assertRaises(ValueError):
                kernel.validate_pairs(valid.cuda())

    def test_factory_double_initialization_noise_freeze_and_repeat_semantics(self):
        L, _ = make_laplacian("path", n=9)
        pairs = torch.tensor([[0, 1], [2, 2], [0, 1]], dtype=DTYPE)
        y = torch.tensor([0.2, -0.1, 0.3], dtype=DTYPE)
        tau, ell, noise = 1.23456789012345, 0.345678901234567, 0.123456789012345
        for backend, learn_noise in itertools.product(("dense_exact", "sparse_chebyshev"), (False, True)):
            with self.subTest(backend=backend, learn_noise=learn_noise):
                model = make_graph_time_gp(L, pairs, y, num_times=7, backend=backend,
                                           heat_time=tau, lengthscale=ell,
                                           noise_variance=noise, learn_noise=learn_noise)
                kernel = model.covar_module
                self.assertNotIsInstance(kernel, gpytorch.kernels.ScaleKernel)
                for actual, expected in ((kernel.heat_time, tau), (kernel.lengthscale, ell),
                                         (model.likelihood.noise, noise)):
                    self.assertEqual(actual.dtype, DTYPE)
                    self.assertAlmostEqual(actual.item(), expected, delta=1e-14)
                self.assertEqual(model.likelihood.noise_covar.raw_noise.requires_grad, learn_noise)
                self.assertEqual(sum(p.numel() for p in model.likelihood.parameters() if p.requires_grad),
                                 int(learn_noise))
                prior = model(pairs)
                latent, observed = prior.covariance_matrix, model.likelihood(prior).covariance_matrix
                torch.testing.assert_close(observed - latent, noise * torch.eye(3, dtype=DTYPE),
                                           rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(latent[0], latent[2], rtol=0, atol=0)
                self.assertAlmostEqual(observed[0, 2].item(), latent[0, 0].item(), delta=1e-12)
        for bad_y in (y.float(), y[:, None], y[:2], torch.tensor([0., float("nan"), 1.], dtype=DTYPE)):
            with self.assertRaises(ValueError):
                make_graph_time_gp(L, pairs, bad_y, num_times=7, backend="dense_exact")

    def test_state_dict_fresh_models_preserve_buffers_parameters_and_predictions(self):
        L, _ = make_laplacian("path", n=9)
        train = torch.tensor([[0, 0], [3, 2], [7, 1], [1, 3], [3, 2]], dtype=DTYPE)
        targets = torch.tensor([[8, 6], [0, 6], [4, 4], [0, 6]], dtype=DTYPE)
        y = torch.tensor([0.3, -0.1, 0.6, -0.2, 0.4], dtype=DTYPE)
        for backend, method in (("dense_exact", "A"), ("sparse_chebyshev", "C")):
            with self.subTest(backend=backend):
                source = make_graph_time_gp(L, train, y, num_times=7, backend=backend,
                                            heat_time=1.2, lengthscale=0.24,
                                            noise_variance=0.06)
                target = make_graph_time_gp(L * 0.5, train, y, num_times=7, backend=backend,
                                            heat_time=0.8, lengthscale=0.5,
                                            noise_variance=0.12)
                target.load_state_dict(source.state_dict())
                for name, value in source.named_buffers():
                    torch.testing.assert_close(dict(target.named_buffers())[name], value, rtol=0, atol=0)
                for name, value in source.named_parameters():
                    torch.testing.assert_close(dict(target.named_parameters())[name], value, rtol=0, atol=0)
                with inference_context(method, len(train), cg_tolerance=1e-8):
                    expected = predict_marginals(source, targets, chunk_size=2)
                    actual = predict_marginals(target, targets, chunk_size=3)
                torch.testing.assert_close(actual.mean, expected.mean, rtol=1e-8, atol=1e-9)
                torch.testing.assert_close(actual.latent_variance, expected.latent_variance,
                                           rtol=1e-8, atol=1e-9)
                torch.testing.assert_close(actual.observation_variance, expected.observation_variance,
                                           rtol=1e-8, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
