import unittest
from unittest.mock import patch

import gpytorch
import numpy as np
import scipy.linalg
import scipy.sparse
import torch

from graphgp.kernels.dense import DenseChebyshevGraphHeatKernel, DenseExactGraphHeatKernel
from graphgp.kernels.heat import GraphHeatKernel
from tests.helpers import DTYPE, path_laplacian, random_tensor


def _laplacian_from_edges(n, edges):
    adjacency = np.zeros((n, n))
    for left, right in edges:
        adjacency[left, right] = adjacency[right, left] = 1
    return scipy.sparse.csr_matrix(np.diag(adjacency.sum(axis=1)) - adjacency)


def _graphs():
    rows, columns = 3, 4
    grid_edges = []
    for row in range(rows):
        for column in range(columns):
            node = row * columns + column
            if row + 1 < rows:
                grid_edges.append((node, node + columns))
            if column + 1 < columns:
                grid_edges.append((node, node + 1))
    rng = np.random.default_rng(123)
    irregular_edges = [(i, i + 1) for i in range(10)]
    irregular_edges.extend((i, j) for i in range(11) for j in range(i + 2, 11)
                           if rng.random() < 0.18)
    return {
        "path": path_laplacian(9),
        "grid": _laplacian_from_edges(rows * columns, grid_edges),
        "irregular": _laplacian_from_edges(11, irregular_edges),
        "disconnected": scipy.sparse.block_diag(
            [path_laplacian(4), path_laplacian(3)], format="csr"),
    }


class DenseBaselineKernelTests(unittest.TestCase):
    def setUp(self):
        self.L = path_laplacian(9)
        self.ids = torch.arange(9)
        self.left = torch.tensor([8, 1, 1, 4])
        self.right = torch.tensor([0, 6, 2])

    def test_exact_covariance_matches_independent_matrix_exponential(self):
        for name, L in _graphs().items():
            with self.subTest(graph=name):
                kernel = DenseExactGraphHeatKernel(L, heat_time=1.1)
                ids = torch.arange(L.shape[0])
                expected = torch.from_numpy(scipy.linalg.expm(-1.1 * L.toarray()))
                torch.testing.assert_close(kernel(ids).to_dense(), expected,
                                           rtol=2e-12, atol=2e-14)
                left, right = ids[[0, -1, 2, 2]], ids[[1, 3]]
                torch.testing.assert_close(kernel(left, right).to_dense(), expected[left][:, right],
                                           rtol=2e-12, atol=2e-14)
                torch.testing.assert_close(kernel(left, diag=True), expected.diagonal()[left],
                                           rtol=2e-12, atol=2e-14)

    def test_exact_raw_gradient_matches_analytic_derivative(self):
        kernel = DenseExactGraphHeatKernel(self.L, heat_time=1.1)
        weights = random_tensor(self.left.numel(), self.right.numel())
        loss = (kernel(self.left, self.right).to_dense() * weights).sum()
        actual, = torch.autograd.grad(loss, kernel.raw_heat_time)
        dense = self.L.toarray()
        derivative = torch.from_numpy(-dense @ scipy.linalg.expm(-1.1 * dense))
        expected = ((derivative[self.left][:, self.right] * weights).sum()
                    * torch.sigmoid(kernel.raw_heat_time.detach()))
        self.assertGreater(abs(actual.item()), 1e-8)
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-14)

    def test_dense_and_sparse_chebyshev_blocks_and_gradients(self):
        for name, L in _graphs().items():
            maximum = 2 * float(L.diagonal().max())
            for degree, heat_time in [(4, 4.0), (16, 1.1), (32, 0.25)]:
                with self.subTest(graph=name, degree=degree, heat_time=heat_time):
                    options = dict(lambda_max=maximum, num_coefficients=degree + 1,
                                   num_nodes=128, heat_time=heat_time)
                    dense = DenseChebyshevGraphHeatKernel(L, **options)
                    sparse = GraphHeatKernel(L, **options)
                    torch.testing.assert_close(dense.spectral_nodes, sparse.spectral_nodes)
                    torch.testing.assert_close(dense.cosine_basis, sparse.cosine_basis)
                    torch.testing.assert_close(dense.raw_heat_time, sparse.raw_heat_time)
                    ids = torch.arange(L.shape[0])
                    left, right = ids[[0, -1, 2, 2]], ids[[1, 3]]
                    for x1, x2 in [(ids, ids), (left, right), (right, right)]:
                        torch.testing.assert_close(dense(x1, x2).to_dense(),
                                                   sparse(x1, x2).to_dense(),
                                                   rtol=2e-11, atol=2e-13)
                    torch.testing.assert_close(dense(left, diag=True), sparse(left, diag=True),
                                               rtol=2e-11, atol=2e-13)
                    weights = random_tensor(left.numel(), right.numel())
                    dense_grad, = torch.autograd.grad(
                        (dense(left, right).to_dense() * weights).sum(), dense.raw_heat_time)
                    sparse_grad, = torch.autograd.grad(
                        (sparse(left, right).to_dense() * weights).sum(), sparse.raw_heat_time)
                    torch.testing.assert_close(dense_grad, sparse_grad, rtol=2e-10, atol=2e-13)

    def test_baselines_do_not_use_custom_operator(self):
        with patch("graphgp.operators.chebyshev.ChebyshevFunctionLinearOperator.__init__",
                   side_effect=AssertionError("dense baseline used custom sparse operator")):
            for cls in (DenseExactGraphHeatKernel, DenseChebyshevGraphHeatKernel):
                kernel = cls(self.L)
                block = kernel.forward(self.left[:, None], self.right[:, None])
                self.assertIsInstance(block, torch.Tensor)
                self.assertEqual(block.layout, torch.strided)
                self.assertEqual(block.shape, (4, 3))
                kernel(self.ids).to_dense().sum().backward()
                self.assertTrue(torch.isfinite(kernel.raw_heat_time.grad))

    def test_eigendecomposition_only_at_construction(self):
        for cls in (DenseExactGraphHeatKernel, DenseChebyshevGraphHeatKernel):
            with self.subTest(kernel=cls.__name__), patch(
                    "torch.linalg.eigh", wraps=torch.linalg.eigh) as eigh:
                kernel = cls(self.L)
                for time in [0.25, 1.1, 4.0]:
                    kernel.heat_time = time
                    kernel(self.ids).to_dense()
                self.assertEqual(eigh.call_count, 1)
                self.assertEqual(set(["evals", "evecs"]) & set(dict(kernel.named_buffers())),
                                 {"evals", "evecs"})

    def test_input_types_and_dtype_conversion(self):
        for cls in (DenseExactGraphHeatKernel, DenseChebyshevGraphHeatKernel):
            for dtype in [torch.float32, torch.float64]:
                dense_input = torch.tensor(self.L.toarray(), dtype=dtype)
                inputs = [self.L.astype(np.float32 if dtype == torch.float32 else np.float64),
                          dense_input, dense_input.to_sparse_csr()]
                for L in inputs:
                    with self.subTest(kernel=cls.__name__, dtype=dtype, input=type(L)):
                        kernel = cls(L, heat_time=1.1)
                        self.assertEqual(kernel.evals.dtype, dtype)
                        self.assertEqual(kernel.evecs.dtype, dtype)
                        self.assertEqual(kernel.raw_heat_time.dtype, dtype)
                        expected = torch.as_tensor(scipy.linalg.expm(-1.1 * self.L.toarray()),
                                                   dtype=dtype)
                        torch.testing.assert_close(kernel(self.ids).to_dense(), expected,
                                                   rtol=2e-5, atol=2e-6)
                        kernel.to(dtype=torch.float64)
                        self.assertTrue(all(buffer.dtype == torch.float64 for buffer in
                                            [kernel.evals, kernel.evecs, kernel.raw_heat_time]))
                        if isinstance(kernel, DenseChebyshevGraphHeatKernel):
                            self.assertEqual(kernel.spectral_nodes.dtype, torch.float64)
                            self.assertEqual(kernel.cosine_basis.dtype, torch.float64)

    def test_invalid_inputs_and_heat_times(self):
        for cls in (DenseExactGraphHeatKernel, DenseChebyshevGraphHeatKernel):
            kernel = cls(self.L)
            for value in [0, -1, float("nan"), float("inf"), [1, 2]]:
                with self.subTest(kernel=cls.__name__, heat_time=value), self.assertRaises(ValueError):
                    kernel.heat_time = value
            for ids in [torch.tensor([-1]), torch.tensor([9]), torch.tensor([1.5]),
                        torch.tensor([float("nan")]), torch.tensor([float("inf")]),
                        torch.ones(2, 2), torch.ones(2, 3, 1), torch.tensor([True]),
                        torch.tensor([1 + 0j])]:
                with self.subTest(kernel=cls.__name__, ids=ids), self.assertRaises(ValueError):
                    kernel(ids).to_dense()
            for L in [torch.empty(0, 0), torch.ones(2, 3), torch.tensor([[1., 2.], [0., 1.]]),
                      torch.tensor([[float("nan")]]), torch.eye(2, requires_grad=True)]:
                with self.subTest(kernel=cls.__name__, L=L), self.assertRaises(ValueError):
                    cls(L)
            with self.assertRaises(TypeError):
                cls(torch.eye(2, dtype=torch.complex128))
            with self.assertRaises(ValueError):
                cls(self.L, batch_shape=torch.Size([2]))
            with self.assertRaises(ValueError):
                kernel(self.ids, last_dim_is_batch=True).to_dense()

    def test_zero_modes_and_edgeless_graph(self):
        L = scipy.sparse.csr_matrix((3, 3), dtype=np.float64)
        exact = DenseExactGraphHeatKernel(L, heat_time=4)
        approximate = DenseChebyshevGraphHeatKernel(L, heat_time=4, lambda_max=1)
        for kernel in (exact, approximate):
            torch.testing.assert_close(kernel(torch.arange(3)).to_dense(), torch.eye(3, dtype=DTYPE),
                                       rtol=2e-12, atol=2e-14)
        with self.assertRaisesRegex(ValueError, "lambda_min < lambda_max"):
            DenseChebyshevGraphHeatKernel(L)

    def test_chebyshev_state_dict_preserves_interval(self):
        original = DenseChebyshevGraphHeatKernel(self.L, lambda_max=4, heat_time=1.1)
        restored = DenseChebyshevGraphHeatKernel(self.L, lambda_max=5, heat_time=2)
        restored.load_state_dict(original.state_dict())
        self.assertEqual(restored.lambda_max, 4)
        torch.testing.assert_close(restored(self.ids).to_dense(), original(self.ids).to_dense())

    def test_scale_kernel(self):
        for cls in (DenseExactGraphHeatKernel, DenseChebyshevGraphHeatKernel):
            kernel = cls(self.L)
            scaled = gpytorch.kernels.ScaleKernel(kernel).double()
            scaled.outputscale = 2
            torch.testing.assert_close(scaled(self.ids).to_dense(), 2 * kernel(self.ids).to_dense())


if __name__ == "__main__":
    unittest.main()
