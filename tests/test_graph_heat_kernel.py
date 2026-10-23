import unittest
from unittest.mock import patch

import gpytorch
import scipy.sparse
import scipy.sparse.linalg
import torch

from graphgp.kernels.heat import GraphHeatKernel
from tests.helpers import DTYPE, dense_heat, path_laplacian, random_tensor


class GraphKernelTests(unittest.TestCase):
    def setUp(self):
        self.L = path_laplacian(9)
        self.kernel = GraphHeatKernel(self.L, lambda_max=4, heat_time=1.1, num_nodes=128)
        self.exact = dense_heat(self.L, 1.1)

    def test_full_covariance(self):
        actual = self.kernel(torch.arange(9)).to_dense()
        torch.testing.assert_close(actual, self.exact, rtol=1e-10, atol=1e-12)

    def test_rectangular_subsets_and_duplicates(self):
        left, right = torch.tensor([8, 1, 1, 4]), torch.tensor([0, 6, 2])
        actual = self.kernel(left[:, None], right[:, None]).to_dense()
        torch.testing.assert_close(actual, self.exact[left][:, right], rtol=1e-10, atol=1e-12)

    def test_exact_diagonal(self):
        ids = torch.tensor([8, 2, 2, 0])
        actual = self.kernel(ids, diag=True)
        torch.testing.assert_close(actual, self.exact.diagonal()[ids], rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(self.kernel(ids).diagonal(), actual)

    def test_gradient_matches_dense(self):
        ids = torch.tensor([8, 1, 4, 0])
        u, v = random_tensor(4), random_tensor(4, seed=567)
        value = u @ (self.kernel(ids) @ v)
        actual, = torch.autograd.grad(value, self.kernel.raw_heat_time)
        raw = self.kernel.raw_heat_time.detach().clone().requires_grad_()
        time = self.kernel.raw_heat_time_constraint.transform(raw)
        exact = u @ dense_heat(self.L, time)[ids][:, ids] @ v
        expected, = torch.autograd.grad(exact, raw)
        self.assertTrue(torch.isfinite(actual))
        self.assertGreater(abs(actual.item()), 1e-8)
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-11)

    def test_dtype_conversion(self):
        ids = torch.arange(9)
        for dtype in [torch.float32, torch.float64]:
            self.kernel.to(dtype=dtype)
            for buffer in [self.kernel.L, self.kernel.spectral_nodes, self.kernel.cosine_basis]:
                self.assertEqual(buffer.dtype, dtype)
            self.assertEqual(self.kernel.raw_heat_time.dtype, dtype)
            # float -> double cannot recover precision lost during float conversion.
            torch.testing.assert_close(self.kernel(ids).to_dense(), self.exact.to(dtype), rtol=2e-5, atol=2e-6)

    def test_bounds_estimated_only_at_construction(self):
        with patch("scipy.sparse.linalg.eigsh", wraps=scipy.sparse.linalg.eigsh) as estimate:
            kernel = GraphHeatKernel(self.L)
            self.assertEqual(estimate.call_count, 1)
            for _ in range(2):
                kernel(torch.arange(9)) @ random_tensor(9)
            self.assertEqual(estimate.call_count, 1)
            self.assertEqual(kernel.lambda_min, 0)

    def test_invalid_ids(self):
        for ids in [torch.tensor([-1]), torch.tensor([9]), torch.tensor([1.5]),
                    torch.tensor([float('nan')]), torch.tensor([float('inf')]),
                    torch.ones(2, 2), torch.ones(2, 3, 1), torch.tensor([True])]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.kernel(ids).to_dense()

    def test_heat_time_validation(self):
        for value in [0, -1, float('nan'), float('inf'), [1, 2]]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.kernel.heat_time = value
        self.kernel.heat_time = 2.5
        self.assertAlmostEqual(self.kernel.heat_time.item(), 2.5, places=12)

    def test_edgeless_graph_interval(self):
        for n in [1, 3]:
            L = scipy.sparse.csr_matrix((n, n))
            with self.subTest(n=n), self.assertRaisesRegex(ValueError, "lambda_min < lambda_max"):
                GraphHeatKernel(L)
            kernel = GraphHeatKernel(L, lambda_max=1)
            torch.testing.assert_close(kernel(torch.arange(n)).to_dense(), torch.eye(n, dtype=DTYPE),
                                       rtol=1e-10, atol=1e-12)

    def test_scale_kernel(self):
        scaled = gpytorch.kernels.ScaleKernel(self.kernel).double()
        scaled.outputscale = 2
        torch.testing.assert_close(scaled(torch.arange(9)).to_dense(), 2 * self.exact,
                                   rtol=1e-10, atol=1e-12)

    def test_state_dict_preserves_interval(self):
        target = GraphHeatKernel(self.L, lambda_max=5, num_nodes=128)
        target.load_state_dict(self.kernel.state_dict())
        self.assertEqual(target.lambda_max, 4)
        torch.testing.assert_close(target(torch.arange(9)).to_dense(), self.exact,
                                   rtol=1e-10, atol=1e-12)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda(self):
        self.kernel.cuda()
        torch.testing.assert_close(self.kernel(torch.arange(9, device='cuda')).to_dense().cpu(),
                                   self.exact, rtol=1e-10, atol=1e-12)
