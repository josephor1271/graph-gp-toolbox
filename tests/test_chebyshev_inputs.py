import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sparse
import torch

from graphgp.operators.coefficients import (
    sparse_symmetric_eigenvalue_bounds, laplacian_eigenvalue_bounds,
    chebyshev_coefficients_for_laplacian, chebyshev_coefficients_on_interval,
    chebyshev_coefficients_on_interval_torch, chebyshev_basis_on_interval_torch,
    chebyshev_coefficients_from_samples_torch,
)
from graphgp.operators.chebyshev import ChebyshevFunctionLinearOperator
from graphgp.validation import polynomial_diagnostics
from tests.helpers import DTYPE, dense_heat, path_laplacian, random_tensor


class CoefficientTests(unittest.TestCase):
    def test_symmetric_bounds_baseline(self):
        L = path_laplacian(24)
        exact = np.linalg.eigvalsh(L.toarray())
        low, high = sparse_symmetric_eigenvalue_bounds(L, tol=1e-12)
        self.assertLess(abs(exact[0]), 1e-12)
        self.assertAlmostEqual(high, exact[-1], places=10)
        # ARPACK's zero-eigenvalue behavior is observed, not assumed reliable.
        self.assertGreaterEqual(low, -1e-10)
        self.assertLessEqual(low, exact[1] + 1e-10)
        print(f"\nBaseline spectral minimum: dense={exact[0]:.3e}, eigsh={low:.3e}")

    def test_laplacian_zero_bound_and_single_estimation(self):
        import scipy.sparse.linalg
        for L in [path_laplacian(24), sparse.block_diag([path_laplacian(4), path_laplacian(6)])]:
            with patch("scipy.sparse.linalg.eigsh", wraps=scipy.sparse.linalg.eigsh) as estimate:
                low, high = laplacian_eigenvalue_bounds(L)
                self.assertEqual(low, 0.0)
                self.assertGreaterEqual(high, np.linalg.eigvalsh(L.toarray())[-1])
                self.assertEqual(estimate.call_count, 1)
                self.assertEqual(estimate.call_args.kwargs["which"], "LA")
        with patch("scipy.sparse.linalg.eigsh", side_effect=AssertionError("unexpected estimation")):
            c = chebyshev_coefficients_for_laplacian(L, np.exp, lambda_max=4)
            reference = chebyshev_coefficients_on_interval(np.exp, lambda_min=0, lambda_max=4)
            torch.testing.assert_close(c, reference)

    def test_generic_symmetric_bounds(self):
        low, high = sparse_symmetric_eigenvalue_bounds(sparse.diags([-3., -1., 2., 4.]))
        self.assertAlmostEqual(low, -3, places=10)
        self.assertAlmostEqual(high, 4, places=10)

    def test_torch_matches_scipy(self):
        functions = [(lambda x: np.exp(-x), lambda x: torch.exp(-x)),
                     (lambda x: np.exp(-2*x), lambda x: torch.exp(-2*x)),
                     (lambda x: 1/(x+1), lambda x: 1/(x+1))]
        for reference_fn, torch_fn in functions:
            kwargs = dict(lambda_min=0, lambda_max=4, num_coefficients=33, num_nodes=128, dtype=DTYPE)
            expected = chebyshev_coefficients_on_interval(reference_fn, **kwargs)
            actual = chebyshev_coefficients_on_interval_torch(torch_fn, **kwargs)
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)

    def test_constant_function_convention(self):
        coefficients = chebyshev_coefficients_on_interval_torch(lambda x: 3., dtype=DTYPE)
        self.assertAlmostEqual(coefficients[0].item(), 6., places=12)
        self.assertLess(coefficients[1:].abs().max().item(), 1e-12)

    def test_heat_time_gradient_matches_exact_derivative(self):
        L = path_laplacian(24)
        t = torch.tensor(1.1, dtype=DTYPE, requires_grad=True)
        nodes, basis = chebyshev_basis_on_interval_torch(
            lambda_min=0, lambda_max=4, num_coefficients=33, num_nodes=128, dtype=DTYPE)
        coefficients = chebyshev_coefficients_from_samples_torch(torch.exp(-t * nodes), basis)
        op = ChebyshevFunctionLinearOperator(L, coefficients, lambda_min=0, lambda_max=4)
        u, v = random_tensor(24), random_tensor(24, seed=456)
        actual, = torch.autograd.grad(u @ (op @ v), t)
        exact = -u @ torch.tensor(L.toarray(), dtype=DTYPE) @ dense_heat(L, 1.1) @ v
        self.assertTrue(torch.isfinite(actual))
        self.assertGreater(actual.abs().item(), 1e-8)
        error = ((actual - exact).abs() / exact.abs()).item()
        self.assertLess(error, 1e-8)
        print(f"\nHeat-time gradient relative error: {error:.3e}")

    def test_coefficient_validation(self):
        for kwargs in [dict(num_coefficients=0), dict(num_coefficients=10, num_nodes=9),
                       dict(lambda_min=1, lambda_max=1), dict(lambda_max=float('nan'))]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                chebyshev_basis_on_interval_torch(**kwargs)
        with self.assertRaises(TypeError):
            chebyshev_basis_on_interval_torch(num_coefficients=1.5)
        with self.assertRaises(TypeError):
            chebyshev_basis_on_interval_torch(dtype=torch.int64)

    def test_heat_polynomial_safe_regime(self):
        for time in [0.25, 1.1, 4.0]:
            result = polynomial_diagnostics({'lambda_max': 4.0}, degree=32, heat_time=time)
            with self.subTest(time=time):
                self.assertGreater(result["scalar_minimum"], 0)
                self.assertLess(result["scalar_max_abs_error"], 1e-12)

    def test_diagnostic_exposes_negative_truncation(self):
        coarse = polynomial_diagnostics({'lambda_max': 4.0}, degree=4, heat_time=4)
        resolved = polynomial_diagnostics({'lambda_max': 4.0}, degree=32, heat_time=4)
        self.assertLess(coarse["scalar_minimum"], -1e-3)
        self.assertGreater(resolved["scalar_minimum"], 0)
        self.assertLess(resolved["scalar_max_abs_error"], coarse["scalar_max_abs_error"])
