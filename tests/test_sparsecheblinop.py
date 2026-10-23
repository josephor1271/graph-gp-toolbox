import unittest

import numpy as np
import scipy.sparse as sparse
import torch
from linear_operator.operators import LinearOperator

from graphgp.operators.coefficients import chebyshev_coefficients_on_interval
from graphgp.operators.chebyshev import ChebyshevFunctionLinearOperator, scipy_csr_to_torch_csr
from tests.helpers import DTYPE, dense_heat, path_laplacian, random_tensor, relative_error


class OperatorTests(unittest.TestCase):
    def test_notebook_mvp_baseline(self):
        L = path_laplacian(24)
        coefficients = chebyshev_coefficients_on_interval(
            lambda x: np.exp(-1.1 * x), lambda_min=0, lambda_max=4,
            num_coefficients=33, num_nodes=128, dtype=DTYPE,
        )
        op = ChebyshevFunctionLinearOperator(L, coefficients, lambda_min=0, lambda_max=4)
        rhs = random_tensor(24, 5)
        error = relative_error(op @ rhs, dense_heat(L, 1.1) @ rhs)
        self.assertLess(error, 1e-10)
        print(f"\nBaseline MVP relative error: {error:.3e}")

    def make_operator(self, L=None, coefficients=None, **bounds):
        return ChebyshevFunctionLinearOperator(
            path_laplacian(8) if L is None else L,
            random_tensor(7) if coefficients is None else coefficients,
            **({"lambda_min": 0, "lambda_max": 4} | bounds),
        )

    def polynomial_reference(self, op):
        values, vectors = torch.linalg.eigh(op.L.to_dense())
        # Independent NumPy polynomial evaluator uses the unhalved convention.
        coeffs = op.cheb_coeffs.detach().numpy().copy()
        coeffs[0] /= 2
        polynomial = np.polynomial.chebyshev.chebval(
            (values.numpy() - op.center) / op.radius, coeffs)
        return (vectors * torch.tensor(polynomial)) @ vectors.T

    def test_constructor_validation(self):
        cases = [
            ({"L": sparse.csr_matrix((3, 4))}, ValueError),
            ({"L": sparse.csr_matrix((0, 0))}, ValueError),
            ({"L": torch.eye(8)}, TypeError),
            ({"L": torch.eye(8).to_sparse()}, TypeError),
            ({"coefficients": torch.ones(2, 2)}, ValueError),
            ({"coefficients": torch.empty(0)}, ValueError),
            ({"coefficients": torch.ones(2, dtype=torch.complex128)}, TypeError),
            ({"lambda_min": 4}, ValueError),
            ({"lambda_min": 5}, ValueError),
            ({"lambda_max": float("nan")}, ValueError),
            ({"lambda_max": float("inf")}, ValueError),
        ]
        for kwargs, exception in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(exception):
                self.make_operator(**kwargs)

    def test_canonical_representation(self):
        L = scipy_csr_to_torch_csr(path_laplacian(8), dtype=torch.float32)
        op = self.make_operator(L=L)
        self.assertEqual(op.L.dtype, DTYPE)
        self.assertIs(op.L, op.representation()[0])
        self.assertIs(op.cheb_coeffs, op.representation()[1])
        reconstructed = op.representation_tree()(*op.representation())
        torch.testing.assert_close(reconstructed @ random_tensor(8), op @ random_tensor(8))
        self.assertIs(type(op).matmul, LinearOperator.matmul)

    def test_rhs_shapes_and_transpose(self):
        op = self.make_operator()
        reference = self.polynomial_reference(op)
        for shape in [(8,), (8, 1), (8, 5), (3, 8, 2), (2, 3, 8, 4)]:
            with self.subTest(shape=shape):
                rhs = random_tensor(*shape)
                torch.testing.assert_close(op @ rhs, reference @ rhs, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(op.mT @ rhs, op @ rhs)

    def test_rhs_validation(self):
        with self.assertRaises(ValueError):
            self.make_operator() @ torch.ones(8, dtype=torch.float32)
        with self.assertRaises((ValueError, RuntimeError)):
            self.make_operator() @ random_tensor(7)

    def test_constant_and_linear_polynomials(self):
        for coeffs in [torch.tensor([2.0], dtype=DTYPE), torch.tensor([2.0, 3.0], dtype=DTYPE)]:
            op = self.make_operator(coefficients=coeffs)
            torch.testing.assert_close(op.to_dense(), self.polynomial_reference(op))

    def test_tiny_to_dense(self):
        op = self.make_operator()
        torch.testing.assert_close(op.to_dense(), self.polynomial_reference(op), rtol=1e-12, atol=1e-12)

    def test_indexing_and_diagonal_gradients(self):
        coefficients = random_tensor(7).requires_grad_()
        op = self.make_operator(coefficients=coefficients)
        reference = self.polynomial_reference(op)
        rows, cols = torch.tensor([[1], [3]]), torch.tensor([[0, 5, 7]])
        torch.testing.assert_close(op[rows, cols], reference[rows, cols])
        torch.testing.assert_close(op.diagonal(), reference.diagonal())
        actual, = torch.autograd.grad(op.diagonal().sum(), coefficients)
        # Trace is linear in the coefficients; compare with dense finite differences.
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(torch.autograd.gradcheck(
            lambda c: self.make_operator(coefficients=c).diagonal(), (coefficients,), fast_mode=True))

    def test_zero_graph_with_explicit_positive_interval(self):
        op = self.make_operator(L=sparse.csr_matrix((8, 8)), coefficients=torch.tensor([2.], dtype=DTYPE))
        torch.testing.assert_close(op @ random_tensor(8), random_tensor(8))

    def test_public_matmul_coefficient_and_rhs_gradients(self):
        coeffs = random_tensor(5).requires_grad_()
        for shape in [(8,), (8, 3), (2, 8, 2)]:
            rhs = random_tensor(*shape).requires_grad_()
            with self.subTest(shape=shape):
                self.assertTrue(torch.autograd.gradcheck(
                    lambda c, v: self.make_operator(coefficients=c) @ v,
                    (coeffs, rhs), fast_mode=True,
                ))
