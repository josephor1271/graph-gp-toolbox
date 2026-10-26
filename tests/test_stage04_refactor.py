"""Compatibility gates for the deliberately bounded Stage 04 extraction."""
from dataclasses import FrozenInstanceError
import json
import subprocess
import sys
import unittest

import gpytorch
import numpy as np
import scipy.sparse
import torch

from graphgp.inference.instrumentation import instrument_operator
from graphgp.inference.solver_settings import IterativeConfig, iterative_context
from graphgp.models.exact_gp import iterative_settings
from graphgp.operators.chebyshev import ChebyshevFunctionLinearOperator
from graphgp.utils import study_domain


def settings_snapshot():
    """Read public settings without relying on their private global storage."""
    names = (
        "cg_tolerance", "eval_cg_tolerance", "max_cg_iterations",
        "num_trace_samples", "max_lanczos_quadrature_iterations",
        "max_preconditioner_size", "max_cholesky_size", "max_eager_kernel_size",
    )
    result = {name: getattr(gpytorch.settings, name).value() for name in names}
    result.update(
        terminate_cg_by_size=gpytorch.settings.terminate_cg_by_size.on(),
        fast_pred_var=gpytorch.settings.fast_pred_var.on(),
        fast_computations={name: getattr(gpytorch.settings.fast_computations, name).on()
                           for name in ("covar_root_decomposition", "log_prob", "solves")},
    )
    return result


class TestStudyDomainCompatibility(unittest.TestCase):
    def test_direct_kernel_import_has_no_model_factory_dependency(self):
        code = (
            "import sys; import graphgp.kernels.graph_time; "
            "assert 'graphgp.models.factory' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    def test_old_public_names_are_identical_reexports(self):
        from graphgp.models import factory
        from graphgp.kernels import graph_time
        for name in ("HEAT_RANGE", "NOISE_RANGE", "AMPLITUDE_RANGE", "validate_study_laplacian"):
            self.assertIs(getattr(factory, name), getattr(study_domain, name))
        self.assertIs(graph_time.LENGTHSCALE_RANGE, study_domain.LENGTHSCALE_RANGE)
        self.assertEqual(study_domain.HEAT_RANGE, (0.05, 4.0))
        self.assertEqual(study_domain.NOISE_RANGE, (0.005, 0.5))
        self.assertEqual(study_domain.AMPLITUDE_RANGE, (0.1, 4.0))
        self.assertEqual(study_domain.LENGTHSCALE_RANGE, (0.08, 0.8))

    def test_domain_preserves_conversion_tolerances_and_errors(self):
        validate = study_domain.validate_study_laplacian
        valid = scipy.sparse.csr_matrix([[0.5, -0.5], [-0.5, 0.5]], dtype=np.float64)
        for matrix in (valid, valid.tocoo(), valid.astype(np.float32),
                       scipy.sparse.csr_matrix([[1, -1], [-1, 1]], dtype=np.int64),
                       scipy.sparse.csr_array(valid)):
            with self.subTest(dtype=matrix.dtype, kind=type(matrix).__name__):
                result = validate(matrix)
                self.assertEqual(result.format, "csr")
                self.assertEqual(result.dtype, np.float64)
                np.testing.assert_array_equal(result.toarray(), matrix.toarray())
        # Existing numeric tolerance is unchanged, including tiny positive off-diagonals.
        tiny = scipy.sparse.csr_matrix([[0, 5e-13], [5e-13, 0]], dtype=np.float64)
        np.testing.assert_array_equal(validate(tiny).toarray(), tiny.toarray())
        with self.assertRaisesRegex(TypeError, "expected a SciPy sparse Laplacian"):
            validate(valid.toarray())
        for malformed in (scipy.sparse.csr_matrix((0, 0)), scipy.sparse.csr_matrix((2, 3))):
            with self.assertRaisesRegex(ValueError, "nonempty and square"):
                validate(malformed)
        with self.assertRaisesRegex(ValueError, "Laplacian must be real"):
            validate(valid.astype(np.complex128))
        invalid = {
            "nonsymmetric": [[0.5, -0.5], [-0.4, 0.4]],
            "positive_off_diagonal": [[0, 0.1], [0.1, 0]],
            "row_sum": [[0.6, -0.5], [-0.5, 0.5]],
            "degree": [[1.1, -1.1], [-1.1, 1.1]],
            "negative_diagonal": [[-0.5, 0.5], [0.5, -0.5]],
            "nonfinite": [[np.nan, -0.5], [-0.5, 0.5]],
        }
        for name, values in invalid.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, "expected symmetric Laplacian with maximum weighted degree <= 1"):
                validate(scipy.sparse.csr_matrix(values, dtype=np.float64))


class TestIterativeConfiguration(unittest.TestCase):
    def test_validation_frozen_state_and_metadata(self):
        config = IterativeConfig()
        self.assertEqual(list(config.to_dict()), [
            "cg_tolerance", "eval_cg_tolerance", "max_cg_iterations", "trace_samples",
            "lanczos_iterations", "max_preconditioner_size", "terminate_cg_by_size",
        ])
        self.assertEqual(IterativeConfig(**config.to_dict()), config)
        self.assertEqual(json.loads(json.dumps(config.to_dict())), config.to_dict())
        numpy_scalars = IterativeConfig(trace_samples=np.int64(8), cg_tolerance=np.float64(1e-4))
        self.assertIs(type(numpy_scalars.trace_samples), int)
        self.assertIs(type(numpy_scalars.cg_tolerance), float)
        json.dumps(numpy_scalars.effective_settings())
        with self.assertRaises(FrozenInstanceError):
            config.trace_samples = 8
        for name in ("cg_tolerance", "eval_cg_tolerance"):
            for value in (0, -1, float("inf"), float("nan"), True, "1e-4"):
                with self.subTest(field=name, value=value), self.assertRaises(ValueError):
                    IterativeConfig(**{name: value})
        for name in ("max_cg_iterations", "trace_samples", "lanczos_iterations"):
            for value in (0, -1, 1.5, True, "8"):
                with self.subTest(field=name, value=value), self.assertRaises(ValueError):
                    IterativeConfig(**{name: value})
        for value in (1, -1, 0.0, False):
            with self.subTest(preconditioner=value), self.assertRaises(ValueError):
                IterativeConfig(max_preconditioner_size=value)
        for value in (0, 1, "false", None):
            with self.subTest(terminate=value), self.assertRaises(ValueError):
                IterativeConfig(terminate_cg_by_size=value)
        with self.assertRaises(TypeError):
            with iterative_context({}):
                pass

    def test_every_setting_nested_normal_and_exception_restoration(self):
        initial = settings_snapshot()
        config = IterativeConfig(cg_tolerance=0.003, eval_cg_tolerance=0.004,
            max_cg_iterations=77, trace_samples=7, lanczos_iterations=9,
            max_preconditioner_size=0, terminate_cg_by_size=True)
        expected = {
            "cg_tolerance": 0.003, "eval_cg_tolerance": 0.004,
            "max_cg_iterations": 77, "num_trace_samples": 7,
            "max_lanczos_quadrature_iterations": 9, "max_preconditioner_size": 0,
            "max_cholesky_size": 0, "max_eager_kernel_size": 0,
            "terminate_cg_by_size": True, "fast_pred_var": False,
            "fast_computations": dict(covar_root_decomposition=True, log_prob=True, solves=True),
        }
        with iterative_context(config):
            self.assertEqual(settings_snapshot(), expected)
            with self.assertRaisesRegex(RuntimeError, "test sentinel"):
                with iterative_context(IterativeConfig()):
                    self.assertFalse(gpytorch.settings.terminate_cg_by_size.on())
                    raise RuntimeError("test sentinel")
            self.assertEqual(settings_snapshot(), expected)
        self.assertEqual(settings_snapshot(), initial)
        with self.assertRaises(RuntimeError):
            with iterative_context(config):
                raise RuntimeError("outer exit")
        self.assertEqual(settings_snapshot(), initial)
        effective = config.effective_settings()
        self.assertEqual(effective["max_cholesky_size"], 0)
        self.assertEqual(effective["max_eager_kernel_size"], 0)
        self.assertEqual(effective["fast_computations"], expected["fast_computations"])
        self.assertFalse(effective["fast_pred_var"])

    def test_legacy_helper_preserves_defaults_and_caller_terminate_override(self):
        initial = settings_snapshot()
        for outer in (False, True):
            with gpytorch.settings.terminate_cg_by_size(outer):
                with iterative_settings():
                    actual = settings_snapshot()
                    self.assertEqual(actual["cg_tolerance"], 1e-5)
                    self.assertEqual(actual["eval_cg_tolerance"], 1e-5)
                    self.assertEqual(actual["max_cg_iterations"], 500)
                    self.assertEqual(actual["num_trace_samples"], 32)
                    self.assertEqual(actual["max_lanczos_quadrature_iterations"], 32)
                    self.assertEqual(actual["max_preconditioner_size"], 0)
                    self.assertEqual(actual["max_cholesky_size"], 0)
                    self.assertEqual(actual["max_eager_kernel_size"], 0)
                    self.assertEqual(actual["terminate_cg_by_size"], outer)
                    self.assertFalse(actual["fast_pred_var"])
                    self.assertTrue(all(actual["fast_computations"].values()))
        with iterative_settings(cg_tolerance=0.0003, trace_samples=5, lanczos_iterations=11):
            actual = settings_snapshot()
            self.assertEqual(actual["cg_tolerance"], 0.0003)
            self.assertEqual(actual["eval_cg_tolerance"], 0.0003)
            self.assertEqual(actual["num_trace_samples"], 5)
            self.assertEqual(actual["max_lanczos_quadrature_iterations"], 11)
        self.assertEqual(settings_snapshot(), initial)


class TestInstrumentationExtraction(unittest.TestCase):
    @staticmethod
    def operator():
        laplacian = scipy.sparse.csr_matrix([[1, -1, 0], [-1, 2, -1], [0, -1, 1]],
                                             dtype=np.float64) / 2
        coefficients = torch.tensor([1.4, -0.3, 0.06, -0.004],
                                    dtype=torch.float64, requires_grad=True)
        return ChebyshevFunctionLinearOperator(
            laplacian, coefficients, lambda_min=0, lambda_max=2), coefficients

    def test_legacy_reexport_counts_widths_and_autograd_are_unchanged(self):
        from graphgp.validation import instrument_operator as legacy
        self.assertIs(legacy, instrument_operator)
        operator, coefficients = self.operator()
        rhs1 = torch.arange(3, dtype=torch.float64)[:, None]
        rhs3 = torch.arange(9, dtype=torch.float64).reshape(3, 3)
        row, col = torch.tensor([0, 1]), torch.tensor([1, 2])
        reference = (operator._matmul(rhs1).sum() + operator._matmul(rhs3).sum()
                     + operator._get_indices(row, col).sum())
        reference_gradient, = torch.autograd.grad(reference, coefficients)
        with instrument_operator(record_widths=True) as stats:
            measured = (operator._matmul(rhs1).sum() + operator._matmul(rhs3).sum()
                        + operator._get_indices(row, col).sum())
            self.assertEqual(stats["operator_matmul_calls"], 4)
            self.assertEqual(stats["estimated_sparse_l_mvps"], 12)
            measured_gradient, = torch.autograd.grad(measured, coefficients)
        torch.testing.assert_close(measured, reference, rtol=0, atol=0)
        torch.testing.assert_close(measured_gradient, reference_gradient, rtol=0, atol=0)
        self.assertGreater(measured_gradient.abs().sum().item(), 0)
        # The two selected-entry custom autograd calls each add one backward MVP.
        self.assertEqual(stats["operator_matmul_calls"], 6)
        self.assertEqual(stats["estimated_sparse_l_mvps"], 18)
        self.assertEqual(stats["rhs_width_counts"], {"1": 5, "3": 1})
        self.assertEqual(stats["get_indices_calls"], 1)
        self.assertEqual(stats["get_indices_entries"], 2)
        self.assertGreater(stats["matmul_seconds"], 0)
        self.assertGreater(stats["get_indices_seconds"], 0)
        with instrument_operator() as stats_without_widths:
            operator._matmul(rhs1)
        self.assertNotIn("rhs_width_counts", stats_without_widths)
        self.assertEqual(stats_without_widths["operator_matmul_calls"], 1)
        self.assertEqual(stats_without_widths["estimated_sparse_l_mvps"], 3)

    def test_method_wrappers_restore_on_exception_and_nested_exit(self):
        original_mm = ChebyshevFunctionLinearOperator._matmul
        original_get = ChebyshevFunctionLinearOperator._get_indices
        with self.assertRaisesRegex(RuntimeError, "test sentinel"):
            with instrument_operator():
                outer_mm = ChebyshevFunctionLinearOperator._matmul
                with instrument_operator():
                    self.assertIsNot(ChebyshevFunctionLinearOperator._matmul, outer_mm)
                self.assertIs(ChebyshevFunctionLinearOperator._matmul, outer_mm)
                raise RuntimeError("test sentinel")
        self.assertIs(ChebyshevFunctionLinearOperator._matmul, original_mm)
        self.assertIs(ChebyshevFunctionLinearOperator._get_indices, original_get)


if __name__ == "__main__":
    unittest.main()
