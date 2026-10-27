"""Scoped observations of the installed CG/SLQ implementation.

These wrappers count calls and returned tridiagonal orders. They do not replace
numerical kernels, alter convergence criteria, sample probes, or infer CG
iteration counts from matrix-product counts.
"""
from contextlib import ExitStack, contextmanager
from functools import wraps
import hashlib
import inspect
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import gpytorch
import linear_operator
from linear_operator import settings, utils as operator_utils
from linear_operator.functions._inv_quad_logdet import InvQuadLogdet
from linear_operator.operators import LinearOperator
from linear_operator.utils.linear_cg import linear_cg as _installed_linear_cg
from linear_operator.utils.stochastic_lq import StochasticLQ
import torch


def effective_solver_settings() -> dict[str, Any]:
    """Read settings without changing them or reseeding the caller's RNG."""
    values = {
        name: getattr(settings, name).value()
        for name in ("cg_tolerance", "max_cg_iterations", "num_trace_samples",
                     "max_lanczos_quadrature_iterations", "max_preconditioner_size",
                     "max_cholesky_size")
    }
    values.update(
        eval_cg_tolerance=gpytorch.settings.eval_cg_tolerance.value(),
        max_eager_kernel_size=gpytorch.settings.max_eager_kernel_size.value(),
        terminate_cg_by_size=settings.terminate_cg_by_size.on(),
        fast_pred_var=gpytorch.settings.fast_pred_var.on(),
        fast_log_prob=settings.fast_computations.log_prob.on(),
        fast_solves=settings.fast_computations.solves.on(),
        fast_covar_root_decomposition=settings.fast_computations.covar_root_decomposition.on(),
    )
    return values


def installed_solver_audit() -> dict[str, Any]:
    """Compact source identities; full numbered local audit is a run artifact."""
    root = Path(linear_operator.__file__).resolve().parent.parent
    sources = {}
    for label, function in (
        ("linear_cg", _installed_linear_cg),
        ("inv_quad_logdet_forward", InvQuadLogdet.forward),
        ("linear_operator_solve", LinearOperator._solve),
        ("stochastic_lq_to_dense", StochasticLQ.to_dense),
    ):
        path = Path(inspect.getfile(function)).resolve()
        lines, first_line = inspect.getsourcelines(function)
        sources[label] = dict(
            source_path=str(path.relative_to(root)), first_line=first_line,
            last_line=first_line + len(lines) - 1,
            signature=str(inspect.signature(function)),
            source_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    signature = inspect.signature(_installed_linear_cg)
    return dict(
        linear_operator_version=linear_operator.__version__,
        gpytorch_version=gpytorch.__version__, sources=sources,
        installed_cg_eps=signature.parameters["eps"].default,
        installed_stop_updating_after=signature.parameters["stop_updating_after"].default,
        iteration_count_instrumented=False,
        limitation="CG calls and matmul workloads are not actual iteration counts. "
                   "Source suggests a small-denominator stagnation mechanism; "
                   "unmodified-solver measurements do not isolate eps as a universal cause.",
    )


@contextmanager
def solver_diagnostics() -> Iterator[dict[str, Any]]:
    """Record actual CG inputs and actual returned/used Lanczos orders.

    Scope this inside the desired inference settings. Settings are merely read;
    ordinary production calls have no wrappers. Patches restore on exceptions
    and nested diagnostic scopes preserve the caller's existing wrappers.
    """
    original_cg = operator_utils.linear_cg
    original_slq = StochasticLQ.to_dense
    signature = inspect.signature(_installed_linear_cg)
    diagnostic: dict[str, Any] = dict(
        cg_calls=0, cg_rhs_shapes=[], cg_tridiag_counts=[],
        cg_actual_tridiagonal_orders=[], cg_call_settings=[],
        slq_calls=0, slq_tridiagonal_orders=[], slq_probe_counts=[],
        effective_settings=effective_solver_settings(),
        actual_cg_iterations=None,
        iteration_count_limitation="Not instrumented; call/workload counts are recorded instead.",
    )

    @wraps(_installed_linear_cg)
    def cg(matmul_closure, rhs, *args, **kwargs):
        arguments = signature.bind_partial(matmul_closure, rhs, *args, **kwargs).arguments
        diagnostic["cg_calls"] += 1
        diagnostic["cg_rhs_shapes"].append(list(rhs.shape))
        diagnostic["cg_tridiag_counts"].append(int(arguments.get("n_tridiag", 0)))
        diagnostic["cg_call_settings"].append(dict(
            tolerance=arguments.get("tolerance") if arguments.get("tolerance") is not None
                else settings.cg_tolerance.value(),
            max_iter=arguments.get("max_iter") if arguments.get("max_iter") is not None
                else settings.max_cg_iterations.value(),
            max_tridiag_iter=arguments.get("max_tridiag_iter")
                if arguments.get("max_tridiag_iter") is not None
                else settings.max_lanczos_quadrature_iterations.value(),
            preconditioner_supplied=arguments.get("preconditioner") is not None,
            eps=arguments.get("eps", signature.parameters["eps"].default),
            stop_updating_after=arguments.get(
                "stop_updating_after", signature.parameters["stop_updating_after"].default),
        ))
        result = original_cg(matmul_closure, rhs, *args, **kwargs)
        if isinstance(result, tuple):
            diagnostic["cg_actual_tridiagonal_orders"].append(int(result[1].shape[-1]))
        return result

    @wraps(original_slq)
    def slq(self, matrix_shape, eigenvalues, eigenvectors, funcs):
        diagnostic["slq_calls"] += 1
        diagnostic["slq_tridiagonal_orders"].append(int(eigenvalues.shape[-1]))
        diagnostic["slq_probe_counts"].append(int(eigenvalues.shape[0]))
        return original_slq(self, matrix_shape, eigenvalues, eigenvectors, funcs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(operator_utils, "linear_cg", cg))
        stack.enter_context(patch.object(StochasticLQ, "to_dense", slq))
        yield diagnostic


def relative_residual(A, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Fresh per-column ``||A x-b|| / max(||b||, 1e-15)``.

    ``A`` is the selected *noisy* covariance supplied by the caller. This
    function invokes a fresh matvec and never substitutes a raw graph factor or
    a solver's recursive residual estimate. Vectors return a scalar tensor;
    matrices return one value per RHS. No full covariance is constructed here.
    """
    if (not torch.is_tensor(x) or not torch.is_tensor(b)
            or x.shape != b.shape or b.ndim not in (1, 2)
            or A.shape != torch.Size((b.shape[0], b.shape[0]))
            or x.dtype != torch.float64 or b.dtype != torch.float64
            or x.device != b.device or not torch.isfinite(x).all()
            or not torch.isfinite(b).all()):
        raise ValueError("A must be square and x,b finite matching float64 vectors or matrices")
    with torch.no_grad():
        difference = A.matmul(x) - b
        if not torch.isfinite(difference).all():
            raise ArithmeticError("nonfinite independently computed residual")
        return torch.linalg.vector_norm(difference, dim=0) / torch.linalg.vector_norm(
            b, dim=0).clamp_min(1e-15)
