"""Small scalar-PSD and CG diagnostics; no dense graph covariance is built."""

import json
import warnings
from unittest.mock import patch

import torch
from linear_operator.utils.warnings import NumericalWarning

from graphgp.operators.coefficients import (
    chebyshev_coefficients_on_interval_torch,
    evaluate_chebyshev_polynomial_torch,
)
from graphgp.demo import iterative_settings, make_problem
from graphgp.kernels.heat import GraphHeatKernel
from graphgp.operators.chebyshev import ChebyshevFunctionLinearOperator


def heat_polynomial_diagnostics(*, heat_time=1.1, degree=32, lambda_max=4.0, grid_size=4097):
    """Report sampled positivity and error; this is not a PSD certificate."""
    nodes = torch.linspace(0, lambda_max, grid_size, dtype=torch.float64)
    coefficients = chebyshev_coefficients_on_interval_torch(
        lambda x: torch.exp(-heat_time * x), lambda_min=0, lambda_max=lambda_max,
        num_coefficients=degree + 1, num_nodes=max(128, 2 * (degree + 1)), dtype=nodes.dtype)
    approximation = evaluate_chebyshev_polynomial_torch(
        coefficients, nodes, lambda_min=0, lambda_max=lambda_max)
    return dict(heat_time=heat_time, degree=degree, lambda_max=lambda_max,
                minimum=approximation.min().item(),
                max_abs_error=(approximation - torch.exp(-heat_time * nodes)).abs().max().item())


def cg_diagnostics(*, n=40, heat_time=1.1, noise=0.12):
    """Solve a full-graph noisy system and measure true residual and MVP calls."""
    L, _, _, _ = make_problem(n=n)
    kernel = GraphHeatKernel(L, lambda_max=4, heat_time=heat_time, num_nodes=128)
    operator = kernel(torch.arange(n)).evaluate_kernel().add_diagonal(torch.tensor(noise, dtype=torch.float64))
    rhs = torch.randn(n, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(123))
    calls = 0
    original = ChebyshevFunctionLinearOperator._matmul

    def counted(base, rhs):
        nonlocal calls
        calls += 1
        return original(base, rhs)

    with torch.no_grad(), iterative_settings(cg_tolerance=1e-4), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", NumericalWarning)
        with patch.object(ChebyshevFunctionLinearOperator, "_matmul", counted):
            solution = operator.solve(rhs)
        residual = ((operator @ solution - rhs).norm(dim=0) / rhs.norm(dim=0)).max().item()
    return dict(n=n, heat_time=heat_time, noise=noise, mvp_calls=calls,
                max_relative_residual=residual,
                cg_warnings=[str(w.message) for w in caught if issubclass(w.category, NumericalWarning)])


if __name__ == "__main__":
    for time in [0.25, 1.1, 4.0, 10.0]:
        print(json.dumps(heat_polynomial_diagnostics(heat_time=time)))
    print(json.dumps(heat_polynomial_diagnostics(heat_time=4, degree=4)))
    for n in [16, 40, 80]:
        for time in [0.25, 1.1, 4.0]:
            for noise in [0.12, 0.0001]:
                print(json.dumps(cg_diagnostics(n=n, heat_time=time, noise=noise)))
