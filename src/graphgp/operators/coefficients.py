"""Chebyshev coefficient helpers for ``sparsecheblinop.py``.

Both paths use ``c[0]/2 + sum(c[k] T_k(x))``. The NumPy/SciPy path is for
offline reference calculations; the Torch path preserves parameter gradients.
"""

from __future__ import annotations

import math
import operator
from typing import Any, Callable

import numpy as np
import scipy as sp
import torch


ScalarFunction = Callable[[Any], Any]
DEFAULT_NUM_CHEBYSHEV_COEFFICIENTS = 64


def choose_heat_chebyshev_degree(max_heat_time, lambda_max, tolerance=1e-6,
                                candidate_degrees=(8, 12, 16, 24, 32, 48, 64, 96, 128)):
    """Choose a fixed degree by a sampled time/spectrum error heuristic.

    Checks 33 heat times in [0,max_heat_time] and 4097 spectral points in
    [0,lambda_max], in float64. This is not a continuous error or PSD
    certificate. Keep the returned degree fixed during training and reject or
    flag trajectories leaving the declared time range. Returns degree plus
    diagnostic metadata so the selection is auditable.
    """
    if not (math.isfinite(max_heat_time) and max_heat_time > 0
            and math.isfinite(lambda_max) and lambda_max > 0
            and math.isfinite(tolerance) and tolerance > 0):
        raise ValueError("time, upper bound, and tolerance must be finite and positive")
    candidates = tuple(candidate_degrees)
    if not candidates or any(operator.index(d) < 0 for d in candidates) or list(candidates) != sorted(set(candidates)):
        raise ValueError("candidate degrees must be nonnegative, increasing integers")
    points = torch.linspace(0, lambda_max, 4097, dtype=torch.float64)
    times = torch.linspace(0, max_heat_time, 33, dtype=torch.float64)
    for degree in candidates:
        nodes, basis = chebyshev_basis_on_interval_torch(lambda_min=0, lambda_max=lambda_max,
            num_coefficients=degree+1, num_nodes=max(128, 2*(degree+1)), dtype=torch.float64)
        worst, minimum = 0.0, float("inf")
        for time in times:
            coeffs = chebyshev_coefficients_from_samples_torch(torch.exp(-time*nodes), basis)
            values = evaluate_chebyshev_polynomial_torch(coeffs, points, lambda_min=0, lambda_max=lambda_max)
            worst = max(worst, (values-torch.exp(-time*points)).abs().max().item())
            minimum = min(minimum, values.min().item())
        if worst <= tolerance:
            return dict(degree=degree, max_heat_time=max_heat_time, lambda_max=lambda_max,
                        tolerance=tolerance, sampled_max_error=worst, sampled_minimum=minimum,
                        spectral_points=4097, time_points=33, certificate=False)
    raise ValueError("no candidate degree meets the sampled error tolerance")


def chebyshev_coefficients_for_laplacian(
    L: Any,
    function: ScalarFunction,
    *,
    num_coefficients: int = DEFAULT_NUM_CHEBYSHEV_COEFFICIENTS,
    num_nodes: int | None = None,
    lambda_min: float | None = None,
    lambda_max: float | None = None,
    eig_tol: float = 1e-6,
    eig_maxiter: int | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return Chebyshev coefficients for ``function`` over the spectrum of ``L``.
    ``L`` may be either a SciPy sparse matrix/array or a PyTorch sparse CSR
    tensor. The Laplacian lower bound defaults to the analytical zero; only a
    missing upper bound is estimated. This is an offline, nondifferentiable path.
    """

    dtype, device = _resolve_output_dtype_device(L, dtype=dtype, device=device)
    lambda_min, lambda_max = laplacian_eigenvalue_bounds(
        L, lambda_min=lambda_min, lambda_max=lambda_max,
        tol=eig_tol, maxiter=eig_maxiter,
    )

    return chebyshev_coefficients_on_interval(
        function,
        lambda_min=float(lambda_min),
        lambda_max=float(lambda_max),
        num_coefficients=num_coefficients,
        num_nodes=num_nodes,
        dtype=dtype,
        device=device,
    )


def chebyshev_coefficients(
    L: Any,
    function: ScalarFunction,
    **kwargs: Any,
) -> torch.Tensor:
    """Convenience wrapper for ``chebyshev_coefficients_for_laplacian``."""

    return chebyshev_coefficients_for_laplacian(L, function, **kwargs)


def chebyshev_coefficients_on_interval(
    function: ScalarFunction,
    *,
    lambda_min: float = -1.0,
    lambda_max: float = 1.0,
    num_coefficients: int = DEFAULT_NUM_CHEBYSHEV_COEFFICIENTS,
    num_nodes: int | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute Chebyshev coefficients for a scalar function on an interval.

    The coefficients use the convention expected by
    ``ChebyshevFunctionLinearOperator``: evaluation starts with
    ``0.5 * coeffs[0] * T_0``.
    """

    num_coefficients = _validate_num_coefficients(num_coefficients)
    num_nodes = _validate_num_nodes(num_nodes, num_coefficients)
    lambda_min = float(lambda_min)
    lambda_max = float(lambda_max)

    if not (math.isfinite(lambda_min) and math.isfinite(lambda_max)
            and lambda_max >= lambda_min):
        raise ValueError("bounds must be finite with lambda_max >= lambda_min")

    center = 0.5 * (lambda_min + lambda_max)
    radius = 0.5 * (lambda_max - lambda_min)

    if radius == 0.0:
        value = _evaluate_function(function, np.array([center], dtype=np.float64))[0]
        coeffs = np.zeros(num_coefficients, dtype=np.float64)
        coeffs[0] = 2.0 * value
        return torch.as_tensor(coeffs, dtype=dtype, device=device)

    node_ids = np.arange(num_nodes, dtype=np.float64)
    theta = (node_ids + 0.5) * math.pi / num_nodes
    spectral_nodes = center + radius * np.cos(theta)
    values = _evaluate_function(function, spectral_nodes)

    coeffs = sp.fft.dct(values, type=2, norm=None)[:num_coefficients] / num_nodes
    return torch.as_tensor(coeffs, dtype=dtype, device=device)


def sparse_symmetric_eigenvalue_bounds(
    L: Any,
    *,
    tol: float = 1e-6,
    maxiter: int | None = None,
) -> tuple[float, float]:
    """Estimate extreme eigenvalues of a generic sparse symmetric matrix.

    These are ARPACK estimates, not certified enclosing bounds. In particular,
    ARPACK can miss a Laplacian's zero mode; use laplacian_eigenvalue_bounds
    for graph Laplacians and supply known enclosing bounds when available.
    """

    matrix = as_scipy_csr(L, dtype=np.float64)
    n = matrix.shape[0]
    if n == 0:
        raise ValueError("L must be nonempty")
    if n == 1:
        value = float(matrix[0, 0])
        return value, value

    eigsh_kwargs = {
        "k": 1,
        "return_eigenvectors": False,
        "tol": float(tol),
        "maxiter": maxiter,
    }

    # Generic symmetric-matrix estimates; Laplacians use the helper below.
    lambda_min = float(sp.sparse.linalg.eigsh(matrix, which="SA", **eigsh_kwargs)[0])
    lambda_max = float(sp.sparse.linalg.eigsh(matrix, which="LA", **eigsh_kwargs)[0])

    if lambda_max < lambda_min:
        lambda_min, lambda_max = lambda_max, lambda_min

    return lambda_min, lambda_max


def laplacian_eigenvalue_bounds(
    L: Any, *, lambda_min: float | None = None, lambda_max: float | None = None,
    tol: float = 1e-6, maxiter: int | None = None,
) -> tuple[float, float]:
    """Use zero as the Laplacian minimum and estimate a missing maximum once.

    The estimated maximum has a small relative safety margin (10*tol), but
    is not a certified enclosure. Known bounds, e.g. 2 for a normalized
    Laplacian or twice the maximum weighted degree, can be supplied instead.
    """
    low = 0.0 if lambda_min is None else float(lambda_min)
    if lambda_max is None:
        matrix = as_scipy_csr(L, dtype=np.float64)
        if matrix.shape[0] == 0:
            raise ValueError("L must be nonempty")
        if matrix.nnz == 0 or not np.any(matrix.data):
            high = 0.0
        elif matrix.shape[0] == 1:
            high = float(matrix[0, 0])
        else:
            high = float(sp.sparse.linalg.eigsh(
                matrix, k=1, which="LA", return_eigenvectors=False,
                tol=tol, maxiter=maxiter,
            )[0])
        high = max(0.0, high) * (1 + max(10 * tol, 1e-12))
    else:
        high = float(lambda_max)
    if not (math.isfinite(low) and math.isfinite(high) and low <= high):
        raise ValueError("bounds must be finite with lambda_min <= lambda_max")
    return low, high


def chebyshev_basis_on_interval_torch(
    *, lambda_min: float = -1.0, lambda_max: float = 1.0,
    num_coefficients: int = DEFAULT_NUM_CHEBYSHEV_COEFFICIENTS,
    num_nodes: int | None = None, dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute spectral nodes and cosine basis (coefficients by nodes).

    Fixed interval bounds must have positive width. Store these tensors as
    module buffers when reusing them throughout optimization.
    """
    num_coefficients = _validate_num_coefficients(num_coefficients)
    num_nodes = _validate_num_nodes(num_nodes, num_coefficients)
    low, high = float(lambda_min), float(lambda_max)
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError("bounds must be finite with lambda_min < lambda_max")
    dtype = torch.get_default_dtype() if dtype is None else dtype
    if not dtype.is_floating_point:
        raise TypeError("dtype must be floating point")
    theta = (torch.arange(num_nodes, dtype=dtype, device=device) + 0.5) * (math.pi / num_nodes)
    nodes = (low / 2 + high / 2) + (high / 2 - low / 2) * theta.cos()
    orders = torch.arange(num_coefficients, dtype=dtype, device=device)
    return nodes, torch.cos(orders[:, None] * theta)


def chebyshev_coefficients_from_samples_torch(values, cosine_basis):
    """Differentiable DCT sum using a precomputed cosine basis."""
    if values.ndim != 1 or cosine_basis.ndim != 2 or values.numel() != cosine_basis.shape[1]:
        raise ValueError("values must match the cosine basis sample dimension")
    return (2.0 / values.numel()) * (cosine_basis @ values)


def chebyshev_coefficients_on_interval_torch(function, **kwargs):
    """Evaluate a Torch callable and compute coefficients without detaching.

    Accepts the same fixed interval/basis arguments as
    chebyshev_basis_on_interval_torch. Specify dtype/device to match the
    callable's trainable parameters. For repeated use, precompute the basis
    and call chebyshev_coefficients_from_samples_torch instead.
    """
    nodes, basis = chebyshev_basis_on_interval_torch(**kwargs)
    values = torch.as_tensor(function(nodes), device=nodes.device)
    if values.is_complex():
        raise ValueError("function must be real-valued")
    values = torch.broadcast_to(values.to(dtype=nodes.dtype), nodes.shape)
    return chebyshev_coefficients_from_samples_torch(values, basis)


def evaluate_chebyshev_polynomial_torch(coefficients, points, *, lambda_min, lambda_max):
    """Evaluate the scalar polynomial for diagnostics, preserving gradients."""
    if coefficients.ndim != 1 or coefficients.numel() == 0:
        raise ValueError("coefficients must be nonempty and 1D")
    low, high = float(lambda_min), float(lambda_max)
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError("bounds must be finite with lambda_min < lambda_max")
    x = (points - (low / 2 + high / 2)) / (high / 2 - low / 2)
    previous = torch.ones_like(x)
    result = coefficients[0] * previous / 2
    if coefficients.numel() > 1:
        current = x
        result = result + coefficients[1] * current
        for coefficient in coefficients[2:]:
            following = 2 * x * current - previous
            result = result + coefficient * following
            previous, current = current, following
    return result


def as_scipy_csr(L: Any, *, dtype: Any | None = None) -> Any:
    """Convert an accepted ``sparsecheblinop.py`` input to SciPy CSR."""

    if sp.sparse.issparse(L):
        matrix = L.tocsr()
    elif torch.is_tensor(L):
        if not L.is_sparse_csr:
            raise TypeError("L must be a PyTorch sparse CSR tensor or SciPy sparse matrix")
        L_cpu = L.detach().cpu()
        matrix = sp.sparse.csr_array(
            (
                L_cpu.values().numpy(),
                L_cpu.col_indices().numpy(),
                L_cpu.crow_indices().numpy(),
            ),
            shape=tuple(L_cpu.shape),
        )
    else:
        raise TypeError("L must be a PyTorch sparse CSR tensor or SciPy sparse matrix")

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("L must be a square matrix")

    if dtype is not None:
        matrix = matrix.astype(dtype, copy=False)

    if not matrix.has_canonical_format:
        matrix = matrix.copy()
        matrix.sum_duplicates()
        matrix.sort_indices()

    return matrix


def _resolve_output_dtype_device(
    L: Any,
    *,
    dtype: torch.dtype | None,
    device: torch.device | str | None,
) -> tuple[torch.dtype | None, torch.device | str | None]:
    if torch.is_tensor(L):
        if dtype is None and L.is_floating_point():
            dtype = L.dtype
        if device is None:
            device = L.device

    return dtype, device


def _validate_num_coefficients(num_coefficients: int) -> int:
    num_coefficients = operator.index(num_coefficients)
    if num_coefficients < 1:
        raise ValueError("num_coefficients must be at least 1")
    return num_coefficients


def _validate_num_nodes(num_nodes: int | None, num_coefficients: int) -> int:
    if num_nodes is None:
        return max(num_coefficients, DEFAULT_NUM_CHEBYSHEV_COEFFICIENTS)

    num_nodes = operator.index(num_nodes)
    if num_nodes < num_coefficients:
        raise ValueError("num_nodes must be at least num_coefficients")
    return num_nodes


def _evaluate_function(function: ScalarFunction, x: np.ndarray) -> np.ndarray:
    """Evaluate a loose NumPy, torch, or scalar-only callable on sample nodes."""

    vector_error = None
    torch_error = None

    try:
        values = function(x)
    except Exception as exc:
        vector_error = exc
    else:
        return _coerce_function_values(values, x.shape)

    try:
        values = function(torch.as_tensor(x, dtype=torch.float64))
    except Exception as exc:
        torch_error = exc
    else:
        return _coerce_function_values(values, x.shape)

    try:
        values = np.vectorize(function)(x)
    except Exception as exc:
        raise TypeError(
            "function must accept a NumPy array, a torch tensor, or scalar inputs"
        ) from (vector_error or torch_error or exc)

    return _coerce_function_values(values, x.shape)


def _coerce_function_values(values: Any, expected_shape: tuple[int, ...]) -> np.ndarray:
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()

    values = np.asarray(values)
    if values.shape == ():
        values = np.full(expected_shape, values.item(), dtype=values.dtype)
    elif values.shape != expected_shape:
        try:
            values = np.broadcast_to(values, expected_shape)
        except ValueError as exc:
            raise ValueError(
                "function must return values broadcastable to the Chebyshev nodes "
                f"(expected {expected_shape}, got {values.shape})"
            ) from exc

    values = np.real_if_close(values)
    if np.iscomplexobj(values):
        raise ValueError("function values must be real-valued")

    return np.asarray(values, dtype=np.float64)
