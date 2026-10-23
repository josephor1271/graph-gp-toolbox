"""Shared small-experiment data, inference controls, and instrumentation.

Exhaustive trace probes and dense scalar/covariance oracles here are for small
validation only, never for the scaling benchmark's matrix-free path.
"""
from contextlib import ExitStack, contextmanager
import csv
import json
import platform
import sys
import time
import warnings
from pathlib import Path
from unittest.mock import patch

import gpytorch
import linear_operator
from linear_operator.operators import LinearOperator
from linear_operator.utils.warnings import NumericalWarning
import networkx as nx
import numpy as np
import scipy
import scipy.sparse
import torch

from graphgp.operators.coefficients import (
    chebyshev_coefficients_from_samples_torch,
    chebyshev_basis_on_interval_torch,
    evaluate_chebyshev_polynomial_torch,
)
from graphgp.demo import GraphGP, iterative_settings
from graphgp.kernels.heat import GraphHeatKernel
from graphgp.operators.chebyshev import ChebyshevFunctionLinearOperator

VARIANTS = {
    "A": "Dense exact / direct", "A2": "Dense exact / iterative",
    "B": "Dense Chebyshev / direct", "B2": "Dense Chebyshev / iterative",
    "C": "Sparse Chebyshev / iterative",
}


def graph_problem(family="path", n=40, seed=123):
    if family == "path":
        graph = nx.path_graph(n)
    elif family == "grid":
        side = int(n ** 0.5)
        if side * side != n:
            raise ValueError("grid n must be a perfect square")
        graph = nx.grid_2d_graph(side, side)
    elif family == "irregular":
        graph = nx.barabasi_albert_graph(n, 2, seed=seed)
    elif family == "disconnected":
        graph = nx.disjoint_union(nx.path_graph(n // 2), nx.path_graph(n - n // 2))
    else:
        raise ValueError(f"unknown graph family: {family}")
    L = nx.laplacian_matrix(graph).astype(np.float64).tocsr()
    upper = 2.0 * float(L.diagonal().max())
    rng = np.random.default_rng(seed)
    if family == "grid":
        x, y = np.meshgrid(np.linspace(0, np.pi, side), np.linspace(0, np.pi, side))
        signal = (np.sin(x) + 0.5 * np.cos(y)).ravel()
    elif family == "irregular":
        signal = rng.standard_normal(n)
        for _ in range(8):
            signal -= (L @ signal) / upper
        signal /= signal.std()
    else:
        signal = np.sin(np.linspace(0, 2 * np.pi, n))
    targets = torch.tensor(signal + 0.04 * rng.standard_normal(n), dtype=torch.float64)
    permutation = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    train = permutation[:n // 2].sort().values
    test = permutation[n // 2:].sort().values
    return dict(L=L, train_ids=train, test_ids=test, targets=targets,
                graph_family=family, n=n, nnz=L.nnz, lambda_min=0.0,
                lambda_max=upper, seed=seed)


def make_model(problem, variant, *, degree=32, heat_time=1.1, noise=0.12):
    from graphgp.kernels.dense import DenseExactGraphHeatKernel, DenseChebyshevGraphHeatKernel
    if variant in ("A", "A2"):
        kernel = DenseExactGraphHeatKernel(problem["L"], heat_time=heat_time)
    else:
        cls = DenseChebyshevGraphHeatKernel if variant in ("B", "B2") else GraphHeatKernel
        kernel = cls(problem["L"], lambda_min=0, lambda_max=problem["lambda_max"],
                     num_coefficients=degree + 1, num_nodes=max(128, 2 * (degree + 1)),
                     heat_time=heat_time)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
    likelihood.noise = noise
    model = GraphGP(problem["train_ids"], problem["targets"][problem["train_ids"]], likelihood, kernel)
    return model, likelihood


@contextmanager
def exhaustive_trace_probes():
    """Exact training-space trace average, with O(n_train^2) probe storage."""
    def probes(operator):
        n = operator.shape[-1]
        return (torch.eye(n, dtype=operator.dtype, device=operator.device),
                torch.full((1, n), n ** 0.5, dtype=operator.dtype, device=operator.device))
    with patch.object(LinearOperator, "_probe_vectors_and_norms", probes):
        yield


@contextmanager
def inference_context(variant, n_train, *, exhaustive=True, seed=123, cg_tolerance=1e-4):
    with ExitStack() as stack:
        stack.enter_context(torch.random.fork_rng(devices=[]))
        torch.manual_seed(seed)
        # Disable eager posterior block conversion for consistent phase accounting.
        stack.enter_context(gpytorch.settings.max_eager_kernel_size(0))
        if variant in ("A", "B"):
            stack.enter_context(gpytorch.settings.max_cholesky_size(10 ** 9))
            stack.enter_context(gpytorch.settings.fast_computations(
                covar_root_decomposition=False, log_prob=False, solves=False))
        else:
            stack.enter_context(iterative_settings(cg_tolerance=cg_tolerance,
                lanczos_iterations=n_train if exhaustive else 32, trace_samples=32))
            stack.enter_context(patch.object(LinearOperator, "_cholesky",
                side_effect=AssertionError("iterative variant attempted Cholesky")))
            if exhaustive:
                stack.enter_context(exhaustive_trace_probes())
        if variant == "C":
            stack.enter_context(patch.object(ChebyshevFunctionLinearOperator, "to_dense",
                side_effect=AssertionError("sparse variant materialized full base covariance")))
        yield


@contextmanager
def numerical_warnings():
    """Record numerical warnings while still emitting them to stderr."""
    messages = []
    with warnings.catch_warnings():
        warnings.simplefilter("always", NumericalWarning)
        original = warnings.showwarning
        def show(message, category, filename, lineno, file=None, line=None):
            if issubclass(category, NumericalWarning):
                messages.append(str(message))
            original(message, category, filename, lineno, file=file, line=line)
        warnings.showwarning = show
        yield messages


@contextmanager
def instrument_operator(*, record_widths=False):
    stats = dict(operator_matmul_calls=0, estimated_sparse_l_mvps=0,
                 get_indices_calls=0, get_indices_entries=0,
                 matmul_seconds=0.0, get_indices_seconds=0.0)
    if record_widths:
        stats['rhs_width_counts'] = {}
    original_mm = ChebyshevFunctionLinearOperator._matmul
    original_get = ChebyshevFunctionLinearOperator._get_indices
    def mm(op, rhs):
        start = time.perf_counter()
        stats["operator_matmul_calls"] += 1
        if record_widths:
            width = str(rhs.shape[-1] if rhs.ndim > 1 else 1)
            stats['rhs_width_counts'][width] = stats['rhs_width_counts'].get(width, 0) + 1
        stats["estimated_sparse_l_mvps"] += op.cheb_coeffs.numel() - 1
        try:
            return original_mm(op, rhs)
        finally:
            stats["matmul_seconds"] += time.perf_counter() - start
    def get(op, row, col, *batch):
        start = time.perf_counter()
        stats["get_indices_calls"] += 1
        stats["get_indices_entries"] += torch.broadcast_tensors(row, col)[0].numel()
        try:
            return original_get(op, row, col, *batch)
        finally:
            stats["get_indices_seconds"] += time.perf_counter() - start
    with patch.object(ChebyshevFunctionLinearOperator, "_matmul", mm), \
         patch.object(ChebyshevFunctionLinearOperator, "_get_indices", get):
        yield stats


def polynomial_diagnostics(problem, degree, heat_time, *, eigenvalues=None):
    nodes, basis = chebyshev_basis_on_interval_torch(lambda_min=0, lambda_max=problem["lambda_max"],
        num_coefficients=degree + 1, num_nodes=max(128, 2*(degree + 1)), dtype=torch.float64)
    coeffs = chebyshev_coefficients_from_samples_torch(torch.exp(-heat_time * nodes), basis)
    points = torch.linspace(0, problem["lambda_max"], 4097, dtype=torch.float64)
    approx = evaluate_chebyshev_polynomial_torch(coeffs, points, lambda_min=0, lambda_max=problem["lambda_max"])
    result = dict(scalar_max_abs_error=(approx-torch.exp(-heat_time*points)).abs().max().item(),
                  scalar_minimum=approx.min().item())
    if eigenvalues is not None:
        result["eigenvalue_polynomial_minimum"] = evaluate_chebyshev_polynomial_torch(
            coeffs, eigenvalues, lambda_min=0, lambda_max=problem["lambda_max"]).min().item()
    return result


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def environment_metadata():
    return dict(python=sys.version, torch=torch.__version__, scipy=scipy.__version__, numpy=np.__version__, networkx=nx.__version__,
                gpytorch=gpytorch.__version__, linear_operator=linear_operator.__version__,
                platform=platform.platform(), torch_threads=torch.get_num_threads(),
                dtype="float64", timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
