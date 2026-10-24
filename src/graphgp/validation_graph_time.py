"""Tiny deterministic graph × time numerical gates and solver diagnostics.

The exhaustive training-space trace probes cost quadratic storage and are a
small-problem validation tool. Production inference remains GPyTorch's installed
CG/SLQ; this module does not modify solver coefficients or convergence criteria.
"""
from contextlib import ExitStack, contextmanager
from typing import Any, Iterator
from unittest.mock import patch
import warnings

import gpytorch
from linear_operator import utils as operator_utils
from linear_operator.operators import KroneckerProductLinearOperator, LinearOperator
from linear_operator.utils.stochastic_lq import StochasticLQ
import numpy as np
import torch

from graphgp.analysis.graph_time_oracle import dense_prior, exact_nll, posterior_marginals
from graphgp.inference.predict import predict_marginals
from graphgp.models.exact_gp import iterative_settings
from graphgp.operators.chebyshev import ChebyshevFunctionLinearOperator
from graphgp.utils.graphs import make_laplacian
from graphgp.validation import exhaustive_trace_probes, instrument_operator


@contextmanager
def sparse_graph_time_guards() -> Iterator[None]:
    """Reject known graph/full-prior densification and direct-solve paths.

    Selected cross-covariance blocks and the dense T×T temporal factor are
    allowed. Batched eigendecomposition of Lanczos tridiagonal matrices is also
    allowed. These method-level guards cover the exercised CPU library paths;
    they are not a proof against every possible future dense allocation.
    """
    original_eigh = torch.linalg.eigh

    def eigh(matrix, *args, **kwargs):
        if matrix.ndim == 2:
            raise AssertionError("sparse graph-time path attempted unbatched dense eigendecomposition")
        return original_eigh(matrix, *args, **kwargs)

    with ExitStack() as stack:
        for cls, name, label in (
            (ChebyshevFunctionLinearOperator, "to_dense", "dense graph factor"),
            (KroneckerProductLinearOperator, "to_dense", "dense full-grid covariance"),
            (LinearOperator, "_cholesky", "operator Cholesky"),
            (torch, "kron", "explicit full Kronecker covariance"),
            (torch.linalg, "cholesky", "dense Cholesky"),
            (torch.linalg, "cholesky_ex", "dense Cholesky"),
            (torch.linalg, "eigvalsh", "dense graph eigenvalues"),
        ):
            stack.enter_context(patch.object(cls, name, side_effect=AssertionError(label)))
        stack.enter_context(patch.object(torch.linalg, "eigh", eigh))
        yield


@contextmanager
def graph_time_inference(backend: str, n_train: int, *, cg_tolerance: float = 1e-8,
                         exhaustive: bool = True) -> Iterator[dict[str, Any]]:
    """Force direct dense or guarded iterative graph-time inference.

    Yield actual installed-library CG/SLQ call diagnostics. Full orthonormal
    probes and n_train Lanczos iterations remove trace sampling error only on
    tiny problems. NumericalWarnings, including the pinned CG denominator floor,
    remain the caller's responsibility to record; no solver defaults are patched.
    """
    if backend not in ("dense_exact", "sparse_chebyshev"):
        raise ValueError("unknown graph-time backend")
    if isinstance(n_train, bool) or not isinstance(n_train, int) or n_train < 1:
        raise ValueError("n_train must be a positive integer")
    if not np.isfinite(cg_tolerance) or cg_tolerance <= 0:
        raise ValueError("cg_tolerance must be finite and positive")
    diagnostic: dict[str, Any] = dict(
        backend=backend, cg_tolerance=cg_tolerance, exhaustive=exhaustive,
        lanczos_iterations=n_train if exhaustive else 32, cg_calls=0, slq_calls=0,
        cg_rhs_shapes=[], cg_tridiag_counts=[], slq_tridiagonal_orders=[],
        installed_cg_eps=1e-10, max_cg_iterations=500,
    )
    original_cg = operator_utils.linear_cg
    original_slq = StochasticLQ.to_dense

    def cg(matmul_closure, rhs, *args, **kwargs):
        diagnostic["cg_calls"] += 1
        diagnostic["cg_rhs_shapes"].append(list(rhs.shape))
        diagnostic["cg_tridiag_counts"].append(int(kwargs.get("n_tridiag", 0)))
        return original_cg(matmul_closure, rhs, *args, **kwargs)

    def slq(self, matrix_shape, eigenvalues, eigenvectors, funcs):
        diagnostic["slq_calls"] += 1
        diagnostic["slq_tridiagonal_orders"].append(int(eigenvalues.shape[-1]))
        return original_slq(self, matrix_shape, eigenvalues, eigenvectors, funcs)

    with ExitStack() as stack:
        stack.enter_context(gpytorch.settings.max_eager_kernel_size(0))
        stack.enter_context(patch.object(operator_utils, "linear_cg", cg))
        stack.enter_context(patch.object(StochasticLQ, "to_dense", slq))
        if backend == "dense_exact":
            stack.enter_context(gpytorch.settings.max_cholesky_size(10**9))
            stack.enter_context(gpytorch.settings.fast_computations(
                covar_root_decomposition=False, log_prob=False, solves=False))
        else:
            stack.enter_context(iterative_settings(
                cg_tolerance=cg_tolerance, lanczos_iterations=n_train if exhaustive else 32,
                trace_samples=n_train if exhaustive else 32))
            stack.enter_context(gpytorch.settings.terminate_cg_by_size(False))
            if exhaustive:
                stack.enter_context(exhaustive_trace_probes())
            stack.enter_context(sparse_graph_time_guards())
        yield diagnostic


def tiny_validation_problem() -> dict[str, Any]:
    """Return a fixed n=9, T=5 case with repeated recordings and forecasts."""
    L, _ = make_laplacian("path", n=9)
    train_pairs = np.array([
        [7, 2], [0, 0], [4, 1], [8, 3], [2, 0], [6, 1],
        [1, 3], [3, 2], [5, 0], [7, 0], [0, 2], [4, 3],
        [8, 1], [2, 2], [6, 3], [1, 1], [3, 0], [4, 1],
    ], dtype=np.float64)
    targets = np.array([[8, 4], [0, 4], [4, 4], [2, 3],
                        [8, 4], [6, 4], [1, 0], [5, 4]], dtype=np.float64)
    rng = np.random.default_rng(872341)
    return dict(L=L, num_times=5, time_grid=np.arange(5, dtype=np.float64) / 4,
                train_pairs=train_pairs, target_pairs=targets,
                train_y=rng.standard_normal(len(train_pairs)))


def _torch_nll_gradients(problem: dict[str, Any], tau: float, ell: float,
                         noise_variance: float) -> dict[str, float]:
    """Independent dense Torch expression used solely for physical derivatives."""
    parameters = [torch.tensor(value, dtype=torch.float64, requires_grad=True)
                  for value in (tau, ell, noise_variance)]
    heat_time, lengthscale, noise = parameters
    L = torch.tensor(problem["L"].toarray(), dtype=torch.float64)
    time = torch.tensor(problem["time_grid"], dtype=torch.float64)
    graph = torch.matrix_exp(-heat_time * L)
    temporal = torch.exp(-(time[:, None] - time[None, :]).square() / (2 * lengthscale.square()))
    prior = torch.kron(graph, temporal)
    pairs = torch.tensor(problem["train_pairs"], dtype=torch.int64)
    ids = pairs[:, 0] * problem["num_times"] + pairs[:, 1]
    covariance = prior[ids][:, ids] + noise * torch.eye(len(ids), dtype=torch.float64)
    values = torch.tensor(problem["train_y"], dtype=torch.float64)
    factor = torch.linalg.cholesky(covariance)
    nll = (2 * factor.diagonal().log().sum()
           + (values[:, None] * torch.cholesky_solve(values[:, None], factor)).sum()
           + len(ids) * np.log(2 * np.pi)) / (2 * len(ids))
    gradients = torch.autograd.grad(nll, parameters)
    return dict(zip(("heat_time", "lengthscale", "noise_variance"),
                    (float(value) for value in gradients)))


def _gradient_measurements(model, reference: dict[str, float]) -> dict[str, Any]:
    kernel = model.covar_module
    parameters = dict(
        heat_time=(kernel.graph_kernel.raw_heat_time,
                   kernel.graph_kernel.raw_heat_time_constraint),
        lengthscale=(kernel.time_kernel.raw_lengthscale,
                     kernel.time_kernel.raw_lengthscale_constraint),
        noise_variance=(model.likelihood.raw_noise,
                        model.likelihood.noise_covar.raw_noise_constraint),
    )
    result = {}
    for name, (raw, constraint) in parameters.items():
        jacobian = torch.autograd.grad(constraint.transform(raw).sum(), raw)[0].item()
        raw_gradient = raw.grad.item()
        physical_gradient = raw_gradient / jacobian
        reference_gradient = reference[name]
        result[name] = dict(
            raw=raw_gradient, physical=physical_gradient, reference_physical=reference_gradient,
            reference_raw=reference_gradient * jacobian,
            physical_abs_error=abs(physical_gradient - reference_gradient),
            raw_abs_error=abs(raw_gradient - reference_gradient * jacobian),
        )
    return result


def measure_validation_case(*, heat_time: float = 0.93, lengthscale: float = 0.29,
                            noise_variance: float = 0.073456789123) -> dict[str, Any]:
    """Measure fixed-parameter MLL, gradients, marginals and actual solver use.

    This deliberately tiny exact-reference check includes learned-noise
    derivatives, duplicate recordings/targets, unseen future indices, chunking,
    and mean-only prediction. Returned data are JSON serializable.
    """
    from graphgp.models.graph_time import make_graph_time_gp

    problem = tiny_validation_problem()
    train = torch.tensor(problem["train_pairs"], dtype=torch.float64)
    values = torch.tensor(problem["train_y"], dtype=torch.float64)
    targets = torch.tensor(problem["target_pairs"], dtype=torch.float64)
    prior = dense_prior(problem["L"], problem["time_grid"], heat_time, lengthscale)
    reference_nll = exact_nll(prior, problem["train_pairs"], problem["train_y"],
                              noise_variance, num_times=problem["num_times"])
    reference_marginals = posterior_marginals(
        prior, problem["train_pairs"], problem["train_y"], problem["target_pairs"],
        noise_variance, num_times=problem["num_times"])
    reference_gradients = _torch_nll_gradients(problem, heat_time, lengthscale, noise_variance)
    result: dict[str, Any] = dict(
        conditions=dict(n=9, num_times=5, n_train=len(train), n_targets=len(targets),
                        heat_time=heat_time, lengthscale=lengthscale,
                        noise_variance=noise_variance, degree=32, cg_tolerance=1e-8),
        reference_nll=reference_nll, reference_gradients=reference_gradients, backends={},
    )
    for backend in ("dense_exact", "sparse_chebyshev"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # Construction is guarded too: sparse factor setup must not use eigh.
            with ExitStack() as construction:
                if backend == "sparse_chebyshev":
                    construction.enter_context(sparse_graph_time_guards())
                model = make_graph_time_gp(
                    problem["L"], train, values, num_times=problem["num_times"],
                    backend=backend, heat_time=heat_time, lengthscale=lengthscale,
                    noise_variance=noise_variance, learn_noise=True)
            model.train()
            model.likelihood.train()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
            with graph_time_inference(backend, len(train)) as solver:
                with instrument_operator(record_widths=True) as forward:
                    loss = -mll(model(train), values)
                with instrument_operator(record_widths=True) as backward:
                    loss.backward()
                gradients = _gradient_measurements(model, reference_gradients)
                with instrument_operator(record_widths=True) as prediction:
                    moments = predict_marginals(model, targets, chunk_size=len(targets))
                    chunked = predict_marginals(model, targets, chunk_size=3)
                    mean_only = predict_marginals(model, targets, mean_only=True, chunk_size=2)
            errors = {}
            for output_name, oracle_name in (("mean", "mean"), ("latent_variance", "latent_var"),
                                              ("observation_variance", "observation_var")):
                actual = getattr(moments, output_name).detach().cpu().numpy()
                errors[output_name + "_max_abs_error"] = float(
                    np.max(np.abs(actual - reference_marginals[oracle_name])))
                errors[output_name + "_chunk_max_abs_error"] = float(
                    (getattr(moments, output_name) - getattr(chunked, output_name)).abs().max())
            errors["mean_only_max_abs_error"] = float((moments.mean - mean_only.mean).abs().max())
            if mean_only.latent_variance is not None or mean_only.observation_variance is not None:
                raise AssertionError("mean_only returned variances")
            nll_value = float(loss.detach())
            result["backends"][backend] = dict(
                nll=nll_value, nll_abs_error=abs(nll_value - reference_nll),
                nll_relative_error=abs(nll_value - reference_nll) / abs(reference_nll),
                gradients=gradients, posterior=errors, solver=solver,
                operator_phases=dict(forward=forward, backward=backward, prediction=prediction),
                warnings=[dict(category=entry.category.__name__, message=str(entry.message))
                          for entry in caught],
            )
    return result


def assert_validation_gates(measurements: dict[str, Any]) -> None:
    """Apply the unchanged Stage 03 numerical thresholds to measured results."""
    for backend, result in measurements["backends"].items():
        dense = backend == "dense_exact"
        if dense:
            if result["nll_relative_error"] > 1e-9:
                raise AssertionError("dense graph-time NLL gate failed")
        elif result["nll_abs_error"] > 1e-5:
            raise AssertionError("sparse graph-time NLL gate failed")
        for name, gradient in result["gradients"].items():
            for kind in ("raw", "physical"):
                actual = gradient[kind]
                reference = gradient["reference_" + kind]
                if not np.isfinite(actual) or abs(actual) <= 1e-12:
                    raise AssertionError(f"nonfinite or disconnected {name} {kind} gradient")
                if abs(actual - reference) > 2e-6 + 2e-4 * abs(reference):
                    raise AssertionError(f"{backend} {name} {kind} gradient gate failed")
        tolerance = 1e-8 if dense else 2e-4
        for name, error in result["posterior"].items():
            if not np.isfinite(error) or error > tolerance:
                raise AssertionError(f"{backend} {name} marginal gate failed: {error}")
        if not dense:
            if result["solver"]["cg_calls"] < 1 or result["solver"]["slq_calls"] < 1:
                raise AssertionError("sparse graph-time did not use installed CG/SLQ")
            for phase in ("forward", "backward", "prediction"):
                if result["operator_phases"][phase]["operator_matmul_calls"] < 1:
                    raise AssertionError(f"sparse graph-time has no Chebyshev MVPs in {phase}")


def measure_validation() -> dict[str, Any]:
    """Measure and enforce two deterministic tiny numerical validation cases."""
    cases = [measure_validation_case(),
             measure_validation_case(heat_time=1.73, lengthscale=0.43, noise_variance=0.16)]
    for case in cases:
        assert_validation_gates(case)
    return dict(
        passed=True, cases=cases,
        guards=dict(graph_to_dense=True, full_kronecker_to_dense=True, torch_kron=True,
                    direct_cholesky=True, unbatched_dense_eigh=True,
                    limitation="Method guards cover exercised CPU paths; selected cross blocks, "
                               "dense temporal factors and batched Lanczos eigensystems are allowed."),
        iterative_solver_note="Pinned linear_operator CG defaults remain unchanged. Requested tolerance "
                              "can be below its eps=1e-10 denominator cutoff residual floor; all emitted "
                              "warnings are retained and actual statistical errors are measured.",
    )
