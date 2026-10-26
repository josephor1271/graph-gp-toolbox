"""One isolated Stage 04 operation; numerical imports follow thread setup."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
import traceback
from unittest.mock import patch
import warnings

THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


class DataConstructionError(RuntimeError):
    """The unchanged prescribed data factorization could not be completed."""


def peak_rss_mib():
    """Whole-process high-water mark, never an allocation delta."""
    if platform.system() not in ("Darwin", "Linux"):
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2 if platform.system() == "Darwin" else 1024)


def current_rss_mib():
    try:
        return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


@contextmanager
def guarded_sparse():
    import numpy as np
    import scipy.linalg
    from graphgp.validation_graph_time import sparse_graph_time_guards
    with ExitStack() as stack:
        stack.enter_context(sparse_graph_time_guards())
        for module, name in ((np, "kron"), (np.linalg, "eigh"), (np.linalg, "eigvalsh"),
                             (scipy.linalg, "expm"), (scipy.linalg, "eigh")):
            stack.enter_context(patch.object(module, name, side_effect=AssertionError("forbidden sparse dense reference")))
        yield


def run(spec):
    before = {key: os.environ.get(key) for key in THREAD_VARIABLES}
    for key in THREAD_VARIABLES:
        os.environ[key] = "1"
    import gpytorch
    import numpy as np
    import torch
    from linear_operator.operators import DiagLinearOperator
    from graphgp.benchmarking.cases import Case, make_problem, probe_seed, profiles
    from graphgp.benchmarking.solver import relative_residual, solver_diagnostics
    from graphgp.inference.instrumentation import instrument_operator
    from graphgp.inference.predict import predict_marginals
    from graphgp.inference.solver_settings import iterative_context
    from graphgp.models.graph_time import make_graph_time_gp
    from graphgp.validation import exhaustive_trace_probes

    torch.set_num_threads(1)
    case_fields = {key: value for key, value in spec["case"].items()
                   if key in ("case_id", "n", "num_times", "m", "q", "groups")}
    case_fields["groups"] = tuple(case_fields["groups"])
    case = Case(**case_fields)
    row = {**spec, "status": "ok", "threads_before": before, "threads_effective": {k: "1" for k in THREAD_VARIABLES},
           "torch_threads": torch.get_num_threads(), "rss_convention": "Darwin bytes / 2^20; Linux KiB / 1024", "iteration_count": None}
    setup_start = time.perf_counter()
    try:
        problem = make_problem(case)
    except Exception as error:
        raise DataConstructionError(f"{type(error).__name__}: {error}") from error
    if spec["campaign"] == "solver":
        problem["target_pairs"] = problem["target_pairs"][:32]
        problem["target_ids"] = problem["target_ids"][:32]
    row["measured_q"] = len(problem["target_pairs"])
    row.update(data_setup_seconds=time.perf_counter()-setup_start, data_fingerprint=problem["fingerprint"],
               nnz=problem["L"].nnz, graph_scale=problem["graph_scale"], streams=problem["stream_metadata"])
    if spec["operation"] == "reference":
        from graphgp.benchmarking.reference import dense_reference
        start = time.perf_counter()
        row["result"] = dense_reference(problem, case, target_count=len(problem["target_pairs"]))
        row.update(elapsed_seconds=time.perf_counter()-start, model_build_seconds=0.0)
        row["rss_peak_mib"] = peak_rss_mib()
        return row

    sparse = spec["backend"] == "sparse_chebyshev"
    config = profiles(case.m)[spec["profile"]] if sparse else None
    if spec["profile"] == "exhaustive_tiny" and (case.n, case.num_times, case.m) != (16, 8, 32):
        raise ValueError("exhaustive probes restricted to declared tiny case")
    effective_seed = int(probe_seed(case, spec["probe_seed"]))
    row.update(effective_probe_seed=effective_seed, settings=config.to_dict() if config else {"direct_cholesky": True},
               fixed_settings={"max_cholesky_size": 0 if sparse else 10**9, "max_eager_kernel_size": 0,
                               "fast_pred_var": False, "fast_computations": sparse}, warmup_count=0)
    train = torch.from_numpy(problem["train_pairs"])
    target = torch.from_numpy(problem["target_pairs"])
    values = torch.from_numpy(problem["train_y"])
    start = time.perf_counter()
    with ExitStack() as construction:
        if sparse:
            construction.enter_context(guarded_sparse())
        model = make_graph_time_gp(problem["L"], train, values, num_times=case.num_times,
            backend=spec["backend"], heat_time=1.2, lengthscale=0.24, noise_variance=0.06)
        model.train()
        model.likelihood.train()
        if sum(p.numel() for p in model.parameters() if p.requires_grad) != 2:
            raise AssertionError("benchmark must have exactly two trainable scalars")
        operator = None
        if spec["operation"] == "mvp":
            operator = model.covar_module._full_grid_operator()
            if not sparse:
                operator = operator.to_dense()
    row["model_build_seconds"] = time.perf_counter()-start
    row["rss_before_operation_mib"] = current_rss_mib()
    caught = []
    with warnings.catch_warnings(record=True) as caught, ExitStack() as stack:
        warnings.simplefilter("always")
        stack.enter_context(torch.random.fork_rng(devices=[]))
        torch.manual_seed(effective_seed)
        stack.enter_context(gpytorch.settings.max_eager_kernel_size(0))
        if sparse:
            stack.enter_context(iterative_context(config))
            stack.enter_context(guarded_sparse())
            if spec["profile"] == "exhaustive_tiny":
                stack.enter_context(exhaustive_trace_probes())
        else:
            stack.enter_context(gpytorch.settings.max_cholesky_size(10**9))
            stack.enter_context(gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False))
        solver = stack.enter_context(solver_diagnostics())
        stats = stack.enter_context(instrument_operator(record_widths=True))
        operation = spec["operation"]
        start = time.perf_counter()
        result = {}
        if operation == "construct":
            # Construction is measured above; no inference is hidden in this phase.
            row["elapsed_seconds"] = row["data_setup_seconds"] + row["model_build_seconds"]
        elif operation == "mvp":
            rhs = torch.from_numpy(problem["mvp_rhs"][:, :spec["rhs_width"]])
            with torch.no_grad():
                operator @ rhs  # exactly one untimed operation warmup
                samples = []
                for _ in range(5):
                    tick = time.perf_counter()
                    product = operator @ rhs
                    samples.append(time.perf_counter()-tick)
            row.update(elapsed_seconds=float(np.median(samples)), measurement_seconds=samples, warmup_count=1,
                       mvp_counter_scope="one warmup plus five measured products")
            result["product"] = product.numpy().tolist()
        elif operation in ("mll", "train3"):
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.05) if operation == "train3" else None
            path = []
            forward_seconds = backward_seconds = 0.0
            for step in range(4 if optimizer else 1):
                model.zero_grad()
                # Fixed common probes at every evaluation in the three-update check.
                torch.manual_seed(effective_seed)
                tick = time.perf_counter()
                loss = -mll(model(train), values)
                forward_seconds += time.perf_counter()-tick
                tick = time.perf_counter()
                loss.backward()
                backward_seconds += time.perf_counter()-tick
                grads = {"heat_time": model.covar_module.graph_kernel.raw_heat_time.grad.item(),
                         "lengthscale": model.covar_module.time_kernel.raw_lengthscale.grad.item()}
                if not np.isfinite([loss.item(), *grads.values()]).all():
                    raise ArithmeticError("nonfinite MLL/gradient")
                path.append(dict(step=step, nll=loss.item(), raw_gradients=grads,
                                 heat_time=model.covar_module.heat_time.item(), lengthscale=model.covar_module.lengthscale.item()))
                if optimizer and step < 3:
                    optimizer.step()
            result.update(nll=loss.item(), raw_gradients=grads, trajectory=path,
                          final_gradient_norm=float(np.linalg.norm(list(grads.values()))), updates=3 if optimizer else 0)
            row.update(forward_seconds=forward_seconds, backward_seconds=backward_seconds, elapsed_seconds=time.perf_counter()-start)
        elif operation in ("mean", "marginal"):
            moments = predict_marginals(model, target, mean_only=operation == "mean", chunk_size=64)
            row["elapsed_seconds"] = time.perf_counter()-start
            result["mean"] = moments.mean.numpy().tolist()
            if operation == "mean":
                assert moments.latent_variance is None and moments.observation_variance is None
                result["variance_returned"] = False
            else:
                result.update(latent_variance=moments.latent_variance.numpy().tolist(),
                              observation_variance=moments.observation_variance.numpy().tolist(),
                              minimum_latent_variance=moments.latent_variance.min().item(),
                              minimum_observation_variance=moments.observation_variance.min().item())
        elif operation == "solve":
            with torch.no_grad():
                A = model.covar_module(train) + DiagLinearOperator(model.likelihood.noise.expand(case.m))
                rhs = torch.from_numpy(problem["solve_rhs"])
                x = A.solve(rhs)
                # Independent fresh noisy covariance product, not a solver warning.
                fresh = model.covar_module(train) + DiagLinearOperator(model.likelihood.noise.expand(case.m))
                residuals = relative_residual(fresh, x, rhs)
            row["elapsed_seconds"] = time.perf_counter()-start
            result.update(relative_residual=residuals.tolist(), solutions=x.numpy().tolist())
        else:
            raise ValueError(f"unknown operation {operation}")
        row.update(result=result, operator=stats, solver=solver)
    row["warnings"] = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
    row["numerical_warning_count"] = sum(w["category"] == "NumericalWarning" for w in row["warnings"])
    row["first_call_seconds"] = row["elapsed_seconds"] if operation in ("mll", "train3", "mean", "marginal", "solve") else None
    row["rss_peak_mib"] = peak_rss_mib()
    if row["rss_peak_mib"] is None:
        row["memory_status"] = "unsupported"
    else:
        row["memory_status"] = "measured"
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    with warnings.catch_warnings(record=True) as outer_warnings:
        warnings.simplefilter("always")
        try:
            result = run(spec)
        except Exception as error:
            result = {**spec, "status": "data_error" if isinstance(error, DataConstructionError) else "error",
                      "error_type": type(error).__name__, "error": str(error),
                      "traceback": traceback.format_exc(), "rss_peak_mib": peak_rss_mib()}
    result["setup_warnings"] = [{"category": w.category.__name__, "message": str(w.message)} for w in outer_warnings]
    Path(args.output).write_text(json.dumps(result, allow_nan=False, indent=2)+"\n")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
