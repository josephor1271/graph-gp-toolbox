"""Fixed predictive study: graph-only, time-only, and separable graph × time GPs.

Training always uses GPyTorch ExactMarginalLogLikelihood. Dense independent
oracles generate the data and diagnose endpoints; they do not train a model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any
import warnings

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

import gpytorch
from linear_operator.utils.warnings import NumericalWarning
import numpy as np
from scipy.linalg import expm
import torch

from graphgp.analysis.graph_time_oracle import dense_prior, exact_nll, posterior_marginals, sample_problem
from graphgp.inference.predict import predict_marginals
from graphgp.kernels.graph_time import GraphTimeHeatRBFKernel, LENGTHSCALE_RANGE
from graphgp.models.exact_gp import GraphGP
from graphgp.models.factory import HEAT_RANGE, NOISE_RANGE, make_heat_gp
from graphgp.models.graph_time import make_graph_time_gp
from graphgp.utils.graphs import make_laplacian
from graphgp.validation import environment_metadata, instrument_operator, write_csv
from graphgp.validation_graph_time import graph_time_inference, measure_validation, sparse_graph_time_guards


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows)
    path.write_text(path.read_text())  # LF in curated Git evidence too.


def configuration(smoke: bool = False) -> dict[str, Any]:
    """Return the fixed design before any sample or fitted outcome is observed."""
    return dict(
        study="smoke" if smoke else "full", starting_main_sha="19b022fd9cadad3026534b4c98f4134d8b3c5d64",
        graph="grid", n=9 if smoke else 16, num_times=5 if smoke else 10, graph_seed=2718,
        laplacian="combinatorial / maximum degree; spectrum enclosed by [0,2]",
        time_grid="j/(T-1), fixed across training and targets", coordinate_order="vertex-major/time-minor: v*T+j",
        true_heat_time=1.2, true_lengthscale=0.24, true_noise_variance=0.06,
        amplitude=1.0, mean=0.0, seeds=[11] if smoke else [11, 22, 33, 44],
        tasks=["interpolation"] if smoke else ["interpolation", "forecast"],
        interpolation_train_count=12 if smoke else 64, forecast_train_count=None if smoke else 48,
        forecast_start=None if smoke else 6, main_models=["joint"] if smoke else ["graph_only", "time_only", "joint"],
        main_steps=3 if smoke else 160, parity_steps=40, parity_seeds=[] if smoke else [11, 22],
        sensitivity_seeds=[] if smoke else [11], learning_rate=0.05, optimizer="torch.optim.Adam",
        initialization=dict(heat_time=0.7, lengthscale=0.4, noise_variance=0.06,
                            sensitivity_noise_variance=0.12),
        known_noise_main=True, parity_backends=["dense_exact", "sparse_chebyshev"], degree=32,
        heat_range=list(HEAT_RANGE), lengthscale_range=list(LENGTHSCALE_RANGE), noise_range=list(NOISE_RANGE),
        parity_cg_tolerance=1e-7, validation_cg_tolerance=1e-8, max_cg_iterations=500,
        max_preconditioner_size=0, exhaustive_probes="complete training-space orthonormal basis, quadratic storage",
        requested_lanczos_iterations="n_train; actual tridiagonal orders are recorded",
        installed_cg_eps=1e-10, solver_defaults_modified=False,
        independent_unit="one latent/noisy full field per seed; tasks and models are paired",
        rng="NumPy SeedSequence(seed), spawn keys latent=(0,), noise=(1,), interpolation=(2,0), forecast=(2,1)",
        representative=dict(seed=11, task="interpolation", model="joint", protocol="main", backend="dense_exact"),
        expected_fit_counts=dict(main=1 if smoke else 24, parity=0 if smoke else 8,
                                 noise_sensitivity=0 if smoke else 2),
        train_time_definition="optimization plus final loss/gradient evaluation; excludes setup and prediction",
        dtype="CPU float64", posterior="requested target marginals, raw latent diagonal, no clipping",
    )


def kernel_validation() -> dict[str, Any]:
    """Recompute small independent covariance/MVP gates and retain maxima."""
    maxima = dict(dense_prior_max_abs=0.0, sparse_mvp_max_abs=0.0,
                  sparse_selected_max_abs=0.0, row_major_action_max_abs=0.0)
    count = 0
    for family in ("path", "grid"):
        for n in (9, 16):
            L, _ = make_laplacian(family, n)
            for total_times in (4, 7):
                time_grid = np.arange(total_times, dtype=np.float64)/(total_times-1)
                all_ids = np.arange(n*total_times)
                pairs = torch.tensor(np.column_stack((all_ids//total_times, all_ids % total_times)), dtype=torch.float64)
                for tau in (0.7, 2.2):
                    for ell in (0.24, 0.56):
                        reference = dense_prior(L, time_grid, tau, ell)
                        dense = GraphTimeHeatRBFKernel(L, num_times=total_times, backend="dense_exact",
                                                      heat_time=tau, lengthscale=ell)
                        actual = dense(pairs).to_dense().detach().numpy()
                        np.testing.assert_allclose(actual, reference, rtol=1e-11, atol=1e-12)
                        maxima["dense_prior_max_abs"] = max(maxima["dense_prior_max_abs"], float(np.max(np.abs(actual-reference))))
                        rng = np.random.default_rng(2718)
                        graph = expm(-tau*L.toarray())
                        temporal = np.exp(-(time_grid[:, None]-time_grid[None, :])**2/(2*ell**2))
                        with sparse_graph_time_guards():
                            sparse = GraphTimeHeatRBFKernel(L, num_times=total_times, backend="sparse_chebyshev",
                                                           heat_time=tau, lengthscale=ell)
                            for width in (1, 3):
                                rhs = rng.standard_normal((n*total_times, width))
                                result = (sparse(pairs) @ torch.from_numpy(rhs)).detach().numpy()
                                np.testing.assert_allclose(result, reference@rhs, rtol=1e-9, atol=1e-10)
                                maxima["sparse_mvp_max_abs"] = max(maxima["sparse_mvp_max_abs"], float(np.max(np.abs(result-reference@rhs))))
                                reshaped = rhs.reshape(n, total_times, width)
                                expected_action = np.stack([graph@reshaped[:, :, k]@temporal.T for k in range(width)], axis=-1).reshape(n*total_times, width)
                                maxima["row_major_action_max_abs"] = max(maxima["row_major_action_max_abs"], float(np.max(np.abs(result-expected_action))))
                            left, right = [n*total_times-1, 1, 1, 4, 7], [0, 5, 2]
                            selected = sparse(pairs[left], pairs[right]).to_dense().detach().numpy()
                            expected = reference[np.ix_(left, right)]
                            np.testing.assert_allclose(selected, expected, rtol=1e-9, atol=1e-10)
                            maxima["sparse_selected_max_abs"] = max(maxima["sparse_selected_max_abs"], float(np.max(np.abs(selected-expected))))
                        count += 1
    return dict(passed=True, cases=count, families=["path", "grid"], nodes=[9, 16],
                num_times=[4, 7], heat_times=[0.7, 2.2], lengthscales=[0.24, 0.56], **maxima)


def model_inputs(pairs: np.ndarray, model_name: str, time_grid: np.ndarray) -> torch.Tensor:
    if model_name == "graph_only":
        return torch.tensor(pairs[:, 0], dtype=torch.long)
    if model_name == "time_only":
        return torch.tensor(time_grid[pairs[:, 1].astype(np.int64), None], dtype=torch.float64)
    return torch.tensor(pairs, dtype=torch.float64)


def construct_model(problem: dict[str, Any], model_name: str, backend: str,
                    learn_noise: bool) -> GraphGP:
    values = torch.tensor(problem["y"][problem["train_ids"]], dtype=torch.float64)
    train = model_inputs(problem["train_pairs"], model_name, problem["time_grid"])
    noise = 0.12 if learn_noise else 0.06
    if model_name == "joint":
        return make_graph_time_gp(problem["L"], train, values, num_times=len(problem["time_grid"]),
            backend=backend, heat_time=0.7, lengthscale=0.4, noise_variance=noise, learn_noise=learn_noise)
    if model_name == "graph_only":
        model = make_heat_gp(problem["L"], train, values, "C", heat_time=0.7, noise=noise,
                             amplitude=1.0, learn_amplitude=False)
        model.likelihood.raw_noise.requires_grad_(False)
        return model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.Interval(*NOISE_RANGE).double()).double()
    likelihood.noise = torch.tensor(noise, dtype=torch.float64)
    likelihood.raw_noise.requires_grad_(False)
    kernel = gpytorch.kernels.RBFKernel(
        lengthscale_constraint=gpytorch.constraints.Interval(*LENGTHSCALE_RANGE).double()).double()
    kernel.lengthscale = torch.tensor(0.4, dtype=torch.float64)
    return GraphGP(train, values, likelihood, kernel).double()


def fitted_parameters(model: GraphGP, model_name: str) -> dict[str, float | None]:
    if model_name == "joint":
        tau, ell = model.covar_module.heat_time.item(), model.covar_module.lengthscale.item()
    elif model_name == "graph_only":
        tau, ell = model.covar_module.base_kernel.heat_time.item(), None
    else:
        tau, ell = None, model.covar_module.lengthscale.item()
    return dict(heat_time=tau, lengthscale=ell, noise_variance=model.likelihood.noise.item())


def independent_endpoint_prior(problem: dict[str, Any], model_name: str,
                               parameters: dict[str, Any]) -> np.ndarray:
    times = problem["time_grid"]
    n = problem["L"].shape[0]
    if model_name == "joint":
        return dense_prior(problem["L"], times, parameters["heat_time"], parameters["lengthscale"])
    if model_name == "graph_only":
        return np.kron(expm(-parameters["heat_time"]*problem["L"].toarray()), np.ones((len(times), len(times))))
    temporal = np.exp(-(times[:, None]-times[None, :])**2/(2*parameters["lengthscale"]**2))
    return np.kron(np.ones((n, n)), temporal)


def fit_one(problem: dict[str, Any], model_name: str, backend: str, protocol: str,
            steps: int, output: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    meta = dict(seed=problem["meta"]["seed"], task=problem["meta"]["split"], model=model_name,
                backend=backend, protocol=protocol)
    fit_id = f"s{meta['seed']}_{meta['task']}_{protocol}_{model_name}_{backend}"
    meta["fit_id"] = fit_id
    learn_noise = protocol == "noise_sensitivity"
    row = meta | dict(steps=steps, learn_noise=learn_noise, fit_status="failed", updates_completed=0,
        initial_heat_time=0.7 if model_name != "time_only" else None,
        initial_lengthscale=0.4 if model_name != "graph_only" else None,
        initial_noise_variance=0.12 if learn_noise else 0.06,
        n_train=len(problem["train_ids"]), n_targets=len(problem["test_ids"]),
        inference="CG/SLQ" if backend == "sparse_chebyshev" else "direct Cholesky")
    metric = meta | dict(fit_status="failed")
    history, signal = [], []
    caught: list[Any] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if backend == "sparse_chebyshev":
                with sparse_graph_time_guards():
                    model = construct_model(problem, model_name, backend, learn_noise)
            else:
                model = construct_model(problem, model_name, backend, learn_noise)
            model.train()
            model.likelihood.train()
            train = model_inputs(problem["train_pairs"], model_name, problem["time_grid"])
            values = torch.tensor(problem["y"][problem["train_ids"]], dtype=torch.float64)
            targets = model_inputs(problem["test_pairs"], model_name, problem["time_grid"])
            if not learn_noise and model.likelihood.raw_noise.requires_grad:
                raise AssertionError("main/parity observation noise is not frozen")
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.05)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
            context_backend = "sparse_chebyshev" if backend == "sparse_chebyshev" else "dense_exact"
            with graph_time_inference(context_backend, len(train), cg_tolerance=1e-7) as solver:
                start = time.perf_counter()
                with instrument_operator() as training_stats:
                    for step in range(steps+1):
                        optimizer.zero_grad()
                        loss = -mll(model(train), values)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("nonfinite training NLL")
                        loss.backward()
                        gradients = [p.grad for p in model.parameters() if p.requires_grad]
                        if any(g is None or not torch.isfinite(g).all() for g in gradients):
                            raise FloatingPointError("nonfinite or disconnected training gradient")
                        norm = float(torch.sqrt(sum(g.square().sum() for g in gradients)))
                        history.append(meta | dict(step=step, nll=loss.item(), gradient_norm=norm,
                                                   **fitted_parameters(model, model_name)))
                        if step < steps:
                            optimizer.step()
                            row["updates_completed"] += 1
                row["training_seconds"] = time.perf_counter()-start
                row.update(first_nll=history[0]["nll"], final_nll=history[-1]["nll"],
                           final_gradient_norm=history[-1]["gradient_norm"], **fitted_parameters(model, model_name))
                prediction_start = time.perf_counter()
                with instrument_operator() as prediction_stats:
                    moments = predict_marginals(model, targets, chunk_size=64)
                    mean_only = predict_marginals(model, targets, mean_only=True, chunk_size=31)
                    full = None
                    if meta["seed"] == 11 and meta["task"] == "interpolation" and protocol == "main" and model_name == "joint":
                        all_inputs = model_inputs(problem["all_pairs"], model_name, problem["time_grid"])
                        full = predict_marginals(model, all_inputs, chunk_size=64)
                metric["prediction_seconds"] = time.perf_counter()-prediction_start
            row.update(training_graph_operator_calls=training_stats["operator_matmul_calls"],
                       training_sparse_laplacian_mvps=training_stats["estimated_sparse_l_mvps"],
                       prediction_graph_operator_calls=prediction_stats["operator_matmul_calls"],
                       cg_calls=solver["cg_calls"], slq_calls=solver["slq_calls"],
                       actual_slq_orders=json.dumps(solver["slq_tridiagonal_orders"]))
            if backend == "sparse_chebyshev" and (row["cg_calls"] < 1 or row["slq_calls"] < 1 or row["training_graph_operator_calls"] < 1):
                raise AssertionError("matrix-free fit did not exercise Chebyshev CG/SLQ")
            parameters = fitted_parameters(model, model_name)
            prior = independent_endpoint_prior(problem, model_name, parameters)
            reference = posterior_marginals(prior, problem["train_pairs"], values.numpy(), problem["test_pairs"],
                parameters["noise_variance"], num_times=len(problem["time_grid"]))
            row["exact_train_nll"] = exact_nll(prior, problem["train_pairs"], values.numpy(), parameters["noise_variance"],
                                             num_times=len(problem["time_grid"]))
            mean = moments.mean.numpy()
            latent_variance, observation_variance = moments.latent_variance.numpy(), moments.observation_variance.numpy()
            mean_error = float((moments.mean-mean_only.mean).abs().max())
            if mean_error > (2e-4 if backend == "sparse_chebyshev" else 1e-8):
                raise FloatingPointError("mean-only prediction disagrees with marginal prediction")
            ids = problem["test_ids"]
            residual = problem["y"][ids]-mean
            metric.update(latent_rmse=float(np.sqrt(np.mean((mean-problem["latent"][ids])**2))),
                observation_rmse=float(np.sqrt(np.mean(residual**2))),
                observation_nlpd=float(np.mean(0.5*(np.log(2*np.pi*observation_variance)+residual**2/observation_variance))),
                coverage95=float(np.mean(np.abs(residual) <= 1.959963984540054*np.sqrt(observation_variance))),
                mean_only_max_error=mean_error, minimum_latent_variance=float(latent_variance.min()),
                minimum_observation_variance=float(observation_variance.min()),
                oracle_mean_max_error=float(np.max(np.abs(mean-reference["mean"]))),
                oracle_latent_variance_max_error=float(np.max(np.abs(latent_variance-reference["latent_var"]))),
                oracle_observation_variance_max_error=float(np.max(np.abs(observation_variance-reference["observation_var"]))))
            np.savez_compressed(output/"predictions"/(fit_id+".npz"), target_ids=ids, mean=mean,
                                latent_variance=latent_variance, observation_variance=observation_variance)
            if full is not None:
                mask = set(problem["train_ids"].tolist())
                for index, pair in enumerate(problem["all_pairs"]):
                    signal.append(dict(seed=meta["seed"], task=meta["task"], fit_id=fit_id,
                        vertex=int(pair[0]), time_index=int(pair[1]), physical_time=float(problem["time_grid"][int(pair[1])]),
                        true_latent=float(problem["latent"][index]), noisy_observation=float(problem["y"][index]),
                        observed=index in mask, posterior_mean=float(full.mean[index]),
                        latent_variance=float(full.latent_variance[index]), observation_variance=float(full.observation_variance[index])))
        messages = sorted(set(str(w.message) for w in caught if issubclass(w.category, NumericalWarning)))
        status = "ok_with_numerical_warnings" if messages else "ok"
        flags = []
        for name, bounds in (("heat_time", HEAT_RANGE), ("lengthscale", LENGTHSCALE_RANGE), ("noise_variance", NOISE_RANGE)):
            value = row.get(name)
            if value is not None and min(value-bounds[0], bounds[1]-value) <= 0.01*(bounds[1]-bounds[0]):
                flags.append(name)
        row.update(fit_status=status, boundary_flags=json.dumps(flags),
                   numerical_warnings=json.dumps(messages), numerical_warning_count=sum(issubclass(w.category, NumericalWarning) for w in caught),
                   other_warnings=json.dumps(sorted(set(str(w.message) for w in caught if not issubclass(w.category, NumericalWarning)))))
        metric["fit_status"] = status
    except Exception as error:
        row.update(error=repr(error), numerical_warnings=json.dumps([str(w.message) for w in caught if issubclass(w.category, NumericalWarning)]))
        metric["error"] = repr(error)
    print(f"{fit_id}: {row['fit_status']}; updates={row['updates_completed']}; "
          f"tau={row.get('heat_time')} ell={row.get('lengthscale')} NLL={row.get('exact_train_nll')}", flush=True)
    return row, metric, history, signal


def summarize(fits: list[dict[str, Any]], metrics: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for task in config["tasks"]:
        for name in config["main_models"]:
            group = [r for r in metrics if r["protocol"] == "main" and r["task"] == task and r["model"] == name]
            valid = [r for r in group if r["fit_status"].startswith("ok")]
            row = dict(task=task, model=name, expected_seeds=len(config["seeds"]), recorded_fits=len(group),
                       valid_fits=len(valid), failed_fits=len(group)-len(valid),
                       seeds=json.dumps([r["seed"] for r in group]))
            for key in ("latent_rmse", "observation_rmse", "observation_nlpd", "coverage95"):
                values = [r[key] for r in valid]
                if values:
                    row.update({key+"_median": float(np.median(values)), key+"_minimum": min(values), key+"_maximum": max(values)})
            rows.append(row)
    return rows


def parity_rows(fits: list[dict[str, Any]], metrics: list[dict[str, Any]], history: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    result = []
    for seed, task in sorted({(r["seed"], r["task"]) for r in fits if r["protocol"] == "parity"}):
        pair = {r["backend"]: r for r in fits if r["protocol"] == "parity" and r["seed"] == seed and r["task"] == task}
        dense, sparse = pair["dense_exact"], pair["sparse_chebyshev"]
        row = dict(seed=seed, task=task, steps=40, dense_status=dense["fit_status"], sparse_status=sparse["fit_status"])
        if dense["fit_status"].startswith("ok") and sparse["fit_status"].startswith("ok"):
            for key in ("heat_time", "lengthscale", "noise_variance", "exact_train_nll"):
                row[key+"_absolute_difference"] = abs(dense[key]-sparse[key])
            curves = [[r["nll"] for r in history if r["fit_id"] == fit["fit_id"]] for fit in (dense, sparse)]
            row["maximum_own_nll_trajectory_difference"] = max(abs(a-b) for a, b in zip(*curves))
            for name in ("mean", "latent_variance", "observation_variance"):
                with np.load(output/"predictions"/(dense["fit_id"]+".npz")) as a, np.load(output/"predictions"/(sparse["fit_id"]+".npz")) as b:
                    row["endpoint_"+name+"_max_difference"] = float(np.max(np.abs(a[name]-b[name])))
            score = next(r for r in metrics if r["fit_id"] == sparse["fit_id"])
            for key in ("oracle_mean_max_error", "oracle_latent_variance_max_error", "oracle_observation_variance_max_error"):
                row["sparse_fixed_endpoint_"+key] = score[key]
            row.update(sparse_graph_operator_calls=sparse["training_graph_operator_calls"],
                       sparse_cg_calls=sparse["cg_calls"], sparse_slq_calls=sparse["slq_calls"],
                       sparse_numerical_warning_count=sparse["numerical_warning_count"])
        result.append(row)
    return result


def run(output: Path, *, smoke: bool = False) -> None:
    if (output/"config.json").exists():
        raise FileExistsError(f"{output} already contains configured results; choose a fresh --output")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("data", "predictions"):
        (output/name).mkdir(exist_ok=True)
    torch.set_num_threads(1)
    config = configuration(smoke)
    source_files = [Path(__file__).resolve().relative_to(Path.cwd().resolve()).as_posix(), "src/graphgp/analysis/graph_time_oracle.py",
        "src/graphgp/kernels/graph_time.py", "src/graphgp/kernels/heat.py", "src/graphgp/models/graph_time.py",
        "src/graphgp/models/factory.py", "src/graphgp/validation_graph_time.py"]
    environment = environment_metadata() | dict(
        matplotlib=importlib.metadata.version("matplotlib"),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        source_hashes={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in source_files},
        blas_thread_environment={name: os.environ.get(name) for name in
            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
    save_json(output/"config.json", config)
    save_json(output/"compile_environment.json", environment)
    validation = measure_validation()
    validation["kernel_sweep"] = kernel_validation()
    validation["installed_kronecker_backend"] = "KroneckerProductLinearOperator; no custom fallback"
    save_json(output/"validation.json", validation)
    fits, metrics, trajectories, signal = [], [], [], []
    for seed in config["seeds"]:
        fields = None
        for task in config["tasks"]:
            kwargs = dict(n=config["n"], num_times=config["num_times"])
            if smoke:
                kwargs["train_count"] = 12
            problem = sample_problem(seed, task, **kwargs)
            if fields is not None:
                np.testing.assert_array_equal(problem["latent"], fields[0])
                np.testing.assert_array_equal(problem["y"], fields[1])
            fields = problem["latent"], problem["y"]
            if set(problem["train_ids"].tolist()) & set(problem["test_ids"].tolist()):
                raise AssertionError("training/held-out split leakage")
            if task == "forecast" and np.any(problem["train_pairs"][:, 1] >= 6):
                raise AssertionError("future measurements leaked into forecasting training")
            np.savez_compressed(output/"data"/f"s{seed}_{task}.npz", **{
                key: problem[key] for key in ("time_grid", "all_pairs", "train_ids", "test_ids", "latent", "y", "noise")})
            save_json(output/"data"/f"s{seed}_{task}_metadata.json", problem["meta"])
            jobs = [(name, "dense_exact" if name == "joint" else "direct_"+name, "main", config["main_steps"])
                    for name in config["main_models"]]
            if seed in config["parity_seeds"]:
                jobs.extend(("joint", backend, "parity", config["parity_steps"]) for backend in config["parity_backends"])
            if seed in config["sensitivity_seeds"]:
                jobs.append(("joint", "dense_exact", "noise_sensitivity", 160))
            for name, backend, protocol, steps in jobs:
                fit, score, history, illustration = fit_one(problem, name, backend, protocol, steps, output)
                fits.append(fit); metrics.append(score); trajectories.extend(history); signal.extend(illustration)
                save_csv(output/"fits.csv", fits)
                save_csv(output/"metrics.csv", metrics)
                save_csv(output/"trajectories.csv", trajectories)
                if signal:
                    save_csv(output/"signal.csv", signal)
    save_csv(output/"summary.csv", summarize(fits, metrics, config))
    save_csv(output/"parity.csv", parity_rows(fits, metrics, trajectories, output))
    completion = dict(independent_fields=len(config["seeds"]), tasks_share_fields=True,
        main_fits=sum(r["protocol"] == "main" for r in fits), parity_fits=sum(r["protocol"] == "parity" for r in fits),
        noise_sensitivity_fits=sum(r["protocol"] == "noise_sensitivity" for r in fits),
        total_fits=len(fits), failed_fits=sum(not r["fit_status"].startswith("ok") for r in fits),
        fits_with_numerical_warnings=sum(r.get("numerical_warning_count", 0) > 0 for r in fits))
    save_json(output/"completion.json", completion)
    print(json.dumps(completion, indent=2), flush=True)
    for protocol, expected in config["expected_fit_counts"].items():
        if completion[protocol+"_fits"] != expected:
            raise AssertionError(f"missing or duplicated {protocol} fits")
    if completion["failed_fits"]:
        raise RuntimeError("One or more fits failed; all outcomes are retained in fits.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.output or Path("results/graph_time_smoke" if args.smoke else "results/graph_time"), smoke=args.smoke)
