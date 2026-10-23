"""Study heat-time/noise identifiability using independent exact-GP draws.

The dense oracle is a small statistical reference. Production graph GP inference
continues to use GPyTorch and the existing sparse Chebyshev operator.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

# Set BLAS controls before importing NumPy/SciPy, without changing dependencies.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import torch

from graphgp.analysis.heat_identifiability import HeatOracle, sample_pool, fit_multistart
from graphgp.models.factory import HEAT_RANGE, NOISE_RANGE
from graphgp.utils.graphs import make_laplacian
from graphgp.validation import write_csv, environment_metadata


FAMILY_CODES = {"path": 1, "grid": 2, "irregular": 3}
REGIMES = ((0.6, 0.03, 0.75), (2.0, 0.12, 0.5))
INITIALS = ((0.3, 0.04), (2.8, 0.2))
GROUP_FIELDS = ("family", "n", "regime", "observation_design", "observed_fraction",
                "n_observed", "replicates", "fit_mode")


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def save_csv(path, rows):
    """Reuse the repository writer, normalizing only line endings for Git."""
    write_csv(path, rows)
    path = Path(path)
    path.write_text(path.read_text())


def configuration(*, smoke=False, secondary=False, graphs=None, pools=None,
                  replicate_counts=None):
    if smoke and secondary:
        raise ValueError("--smoke and --secondary are separate study designs")
    n = 16 if smoke else (100 if secondary else 49)
    families = ["path"] if smoke else (["grid", "irregular"] if secondary else list(FAMILY_CODES))
    families = families if graphs is None else list(dict.fromkeys(graphs))
    counts = ([1, 5] if smoke else ([1, 20] if secondary else [1, 5, 20]))
    counts = counts if replicate_counts is None else sorted(set(replicate_counts))
    n_pools = (2 if smoke else (10 if secondary else 20)) if pools is None else pools
    if n_pools < 1 or not counts or counts[0] < 1 or not families:
        raise ValueError("positive pool/replicate counts required")
    return dict(
        study="smoke" if smoke else ("secondary" if secondary else "primary"),
        n=n, families=families, graph_seed=2718, graph_family_codes=FAMILY_CODES,
        laplacian="combinatorial / maximum degree (spectrum enclosed by [0,2])",
        amplitude=1.0, mean=0.0, regimes=[list(x) for x in (REGIMES[:1] if smoke else REGIMES)],
        observation_designs=["full"] if secondary else ["partial", "full"],
        pool_seeds=list(range(n_pools)), mask_seed=44001,
        rng="NumPy SeedSequence with separate latent, noise and mask streams; PCG64",
        stream_key="(n, graph_family_code, regime); topology uses separate graph_seed",
        mask_policy="one fixed permutation per graph/regime, shared across pools and replicate budgets",
        paired_observations="partial/full use identical full latent/noisy draws in each pool",
        nested_replicates="prefixes of one max-R draw sequence, with fresh latent AND noise per column",
        replicates=counts, fit_modes=["joint", "noise_known"],
        heat_range=list(HEAT_RANGE), noise_variance_range=list(NOISE_RANGE),
        optimizer="scipy.optimize.minimize, bounded L-BFGS-B in log parameters",
        initials=[list(x) for x in INITIALS], maxiter=250, gtol=1e-9, ftol=1e-13,
        stationarity_tolerance=1e-6, boundary_definition="within 1% of physical range width",
        selection="lowest evaluated exact NLL among numerically valid starts, including nonconverged starts",
        extra_grid_starts=False,
        representative=dict(family="grid" if "grid" in families else families[0], n=n, regime=0,
                            observation_design="full" if secondary else "partial",
                            pool_seed=0, replicates=1),
        landscape_grid=dict(tau_log_nodes=45, noise_log_nodes=45,
                            include_truth=True, include_selected_fit=True),
        dtype="float64", torch_threads=1,
        normalization="NLL per observed node per independent realization; KL per observed node",
        fisher_normalization="total log-parameter information across R independent vectors, not divided by nodes or R",
    )


def validate_existing_paths():
    """Independent dense/GpyTorch comparisons at three non-optimum parameter pairs."""
    import gpytorch
    from graphgp.recovery import exact_nll
    from graphgp.models.factory import make_heat_gp
    from graphgp.validation import inference_context

    L, _ = make_laplacian("path", 16)
    oracle = HeatOracle(L)
    pool = sample_pool(oracle, 0.6, 0.03, 5, 2026, mask_seed=44001,
                       stream_key=(16, 1, 0))
    observed = np.asarray(pool["permutation"][:12], dtype=np.int64)
    Y = pool["observations"][observed, :1]
    ids = torch.from_numpy(observed)
    y = torch.from_numpy(Y[:, 0].copy())
    evals, evecs = torch.linalg.eigh(torch.from_numpy(L.toarray()))
    problem = dict(train=ids, y=torch.from_numpy(pool["observations"][:, 0].copy()),
                   evals=evals, evecs=evecs)
    comparisons = []
    for tau, noise in ((0.35, 0.06), (1.3, 0.18), (3.2, 0.31)):
        value = oracle.nll(tau, noise, Y, observed)
        old = exact_nll(problem, tau, noise)
        row = dict(tau=tau, noise_variance=noise, oracle_nll=value,
                   recovery_nll=old, recovery_absolute_error=abs(value-old))
        for method in ("A", "C"):
            model = make_heat_gp(L, ids, y, method, heat_time=tau, noise=noise)
            # GaussianLikelihood's Python-scalar setter can first create a
            # float32 tensor, even on a double model. Match the physical
            # covariance parameters exactly for this float64 validation gate.
            model.likelihood.noise = torch.tensor(noise, dtype=torch.float64)
            row[method+"_stored_tau"] = model.covar_module.base_kernel.heat_time.item()
            row[method+"_stored_noise_variance"] = model.likelihood.noise.item()
            model.train()
            model.likelihood.train()
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
            with inference_context(method, len(ids)):
                loss = -mll(model(ids), y)
                row[method+"_nll"] = loss.item()
                row[method+"_absolute_error"] = abs(loss.item()-value)
        if row["recovery_absolute_error"] > 1e-9 or row["A_absolute_error"] > 1e-9 or row["C_absolute_error"] > 1e-7:
            raise ArithmeticError(f"Likelihood normalization/parity gate failed: {row}")
        comparisons.append(row)
    # Detailed deterministic derivative, KL, Fisher, boundary and sampling
    # invariants live in tests/test_heat_identifiability.py and run in CI.
    return dict(status="passed", fixed_parameter_comparisons=comparisons,
                mathematical_tests="tests/test_heat_identifiability.py (run full suite before scientific studies)",
                dense_sparse_regression="existing tests/test_recovery.py and tests/test_graph_gp_validation.py retained",
                gpytorch_noise_assignment="explicit torch.float64 tensor after factory construction to avoid Python-scalar default-float32 rounding",
                no_jitter=True, no_matrix_inverse=True)


def fisher_rows(oracle, observed, truth, meta, replicate_counts):
    result = []
    for count in replicate_counts:
        diagnostics = oracle.fisher_diagnostics(*truth, observed, replicates=count,
                                                log_coordinates=True)
        result.append(meta | dict(replicates=count,
            coordinates="log(tau), log(noise_variance)") | diagnostics)
    return result


def summary_rows(selected):
    """Summarize one selected fit per independent pool, with failures counted."""
    groups = {}
    for row in selected:
        key = tuple(row[field] for field in GROUP_FIELDS)
        groups.setdefault(key, []).append(row)
    summary = []
    for key, rows in groups.items():
        valid = [row for row in rows if row["valid"]]
        record = dict(zip(GROUP_FIELDS, key)) | dict(
            pools_attempted=len(rows), pools_valid=len(valid), pools_failed=len(rows)-len(valid),
            optimizer_success_count=sum(row["optimizer_success"] for row in valid),
            stationary_count=sum(row.get("stationary", False) for row in valid),
            boundary_count=sum(bool(json.loads(row["boundary_flags"])) for row in valid),
            tau_boundary_count=sum(row["tau_near_boundary"] for row in valid),
            noise_boundary_count=sum(row["noise_near_boundary"] for row in valid),
            start_disagreement_count=sum(row.get("start_disagreement", False) for row in valid),
        )
        for parameter in ("tau", "noise"):
            if parameter == "noise" and key[-1] == "noise_known":
                continue
            value_field = "fit_tau" if parameter == "tau" else "fit_noise_variance"
            truth_field = "true_tau" if parameter == "tau" else "true_noise_variance"
            values = np.array([row[value_field] for row in valid])
            errors = np.array([row[value_field]-row[truth_field] for row in valid])
            relative = np.array([row[parameter+"_relative_error"] for row in valid])
            if not len(values):
                continue
            record.update({f"{parameter}_estimate_median": float(np.median(values)),
                           f"{parameter}_estimate_p10": float(np.percentile(values, 10)),
                           f"{parameter}_estimate_p90": float(np.percentile(values, 90)),
                           f"{parameter}_rmse": float(np.sqrt(np.mean(errors**2))),
                           f"{parameter}_relative_error_median": float(np.median(relative)),
                           f"{parameter}_relative_error_p10": float(np.percentile(relative, 10)),
                           f"{parameter}_relative_error_p90": float(np.percentile(relative, 90))})
        summary.append(record)
    return summary


def paired_rows(selected, replicate_counts):
    """Paired R=1 versus larger R; neither starts nor R count as new pools."""
    index = {}
    for row in selected:
        key = tuple(row[k] for k in GROUP_FIELDS if k != "replicates")
        index.setdefault(key, {}).setdefault(row["pool_seed"], {})[row["replicates"]] = row
    result = []
    if 1 not in replicate_counts:
        return result
    for key, pools in index.items():
        for count in replicate_counts:
            if count == 1:
                continue
            pairs = [(pool[1], pool[count]) for pool in pools.values()
                     if 1 in pool and count in pool and pool[1]["valid"] and pool[count]["valid"]]
            record = dict(zip([k for k in GROUP_FIELDS if k != "replicates"], key))
            record.update(replicates_reference=1, replicates_comparison=count, paired_pools=len(pairs))
            for parameter in ("tau", "noise"):
                if parameter == "noise" and record["fit_mode"] == "noise_known":
                    continue
                field = parameter+"_relative_error"
                delta = np.array([b[field]-a[field] for a, b in pairs])
                if len(delta):
                    record.update({parameter+"_error_delta_median": float(np.median(delta)),
                                   parameter+"_improved_fraction": float(np.mean(delta < 0))})
            result.append(record)
    return result


def check_output_directory(output):
    if (output/"config.json").exists():
        raise FileExistsError(f"{output} already contains a study; choose a new --output to preserve its evidence")
    output.mkdir(parents=True, exist_ok=True)
    (output/"pools").mkdir(exist_ok=True)


def run_study(output, config):
    output = Path(output)
    check_output_directory(output)
    torch.set_num_threads(1)
    config = dict(config)
    config["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config["source_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    config["driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    oracle_file = Path(__file__).parents[1]/"src/graphgp/analysis/heat_identifiability.py"
    config["oracle_sha256"] = hashlib.sha256(oracle_file.read_bytes()).hexdigest()
    config["expected_selected_fits"] = (len(config["families"])*len(config["regimes"])*
        len(config["observation_designs"])*len(config["replicates"])*len(config["pool_seeds"])*len(config["fit_modes"]))
    dump_json(output/"config.json", config)
    environment = environment_metadata()
    environment["blas_thread_environment"] = {key: os.environ.get(key) for key in
        ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
    dump_json(output/"environment.json", environment)
    validation = validate_existing_paths()
    dump_json(output/"validation.json", validation)
    starts, selected, information, landscapes, masks = [], [], [], [], []
    wall_start = time.perf_counter()
    for family in config["families"]:
        L, scale = make_laplacian(family, config["n"], seed=config["graph_seed"])
        oracle = HeatOracle(L)
        for regime, (tau, noise, fraction) in enumerate(config["regimes"]):
            for pool_seed in config["pool_seeds"]:
                pool = sample_pool(oracle, tau, noise, max(config["replicates"]), pool_seed,
                    mask_seed=config["mask_seed"], stream_key=(config["n"], FAMILY_CODES[family], regime))
                permutation = pool["permutation"]
                pool_name = f"{family}_n{config['n']}_r{regime}_s{pool_seed}"
                np.savez_compressed(output/"pools"/(pool_name+".npz"), **pool)
                for design in config["observation_designs"]:
                    count = int(config["n"]*fraction) if design == "partial" else config["n"]
                    observed = np.asarray(permutation[:count], dtype=np.int64)
                    meta = dict(family=family, n=config["n"], regime=regime,
                                observation_design=design, observed_fraction=count/config["n"],
                                requested_fraction=fraction if design == "partial" else 1.0,
                                n_observed=count, true_tau=tau, true_noise_variance=noise,
                                laplacian_scale=scale)
                    if pool_seed == config["pool_seeds"][0]:
                        masks.append(meta | dict(observed_ids=json.dumps(observed.tolist())))
                        information.extend(fisher_rows(oracle, observed, (tau, noise), meta, config["replicates"]))
                    for replicates in config["replicates"]:
                        Y = pool["observations"][observed, :replicates]
                        truth_nll = oracle.nll(tau, noise, Y, observed)
                        for mode in config["fit_modes"]:
                            fit_start = time.perf_counter()
                            fit_starts, best = fit_multistart(oracle, Y, observed,
                                known_noise=noise if mode == "noise_known" else None,
                                initials=tuple(map(tuple, config["initials"])),
                                maxiter=config["maxiter"], gtol=config["gtol"], ftol=config["ftol"])
                            common = meta | dict(pool_seed=pool_seed, replicates=replicates, fit_mode=mode,
                                                 truth_nll_per_observation=truth_nll)
                            for record in fit_starts + [best]:
                                record.update(nll_gap_fit_minus_truth=(record["nll_per_observation"]-truth_nll)
                                              if record["valid"] else "",
                                    tau_relative_error=abs(record["fit_tau"]-tau)/tau if record["valid"] else "",
                                    noise_relative_error=(abs(record["fit_noise_variance"]-noise)/noise)
                                              if record["valid"] and mode == "joint" else "")
                            starts.extend(common | point for point in fit_starts)
                            selected.append(common | best | dict(fit_seconds=time.perf_counter()-fit_start))
                            scenario = config["representative"]
                            if (family == scenario["family"] and regime == scenario["regime"] and
                                design == scenario["observation_design"] and pool_seed == scenario["pool_seed"] and
                                replicates == scenario["replicates"] and mode == "joint"):
                                grids = (np.unique(np.r_[np.geomspace(*HEAT_RANGE, 45), tau,
                                           best["fit_tau"] if best["valid"] else tau]),
                                         np.unique(np.r_[np.geomspace(*NOISE_RANGE, 45), noise,
                                           best["fit_noise_variance"] if best["valid"] else noise]))
                                for grid_tau in grids[0]:
                                    for grid_noise in grids[1]:
                                        landscapes.append(common | dict(grid_tau=float(grid_tau),
                                            grid_noise_variance=float(grid_noise),
                                            sample_nll_per_observation=oracle.nll(grid_tau, grid_noise, Y, observed),
                                            expected_excess_nll=oracle.expected_excess(grid_tau, grid_noise, tau, noise, observed),
                                            fit_tau=best["fit_tau"], fit_noise_variance=best["fit_noise_variance"]))
                # Checkpoint after each independent pool, with every start retained.
                save_csv(output/"fit_starts.csv", starts)
                save_csv(output/"fit_selected.csv", selected)
                print(f"{pool_name}: {len(selected)}/{config['expected_selected_fits']} selected fits; "
                      f"{sum(not row['valid'] for row in selected)} invalid", flush=True)
    save_csv(output/"summary.csv", summary_rows(selected))
    save_csv(output/"paired_comparisons.csv", paired_rows(selected, config["replicates"]))
    save_csv(output/"fisher.csv", information)
    save_csv(output/"masks.csv", masks)
    save_csv(output/"landscapes.csv", landscapes)
    counts = dict(selected_fits=len(selected), optimizer_starts=len(starts),
                  independent_pools=len(config["families"])*len(config["regimes"])*len(config["pool_seeds"]),
                  invalid_selected=sum(not row["valid"] for row in selected),
                  invalid_starts=sum(not row["valid"] for row in starts),
                  selected_optimizer_non_success=sum(row["valid"] and not row["optimizer_success"] for row in selected),
                  selected_nonstationary=sum(row["valid"] and not row.get("stationary", False) for row in selected),
                  selected_boundary_count=sum(bool(json.loads(row["boundary_flags"])) for row in selected if row["valid"]),
                  selected_start_disagreements=sum(row.get("start_disagreement", False) for row in selected),
                  elapsed_seconds=time.perf_counter()-wall_start)
    dump_json(output/"completion.json", counts)
    print(json.dumps(counts, indent=2), flush=True)
    if len(selected) != config["expected_selected_fits"] or counts["invalid_selected"]:
        raise RuntimeError("study has missing/invalid selected fits; failures are retained in CSV")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--secondary", action="store_true")
    parser.add_argument("--graphs", nargs="+", choices=list(FAMILY_CODES))
    parser.add_argument("--pools", type=int)
    parser.add_argument("--replicates", nargs="+", type=int)
    args = parser.parse_args()
    try:
        config = configuration(smoke=args.smoke, secondary=args.secondary, graphs=args.graphs,
                               pools=args.pools, replicate_counts=args.replicates)
    except ValueError as error:
        parser.error(str(error))
    default = "results/identifiability_smoke" if args.smoke else (
        "results/identifiability_secondary" if args.secondary else "results/identifiability")
    run_study(args.output or Path(default), config)
