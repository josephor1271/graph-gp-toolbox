"""Sequential, bounded subprocess orchestration; imports no numerical stack."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from graphgp.benchmarking.cases import dense_eligible, primary_cases, smoke_case
from graphgp.benchmarking.report import compare_accuracy, flatten, save_csv, summarize, validate_records

TIMEOUT_SECONDS = 180


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def command_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment():
    sources = [*Path("src/graphgp").rglob("*.py"), *Path("experiments").glob("04_*.py")]
    return dict(source_commit=command_output(["git", "rev-parse", "HEAD"]), source_dirty=bool(command_output(["git", "status", "--porcelain"])),
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}, python=sys.version,
        platform=platform.platform(), cpu=command_output(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor(),
        logical_cores=os.cpu_count(), physical_cores=command_output(["sysctl", "-n", "hw.physicalcpu"]),
        versions={name: importlib.metadata.version(name) for name in ("torch", "gpytorch", "linear_operator", "numpy", "scipy", "networkx", "matplotlib")},
        command_line=sys.argv, timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        host_memory_limit_enforced=False, memory_note="No portable host memory ceiling imposed; per-worker wall timeout is enforced.",
        rss_note="Whole isolated child peak: macOS ru_maxrss bytes / 2^20; Linux KiB / 1024. Imports/allocator/setup included. No peak subtraction.")


def request(case, operation, backend, profile="balanced_probe", *, campaign="scaling", seed=301, repeat=0, width=0):
    if backend == "dense_exact":
        profile = "dense"
    return dict(run_id=f"{campaign}-{case.case_id}-q{case.q}-{operation}-{backend}-{profile}-s{seed}-r{repeat}-w{width}",
                case=case.to_dict(), operation=operation, backend=backend, profile=profile, campaign=campaign,
                probe_seed=seed, repeat=repeat, rhs_width=width)


def solver_requests(cases, smoke=False):
    result = []
    for case in cases:
        if case.n not in (16, 49, 100) or case.num_times != 8 or case.m != int(.25*case.n*8) or "N" not in case.groups:
            continue
        result.append(request(case, "reference", "dense_exact", campaign="solver"))
        profiles = ["exhaustive_tiny"] if smoke else (["fast_probe", "balanced_probe", "strict_probe"] if case.n < 100 else ["fast_probe", "balanced_probe"])
        if not smoke and case.n == 16:
            profiles += ["exhaustive_tiny"]
        for profile in profiles:
            seeds = [301] if smoke or profile == "exhaustive_tiny" else ([301, 302, 303] if case.n < 100 else [301, 302])
            for seed in seeds:
                for op in ("mll", "mean", "marginal", "solve"):
                    result.append(request(case, op, "sparse_chebyshev", profile, campaign="solver", seed=seed))
    if smoke and not result:
        case = cases[0]
        result.append(request(case, "reference", "dense_exact", campaign="solver"))
        result.extend(request(case, op, "sparse_chebyshev", "exhaustive_tiny", campaign="solver") for op in ("mll", "mean", "marginal", "solve"))
    return result


def scaling_requests(cases, smoke=False):
    result = []
    for case in cases:
        small = case.n <= 100 and case.num_times < 32
        for backend in (["dense_exact", "sparse_chebyshev"] if dense_eligible(case) else ["sparse_chebyshev"]):
            operations = ["mean", "marginal"]
            if set(case.groups) & {"N", "T", "M", "smoke"}:
                operations += ["mll"]
                if dense_eligible(case):
                    operations += ["train3"]
            if set(case.groups) & {"N", "T", "smoke"}:
                operations += ["construct"]
            operations += ["mvp"]
            for op in operations:
                repeats = 1 if smoke or not small else (3 if op in ("mvp", "construct") else 2)
                for repeat in range(repeats):
                    for width in ([1, 8] if op == "mvp" else [0]):
                        result.append(request(case, op, backend, repeat=repeat, width=width))
        if dense_eligible(case):
            result.append(request(case, "reference", "dense_exact"))
    return result


def execute(spec, output, *, timeout=TIMEOUT_SECONDS, runner=subprocess.run,
            worker_module="graphgp.benchmarking.worker"):
    if worker_module not in ("graphgp.benchmarking.worker", "graphgp.benchmarking.stage05_worker"):
        raise ValueError(f"unknown worker module: {worker_module}")
    run_id = spec["run_id"]
    spec_path, result_path = output/"workers"/(run_id+".spec.json"), output/"workers"/(run_id+".json")
    write_json(spec_path, spec)
    command = [sys.executable, "-m", worker_module, "--spec", str(spec_path), "--output", str(result_path)]
    start = time.perf_counter()
    try:
        child = runner(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        return {**spec, "status": "timeout", "child_wall_seconds": time.perf_counter()-start, "error": f"hard wall limit {timeout}s", "returncode": None}
    except OSError as error:
        return {**spec, "status": "launch_error", "child_wall_seconds": time.perf_counter()-start, "error": str(error), "returncode": None}
    (output/"workers"/(run_id+".log")).write_text(child.stdout+child.stderr)
    elapsed = time.perf_counter()-start
    if result_path.exists():
        try:
            row = json.loads(result_path.read_text())
            if not isinstance(row, dict) or row.get("run_id") != run_id or "status" not in row:
                raise ValueError("worker output does not match requested measurement")
        except (OSError, ValueError) as error:
            row = {**spec, "status": "invalid_output", "error": str(error)}
    else:
        status = "signal" if child.returncode < 0 else ("nonzero_exit" if child.returncode else "missing_output")
        row = {**spec, "status": status, "error": child.stderr[-2000:], "oom_indicator": "out of memory" in child.stderr.lower()}
    row.update(child_wall_seconds=elapsed, returncode=child.returncode)
    if child.returncode and row["status"] == "ok":
        row["status"] = "nonzero_exit"
    return row


def orchestrate(output, *, smoke=False, max_cases=None, solver_only=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a fresh output directory; existing runs are never overwritten")
    output.mkdir(parents=True)
    (output/"workers").mkdir()
    cases = [smoke_case()] if smoke else primary_cases()
    if max_cases is not None:
        if max_cases < 1:
            raise ValueError("max_cases must be positive")
        cases = cases[:max_cases]
    specs = solver_requests(cases, smoke)
    if not solver_only:
        specs += scaling_requests(cases, smoke)
    config = dict(study="smoke" if smoke else "fixed_full", solver_only=solver_only, debug_max_cases=max_cases,
        baseline_main="4f90ba5d64f7fe08f9f89697f3be06ce508c450a", cases=[c.to_dict() for c in cases],
        model=dict(heat_time=1.2, lengthscale=0.24, noise_variance=0.06, amplitude=1, mean=0, degree=32),
        profiles={"fast_probe": [8,16,1e-3,200], "balanced_probe": [16,32,1e-4,500], "strict_probe": [16,32,1e-6,500], "exhaustive_tiny": [32,32,1e-8,500]},
        profile_columns=["trace_samples", "lanczos_iterations", "cg_tolerance_and_eval_cg_tolerance", "max_cg_iterations"],
        fixed_settings=dict(max_preconditioner_size=0, terminate_cg_by_size=False, max_cholesky_size=0, max_eager_kernel_size=0, fast_pred_var=False),
        screening=dict(nll_abs=0.05, raw_gradient_atol=0.02, raw_gradient_rtol=0.20, posterior_mean_abs=0.01, latent_variance_abs=0.01, residual_fast=0.01, residual_other=0.001),
        probe_seed_labels=[301,302,303], master_seed=202604, probe_policy="SeedSequence-derived Torch seed; labels and effective seeds recorded; fixed common probes at all three-update evaluations",
        timeout_seconds=TIMEOUT_SECONDS, expected_rows=len(specs), worker_rule="one fresh sequential subprocess per operation/backend/profile/case/repeat",
        primary_profile="balanced_probe", warmup="exactly one for MVP only, then five products; no GP warmup or cache reuse",
        repeats="small construct/MVP 3; small GP 2; n>=225 or T32 1; smoke1", data="approximate P32(exp(-tau L/2)) Z chol(KT).T, independent noise; not an exact sampling oracle")
    write_json(output/"config.json", config)
    write_json(output/"environment.json", environment())
    write_json(output/"requests.json", specs)
    records = []
    for i, spec in enumerate(specs):
        row = execute(spec, output)
        records.append(row)
        # Crash-visible journal; measurements are never retried or resumed implicitly.
        with (output/"records.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
        print(f"{i+1}/{len(specs)} {spec['run_id']}: {row['status']} wall={row['child_wall_seconds']:.3f}s", flush=True)
    checks, accuracy = compare_accuracy(records)
    raw = [flatten(r) | ({"polynomial_mvp_max_abs_error": r["polynomial_mvp_max_abs_error"]} if "polynomial_mvp_max_abs_error" in r else {}) for r in records]
    summary = summarize(raw)
    completion = validate_records(specs, records)
    completion.update(solver_accuracy_rows=len(accuracy), solver_check_rows=len(checks),
                      profile_screening={name: dict(attempts=sum(r["profile"] == name and r.get("screening_scope") == "profile_seed" for r in accuracy),
                                                   failures=sum(r["profile"] == name and r.get("screening_scope") == "profile_seed" and r["screen_status"] != "pass" for r in accuracy)) for name in config["profiles"]},
                      profiling_gaps=[dict(run_id=r["run_id"], status=r["status"], error=r.get("error")) for r in records if r["status"] != "ok"])
    for name, rows in (("raw_runs.csv", raw), ("solver_checks.csv", checks), ("accuracy.csv", accuracy), ("summary.csv", summary)):
        save_csv(output/name, rows)
    write_json(output/"completion.json", completion)
    write_json(output/"records.json", records)
    if smoke and (any(r["status"] != "ok" for r in records) or
                  any(r["screening_scope"] == "profile_seed" and r["screen_status"] != "pass" for r in accuracy)):
        raise RuntimeError("Stage 04 smoke failed; retained outputs explain failures")
    print(json.dumps(completion, indent=2), flush=True)
    return completion
