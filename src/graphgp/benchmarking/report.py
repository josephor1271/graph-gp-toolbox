"""Standard-library-only Stage 04 accounting and independent result comparison."""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


def save_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v
                         for k, v in row.items()} for row in rows)


def flatten(record):
    row = {key: record.get(key) for key in ("run_id", "campaign", "operation", "backend", "profile", "probe_seed", "repeat", "rhs_width", "status",
        "child_wall_seconds", "elapsed_seconds", "import_seconds", "common_diagnostic_imports_preloaded", "data_setup_seconds", "model_build_seconds", "first_call_seconds", "forward_seconds", "backward_seconds",
        "rss_peak_mib", "rss_before_operation_mib", "memory_status", "error_type", "error", "returncode", "numerical_warning_count", "warnings",
        "nnz", "graph_scale", "settings", "fixed_settings", "effective_probe_seed", "operator", "solver", "measurement_seconds", "warmup_count", "data_fingerprint", "setup_warnings")}
    case = record["case"]
    row.update({k: case[k] for k in ("case_id", "n", "num_times", "m", "q", "groups")})
    row.update(nT=case["n"]*case["num_times"], observed_fraction=case["m"]/(case["n"]*case["num_times"]))
    row["q"] = record.get("measured_q", case["q"])
    result = record.get("result", {})
    for key in ("nll", "raw_gradients", "final_gradient_norm", "updates", "trajectory", "minimum_latent_variance", "minimum_observation_variance", "variance_returned"):
        if key in result:
            row[key] = result[key]
    return row


def summarize(rows):
    keys = ("campaign", "case_id", "n", "num_times", "nT", "m", "q", "groups", "operation", "backend", "profile", "rhs_width")
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(json.dumps(row.get(k), sort_keys=True) for k in keys)].append(row)
    output = []
    for grouped in groups.values():
        result = {k: grouped[0].get(k) for k in keys}
        success = [r for r in grouped if r["status"] == "ok"]
        result.update(attempted=len(grouped), success=len(success), timeout=sum(r["status"] == "timeout" for r in grouped),
                      failed=sum(r["status"] not in ("ok", "timeout") for r in grouped))
        for field, prefix in (("elapsed_seconds", "seconds"), ("rss_peak_mib", "rss_peak_mib")):
            values = [r[field] for r in success if r.get(field) is not None]
            if values:
                if not all(math.isfinite(v) and v > 0 for v in values):
                    raise ValueError(f"invalid successful {field}")
                result.update({prefix+"_median": statistics.median(values), prefix+"_minimum": min(values), prefix+"_maximum": max(values)})
        output.append(result)
    return output


def maximum_error(a, b):
    if isinstance(a, list):
        if len(a) != len(b):
            raise ValueError("reference shape mismatch")
        return max((maximum_error(x, y) for x, y in zip(a, b)), default=0.0)
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("nonfinite accuracy input")
    return abs(a-b)


def relative_column_errors(actual, exact):
    if (len(actual) != len(exact) or not actual
            or any(len(row) != 3 for row in (*actual, *exact))
            or any(not math.isfinite(value) for row in (*actual, *exact) for value in row)):
        raise ValueError("invalid exact-solve comparison shape or nonfinite values")
    return [math.sqrt(sum((a[j]-b[j])**2 for a, b in zip(actual, exact))) /
            max(math.sqrt(sum(b[j]**2 for b in exact)), 1e-15) for j in range(3)]


def compare_accuracy(records):
    """Never convert an absent/failed measurement to zero error or success."""
    refs = {(r["case"]["case_id"], r.get("measured_q", r["case"]["q"])): r["result"] for r in records
            if r["operation"] == "reference" and r["status"] == "ok"}
    grouped = defaultdict(dict)
    for r in records:
        if r["campaign"] == "solver" and r["backend"] == "sparse_chebyshev":
            grouped[(r["case"]["case_id"], r["profile"], r["probe_seed"])][r["operation"]] = r
    checks, accuracy = [], []
    for (case_id, profile, seed), operations in grouped.items():
        sample = next(iter(operations.values()))
        case = sample["case"]
        common = dict(case_id=case_id, n=case["n"], num_times=case["num_times"], m=case["m"], q=sample.get("measured_q", min(case["q"],32)),
                      profile=profile, probe_seed=seed, screening_scope="profile_seed")
        reference = refs.get((case_id, common["q"]))
        row, failures = dict(common), []
        if reference is None:
            failures.append("missing_exact_reference")
        for operation in ("mll", "mean", "marginal", "solve"):
            if operation not in operations or operations[operation]["status"] != "ok":
                failures.append(operation+"_not_successful")
        if reference is not None and operations.get("mll", {}).get("status") == "ok":
            mll = operations["mll"]["result"]
            row["nll_dense"] = reference["nll"]
            row["nll_sparse"] = mll["nll"]
            row["nll_abs_error"] = abs(mll["nll"]-reference["nll"])
            if not math.isfinite(row["nll_abs_error"]) or row["nll_abs_error"] > 0.05:
                failures.append("nll")
            for param in ("heat_time", "lengthscale"):
                exact = reference["raw_gradients"][param]
                error = abs(mll["raw_gradients"][param]-exact)
                row[param+"_raw_gradient_dense"] = exact
                row[param+"_raw_gradient_sparse"] = mll["raw_gradients"][param]
                row[param+"_raw_gradient_abs_error"] = error
                if not math.isfinite(error) or error > 0.02+0.20*abs(exact):
                    failures.append(param+"_gradient")
        for name, operation, output in (("posterior_mean", "mean", "mean"), ("latent_variance", "marginal", "latent_variance"),
                                         ("observation_variance", "marginal", "observation_variance")):
            if reference is not None and operations.get(operation, {}).get("status") == "ok":
                try:
                    error = maximum_error(operations[operation]["result"][output], reference[output])
                except (TypeError, ValueError):
                    failures.append(name+"_invalid")
                    continue
                row[name+"_max_abs_error"] = error
                if error > 0.01:
                    failures.append(name)
        solved = operations.get("solve")
        if solved is not None and solved["status"] == "ok":
            residuals = solved["result"]["relative_residual"]
            if len(residuals) != 3 or not all(math.isfinite(value) for value in residuals):
                failures.append("true_residual_invalid")
                residuals = [None] * 3
            else:
                row["maximum_relative_residual"] = max(residuals)
            errors = [None] * 3
            if reference is not None:
                try:
                    errors = relative_column_errors(solved["result"]["solutions"], reference["exact_solutions"])
                except (TypeError, ValueError):
                    failures.append("exact_solve_comparison_invalid")
            tolerance = 1e-2 if profile == "fast_probe" else 1e-3
            if residuals[0] is not None and max(residuals) > tolerance:
                failures.append("true_residual")
            for j, residual in enumerate(residuals):
                checks.append({**common, "rhs": ("train_y", "normal_1", "normal_2")[j], "requested_tolerance": solved["settings"]["cg_tolerance"],
                               "relative_residual": residual, "relative_solution_error": errors[j],
                               "warning_count": solved["numerical_warning_count"],
                               "status": "ok" if residual is not None else "invalid",
                               "reference_status": "available" if reference is not None else "missing"})
        else:
            for label in ("train_y", "normal_1", "normal_2"):
                checks.append({**common, "rhs": label, "status": "incomplete",
                               "operation_status": solved["status"] if solved else "missing", "error": list(failures)})
        row.update(screen_status="fail" if failures else "pass", screen_failures=failures)
        accuracy.append(row)
    accuracy.extend(_scaling_accuracy(records, refs))
    return checks, accuracy


def _scaling_accuracy(records, refs):
    """Report every fixed-point small-case measurement against its oracle.

    These per-operation checks include every timing repeat and both backends.
    They do not perform the across-seed/profile screening: in particular a
    passing case check alone cannot qualify an inference profile without the
    independent solve residuals required by ``profile_seed`` rows.
    """
    output = []
    for record in records:
        case = record["case"]
        if (record["campaign"] != "scaling"
                or record["operation"] not in ("mll", "mean", "marginal", "mvp")
                or not (case["n"] <= 100 and case["n"] * case["num_times"] <= 800
                        and case["m"] <= 400 and case["q"] <= 256)):
            continue
        q = record.get("measured_q", case["q"])
        row = dict(run_id=record["run_id"], case_id=case["case_id"], n=case["n"],
                   num_times=case["num_times"], m=case["m"], q=q,
                   backend=record["backend"], profile=record["profile"],
                   probe_seed=record["probe_seed"], repeat=record["repeat"],
                   operation=record["operation"], rhs_width=record["rhs_width"],
                   screening_scope="case_check", operation_status=record["status"])
        failures = []
        reference = refs.get((case["case_id"], q))
        if record["status"] != "ok":
            failures.append("operation_not_successful")
        if reference is None:
            failures.append("missing_exact_reference")
        if not failures:
            result = record["result"]
            try:
                if record["operation"] == "mll":
                    row.update(nll_dense=reference["nll"], nll_actual=result["nll"],
                               nll_abs_error=maximum_error(result["nll"], reference["nll"]))
                    if row["nll_abs_error"] > 0.05:
                        failures.append("nll")
                    for parameter in ("heat_time", "lengthscale"):
                        exact = reference["raw_gradients"][parameter]
                        actual = result["raw_gradients"][parameter]
                        error = maximum_error(actual, exact)
                        row.update({parameter+"_raw_gradient_dense": exact,
                                    parameter+"_raw_gradient_actual": actual,
                                    parameter+"_raw_gradient_abs_error": error})
                        if error > 0.02+0.20*abs(exact):
                            failures.append(parameter+"_gradient")
                elif record["operation"] in ("mean", "marginal"):
                    comparisons = [("posterior_mean", "mean")]
                    if record["operation"] == "marginal":
                        comparisons += [("latent_variance", "latent_variance"),
                                        ("observation_variance", "observation_variance")]
                    for name, field in comparisons:
                        error = maximum_error(result[field], reference[field])
                        row[name+"_max_abs_error"] = error
                        if error > 0.01:
                            failures.append(name)
                else:
                    exact = [values[:record["rhs_width"]] for values in reference["prior_mvp"]]
                    error = maximum_error(result["product"], exact)
                    row["kernel_mvp_max_abs_error"] = error
                    # This is polynomial error only for the polynomial backend;
                    # no CG/SLQ occurs in a bare prior product. Report the
                    # measured number without inventing a new acceptance gate.
                    if record["backend"] == "sparse_chebyshev":
                        row["polynomial_mvp_max_abs_error"] = error
                        record["polynomial_mvp_max_abs_error"] = error
            except (KeyError, TypeError, ValueError, OverflowError):
                failures.append("invalid_accuracy_input")
        row.update(screen_status="fail" if failures else "pass", screen_failures=failures)
        output.append(row)
    return output


def validate_records(specs, records):
    def measurement_key(row):
        return (row.get("campaign"), row["case"]["case_id"], row.get("operation"),
                row.get("backend"), row.get("profile"), row.get("probe_seed"),
                row.get("repeat"), row.get("rhs_width"))

    def check_finite(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                check_finite(item, path+"."+str(key))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                check_finite(item, path+f"[{index}]")
        elif isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError(f"nonfinite successful numeric field: {path}")

    expected = {s["run_id"] for s in specs}
    observed = [r["run_id"] for r in records]
    if (len(expected) != len(specs) or len(set(observed)) != len(observed)
            or len({measurement_key(s) for s in specs}) != len(specs)
            or len({measurement_key(r) for r in records}) != len(records)):
        raise ValueError("duplicate measurement keys")
    if set(observed) != expected:
        raise ValueError("missing or unexpected measurement rows")
    expected_by_id = {spec["run_id"]: spec for spec in specs}
    for r in records:
        if measurement_key(r) != measurement_key(expected_by_id[r["run_id"]]):
            raise ValueError("worker result does not match its requested measurement")
        if r["status"] == "ok":
            check_finite(r, r["run_id"])
            if r["elapsed_seconds"] <= 0:
                raise ValueError("invalid successful operation time")
            if r.get("rss_peak_mib") is not None and r["rss_peak_mib"] <= 0:
                raise ValueError("invalid successful peak RSS")
    return dict(expected_rows=len(specs), actual_rows=len(records), cases_attempted=sorted({r["case"]["case_id"] for r in records}),
                status_counts={s: sum(r["status"] == s for r in records) for s in sorted({r["status"] for r in records})},
                missing_rows=[], duplicate_keys=False)
