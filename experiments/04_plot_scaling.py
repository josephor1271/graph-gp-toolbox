#!/usr/bin/env python3
"""Regenerate Stage 04 figures from saved CSVs only; never sample or fit a GP."""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {"dense_exact": "#3366a6", "sparse_chebyshev": "#d17822"}
LABELS = {"dense_exact": "Dense exact", "sparse_chebyshev": "Sparse balanced"}
PROFILES = ("fast_probe", "balanced_probe", "strict_probe", "exhaustive_tiny")
PROFILE_COLORS = ("#649451", "#d17822", "#3366a6", "#855a93")
OPERATIONS = (("construct", "Construction"), ("mvp", "Prior MVP, RHS width 8"),
              ("mll", "MLL + backward"), ("train3", "Three Adam updates"),
              ("mean", "Mean only"), ("marginal", "Exact target marginals"))


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def integer(row, key):
    value = number(row, key)
    return int(value) if value is not None else None


def primary(row):
    return (row.get("campaign") == "scaling"
            and row.get("backend") in COLORS
            and row.get("profile") in ("balanced_probe", "dense"))


def dense_eligible(row):
    n, times, m, q = (integer(row, name) for name in ("n", "num_times", "m", "q"))
    return (None not in (n, times, m, q) and n <= 100 and n * times <= 800
            and m <= 400 and q <= 256)


def n_sweep(row):
    n = integer(row, "n")
    return (n in (16, 49, 100, 225, 400) and integer(row, "num_times") == 8
            and integer(row, "m") == n * 8 // 4 and integer(row, "q") == 64)


def failure_count(row):
    return sum(integer(row, name) or 0 for name in ("timeout", "failed"))


def failure_marker(ax, row, x, label=True):
    """Place failed attempts in axes coordinates, never at a fabricated time."""
    count = failure_count(row)
    if not count:
        return
    color = COLORS[row["backend"]]
    height = 0.04 if row["backend"] == "dense_exact" else 0.12
    ax.plot(x, height, marker="x", color=color, transform=ax.get_xaxis_transform(),
            clip_on=False, markersize=6)
    if label:
        timeouts = integer(row, "timeout") or 0
        text = f"{count} {'timeout' if timeouts == count else 'failed/timeout'}"
        ax.annotate(text, xy=(x, height), xycoords=ax.get_xaxis_transform(),
                    xytext=(3, 3), textcoords="offset points", fontsize=6, color=color)


def draw_times(ax, rows, *, x_name, label, color, marker="o", linestyle="-"):
    valid = [(number(row, x_name), number(row, "seconds_median"), row) for row in rows]
    valid = sorted((x, y, row) for x, y, row in valid
                   if x is not None and y is not None and y > 0 and (integer(row, "success") or 0))
    if valid:
        xs, ys = [item[0] for item in valid], [item[1] for item in valid]
        lower = [max(0, y - (number(row, "seconds_minimum") or number(row, "seconds_min") or y))
                 for _, y, row in valid]
        upper = [max(0, (number(row, "seconds_maximum") or number(row, "seconds_max") or y) - y)
                 for _, y, row in valid]
        ax.errorbar(xs, ys, yerr=[lower, upper], label=label, color=color,
                    marker=marker, linestyle=linestyle, linewidth=1.2, capsize=2, markersize=4)
    for row in rows:
        x = number(row, x_name)
        if x is not None:
            failure_marker(ax, row, x)


def save(fig, output, name):
    fig.savefig(output / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def runtime_plot(summary, output):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.3), constrained_layout=True)
    for ax, (operation, title) in zip(axes.flat, OPERATIONS):
        selected = [row for row in summary if primary(row) and n_sweep(row)
                    and row.get("operation") == operation
                    and (operation != "mvp" or integer(row, "rhs_width") == 8)]
        for backend in COLORS:
            matching = [row for row in selected if row["backend"] == backend]
            paired = [row for row in matching if dense_eligible(row)]
            sparse_only = [row for row in matching if not dense_eligible(row)]
            draw_times(ax, paired, x_name="m", label=LABELS[backend], color=COLORS[backend])
            draw_times(ax, sparse_only, x_name="m", label="Sparse only; dense ineligible",
                       color=COLORS[backend], marker="^", linestyle="none")
        ax.axvline(325, color="0.65", linestyle=":", linewidth=0.8)
        ax.set(title=title, xlabel="Observed pairs m (N sweep)", ylabel="Operation time (s)",
               yscale="log", xlim=(0, 850))
        ax.grid(alpha=0.2)
        if operation == "train3":
            ax.text(.98, .02, "Large sizes: not requested", ha="right",
                    transform=ax.transAxes, fontsize=7, color="0.35")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=6, loc="best")
    fig.suptitle("Fixed T=8, observed fraction=0.25, q=64; medians and timing ranges\n"
                 "Separate workers; crosses: failures. Accuracy qualification is reported separately.", fontsize=11)
    save(fig, output, "scaling_stage04_runtime.png")


def memory_plot(summary, output):
    rows = [row for row in summary if primary(row)]
    case_data = {}
    for row in rows:
        case_data[row["case_id"]] = row
    cases = sorted(case_data, key=lambda name: tuple(integer(case_data[name], key) or 0
                                                    for key in ("n", "num_times", "m", "q")))
    positions = {case: index for index, case in enumerate(cases)}
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6), sharex=True, constrained_layout=True)
    for ax, operation in zip(axes, ("mll", "marginal")):
        for backend in COLORS:
            matching = [row for row in rows if row.get("operation") == operation
                        and row["backend"] == backend]
            xs, ys = [], []
            for row in matching:
                x = positions[row["case_id"]] + (-0.12 if backend == "dense_exact" else 0.12)
                peak = number(row, "rss_peak_mib_median")
                if peak is not None and peak > 0 and (integer(row, "success") or 0):
                    xs.append(x)
                    ys.append(peak)
                failure_marker(ax, row, x, label=False)
            ax.scatter(xs, ys, color=COLORS[backend], s=24, label=LABELS[backend])
        ax.set(title="MLL + backward" if operation == "mll" else "Exact target marginals",
               ylabel="Whole worker peak RSS (MiB)")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    for index, case in enumerate(cases):
        if not dense_eligible(case_data[case]):
            for ax in axes:
                ax.axvspan(index - .45, index + .45, color="0.9", zorder=0)
    labels = ["n{n}/T{num_times}\nm{m}/q{q}".format(**case_data[name]) for name in cases]
    axes[-1].set_xticks(range(len(cases)), labels, fontsize=7, rotation=45, ha="right")
    fig.suptitle("Worker RSS includes imports, model/data preparation and allocator effects\n"
                 "macOS: bytes / 2²⁰; Linux: KiB / 2¹⁰. Gray: dense ineligible; ×: failure; blank: unrequested/unavailable.", fontsize=10)
    save(fig, output, "scaling_stage04_memory.png")


def solver_plot(checks, accuracy, output):
    # Primary per-case checks share the evidence table but are not probe-seed screens.
    accuracy = [row for row in accuracy
                if row.get("screening_scope", "profile_seed") == "profile_seed"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), constrained_layout=True)
    residual_ax, nll_ax, gradient_ax, posterior_ax = axes.flat
    metric_values = {nll_ax: [0.05], gradient_ax: [], posterior_ax: [0.01]}
    for index, (profile, color) in enumerate(zip(PROFILES, PROFILE_COLORS)):
        matching = [row for row in checks if row.get("profile") == profile]
        for warned, marker in ((False, "o"), (True, "x")):
            values = [(number(row, "requested_tolerance"), number(row, "relative_residual"))
                      for row in matching if bool(integer(row, "warning_count") or 0) == warned]
            values = [(x, y) for x, y in values if x is not None and x > 0 and y is not None and y > 0]
            if values:
                residual_ax.scatter([x for x, _ in values], [y for _, y in values],
                                    color=color, marker=marker, s=25,
                                    label=profile + ("; warning" if warned else ""), alpha=.7)
        matching = [row for row in accuracy if row.get("profile") == profile]
        metrics = ((nll_ax, "nll_abs_error"),
                   (gradient_ax, "heat_time_raw_gradient_abs_error"),
                   (gradient_ax, "lengthscale_raw_gradient_abs_error"),
                   (posterior_ax, "posterior_mean_max_abs_error"),
                   (posterior_ax, "latent_variance_max_abs_error"))
        for axis, metric in metrics:
            values = [(row, number(row, metric)) for row in matching]
            valid = [(row, value) for row, value in values if value is not None]
            for j, (row, value) in enumerate(valid):
                offset = 0.24 * (j / max(len(valid) - 1, 1) - .5)
                metric_values[axis].append(value)
                axis.scatter(index + offset, value, marker="x" if row.get("screen_status") == "fail" else "o",
                             s=22, color=color, alpha=.7)
    residual_ax.plot([1e-8, 1e-3], [1e-8, 1e-3], linestyle=":", color="0.4", label="Measured = requested")
    residual_ax.set(xscale="log", yscale="log", xlabel="Requested CG tolerance",
                    ylabel="Computed ‖Ax−b‖₂ / ‖b‖₂", title="Fresh noisy-covariance residual; all RHS/seeds")
    residual_ax.legend(fontsize=6, loc="best")
    nll_ax.set(title="NLL error per observed point; all probe seeds", ylabel="Absolute error")
    nll_ax.axhline(.05, color="0.4", linestyle=":", label="Screen: 0.05")
    nll_ax.legend(fontsize=7)
    gradient_ax.set(title="Both raw parameter gradient errors", ylabel="Absolute error")
    gradient_ax.text(.02, .98, "Screen: 0.02 + 0.20 × |dense gradient|", va="top",
                     transform=gradient_ax.transAxes, fontsize=7)
    posterior_ax.set(title="Mean and latent-variance errors", ylabel="Maximum absolute error")
    posterior_ax.axhline(.01, color="0.4", linestyle=":", label="Screen: 0.01")
    posterior_ax.legend(fontsize=7)
    for ax in (nll_ax, gradient_ax, posterior_ax):
        ax.set_xticks(range(4), [name.replace("_", "\n") for name in PROFILES], fontsize=8)
        positive = [value for value in metric_values[ax] if value > 0]
        if positive:
            ax.set_yscale("symlog", linthresh=min(positive) / 10)
            ax.set_ylim(0, max(positive) * 1.5)
        else:
            ax.set_ylim(0, 1)
            ax.text(.5, .5, "No positive error measurements", ha="center", transform=ax.transAxes)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    missing = sum(any(number(row, metric) is None for metric in
                      ("nll_abs_error", "posterior_mean_max_abs_error", "maximum_relative_residual"))
                  for row in accuracy)
    passed = sum(row.get("screen_status") == "pass" for row in accuracy)
    fig.suptitle(f"Fixed hyperparameters; {passed}/{len(accuracy)} screens pass; {missing} screens have missing metrics\n"
                 "All seeds; ×: warnings (residual) or failed screens (errors). Exhaustive: tiny only.", fontsize=10)
    save(fig, output, "scaling_stage04_solver.png")


def prediction_plot(summary, output):
    rows = [row for row in summary if primary(row) and integer(row, "n") == 100
            and integer(row, "num_times") == 8 and integer(row, "m") == 200
            and integer(row, "q") in (16, 64, 256) and row.get("operation") in ("mean", "marginal")]
    fig, (timing, ratio) = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    for backend in COLORS:
        for operation, style in (("mean", "--"), ("marginal", "-")):
            matching = [row for row in rows if row["backend"] == backend and row["operation"] == operation]
            draw_times(timing, matching, x_name="q", color=COLORS[backend], linestyle=style,
                       marker="o" if operation == "mean" else "s",
                       label=LABELS[backend] + (": mean only" if operation == "mean" else ": marginals"))
        lookup = {(integer(row, "q"), row["operation"]): row for row in rows if row["backend"] == backend}
        valid = []
        for q in (16, 64, 256):
            mean, marginal = lookup.get((q, "mean"), {}), lookup.get((q, "marginal"), {})
            a, b = number(mean, "seconds_median"), number(marginal, "seconds_median")
            if (a is not None and a > 0 and b is not None and b > 0
                    and (integer(mean, "success") or 0) > 0 and (integer(marginal, "success") or 0) > 0):
                valid.append((q, b / a))
        if valid:
            ratio.plot([x for x, _ in valid], [y for _, y in valid], marker="o",
                       color=COLORS[backend], label=LABELS[backend])
    timing.set(yscale="log", xlabel="Requested targets q", ylabel="Operation time (s)",
               title="Cold prediction; medians and timing ranges")
    ratio.set(xlabel="Requested targets q", ylabel="Marginal / mean-only median time", title="Matched prediction cost ratio")
    for ax in (timing, ratio):
        ax.set_xticks((16, 64, 256))
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle("Same n=100, T=8, m=200; chunk size 64; independent workers per route\n"
                 "Crosses mark failed attempts. Ratios require two finite successful timings.", fontsize=10)
    save(fig, output, "scaling_stage04_prediction.png")


def plot_all(input_dir, output_dir):
    source, output = Path(input_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = read_csv(source / "summary.csv")
    checks = read_csv(source / "solver_checks.csv")
    accuracy = read_csv(source / "accuracy.csv")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False}):
        runtime_plot(summary, output)
        memory_plot(summary, output)
        solver_plot(checks, accuracy, output)
        prediction_plot(summary, output)
    return [output / f"scaling_stage04_{name}.png" for name in ("runtime", "memory", "solver", "prediction")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("docs/evidence/scaling_stage04"))
    parser.add_argument("--output", type=Path, default=Path("docs/figures"))
    arguments = parser.parse_args()
    for path in plot_all(arguments.input, arguments.output):
        print(path)
