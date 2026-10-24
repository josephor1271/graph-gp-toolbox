"""Regenerate at most three Stage 03 figures from saved CSV evidence only.

No model is fit and no synthetic data is sampled here. The main predictive
comparison uses individual paired seeds, retains finite unfavorable outcomes,
and reports recorded failures. Parity and noise-sensitivity fits are separate.
The two prediction tasks share one latent/noisy realization per seed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np


MODEL_ORDER = ("graph_only", "time_only", "joint")
MODEL_LABELS = {"graph_only": "Graph only", "time_only": "Time only", "joint": "Graph × time"}
TASK_ORDER = ("interpolation", "forecast")
TASK_LABELS = {"interpolation": "Random-pair interpolation", "forecast": "Future forecasting"}
SEED_COLORS = {11: "#2474a9", 22: "#d47d21", 33: "#328653", 44: "#a6548c"}
BACKEND_COLORS = {"dense_exact": "#225d98", "sparse_chebyshev": "#d47427"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def _failed(row: dict[str, str]) -> bool:
    status = row.get("fit_status", "").lower()
    return any(word in status for word in ("fail", "invalid", "nonfinite", "error"))


def _style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _predictive_figure(rows: list[dict[str, str]], path: Path) -> Path:
    main = [row for row in rows if row.get("protocol") == "main"
            and row.get("model") in MODEL_ORDER]
    if not main:
        raise ValueError("metrics.csv contains no main predictive comparison rows")
    tasks = [task for task in TASK_ORDER if any(row.get("task") == task for row in main)]
    if not tasks:
        raise ValueError("No recognized prediction task in main metrics")
    seeds = sorted({int(row["seed"]) for row in main})
    fallback_colors = plt.get_cmap("tab10")
    colors = {seed: SEED_COLORS.get(seed, fallback_colors(index % 10))
              for index, seed in enumerate(seeds)}
    metrics = (("latent_rmse", "Latent RMSE (signal units) ↓"),
               ("observation_nlpd", "Observation NLPD (nats/target) ↓"),
               ("coverage95", "95% observation coverage (fraction)"))
    fig, axes = plt.subplots(len(tasks), 3, figsize=(13.5, 3.6 * len(tasks) + 1.4),
                             squeeze=False)
    for row_index, task in enumerate(tasks):
        panel = [row for row in main if row["task"] == task]
        cells: dict[tuple[int, str], dict[str, str]] = {}
        for row in panel:
            key = int(row["seed"]), row["model"]
            if key in cells:
                raise ValueError(f"Duplicate main seed/task/model metric row: {key}, {task}")
            cells[key] = row
        for ax, (metric, label) in zip(axes[row_index], metrics):
            nonfinite = 0
            for seed in seeds:
                values = []
                for model in MODEL_ORDER:
                    record = cells.get((seed, model))
                    value = _number(record, metric) if record is not None else np.nan
                    values.append(value)
                    nonfinite += int(record is not None and not np.isfinite(value))
                ax.plot(range(3), values, color=colors[seed], marker="o", markersize=5,
                        linewidth=1.1, alpha=0.85)
                for position, model in enumerate(MODEL_ORDER):
                    record = cells.get((seed, model))
                    if record is not None and _failed(record) and np.isfinite(values[position]):
                        ax.plot(position, values[position], marker="x", color="black",
                                markersize=9, markeredgewidth=1.3, linestyle="")
            if metric == "coverage95":
                ax.axhline(0.95, color="#555555", linewidth=0.9, linestyle="--")
                low, high = ax.get_ylim()
                ax.set_ylim(min(low, 0), max(high, 1.02))
            failed = sum(_failed(row) for row in panel)
            ax.set_title(TASK_LABELS[task], fontsize=11, pad=25)
            ax.set_ylabel(label)
            ax.set_xticks(range(3), [MODEL_LABELS[model] for model in MODEL_ORDER], fontsize=10)
            ax.text(0.5, 1.015,
                    f"Recorded fits: {len(panel)}; failed: {failed}; nonfinite metric: {nonfinite}",
                    transform=ax.transAxes, fontsize=7.7, va="bottom", ha="center")
            _style(ax)
    handles = [Line2D([], [], color=colors[seed], marker="o", linewidth=1.1,
                      label=f"Seed {seed}") for seed in seeds]
    if any(_failed(row) for row in main):
        handles.append(Line2D([], [], color="black", marker="x", linestyle="",
                              label="Failed fit with finite recorded metric"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.95),
               ncol=min(len(handles), 5), frameon=False, fontsize=9)
    fig.suptitle("Known-noise predictive comparison: individual paired seeds", fontsize=14, y=0.995)
    fig.text(0.5, 0.015,
             f"Main protocol only; {len(seeds)} recorded independent field(s), with the same field reused across tasks. "
             "Lines connect each seed's model outcomes.\n"
             "All finite metrics retained; missing/nonfinite points leave gaps. "
             "No confidence intervals; sparse parity and learned-noise diagnostics are separate.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.075, 1, 0.895))
    return _save(fig, path)


def _training_figure(rows: list[dict[str, str]], path: Path) -> Path | None:
    selected = [row for row in rows if row.get("protocol") == "parity"
                and row.get("task") == "interpolation" and int(row["seed"]) == 11
                and row.get("model") == "joint"]
    if not selected:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    panels = (("nll", "Training NLL (nats/observation)"),
              ("heat_time", "Heat time τ"), ("lengthscale", "Temporal lengthscale ℓ"))
    present_backends = []
    maximum_step = 0
    for backend in ("dense_exact", "sparse_chebyshev"):
        backend_rows = sorted((row for row in selected if row.get("backend") == backend),
                              key=lambda row: int(row["step"]))
        if not backend_rows:
            continue
        fit_ids = {row["fit_id"] for row in backend_rows}
        if len(fit_ids) != 1:
            raise ValueError(f"Expected one seed-11 interpolation parity fit for {backend}")
        steps = [int(row["step"]) for row in backend_rows]
        if len(set(steps)) != len(steps):
            raise ValueError(f"Duplicate trajectory steps for {backend}")
        maximum_step = max(maximum_step, max(steps))
        present_backends.append(backend)
        for ax, (key, label) in zip(axes, panels):
            values = [_number(row, key) for row in backend_rows]
            ax.plot(steps, values, color=BACKEND_COLORS[backend],
                    linestyle="-" if backend == "dense_exact" else "--",
                    linewidth=1.8, label="Dense exact" if backend == "dense_exact" else "Sparse Chebyshev")
            if not np.isfinite(values).all():
                ax.text(0.02, 0.02, "Nonfinite saved values create gaps", transform=ax.transAxes,
                        fontsize=8, color="#9c3333")
    for ax, (_, label) in zip(axes, panels):
        ax.set_xlabel("Adam update step")
        ax.set_ylabel(label)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Matched dense/sparse training: seed 11, interpolation", fontsize=14)
    missing = ({"dense_exact", "sparse_chebyshev"} - set(present_backends))
    missing_text = f" Missing backend(s): {', '.join(sorted(missing))}." if missing else ""
    fig.text(0.5, 0.015,
             f"Separate parity protocol: saved steps through {maximum_step} (planned budget: 40 updates); "
             "main predictive comparison uses 160 updates.\n"
             "Small-problem numerical comparison with exhaustive trace probes; no speedup claim." + missing_text,
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.10, 1, 0.92))
    return _save(fig, path)


def _observed(value: str) -> bool:
    if value.lower() in {"true", "1", "1.0"}:
        return True
    if value.lower() in {"false", "0", "0.0"}:
        return False
    raise ValueError(f"Unrecognized observed-mask value {value!r}")


def _signal_figure(rows: list[dict[str, str]], path: Path) -> Path | None:
    if not rows:
        return None
    vertices = sorted({int(row["vertex"]) for row in rows})
    times = sorted({int(row["time_index"]) for row in rows})
    if len(times) < 2:
        raise ValueError("signal.csv requires at least two declared times")
    if vertices != list(range(len(vertices))) or times != list(range(len(times))):
        raise ValueError("signal.csv must contain the complete zero-based graph/time grid")
    if len(rows) != len(vertices) * len(times):
        raise ValueError("signal.csv must contain one row per full-grid coordinate")
    shape = len(vertices), len(times)
    truth, mean, observed = (np.full(shape, np.nan) for _ in range(3))
    physical_times: dict[int, float] = {}
    seen = set()
    for row in rows:
        coordinate = int(row["vertex"]), int(row["time_index"])
        if coordinate in seen:
            raise ValueError(f"Duplicate signal.csv coordinate {coordinate}")
        seen.add(coordinate)
        truth[coordinate] = _number(row, "true_latent")
        mean[coordinate] = _number(row, "posterior_mean")
        observed[coordinate] = _observed(row["observed"])
        physical = _number(row, "physical_time")
        index = coordinate[1]
        if index in physical_times and physical_times[index] != physical:
            raise ValueError("Inconsistent physical_time for a declared time index")
        physical_times[index] = physical
    if not np.isfinite(truth).all() or not np.isfinite(list(physical_times.values())).all():
        raise ValueError("Signal truth and physical times must be finite")
    expected_times = np.arange(len(times), dtype=np.float64) / (len(times) - 1)
    if not np.allclose([physical_times[index] for index in times], expected_times,
                       rtol=0, atol=1e-14):
        raise ValueError("Signal physical times must use the full normalized declared grid")
    signal_values = np.concatenate((truth.reshape(-1), mean[np.isfinite(mean)]))
    low, high = float(signal_values.min()), float(signal_values.max())
    if low == high:
        low, high = low - 0.5, high + 0.5
    norm = Normalize(vmin=low, vmax=high)
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.3), constrained_layout=True)
    image = None
    for ax, values, title in zip(axes[:2], (truth, mean), ("True latent field f", "Graph × time posterior mean")):
        image = ax.imshow(np.ma.masked_invalid(values), aspect="auto", origin="lower",
                          interpolation="nearest", cmap="coolwarm", norm=norm)
        ax.set_title(title, fontsize=11)
    fig.colorbar(image, ax=axes[:2], label="Signal units (shared scale)", shrink=0.8)
    mask_image = axes[2].imshow(observed, aspect="auto", origin="lower", interpolation="nearest",
                                cmap=ListedColormap(["#ececec", "#293f55"]), vmin=0, vmax=1)
    axes[2].set_title(f"Training mask ({int(observed.sum())} observed pairs)", fontsize=11)
    colorbar = fig.colorbar(mask_image, ax=axes[2], ticks=[0, 1], shrink=0.8)
    colorbar.ax.set_yticklabels(["Held out", "Observed"])
    ticks = np.unique(np.linspace(0, len(times) - 1, min(4, len(times)), dtype=int))
    labels = [f"j={index}\nt={physical_times[index]:.2f}" for index in ticks]
    for ax in axes:
        ax.set_xticks(ticks, labels, fontsize=9)
        ax.set_xlabel("Declared time index j / normalized physical time t")
        ax.set_ylabel("Vertex index")
        ax.set_yticks(np.unique(np.linspace(0, len(vertices) - 1, min(6, len(vertices)), dtype=int)))
    missing = int((~np.isfinite(mean)).sum())
    failure_text = f"; {missing} nonfinite posterior means shown blank" if missing else ""
    fig.suptitle("Preregistered illustration: seed 11, interpolation, main joint dense fit\n"
                 "One realization; predictions include training and held-out pairs" + failure_text,
                 fontsize=13)
    return _save(fig, path)


def plot_all(input_dir: Path | str, output_dir: Path | str) -> list[Path]:
    """Regenerate available figures from CSVs and return their image paths.

    ``metrics.csv`` is required. Optional ``trajectories.csv`` and ``signal.csv``
    enable the fixed parity and illustrative panels. This also supports a smoke
    directory containing fewer seeds/tasks/models, without implying a full study.
    """
    source, destination = Path(input_dir), Path(output_dir)
    metrics = _read_csv(source / "metrics.csv")
    if not metrics:
        raise ValueError(f"No metric rows in {source / 'metrics.csv'}")
    destination.mkdir(parents=True, exist_ok=True)
    paths = [_predictive_figure(metrics, destination / "graph_time_predictive.png")]
    for name, function, filename in (
        ("trajectories.csv", _training_figure, "graph_time_training.png"),
        ("signal.csv", _signal_figure, "graph_time_signal.png"),
    ):
        result = function(_read_csv(source / name), destination / filename)
        if result is not None:
            paths.append(result)
        else:
            print(f"No preregistered {name} rows; skipped {filename}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/graph_time"))
    parser.add_argument("--output", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    for path in plot_all(args.input, args.output):
        print(path)


if __name__ == "__main__":
    main()
