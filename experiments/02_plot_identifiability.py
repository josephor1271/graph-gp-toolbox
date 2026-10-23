"""Plot the Stage 02 study using saved CSVs only; never fit or sample a model.

Every distribution uses one selected fit per independent pool and condition.
Numerically valid boundary/nonconverged fits remain in the distributions. The
plots exclude explicitly invalid rows and nonfinite values, and report counts.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogFormatterSciNotation, LogLocator
import numpy as np


FAMILY_ORDER = {"path": 0, "grid": 1, "irregular": 2}
COLORS = {"joint": "#2563a6", "noise_known": "#ce6a22"}
MODE_LABELS = {"joint": "joint fit", "noise_known": "noise fixed to truth"}
DESIGN_LABELS = {"partial": "partial observations", "full": "full observations"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def _regime(row: dict[str, str]) -> str:
    return row.get("regime", "")


def _design(row: dict[str, str]) -> str:
    if row.get("observation_design"):
        return row["observation_design"]
    return "full" if _number(row, "observed_fraction") == 1.0 else "partial"


def _valid(row: dict[str, str]) -> bool:
    return str(row.get("valid", "True")).lower() not in {"false", "0"}


def _sort_key(key: tuple[str, str]) -> tuple[int, str, str]:
    return FAMILY_ORDER.get(key[0], 99), key[0], key[1]


def _save(fig: plt.Figure, output: Path) -> Path:
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def _error_figure(
    rows: list[dict[str, str]], output: Path, parameter: str, joint_only: bool = False
) -> Path:
    rows = [row for row in rows if not joint_only or row["fit_mode"] == "joint"]
    panel_keys = sorted({(row["family"], _regime(row)) for row in rows}, key=_sort_key)
    if not panel_keys:
        raise ValueError("No selected fit rows are available for an error figure")
    ncols = min(2, len(panel_keys))
    nrows = (len(panel_keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(12, 3.0 * nrows + 1.6), squeeze=False, sharey=True
    )
    budgets = sorted({int(row["replicates"]) for row in rows})
    series = [
        (mode, design)
        for mode in ("joint", "noise_known")
        for design in ("partial", "full")
        if any(row["fit_mode"] == mode and _design(row) == design for row in rows)
    ]
    width = 0.68 / max(len(series), 1)
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * width
    counts: list[int] = []
    retained = 0
    for ax, panel_key in zip(axes.flat, panel_keys):
        panel = [row for row in rows if (row["family"], _regime(row)) == panel_key]
        first = panel[0]
        for index, (mode, design) in enumerate(series):
            for budget_index, budget in enumerate(budgets):
                cell = [
                    row for row in panel
                    if row["fit_mode"] == mode and _design(row) == design
                    and int(row["replicates"]) == budget
                ]
                values = [
                    100 * abs(_number(row, f"fit_{parameter}") - _number(row, f"true_{parameter}"))
                    / _number(row, f"true_{parameter}")
                    for row in cell if _valid(row)
                    and np.isfinite(_number(row, f"fit_{parameter}"))
                    and np.isfinite(_number(row, f"true_{parameter}"))
                    and _number(row, f"true_{parameter}") > 0
                ]
                if not cell:
                    continue
                counts.append(len(values))
                retained += len(values)
                if not values:
                    continue
                box = ax.boxplot(
                    [values], positions=[budget_index + offsets[index]], widths=width * 0.85,
                    whis=(10, 90), patch_artist=True, manage_ticks=False,
                    medianprops={"color": "black", "linewidth": 1.2},
                    boxprops={"linewidth": 0.9}, whiskerprops={"linewidth": 0.9},
                    capprops={"linewidth": 0.9},
                    flierprops={"marker": ".", "markersize": 3.5, "alpha": 0.65},
                )
                box["boxes"][0].set_facecolor(COLORS[mode])
                box["boxes"][0].set_alpha(0.65)
                box["boxes"][0].set_hatch("///" if design == "full" else "")
                for item in (*box["whiskers"], *box["caps"], *box["fliers"]):
                    item.set_color(COLORS[mode])
        tau = _number(first, "true_tau")
        noise = _number(first, "true_noise_variance")
        ax.set_title(f"{panel_key[0]} · n={first['n']} · τ*={tau:g}, σ²*={noise:g}", fontsize=11)
        ax.set_xticks(range(len(budgets)), [str(budget) for budget in budgets])
        ax.set_xlabel("Independent realizations R")
        ax.set_yscale("symlog", linthresh=10, linscale=0.7)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.22, which="both")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("Absolute relative error (%)\nlinear to 10%, log above")
    for ax in list(axes.flat)[len(panel_keys):]:
        ax.set_visible(False)
    legend = [
        Patch(facecolor=COLORS[mode], alpha=0.65, edgecolor="black",
              hatch="///" if design == "full" else "",
              label=f"{MODE_LABELS[mode]} · {DESIGN_LABELS[design]}")
        for mode, design in series
    ]
    label = "Heat-time" if parameter == "tau" else "Noise-variance (joint fit only)"
    fig.suptitle(f"{label} estimation error across independent pools", fontsize=14, y=0.99)
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, fontsize=9)
    count_text = str(counts[0]) if counts and len(set(counts)) == 1 else f"{min(counts, default=0)}–{max(counts, default=0)}"
    fig.text(
        0.5, 0.012,
        f"Boxes: quartiles; whiskers: 10th–90th percentiles; dots: remaining extremes. "
        f"Valid pools/cell: {count_text}.\n"
        f"{retained}/{len(rows)} selected fits plotted; valid boundary and nonconverged fits retained. "
        "R budgets use paired pool seeds, not independent samples across R.",
        ha="center", fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    return _save(fig, output)


def _likelihood_figure(rows: list[dict[str, str]], output: Path) -> Path:
    # The runner records a scenario chosen before fitting. If a file contains
    # more than one scenario, its first recorded scenario is used, never a
    # selection based on fitted errors or the appearance of its landscape.
    metadata = ("family", "n", "regime", "observation_design", "pool_seed", "replicates")
    key = tuple(rows[0].get(name, "") for name in metadata)
    rows = [row for row in rows if tuple(row.get(name, "") for name in metadata) == key]
    taus = np.asarray(sorted({_number(row, "grid_tau") for row in rows}))
    noises = np.asarray(sorted({_number(row, "grid_noise_variance") for row in rows}))
    if len(taus) < 2 or len(noises) < 2:
        raise ValueError("Likelihood landscape needs at least two tau and noise values")
    tau_index = {value: index for index, value in enumerate(taus)}
    noise_index = {value: index for index, value in enumerate(noises)}
    surfaces = [np.full((len(noises), len(taus)), np.nan) for _ in range(2)]
    for row in rows:
        index = noise_index[_number(row, "grid_noise_variance")], tau_index[_number(row, "grid_tau")]
        surfaces[0][index] = _number(row, "sample_nll_per_observation")
        surfaces[1][index] = _number(row, "expected_excess_nll")
    surfaces[0] -= np.nanmin(surfaces[0])
    first = rows[0]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), constrained_layout=True)
    titles = ["Finite-sample NLL minus grid minimum", "Expected excess NLL / KL per observed node"]
    for ax, surface, title in zip(axes, surfaces, titles):
        minimum = min(0.0, float(np.nanmin(surface)))
        maximum = float(np.nanmax(surface))
        # A shared square-root color transform exposes shallow valleys without
        # truncating either surface. Each colorbar preserves its own NLL units.
        mesh = ax.pcolormesh(
            taus, noises, surface, shading="auto", cmap="viridis",
            norm=PowerNorm(gamma=0.5, vmin=minimum, vmax=maximum if maximum > minimum else minimum + 1),
        )
        levels = [value for value in (0.001, 0.01, 0.1, 1.0) if minimum < value < maximum]
        if levels:
            contours = ax.contour(taus, noises, surface, levels=levels, colors="white", linewidths=0.6, alpha=0.7)
            ax.clabel(contours, fmt="%g", fontsize=7)
        ax.scatter(_number(first, "true_tau"), _number(first, "true_noise_variance"),
                   marker="*", s=180, facecolor="white", edgecolor="black", linewidth=1.0, zorder=5, clip_on=False)
        if np.isfinite(_number(first, "fit_tau")) and np.isfinite(_number(first, "fit_noise_variance")):
            ax.scatter(_number(first, "fit_tau"), _number(first, "fit_noise_variance"),
                       marker="X", s=90, facecolor="#f28e2b", edgecolor="black", linewidth=0.8, zorder=6, clip_on=False)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Heat time τ (log axis)")
        ax.set_ylabel("Observation-noise variance σ² (log axis)")
        ax.set_title(title, fontsize=11)
        fig.colorbar(mesh, ax=ax, label="NLL per observed node per realization", shrink=0.85)
    fig.legend(handles=[
        Line2D([], [], marker="*", linestyle="", markerfacecolor="white", markeredgecolor="black", markersize=11, label="generating truth"),
        Line2D([], [], marker="X", linestyle="", markerfacecolor="#f28e2b", markeredgecolor="black", markersize=8, label="selected joint fit"),
    ], loc="outside lower center", ncol=2, fontsize=9)
    fig.suptitle(
        f"Recorded scenario: {first['family']}, n={first['n']}, regime {first['regime']}, "
        f"{_design(first)}, pool {first['pool_seed']}, R={first['replicates']}\n"
        "Square-root color scales; full saved ranges shown; finite-sample optimum need not be truth",
        fontsize=11,
    )
    return _save(fig, output)


def _fisher_figure(rows: list[dict[str, str]], output: Path) -> Path:
    # Different R rows repeat the same fixed graph/mask diagnostic. Retain the
    # smallest recorded R for each mask, then divide total information by R.
    masks: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = (row["family"], row["n"], _regime(row), _design(row), row.get("pool_seed", ""))
        if key not in masks or int(row["replicates"]) < int(masks[key]["replicates"]):
            masks[key] = row
    unique = list(masks.values())
    keys = sorted(
        {(row["family"], _regime(row), _design(row)) for row in unique},
        key=lambda key: (*_sort_key(key[:2]), 0 if key[2] == "partial" else 1),
    )
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.6), sharex=True)
    labels = []
    singular = 0
    for position, key in enumerate(keys):
        group = [row for row in unique if (row["family"], _regime(row), _design(row)) == key]
        labels.append(f"{key[0]}\nregime {key[1]} · {key[2]}")
        color = "#2563a6" if key[2] == "partial" else "#ce6a22"
        values = [
            [_number(row, "lambda_min") / int(row["replicates"]) for row in group],
            [_number(row, "log_parameter_correlation") for row in group],
        ]
        for ax, raw in zip(axes, values):
            finite = np.asarray([value for value in raw if np.isfinite(value)])
            if len(finite):
                low, median, high = np.percentile(finite, [10, 50, 90])
                ax.errorbar(position, median, yerr=[[median - low], [high - median]],
                            fmt="o", color=color, capsize=4, markersize=5)
            else:
                ax.text(position, 0, "undefined", rotation=90, ha="center", va="center", fontsize=8)
        singular += sum(not np.isfinite(value) for value in values[1])
    eigenvalues = [_number(row, "lambda_min") / int(row["replicates"]) for row in unique]
    if all(value > 0 for value in eigenvalues):
        axes[0].set_yscale("log")
        axes[0].yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        axes[0].yaxis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(np.inf, np.inf)))
    else:
        axes[0].set_yscale("symlog", linthresh=1e-4, linscale=0.7)
    axes[0].set_ylabel("Smallest eigenvalue of I(log τ, log σ²) / R\n(dimensionless, per realization)")
    axes[0].set_title("Local information at generating truth", fontsize=12)
    axes[1].set_ylabel("Predicted correlation of\nlog-parameter estimates")
    axes[1].set_ylim(-1.05, 1.05)
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_xticks(range(len(keys)), labels, fontsize=8.5)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25, which="both")
        ax.spines[["top", "right"]].set_visible(False)
    counts = [sum((row["family"], _regime(row), _design(row)) == key for row in unique) for key in keys]
    count_text = str(counts[0]) if len(set(counts)) == 1 else f"{min(counts)}–{max(counts)}"
    fig.suptitle("Fisher information in dimensionless log-parameter coordinates", fontsize=14, y=0.99)
    fig.text(
        0.5, 0.012,
        f"Points: median; whiskers: 10th–90th percentiles across {count_text} recorded mask(s) per condition. "
        "One row per graph/mask; repeated R diagnostics removed.\n"
        f"Undefined correlations: {singular}/{len(unique)} masks. Full-observation masks coincide within a graph/regime. "
        "Local asymptotic diagnostics are not finite-sample guarantees.",
        ha="center", fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.96))
    return _save(fig, output)


def _secondary_figure(
    primary: list[dict[str, str]], secondary: list[dict[str, str]], output: Path
) -> Path | None:
    rows = [row for row in primary + secondary if _design(row) == "full"]
    sizes = sorted({int(row["n"]) for row in rows})
    if len(sizes) < 2:
        return None
    # Compare matching graph/regime/design conditions, not an aggregate of
    # potentially different information regimes or observation masks.
    keys = sorted({(row["family"], _regime(row)) for row in secondary}, key=_sort_key)
    budgets = sorted({int(row["replicates"]) for row in secondary})
    ncols = min(2, len(keys))
    nrows = (len(keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.8, 3.3 * nrows + 1.3), squeeze=False, sharey=True)
    counts: dict[int, set[int]] = {size: set() for size in sizes}
    for ax, key in zip(axes.flat, keys):
        panel = [row for row in rows if (row["family"], _regime(row)) == key]
        for size in sizes:
            for mode in ("joint", "noise_known"):
                medians, lower, upper = [], [], []
                for budget in budgets:
                    values = [
                        abs(_number(row, "fit_tau") - _number(row, "true_tau"))
                        / _number(row, "true_tau") * 100
                        for row in panel if int(row["n"]) == size and row["fit_mode"] == mode
                        and int(row["replicates"]) == budget
                        and _valid(row) and np.isfinite(_number(row, "fit_tau"))
                    ]
                    counts[size].add(len(values))
                    if not values:
                        medians.append(np.nan)
                        lower.append(np.nan)
                        upper.append(np.nan)
                        continue
                    p10, median, p90 = np.percentile(values, [10, 50, 90])
                    medians.append(median)
                    lower.append(median - p10)
                    upper.append(p90 - median)
                ax.errorbar(range(len(budgets)), medians, yerr=[lower, upper], capsize=4,
                            marker="o" if size == sizes[0] else "s", color=COLORS[mode],
                            linestyle="--" if size == sizes[0] else "-",
                            label=f"n={size} · {MODE_LABELS[mode]}")
        ax.set_title(f"{key[0]} · τ*={_number(panel[0], 'true_tau'):g}, σ²*={_number(panel[0], 'true_noise_variance'):g}")
        ax.set_xticks(range(len(budgets)), [str(value) for value in budgets])
        ax.set_xlabel("Independent realizations R")
        ax.set_ylabel("Heat-time absolute relative error (%)")
        ax.set_yscale("symlog", linthresh=10, linscale=0.7)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25, which="both")
    for ax in list(axes.flat)[len(keys):]:
        ax.set_visible(False)
    fig.suptitle(
        "Secondary size check: heat-time estimation with full observations",
        fontsize=14, y=0.99,
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.955), fontsize=9)
    count_text = "; ".join(f"n={size}: {','.join(str(value) for value in sorted(counts[size]))} valid pools/cell" for size in sizes)
    fig.text(0.5, 0.012,
             "Points: median; whiskers: 10th–90th percentiles across independent pools. " + count_text + ".\n"
             "Graphs differ in spectrum as well as size; this is a bounded secondary comparison, not a size-scaling law.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.075, 1, 0.88))
    return _save(fig, output)


def plot_all(input_dir: Path | str, output_dir: Path | str, secondary_dir: Path | str | None = None) -> list[Path]:
    """Regenerate figures from CSV evidence and return the written image paths."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fits = _read_csv(input_dir / "fit_selected.csv")
    if not fits:
        raise ValueError(f"No selected fits in {input_dir / 'fit_selected.csv'}")
    paths = [_error_figure(fits, output_dir / "identifiability_replicates.png", "tau")]
    if any(row["fit_mode"] == "joint" for row in fits):
        paths.append(_error_figure(fits, output_dir / "identifiability_noise.png", "noise_variance", joint_only=True))
    for filename, function, target in (
        ("landscapes.csv", _likelihood_figure, "identifiability_likelihood.png"),
        ("fisher.csv", _fisher_figure, "identifiability_fisher.png"),
    ):
        source = input_dir / filename
        rows = _read_csv(source) if source.exists() else []
        if rows:
            paths.append(function(rows, output_dir / target))
        else:
            print(f"No {filename} rows; skipped {target}")
    if secondary_dir is not None:
        secondary_path = Path(secondary_dir) / "fit_selected.csv"
        if secondary_path.exists():
            result = _secondary_figure(fits, _read_csv(secondary_path), output_dir / "identifiability_size.png")
            if result is not None:
                paths.append(result)
        else:
            print(f"No secondary selected fits: {secondary_path}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/identifiability"))
    parser.add_argument("--output", type=Path, default=Path("docs/figures"))
    parser.add_argument("--secondary-input", type=Path)
    args = parser.parse_args()
    for path in plot_all(args.input, args.output, args.secondary_input):
        print(path)


if __name__ == "__main__":
    main()
