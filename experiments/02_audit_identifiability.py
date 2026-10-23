"""Bounded extra-start audit of flagged Stage 02 joint fits; preserve originals.

Selection is triggered by recorded optimizer/stationarity/start-disagreement
flags, never by distance from the generating parameters. The fixed 9x9 search
grid supplies two separated starts, plus a restart at the original selected fit.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
import torch
from graphgp.analysis.heat_identifiability import HeatOracle, fit_multistart, HEAT_RANGE, NOISE_RANGE
from graphgp.utils.graphs import make_laplacian
from graphgp.validation import write_csv, environment_metadata


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def run_audit(source, output):
    source, output = Path(source), Path(output)
    if (output/"config.json").exists():
        raise FileExistsError("audit output exists; choose a fresh --output")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    config = json.loads((source/"config.json").read_text())
    candidates = [r for r in read_csv(source/"fit_selected.csv") if r["valid"] == "True" and
        (r["start_disagreement"] == "True" or r["stationary"] != "True" or r["optimizer_success"] != "True")]
    if any(row["fit_mode"] != "joint" for row in candidates):
        raise ValueError("this audit is limited to flagged joint fits")
    (output/"config.json").write_text(json.dumps(dict(source_config=config,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        audit_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        trigger="start_disagreement OR nonstationary OR optimizer_non_success",
        grid_shape=[9, 9], grid_spacing="logarithmic over unchanged physical bounds",
        extra_starts="two lowest-NLL grid points separated by >=0.4 in max log distance, plus original selected estimate",
        original_results="preserved, never replaced", selected_cases=len(candidates)), indent=2)+"\n")
    (output/"environment.json").write_text(json.dumps(environment_metadata(), indent=2)+"\n")
    masks = {(r["family"], r["regime"], r["observation_design"]): np.array(json.loads(r["observed_ids"]))
             for r in read_csv(source/"masks.csv")}
    oracles, starts, selected, grids = {}, [], [], []
    for row in candidates:
        family = row["family"]
        if family not in oracles:
            oracles[family] = HeatOracle(make_laplacian(family, config["n"], config["graph_seed"])[0])
        oracle = oracles[family]
        ids = masks[family, row["regime"], row["observation_design"]]
        pool_name = f"{family}_n{config['n']}_r{row['regime']}_s{row['pool_seed']}"
        with np.load(source/"pools"/(pool_name+".npz")) as data:
            Y = data["observations"][ids, :int(row["replicates"])]
        grid = [(oracle.nll(tau, noise, Y, ids), float(tau), float(noise))
                for tau in np.geomspace(*HEAT_RANGE, 9) for noise in np.geomspace(*NOISE_RANGE, 9)]
        grid.sort()
        first = grid[0][1:]
        second = next(point[1:] for point in grid
                      if np.max(np.abs(np.log(np.array(point[1:])/first))) >= .4)
        additional, best = fit_multistart(oracle, Y, ids,
            initials=(first, second, (float(row["fit_tau"]), float(row["fit_noise_variance"]))),
            maxiter=config["maxiter"], gtol=config["gtol"], ftol=config["ftol"])
        meta = {key: row[key] for key in ("family", "n", "regime", "observation_design", "pool_seed",
                "replicates", "fit_mode", "true_tau", "true_noise_variance")}
        starts.extend(meta | point for point in additional)
        grids.extend(meta | dict(grid_nll=value, grid_tau=tau, grid_noise_variance=noise) for value, tau, noise in grid)
        selected.append(meta | dict(original_fit_tau=float(row["fit_tau"]),
            original_fit_noise_variance=float(row["fit_noise_variance"]),
            original_nll=float(row["nll_per_observation"]), original_stationary=row["stationary"],
            original_optimizer_success=row["optimizer_success"]) | best)
    for name, rows in (("fit_starts.csv", starts), ("fit_selected.csv", selected), ("grid.csv", grids)):
        write_csv(output/name, rows)
        (output/name).write_text((output/name).read_text())
    completion = dict(cases=len(selected), extra_optimizer_starts=len(starts),
        invalid_extra_starts=sum(not row["valid"] for row in starts),
        selected_nonstationary=sum(not row["stationary"] for row in selected),
        selected_optimizer_non_success=sum(not row["optimizer_success"] for row in selected),
        max_nll_improvement=max((row["original_nll"]-row["nll_per_observation"]
            for row in selected if row["valid"]), default=0))
    (output/"completion.json").write_text(json.dumps(completion, indent=2)+"\n")
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/identifiability"))
    parser.add_argument("--output", type=Path, default=Path("results/identifiability_audit"))
    args = parser.parse_args()
    run_audit(args.source, args.output)
