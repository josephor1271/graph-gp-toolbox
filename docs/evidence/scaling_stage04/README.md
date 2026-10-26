# Stage 04 evidence: isolated operations and solver diagnostics

This package retains the fixed benchmark's successful measurements and failed
attempts. The full combined command runs the solver sweep **before** the scaling
table. It starts from main `4f90ba5d64f7fe08f9f89697f3be06ce508c450a`.
The measured computational source is clean commit
`060a3089cbfb008ca2c8775e35cb9503f967110c`; later commits add evidence/reporting.
Source hashes and the actual interpreter/platform/package identities are recorded.

| File | Unit and interpretation |
|---|---|
| `baseline.json` | Actual initial clean main, versions/hardware, baseline import relationships, old test/smoke outcomes and hashes of historical documentation/evidence/figures. The roadmap is excluded because this stage updates it. |
| `config.json` | All 12 fixed cases, profile table, seeds, screening thresholds, repeats, model and measurement rules. Debug case limits are null in the full run. |
| `environment.json` | Clean computational source SHA and hashes, exact CLI, packages, CPU/OS and memory/timeout conventions. |
| `measurement_correction.json` | Reproduced cold-guard import bias, correction commits and explanation of the superseded first run. |
| `source_audit.json` | Installed CG/SLQ signatures, package-relative source paths/line citations/file hashes and limits of causal attribution. Full upstream source listings are not copied into Git. |
| `raw_runs.csv` | One fresh child process per campaign/case/operation/backend/profile/probe seed/repeat/RHS width, including non-successes. `q` is the actual number of measured targets (32 in solver checks; declared 64 in those full scaling cases). |
| `solver_checks.csv` | One independently checked noisy selected-covariance solve per RHS: training y and two independent normal vectors. Requested tolerance, true fresh-matvec residual, exact Cholesky solution error, warning count and status are distinct. |
| `accuracy.csv` | `screening_scope=profile_seed` combines a fixed profile/seed's MLL, both posterior routes and three solve checks. `case_check` retains each matched primary fixed-point operation/repeat/backend's independent errors, including polynomial products. It is not a complete profile qualification. |
| `summary.csv` | Grouped raw timing/peak-RSS medians and observed min/max, retaining attempted/success/timeout/failure counts. Numeric summaries use successful measurements only. |
| `completion.json` | Expected/actual row counts, all attempted case IDs, screening counts and explicit non-success reason codes. Its `solver_accuracy_rows` field counts all combined accuracy rows; the scope column distinguishes the solver screens. |

Times are seconds. RSS is MiB for the **whole isolated worker**, including
imports, data generation, construction and allocator effects: macOS `ru_maxrss`
is bytes divided by 2²⁰; Linux is KiB divided by 1024. Current RSS, where
available, is the OS process reading before the operation, not peak subtraction.
No claim about per-operation native Torch allocation is made. A missing memory
metric is null/unsupported. No host memory ceiling was imposed; each worker has
a hard 180-second wall timeout and the coordinator kills/reaps timed-out children.
Signals/nonzero exits/invalid output/launch errors have separate statuses. Failed
rows have no invented successful operation duration or score.

The parent imports no NumPy/SciPy/Torch. Workers unconditionally set all four
BLAS/OpenMP thread environment variables to one before numerical imports, then
set Torch threads to one. Detailed inherited/effective values are retained in
the ignored worker JSON; the environment's derived thread summary describes
those observed values. Measurements run sequentially on the same machine.

The measured full output is `results/scaling_stage04_corrected/`. The initial
`results/scaling_stage04/` run is preserved but superseded because sparse-only
lazy diagnostic imports contaminated construction and RSS. Those comparisons
are not pooled with this corrected run. Common diagnostic dependencies are now
loaded for both backends before setup timers; `import_seconds` is separate,
and entering diagnostic guards is outside the model construction timer.

`model_build_seconds` is model/operator setup after data creation;
`data_setup_seconds` includes graph and approximate field generation.
`elapsed_seconds` excludes imports and those setup phases, except `construct`,
which explicitly sums data/model setup. `child_wall_seconds` is the parent's
entire subprocess elapsed time including imports and setup. `first_call_seconds`
is a cold GP operation, with fresh model/prediction cache, not a warm repetition.
There is no GP warmup. Each MVP child warms up exactly once, then takes five
timings; its elapsed value is their median and all five values are retained.
MVP counters include that warmup plus the five measured products. Independent
small timing repeats are separate child processes, not five cached model fits.
No confidence intervals are inferred from two or three timing repeats.

Every model has zero mean, unit amplitude, heat time 1.2, temporal lengthscale
0.24, observation noise **variance** 0.06 and degree 32. The grid is the unchanged
degree-scaled combinatorial graph, with `v*T+j` ordering and physical time
`j/(T-1)`. Main measurements hold parameters fixed; `train3` makes exactly three
Adam updates at 0.05 plus a fourth endpoint loss/gradient evaluation. Its four
recorded points are a compatibility/cost check, not convergence or recovery.

The benchmark signal is the **approximate**, correlated field
`P32(exp(-tau L/2)) @ Z @ chol(KT).T` with independent observation noise.
It is not the independent exact-GP sampling oracle. NumPy SeedSequence master
202604 separates latent, noise, training/target masks, probe seeds and RHS streams.
Explicit case/stream identifiers, never Python `hash`, are used. Density masks
are prefixes of one fixed permutation; targets are fixed disjoint prefixes
outside even the largest mask. Probe labels 301/302/303 map through the separate
SeedSequence probe stream to recorded effective Torch seeds. The same seed is
reset at each three-update loss evaluation to reuse common stochastic probes.
Production stochastic probe generation is not patched; only the n16/T8/m32
exhaustive control uses the old orthonormal diagnostic policy.

`operator` JSON columns count Chebyshev `_matmul` calls, degree-times-call
recurrence applications, RHS-width histograms and selected-entry work/timing.
Multiple vectors share an application: neither counter equals scalar SpMV work.
Wrapped-method time is not complete algorithm time. `solver` JSON columns record
actual CG calls/RHS shapes and actual returned/used tridiagonal orders. Actual CG
iterations are **not instrumented** and cannot be inferred from those counts.
Requested tolerances and Lanczos maxima are never substituted for achieved values.
Warnings retain their categories and full messages. Curated text normalizes only
upstream PyTorch's build-path prefix to `<upstream-pytorch-build>/`; CSV line endings
are LF. Numerical values and adverse outcomes are not filtered.

Only explicitly labeled dense/reference workers may form dense graph/full-grid
covariances, within n≤100, nT≤800, m≤400, q≤256. Sparse inference guards reject
known graph/full-prior conversion, direct Cholesky and graph eigendecomposition
paths. Data generation permits only the declared temporal-factor Cholesky before
guarded inference. Dense temporal factors, selected cross blocks and small batched
Lanczos eigensystems remain allowed. These are guards of exercised paths, not a
proof against every future allocation. Full-grid RHS workspace and T² temporal
storage remain.

Run from repository root after `python -m pip install -e .`, using a new directory:

```bash
python -m unittest discover -s tests -v
python experiments/04_scaling.py --smoke --output results/scaling_stage04_smoke
python experiments/04_scaling.py --output results/scaling_stage04
# Optional separate reproduction of the solver-only subset:
python experiments/04_solver_diagnostics.py --output results/solver_stage04
# Plot using committed CSVs only; no training, sampling or worker imports:
python experiments/04_plot_scaling.py --input docs/evidence/scaling_stage04 --output docs/figures
```

The full ignored output also contains exact request specifications, per-worker
JSON/logs, a crash-visible JSONL journal and arrays returned for comparison. No
resume/retry is implicit. None of those large arrays or per-method traces are
needed by the four plot-only figures or committed here. Consult
[the findings report](../../experiment_04_findings.md) for measured outcomes,
accuracy qualification, failures and the prioritized next step.
