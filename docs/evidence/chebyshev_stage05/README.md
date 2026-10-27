# Stage 05 evidence

This package supports [the findings](../../experiment_05_findings.md). The primary
study separates A (dense heat/direct Cholesky), B (the identical dense heat/CG+SLQ),
C32 (CSR Chebyshev degree 32/CG+SLQ), and C16 (degree 16/CG+SLQ). The scaled-CSR
candidate is a separate, interleaved operator experiment. The four-arm primary
study keeps the original recurrence. Defaults remain degree 32 and unscaled CSR.

The primary signal is the unchanged Stage04 approximate-field generator with new
master seed 202605, independent recording noise, one realization per n/T, and
separate mask/RHS/probe streams. It is not an exact-GP sampling or parameter-recovery
study. q16/q64/q256 are nested prefixes; training data and validation RHS are
identical across those cases and all arms. Data fingerprints include requested
targets; base fingerprints exclude targets. The q variants use their canonical
n100/n400 operator measurements because graph, time and MVP RHS are identical.

## Reproduce

Use Python 3.12 with the repository constraints. These commands create fresh
ignored directories; existing runs are never overwritten. Run the profiler gate
before requesting the optional scaled candidate. Do not run simultaneous timing
workers or compete with them using training/tests/plotting.

```bash
python -m unittest discover -s tests -v
python -m graphgp.benchmarking.stage05_validation --output results/stage05_validation.json
python experiments/05_optimize_chebyshev.py --smoke --output results/chebyshev_stage05_smoke
python experiments/05_optimize_chebyshev.py --profile-only --output results/chebyshev_stage05_profile
# Only if the predeclared profile gate passes:
python experiments/05_optimize_chebyshev.py --scaled-candidate --output results/stage05_scaled_candidate
python experiments/05_optimize_chebyshev.py --output results/chebyshev_stage05
python experiments/05_curate_evidence.py --output results/chebyshev_stage05_curated
python experiments/05_plot_optimization.py --evidence docs/evidence/chebyshev_stage05 --output results/stage05_figures_check
```

The curation command additionally reads the original ignored baseline controls,
dispatch trace, before/after entry diagnostic, and prebenchmark check logs. Those
were produced before the full study. Their compact committed equivalents are
baseline.json, baseline_controls.json, batched_controls.json, dispatch.json and
profile_gate.json. The baseline control used a single diagnostic process: its
ordered-arm times are **not headline performance evidence**. Their retained values
and invocation counts establish the original path. Reproducing that baseline
requires the recorded original main SHA, rather than timing current code and
calling it the old method. The original scalar implementation also remains a
scoped benchmark-only reference, and all original/new timings in entry_diagonal.csv
use the same present source, data and solver settings.

## Interpretation and scope

- config.json fixes all case shapes, settings, repeats, numerical gates and
  resource bounds. requests.json and raw per-worker JSON remain ignored under
  results/; all 1,275 primary requests were serialized before the first child.
- environment.json records clean measured source SHA/hashes, package versions and
  CPU. Completion includes every attempted worker, total wall time, errors and
  timeouts. A failure is never a zero timing. No portable host memory ceiling was
  imposed; every child had a 180-second hard timeout, kill/reap and no retry.
- operator_micro.csv keeps one warmup and five product times, graph/time widths,
  prebuild/setup times and whole-worker RSS. Autograd graph microbenchmarks use one
  untimed forward/backward and one timed forward/backward in each fresh worker.
  Operator measurements use three processes; GP timings use two at n<=400 and one
  at n1600/T32. Within-process product samples are not independent data draws.
- inference.csv contains build, MLL forward/backward, three actual Adam updates
  plus final evaluation, cold prediction and separately labeled warm prediction.
  Warm counts are in warm_operator/warm_solver; their outer counters include cold
  preparation too. Train3 endpoint marginals are untimed diagnostics with separate
  counters; they are excluded from the measured training interval.
  Train3's outer actions/direct counters include the endpoint; its operator/solver
  snapshots describe training, and endpoint_operator/endpoint_solver describe the
  separate endpoint prediction.
- accuracy.csv compares each operation against an independent selected-space
  SciPy expm/NumPy-RBF/Cholesky oracle through n400. Torch matrix-exp derivatives
  independently check raw parameters. Full-product spot checks use 32 fixed rows
  and all RHS columns; the complete small-domain numerical tests are separate.
  No exact large-case reference is invented. Missing-reference rows fail the
  qualification screen. Timing rows and finite numerical results remain visible.
- The inherited operational screen is NLL <=0.05, each raw-gradient error
  <=0.02+0.20*abs(reference), posterior absolute error <=0.01, and true fresh noisy
  solve residual <=0.001. It is an operational comparison, not calibrated training.
  A wrong sign for a raw reference magnitude >0.001 additionally disqualifies a
  training crossover even when the loose screen passes. Strict and exhaustive
  tiny controls have separate labels and never enter primary timing medians.
- solver.csv retains actual CG calls/RHS shapes, returned Lanczos orders, warnings
  and three fresh noisy-covariance residuals. CG iteration counts remain null;
  polynomial recurrence applications are not scalar-vector FLOPs. Inclusive
  method timers overlap outer wall time and may not be added into a cost budget.
- summary.csv reports all attempts and successful timing min/median/max. Qualified
  ratios require every repeated result and its fresh solve to pass. A sparse GP
  crossover requires beating **both A and B** on that operation. Dense ratios stay
  null at sparse-only sizes. Build + 1/20/100 MLL costs are arithmetic projections
  from measured medians; train3 is a separate measured compatibility test.
- degree.csv contains the analytic certificate, 480 full parameter-domain rows,
  and every paired C16/C32 GP result including failed pairs. Tiny validation also
  checks learned-noise derivatives, state/cache mutations and three-update parity.
- scaled_candidate.csv/json retain all 90 candidate-study attempts, the same-width
  10% elapsed-reduction test in at least two larger sizes, every <=10% regression
  check and the <=15% extra resident-memory test. The memory metric includes CSR
  indices/values, spectral nodes, cosine basis and raw tau. Scaled storage replaces
  the raw CSR and removes exact zeros; it does not cache a duplicate graph. This
  byte count is distinct from whole-worker peak RSS. Candidate plot lines come
  from this separate interleaved microstudy; admission ratios use its matched
  baseline rather than a main-study timing from a different execution block.
- t32_supplement.json/csv label the secondary-only spectral PSD square root used
  for noiseless data. Only tolerated negative temporal eigenvalues were clipped
  for sampling. The inference kernel and historical Stage04 failures were not
  repaired or overwritten.

Exactly three figures read the committed CSVs only. Their PNG bytes are
reproducible in the recorded Matplotlib/font environment. They do not run GP
sampling, fitting or numerical reference generation.
