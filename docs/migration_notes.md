# Migration notes

The source workspace was `gpytorch-ex`, an unversioned directory: `git status`
reported no Git repository. Its Python, Markdown and notebook source files were
left unchanged. The four development/results reports were read before migration.
The actual source and tests, rather than plans, determined what was reused.

Baseline: Python 3.12.0, Torch 2.11.0, GPyTorch 1.15.2, linear_operator 0.6.1.
The legacy suite ran 68 tests: 67 passed, one skipped (CUDA unavailable).
No numerical dependencies were upgraded. The local development venv uses the
existing system packages; the package also installs in a fresh Python 3.12 venv.

| Legacy file | New file | Retained or changed behavior |
|---|---|---|
| `sparsecheblinop.py` | `src/graphgp/operators/chebyshev.py` | CSR recurrence, coefficient/RHS autodiff, exact selected entries; unchanged numerical implementation |
| `chebyshev_inputs.py` | `src/graphgp/operators/coefficients.py` | Torch differentiable coefficients, SciPy reference, bounds and degree diagnostics |
| `graph_kernels.py` | `src/graphgp/kernels/heat.py` | Trainable heat time, InterpolatedLinearOperator node selection, repeated IDs |
| `baseline_kernels.py` | `src/graphgp/kernels/dense.py` | Cached dense eigendecomposition; exact and polynomial references independent of sparse operator |
| `graph_gp_demo.py` | `src/graphgp/models/exact_gp.py`, `src/graphgp/demo.py` | Model/settings separated from original small sine-signal demo; demo is not a recovery experiment |
| `experiment_utils.py` | `src/graphgp/validation.py` | Small-problem five-method checks, exhaustive probes, guards, diagnostics and CSV/environment helpers |
| `graph_gp_diagnostics.py` | `src/graphgp/diagnostics.py` | Original scalar/CG diagnostics preserved |
| Six core test modules + helpers | `tests/` | All 48 core tests retained, imports remapped |

[Migration manifest](migration_manifest.json) records SHA-256 hashes of the original
files, since no legacy commit exists. Final source differs where described above.

New components: scaled deterministic graph construction; bounded model factory;
heat interpolation positivity analysis; requested-target marginal/mean-only helper;
exact-GP sampling and recovery harness; four additional regression tests (52 total).
The new prediction helper delegates conditioning to GPyTorch. It resets its cache
per call and reads raw covariance diagonals so negative variances are not hidden by
GPyTorch's public variance clamp. It does not promise bounded-memory exact variance
computation for very large training or target sets.

The generic migrated kernels retain positive, unbounded heat times for legacy
compatibility. Use `make_heat_gp` for the study's enforced range. The study uses
33 interpolation nodes rather than legacy oversampling at 128 nodes; a real-arithmetic
interpolation remainder bound applies specifically to this choice. Bounds, tests,
and float64 checks are documented in [numerical safety](numerical_safety.md).

Not migrated: notebooks, benchmark runners, historical raw results, stochastic and
adaptive posterior diagonal estimators, LOVE modes, supervisor documents. No
historical speedup is claimed for this repository. Spectral mixtures and graph×time
models remain subsequent milestones, as requested for the first A–C pass.
