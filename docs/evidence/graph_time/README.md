# Stage 03 measured evidence

These files summarize the single fixed full run of `experiments/03_graph_time.py`
at clean source commit `6d1f850964098fd98f353739f3be46618d85071a`, starting
from main `19b022fd9cadad3026534b4c98f4134d8b3c5d64`. Source SHA-256 hashes,
package versions and CPU/thread settings are in `compile_environment.json`.
Later commits add this evidence and its report. Previous stage evidence is unchanged.

There are four independent latent/noisy fields, one per seed. Interpolation and
forecasting share each seed's field; model comparisons and tasks are paired.
All 34 fits are retained, including four sparse fits with numerical warnings.
No fit failed. The two noise-learning fits and eight 40-update parity fits are
excluded from the 24-fit main ranking.

| File | Row unit / definition |
|---|---|
| `config.json` | Predeclared graph, physical grid, seeds, parameters, masks, bounds and optimizer budgets. |
| `compile_environment.json` | Environment and source provenance for the full run. |
| `validation.json` | Two fixed-parameter GP gates and 32 kernel configurations, numeric errors, actual solver/operator calls and emitted warnings. |
| `fits.csv` | One fit (34 rows): initialization, completed Adam updates, first/final normalized training NLL, endpoint parameters, raw-gradient norm, runtime, boundary flags, calls and warnings. |
| `metrics.csv` | One scored fit (34 rows): held-out latent/observation RMSE, observation NLPD, 95% observation marginal coverage, variance minima, mean-only and independent-oracle errors. |
| `summary.csv` | One main task/model cell (six rows): exactly four seeds, counts, median and observed minimum/maximum for each score. No confidence intervals. |
| `parity.csv` | One matched dataset (four rows): dense/sparse differences after 40 Adam updates, trajectory maximum error and fixed-endpoint oracle errors. |
| `trajectories.csv` | Only seed 11 interpolation parity, dense/sparse, steps 0–40 inclusive (82 rows). Step zero precedes all updates; step 40 follows exactly 40 updates. |
| `signal.csv` | Full 16×10 field, vertex-major, for seed 11 interpolation main joint dense fit (160 rows): true latent, noisy observation, explicit 64-position training mask, full-grid prediction and both variances. |
| `completion.json` | Recorded protocol counts, failures and numerical-warning fit counts; expected counts are in `config.json`. |

The kernel is unit amplitude and mean zero. `heat_time` uses the unchanged
degree-scaled combinatorial Laplacian. `lengthscale` uses physical normalized
time `j/(T-1)`. `noise_variance` is a variance, not a standard deviation.
Inputs use integer time **indices**. Training NLL is per recorded observation;
NLPD is the mean held-out noisy-observation marginal negative log density.
RMSE is in signal units; coverage is a fraction. Empty CSV parameter cells mean
the parameter does not exist in that baseline. `boundary_flags` uses distance
within 1% of a parameter's allowed interval width; an empty list is not a
certificate of optimization convergence. `final_gradient_norm` is the norm
over trainable raw parameters. JSON lists in CSV cells preserve unique warning
messages; `numerical_warning_count` counts all emitted numerical warnings.
`training_seconds` excludes construction and prediction but includes the final
loss/gradient evaluation after the last update. It is not a speed benchmark.

The graph-only baseline uses the existing degree-32 sparse heat covariance
with direct Cholesky of the selected training covariance; the time-only
baseline uses direct RBF covariance. Main joint fits use dense exact GPyTorch.
Only explicitly labeled sparse parity fits force CG/SLQ and graph/full-grid
no-dense guards. Those fits retain requested tolerance, actual Lanczos orders
and residual warnings separately: requesting `1e-7` did not achieve that
residual tolerance. No installed solver defaults were changed.
`training_sparse_laplacian_mvps` counts recurrence applications (32 per heat
operator matmul), with multiple RHS columns in each application; it is not a
count of independent vector operations. See validation's RHS-width counts.

Run from the repository root after `python -m pip install -e .`:

```bash
python -m unittest discover -s tests -v
python experiments/03_graph_time.py --smoke --output results/graph_time_smoke
python experiments/03_graph_time.py --output results/graph_time
python experiments/03_plot_graph_time.py --input results/graph_time --output docs/figures
```

Study output directories must not already contain a configuration. Use a new
output directory for an intentional second run. The full ignored output holds
all trajectories plus deterministic data and prediction NPZs; these large raw
arrays are deliberately excluded from Git. To regenerate all three committed
figures using **only committed CSVs**, without fitting or sampling:

```bash
python experiments/03_plot_graph_time.py --input docs/evidence/graph_time --output docs/figures
```

Regeneration into a separate directory produced identical SHA-256 hashes for
all three committed PNGs in the recorded environment. Other Matplotlib/font
environments may render pixels differently. The compact trajectory subset is
enough for the preregistered training figure; it is not the full run archive.
Curated CSVs use LF line endings. Their values match the raw outputs except
for this documented normalization:
upstream PyTorch build paths in warning text are replaced by
`<upstream-pytorch-build>/`. Warning categories, numbers and messages otherwise
remain intact. No covariance, metric or fit was filtered based on outcome.

Full-grid `nT` workspace, dense temporal `T×T` factors, exhaustive quadratic
probe storage and exact target-variance work remain. Small guarded examples
establish the exercised paths; they do not establish general scalability or
prove absence of every possible allocation. See
[the measured report](../../experiment_03_findings.md) for the numerical gates,
solver limitations and interpretation.
