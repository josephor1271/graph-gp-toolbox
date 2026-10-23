# Stage 02 evidence and column definitions

This directory contains the complete small tabular outputs of the primary study.
`secondary/` contains the separate 100-node check; `optimizer_audit/` contains a
bounded follow-up of flagged primary fits. Full latent/noise arrays remain in
ignored `results/identifiability/pools/` and the secondary equivalent. They can be
regenerated from the saved configuration and the source commits/hashes recorded
there. All data are synthetic. No earlier experiment evidence was changed.

| File | Unit of one row |
|---|---|
| `fit_starts.csv` | One optimizer start, including invalid or nonconverged outcomes |
| `fit_selected.csv` | One lowest-valid-exact-NLL selection per pool and fitting condition |
| `summary.csv` | One graph/regime/mask/R/mode cell, aggregated over 20 independent primary pools (10 secondary pools) |
| `paired_comparisons.csv` | Within-pool changes from R=1 to larger R, summarized by cell |
| `fisher.csv` | Total log-parameter Fisher information at truth for one fixed graph/mask and R |
| `masks.csv` | Explicit observed node IDs and the graph scaling, once per graph/regime/design |
| `landscapes.csv` | One candidate parameter pair for the preregistered sample/expected-likelihood comparison |
| `config.json`, `environment.json` | Complete design, seeds, versions, numerical settings and source provenance |
| `validation.json` | Measured direct/Torch/GPyTorch A/C fixed-parameter likelihood agreement |
| `completion.json` | Counts, failures, stationarity and boundary diagnostics |

There are **120 independent primary full-draw pools** (3 families × 2 regimes ×
20 seeds), each with 20 independently drawn latent functions and noise vectors.
Partial/full masks share those arrays. The three R budgets are nested prefixes;
R budgets, fit modes and optimizer starts do not create extra independent pools.
Thus 240 mask conditions lead to 1,440 selected fits and 2,880 original starts.
The secondary study has 40 independent pools, 160 fits and 320 starts. Its graph
and data streams are separate from the primary study.

Fit keys are `family,n,regime,observation_design,pool_seed,replicates,fit_mode`;
`init_id` adds the start index. `observed_fraction` is the **actual** count/n,
while `requested_fraction` is the original 0.75/0.5 design fraction. Counts use
floor rounding: 36/49 or 24/49 observed nodes. `true_tau` and
`true_noise_variance` describe the fixed generating covariance; amplitude is one.

`nll_per_observation` is normalized per observed node **and realization**.
`truth_nll_per_observation` uses the same sample at its generating parameters;
`nll_gap_fit_minus_truth` is fit minus truth. Finite-sample NLL need not be lowest
at truth. `tau_relative_error` and `noise_relative_error` are **absolute** relative
errors. Noise-error fields are empty for `noise_known`; summaries omit their
noise-estimation statistics rather than treating known noise as an estimated zero
error. Estimate quantiles, absolute-relative-error quantiles and physical-unit
RMSE are separately named. Counts are valid/attempted pools, not start counts.

`optimizer_success` preserves SciPy's flag and `optimizer_message` its termination
reason. `gradient_norm` is the L2 norm of the final **log-coordinate** gradient;
`projected_gradient_norm` is its infinity norm after projecting outward-pointing
components at active bounds. `stationary` requires the latter ≤1e-6.
`log_gradient` and `projected_log_gradient` retain the vectors as JSON arrays.
`boundary_flags` is a JSON list: a fit within 1% of the unchanged physical range
width is flagged, including exact endpoints. A fixed known noise value is never
flagged as a noise estimate. These thresholds are diagnostics, not statistical
recovery tolerances.

`valid` means finite exact NLL/gradients with a successful Cholesky evaluation;
it is separate from optimizer success/stationarity. A valid nonconverged fit is
retained and may be selected if its actual NLL is lowest. `selection_status` is
`selected_stationary`, `selected_not_converged`, or `no_valid_start`. The case-level
selection status and spread are copied onto each start row; its own optimizer
diagnostics still describe that start. `nll_start_spread` is the maximum minus
minimum valid-start NLL. `start_disagreement` is true for spread >1e-7 or any
absolute log-parameter ratio to the selected fit >0.01. Neither lowest-start
selection nor stationarity certifies a global optimum.

Fisher entries (`information_00/01/11`, eigenvalues) are **total information across
R independent vectors**, without division by observed count or R. The report
plot divides by R to compare designs. `log_parameter_correlation` is the
correlation implied by inverse information, calculated algebraically for 2×2
matrices without forming an inverse. It is absent for numerically singular
information. The local diagnostics are not finite-sample confidence guarantees.

The primary landscape is fixed in advance to grid/n=49/regime 0/partial/pool 0/R=1.
The grid includes truth and the selected fit. `expected_excess_nll` is
KL(true || candidate) per observed node; it is not the observed sample NLL.
The plot subtracts only the finite-sample grid minimum and shows full saved
ranges using a square-root color scale. Raw negative roundoff-sized KL values,
if present, are retained.

The separate audit uses only recorded start/convergence flags to select 84
conditions, then evaluates a fixed 9×9 logarithmic grid and three extra starts
per condition. `grid.csv` contains all 6,804 grid evaluations. The audit's
`fit_selected.csv` retains original estimates/NLL beside the best additional fit;
the original primary files and report distributions are preserved. This adds
252 optimization attempts and **zero** independent pools. Its config labels the
follow-up explicitly.

The audit records `source_dirty=true` because its five generated report figures
were still untracked at execution. Its recorded driver hash matches the tracked
source at commit `fe2ea24` and the included source file; the tracked audit code
was unchanged. Primary/secondary runs record clean source commit `d57b30a`.

Regenerate all report figures from these CSVs, without sampling or fitting:

```bash
python experiments/02_plot_identifiability.py \
  --input docs/evidence/identifiability \
  --secondary-input docs/evidence/identifiability/secondary \
  --output docs/figures
```
