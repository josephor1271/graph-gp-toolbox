# Graph GP toolbox

A small research codebase for Gaussian processes on fixed graphs, using GPyTorch
and sparse Chebyshev matrix products. It implements trainable graph heat kernels
and a separable **graph vertex × time** heat/RBF kernel, with controlled synthetic
recovery, identifiability and predictive studies.

[Stage 01](docs/experiment_01_findings.md) establishes numerical agreement between
dense exact and sparse polynomial GP fits; it does not establish reliable
hyperparameter recovery. [Stage 02](docs/experiment_02_findings.md) studies
finite-data identifiability using an independent dense statistical oracle,
including multiple independent realizations sharing one covariance. These small
studies make no claim of general speedup or superiority over GNNs.
[Stage 03](docs/experiment_03_findings.md) documents the finite-grid graph × time
model, independent dense oracle, numerical validation and measured predictive results.

## Install and run

Python 3.12 and CPU float64 are the tested setup.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python -m graphgp.demo --steps 20
python experiments/01_heat_hyperparameter_recovery.py --smoke
python experiments/02_heat_identifiability.py --smoke --output results/identifiability_smoke
python experiments/03_graph_time.py --smoke --output results/graph_time_smoke
```

The original short demo fits a smooth sine signal; it is an API demonstration. The scientific study samples independent latent
GPs from the dense **exact** heat covariance and adds independent Gaussian noise.

```bash
# 3 graph families × 2 regimes × 3 data seeds × 2 starts × 3 methods
python experiments/01_heat_hyperparameter_recovery.py
# Repeat with trainable signal amplitude
python experiments/01_heat_hyperparameter_recovery.py --learn-amplitude --output results/free_amplitude
# Diagnose the three largest fixed-amplitude initialization discrepancies
python experiments/01_optimizer_audit.py
# Regenerate summaries and all plots from saved CSV, without retraining
python experiments/01_heat_hyperparameter_recovery.py --plot-only
# Stage 02 primary study and bounded graph-size check
python experiments/02_heat_identifiability.py
python experiments/02_heat_identifiability.py --secondary
# Bounded additional-start diagnostic; original study results are preserved
python experiments/02_audit_identifiability.py
# Regenerate Stage 02 figures from saved CSV
python experiments/02_plot_identifiability.py --input results/identifiability --secondary-input results/identifiability_secondary --output docs/figures
# Stage 03 fixed four-seed predictive study; refuses an existing configured output
python experiments/03_graph_time.py --output results/graph_time
# Regenerate its three figures from saved CSV, without training or sampling
python experiments/03_plot_graph_time.py --input results/graph_time --output docs/figures
```

Stage 01 saves config/environment JSON, diagnostic CSV, data/prediction NPZ arrays,
and figures under `results/`. Stage 02 saves config/environment JSON and compact
statistical CSV; its plot command reads those saved results. Generated results
are gitignored. Selected evidence and report figures are retained under
`docs/evidence/` and `docs/figures/`.

Stage 03 keeps its main comparison, matched dense/sparse training parity, and
learned-noise diagnostic separate in the saved CSV. Its smoke command is a small
API/numerical check and does not reproduce the full scientific study.

[Migration notes](docs/migration_notes.md) identify source provenance and tests.
[Numerical safety](docs/numerical_safety.md) explains the enforced parameter domain,
polynomial positivity analysis, and exact/approximate inference distinction.

## Minimal API

```python
import torch
from graphgp.utils.graphs import make_laplacian
from graphgp.models.factory import make_heat_gp
from graphgp.models.exact_gp import iterative_settings
from graphgp.inference.predict import predict_marginals

L, scale = make_laplacian("path", n=40)
train_ids = torch.arange(0, 40, 2)
y = torch.sin(train_ids.double() / 6)
model = make_heat_gp(L, train_ids, y, method="C", heat_time=1.0, noise=0.08)
# See graphgp.demo for Adam + ExactMarginalLogLikelihood training.
with iterative_settings(cg_tolerance=1e-4):
    moments = predict_marginals(model, torch.tensor([1, 3, 7]))
    mean_only = predict_marginals(model, torch.tensor([1, 3]), mean_only=True)
```

`noise` is observation **variance**, not standard deviation. `moments` contains
mean, latent variance and observation variance. The model is zero-mean; amplitude
is fixed to one unless `learn_amplitude=True`. Node IDs may be unordered or repeated.
The factory uses the degree-scaled combinatorial Laplacian with spectrum in [0,2]
and heat time in (0.05,4). Raw migrated kernels support broader configurations but
require their own approximation/PSD validation.

## Graph × time example

```python
import gpytorch
import torch
from graphgp.analysis.graph_time_oracle import sample_problem
from graphgp.models.graph_time import make_graph_time_gp
from graphgp.models.exact_gp import iterative_settings
from graphgp.inference.predict import predict_marginals

# Tiny independently sampled problem: historical training, declared future targets.
problem = sample_problem(11, "forecast", graph_kind="path", n=9, num_times=5,
                         forecast_start=3, train_count=12)
pairs = torch.tensor(problem["train_pairs"], dtype=torch.float64)
y = torch.tensor(problem["y"][problem["train_ids"]], dtype=torch.float64)
targets = torch.tensor(problem["test_pairs"][:6], dtype=torch.float64)
model = make_graph_time_gp(problem["L"], pairs, y, num_times=5,
                           backend="sparse_chebyshev", noise_variance=0.06)
model.train()
model.likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
torch.manual_seed(7)
with iterative_settings(cg_tolerance=1e-4, trace_samples=16, lanczos_iterations=12):
    for _ in range(5):
        optimizer.zero_grad()
        loss = -mll(model(pairs), y)
        loss.backward()
        optimizer.step()
    moments = predict_marginals(model, targets, chunk_size=3)
    means = predict_marginals(model, targets, mean_only=True).mean
print(moments.mean, moments.latent_variance, moments.observation_variance)
```

Inputs are unbatched CPU float64 `(m, 2)` tensors containing integer-valued
`(vertex_id, time_index)` coordinates. Declare the complete finite time grid when
constructing the model: index `j` means physical normalized time `j/(num_times-1)`.
Future indices remain on that same grid, without rescaling to the observed time
range; arbitrary continuous observation times and extending the grid after
construction are outside this API. Unordered, missing and repeated pairs are
supported, with independent noise per recorded observation. Latent amplitude is
fixed at one; noise is frozen unless `learn_noise=True`. Both heat time and temporal
lengthscale are trainable. `mean_only=True` avoids target variance extraction.

## Scope

The sparse covariance applies a polynomial of the Laplacian without constructing
the full graph covariance. Exact target variance extraction remains expensive.
The tiny numerical recovery comparison intentionally uses exhaustive trace probes
with quadratic storage. Stage 02's dense statistical oracle also targets small
graphs; neither study is a scaling benchmark. The standalone demo uses random probes.
Inherited CSR/deprecation warnings remain visible; numerical warnings are logged
separately. GPU behavior is unverified.

The graph × time sparse path also retains full `n*T` workspace and a dense `T*T`
temporal factor. Its exact requested target variances can be expensive; sparse
graph products alone do not establish general GP scalability. CPU float64 is the
validated configuration.

Kernel mixtures, structured observation compression and scalable target-variance
estimators remain future work. See the [roadmap](docs/research_roadmap.md) and clearly separated
[historical results](docs/prior_results.md). No large old results tree is included.

## Collaboration

The GitHub repository is public. No license has been selected; redistribution
terms have not been specified.
