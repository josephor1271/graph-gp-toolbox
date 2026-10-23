# Graph GP toolbox

A small research codebase for Gaussian processes on fixed graphs, using GPyTorch
and sparse Chebyshev matrix products. The intended direction is compositional
kernels and functions of **graph vertex plus time**. The current implementation
covers a trainable heat kernel and controlled synthetic recovery and
identifiability studies.

[Stage 01](docs/experiment_01_findings.md) establishes numerical agreement between
dense exact and sparse polynomial GP fits; it does not establish reliable
hyperparameter recovery. [Stage 02](docs/experiment_02_findings.md) studies
finite-data identifiability using an independent dense statistical oracle,
including multiple independent realizations sharing one covariance. These small
studies make no claim of general speedup or superiority over GNNs.

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
# Regenerate Stage 02 figures from saved CSV
python experiments/02_plot_identifiability.py --input results/identifiability --secondary-input results/identifiability_secondary --output docs/figures
```

Stage 01 saves config/environment JSON, diagnostic CSV, data/prediction NPZ arrays,
and figures under `results/`. Stage 02 saves config/environment JSON and compact
statistical CSV; its plot command reads those saved results. Generated results
are gitignored. Selected evidence and report figures are retained under
`docs/evidence/` and `docs/figures/`.

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

## Scope

The sparse covariance applies a polynomial of the Laplacian without constructing
the full graph covariance. Exact target variance extraction remains expensive.
The tiny numerical recovery comparison intentionally uses exhaustive trace probes
with quadratic storage. Stage 02's dense statistical oracle also targets small
graphs; neither study is a scaling benchmark. The standalone demo uses random probes.
Inherited CSR/deprecation warnings remain visible; numerical warnings are logged
separately. GPU behavior is unverified.

Kernel mixtures, graph×time/Kronecker models and adaptive uncertainty are not yet
implemented. See the [roadmap](docs/research_roadmap.md) and clearly separated
[historical results](docs/prior_results.md). No large old results tree is included.

## Collaboration

The GitHub repository is public. No license has been selected; redistribution
terms have not been specified.
