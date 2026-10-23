# Historical context (not measurements from this repository)

The legacy development, validation, performance and adaptive-uncertainty reports
established a useful sparse Chebyshev/GPyTorch implementation. The original
files remain in the separate `gpytorch-ex` workspace; they are not copied here.

Reported historical 8,100-node, 50%-observed grid, Apple M4 CPU, float64, one
Torch thread: fixed P128/Q16 achieved 7.24× end-to-end speedup at 20 training steps
and 4.24× at 50 steps against the selected dense exact baseline. At 50 steps,
peak RSS was 685 versus 2,509 MiB; latent variance relative error was about 16.47%.
A later adaptive structured-probe setup reported 3.58× speedup, 7.76% latent and
1.92% observation variance error on the comparable 50-step workload. At only
20% observed, speedup fell to about 1.05× at 8,100 nodes and below one at 6,400.

These workload-specific numbers have **not been rerun here**. They do not
establish speedups for new kernels, mixtures, graph×time models, GPUs or general
observation patterns. No large benchmark tree or old result plots were migrated.
The new study asks about numerical agreement and finite-data learning.
