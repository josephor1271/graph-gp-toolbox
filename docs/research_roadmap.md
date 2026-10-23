# Research roadmap

The first implementation pass covers A–C: selective migration, numerical fidelity,
and controlled synthetic heat-kernel recovery. Statistical non-identifiability is
an experimental finding, not something to hide with a favorable seed.

Next, expand independent draws and assess optimizer convergence in regimes where
initializations disagree. Distinguish finite-sample uncertainty from a constrained
boundary optimum, and keep a dense exact likelihood surface as the diagnostic.

Stage D: one nonnegative weighted sum of two heat kernels with an exact dense
reference. Assess covariance/spectral-response recovery as well as weights and
shape parameters; account for label symmetry. Matrix products f(L)g(L) and
Hadamard kernel products are different. Two heat matrix factors merely add heat
times, so their matrix product adds no new family. A latent nugget and observation
noise should not be learned redundantly without an identification argument.

Stage E (main strategic extension): a tiny separable graph×time heat/RBF GP,
with explicitly documented vertex-major/time-minor order and dense exact
K_graph ⊗ K_time sampling. Validate gradients, likelihood and requested target
marginals. Then use a sparse graph factor in a Kronecker action K_graph X K_timeᵀ,
with gather/scatter for missing or repeated node-time pairs. This still uses full
vertex×time storage and a dense temporal factor. Compare graph-only, time-only and
joint models; distinguish random interpolation splits from future forecasting.

Later: temporal compression, validated approximate variance estimators, realistic
data and appropriate baselines. Broad kernel catalogs, GNN benchmark suites,
cross-graph transfer, custom solvers and GPU specialization are deferred.
