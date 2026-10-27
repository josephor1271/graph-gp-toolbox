# Research roadmap

Stage 01 is complete: selective migration, numerical fidelity and controlled
synthetic heat-kernel recovery. Dense exact, dense polynomial and sparse
polynomial methods agree numerically; reliable finite-data parameter recovery
was not established. See [the Stage 01 report](experiment_01_findings.md).

Stage 02 is complete: an independent dense statistical oracle, likelihood/KL and
Fisher diagnostics, independent-draw experiments and additional optimizer audits.
More independent draws improved several recovery measures, while joint noise
recovery and some optimizer disagreements remain limitations. Weak finite-data
identification is a measured finding even where local Fisher information is
nonsingular. See [the Stage 02 report](experiment_02_findings.md).

Stage 03 is complete: one separable graph × finite-time-grid heat/RBF GP through
GPyTorch, with unit latent amplitude and vertex-major/time-minor coordinates.
An independent dense oracle checks covariance action, gradients, likelihood and
requested target marginals. The sparse graph factor acts through a Kronecker
operator, while arbitrary missing and repeated node-time pairs use observation
selection. This retains full `n*T` workspace and a dense `T*T` temporal factor.
The fixed predictive study compares graph-only, time-only and joint models,
separating random-pair interpolation from future forecasting and keeping matched
dense/sparse training parity separate from the main comparison. The
[Stage 03 report](experiment_03_findings.md) records measured outcomes and the
prioritized next step; predictions do not have to favor the joint model.

Stage 04 measures the current operator and GPyTorch inference pathway under
fixed graph/time/observation/target workloads. It separates polynomial error,
independently computed noisy-solve residuals and stochastic MLL/gradient error;
isolated processes measure cold operation time and whole-worker peak RSS.
The bounded refactor preserves old APIs and scientific thresholds. See the
[Stage 04 report](experiment_04_findings.md) for actual accuracy qualification,
performance limits, failed configurations and the selected next-stage decision.

Stage 04 prioritizes a separate accuracy-and-cost study of approximate target
marginal variances. Exact marginal variance extraction remains the reference;
an estimator must meet independent error gates before replacing it. Improved
iterative solves, trace estimation, structured observation compression and
temporal compression remain separate later decisions.

Kernel composition is a separate future stage: start with a nonnegative weighted
sum of two heat kernels and an independent dense reference, assess covariance and
spectral-response recovery as well as weights, and account for label symmetry.
Matrix products `f(L)g(L)` and Hadamard kernel products are different; two heat
matrix factors merely add heat times. Learning a latent nugget alongside
observation noise requires an identification argument.

Applied datasets and suitable baselines follow numerical validation and a bounded
cost study. Broad kernel catalogs, GNN benchmark suites, cross-graph transfer,
custom solvers and GPU specialization remain deferred.
