# Numerical contract for experiment 01

Graphs are undirected, connected, unit-weight paths, grids, and fixed-seed
Barabási–Albert graphs. We divide the combinatorial Laplacian by its maximum degree.
This is scalar rescaling, not the symmetric normalized Laplacian. The resulting
maximum weighted degree is one and its spectrum is enclosed by [0,2]. Heat-time
units are tied to this scaling; comparisons with unscaled historical graphs need
conversion. The graph topology is fixed across data draws (graph seed 2718).

The model factory enforces heat time in (0.05,4), observation noise **variance** in
(0.005,0.5), and amplitude in (0.1,4). Amplitude is fixed to one unless requested.
All use GPyTorch Interval transforms; initial values must be strictly inside the
bounds. Generic migrated kernels are unrestricted building blocks and carry no
such domain guarantee. The factory validates symmetric Laplacian sign/row-sum and
degree conditions and rejects inappropriate inputs.

For degree d=32 and m=d+1=33 Chebyshev interpolation nodes, on the rescaled interval
[-1,1], f(x)=exp(-z(x+1)), z=tau*lambda_max/2 <= 4. Its m-th derivative is bounded
by z^m. The monic product of Chebyshev nodes has maximum norm 2^(1-m). The Lagrange
remainder therefore gives a uniform real-arithmetic error bound

    |f(x)-p_d(x)| <= 2*(z/2)^m/m! <= 1.979e-27.

The exact spectral response is at least exp(-8)=0.000335462628. Thus the polynomial
is positive over the entire spectral and heat-time range in real arithmetic.
This is stronger than checking a sampled scalar grid. The implementation also
requires a conservative practical roundoff allowance of 100*m²*epsilon, about
2.42e-11, below that margin. This allowance is a numerical engineering margin,
**not a formally verified floating-point error bound**. Float64 tests sweep the
training domain and actual graph eigenvalues, and compare against independent
SciPy matrix exponentials. Coarse degrees can fail positivity and are rejected
by the factory. Outside this contract, revalidate bounds, degree and arithmetic.

Dense A uses exact eigendecomposition and Cholesky. Dense B evaluates the same
polynomial at eigenvalues and uses Cholesky. Sparse C applies the migrated CSR
recurrence using CG and SLQ. C has max_cholesky_size=0, max_preconditioner_size=0,
max_eager_kernel_size=0, CG tolerance 1e-4, maximum 500 CG iterations and n_train
Lanczos steps. Exhaustive orthonormal trace probes eliminate trace sampling error
for these tiny comparisons, at O(n_train²) storage. They are a small-validation
oracle, not the scalable production setting. Tests also exercise random probes.
No custom CG solver or logdet estimator is implemented.

The inference context guards against base graph covariance `to_dense()` and
LinearOperator Cholesky on C. These are regression guards for known fallback
routes, not a proof against all possible future densification paths. Small dense
oracles and explicit tiny matrix checks intentionally materialize covariance.
Exact target variances still have substantial cost; approximate large-scale
uncertainty is deferred. Prediction reports latent variance and observation
variance = latent + noise. Target marginals do not support joint posterior
sampling or joint event probabilities.
