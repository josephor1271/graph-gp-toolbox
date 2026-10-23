"""Exact dense reference for the fixed-amplitude heat identifiability study.

This CPU float64 oracle deliberately materializes small observed covariances.
It is independent of production GPyTorch, Chebyshev approximation and plotting.
Noise parameters are variances. No factorization failure receives hidden jitter.
"""

import json

import numpy as np
from scipy.linalg import cholesky, cho_solve, eigh
from scipy.optimize import minimize

# The same physical domain as models.factory, including endpoints for this
# bounded statistical reference (the production Interval transform is open).
HEAT_RANGE = (0.05, 4.0)
NOISE_RANGE = (0.005, 0.5)
DEFAULT_INITIALS = ((0.3, 0.04), (2.8, 0.2))


class HeatOracle:
    """Cache one full-graph eigensystem; select nodes after exponentiation."""

    def __init__(self, laplacian):
        dense = laplacian.toarray() if hasattr(laplacian, "toarray") else laplacian
        self.laplacian = np.array(dense, dtype=np.float64, copy=True)
        if (self.laplacian.ndim != 2 or self.laplacian.shape[0] == 0
                or self.laplacian.shape[0] != self.laplacian.shape[1]
                or not np.isfinite(self.laplacian).all()
                or not np.allclose(self.laplacian, self.laplacian.T,
                                   rtol=0, atol=1e-12)):
            raise ValueError("laplacian must be a finite nonempty symmetric matrix")
        self.eigenvalues, self.eigenvectors = eigh(self.laplacian)
        if self.eigenvalues[0] < -1e-12:
            raise ValueError("laplacian must be positive semidefinite")
        self.n = len(self.eigenvalues)

    def _ids(self, observed_ids):
        if observed_ids is None:
            return np.arange(self.n)
        ids = np.asarray(observed_ids)
        if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "iu"
                or np.any(ids < 0) or np.any(ids >= self.n)
                or len(np.unique(ids)) != len(ids)):
            raise ValueError("observed_ids must be distinct valid integer node indices")
        return ids

    @staticmethod
    def _parameters(tau, noise_variance):
        if (not np.isfinite(tau) or tau <= 0
                or not np.isfinite(noise_variance) or noise_variance < 0):
            raise ValueError("heat time must be positive and noise variance nonnegative")

    def _covariance_derivatives(self, tau, noise_variance, observed_ids):
        self._parameters(tau, noise_variance)
        ids = self._ids(observed_ids)
        vectors = self.eigenvectors[ids]
        response = np.exp(-tau * self.eigenvalues)
        # These are submatrices of the FULL graph heat kernel and its derivative;
        # exponentiating the principal Laplacian submatrix is a different model.
        heat = (vectors * response) @ vectors.T
        heat_derivative = (vectors * (-self.eigenvalues * response)) @ vectors.T
        identity = np.eye(len(ids))
        return heat + noise_variance * identity, (heat_derivative, identity)

    def covariance(self, tau, noise_variance, observed_ids=None):
        """Return [exp(-tau L)]_OO + noise_variance I in supplied ID order."""
        return self._covariance_derivatives(tau, noise_variance, observed_ids)[0]

    def _data(self, Y, observed_ids):
        ids = self._ids(observed_ids)
        data = np.asarray(Y, dtype=np.float64)
        if data.ndim == 1:
            data = data[:, None]
        if (data.ndim != 2 or data.shape[0] != len(ids)
                or data.shape[1] < 1 or not np.isfinite(data).all()):
            raise ValueError("Y must be finite observed_nodes by independent_realizations")
        return data, ids

    def physical_nll_and_gradient(self, tau, noise_variance, Y, observed_ids=None):
        """NLL per observed node per realization; physical-parameter gradient."""
        data, ids = self._data(Y, observed_ids)
        covariance, derivatives = self._covariance_derivatives(tau, noise_variance, ids)
        factor = cholesky(covariance, lower=True)
        alpha = cho_solve((factor, True), data)
        m, r = data.shape
        logdet = 2 * np.log(np.diag(factor)).sum()
        value = (r * logdet + np.sum(data * alpha) + r*m*np.log(2*np.pi)) / (2*m*r)
        gradient = np.array([
            (r*np.trace(cho_solve((factor, True), derivative))
             - np.sum(alpha * (derivative @ alpha))) / (2*m*r)
            for derivative in derivatives
        ])
        if not np.isfinite(value) or not np.isfinite(gradient).all():
            raise FloatingPointError("nonfinite exact NLL or gradient")
        return float(value), gradient

    def nll_and_gradient(self, log_parameters, Y, observed_ids=None, known_noise=None):
        """Analytic log-coordinate objective, with optional fixed-noise control."""
        eta = np.asarray(log_parameters, dtype=np.float64)
        expected_shape = (2,) if known_noise is None else (1,)
        if eta.shape != expected_shape or not np.isfinite(eta).all():
            raise ValueError(f"log_parameters must have shape {expected_shape}")
        parameters = np.exp(eta)
        tau = parameters[0]
        noise = parameters[1] if known_noise is None else known_noise
        value, gradient = self.physical_nll_and_gradient(tau, noise, Y, observed_ids)
        return value, gradient[:len(eta)] * parameters

    def nll(self, tau, noise_variance, Y, observed_ids=None):
        """Exact average NLL using Cholesky solves, with no adaptive jitter."""
        data, ids = self._data(Y, observed_ids)
        factor = cholesky(self.covariance(tau, noise_variance, ids), lower=True)
        m, r = data.shape
        value = (2*np.log(np.diag(factor)).sum()
                 + np.sum(data * cho_solve((factor, True), data))/r
                 + m*np.log(2*np.pi))/(2*m)
        if not np.isfinite(value):
            raise FloatingPointError("nonfinite exact NLL")
        return float(value)

    def expected_excess(self, tau, noise, true_tau, true_noise, observed_ids=None):
        """KL(true || candidate) per observed node; not a sample objective."""
        candidate = self.covariance(tau, noise, observed_ids)
        truth = self.covariance(true_tau, true_noise, observed_ids)
        factor = cholesky(candidate, lower=True)
        truth_factor = cholesky(truth, lower=True)
        m = len(candidate)
        return float((np.trace(cho_solve((factor, True), truth)) - m
                      + 2*np.log(np.diag(factor)).sum()
                      - 2*np.log(np.diag(truth_factor)).sum()) / (2*m))

    def fisher(self, true_tau, true_noise, observed_ids=None, *, replicates=1,
               log_coordinates=True):
        """Unnormalized total information for independent draws on one mask."""
        if not isinstance(replicates, (int, np.integer)) or replicates < 1:
            raise ValueError("replicates must be a positive integer")
        covariance, derivatives = self._covariance_derivatives(
            true_tau, true_noise, observed_ids)
        if log_coordinates:
            derivatives = (true_tau * derivatives[0], true_noise * derivatives[1])
        factor = cholesky(covariance, lower=True)
        solved = [cho_solve((factor, True), derivative) for derivative in derivatives]
        result = np.array([[replicates * np.sum(left * right.T) / 2
                            for right in solved] for left in solved])
        return (result + result.T) / 2

    def fisher_diagnostics(self, true_tau, true_noise, observed_ids=None, *,
                           replicates=1, log_coordinates=True):
        """Dimensionless diagnostics by default; no singular-matrix correlation."""
        information = self.fisher(true_tau, true_noise, observed_ids,
                                  replicates=replicates, log_coordinates=log_coordinates)
        eigenvalues = np.linalg.eigvalsh(information)
        nonsingular = bool(eigenvalues[0] > 100*np.finfo(float).eps*max(eigenvalues[-1], 1))
        # For a nonsingular 2x2 information matrix, the inverse-information
        # correlation is exactly -I_12/sqrt(I_11 I_22); no inverse is needed.
        correlation = (-information[0, 1] / np.sqrt(information[0, 0]*information[1, 1])
                       if nonsingular else None)
        return dict(lambda_min=float(eigenvalues[0]), lambda_max=float(eigenvalues[1]),
                    condition_number=float(eigenvalues[1]/eigenvalues[0]) if nonsingular else None,
                    log_parameter_correlation=float(correlation) if nonsingular else None,
                    information_status="nonsingular" if nonsingular else "numerically_singular",
                    information_00=float(information[0, 0]),
                    information_01=float(information[0, 1]),
                    information_11=float(information[1, 1]))


def sample_pool(oracle, true_tau, true_noise, max_replicates, pool_seed, *,
                mask_seed=None, stream_key=()):
    """Full-graph independent latent/noise draws with stable nested prefixes.

    SeedSequence spawn keys 0, 1, 2 separate latent, noise and mask streams.
    Numeric stream_key entries identify a scenario without unstable Python hashes.
    Mask generation is independent of R and can use a shared explicit mask_seed.
    Arrays have shape (vertices, realizations). A shorter budget is a column prefix.
    """
    if not isinstance(max_replicates, (int, np.integer)) or max_replicates < 1:
        raise ValueError("max_replicates must be a positive integer")
    oracle._parameters(true_tau, true_noise)
    seed_key = tuple(stream_key)
    latent_rng = np.random.default_rng(np.random.SeedSequence([pool_seed, *seed_key], spawn_key=(0,)))
    noise_rng = np.random.default_rng(np.random.SeedSequence([pool_seed, *seed_key], spawn_key=(1,)))
    permutation_rng = np.random.default_rng(np.random.SeedSequence(
        [pool_seed if mask_seed is None else mask_seed, *seed_key], spawn_key=(2,)))
    factor = cholesky(oracle.covariance(true_tau, 0), lower=True)
    latent = factor @ latent_rng.standard_normal((max_replicates, oracle.n)).T
    noise = np.sqrt(true_noise) * noise_rng.standard_normal((max_replicates, oracle.n)).T
    return dict(latent=latent, noise=noise, observations=latent+noise,
                permutation=permutation_rng.permutation(oracle.n))


def _projected_gradient(eta, gradient, bounds):
    projected = gradient.copy()
    for index, (lower, upper) in enumerate(bounds):
        if ((eta[index] <= lower + 1e-10 and gradient[index] > 0)
                or (eta[index] >= upper - 1e-10 and gradient[index] < 0)):
            projected[index] = 0
    return projected


def fit_multistart(oracle, Y, observed_ids=None, *, known_noise=None,
                   initials=DEFAULT_INITIALS, maxiter=250, gtol=1e-9, ftol=1e-13):
    """Retain all bounded L-BFGS-B outcomes and select lowest actual valid NLL.

    Numerical validity, optimizer success and projected-gradient stationarity
    are separate fields. Selection is not a global-optimum certificate.
    """
    data, ids = oracle._data(Y, observed_ids)
    if known_noise is not None and not NOISE_RANGE[0] <= known_noise <= NOISE_RANGE[1]:
        raise ValueError("known_noise must lie in the study's noise-variance range")
    if len(initials) < 2:
        raise ValueError("at least two initialization points are required")
    physical_bounds = [HEAT_RANGE] if known_noise is not None else [HEAT_RANGE, NOISE_RANGE]
    bounds = [tuple(np.log(interval)) for interval in physical_bounds]
    rows = []
    for init_id, initial in enumerate(initials):
        if len(initial) != 2 or any(not np.isfinite(x) or not lo <= x <= hi
                                    for x, (lo, hi) in zip(initial, (HEAT_RANGE, NOISE_RANGE))):
            raise ValueError("each initial must be a heat/noise pair inside physical bounds")
        x0 = np.log(initial[:len(bounds)])
        row = dict(init_id=init_id, initial_tau=float(initial[0]),
                   initial_noise_variance=float(known_noise if known_noise is not None else initial[1]),
                   fit_tau=None, fit_noise_variance=None, nll_per_observation=None,
                   valid=False, optimizer_success=False, optimizer_message="", num_iterations=0,
                   gradient_norm=None, projected_gradient_norm=None, stationary=False,
                   log_gradient="[]", projected_log_gradient="[]",
                   boundary_flags="[]", tau_near_boundary=False, noise_near_boundary=False)
        try:
            result = minimize(oracle.nll_and_gradient, x0,
                              args=(data, ids, known_noise), method="L-BFGS-B", jac=True,
                              bounds=bounds, options=dict(maxiter=maxiter, gtol=gtol, ftol=ftol,
                                                         maxls=40))
            estimate = np.exp(result.x)
            tau = float(estimate[0])
            noise = float(known_noise if known_noise is not None else estimate[1])
            actual_nll, gradient = oracle.nll_and_gradient(result.x, data, ids, known_noise)
            projected = _projected_gradient(result.x, gradient, bounds)
            flags = []
            for name, value, interval in (("tau", tau, HEAT_RANGE), ("noise", noise, NOISE_RANGE)):
                near = min(value-interval[0], interval[1]-value) <= 0.01*(interval[1]-interval[0])
                if name == "noise" and known_noise is not None:
                    near = False
                row[name+"_near_boundary"] = bool(near)
                if near:
                    side = "lower" if value-interval[0] < interval[1]-value else "upper"
                    flags.append(name+"_"+side)
            row.update(fit_tau=tau, fit_noise_variance=noise,
                       nll_per_observation=float(actual_nll), valid=True,
                       optimizer_success=bool(result.success), optimizer_message=str(result.message),
                       num_iterations=int(result.nit), gradient_norm=float(np.linalg.norm(gradient)),
                       projected_gradient_norm=float(np.linalg.norm(projected, ord=np.inf)),
                       log_gradient=json.dumps(gradient.tolist()),
                       projected_log_gradient=json.dumps(projected.tolist()),
                       stationary=bool(np.linalg.norm(projected, ord=np.inf) <= 1e-6),
                       boundary_flags=json.dumps(flags))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            row["optimizer_message"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    valid = [row for row in rows if row["valid"]]
    if valid:
        selected = min(valid, key=lambda row: row["nll_per_observation"]).copy()
        spread = max(row["nll_per_observation"] for row in valid)-selected["nll_per_observation"]
        parameter_spread = max(np.max(np.abs(np.log([
            row["fit_tau"]/selected["fit_tau"], row["fit_noise_variance"]/selected["fit_noise_variance"]])))
                               for row in valid)
        disagreement = bool(spread > 1e-7 or parameter_spread > 0.01)
        status = ("selected_stationary" if selected["stationary"] and selected["optimizer_success"]
                  else "selected_not_converged")
    else:
        selected = rows[0].copy()
        spread, disagreement, status = None, False, "no_valid_start"
    for row in [*rows, selected]:
        row.update(nll_start_spread=spread, start_disagreement=disagreement, selection_status=status)
    return rows, selected
