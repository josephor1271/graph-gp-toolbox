"""Independent small dense benchmark references, never for sparse workers.

The covariance/conditioning reference uses Stage 03's independent SciPy/NumPy
oracle. Raw derivatives use an independent Torch matrix-exponential expression
and the public Interval transforms defining the model's raw coordinates. No
production kernel is imported.
"""
from typing import Any

import gpytorch
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import torch

from graphgp.analysis.graph_time_oracle import (
    dense_prior, exact_nll, flat_pair_indices, observed_covariance, posterior_marginals,
)
from graphgp.utils.study_domain import HEAT_RANGE, LENGTHSCALE_RANGE


def _raw_gradients(problem: dict[str, Any], tau: float, ell: float,
                   noise_variance: float) -> dict[str, float]:
    raw = []
    physical = []
    for value, (lower, upper) in ((tau, HEAT_RANGE), (ell, LENGTHSCALE_RANGE)):
        if not lower < value < upper:
            raise ValueError("reference parameters must lie strictly within study bounds")
        # Match the installed public constraint's stored bound precision: the
        # factory constructs Interval before converting it to float64. Using
        # ideal Python bounds instead would define slightly different *raw*
        # coordinates even when physical tau and ell agree exactly.
        constraint = gpytorch.constraints.Interval(lower, upper).double()
        parameter = constraint.inverse_transform(torch.tensor(value, dtype=torch.float64))
        parameter = parameter.detach().requires_grad_(True)
        raw.append(parameter)
        physical.append(constraint.transform(parameter))
    heat_time, lengthscale = physical
    laplacian = torch.tensor(problem["L"].toarray(), dtype=torch.float64)
    time = torch.tensor(problem["time_grid"], dtype=torch.float64)
    graph = torch.matrix_exp(-heat_time * laplacian)
    temporal = torch.exp(-0.5 * (time[:, None] - time[None, :]).square() / lengthscale.square())
    covariance = torch.kron(graph, temporal)
    pairs = torch.tensor(problem["train_pairs"], dtype=torch.int64)
    ids = pairs[:, 0] * len(time) + pairs[:, 1]
    covariance = covariance[ids][:, ids] + noise_variance * torch.eye(len(ids), dtype=torch.float64)
    observations = torch.tensor(problem["train_y"], dtype=torch.float64)
    factor = torch.linalg.cholesky(covariance)
    loss = (2 * factor.diagonal().log().sum()
            + observations @ torch.cholesky_solve(observations[:, None], factor).squeeze(-1)
            + len(ids) * np.log(2 * np.pi)) / (2 * len(ids))
    gradients = torch.autograd.grad(loss, raw)
    return dict(zip(("heat_time", "lengthscale"), (float(value) for value in gradients)))


def dense_reference(problem: dict[str, Any], case, *, tau: float = 1.2,
                    ell: float = 0.24, noise_variance: float = 0.06,
                    target_count: int | None = None) -> dict[str, Any]:
    """JSON-ready exact NLL, raw gradients, target moments and solve reference.

    This must be called in an explicitly labeled independent reference worker.
    The small-case ceiling is enforced before constructing dense matrices.
    ``problem`` supplies the case's deterministic ``solve_rhs`` and ``mvp_rhs``;
    corresponding reference products use the same arrays as sparse workers.
    ``target_count`` explicitly declares the fixed solver-screening target
    prefix; omitting it requires all ``case.q`` targets as before.
    """
    n, total_times, m, q = case.n, case.num_times, case.m, case.q
    if not (n <= 100 and n * total_times <= 800 and m <= 400 and q <= 256):
        raise ValueError("case exceeds the explicit small dense-reference ceiling")
    if target_count is not None:
        if isinstance(target_count, bool) or not isinstance(target_count, int) or not 1 <= target_count <= q:
            raise ValueError("target_count must be an integer in [1,case.q]")
        q = target_count
    if (problem["L"].shape != (n, n) or len(problem["time_grid"]) != total_times
            or np.shape(problem["train_pairs"]) != (m, 2)
            or np.shape(problem["train_y"]) != (m,)
            or np.shape(problem["target_pairs"]) != (q, 2)):
        raise ValueError("problem dimensions do not match the declared reference case")
    rhs = np.asarray(problem["solve_rhs"], dtype=np.float64)
    mvp_rhs = np.asarray(problem["mvp_rhs"], dtype=np.float64)
    if (rhs.shape != (m, 3) or mvp_rhs.shape != (n * total_times, 8)
            or not np.isfinite(rhs).all() or not np.isfinite(mvp_rhs).all()
            or not np.array_equal(rhs[:, 0], problem["train_y"])):
        raise ValueError("reference RHS must be finite paired solve (m,3) and MVP (nT,8) arrays")
    prior = dense_prior(problem["L"], problem["time_grid"], tau, ell)
    selected = observed_covariance(prior, problem["train_pairs"],
                                   noise_variance=noise_variance, num_times=total_times)
    factor = cho_factor(selected, lower=True)
    moments = posterior_marginals(
        prior, problem["train_pairs"], problem["train_y"], problem["target_pairs"],
        noise_variance, num_times=total_times)
    # Explicit index validation here also documents that the solve is selected
    # recording space with noise on its diagonal, not a graph-only solve.
    flat_pair_indices(problem["train_pairs"], n, total_times)
    return dict(
        nll=exact_nll(prior, problem["train_pairs"], problem["train_y"],
                      noise_variance, num_times=total_times),
        raw_gradients=_raw_gradients(problem, tau, ell, noise_variance),
        mean=moments["mean"].tolist(), latent_variance=moments["latent_var"].tolist(),
        observation_variance=moments["observation_var"].tolist(),
        exact_solutions=cho_solve(factor, rhs).tolist(),
        prior_mvp=(prior @ mvp_rhs).tolist(),
        reference="independent scipy expm/NumPy RBF/Cholesky; independent Torch raw derivatives",
    )
