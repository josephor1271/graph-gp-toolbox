"""Bounded GPyTorch ExactGP factory for a fixed graph × finite time grid."""

import gpytorch
import torch

from graphgp.kernels.graph_time import GraphTimeHeatRBFKernel, _initial_value
from graphgp.models.exact_gp import GraphGP
from graphgp.models.factory import NOISE_RANGE


def make_graph_time_gp(L, train_pairs: torch.Tensor, train_y: torch.Tensor, *,
                       num_times: int, backend: str, heat_time: float = 0.7,
                       lengthscale: float = 0.4, noise_variance: float = 0.06,
                       learn_noise: bool = False, degree: int = 32,
                       scaled_laplacian: bool = False) -> GraphGP:
    """Make a zero-mean, unit-amplitude graph × time ExactGP in CPU float64.

    ``train_pairs`` is nonempty ``(m, 2)`` with integer-valued vertex/time
    indices; ``train_y`` is finite ``(m,)``. The time grid is fixed at
    ``j/(num_times-1)`` including forecast positions. Noise is a variance,
    independent per recorded observation, and its raw parameter is frozen
    unless ``learn_noise=True``. Both kernel parameters remain trainable.

    Use explicit direct/iterative inference settings appropriate to ``backend``
    and ``predict_marginals`` for means and requested target variances. Sparse
    MVPs retain full-grid workspace; exact marginal variances can be expensive.
    """
    if not isinstance(learn_noise, bool):
        raise ValueError("learn_noise must be a bool")
    kernel = GraphTimeHeatRBFKernel(
        L, num_times=num_times, backend=backend, heat_time=heat_time,
        lengthscale=lengthscale, degree=degree, scaled_laplacian=scaled_laplacian)
    kernel.validate_pairs(train_pairs)
    if (not torch.is_tensor(train_y) or train_y.ndim != 1
            or train_y.shape[0] != train_pairs.shape[0]
            or train_y.dtype != torch.float64 or train_y.device.type != "cpu"
            or not torch.isfinite(train_y).all()):
        raise ValueError("train_y must be finite CPU float64 with shape (m,)")
    noise = _initial_value(noise_variance, NOISE_RANGE, "noise_variance")
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.Interval(*NOISE_RANGE).double()).double()
    likelihood.noise = noise
    likelihood.noise_covar.raw_noise.requires_grad_(learn_noise)
    return GraphGP(train_pairs, train_y, likelihood, kernel).double()
