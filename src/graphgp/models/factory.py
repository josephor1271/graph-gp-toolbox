"""Matched, bounded models for the first recovery experiment."""
import gpytorch
import numpy as np
import scipy.sparse
import torch
from graphgp.kernels.heat import GraphHeatKernel
from graphgp.kernels.dense import DenseExactGraphHeatKernel, DenseChebyshevGraphHeatKernel
from graphgp.kernels.safety import heat_interpolation_certificate
from graphgp.models.exact_gp import GraphGP

HEAT_RANGE = (0.05, 4.0)
NOISE_RANGE = (0.005, 0.5)
AMPLITUDE_RANGE = (0.1, 4.0)


def validate_study_laplacian(L) -> scipy.sparse.csr_matrix:
    """Validate a fixed SciPy Laplacian with maximum weighted degree <= 1.

    Return float64 CSR, retaining the degree-scaled combinatorial convention
    and its analytically enclosing spectral interval [0, 2].
    """
    if not scipy.sparse.issparse(L):
        raise TypeError("expected a SciPy sparse Laplacian")
    if L.ndim != 2 or not L.shape[0] or L.shape[0] != L.shape[1]:
        raise ValueError("Laplacian must be nonempty and square")
    if np.iscomplexobj(L.data):
        raise ValueError("Laplacian must be real")
    L = L.astype(np.float64).tocsr()
    off = L - scipy.sparse.diags(L.diagonal())
    asymmetry = L-L.T
    if (not np.isfinite(L.data).all() or
        (asymmetry.nnz and np.max(np.abs(asymmetry.data)) > 1e-12) or
        np.max(np.abs(np.asarray(L.sum(axis=1)))) > 1e-12 or
        np.any(off.data > 1e-12) or np.any(L.diagonal() < 0) or
        np.max(L.diagonal()) > 1+1e-12):
        raise ValueError("expected symmetric Laplacian with maximum weighted degree <= 1")
    return L


def make_heat_gp(L, train_ids, train_y, method="C", *, heat_time=1.0,
                 noise=0.08, amplitude=1.0, learn_amplitude=False, degree=32):
    """Float64 GP on a fixed Laplacian with spectrum enclosed by [0,2].

    Generic migrated kernels remain available for other validated domains.
    This factory enforces the study's narrower numerical contract.
    """
    L = validate_study_laplacian(L)
    heat_interpolation_certificate(degree)
    for name, value, bounds in [('heat_time',heat_time,HEAT_RANGE),
                                ('noise',noise,NOISE_RANGE),
                                ('amplitude',amplitude,AMPLITUDE_RANGE)]:
        if not bounds[0] < value < bounds[1]:
            raise ValueError(f"{name} initialization must lie strictly inside {bounds}")
    if method in ('A','A2'):
        kernel = DenseExactGraphHeatKernel(L, heat_time=heat_time)
    elif method in ('B','B2','C'):
        cls = GraphHeatKernel if method == 'C' else DenseChebyshevGraphHeatKernel
        kernel = cls(L, lambda_max=2, num_coefficients=degree+1,
                     num_nodes=degree+1, heat_time=heat_time)
    else:
        raise ValueError(f"unknown method: {method}")
    kernel.register_constraint('raw_heat_time', gpytorch.constraints.Interval(*HEAT_RANGE).double())
    kernel.heat_time = heat_time
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.Interval(*NOISE_RANGE).double()).double()
    likelihood.noise = torch.as_tensor(noise, dtype=torch.float64)
    scaled = gpytorch.kernels.ScaleKernel(kernel,
        outputscale_constraint=gpytorch.constraints.Interval(*AMPLITUDE_RANGE).double()).double()
    scaled.outputscale = torch.as_tensor(amplitude, dtype=torch.float64)
    scaled.raw_outputscale.requires_grad_(learn_amplitude)
    return GraphGP(train_ids, train_y, likelihood, scaled).double()
