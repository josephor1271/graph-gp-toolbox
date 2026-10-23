"""Ordinary dense graph heat kernels for small-graph GPyTorch references.

These kernels intentionally pay the dense eigendecomposition/storage cost at
construction and return dense covariance blocks. They are independent of the
custom sparse operator, so comparisons can separate polynomial approximation,
iterative inference, and operator representation errors.
"""

import gpytorch
import scipy.sparse
import torch

from graphgp.operators.coefficients import (
    chebyshev_basis_on_interval_torch,
    chebyshev_coefficients_from_samples_torch,
    evaluate_chebyshev_polynomial_torch,
    laplacian_eigenvalue_bounds,
)


def _dense_laplacian(L):
    """Validate and densify once without importing the sparse operator module."""
    if scipy.sparse.issparse(L):
        dense = torch.as_tensor(L.toarray())
    elif torch.is_tensor(L) and L.layout in (torch.strided, torch.sparse_csr):
        if L.requires_grad:
            raise ValueError("L must be fixed (requires_grad=False)")
        dense = L.to_dense() if L.layout == torch.sparse_csr else L
    else:
        raise TypeError("L must be a Torch dense/CSR tensor or SciPy sparse matrix")
    if dense.is_complex():
        raise TypeError("L must be real")
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1] or dense.shape[0] == 0:
        raise ValueError("L must be a nonempty square matrix")
    if not dense.is_floating_point():
        dense = dense.to(dtype=torch.get_default_dtype())
    if not torch.isfinite(dense).all():
        raise ValueError("L must have finite entries")
    if not torch.allclose(dense, dense.T):
        raise ValueError("L must be symmetric")
    return dense


class DenseExactGraphHeatKernel(gpytorch.kernels.Kernel):
    """Exact ``exp(-heat_time * L)`` covariance blocks for integer node IDs.

    The fixed symmetric graph Laplacian can be SciPy sparse or Torch dense/CSR.
    An eigendecomposition is computed once and registered as module buffers.
    ``heat_time`` has the same positive scalar parameterization as GraphHeatKernel.
    Inputs have shape ``(N,)`` or ``(N, 1)``; batching is unsupported.
    This intentionally dense baseline is suitable only for small graphs.
    """

    has_lengthscale = False

    def __init__(self, L, *, heat_time=1.0, **kwargs):
        super().__init__(**kwargs)
        if self.batch_shape:
            raise ValueError("DenseExactGraphHeatKernel supports one unbatched graph")
        dense = _dense_laplacian(L)
        evals, evecs = torch.linalg.eigh(dense)
        self.register_buffer("evals", evals)
        self.register_buffer("evecs", evecs)
        self.register_parameter("raw_heat_time", torch.nn.Parameter(
            torch.zeros((), dtype=dense.dtype, device=dense.device)))
        self.register_constraint("raw_heat_time", gpytorch.constraints.Positive())
        self.heat_time = heat_time

    @property
    def heat_time(self):
        return self.raw_heat_time_constraint.transform(self.raw_heat_time)

    @heat_time.setter
    def heat_time(self, value):
        value = torch.as_tensor(value, dtype=self.raw_heat_time.dtype,
                                device=self.raw_heat_time.device)
        if value.numel() != 1 or not torch.isfinite(value).all() or not (value > 0).all():
            raise ValueError("heat_time must be a finite positive scalar")
        self.initialize(raw_heat_time=self.raw_heat_time_constraint.inverse_transform(value.reshape(())))

    def _node_indices(self, x):
        if x.ndim != 2 or x.shape[-1] != 1:
            raise ValueError("node IDs must have shape (N,) or (N, 1); batching is unsupported")
        if x.device != self.evals.device:
            raise ValueError("node IDs must be on the same device as the graph")
        if x.is_complex() or x.dtype == torch.bool:
            raise ValueError("node IDs must be integers")
        if x.is_floating_point() and (not torch.isfinite(x).all() or not (x == x.round()).all()):
            raise ValueError("node IDs must be finite integers")
        if ((x < 0) | (x >= self.evals.numel())).any():
            raise ValueError(f"node IDs must be in [0, {self.evals.numel() - 1}]")
        return x[:, 0].long()

    def spectral_values(self):
        """Return exact heat eigenvalues with their heat-time gradient intact."""
        return torch.exp(-self.heat_time * self.evals)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise ValueError("last_dim_is_batch is unsupported")
        i1, i2 = self._node_indices(x1), self._node_indices(x2)
        left, right = self.evecs[i1], self.evecs[i2]
        weighted_left = left * self.spectral_values()
        if diag:
            size = min(i1.numel(), i2.numel())
            return (weighted_left[:size] * right[:size]).sum(dim=-1)
        return weighted_left @ right.T


class DenseChebyshevGraphHeatKernel(DenseExactGraphHeatKernel):
    """Dense eigenspace evaluation of the same Chebyshev heat approximation.

    ``num_coefficients`` is polynomial degree plus one. Pass the same spectral
    interval and ``num_nodes`` as the sparse kernel for an identical polynomial.
    A missing upper bound uses the same Laplacian bound helper as GraphHeatKernel.
    No positivity correction is applied: a truncated polynomial can be negative.
    """

    def __init__(self, L, *, lambda_max=None, lambda_min=0.0,
                 num_coefficients=33, num_nodes=128, heat_time=1.0, **kwargs):
        super().__init__(L, heat_time=heat_time, **kwargs)
        bounds_input = L
        if torch.is_tensor(L) and L.layout == torch.strided and lambda_max is None:
            bounds_input = L.to_sparse_csr()
        self.lambda_min, self.lambda_max = laplacian_eigenvalue_bounds(
            bounds_input, lambda_min=lambda_min, lambda_max=lambda_max)
        nodes, basis = chebyshev_basis_on_interval_torch(
            lambda_min=self.lambda_min, lambda_max=self.lambda_max,
            num_coefficients=num_coefficients, num_nodes=num_nodes,
            dtype=self.evals.dtype, device=self.evals.device,
        )
        self.register_buffer("spectral_nodes", nodes)
        self.register_buffer("cosine_basis", basis)

    def spectral_values(self):
        """Evaluate the differentiable polynomial at the graph eigenvalues."""
        coefficients = chebyshev_coefficients_from_samples_torch(
            torch.exp(-self.heat_time * self.spectral_nodes), self.cosine_basis)
        return evaluate_chebyshev_polynomial_torch(
            coefficients, self.evals, lambda_min=self.lambda_min,
            lambda_max=self.lambda_max)

    def get_extra_state(self):
        return {"lambda_min": self.lambda_min, "lambda_max": self.lambda_max}

    def set_extra_state(self, state):
        self.lambda_min = state["lambda_min"]
        self.lambda_max = state["lambda_max"]
