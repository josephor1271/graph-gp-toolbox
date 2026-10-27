"""Trainable graph kernels using sparse Laplacian matrix products."""

import gpytorch
import scipy.sparse
import torch
from linear_operator.operators import InterpolatedLinearOperator

from graphgp.operators.coefficients import (
    chebyshev_basis_on_interval_torch,
    chebyshev_coefficients_from_samples_torch,
    laplacian_eigenvalue_bounds,
)
from graphgp.operators.chebyshev import ChebyshevFunctionLinearOperator, scipy_csr_to_torch_csr


class GraphHeatKernel(gpytorch.kernels.Kernel):
    """Approximate ``exp(-heat_time * L)`` for integer graph node IDs.

    ``L`` is a fixed symmetric Laplacian, stored as sparse CSR. Bounds and
    the coefficient basis are computed once. ``num_coefficients`` is degree+1.
    Inputs have shape ``(N,)`` or ``(N,1)``; graph/input batching is unsupported.
    Use ScaleKernel for a trainable amplitude. A finite polynomial need not
    stay positive for arbitrary heat times: validate the chosen degree/range.

    The Laplacian minimum defaults to zero. A missing maximum is estimated
    with ARPACK; known enclosing bounds are preferable. Exact diagonals are
    correct but expensive. Force max_cholesky_size(0) and initially disable
    preconditioning to exercise the iterative inference path.

    ``scaled_laplacian=True`` preprocesses a SciPy graph once, storing its
    scaled CSR instead of a second graph copy. Coefficient nodes keep the
    original spectral interval. Raw Torch CSR inputs use the default route.
    Restore state into a kernel constructed with the same graph/storage mode.
    """

    has_lengthscale = False

    def __init__(self, L, *, lambda_max=None, lambda_min=0.0,
                 num_coefficients=33, num_nodes=None, heat_time=1.0,
                 scaled_laplacian=False, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(scaled_laplacian, bool):
            raise TypeError("scaled_laplacian must be a bool")
        if self.batch_shape:
            raise ValueError("GraphHeatKernel supports one unbatched graph")
        scipy_graph = L if scipy.sparse.issparse(L) else None
        if scipy_graph is not None:
            L = scipy_csr_to_torch_csr(L)
        if not torch.is_tensor(L) or L.layout != torch.sparse_csr:
            raise TypeError("L must be a Torch sparse CSR tensor or SciPy sparse matrix")
        if L.is_complex():
            raise TypeError("L must be real")
        if L.ndim != 2 or L.shape[0] != L.shape[1] or L.shape[0] == 0:
            raise ValueError("L must be a nonempty square matrix")
        if L.requires_grad:
            raise ValueError("L must be fixed (requires_grad=False)")
        if not L.is_floating_point():
            L = L.to(dtype=torch.get_default_dtype())
        self.lambda_min, self.lambda_max = laplacian_eigenvalue_bounds(
            L, lambda_min=lambda_min, lambda_max=lambda_max)
        # Opt-in preprocessing replaces the resident graph; it does not keep
        # a second CSR copy. Raw Torch graphs retain the original runtime path.
        self.scaled_laplacian = scaled_laplacian and scipy_graph is not None
        if self.scaled_laplacian:
            center = (self.lambda_min+self.lambda_max)/2
            radius = (self.lambda_max-self.lambda_min)/2
            scaled = (scipy_graph.tocsr()-center*scipy.sparse.eye(L.shape[0],format="csr"))/radius
            scaled.sum_duplicates(); scaled.eliminate_zeros(); scaled.sort_indices()
            L = scipy_csr_to_torch_csr(scaled,dtype=L.dtype,device=L.device)
        nodes, basis = chebyshev_basis_on_interval_torch(
            lambda_min=self.lambda_min, lambda_max=self.lambda_max,
            num_coefficients=num_coefficients, num_nodes=num_nodes,
            dtype=L.dtype, device=L.device,
        )
        self.register_buffer("L", L)
        self.register_buffer("spectral_nodes", nodes)
        self.register_buffer("cosine_basis", basis)
        self.register_parameter("raw_heat_time", torch.nn.Parameter(
            torch.zeros((), dtype=L.dtype, device=L.device)))
        self.register_constraint("raw_heat_time", gpytorch.constraints.Positive())
        self.heat_time = heat_time

    @property
    def heat_time(self):
        return self.raw_heat_time_constraint.transform(self.raw_heat_time)

    @heat_time.setter
    def heat_time(self, value):
        value = torch.as_tensor(value, dtype=self.raw_heat_time.dtype, device=self.raw_heat_time.device)
        if value.numel() != 1 or not torch.isfinite(value).all() or not (value > 0).all():
            raise ValueError("heat_time must be a finite positive scalar")
        self.initialize(raw_heat_time=self.raw_heat_time_constraint.inverse_transform(value.reshape(())))

    def get_extra_state(self):
        # Keep Python interval constants consistent with the loaded L/basis.
        return {"lambda_min": self.lambda_min, "lambda_max": self.lambda_max,
                "scaled_laplacian": self.scaled_laplacian}

    def set_extra_state(self, state):
        self.lambda_min = state["lambda_min"]
        self.lambda_max = state["lambda_max"]
        self.scaled_laplacian = state.get("scaled_laplacian", False)

    def _node_indices(self, x):
        if x.ndim != 2 or x.shape[-1] != 1:
            raise ValueError("node IDs must have shape (N,) or (N, 1); batching is unsupported")
        if x.device != self.L.device:
            raise ValueError("node IDs must be on the same device as the graph")
        if x.is_complex() or x.dtype == torch.bool:
            raise ValueError("node IDs must be integers")
        if x.is_floating_point() and (not torch.isfinite(x).all() or not (x == x.round()).all()):
            raise ValueError("node IDs must be finite integers")
        if ((x < 0) | (x >= self.L.shape[0])).any():
            raise ValueError(f"node IDs must be in [0, {self.L.shape[0] - 1}]")
        return x.long()

    def full_graph_operator(self) -> ChebyshevFunctionLinearOperator:
        """Return differentiable ``exp(-heat_time * L)`` polynomial MVPs.

        Reuse the registered CSR graph and coefficient basis. This constructs
        an operator, without materializing its graph covariance or copying L.
        """
        coefficients = chebyshev_coefficients_from_samples_torch(
            torch.exp(-self.heat_time * self.spectral_nodes), self.cosine_basis)
        return ChebyshevFunctionLinearOperator(
            self.L, coefficients, lambda_min=self.lambda_min, lambda_max=self.lambda_max,
            pre_scaled=self.scaled_laplacian)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise ValueError("last_dim_is_batch is unsupported")
        i1, i2 = self._node_indices(x1), self._node_indices(x2)
        base = self.full_graph_operator()
        result = InterpolatedLinearOperator(
            base_linear_op=base, left_interp_indices=i1, right_interp_indices=i2)
        return result.diagonal() if diag else result
