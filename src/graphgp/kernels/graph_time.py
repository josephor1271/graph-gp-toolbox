"""Separable heat graph × RBF time covariance on a declared finite grid."""

from numbers import Integral

import gpytorch
import torch
from linear_operator.operators import (
    DenseLinearOperator,
    InterpolatedLinearOperator,
    KroneckerProductLinearOperator,
)

from graphgp.kernels.dense import DenseExactGraphHeatKernel
from graphgp.kernels.heat import GraphHeatKernel
from graphgp.kernels.safety import heat_interpolation_certificate
from graphgp.utils.study_domain import (
    HEAT_RANGE, LENGTHSCALE_RANGE, validate_study_laplacian,
)


def _initial_value(value: float, bounds: tuple[float, float], name: str) -> torch.Tensor:
    """Construct a bounded scalar directly in the validated float64 dtype."""
    scalar = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if (scalar.numel() != 1 or not torch.isfinite(scalar).all()
            or not ((scalar > bounds[0]) & (scalar < bounds[1])).all()):
        raise ValueError(f"{name} initialization must lie strictly inside {bounds}")
    return scalar.reshape(())


class GraphTimeHeatRBFKernel(gpytorch.kernels.Kernel):
    """Unit-amplitude ``exp(-tau L) ⊗ RBF(t, t; ell)`` with pair selection.

    Inputs are nonempty unbatched CPU float64 ``(m, 2)`` tensors containing
    exact integer ``(vertex_id, time_index)`` coordinates. Time index j maps
    to the fixed grid ``j/(num_times-1)``. Missing, unordered and repeated
    observations are retained in vertex-major/time-minor order ``v*T+j``.

    The sparse backend retains a CSR graph factor and a dense T×T temporal
    factor; MVPs still use full-grid nT workspace. Exact selected diagonals
    are available but can be expensive. The dense backend is a small-graph
    reference. Graph/input batching and other devices/dtypes are unsupported.
    """

    has_lengthscale = False

    def __init__(self, L, *, num_times: int, backend: str, heat_time: float = 0.7,
                 lengthscale: float = 0.4, degree: int = 32, **kwargs):
        super().__init__(**kwargs)
        if self.batch_shape or self.active_dims is not None:
            raise ValueError("graph/input batching and active_dims are unsupported")
        if (isinstance(num_times, bool) or not isinstance(num_times, Integral)
                or not 2 <= num_times <= 2**53 - 1):
            raise ValueError("num_times must be an integer >= 2 in the float64 exact-int domain")
        if backend not in ("dense_exact", "sparse_chebyshev"):
            raise ValueError("backend must be 'dense_exact' or 'sparse_chebyshev'")
        laplacian = validate_study_laplacian(L)
        self.num_vertices = laplacian.shape[0]
        self.num_times = int(num_times)
        self.backend = backend
        tau = _initial_value(heat_time, HEAT_RANGE, "heat_time")
        ell = _initial_value(lengthscale, LENGTHSCALE_RANGE, "lengthscale")
        if backend == "sparse_chebyshev":
            heat_interpolation_certificate(degree)
            graph_kernel = GraphHeatKernel(
                laplacian, lambda_min=0.0, lambda_max=2.0,
                num_coefficients=degree + 1, num_nodes=degree + 1,
                heat_time=tau)
        else:
            graph_kernel = DenseExactGraphHeatKernel(laplacian, heat_time=tau)
        graph_kernel.register_constraint(
            "raw_heat_time", gpytorch.constraints.Interval(*HEAT_RANGE).double())
        graph_kernel.heat_time = tau
        self.graph_kernel = graph_kernel
        self.time_kernel = gpytorch.kernels.RBFKernel(
            lengthscale_constraint=gpytorch.constraints.Interval(*LENGTHSCALE_RANGE).double()
        ).double()
        self.time_kernel.lengthscale = ell
        self.register_buffer(
            "time_grid", torch.arange(self.num_times, dtype=torch.float64) / (self.num_times - 1))

    @property
    def heat_time(self) -> torch.Tensor:
        """Spatial heat time; set through ``graph_kernel.heat_time`` if needed."""
        return self.graph_kernel.heat_time

    @property
    def lengthscale(self) -> torch.Tensor:
        """Temporal lengthscale in normalized full-grid time units."""
        return self.time_kernel.lengthscale.reshape(())

    def validate_pairs(self, pairs: torch.Tensor) -> None:
        """Reject coordinates outside the declared CPU float64 pair contract."""
        if not torch.is_tensor(pairs):
            raise TypeError("pairs must be a Torch tensor")
        if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.shape[0] == 0:
            raise ValueError("pairs must have nonempty unbatched shape (m, 2)")
        if (pairs.dtype != torch.float64 or pairs.device.type != "cpu"
                or self.time_grid.dtype != torch.float64 or self.time_grid.device.type != "cpu"):
            raise ValueError("pairs and kernel must use CPU float64")
        if (not torch.isfinite(pairs).all() or not (pairs == pairs.round()).all()
                or (pairs.abs() > 2**53 - 1).any()):
            raise ValueError("pair coordinates must be finite exact integers in the float64 exact-int domain")
        if ((pairs[:, 0] < 0) | (pairs[:, 0] >= self.num_vertices)
                | (pairs[:, 1] < 0) | (pairs[:, 1] >= self.num_times)).any():
            raise ValueError("vertex or time index is outside the declared graph/time grid")

    def _flat_indices(self, pairs: torch.Tensor) -> torch.Tensor:
        self.validate_pairs(pairs)
        indices = pairs.to(dtype=torch.long)
        return (indices[:, 0] * self.num_times + indices[:, 1]).unsqueeze(-1)

    def _full_grid_operator(self) -> KroneckerProductLinearOperator:
        if self.backend == "sparse_chebyshev":
            graph_factor = self.graph_kernel.full_graph_operator()
        else:
            graph = self.graph_kernel
            graph_factor = DenseLinearOperator(
                (graph.evecs * graph.spectral_values()) @ graph.evecs.T)
        temporal = self.time_kernel(self.time_grid[:, None]).to_dense()
        return KroneckerProductLinearOperator(graph_factor, DenseLinearOperator(temporal))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False,
                last_dim_is_batch: bool = False, **params):
        if last_dim_is_batch:
            raise ValueError("last_dim_is_batch is unsupported")
        left, right = self._flat_indices(x1), self._flat_indices(x2)
        base = self._full_grid_operator()
        if diag:
            size = min(left.shape[0], right.shape[0])
            return base._get_indices(left[:size, 0], right[:size, 0])
        return InterpolatedLinearOperator(
            base_linear_op=base, left_interp_indices=left, right_interp_indices=right)
