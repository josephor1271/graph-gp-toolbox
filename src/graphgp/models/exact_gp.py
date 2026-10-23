"""Zero-mean GPyTorch model and explicit iterative inference settings."""
from contextlib import ExitStack, contextmanager
import gpytorch

class GraphGP(gpytorch.models.ExactGP):
    def __init__(self, train_ids, train_y, likelihood, kernel):
        super().__init__(train_ids, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, node_ids):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(node_ids), self.covar_module(node_ids))


@contextmanager
def iterative_settings(*, cg_tolerance=1e-5, trace_samples=32, lanczos_iterations=32):
    """Explicit CG/SLQ settings; diagonal-based preconditioning is disabled."""
    with ExitStack() as stack:
        for setting in [
            gpytorch.settings.max_cholesky_size(0),
            gpytorch.settings.max_preconditioner_size(0),
            gpytorch.settings.max_eager_kernel_size(0),
            gpytorch.settings.cg_tolerance(cg_tolerance),
            gpytorch.settings.eval_cg_tolerance(cg_tolerance),
            gpytorch.settings.max_cg_iterations(500),
            gpytorch.settings.max_lanczos_quadrature_iterations(lanczos_iterations),
            gpytorch.settings.num_trace_samples(trace_samples),
            gpytorch.settings.fast_computations(covar_root_decomposition=True, log_prob=True, solves=True),
            gpytorch.settings.fast_pred_var(False),
        ]:
            stack.enter_context(setting)
        yield
