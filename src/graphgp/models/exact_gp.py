"""Zero-mean GPyTorch model and explicit iterative inference settings."""
from contextlib import contextmanager
import gpytorch

from graphgp.inference.solver_settings import IterativeConfig, iterative_context

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
    """Legacy scoped CG/SLQ defaults, with preconditioning disabled.

    Historically this helper inherited ``terminate_cg_by_size`` from its caller
    (the installed default is False). Preserve that behavior even inside a
    caller's overriding context; new measurements should set it explicitly in
    ``IterativeConfig``. All other former settings retain their exact values.
    """
    config = IterativeConfig(
        cg_tolerance=cg_tolerance, eval_cg_tolerance=cg_tolerance,
        max_cg_iterations=500, trace_samples=trace_samples,
        lanczos_iterations=lanczos_iterations, max_preconditioner_size=0,
        terminate_cg_by_size=gpytorch.settings.terminate_cg_by_size.on(),
    )
    with iterative_context(config):
        yield
