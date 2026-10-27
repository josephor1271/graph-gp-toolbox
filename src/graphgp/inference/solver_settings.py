"""Explicit, scoped settings for the installed iterative GP inference route.

This module configures public GPyTorch settings only. It neither changes the
installed CG implementation/epsilon nor chooses or seeds stochastic probes.
Profiles here are configuration containers, not production recommendations.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Iterator

import gpytorch


@dataclass(frozen=True)
class IterativeConfig:
    """All variable settings for CPU sparse CG/SLQ measurements.

    Defaults reproduce the legacy helper under installed default settings.
    Explicit train/evaluation tolerances avoid inheriting different library
    defaults. Preconditioning remains disabled throughout this study.
    """

    cg_tolerance: float = 1e-5
    eval_cg_tolerance: float = 1e-5
    max_cg_iterations: int = 500
    trace_samples: int = 32
    lanczos_iterations: int = 32
    max_preconditioner_size: int = 0
    terminate_cg_by_size: bool = False

    def __post_init__(self) -> None:
        for name in ("cg_tolerance", "eval_cg_tolerance"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be a finite positive real number")
            object.__setattr__(self, name, float(value))
        for name in ("max_cg_iterations", "trace_samples", "lanczos_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if (isinstance(self.max_preconditioner_size, bool)
                or not isinstance(self.max_preconditioner_size, Integral)
                or self.max_preconditioner_size != 0):
            raise ValueError("max_preconditioner_size must be the integer zero for this study")
        object.__setattr__(self, "max_preconditioner_size", int(self.max_preconditioner_size))
        if not isinstance(self.terminate_cg_by_size, bool):
            raise ValueError("terminate_cg_by_size must be an explicit boolean")

    def to_dict(self) -> dict:
        """Return the constructor fields in deterministic declaration order."""
        return asdict(self)

    def effective_settings(self) -> dict:
        """Return variable and fixed public settings for complete metadata."""
        return {
            **self.to_dict(),
            "max_cholesky_size": 0,
            "max_eager_kernel_size": 0,
            "fast_computations": {
                "covar_root_decomposition": True, "log_prob": True, "solves": True,
            },
            "fast_pred_var": False,
        }


@contextmanager
def iterative_context(config: IterativeConfig) -> Iterator[None]:
    """Force installed iterative inference and restore every setting on exit.

    Zero Cholesky/eager thresholds prevent direct factorization and eager
    posterior block conversion. All three fast-computations flags retain the
    legacy iterative root/log-probability/solve behavior. ``fast_pred_var=False``
    retains exact requested posterior marginal-variance semantics rather than
    enabling LOVE variance approximation. No settings are changed globally.
    """
    if not isinstance(config, IterativeConfig):
        raise TypeError("config must be an IterativeConfig")
    with ExitStack() as stack:
        for setting in (
            gpytorch.settings.max_cholesky_size(0),
            gpytorch.settings.max_preconditioner_size(config.max_preconditioner_size),
            gpytorch.settings.max_eager_kernel_size(0),
            gpytorch.settings.cg_tolerance(config.cg_tolerance),
            gpytorch.settings.eval_cg_tolerance(config.eval_cg_tolerance),
            gpytorch.settings.max_cg_iterations(config.max_cg_iterations),
            gpytorch.settings.max_lanczos_quadrature_iterations(config.lanczos_iterations),
            gpytorch.settings.num_trace_samples(config.trace_samples),
            gpytorch.settings.terminate_cg_by_size(config.terminate_cg_by_size),
            gpytorch.settings.fast_computations(
                covar_root_decomposition=True, log_prob=True, solves=True),
            gpytorch.settings.fast_pred_var(False),
        ):
            stack.enter_context(setting)
        yield
