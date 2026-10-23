"""Requested-target marginal moments, with an explicit mean-only path."""
from dataclasses import dataclass
import gpytorch
import torch


@dataclass
class Marginals:
    mean: torch.Tensor
    latent_variance: torch.Tensor | None
    observation_variance: torch.Tensor | None


def predict_marginals(model, targets, *, mean_only=False, chunk_size=64):
    """Condition using the caller's inference settings; never return joint covariance.

    Variances use ordinary GP conditioning, subject to the selected kernel and
    solve accuracy. Negative or nonfinite variances raise rather than being clipped.
    Cache is reset per call to avoid stale caches after parameter/data changes.
    """
    if chunk_size < 1 or targets.numel() == 0:
        raise ValueError("positive chunk_size and nonempty targets required")
    model.eval()
    model.likelihood.eval()
    model.prediction_strategy = None
    means, variances = [], []
    with torch.no_grad(), gpytorch.settings.fast_pred_var(False), \
            gpytorch.settings.skip_posterior_variances(mean_only):
        for chunk in targets.split(chunk_size):
            posterior = model(chunk)
            means.append(posterior.mean)
            if not mean_only:
                # Avoid MultivariateNormal.variance's automatic minimum clamp.
                value = posterior.lazy_covariance_matrix.diagonal()
                if not torch.isfinite(value).all() or (value < 0).any():
                    raise ArithmeticError("invalid latent marginal variance")
                variances.append(value)
    mean = torch.cat(means)
    if not torch.isfinite(mean).all():
        raise ArithmeticError("invalid predictive mean")
    latent = None if mean_only else torch.cat(variances)
    observed = None if mean_only else latent + model.likelihood.noise.detach()
    return Marginals(mean, latent, observed)
