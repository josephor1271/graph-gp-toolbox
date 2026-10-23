"""Small matrix-free ExactGP example: python -m graphgp.demo."""

from graphgp.models.exact_gp import GraphGP, iterative_settings

import argparse

import gpytorch
import numpy as np
import scipy.sparse
import torch

from graphgp.kernels.heat import GraphHeatKernel


def make_problem(n=40, seed=123):
    """A path graph and noisy smooth signal; no dense graph matrices."""
    if n < 4:
        raise ValueError("n must be at least 4")
    degrees = np.full(n, 2.0)
    degrees[[0, -1]] = 1.0
    L = scipy.sparse.diags([degrees, -np.ones(n - 1), -np.ones(n - 1)],
                           [0, -1, 1], format="csr")
    ids = torch.arange(n)
    generator = torch.Generator().manual_seed(seed)
    signal = torch.sin(torch.linspace(0, 2 * torch.pi, n, dtype=torch.float64))
    y = signal + 0.04 * torch.randn(n, generator=generator, dtype=torch.float64)
    train_ids, test_ids = ids[::2], ids[1::2]
    return L, train_ids, y[train_ids], test_ids


def run_demo(steps=20, seed=123):
    L, train_ids, train_y, test_ids = make_problem(seed=seed)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
    likelihood.noise = 0.12
    kernel = GraphHeatKernel(L, lambda_max=4, num_coefficients=33, num_nodes=128, heat_time=1.1)
    model = GraphGP(train_ids, train_y, likelihood, kernel)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
    losses = []
    model.train()
    likelihood.train()
    with iterative_settings(), torch.random.fork_rng(devices=[]):
        # Common random probes make the demo's optimization curve comparable.
        # The integration tests separately use exhaustive trace probes.
        for step in range(steps + 1):
            torch.manual_seed(seed)
            optimizer.zero_grad()
            loss = -mll(model(train_ids), train_y)
            losses.append(loss.item())
            if step < steps:
                loss.backward()
                optimizer.step()
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            prediction = likelihood(model(test_ids))
            mean, variance = prediction.mean, prediction.variance
    print(f"Path graph: {L.shape[0]} nodes; {len(train_ids)} train / {len(test_ids)} test")
    print("Inference: CG/SLQ, max_cholesky_size=0, max_preconditioner_size=0")
    print(f"Loss: {losses[0]:.6f} -> {losses[-1]:.6f} ({steps} Adam steps)")
    print(f"heat_time={kernel.heat_time.item():.6f}, noise={likelihood.noise.item():.6f}")
    print(f"Finite predictions: mean={bool(mean.isfinite().all())}, variance={bool(variance.isfinite().all())}")
    return model, likelihood, losses, mean, variance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    run_demo(steps=parser.parse_args().steps)
