"""Small, dense validation oracles (never used by production code)."""
import numpy as np
import scipy.sparse as sparse
import torch

DTYPE = torch.float64


def path_laplacian(n):
    degree = np.full(n, 2.0)
    degree[[0, -1]] = 1.0 if n > 1 else 0.0
    return sparse.diags([degree, -np.ones(n - 1), -np.ones(n - 1)],
                        [0, -1, 1], shape=(n, n), format="csr")


def dense_heat(L, time):
    eigenvalues, eigenvectors = torch.linalg.eigh(torch.tensor(L.toarray(), dtype=DTYPE))
    return (eigenvectors * torch.exp(-time * eigenvalues)) @ eigenvectors.T


def random_tensor(*shape, seed=123):
    return torch.randn(*shape, dtype=DTYPE, generator=torch.Generator().manual_seed(seed))


def relative_error(actual, expected):
    return ((actual - expected).norm() / expected.norm()).item()
