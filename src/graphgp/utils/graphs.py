"""Deterministic graphs with a declared, analytically enclosing spectrum."""
import networkx as nx
import numpy as np


def make_laplacian(family="path", n=49, seed=2718):
    """Return L_combinatorial / max_degree; its spectrum lies in [0, 2].

    This scalar rescaling is NOT the symmetric normalized Laplacian. Heat times
    in the experiments refer to this scaled L, not the original graph units.
    """
    if n < 4:
        raise ValueError("n must be at least four")
    if family == "path":
        graph = nx.path_graph(n)
    elif family == "grid":
        side = int(n ** 0.5)
        if side * side != n:
            raise ValueError("grid size must be a perfect square")
        graph = nx.grid_2d_graph(side, side)
    elif family == "irregular":
        graph = nx.barabasi_albert_graph(n, 2, seed=seed)
    else:
        raise ValueError(f"unknown graph family: {family}")
    laplacian = nx.laplacian_matrix(graph).astype(np.float64).tocsr()
    scale = float(laplacian.diagonal().max())
    return laplacian / scale, scale
