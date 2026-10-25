"""Independent small dense references for the finite graph × time model.

Coordinates are ``(vertex_id, time_index)`` and flatten as ``vertex*T + time``.
This module deliberately uses SciPy ``expm`` and an explicit NumPy RBF formula;
it never imports a production covariance kernel or Chebyshev implementation.
Noise is a variance and is independent per recording, including repeated pairs.
Cholesky failures propagate without jitter or an explicit matrix inverse.
"""

from typing import Any

import numpy as np
from scipy.linalg import cholesky, cho_solve, expm, solve_triangular

from graphgp.utils.graphs import make_laplacian


def _dimension(value: int, name: str, *, minimum: int = 1) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum or value > 2**53):
        raise ValueError(f"{name} must be an integer in [{minimum}, 2**53]")
    return int(value)


def _positive_parameter(value: float, name: str, *, allow_zero: bool = False) -> float:
    scalar = np.asarray(value)
    if (scalar.ndim != 0 or scalar.dtype.kind not in "fiu"
            or not np.isfinite(scalar)
            or (scalar < 0 if allow_zero else scalar <= 0)):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} scalar")
    return float(scalar)


def _laplacian_array(laplacian: Any) -> np.ndarray:
    dense = laplacian.toarray() if hasattr(laplacian, "toarray") else laplacian
    matrix = np.asarray(dense, dtype=np.float64)
    if (matrix.ndim != 2 or not matrix.shape[0]
            or matrix.shape[0] != matrix.shape[1]
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12)):
        raise ValueError("L must be a finite nonempty symmetric square matrix")
    if np.linalg.eigvalsh(matrix)[0] < -1e-12:
        raise ValueError("L must be positive semidefinite")
    return matrix


def _time_array(time_grid: np.ndarray) -> np.ndarray:
    times = np.asarray(time_grid, dtype=np.float64)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("time_grid must be a finite strictly increasing vector of length >=2")
    expected = np.arange(len(times), dtype=np.float64) / (len(times) - 1)
    if not np.array_equal(times, expected):
        raise ValueError("time_grid must equal arange(T)/(T-1) on the full declared grid")
    return times


def _factors(L: Any, time_grid: np.ndarray, tau: float,
             ell: float) -> tuple[np.ndarray, np.ndarray]:
    matrix = _laplacian_array(L)
    times = _time_array(time_grid)
    heat_time = _positive_parameter(tau, "tau")
    lengthscale = _positive_parameter(ell, "ell")
    graph = expm(-heat_time * matrix)
    distance = times[:, None] - times[None, :]
    temporal = np.exp(-(distance * distance) / (2 * lengthscale**2))
    return graph, temporal


def dense_prior(L: Any, time_grid: np.ndarray, tau: float,
                ell: float) -> np.ndarray:
    """Return exact ``expm(-tau*L) ⊗ RBF(time_grid, ell)`` in float64.

    ``time_grid`` is the full normalized grid ``arange(T)/(T-1)``, ``T>=2``.
    The returned ``(n*T, n*T)`` array is vertex-major / time-minor, has unit
    latent amplitude, and contains no observation noise. Dense memory is
    intentional here and is suitable only for small reference problems.
    """
    graph, temporal = _factors(L, time_grid, tau, ell)
    return np.kron(graph, temporal)


def flat_pair_indices(pairs: np.ndarray, num_vertices: int,
                      num_times: int) -> np.ndarray:
    """Validate float64 ``(m,2)`` integer coordinates and return ``v*T+j``.

    Unordered and repeated rows are preserved. Inputs must be unbatched,
    nonempty, finite NumPy float64 coordinates within the declared graph and
    time grid; fractional values, integers of another dtype, and invalid
    indices are rejected rather than coerced.
    """
    n = _dimension(num_vertices, "num_vertices")
    total_times = _dimension(num_times, "num_times", minimum=2)
    if n * total_times > np.iinfo(np.int64).max:
        raise ValueError("the flattened grid must fit in int64")
    coordinates = np.asarray(pairs)
    if (coordinates.dtype != np.dtype(np.float64)
            or coordinates.ndim != 2 or coordinates.shape[1] != 2
            or not len(coordinates) or not np.isfinite(coordinates).all()
            or np.any(coordinates != np.floor(coordinates))
            or np.any(coordinates < 0)
            or np.any(coordinates[:, 0] >= n)
            or np.any(coordinates[:, 1] >= total_times)):
        raise ValueError("pairs must be nonempty float64 (m,2) valid integer (vertex,time) indices")
    integer_pairs = coordinates.astype(np.int64)
    return integer_pairs[:, 0] * total_times + integer_pairs[:, 1]


def _full_covariance(full_K: np.ndarray, num_times: int) -> tuple[np.ndarray, int]:
    total_times = _dimension(num_times, "num_times", minimum=2)
    covariance = np.asarray(full_K, dtype=np.float64)
    if (covariance.ndim != 2 or not covariance.shape[0]
            or covariance.shape[0] != covariance.shape[1]
            or covariance.shape[0] % total_times
            or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12)):
        raise ValueError("full_K must be finite symmetric square with size divisible by num_times")
    return covariance, len(covariance) // total_times


def observed_covariance(full_K: np.ndarray, pairs: np.ndarray, *,
                        noise_variance: float, num_times: int) -> np.ndarray:
    """Select full-prior rows/columns, then add independent recording noise.

    ``num_times`` is explicit because a plain full covariance array cannot
    identify its graph/time factor dimensions. Duplicate pairs produce equal
    latent rows, while observation noise is added only on the selected diagonal.
    """
    covariance, n = _full_covariance(full_K, num_times)
    ids = flat_pair_indices(pairs, n, num_times)
    noise = _positive_parameter(noise_variance, "noise_variance", allow_zero=True)
    return covariance[np.ix_(ids, ids)] + noise * np.eye(len(ids), dtype=np.float64)


def _observations(train_y: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(train_y, dtype=np.float64)
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError("train_y must be a finite vector with one value per training pair")
    return values


def posterior_marginals(full_K: np.ndarray, train_pairs: np.ndarray,
                        train_y: np.ndarray, target_pairs: np.ndarray,
                        noise_variance: float, *, num_times: int) -> dict[str, np.ndarray]:
    """Return exact conditional mean, latent variance, and observation variance.

    Outputs ``mean``, ``latent_var`` and ``observation_var`` are target-length
    arrays. Targets may be repeated, absent from training, or forecast indices
    on the same declared grid. Only marginal variances are constructed; Cholesky
    solves use the selected noisy training covariance without hidden jitter.
    """
    covariance, n = _full_covariance(full_K, num_times)
    train_ids = flat_pair_indices(train_pairs, n, num_times)
    target_ids = flat_pair_indices(target_pairs, n, num_times)
    values = _observations(train_y, len(train_ids))
    noise = _positive_parameter(noise_variance, "noise_variance", allow_zero=True)
    train_covariance = covariance[np.ix_(train_ids, train_ids)] + noise * np.eye(len(train_ids))
    factor = cholesky(train_covariance, lower=True)
    target_train = covariance[np.ix_(target_ids, train_ids)]
    mean = target_train @ cho_solve((factor, True), values)
    whitened = solve_triangular(factor, target_train.T, lower=True)
    latent_var = covariance[target_ids, target_ids] - np.sum(whitened**2, axis=0)
    observation_var = latent_var + noise
    if (not np.isfinite(mean).all() or not np.isfinite(latent_var).all()
            or np.any(latent_var < 0)):
        raise FloatingPointError("nonfinite posterior moments or negative latent variance")
    return dict(mean=mean, latent_var=latent_var, observation_var=observation_var)


def exact_nll(full_K: np.ndarray, train_pairs: np.ndarray,
              train_y: np.ndarray, noise_variance: float, *, num_times: int) -> float:
    """Return exact zero-mean Gaussian NLL per recorded observation."""
    covariance = observed_covariance(full_K, train_pairs, noise_variance=noise_variance,
                                     num_times=num_times)
    values = _observations(train_y, len(covariance))
    factor = cholesky(covariance, lower=True)
    result = (2 * np.log(np.diag(factor)).sum()
              + values @ cho_solve((factor, True), values)
              + len(values) * np.log(2 * np.pi)) / (2 * len(values))
    if not np.isfinite(result):
        raise FloatingPointError("nonfinite exact NLL")
    return float(result)


def sample_problem(seed: int, split: str, *, graph_kind: str = "grid", n: int = 16,
                   num_times: int = 10, tau: float = 1.2, ell: float = 0.24,
                   noise_variance: float = 0.06, graph_seed: int = 2718,
                   train_count: int | None = None,
                   forecast_start: int | None = None) -> dict[str, Any]:
    """Sample independent dense latent/noise fields and a predetermined split.

    Defaults implement the fixed Stage 03 design: 4×4 grid, T=10, 64 random
    interpolation observations, or 48 observations among j<6 with all j>=6
    held out for forecasting. Optional sizes/counts support tiny validation
    problems without changing experiment defaults. The latent ``F`` is sampled
    as ``chol(KG) @ Z @ chol(KT).T`` and flattened vertex-major as ``latent``;
    ``y`` adds independent ``sqrt(noise_variance)`` Gaussian recording noise.

    SeedSequence spawn keys (0,), (1,), (2,0), (2,1) respectively identify
    latent, noise, interpolation-mask and forecasting-mask streams. For a fixed
    seed and problem configuration, both split designs share identical full
    latent/noisy arrays. Masks depend only on their own numeric RNG streams.
    """
    draw_seed = _dimension(seed, "seed", minimum=0)
    vertices = _dimension(n, "n", minimum=4)
    total_times = _dimension(num_times, "num_times", minimum=2)
    graph_seed = _dimension(graph_seed, "graph_seed", minimum=0)
    noise = _positive_parameter(noise_variance, "noise_variance", allow_zero=True)
    if split not in ("interpolation", "forecast"):
        raise ValueError("split must be 'interpolation' or 'forecast'")
    L, degree_scale = make_laplacian(graph_kind, n=vertices, seed=graph_seed)
    times = np.arange(total_times, dtype=np.float64) / (total_times - 1)
    graph, temporal = _factors(L, times, tau, ell)
    latent_rng = np.random.default_rng(np.random.SeedSequence(draw_seed, spawn_key=(0,)))
    noise_rng = np.random.default_rng(np.random.SeedSequence(draw_seed, spawn_key=(1,)))
    mask_key = (2, 0 if split == "interpolation" else 1)
    mask_rng = np.random.default_rng(np.random.SeedSequence(draw_seed, spawn_key=mask_key))
    latent_matrix = (cholesky(graph, lower=True)
                     @ latent_rng.standard_normal((vertices, total_times))
                     @ cholesky(temporal, lower=True).T)
    latent = latent_matrix.reshape(-1)
    noise_field = np.sqrt(noise) * noise_rng.standard_normal(vertices * total_times)
    y = latent + noise_field
    all_ids = np.arange(vertices * total_times, dtype=np.int64)
    all_pairs = np.column_stack((all_ids // total_times, all_ids % total_times)).astype(np.float64)
    if split == "interpolation":
        count = int(0.4 * len(all_ids)) if train_count is None else train_count
        count = _dimension(count, "train_count")
        if count >= len(all_ids):
            raise ValueError("interpolation train_count must leave at least one test pair")
        train_ids = mask_rng.choice(all_ids, size=count, replace=False)
        test_ids = np.setdiff1d(all_ids, train_ids, assume_unique=True)
        start = None
    else:
        start = max(1, int(0.6 * total_times)) if forecast_start is None else forecast_start
        start = _dimension(start, "forecast_start")
        if start >= total_times:
            raise ValueError("forecast_start must precede at least one declared time")
        candidates = all_ids[all_pairs[:, 1] < start]
        count = len(candidates) // 2 if train_count is None else train_count
        count = _dimension(count, "train_count")
        if count > len(candidates):
            raise ValueError("forecast train_count exceeds the historical candidate count")
        train_ids = mask_rng.choice(candidates, size=count, replace=False)
        test_ids = all_ids[all_pairs[:, 1] >= start]
    metadata = dict(seed=draw_seed, split=split, graph_kind=graph_kind, n=vertices,
                    num_times=total_times, graph_seed=graph_seed, degree_scale=degree_scale,
                    tau=float(tau), ell=float(ell), noise_variance=noise,
                    train_count=count, test_count=len(test_ids), forecast_start=start,
                    latent_stream_key=[0], noise_stream_key=[1],
                    split_stream_key=list(mask_key), independent_unit="full latent/noisy field per seed")
    return dict(L=L, time_grid=times, all_pairs=all_pairs,
                train_pairs=all_pairs[train_ids], test_pairs=all_pairs[test_ids],
                train_ids=train_ids, test_ids=test_ids, latent=latent, y=y,
                noise=noise_field, meta=metadata)
