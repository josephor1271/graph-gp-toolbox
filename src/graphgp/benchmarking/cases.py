"""Frozen Stage 04 workloads and performance-independent random streams.

Only standard-library modules are imported at module scope, so the benchmark
coordinator can set child thread environments before importing numerical code.
The generated field is a degree-32 approximate GP-like benchmark signal. It is
not the independent exact-GP sampling oracle used for numerical validation.
"""

from dataclasses import asdict, dataclass
import hashlib
import math


MASTER_SEED = 202604
TAU = 1.2
LENGTHSCALE = 0.24
NOISE_VARIANCE = 0.06
DEGREE = 32
PROBE_SEED_LABELS = (301, 302, 303)


# These numeric identifiers are data, not a hash of a Python object or a label.
# Cases with the same (n,T) share the canonical stream identifier in column 8.
_TABLE = (
    (1, "n016_t08_m0032_q0064", 16, 8, 32, 64, ("N",), 1),
    (2, "n049_t08_m0098_q0064", 49, 8, 98, 64, ("N",), 2),
    (3, "n100_t08_m0200_q0064", 100, 8, 200, 64, ("N", "T", "M", "Q"), 3),
    (4, "n225_t08_m0450_q0064", 225, 8, 450, 64, ("N",), 4),
    (5, "n400_t08_m0800_q0064", 400, 8, 800, 64, ("N",), 5),
    (6, "n100_t04_m0200_q0064", 100, 4, 200, 64, ("T",), 6),
    (7, "n100_t16_m0200_q0064", 100, 16, 200, 64, ("T",), 7),
    (8, "n100_t32_m0200_q0064", 100, 32, 200, 64, ("T",), 8),
    (9, "n100_t08_m0080_q0064", 100, 8, 80, 64, ("M",), 3),
    (10, "n100_t08_m0400_q0064", 100, 8, 400, 64, ("M",), 3),
    (11, "n100_t08_m0200_q0016", 100, 8, 200, 16, ("Q",), 3),
    (12, "n100_t08_m0200_q0256", 100, 8, 200, 256, ("Q",), 3),
)
_IDENTIFIERS = {row[1]: (row[0], row[7]) for row in _TABLE}
_IDENTIFIERS["smoke_n016_t08_m0032_q0008"] = (0, 1)


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class Case:
    """One immutable workload; ``m`` and ``q`` count distinct coordinates."""

    case_id: str
    n: int
    num_times: int
    m: int
    q: int
    groups: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.case_id, str) or self.case_id not in _IDENTIFIERS:
            raise ValueError("case_id must be one of the fixed Stage 04 identifiers")
        _integer(self.n, "n", 4)
        _integer(self.num_times, "num_times", 2)
        _integer(self.m, "m")
        _integer(self.q, "q")
        if math.isqrt(self.n)**2 != self.n:
            raise ValueError("n must be a square grid size")
        if self.m + self.q > self.nT:
            raise ValueError("m + q must fit the declared full grid")
        if (not isinstance(self.groups, tuple) or not self.groups
                or any(group not in ("N", "T", "M", "Q", "smoke")
                       for group in self.groups)
                or len(set(self.groups)) != len(self.groups)):
            raise ValueError("groups must be a nonempty tuple of distinct declared groups")
        expected = next((row for row in _TABLE if row[1] == self.case_id), None)
        if expected is not None and (self.n, self.num_times, self.m, self.q, self.groups) != expected[2:7]:
            raise ValueError("the fixed case identifier does not match its workload")
        if expected is None and (self.n, self.num_times, self.m, self.q, self.groups) != (16, 8, 32, 8, ("smoke",)):
            raise ValueError("the smoke identifier does not match its workload")

    @property
    def numeric_id(self):
        return _IDENTIFIERS[self.case_id][0]

    @property
    def integer_id(self):
        """Alias retained for case-stream metadata readability."""
        return self.numeric_id

    @property
    def stream_case_id(self):
        return _IDENTIFIERS[self.case_id][1]

    @property
    def nT(self):
        return self.n * self.num_times

    @property
    def observed_fraction(self):
        return self.m / self.nT

    def to_dict(self):
        result = asdict(self)
        result["groups"] = list(self.groups)
        result.update(numeric_id=self.numeric_id, integer_id=self.integer_id,
                      stream_case_id=self.stream_case_id,
                      nT=self.nT, observed_fraction=self.observed_fraction,
                      dense_eligible=dense_eligible(self))
        return result


def primary_cases():
    """Return all twelve predeclared configurations in stable table order."""
    return [Case(row[1], *row[2:7]) for row in _TABLE]


def smoke_case():
    return Case("smoke_n016_t08_m0032_q0008", 16, 8, 32, 8, ("smoke",))


def dense_eligible(case):
    return case.n <= 100 and case.nT <= 800 and case.m <= 400 and case.q <= 256


def profiles(m):
    """Return experimental settings; these are not production recommendations.

    ``exhaustive_tiny`` only describes the settings. Its separate orthonormal
    trace-probe policy must be applied by diagnostics and restricted to the
    declared n=16,T=8,m=32 case, never to a larger benchmark.
    """
    from graphgp.inference.solver_settings import IterativeConfig

    _integer(m, "m")
    configurations = {
        "fast_probe": (1e-3, 200, 8, 16),
        "balanced_probe": (1e-4, 500, 16, 32),
        "strict_probe": (1e-6, 500, 16, 32),
    }
    if m == 32:
        configurations["exhaustive_tiny"] = (1e-8, 500, m, m)
    return {
        name: IterativeConfig(cg_tolerance=tolerance, eval_cg_tolerance=tolerance,
                              max_cg_iterations=iterations, trace_samples=traces,
                              lanczos_iterations=lanczos, max_preconditioner_size=0,
                              terminate_cg_by_size=False)
        for name, (tolerance, iterations, traces, lanczos) in configurations.items()
    }


def _sequence(case, *stream_key):
    import numpy as np

    return np.random.SeedSequence(MASTER_SEED,
                                  spawn_key=(case.stream_case_id, *stream_key))


def probe_seed(case, seed_label):
    """Map the declared integer probe-seed label to an independent Torch seed."""
    import numpy as np

    _integer(seed_label, "seed_label", 0)
    return int(_sequence(case, 3, seed_label).generate_state(1, dtype=np.uint32)[0])


def _approximate_field(L, time_grid, Z):
    """Apply degree-32 heat-half and temporal Cholesky, without hidden jitter.

    The prescribed T=32 RBF factor can fail float64 Cholesky because it is
    numerically singular. The original factorization exception propagates so
    workers can record a data-construction failure for the unchanged workload.
    """
    import numpy as np
    import torch
    from graphgp.kernels.heat import GraphHeatKernel

    distance = time_grid[:, None] - time_grid[None, :]
    temporal = np.exp(-(distance * distance) / (2 * LENGTHSCALE**2))
    # Only the declared T×T temporal factor is factorized for data generation.
    temporal_factor = np.linalg.cholesky(temporal)
    graph_half = GraphHeatKernel(
        L, lambda_min=0.0, lambda_max=2.0, num_coefficients=DEGREE + 1,
        heat_time=torch.tensor(TAU / 2, dtype=torch.float64)).double()
    with torch.no_grad():
        filtered = graph_half.full_graph_operator() @ torch.from_numpy(Z)
    return (filtered.numpy() @ temporal_factor.T).reshape(-1)


def _fingerprint(L, arrays):
    """SHA256 of canonical sparse graph/data bytes for paired-worker audits."""
    import numpy as np

    digest = hashlib.sha256()
    for array in (L.data, L.indices, L.indptr, *arrays):
        values = np.asarray(array)
        # Fix integer width/endian so the definition is independent of CSR's
        # platform-selected index width. All benchmark floating arrays are f64.
        canonical = np.asarray(values, dtype="<i8" if values.dtype.kind in "iu" else "<f8")
        digest.update(str(canonical.shape).encode("ascii"))
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def make_problem(case):
    """Construct paired sparse graph/data without a dense graph covariance.

    The n=100,T=8 family reserves 400 possible training coordinates before
    choosing targets; hence its three training masks are prefixes of one fixed
    permutation and its target sets are fixed prefixes of another permutation.
    No target belongs to even the largest training mask. Every other family
    reserves its declared m coordinates. Outcomes never influence selection.
    """
    import numpy as np
    from graphgp.utils.graphs import make_laplacian

    if not isinstance(case, Case):
        raise TypeError("case must be a Stage 04 Case")
    L, graph_scale = make_laplacian("grid", n=case.n)
    times = np.arange(case.num_times, dtype=np.float64) / (case.num_times - 1)
    latent_rng = np.random.default_rng(_sequence(case, 0))
    noise_rng = np.random.default_rng(_sequence(case, 1))
    train_rng = np.random.default_rng(_sequence(case, 2, 0))
    target_rng = np.random.default_rng(_sequence(case, 2, 1))
    solve_rng = np.random.default_rng(_sequence(case, 4, 0))
    mvp_rng = np.random.default_rng(_sequence(case, 4, 1))
    latent = _approximate_field(L, times, latent_rng.standard_normal((case.n, case.num_times)))
    noise = math.sqrt(NOISE_VARIANCE) * noise_rng.standard_normal(case.nT)
    y = latent + noise
    permutation = train_rng.permutation(case.nT)
    reserve = 400 if (case.n, case.num_times) == (100, 8) else case.m
    train_ids = permutation[:case.m].copy()
    target_pool = np.setdiff1d(np.arange(case.nT, dtype=np.int64),
                             permutation[:reserve], assume_unique=True)
    target_ids = target_rng.permutation(target_pool)[:case.q].copy()
    all_ids = np.arange(case.nT, dtype=np.int64)
    all_pairs = np.column_stack((all_ids // case.num_times,
                                 all_ids % case.num_times)).astype(np.float64)
    train_y = y[train_ids]
    # Generate normals on the full grid and select training IDs, keeping the
    # two independent RHS streams identical across the nested density masks.
    independent_solve_rhs = solve_rng.standard_normal((case.nT, 2))[train_ids]
    solve_rhs = np.column_stack((train_y, independent_solve_rhs))
    mvp_rhs = mvp_rng.standard_normal((case.nT, 8))
    fingerprint = _fingerprint(L, (times, train_ids, target_ids, latent, y,
                                   solve_rhs, mvp_rhs))
    stream_metadata = {
        "master_seed": MASTER_SEED,
        "case_integer_id": case.integer_id,
        "stream_case_id": case.stream_case_id,
        "seed_sequence_scheme": "SeedSequence(202604, spawn_key=(stream_case_id, *stream_key))",
        "latent_stream_key": [case.stream_case_id, 0],
        "noise_stream_key": [case.stream_case_id, 1],
        "training_mask_stream_key": [case.stream_case_id, 2, 0],
        "target_mask_stream_key": [case.stream_case_id, 2, 1],
        "probe_stream_key_prefix": [case.stream_case_id, 3],
        "probe_seed_policy": "append seed label (301,302,303); first uint32 state is Torch seed",
        "solve_rhs_stream_key": [case.stream_case_id, 4, 0],
        "mvp_rhs_stream_key": [case.stream_case_id, 4, 1],
        "reserved_training_count": reserve,
        "sampling": "P_32(exp(-tau*L/2)) @ Z @ chol(KT).T; controlled approximate GP-like field",
        "ordering": "vertex-major/time-minor: v*T+j",
        "fingerprint": fingerprint,
    }
    return dict(L=L, graph_scale=graph_scale, time_grid=times,
                all_pairs=all_pairs, train_pairs=all_pairs[train_ids],
                target_pairs=all_pairs[target_ids], train_y=train_y,
                latent=latent, noise=noise, y=y, train_ids=train_ids,
                target_ids=target_ids, solve_rhs=solve_rhs, mvp_rhs=mvp_rhs,
                stream_metadata=stream_metadata, fingerprint=fingerprint)
