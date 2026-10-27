"""Predeclared Stage 05 shapes and streams; no numerical imports at module scope."""
from dataclasses import asdict, dataclass

MASTER_SEED = 202605
BASELINE_SHA = "ac8bffe0443436d623840130609582f599824d9d"
ARMS = ("A_dense_direct", "B_dense_iterative", "C32_sparse_iterative", "C16_sparse_iterative")
ARM_CONFIG = {
    ARMS[0]: dict(backend="dense_exact", inference="direct", degree=32),
    ARMS[1]: dict(backend="dense_exact", inference="iterative", degree=32),
    ARMS[2]: dict(backend="sparse_chebyshev", inference="iterative", degree=32),
    ARMS[3]: dict(backend="sparse_chebyshev", inference="iterative", degree=16),
}
_TABLE = (
    ("s05_n016",16,8,32,32), ("s05_n049",49,8,98,64),
    ("s05_n100",100,8,200,64), ("s05_n400",400,8,800,64),
    ("s05_n1600",1600,8,3200,64), ("s05_n6400_mvp",6400,8,0,0),
    ("s05_q016",100,8,200,16), ("s05_q256",100,8,200,256),
    ("s05_q256_n400",400,8,800,256), ("s05_t32_secondary",100,32,200,64),
)
_STREAM = {(16,8):1,(49,8):2,(100,8):3,(400,8):4,(1600,8):5,(6400,8):6,(100,32):7}


@dataclass(frozen=True)
class Stage05Case:
    case_id: str
    n: int
    num_times: int
    m: int
    q: int

    def __post_init__(self):
        if (self.case_id,self.n,self.num_times,self.m,self.q) not in _TABLE:
            raise ValueError("case must match the fixed Stage 05 table")
        if any(isinstance(x,bool) or not isinstance(x,int) for x in (self.n,self.num_times,self.m,self.q)):
            raise ValueError("case dimensions must be integers")

    @property
    def stream_id(self): return _STREAM[self.n,self.num_times]

    @property
    def nT(self): return self.n*self.num_times

    @property
    def secondary(self): return self.num_times==32

    @property
    def mvp_only(self): return self.m==0

    def to_dict(self): return asdict(self)


def cases(): return [Stage05Case(*row) for row in _TABLE]


def arms(case):
    return ARMS if case.n<=400 and not case.secondary else ARMS[2:]


def profile(name, m):
    from graphgp.inference.solver_settings import IterativeConfig
    if name not in ("balanced","strict","exhaustive_tiny"):
        raise ValueError("unknown Stage 05 profile")
    return IterativeConfig(cg_tolerance=1e-8 if name=="exhaustive_tiny" else (1e-6 if name=="strict" else 1e-4),
        eval_cg_tolerance=1e-8 if name=="exhaustive_tiny" else (1e-6 if name=="strict" else 1e-4),
        max_cg_iterations=500,trace_samples=m if name=="exhaustive_tiny" else 16,
        lanczos_iterations=m if name=="exhaustive_tiny" else 32,
        max_preconditioner_size=0,terminate_cg_by_size=False)


def sequence(case,*key):
    import numpy as np
    return np.random.SeedSequence(MASTER_SEED,spawn_key=(case.stream_id,*key))


def probe_seed(case,label):
    if isinstance(label,bool) or not isinstance(label,int) or label not in (301,302,303):
        raise ValueError("probe label must be 301,302,303")
    return int(sequence(case,3,label).generate_state(1)[0])


def make_problem(case):
    import numpy as np
    import torch
    from graphgp.benchmarking.cases import _approximate_field,_fingerprint
    from graphgp.kernels.heat import GraphHeatKernel
    from graphgp.utils.graphs import make_laplacian
    rng=lambda *key: np.random.default_rng(sequence(case,*key))
    L,scale=make_laplacian("grid",case.n)
    time=np.arange(case.num_times,dtype=np.float64)/(case.num_times-1)
    z=rng(0).standard_normal((case.n,case.num_times))
    temporal_metadata=dict(method="plain temporal Cholesky",correction=0.0)
    if not case.secondary:
        latent=_approximate_field(L,time,z)
    else:
        temporal=np.exp(-(time[:,None]-time[None,:])**2/(2*.24**2))
        symmetric=(temporal+temporal.T)/2
        values,vectors=np.linalg.eigh(symmetric)
        tolerance=1e-12*max(1.0,float(values[-1]))
        if not np.isfinite(values).all() or values[0] < -tolerance:
            raise ArithmeticError("T32 data temporal factor violates the declared PSD tolerance")
        clipped=np.maximum(values,0)
        factor=(vectors*np.sqrt(clipped))@vectors.T
        correction=float(np.max(np.abs((vectors*clipped)@vectors.T-symmetric)))
        graph=GraphHeatKernel(L,lambda_min=0,lambda_max=2,num_coefficients=33,num_nodes=33,
                              heat_time=torch.tensor(.6,dtype=torch.float64)).double()
        with torch.no_grad():latent=((graph.full_graph_operator()@torch.from_numpy(z)).numpy()@factor.T).reshape(-1)
        temporal_metadata=dict(method="secondary-only spectral PSD data square root",minimum_eigenvalue=float(values[0]),
            maximum_eigenvalue=float(values[-1]),eigenvalue_tolerance=tolerance,effective_rank=int(np.sum(values>tolerance)),
            positive_eigenvalues=int(np.sum(values>0)),maximum_covariance_correction=correction,
            maximum_eigenvalue_correction=float(np.max(np.abs(clipped-values))))
    noise=np.sqrt(.06)*rng(1).standard_normal(case.nT);y=latent+noise
    permutation=rng(2,0).permutation(case.nT);train_ids=permutation[:case.m]
    target_ids=rng(2,1).permutation(np.setdiff1d(np.arange(case.nT),train_ids))[:case.q]
    ids=np.arange(case.nT);pairs=np.column_stack((ids//case.num_times,ids%case.num_times)).astype(np.float64)
    train_y=y[train_ids]
    rhs=np.column_stack((train_y,rng(4,0).standard_normal(case.nT)[train_ids],rng(4,1).standard_normal(case.nT)[train_ids]))
    graph_rhs=rng(5,0).standard_normal((case.n,128));full_rhs=rng(5,1).standard_normal((case.nT,32))
    if not np.isfinite(y).all():raise ArithmeticError("nonfinite approximate benchmark signal")
    fingerprint=_fingerprint(L,(time,latent,y,train_ids,target_ids,rhs,graph_rhs,full_rhs))
    base_fingerprint=_fingerprint(L,(time,latent,y,train_ids,rhs,graph_rhs,full_rhs))
    metadata=dict(master_seed=MASTER_SEED,stream_id=case.stream_id,stream_scheme="SeedSequence(202605,spawn_key=(stream_id,*key))",
        latent=[case.stream_id,0],noise=[case.stream_id,1],train_mask=[case.stream_id,2,0],target_mask=[case.stream_id,2,1],
        probe=[case.stream_id,3,"label"],solve_normals=[[case.stream_id,4,0],[case.stream_id,4,1]],
        graph_rhs=[case.stream_id,5,0],full_rhs=[case.stream_id,5,1],
        sampling="degree32 heat-half @ Z @ temporal-factor.T; approximate GP-like signal",temporal_factor=temporal_metadata)
    return dict(L=L,graph_scale=scale,time_grid=time,train_pairs=pairs[train_ids],target_pairs=pairs[target_ids],train_y=train_y,
        latent=latent,y=y,noise=noise,train_ids=train_ids,target_ids=target_ids,solve_rhs=rhs,
        graph_rhs=graph_rhs,full_rhs=full_rhs,mvp_rhs=full_rhs[:,:8],fingerprint=fingerprint,
        base_fingerprint=base_fingerprint,stream_metadata=metadata)
