"""Scoped Stage 05 controls, reference entry path and prediction cache diagnostics."""
from contextlib import ExitStack, contextmanager
from unittest.mock import patch


@contextmanager
def action_diagnostics(n,num_times):
    """Graph/time factor actions and outer entry requests, with overlapping scopes explicit."""
    import time,torch
    from linear_operator.operators import DenseLinearOperator
    from graphgp.operators.chebyshev import ChebyshevFunctionLinearOperator
    result=dict(dense_graph_calls=0,dense_graph_seconds=0.0,dense_graph_widths={},
                temporal_calls=0,temporal_seconds=0.0,entry_unique_columns=[],entry_counts=[])
    original=DenseLinearOperator._matmul;get=ChebyshevFunctionLinearOperator._get_indices
    def mm(op,rhs):
        prefix='dense_graph' if tuple(op.shape)==(n,n) else ('temporal' if tuple(op.shape)==(num_times,num_times) else None)
        tick=time.perf_counter()
        try:return original(op,rhs)
        finally:
            if prefix:
                result[prefix+'_calls']+=1;result[prefix+'_seconds']+=time.perf_counter()-tick
                if prefix=='dense_graph':
                    width=str(rhs.shape[-1]);result['dense_graph_widths'][width]=result['dense_graph_widths'].get(width,0)+1
    def entries(op,row,col,*batch):
        result['entry_unique_columns'].append(int(torch.unique(col).numel()))
        result['entry_counts'].append(int(torch.broadcast_tensors(row,col)[0].numel()))
        return get(op,row,col,*batch)
    with patch.object(DenseLinearOperator,'_matmul',mm),patch.object(ChebyshevFunctionLinearOperator,'_get_indices',entries):yield result


def scalar_entries(operator,row_index,col_index,*batch):
    """Unmodified pre-Stage-05 scalar algorithm, for labeled timing controls."""
    import torch
    if batch:raise ValueError("batched graph indexing unsupported")
    rows,cols=torch.broadcast_tensors(row_index,col_index)
    result=[]
    for row,col in zip(rows.reshape(-1),cols.reshape(-1)):
        basis=operator.cheb_coeffs.new_zeros(operator.n,1);basis[col,0]=1
        result.append((operator@basis)[row,0])
    return torch.stack(result).reshape(rows.shape) if result else operator.cheb_coeffs.new_empty(rows.shape)


@contextmanager
def inference_context(arm,config,*,entry_strategy="batched"):
    import gpytorch
    import torch
    from linear_operator.operators import LinearOperator,KroneckerProductLinearOperator
    from graphgp.benchmarking.stage05_cases import ARM_CONFIG
    from graphgp.benchmarking.worker import guarded_sparse
    from graphgp.inference.solver_settings import iterative_context
    from graphgp.operators.chebyshev import ChebyshevFunctionLinearOperator
    from graphgp.validation import exhaustive_trace_probes
    mode=ARM_CONFIG[arm]
    counts=dict(operator_cholesky_calls=0,torch_cholesky_calls=0)
    original_operator=LinearOperator._cholesky
    original_cholesky=torch.linalg.cholesky_ex
    def op_chol(self,*args,**kwargs):
        counts['operator_cholesky_calls']+=1;return original_operator(self,*args,**kwargs)
    def tensor_chol(*args,**kwargs):
        counts['torch_cholesky_calls']+=1;return original_cholesky(*args,**kwargs)
    with ExitStack() as stack:
        stack.enter_context(gpytorch.settings.max_eager_kernel_size(0))
        if entry_strategy=='scalar_reference':
            stack.enter_context(patch.object(ChebyshevFunctionLinearOperator,'_get_indices',scalar_entries))
        elif entry_strategy!='batched':raise ValueError('unknown entry strategy')
        if mode['inference']=='direct':
            stack.enter_context(gpytorch.settings.max_cholesky_size(10**9))
            stack.enter_context(gpytorch.settings.fast_computations(covar_root_decomposition=False,log_prob=False,solves=False))
            stack.enter_context(patch.object(LinearOperator,'_cholesky',op_chol))
            stack.enter_context(patch.object(torch.linalg,'cholesky_ex',tensor_chol))
        else:
            stack.enter_context(iterative_context(config))
            if config.trace_samples==config.lanczos_iterations and config.cg_tolerance==1e-8:
                stack.enter_context(exhaustive_trace_probes())
            if mode['backend']=='sparse_chebyshev':stack.enter_context(guarded_sparse())
            else:
                for module,name in [(LinearOperator,'_cholesky'),(torch.linalg,'cholesky'),(torch.linalg,'cholesky_ex'),(KroneckerProductLinearOperator,'to_dense')]:
                    stack.enter_context(patch.object(module,name,side_effect=AssertionError('B iterative attempted direct solve/full prior')))
        yield counts


def warm_prediction(model,targets,*,mean_only=False):
    """Call GPyTorch with its existing strategy; never change public prediction API.

    Used only by the explicitly labeled benchmark after an unchanged model's
    cold public call. It retains GPyTorch conditioning and exact lazy diagonals.
    """
    import torch,gpytorch
    from graphgp.inference.predict import Marginals
    if model.prediction_strategy is None:raise ValueError('warm measurement needs a cold prediction first')
    means,variances=[],[]
    with torch.no_grad(),gpytorch.settings.fast_pred_var(False),gpytorch.settings.skip_posterior_variances(mean_only):
        for part in targets.split(64):
            posterior=model(part);means.append(posterior.mean)
            if not mean_only:
                value=posterior.lazy_covariance_matrix.diagonal()
                if not torch.isfinite(value).all() or (value<0).any():raise ArithmeticError('invalid warm latent variance')
                variances.append(value)
    mean=torch.cat(means)
    if not torch.isfinite(mean).all():raise ArithmeticError('invalid warm mean')
    variance=None if mean_only else torch.cat(variances)
    return Marginals(mean,variance,None if mean_only else variance+model.likelihood.noise.detach())


def selected_dense_reference(problem,case):
    """Independent selected-space oracle up to n400, without a full nT prior."""
    import numpy as np
    import torch,gpytorch
    from scipy.linalg import expm,cho_factor,cho_solve
    from graphgp.utils.study_domain import HEAT_RANGE,LENGTHSCALE_RANGE
    if case.n>400 or case.secondary:raise ValueError('independent dense reference restricted to primary n<=400')
    L=problem['L'].toarray();G=expm(-1.2*L);t=problem['time_grid']
    train=problem['train_pairs'].astype(np.int64);target=problem['target_pairs'].astype(np.int64)
    def block(left,right):
        return G[left[:,0,None],right[None,:,0]]*np.exp(-(t[left[:,1,None]]-t[right[None,:,1]])**2/(2*.24**2))
    A=block(train,train)+.06*np.eye(case.m);factor=cho_factor(A,lower=True);y=problem['train_y']
    alpha=cho_solve(factor,y);cross=block(target,train)
    mean=cross@alpha;variance=np.diag(G)[target[:,0]]-np.sum(cross*cho_solve(factor,cross.T).T,axis=1)
    nll=float((np.log(np.diag(factor[0])).sum()*2+y@alpha+case.m*np.log(2*np.pi))/(2*case.m))
    raw=[];physical=[]
    for value,bounds in [(1.2,HEAT_RANGE),(.24,LENGTHSCALE_RANGE)]:
        constraint=gpytorch.constraints.Interval(*bounds).double();parameter=constraint.inverse_transform(torch.tensor(value,dtype=torch.float64)).detach().requires_grad_(True)
        raw.append(parameter);physical.append(constraint.transform(parameter))
    tau,ell=physical;graph=torch.matrix_exp(-tau*torch.from_numpy(L));pairs=torch.from_numpy(train);times=torch.from_numpy(t)
    cov=graph[pairs[:,0,None],pairs[None,:,0]]*torch.exp(-(times[pairs[:,1,None]]-times[pairs[None,:,1]])**2/(2*ell**2))+.06*torch.eye(case.m,dtype=torch.float64)
    chol=torch.linalg.cholesky(cov);v=torch.from_numpy(y)
    loss=(2*chol.diagonal().log().sum()+v@torch.cholesky_solve(v[:,None],chol)[:,0]+case.m*np.log(2*np.pi))/(2*case.m)
    gradients=torch.autograd.grad(loss,raw)
    products={}
    temporal=np.exp(-(t[:,None]-t[None,:])**2/(2*.24**2))
    for width in (1,8,32):
        graph_product=G@problem['graph_rhs'][:,:width]
        X=problem['full_rhs'][:,:width].reshape(case.n,case.num_times,width)
        full_product=np.stack([G@X[:,:,k]@temporal.T for k in range(width)],axis=-1).reshape(case.nT,width)
        for kind,array in [('graph_mvp',graph_product),('full_mvp',full_product)]:
            indices=np.linspace(0,len(array)-1,min(32,len(array)),dtype=np.int64)
            products[kind+str(width)]=dict(sample_rows=indices.tolist(),product_sample=array[indices].tolist())
    return dict(nll=nll,raw_gradients=dict(zip(('heat_time','lengthscale'),[float(x) for x in gradients])),
        mean=mean.tolist(),latent_variance=variance.tolist(),observation_variance=(variance+.06).tolist(),
        exact_solutions=cho_solve(factor,problem['solve_rhs']).tolist(),products=products,
        reference_kind='independent scipy expm/selected NumPy RBF/Cholesky and Torch matrix-exp derivatives')
