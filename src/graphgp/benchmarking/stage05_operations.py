"""Stage 05 operations/profiling helpers; imported only inside isolated workers."""
from contextlib import ExitStack, contextmanager
import statistics
import time
from unittest.mock import patch

import gpytorch
from linear_operator.operators import DenseLinearOperator, KroneckerProductLinearOperator, LinearOperator
import numpy as np
import torch

from graphgp.benchmarking.stage05_cases import ARM_CONFIG
from graphgp.benchmarking.worker import guarded_sparse
from graphgp.inference.predict import Marginals
from graphgp.inference.solver_settings import iterative_context
from graphgp.operators.chebyshev import ChebyshevFunctionLinearOperator
from graphgp.validation import exhaustive_trace_probes


def product_measurement(action,rhs):
    """One warmup and five measured products; output summaries are untimed."""
    with torch.no_grad():
        action(rhs)
        samples=[]
        for _ in range(5):
            start=time.perf_counter();value=action(rhs);samples.append(time.perf_counter()-start)
    array=value.detach().numpy() if torch.is_tensor(value) else np.asarray(value)
    indices=np.linspace(0,len(array)-1,min(32,len(array)),dtype=np.int64)
    if not np.isfinite(array).all():raise ArithmeticError('nonfinite product')
    return dict(elapsed_seconds=statistics.median(samples),measurement_seconds=samples,warmup_count=1,
                product_shape=list(array.shape),sample_rows=indices.tolist(),product_sample=array[indices].tolist(),
                product_norm=float(np.linalg.norm(array)))


def entry_measurement(operator,count,unique_count):
    rows=(torch.arange(count,device=operator.device)*17)%operator.n
    cols=torch.arange(count,device=operator.device)%min(unique_count,operator.n)
    return product_measurement(lambda _:operator._get_indices(rows,cols),rows)


def mll_measurement(model,train,values,seed,updates=0):
    objective=gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood,model)
    start=time.perf_counter()
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.05) if updates else None
    path=[];forward=backward=0.0
    for step in range(updates+1):
        model.zero_grad();torch.manual_seed(seed)
        tick=time.perf_counter();loss=-objective(model(train),values);forward+=time.perf_counter()-tick
        tick=time.perf_counter();loss.backward();backward+=time.perf_counter()-tick
        kernel=model.covar_module
        grads=dict(heat_time=kernel.graph_kernel.raw_heat_time.grad.item(),lengthscale=kernel.time_kernel.raw_lengthscale.grad.item())
        physical={}
        for name,raw,constraint in [('heat_time',kernel.graph_kernel.raw_heat_time,kernel.graph_kernel.raw_heat_time_constraint),
                                    ('lengthscale',kernel.time_kernel.raw_lengthscale,kernel.time_kernel.raw_lengthscale_constraint)]:
            jacobian=torch.autograd.grad(constraint.transform(raw).sum(),raw)[0].item();physical[name]=grads[name]/jacobian
        if model.likelihood.raw_noise.requires_grad:
            grads['noise_variance']=model.likelihood.raw_noise.grad.item()
        if not np.isfinite([loss.item(),*grads.values()]).all():raise ArithmeticError('nonfinite MLL/gradient')
        path.append(dict(step=step,nll=loss.item(),raw_gradients=grads,physical_gradients=physical,
                         heat_time=kernel.heat_time.item(),lengthscale=kernel.lengthscale.item(),noise_variance=model.likelihood.noise.item()))
        if optimizer and step<updates:optimizer.step()
    return dict(elapsed_seconds=time.perf_counter()-start,forward_seconds=forward,backward_seconds=backward,
                nll=loss.item(),raw_gradients=grads,physical_gradients=physical,trajectory=path,updates=updates,
                final_gradient_norm=float(np.linalg.norm(list(grads.values()))))


def profile_mll(model,train,values,seed):
    """Nonoverlapping graph ranges inside a single inclusive profiled MLL/backward."""
    from torch.profiler import profile,ProfilerActivity,record_function
    original=ChebyshevFunctionLinearOperator._matmul
    depth=0;maximum_depth=0
    def mm(operator,rhs):
        nonlocal depth,maximum_depth
        depth+=1;maximum_depth=max(maximum_depth,depth)
        try:
            with record_function('stage05_chebyshev_graph_mvp'):return original(operator,rhs)
        finally:depth-=1
    with patch.object(ChebyshevFunctionLinearOperator,'_matmul',mm):
        with profile(activities=[ProfilerActivity.CPU],record_shapes=True,profile_memory=True) as profiler:
            start=time.perf_counter()
            with record_function('stage05_inclusive_mll_backward'):
                result=mll_measurement(model,train,values,seed)
            profiled_wall=time.perf_counter()-start
    events=profiler.key_averages()
    ranges={e.key:e for e in events if e.key.startswith('stage05_')}
    graph=ranges['stage05_chebyshev_graph_mvp'];outer=ranges['stage05_inclusive_mll_backward']
    if maximum_depth!=1:raise AssertionError('graph ranges overlap; cannot use their sum for the gate')
    top=sorted(events,key=lambda x:x.self_cpu_time_total,reverse=True)[:25]
    result.update(profiled_wall_seconds=profiled_wall,profile_inclusive_cpu_us=outer.cpu_time_total,
        graph_range_cpu_us=graph.cpu_time_total,graph_fraction_of_inclusive=graph.cpu_time_total/outer.cpu_time_total,
        graph_fraction_of_profiled_wall=graph.cpu_time_total/(profiled_wall*1e6),maximum_graph_range_depth=maximum_depth,
        events=[dict(name=e.key,calls=e.count,self_cpu_us=e.self_cpu_time_total,inclusive_cpu_us=e.cpu_time_total,
                     net_cpu_memory_bytes=e.cpu_memory_usage) for e in top],
        profiler_note='CPU event net memory is not peak allocation; profile timings excluded from headline performance')
    return result


def gather_scatter_measurement(problem,width,kind):
    from linear_operator.utils.interpolation import left_interp,left_t_interp
    indices=torch.from_numpy((problem['train_pairs'][:,0]*len(problem['time_grid'])+problem['train_pairs'][:,1]).astype(np.int64))[:,None]
    weights=torch.ones(indices.shape,dtype=torch.float64)
    full=torch.from_numpy(problem['full_rhs'][:,:width]);selected=full[indices[:,0]]
    if kind=='gather':return product_measurement(lambda x:left_interp(indices,weights,x),full)
    return product_measurement(lambda x:left_t_interp(indices,weights,x,output_dim=len(full)),selected)


@contextmanager
def action_diagnostics(n, num_times):
    """Count graph/time factor actions and overlapping outer entry requests."""
    result = dict(
        dense_graph_calls=0, dense_graph_seconds=0.0, dense_graph_widths={},
        temporal_calls=0, temporal_seconds=0.0, entry_unique_columns=[], entry_counts=[],
    )
    original = DenseLinearOperator._matmul
    get = ChebyshevFunctionLinearOperator._get_indices

    def mm(op, rhs):
        prefix = (
            'dense_graph' if tuple(op.shape) == (n, n)
            else ('temporal' if tuple(op.shape) == (num_times, num_times) else None)
        )
        tick = time.perf_counter()
        try:
            return original(op, rhs)
        finally:
            if prefix:
                result[prefix + '_calls'] += 1
                result[prefix + '_seconds'] += time.perf_counter() - tick
                if prefix == 'dense_graph':
                    width = str(rhs.shape[-1])
                    result['dense_graph_widths'][width] = result['dense_graph_widths'].get(width, 0) + 1

    def entries(op, row, col, *batch):
        result['entry_unique_columns'].append(int(torch.unique(col).numel()))
        result['entry_counts'].append(int(torch.broadcast_tensors(row, col)[0].numel()))
        return get(op, row, col, *batch)

    with (
        patch.object(DenseLinearOperator, '_matmul', mm),
        patch.object(ChebyshevFunctionLinearOperator, '_get_indices', entries),
    ):
        yield result


def scalar_entries(operator, row_index, col_index, *batch):
    """Keep the historical scalar entry algorithm for labeled timing controls."""
    if batch:
        raise ValueError('batched graph indexing unsupported')
    rows, cols = torch.broadcast_tensors(row_index, col_index)
    result = []
    for row, col in zip(rows.reshape(-1), cols.reshape(-1)):
        basis = operator.cheb_coeffs.new_zeros(operator.n, 1)
        basis[col, 0] = 1
        result.append((operator @ basis)[row, 0])
    return torch.stack(result).reshape(rows.shape) if result else operator.cheb_coeffs.new_empty(rows.shape)


@contextmanager
def inference_context(arm, config, *, entry_strategy='batched'):
    """Match each benchmark arm's direct/iterative settings and sparse guards."""
    mode = ARM_CONFIG[arm]
    counts = dict(operator_cholesky_calls=0, torch_cholesky_calls=0)
    original_operator = LinearOperator._cholesky
    original_cholesky = torch.linalg.cholesky_ex

    def op_chol(self, *args, **kwargs):
        counts['operator_cholesky_calls'] += 1
        return original_operator(self, *args, **kwargs)

    def tensor_chol(*args, **kwargs):
        counts['torch_cholesky_calls'] += 1
        return original_cholesky(*args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(gpytorch.settings.max_eager_kernel_size(0))
        if entry_strategy == 'scalar_reference':
            stack.enter_context(patch.object(ChebyshevFunctionLinearOperator, '_get_indices', scalar_entries))
        elif entry_strategy != 'batched':
            raise ValueError('unknown entry strategy')
        if mode['inference'] == 'direct':
            stack.enter_context(gpytorch.settings.max_cholesky_size(10**9))
            stack.enter_context(gpytorch.settings.fast_computations(
                covar_root_decomposition=False, log_prob=False, solves=False,
            ))
            stack.enter_context(patch.object(LinearOperator, '_cholesky', op_chol))
            stack.enter_context(patch.object(torch.linalg, 'cholesky_ex', tensor_chol))
        else:
            stack.enter_context(iterative_context(config))
            # Exhaustive probes belong only to the predeclared tiny accuracy profile.
            if config.trace_samples == config.lanczos_iterations and config.cg_tolerance == 1e-8:
                stack.enter_context(exhaustive_trace_probes())
            if mode['backend'] == 'sparse_chebyshev':
                stack.enter_context(guarded_sparse())
            else:
                for module, name in [
                    (LinearOperator, '_cholesky'), (torch.linalg, 'cholesky'),
                    (torch.linalg, 'cholesky_ex'), (KroneckerProductLinearOperator, 'to_dense'),
                ]:
                    stack.enter_context(patch.object(
                        module, name, side_effect=AssertionError('B iterative attempted direct solve/full prior'),
                    ))
        yield counts


def warm_prediction(model, targets, *, mean_only=False):
    """Reuse conditioning only after the explicitly labeled cold benchmark call."""
    if model.prediction_strategy is None:
        raise ValueError('warm measurement needs a cold prediction first')
    means, variances = [], []
    with (
        torch.no_grad(), gpytorch.settings.fast_pred_var(False),
        gpytorch.settings.skip_posterior_variances(mean_only),
    ):
        for part in targets.split(64):
            posterior = model(part)
            means.append(posterior.mean)
            if not mean_only:
                value = posterior.lazy_covariance_matrix.diagonal()
                if not torch.isfinite(value).all() or (value < 0).any():
                    raise ArithmeticError('invalid warm latent variance')
                variances.append(value)
    mean = torch.cat(means)
    if not torch.isfinite(mean).all():
        raise ArithmeticError('invalid warm mean')
    variance = None if mean_only else torch.cat(variances)
    return Marginals(mean, variance, None if mean_only else variance + model.likelihood.noise.detach())
