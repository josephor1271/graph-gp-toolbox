"""Small exact-GP recovery study; dense oracles/exhaustive probes are intentional.

This is a scientific validation harness, not a scalable training recipe.
"""
import csv
import json
import math
from pathlib import Path
import time
import warnings

import gpytorch
import numpy as np
import torch
from linear_operator.utils.warnings import NumericalWarning

from graphgp.inference.predict import predict_marginals
from graphgp.kernels.dense import DenseChebyshevGraphHeatKernel
from graphgp.kernels.heat import GraphHeatKernel
from graphgp.kernels.safety import heat_interpolation_certificate
from graphgp.models.factory import make_heat_gp, HEAT_RANGE, NOISE_RANGE, AMPLITUDE_RANGE
from graphgp.utils.graphs import make_laplacian
from graphgp.validation import inference_context, write_csv, environment_metadata

METHODS = ('A', 'B', 'C')


def sample_problem(family, *, n=49, seed=11, heat_time=0.6, noise=0.03,
                   amplitude=1.0, train_fraction=0.75):
    L, scale = make_laplacian(family, n)
    values, vectors = torch.linalg.eigh(torch.from_numpy(L.toarray()))
    covariance = amplitude * (vectors * torch.exp(-heat_time*values)) @ vectors.T
    rng = torch.Generator().manual_seed(seed)
    latent = torch.linalg.cholesky(covariance) @ torch.randn(n, generator=rng, dtype=torch.float64)
    y = latent + math.sqrt(noise)*torch.randn(n, generator=rng, dtype=torch.float64)
    permutation = torch.randperm(n, generator=rng)
    count = int(n*train_fraction)
    if not 1 < count < n:
        raise ValueError('split needs at least two observations and one target')
    return dict(L=L, scale=scale, evals=values, evecs=vectors, y=y, latent=latent,
                train=permutation[:count], test=permutation[count:], family=family,
                seed=seed, heat_time=heat_time, noise=noise, amplitude=amplitude,
                train_fraction=train_fraction)


def exact_nll(problem, heat_time, noise, amplitude=1.0):
    """Independent dense direct scalar oracle; normalized per observation."""
    train = problem['train']
    vectors = problem['evecs'][train]
    cov = amplitude*(vectors*torch.exp(-heat_time*problem['evals']))@vectors.T
    cov = cov + noise*torch.eye(len(train), dtype=torch.float64)
    chol = torch.linalg.cholesky(cov)
    y = problem['y'][train]
    return float((y @ torch.cholesky_solve(y[:, None], chol)[:, 0]
                  + 2*chol.diagonal().log().sum() + len(train)*math.log(2*math.pi))/(2*len(train)))


def parameters(model):
    return dict(heat_time=model.covar_module.base_kernel.heat_time.item(),
                noise=model.likelihood.noise.item(), amplitude=model.covar_module.outputscale.item())


def fit(problem, method, *, initial=(0.3, 0.04), steps=180, learn_amplitude=False,
        degree=32, cg_tolerance=1e-4):
    train, target = problem['train'], problem['test']
    model = make_heat_gp(problem['L'], train, problem['y'][train], method,
        heat_time=initial[0], noise=initial[1], learn_amplitude=learn_amplitude, degree=degree)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.07)
    trajectory = []
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with inference_context(method, len(train), seed=problem['seed'], cg_tolerance=cg_tolerance):
            for step in range(steps+1):
                optimizer.zero_grad()
                loss = -mll(model(train), problem['y'][train])
                if not torch.isfinite(loss):
                    raise ArithmeticError('nonfinite loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ArithmeticError('nonfinite gradient')
                grad_norm = math.sqrt(sum(float(p.grad.square().sum()) for p in model.parameters() if p.grad is not None))
                trajectory.append(dict(step=step, loss=loss.item(), grad_norm=grad_norm, **parameters(model)))
                if step < steps:
                    optimizer.step()
            training_seconds = time.perf_counter()-start
            moments = predict_marginals(model, target)
    elapsed = time.perf_counter()-start
    fitted = parameters(model)
    oracle = make_heat_gp(problem['L'], train, problem['y'][train], 'A', **fitted,
                          learn_amplitude=learn_amplitude)
    with inference_context('A', len(train)):
        reference = predict_marginals(oracle, target)
    residual = problem['y'][target]-moments.mean
    variance = moments.observation_variance
    poly = DenseChebyshevGraphHeatKernel(problem['L'], heat_time=fitted['heat_time'],
        lambda_max=2, num_coefficients=degree+1, num_nodes=degree+1)
    approximate_spectrum = poly.spectral_values().detach()
    exact_spectrum = torch.exp(-fitted['heat_time']*problem['evals'])
    result = dict(status='ok', **fitted, own_nll=trajectory[-1]['loss'],
        exact_nll=exact_nll(problem, **fitted),
        truth_nll=exact_nll(problem, problem['heat_time'],problem['noise'],problem['amplitude']),
        rmse=float(residual.square().mean().sqrt()),
        marginal_nlpd=float((0.5*(math.log(2*math.pi)+variance.log()+residual.square()/variance)).mean()),
        coverage95=float((residual.abs() <= 1.959963984540054*variance.sqrt()).double().mean()),
        latent_rmse=float((problem['latent'][target]-moments.mean).square().mean().sqrt()),
        covariance_relative_error=float((approximate_spectrum-exact_spectrum).norm()/exact_spectrum.norm()),
        polynomial_spectral_min=float(approximate_spectrum.min()),
        posterior_mean_max_error=float((moments.mean-reference.mean).abs().max()),
        posterior_variance_max_error=float((moments.latent_variance-reference.latent_variance).abs().max()),
        posterior_mean_relative_error=float((moments.mean-reference.mean).norm()/reference.mean.norm().clamp_min(1e-15)),
        posterior_variance_relative_error=float((moments.latent_variance-reference.latent_variance).norm()/reference.latent_variance.norm()),
        training_seconds=training_seconds, fit_predict_seconds=elapsed,
        final_gradient_norm=trajectory[-1]['grad_norm'],
        numerical_warnings=json.dumps(sorted(set(str(w.message) for w in caught
            if issubclass(w.category, NumericalWarning)))),
        other_warnings=json.dumps(sorted(set(str(w.message) for w in caught
            if not issubclass(w.category, NumericalWarning)))))
    for name, bounds in [('heat_time',HEAT_RANGE),('noise',NOISE_RANGE),('amplitude',AMPLITUDE_RANGE)]:
        result[name+'_absolute_error'] = abs(fitted[name]-problem[name])
        result[name+'_relative_error'] = result[name+'_absolute_error']/problem[name]
        result[name+'_near_boundary'] = min(fitted[name]-bounds[0],bounds[1]-fitted[name]) < 0.01*(bounds[1]-bounds[0])
    return result, trajectory, moments


def validate_numerics(output, n=49):
    """Degree sweep against an independent matrix exponential and its derivative."""
    import scipy.linalg
    rows=[]
    for family in ('path','grid','irregular'):
        L, _ = make_laplacian(family,n)
        rhs = torch.randn(n,3,dtype=torch.float64,generator=torch.Generator().manual_seed(91))
        for heat_time in (0.05,0.6,2.0,4.0):
            exact = torch.from_numpy(scipy.linalg.expm(-heat_time*L.toarray()))
            expected = exact@rhs
            derivative = -torch.from_numpy(L.toarray())@expected
            for degree in (4,8,16,32):
                kernel=GraphHeatKernel(L, lambda_max=2, num_coefficients=degree+1,
                    num_nodes=degree+1, heat_time=heat_time)
                actual=kernel(torch.arange(n))@rhs
                gradient,=torch.autograd.grad(actual.square().sum()/2,kernel.raw_heat_time)
                reference=(expected*derivative).sum()*torch.sigmoid(kernel.raw_heat_time.detach())
                poly=DenseChebyshevGraphHeatKernel(L,lambda_max=2,num_coefficients=degree+1,
                    num_nodes=degree+1,heat_time=heat_time)
                row=dict(family=family,heat_time=heat_time,degree=degree,
                    mvp_relative_error=float((actual-expected).norm().detach()/expected.norm()),
                    gradient_absolute_error=float(abs(gradient-reference)),
                    gradient_relative_error=float(abs(gradient-reference)/reference.abs()),
                    spectral_min=float(poly.spectral_values().min().detach()))
                rows.append(row)
                if degree==32 and (row['mvp_relative_error']>1e-10 or row['gradient_relative_error']>1e-8 or row['spectral_min']<=0):
                    raise ArithmeticError(f'numerical prerequisite failed: {row}')
    write_csv(Path(output)/'numerical_validation.csv',rows)
    return rows


def run_study(output, *, smoke=False, steps=180, learn_amplitude=False):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    n=16 if smoke else 49
    families=('path',) if smoke else ('path','grid','irregular')
    seeds=(11,) if smoke else (11,22,33)
    regimes=((0.6,0.03,0.75),) if smoke else ((0.6,0.03,0.75),(2.0,0.12,0.5))
    initials=((0.3,0.04),(2.8,0.2)) if not smoke else ((0.3,0.04),)
    config=dict(n=n,graph_seed=2718,families=families,seeds=seeds,regimes=regimes,
        initials=initials,steps=steps,learning_rate=0.07,optimizer='Adam',learn_amplitude=learn_amplitude,
        heat_range=HEAT_RANGE,noise_variance_range=NOISE_RANGE,amplitude_range=AMPLITUDE_RANGE,
        polynomial=heat_interpolation_certificate(),cg_tolerance=1e-4,max_cg_iterations=500,
        trace_probes='complete orthonormal basis: n_train probes, quadratic storage',
        lanczos_iterations='n_train',max_preconditioner_size=0,prediction='target marginals, ordinary GP conditioning',
        dense_inference='Cholesky',sparse_inference='CG/SLQ',dtype='float64',torch_threads=1)
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (output/'environment.json').write_text(json.dumps(environment_metadata(),indent=2)+'\n')
    validate_numerics(output,n)
    rows,trajectories,landscape=[],[],[]
    representative=None
    for family in families:
        for regime,(heat_time,noise,fraction) in enumerate(regimes):
            for seed in seeds:
                p=sample_problem(family,n=n,seed=seed,heat_time=heat_time,noise=noise,train_fraction=fraction)
                dataset=f'{family}_r{regime}_s{seed}'
                if representative is None:
                    representative=p
                np.savez_compressed(output/(dataset+'.npz'),y=p['y'].numpy(),latent=p['latent'].numpy(),
                    train=p['train'].numpy(),test=p['test'].numpy(),laplacian=p['L'].toarray())
                for init_id,initial in enumerate(initials):
                    for method in METHODS:
                        meta=dict(dataset=dataset,family=family,regime=regime,seed=seed,init_id=init_id,method=method,
                            true_heat_time=heat_time,true_noise=noise,true_amplitude=1.0,train_fraction=fraction,
                            initial_heat_time=initial[0],initial_noise=initial[1],initial_amplitude=1.0,
                            learn_amplitude=learn_amplitude,laplacian_scale=p['scale'])
                        try:
                            metrics,history,moments=fit(p,method,initial=initial,steps=steps,learn_amplitude=learn_amplitude)
                            rows.append(meta|metrics)
                            trajectories.extend(meta|point for point in history)
                            np.savez_compressed(output/f'{dataset}_i{init_id}_{method}_predictions.npz',
                                target_ids=p['test'].numpy(),mean=moments.mean.numpy(),
                                latent_variance=moments.latent_variance.numpy(),
                                observation_variance=moments.observation_variance.numpy())
                            print(f'{dataset} init={init_id} {method}: tau={metrics["heat_time"]:.4f} noise={metrics["noise"]:.4f} nll={metrics["exact_nll"]:.6f}',flush=True)
                        except Exception as error:
                            rows.append(meta|dict(status='failed',error=repr(error)))
                            print(f'FAILED {dataset} {method}: {error}',flush=True)
                        write_csv(output/'fits.csv',rows)
                        write_csv(output/'trajectories.csv',trajectories)
    # A two-parameter exact likelihood slice at true amplitude=1. For free
    # amplitude, fitted points are projections and their NLL differs from slice.
    for tau in np.linspace(0.055,3.995,65):
        for noise in np.geomspace(0.0051,0.499,60):
            landscape.append(dict(heat_time=tau,noise=noise,nll=exact_nll(representative,tau,noise)))
    write_csv(output/'landscape.csv',landscape)
    summarize(output)
    plot_results(output)
    if any(row['status']!='ok' for row in rows):
        raise RuntimeError('Some fits failed; inspect fits.csv')
    return rows


def read_rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def summarize(output):
    """Use best exact NLL initialization per dataset; do not count starts as draws."""
    output=Path(output)
    rows=[r for r in read_rows(output/'fits.csv') if r['status']=='ok']
    best={}
    for row in rows:
        key=(row['dataset'],row['method'])
        if key not in best or float(row['exact_nll'])<float(best[key]['exact_nll']):
            best[key]=row
    summary=[]
    for family,regime,method in sorted(set((r['family'],r['regime'],r['method']) for r in best.values())):
        group=[r for r in best.values() if (r['family'],r['regime'],r['method'])==(family,regime,method)]
        record=dict(family=family,regime=regime,method=method,datasets=len(group))
        for key in ('heat_time','noise','amplitude','heat_time_relative_error','noise_relative_error','rmse','marginal_nlpd','coverage95'):
            values=[float(r[key]) for r in group]
            for stat, value in [('median',np.median(values)),('min',min(values)),('max',max(values))]:
                record[f'{key}_{stat}']=value
        summary.append(record)
    write_csv(output/'summary.csv',summary)
    return summary


def plot_results(output):
    """Regenerate every figure from CSV only, without fitting or sampling."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output)
    fits=[r for r in read_rows(output/'fits.csv') if r['status']=='ok']
    colors=dict(A='#2563eb',B='#e69f00',C='#009e73')
    best={}
    for row in fits:
        key=(row['dataset'],row['method'])
        if key not in best or float(row['exact_nll'])<float(best[key]['exact_nll']): best[key]=row
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,param,label in zip(axes,('heat_time','noise'),('Heat time','Noise variance')):
        for j,method in enumerate(METHODS):
            group=[r for r in best.values() if r['method']==method]
            x=np.array([float(r['true_'+param]) for r in group]); y=[float(r[param]) for r in group]
            ax.scatter(x*(1+(j-1)*0.025),y,label=method,color=colors[method],alpha=.65,s=25)
        bounds=HEAT_RANGE if param=='heat_time' else NOISE_RANGE
        ax.plot(bounds,bounds,'k--',lw=.8); ax.set(xlabel='True '+label.lower(),ylabel='Fitted '+label.lower())
        ax.legend(title='Method')
    fig.suptitle('Independent GP draws; best of two starts by exact NLL (x offset for visibility)')
    fig.tight_layout(); fig.savefig(output/'heat_parameter_recovery.png',dpi=160); plt.close(fig)
    representative=fits[0]['dataset']
    data=[r for r in read_rows(output/'trajectories.csv') if r['dataset']==representative and r['init_id']=='0']
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for method in METHODS:
        group=[r for r in data if r['method']==method]
        for ax,key in zip(axes,('loss','heat_time','noise')):
            ax.plot([int(r['step']) for r in group],[float(r[key]) for r in group],label=method,color=colors[method],alpha=.8)
            ax.set(xlabel='Adam step',ylabel=key)
    axes[0].legend(); fig.suptitle(representative); fig.tight_layout()
    fig.savefig(output/'heat_training_curves.png',dpi=160); plt.close(fig)
    rows=read_rows(output/'landscape.csv')
    x=np.unique([float(r['heat_time']) for r in rows]); y=np.unique([float(r['noise']) for r in rows])
    z=np.array([float(r['nll']) for r in rows]).reshape(len(x),len(y)).T
    fig,ax=plt.subplots(figsize=(7,5))
    # Show the shape near the optimum; retain the full unmodified NLL in CSV.
    delta=z-z.min()
    cs=ax.contourf(x,y,delta,levels=np.linspace(0,.35,29),extend='max',cmap='viridis')
    fig.colorbar(cs,ax=ax,label='Δ exact NLL / observation (colors capped at 0.35)')
    ax.contour(x,y,delta,levels=[.005,.01,.02,.05,.1],colors='white',linewidths=.5,alpha=.6)
    ax.scatter(float(fits[0]['true_heat_time']),float(fits[0]['true_noise']),marker='*',s=160,c='white',edgecolors='black',label='Truth')
    for method,marker in zip(METHODS,('o','s','x')):
        group=[r for r in fits if r['dataset']==representative and r['method']==method]
        ax.scatter([float(r['heat_time']) for r in group],[float(r['noise']) for r in group],marker=marker,color=colors[method],label=method)
    ax.set(xlabel='Heat time',ylabel='Noise variance',yscale='log',title='Exact NLL slice at amplitude=1; fitted points projected')
    ax.legend(); fig.tight_layout(); fig.savefig(output/'heat_loss_landscape.png',dpi=160); plt.close(fig)
