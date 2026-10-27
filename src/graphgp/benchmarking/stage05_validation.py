"""Independent pre-benchmark degree/domain and tiny GP admission gates."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import warnings


def degree_domain():
    import numpy as np,scipy.linalg,scipy.sparse,torch
    from graphgp.kernels.graph_time import GraphTimeHeatRBFKernel
    from graphgp.kernels.safety import heat_interpolation_certificate
    from graphgp.kernels.dense import DenseChebyshevGraphHeatKernel
    from graphgp.utils.graphs import make_laplacian
    from graphgp.benchmarking.worker import guarded_sparse
    certificate=heat_interpolation_certificate(16,max_heat_time=4,lambda_max=2)
    assert certificate['positivity_margin']>0 and certificate['num_nodes']==17
    rows=[]
    for family in ('path','grid','irregular','disconnected'):
        for n in (9,16,49,100):
            if family=='disconnected':
                base,_=make_laplacian('path',n-1)
                L=scipy.sparse.block_diag((base,scipy.sparse.csr_matrix((1,1),dtype=np.float64)),format='csr')
            else:L,_=make_laplacian(family,n)
            for tau in (.051,.7,1.2,2.,3.95):
                graph=scipy.linalg.expm(-tau*L.toarray())
                dense_poly=DenseChebyshevGraphHeatKernel(L,lambda_min=0,lambda_max=2,num_coefficients=17,num_nodes=17,heat_time=torch.tensor(tau,dtype=torch.float64)).double()
                dense_ids=torch.arange(n)[:,None]
                eigen_floor=float(np.linalg.eigvalsh(dense_poly(dense_ids).to_dense().detach().numpy()).min())
                if eigen_floor<=0:raise AssertionError('degree16 covariance is not positive on domain sweep')
                for ell in (.18,.24,.56):
                    for T in (4,8):
                        times=np.arange(T,dtype=np.float64)/(T-1);temporal=np.exp(-(times[:,None]-times[None,:])**2/(2*ell**2))
                        ids=np.arange(n*T);pairs=torch.from_numpy(np.column_stack((ids//T,ids%T)).astype(np.float64))
                        maxima=dict(graph_mvp_abs=0.,full_mvp_abs=0.)
                        rng=np.random.default_rng(5016)
                        with guarded_sparse(),torch.no_grad():
                            kernel=GraphTimeHeatRBFKernel(L,num_times=T,backend='sparse_chebyshev',heat_time=tau,lengthscale=ell,degree=16)
                            assert kernel.graph_kernel.spectral_nodes.numel()==17 and kernel.graph_kernel.cosine_basis.shape==(17,17)
                            for width in (1,8):
                                graph_rhs=rng.standard_normal((n,width));actual=(kernel.graph_kernel.full_graph_operator()@torch.from_numpy(graph_rhs)).numpy()
                                expected=graph@graph_rhs;np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-10)
                                maxima['graph_mvp_abs']=max(maxima['graph_mvp_abs'],float(np.max(np.abs(actual-expected))))
                                rhs=rng.standard_normal((n*T,width));actual=(kernel(pairs)@torch.from_numpy(rhs)).numpy()
                                X=rhs.reshape(n,T,width);expected=np.stack([graph@X[:,:,k]@temporal.T for k in range(width)],axis=-1).reshape(n*T,width)
                                np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-10)
                                maxima['full_mvp_abs']=max(maxima['full_mvp_abs'],float(np.max(np.abs(actual-expected))))
                        rows.append(dict(family=family,n=n,heat_time=tau,lengthscale=ell,num_times=T,degree=16,
                                         polynomial_minimum_eigenvalue=eigen_floor,passed=True,**maxima))
    return dict(certificate=certificate,domain_rows=rows,passed=True)


def tiny_gates():
    import numpy as np,torch
    from graphgp.benchmarking.stage05_cases import cases,ARMS,ARM_CONFIG,make_problem,profile,probe_seed
    from graphgp.benchmarking.stage05_runtime import inference_context,selected_dense_reference
    from graphgp.benchmarking.stage05_operations import mll_measurement
    from graphgp.benchmarking.solver import solver_diagnostics
    from graphgp.inference.instrumentation import instrument_operator
    from graphgp.inference.predict import predict_marginals
    from graphgp.models.graph_time import make_graph_time_gp
    from graphgp.validation_graph_time import _torch_nll_gradients
    case=cases()[0];problem=make_problem(case);problem['num_times']=case.num_times
    reference=selected_dense_reference(problem,case)
    physical_reference=_torch_nll_gradients(problem,1.2,.24,.06)
    train=torch.from_numpy(problem['train_pairs']);y=torch.from_numpy(problem['train_y']);target=torch.from_numpy(problem['target_pairs'])
    outputs={};history={}
    for arm in ARMS:
        mode=ARM_CONFIG[arm]
        model=make_graph_time_gp(problem['L'],train,y,num_times=case.num_times,backend=mode['backend'],heat_time=1.2,lengthscale=.24,noise_variance=.06,degree=mode['degree'],learn_noise=True)
        model.train();model.likelihood.train()
        with inference_context(arm,profile('exhaustive_tiny',case.m)),solver_diagnostics() as solver,instrument_operator(record_widths=True) as stats:
            measured=mll_measurement(model,train,y,probe_seed(case,301))
            moments=predict_marginals(model,target,chunk_size=13)
        error=abs(measured['nll']-reference['nll'])
        if mode['inference']=='direct':assert error/max(abs(reference['nll']),1e-15)<=1e-9
        else:assert error<=1e-5 and solver['cg_calls']>0 and solver['slq_calls']>0
        physical=dict(measured['physical_gradients'])
        noise=model.likelihood.raw_noise;constraint=model.likelihood.noise_covar.raw_noise_constraint
        jac=torch.autograd.grad(constraint.transform(noise).sum(),noise)[0].item()
        physical['noise_variance']=noise.grad.item()/jac
        for name,value in physical.items():np.testing.assert_allclose(value,physical_reference[name],rtol=2e-4,atol=2e-6)
        errors={}
        for name in ('mean','latent_variance','observation_variance'):
            actual=getattr(moments,name).numpy();expected=np.asarray(reference[name]);errors[name]=float(np.max(np.abs(actual-expected)))
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-8 if mode['inference']=='direct' else 2e-4)
        outputs[arm]=dict(nll=measured['nll'],nll_abs_error=error,raw_gradients=measured['raw_gradients'],physical_gradients=physical,
                          posterior_errors=errors,solver=solver,operator=stats,passed=True)
        if arm.startswith('C'):
            fresh=make_graph_time_gp(problem['L'],train,y,num_times=case.num_times,backend=mode['backend'],heat_time=1.2,lengthscale=.24,noise_variance=.06,degree=mode['degree'])
            fresh.train();fresh.likelihood.train()
            with inference_context(arm,profile('balanced',case.m)):
                history[arm]=mll_measurement(fresh,train,y,probe_seed(case,301),updates=3)
                m=predict_marginals(fresh,target)
            history[arm]['mean']=m.mean.tolist();history[arm]['latent_variance']=m.latent_variance.tolist();history[arm]['observation_variance']=m.observation_variance.tolist()
    high,low=history[ARMS[2]],history[ARMS[3]]
    for a,b in zip(high['trajectory'],low['trajectory']):
        assert abs(a['nll']-b['nll'])<=2e-4
        for key in a['raw_gradients']:assert abs(a['raw_gradients'][key]-b['raw_gradients'][key])<=2e-3
    for name in ('mean','latent_variance','observation_variance'):np.testing.assert_allclose(high[name],low[name],rtol=0,atol=2e-4)
    return dict(arms=outputs,three_update_pair=history,passed=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--tiny-only',action='store_true');args=parser.parse_args()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
    import torch
    torch.set_num_threads(1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result=dict(tiny=tiny_gates())
        if not args.tiny_only:result['degree']=degree_domain()
    result['warnings']=[dict(category=x.category.__name__,message=str(x.message)) for x in caught]
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
