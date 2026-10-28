"""The single gated CSR candidate: sparse preprocessing, state and derivatives."""
import io
import unittest
import numpy as np
import scipy.sparse as sp
import torch
from graphgp.kernels.heat import GraphHeatKernel
from graphgp.kernels.graph_time import GraphTimeHeatRBFKernel
from graphgp.operators.chebyshev import scipy_csr_to_torch_csr
from graphgp.utils.graphs import make_laplacian
from graphgp.benchmarking.worker import guarded_sparse
from graphgp.benchmarking.stage05_cases import cases,make_problem,profile,probe_seed,ARMS
from graphgp.benchmarking.stage05_operations import inference_context,mll_measurement
from graphgp.models.graph_time import make_graph_time_gp
from graphgp.inference.predict import predict_marginals


class Stage05Scaled(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.previous)

    def test_sparse_preprocessing_representation_batch_gradients_and_roundtrip(self):
        L,_=make_laplacian('grid',16)
        duplicate=sp.coo_matrix((np.r_[L.data/2,L.data/2],(np.tile(L.tocoo().row,2),np.tile(L.tocoo().col,2))),shape=L.shape)
        for matrix in (L,duplicate,sp.csr_matrix((16,16),dtype=np.float64),sp.diags([0.,.1,.8,1.],format='csr')):
            for width in (1,8,32,128):
                with guarded_sparse():
                    baseline=GraphHeatKernel(matrix,lambda_max=2,heat_time=torch.tensor(1.2,dtype=torch.float64)).double()
                    candidate=GraphHeatKernel(matrix,lambda_max=2,heat_time=torch.tensor(1.2,dtype=torch.float64),scaled_laplacian=True).double()
                    n=matrix.shape[0];rhs=torch.linspace(-1,1,2*n*width,dtype=torch.float64).reshape(2,n,width)
                    a=baseline.full_graph_operator();b=candidate.full_graph_operator()
                    torch.testing.assert_close(a@rhs,b@rhs,rtol=1e-11,atol=1e-11)
                    ga=torch.autograd.grad((a@rhs).square().sum(),baseline.raw_heat_time)[0]
                    gb=torch.autograd.grad((b@rhs).square().sum(),candidate.raw_heat_time)[0]
                    torch.testing.assert_close(ga,gb,rtol=1e-8,atol=1e-9)
                    coefficients=b.cheb_coeffs.detach().requires_grad_();rebuilt=b.representation_tree()(b.L,coefficients)
                    reference=a.representation_tree()(a.L,coefficients)
                    ca=torch.autograd.grad((reference@rhs).sum(),coefficients)[0]
                    cb=torch.autograd.grad((rebuilt@rhs).sum(),coefficients)[0]
                    torch.testing.assert_close(ca,cb,rtol=1e-8,atol=1e-9)
                    torch.testing.assert_close(b@rhs[0,:,0],(b@rhs[0])[:,0],rtol=1e-11,atol=1e-11)
                    state=io.BytesIO();torch.save(candidate.state_dict(),state);state.seek(0)
                    # Same graph/representation gives sparse buffers compatible nnz.
                    fresh=GraphHeatKernel(matrix,lambda_max=2,scaled_laplacian=True).double()
                    fresh.load_state_dict(torch.load(state,weights_only=True));fresh=fresh.to(dtype=torch.float64)
                    torch.testing.assert_close(b@rhs,fresh.full_graph_operator()@rhs,rtol=0,atol=0)
                    loaded_gradient=torch.autograd.grad((fresh.full_graph_operator()@rhs).square().sum(),fresh.raw_heat_time)[0]
                    torch.testing.assert_close(ga,loaded_gradient,rtol=1e-8,atol=1e-9)
                    torch.testing.assert_close(b@rhs,b.to(device='cpu',dtype=torch.float64)@rhs,rtol=0,atol=0)
                    self.assertTrue(candidate.L.is_sparse_csr)
                    self.assertEqual(candidate.L.values().numel(),candidate.L.col_indices().numel())
        raw=scipy_csr_to_torch_csr(L,dtype=torch.float64)
        fallback=GraphHeatKernel(raw,lambda_max=2,scaled_laplacian=True)
        self.assertFalse(fallback.scaled_laplacian)

    def test_complete_tiny_gp_three_updates_and_learned_noise(self):
        case=cases()[0];p=make_problem(case);x=torch.from_numpy(p['train_pairs']);y=torch.from_numpy(p['train_y']);target=torch.from_numpy(p['target_pairs'])
        for degree,arm in [(32,ARMS[2]),(16,ARMS[3])]:
            results=[]
            for candidate in (False,True):
                model=make_graph_time_gp(p['L'],x,y,num_times=8,backend='sparse_chebyshev',heat_time=1.2,lengthscale=.24,degree=degree,scaled_laplacian=candidate,learn_noise=True)
                model.train();model.likelihood.train()
                with inference_context(arm,profile('balanced',case.m)):
                    r=mll_measurement(model,x,y,probe_seed(case,301),updates=3)
                    moments=predict_marginals(model,target)
                results.append((r,moments))
            for a,b in zip(results[0][0]['trajectory'],results[1][0]['trajectory']):
                self.assertLess(abs(a['nll']-b['nll']),2e-4)
                for key in a['raw_gradients']:self.assertLess(abs(a['raw_gradients'][key]-b['raw_gradients'][key]),2e-3)
            for name in ('mean','latent_variance','observation_variance'):
                torch.testing.assert_close(getattr(results[0][1],name),getattr(results[1][1],name),rtol=0,atol=2e-4)
            self.assertTrue(model.covar_module.graph_kernel.scaled_laplacian)


if __name__=='__main__':unittest.main()
