"""Exact bounded entry batching, its derivatives, and actual posterior dispatch."""
import io
import math
import unittest
from unittest.mock import patch

import gpytorch
import numpy as np
import scipy.sparse
import torch

from graphgp.analysis.graph_time_oracle import dense_prior, posterior_marginals
from graphgp.inference.instrumentation import instrument_operator
from graphgp.inference.predict import predict_marginals
from graphgp.models.graph_time import make_graph_time_gp
from graphgp.operators.chebyshev import ChebyshevFunctionLinearOperator
from graphgp.utils.graphs import make_laplacian
from graphgp.validation_graph_time import graph_time_inference, sparse_graph_time_guards


def scalar_reference(operator, rows, cols):
    rows, cols = torch.broadcast_tensors(rows, cols)
    values = []
    for row, col in zip(rows.reshape(-1), cols.reshape(-1)):
        vector = operator.cheb_coeffs.new_zeros(operator.n, 1)
        vector[col, 0] = 1
        values.append((operator @ vector)[row, 0])
    return torch.stack(values).reshape(rows.shape) if values else operator.cheb_coeffs.new_empty(rows.shape)


def dense_polynomial(L, coefficients):
    matrix = torch.from_numpy(L.toarray())-torch.eye(L.shape[0], dtype=torch.float64)
    previous = torch.eye(L.shape[0], dtype=torch.float64)
    result = .5*coefficients[0]*previous
    if len(coefficients)>1:
        current = matrix
        result = result+coefficients[1]*current
        for k in range(2,len(coefficients)):
            following = 2*matrix@current-previous
            result = result+coefficients[k]*following
            previous,current = current,following
    return result


class Stage05Entries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def operator(self,n=16):
        L,_=make_laplacian('grid',n)
        c=torch.tensor([1.4,-.3,.06,-.004,.0002],dtype=torch.float64,requires_grad=True)
        return L,c,ChebyshevFunctionLinearOperator(L,c,lambda_min=0,lambda_max=2)

    def test_broadcast_unsorted_duplicates_values_and_coefficient_gradients(self):
        L,c,op=self.operator()
        row=torch.tensor([[13],[2],[13]],dtype=torch.int64)
        col=torch.tensor([[7,2,7,0]],dtype=torch.int64)
        expected=dense_polynomial(L,c)[row,col]
        original=scalar_reference(op,row,col)
        with sparse_graph_time_guards():
            actual=op._get_indices(row,col)
        torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-11)
        torch.testing.assert_close(actual,original,rtol=1e-11,atol=1e-11)
        weights=torch.arange(1,13,dtype=torch.float64).reshape(3,4)
        gradients=[torch.autograd.grad((v*weights).sum(),c,retain_graph=True)[0] for v in [actual,expected,original]]
        for gradient in gradients[1:]:
            torch.testing.assert_close(gradients[0],gradient,rtol=1e-8,atol=1e-9)
        for k in range(len(c)):
            step=1e-6
            perturbed=c.detach().clone();perturbed[k]+=step
            other=c.detach().clone();other[k]-=step
            plus=(dense_polynomial(L,perturbed)[row,col]*weights).sum()
            minus=(dense_polynomial(L,other)[row,col]*weights).sum()
            self.assertAlmostEqual(gradients[0][k].item(),((plus-minus)/(2*step)).item(),places=8)

    def test_entry_call_bounds_and_basis_workspace(self):
        _,_,op=self.operator(400)
        for unique_count in [50,256]:
            rows=(torch.arange(256)*17)%400;cols=torch.arange(256)%unique_count
            widths=[];original=ChebyshevFunctionLinearOperator._matmul
            def mm(operator,rhs):
                widths.append(tuple(rhs.shape));return original(operator,rhs)
            with sparse_graph_time_guards(),torch.no_grad(),patch.object(ChebyshevFunctionLinearOperator,'_matmul',mm):
                result=op._get_indices(rows,cols)
            self.assertEqual(result.shape,(256,))
            self.assertLessEqual(len(widths),math.ceil(unique_count/16))
            self.assertTrue(all(n==400 and 1<=width<=16 for n,width in widths))
            self.assertLessEqual(max(n*width for n,width in widths),400*16)

    def test_empty_validation_scalar_and_representation_reconstruction(self):
        _,c,op=self.operator()
        empty=op._get_indices(torch.empty(0,3,dtype=torch.long),torch.empty(1,3,dtype=torch.long))
        self.assertEqual(empty.shape,(0,3));empty.sum().backward();torch.testing.assert_close(c.grad,torch.zeros_like(c))
        for row,col,error in [(torch.tensor([-1]),torch.tensor([0]),IndexError),(torch.tensor([0]),torch.tensor([16]),IndexError),
                              (torch.tensor([.5]),torch.tensor([0]),TypeError)]:
            with self.assertRaises(error):op._get_indices(row,col)
        with self.assertRaises(ValueError):op._get_indices(torch.tensor([0]),torch.tensor([0]),torch.tensor([0]))
        reconstructed=op.representation_tree()(*op.representation())
        rhs=torch.randn(2,16,3,dtype=torch.float64)
        torch.testing.assert_close(op@rhs,reconstructed@rhs,rtol=0,atol=0)
        torch.testing.assert_close(op@rhs[0,:,0],(op@rhs[0])[:,0],rtol=1e-11,atol=1e-11)
        one=ChebyshevFunctionLinearOperator(scipy.sparse.csr_matrix((1,1),dtype=np.float64),c,lambda_min=0,lambda_max=2)
        self.assertEqual(one._get_indices(torch.tensor([0,0]),torch.tensor([0,0])).shape,(2,))

    def test_actual_gp_dispatch_cross_diagonal_and_stale_cache_detection(self):
        L,_=make_laplacian('grid',16);rng=np.random.default_rng(5505)
        all_ids=np.arange(128);pairs=np.column_stack((all_ids//8,all_ids%8)).astype(np.float64)
        train=torch.from_numpy(pairs[rng.permutation(128)[:32]])
        y=torch.from_numpy(rng.standard_normal(32))
        target=torch.tensor([[1,2],[1,5],[1,2],[7,3],[14,6]],dtype=torch.float64)
        model=make_graph_time_gp(L,train,y,num_times=8,backend='sparse_chebyshev',heat_time=1.2,lengthscale=.24)
        kernel=model.covar_module
        other=target.flip(0)
        reference=dense_prior(L,np.arange(8)/7,1.2,.24)
        left=(target[:,0]*8+target[:,1]).long().numpy();right=(other[:,0]*8+other[:,1]).long().numpy()
        np.testing.assert_allclose(kernel(target,other,diag=True).detach().numpy(),reference[left,right],rtol=1e-11,atol=1e-11)
        with graph_time_inference('sparse_chebyshev',32,cg_tolerance=1e-4,exhaustive=False):
            with instrument_operator(record_widths=True) as stats:
                initial=predict_marginals(model,target,chunk_size=5)
            self.assertEqual(stats['get_indices_calls'],1)
            self.assertEqual(stats['get_indices_entries'],5)
            self.assertIn('3',stats['rhs_width_counts'])
            with instrument_operator() as mean_stats:
                only=predict_marginals(model,target,mean_only=True,chunk_size=2)
            self.assertEqual(mean_stats['get_indices_calls'],0)
            torch.testing.assert_close(initial.mean,only.mean,rtol=1e-10,atol=1e-10)
            kernel.graph_kernel.heat_time=torch.tensor(2.1,dtype=torch.float64)
            updated=predict_marginals(model,target,chunk_size=2)
        independent=posterior_marginals(dense_prior(L,np.arange(8)/7,2.1,.24),train.numpy(),y.numpy(),target.numpy(),.06,num_times=8)
        np.testing.assert_allclose(updated.mean.numpy(),independent['mean'],rtol=0,atol=2e-4)
        self.assertGreater((updated.mean-initial.mean).abs().max().item(),1e-3)
        model.set_train_data(inputs=train.flip(0),targets=-y.flip(0),strict=False)
        saved=io.BytesIO();torch.save(model.state_dict(),saved);saved.seek(0)
        fresh=make_graph_time_gp(L,train.flip(0),-y.flip(0),num_times=8,backend='sparse_chebyshev')
        fresh.load_state_dict(torch.load(saved,weights_only=True))
        with graph_time_inference('sparse_chebyshev',32,cg_tolerance=1e-4,exhaustive=False):
            a=predict_marginals(model,target);b=predict_marginals(fresh,target)
        torch.testing.assert_close(a.mean,b.mean,rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(a.mean.numpy(),-independent['mean'],rtol=0,atol=2e-4)


if __name__=='__main__':unittest.main()
