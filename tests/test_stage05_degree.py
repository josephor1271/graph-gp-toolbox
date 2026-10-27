"""Degree16 stays opt-in and passes the existing independent oracle gates."""
import unittest
import torch
from graphgp.benchmarking.stage05_validation import degree_domain,tiny_gates


class Stage05Degree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old=torch.get_num_threads();torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.old)

    def test_full_parameter_domain_and_positive_certificate(self):
        result=degree_domain()
        self.assertTrue(result['passed']);self.assertEqual(len(result['domain_rows']),480)
        self.assertGreater(result['certificate']['positivity_margin'],0)

    def test_four_arm_exhaustive_tiny_gradients_noise_and_three_updates(self):
        self.assertTrue(tiny_gates()['passed'])

    def test_degree16_entry_and_product_tau_derivatives_against_expm(self):
        import numpy as np
        import scipy.linalg
        import scipy.sparse
        from graphgp.kernels.graph_time import GraphTimeHeatRBFKernel
        from graphgp.utils.graphs import make_laplacian
        from graphgp.benchmarking.worker import guarded_sparse
        for family in ('path','grid','irregular','disconnected'):
            L,_=make_laplacian('path' if family=='disconnected' else family,9)
            if family=='disconnected':
                L=scipy.sparse.block_diag((L,scipy.sparse.csr_matrix((1,1))),format='csr')
            n=L.shape[0];rhs=torch.from_numpy(np.random.default_rng(502).normal(size=(n,8)))
            rows=torch.tensor([n-1,1,1,4]);cols=torch.tensor([0,3,3,4])
            for tau in (.051,.7,1.2,2.,3.95):
                derivative=-L.toarray()@scipy.linalg.expm(-tau*L.toarray())
                with guarded_sparse():
                    kernel=GraphTimeHeatRBFKernel(L,num_times=8,backend='sparse_chebyshev',heat_time=tau,lengthscale=.24,degree=16)
                    op=kernel.graph_kernel.full_graph_operator();raw=kernel.graph_kernel.raw_heat_time
                    jac=torch.autograd.grad(kernel.heat_time,raw,retain_graph=True)[0].item()
                    product=(op@rhs).sum();entries=op._get_indices(rows,cols).sum()
                    for actual,expected in [(product,float((derivative@rhs.numpy()).sum())),(entries,float(derivative[rows.numpy(),cols.numpy()].sum()))]:
                        measured=torch.autograd.grad(actual,raw,retain_graph=True)[0].item()/jac
                        np.testing.assert_allclose(measured,expected,rtol=1e-8,atol=1e-9)


if __name__=='__main__':unittest.main()
