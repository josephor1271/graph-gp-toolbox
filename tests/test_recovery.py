"""Deterministic numerical prerequisites, not statistical recovery thresholds."""
import unittest
from unittest.mock import patch
import warnings
import gpytorch
import torch
from graphgp.inference.predict import predict_marginals
from graphgp.kernels.safety import heat_interpolation_certificate
from graphgp.kernels.dense import DenseChebyshevGraphHeatKernel
from graphgp.models.factory import make_heat_gp
from graphgp.recovery import sample_problem, exact_nll, fit, parameters
from graphgp.validation import inference_context

torch.set_num_threads(1)


class RecoveryTests(unittest.TestCase):
    def test_certificate_and_range_validation(self):
        cert=heat_interpolation_certificate()
        self.assertGreater(cert['positivity_margin'],0.0003)
        with self.assertRaises(ValueError):
            heat_interpolation_certificate(degree=4)
        p=sample_problem('path',n=16)
        with self.assertRaises(ValueError):
            make_heat_gp(2*p['L'],p['train'],p['y'][p['train']])
        with self.assertRaises(ValueError):
            make_heat_gp(p['L'],p['train'],p['y'][p['train']],heat_time=5)
        # Whole-domain analytic certificate above; floating-point sampled checks
        # below are additional diagnostics, not the reason we assert PSD.
        for family in ('path','grid','irregular'):
            p=sample_problem(family,n=16)
            for tau in torch.linspace(.05,4,15).tolist():
                kernel=DenseChebyshevGraphHeatKernel(p['L'],lambda_max=2,
                    num_coefficients=33,num_nodes=33,heat_time=tau)
                values=kernel.spectral_values()
                self.assertGreater(values.min().item(),cert['positivity_margin'])
                torch.testing.assert_close(values,torch.exp(-tau*kernel.evals),rtol=1e-10,atol=1e-13)

    def test_bounded_five_path_semantics(self):
        for family in ('path','grid','irregular'):
            p=sample_problem(family,n=16)
            results={}
            for method in ('A','A2','B','B2','C'):
                model=make_heat_gp(p['L'],p['train'],p['y'][p['train']],method,learn_amplitude=True)
                mll=gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood,model)
                with inference_context(method,len(p['train'])):
                    loss=-mll(model(p['train']),p['y'][p['train']]); loss.backward()
                    grads=torch.cat([v.grad.flatten() for v in model.parameters() if v.requires_grad])
                    moments=predict_marginals(model,p['test'])
                results[method]=(loss.detach(),grads,moments.mean,moments.latent_variance)
                self.assertAlmostEqual(loss.item(),exact_nll(p,**parameters(model)),places=7)
            for method in ('A2','B','B2','C'):
                for actual,expected in zip(results[method],results['A']):
                    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
            for actual,expected in zip(results['C'],results['B2']):
                torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-11)

    def test_mean_only_and_repeated_requested_targets(self):
        p=sample_problem('grid',n=16)
        targets=p['test'][torch.tensor([2,0,2,1])]
        for method in ('A','C'):
            model=make_heat_gp(p['L'],p['train'],p['y'][p['train']],method)
            with inference_context(method,len(p['train'])):
                full=predict_marginals(model,targets,chunk_size=2)
                mean=predict_marginals(model,targets,mean_only=True,chunk_size=3)
            self.assertIsNone(mean.latent_variance)
            self.assertIsNone(mean.observation_variance)
            torch.testing.assert_close(mean.mean,full.mean,rtol=1e-4,atol=1e-5)
            torch.testing.assert_close(full.mean[0],full.mean[2])
            torch.testing.assert_close(full.observation_variance-full.latent_variance,
                model.likelihood.noise.detach().expand_as(full.latent_variance))

    def test_synthetic_training_smoke(self):
        p=sample_problem('path',n=16)
        dense,_,_=fit(p,'A',steps=4)
        sparse,history,_=fit(p,'C',steps=4)
        self.assertLess(history[-1]['loss'],history[0]['loss'])
        self.assertAlmostEqual(dense['heat_time'],sparse['heat_time'],places=5)
        self.assertLess(sparse['posterior_variance_max_error'],1e-5)
        self.assertEqual(sparse['numerical_warnings'],'[]')

    def test_fit_trains_model_returned_in_eval_mode(self):
        p=sample_problem('path',n=16)
        expected,expected_history,expected_moments=fit(p,'A',steps=3)

        def eval_factory(*args,**kwargs):
            model=make_heat_gp(*args,**kwargs)
            model.eval()
            model.likelihood.eval()
            return model

        with patch('graphgp.recovery.make_heat_gp',side_effect=eval_factory):
            actual,actual_history,actual_moments=fit(p,'A',steps=3)
        for name in ('heat_time','noise','own_nll','exact_nll'):
            self.assertAlmostEqual(actual[name],expected[name],places=12)
        torch.testing.assert_close(
            torch.tensor([point['loss'] for point in actual_history],dtype=torch.float64),
            torch.tensor([point['loss'] for point in expected_history],dtype=torch.float64),
            rtol=0,atol=0)
        torch.testing.assert_close(actual_moments.mean,expected_moments.mean)
        torch.testing.assert_close(actual_moments.latent_variance,expected_moments.latent_variance)


if __name__=='__main__':
    unittest.main()
