"""Small deterministic checks separating the three approximation sources.

Ordinary dense kernels supply the GP references. Exhaustive trace probes remove
sampling variance, while CG/Lanczos remain active in A2, B2, and C. The inference
context rejects Cholesky for all iterative variants and full-base densification
for C. Benchmarks deliberately do not run as part of these regression tests.
"""

import unittest

import gpytorch
import torch

from graphgp.validation import (
    VARIANTS,
    graph_problem,
    inference_context,
    instrument_operator,
    make_model,
    numerical_warnings,
)


def _evaluate_variant(problem, variant):
    """Exercise ordinary ExactGP directly, independent of the result harness."""
    model, likelihood = make_model(problem, variant, degree=32, heat_time=1.1, noise=0.12)
    train, test = problem["train_ids"], problem["test_ids"]
    target = problem["targets"][train]
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    result = {"phases": {}}
    with numerical_warnings() as caught, inference_context(variant, train.numel()):
        for name, left, right in [("train_covariance", train, train),
                                   ("cross_covariance", test, train),
                                   ("test_covariance", test, test)]:
            result[name] = model.covar_module(left, right).to_dense().detach()
        with instrument_operator() as forward_stats:
            loss = -mll(model(train), target)
        with instrument_operator() as backward_stats:
            loss.backward()
        result["loss"] = loss.detach()
        result["heat_gradient"] = model.covar_module.raw_heat_time.grad.detach().clone()
        result["noise_gradient"] = likelihood.raw_noise.grad.detach().clone()
        model.eval()
        likelihood.eval()
        with torch.no_grad(), instrument_operator() as prediction_stats:
            prediction = likelihood(model(test))
            result["mean"] = prediction.mean.detach()
            result["variance"] = prediction.variance.detach()
        result["phases"] = dict(forward=forward_stats, backward=backward_stats,
                                prediction=prediction_stats)
    result["warnings"] = caught
    return result


class GraphGPValidationTests(unittest.TestCase):
    def _assert_graph_equivalence(self, family):
        problem = graph_problem(family, n=16, seed=123)
        results = {variant: _evaluate_variant(problem, variant) for variant in VARIANTS}
        covariance_fields = ("train_covariance", "cross_covariance", "test_covariance")
        inference_fields = ("loss", "heat_gradient", "noise_gradient", "mean", "variance")
        for variant, result in results.items():
            with self.subTest(graph=family, variant=variant):
                self.assertEqual(result["warnings"], [])
                for field in covariance_fields + inference_fields:
                    self.assertTrue(torch.isfinite(result[field]).all(), field)
                self.assertTrue((result["variance"] > 0).all())
                for phase in ("forward", "backward", "prediction"):
                    calls = result["phases"][phase]["operator_matmul_calls"]
                    if variant == "C":
                        self.assertGreater(calls, 0, phase)
                        self.assertEqual(result["phases"][phase]["estimated_sparse_l_mvps"],
                                         calls * 32)
                    else:
                        self.assertEqual(calls, 0, phase)

        # A/A2 isolates CG/SLQ, A/B isolates polynomial truncation, B/B2
        # isolates CG/SLQ for that polynomial, and B2/C isolates representation.
        # A/C additionally verifies the total error against the exact dense GP.
        for reference, actual in [("A", "A2"), ("A", "B"), ("B", "B2"),
                                   ("B2", "C"), ("A", "C")]:
            with self.subTest(graph=family, comparison=f"{reference}/{actual}"):
                for field in covariance_fields:
                    torch.testing.assert_close(results[actual][field], results[reference][field],
                                               rtol=1e-10, atol=2e-12, msg=field)
                for field in inference_fields:
                    if (reference, actual) == ("A", "B"):
                        # Degree 32 makes approximation negligible at this fixed
                        # heat time even for the irregular graph's wider bound.
                        rtol, atol = 1e-9, 2e-11
                    else:
                        # Matches the configured 1e-4 relative CG stopping
                        # criterion; the absolute tolerance covers near-zero
                        # means/gradients. No stochastic SLQ error is present.
                        rtol, atol = 1e-4, 1e-5
                    torch.testing.assert_close(results[actual][field], results[reference][field],
                                               rtol=rtol, atol=atol, msg=field)

    def test_path_five_variant_equivalence(self):
        self._assert_graph_equivalence("path")

    def test_grid_five_variant_equivalence(self):
        self._assert_graph_equivalence("grid")

    def test_irregular_five_variant_equivalence(self):
        self._assert_graph_equivalence("irregular")

    def test_disconnected_five_variant_equivalence(self):
        self._assert_graph_equivalence("disconnected")


if __name__ == "__main__":
    unittest.main()
