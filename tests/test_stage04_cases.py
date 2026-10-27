"""Fixed Stage 04 cases, independent streams, and sparse signal construction."""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm
import scipy.sparse
import torch

from graphgp.analysis.graph_time_oracle import dense_prior
from graphgp.benchmarking.cases import (
    Case, DEGREE, LENGTHSCALE, MASTER_SEED, NOISE_VARIANCE, TAU,
    _approximate_field, dense_eligible, make_problem, primary_cases,
    probe_seed, profiles, smoke_case,
)
from graphgp.operators.chebyshev import ChebyshevFunctionLinearOperator
from graphgp.utils.graphs import make_laplacian


class Stage04CasesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_exact_fixed_table_and_frozen_validation(self):
        cases = primary_cases()
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({case.case_id for case in cases}), 12)
        self.assertEqual([case.numeric_id for case in cases], list(range(1, 13)))
        expected = {
            "N": {(16, 8, 32, 64), (49, 8, 98, 64), (100, 8, 200, 64),
                  (225, 8, 450, 64), (400, 8, 800, 64)},
            "T": {(100, 4, 200, 64), (100, 8, 200, 64),
                  (100, 16, 200, 64), (100, 32, 200, 64)},
            "M": {(100, 8, 80, 64), (100, 8, 200, 64), (100, 8, 400, 64)},
            "Q": {(100, 8, 200, 16), (100, 8, 200, 64), (100, 8, 200, 256)},
        }
        for group, configurations in expected.items():
            self.assertEqual({(case.n, case.num_times, case.m, case.q)
                              for case in cases if group in case.groups}, configurations)
        self.assertEqual(sum(dense_eligible(case) for case in cases), 8)
        for case in cases:
            self.assertEqual(case.nT, case.n * case.num_times)
            self.assertEqual(case.observed_fraction, case.m / case.nT)
            self.assertEqual(case.to_dict()["numeric_id"], case.numeric_id)
            self.assertEqual(json.loads(json.dumps(case.to_dict())), case.to_dict())
        with self.assertRaises(FrozenInstanceError):
            cases[0].m = 1
        valid = dict(case_id=cases[0].case_id, n=16, num_times=8, m=32,
                     q=64, groups=("N",))
        for change in ({"n": True}, {"n": 15}, {"m": 0}, {"m": 32.0},
                       {"q": 200}, {"num_times": 1}, {"groups": ["N"]},
                       {"groups": ("N", "N")}, {"case_id": "arbitrary"},
                       {"m": 31}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                Case(**(valid | change))

    def test_case_import_does_not_import_numerical_or_model_modules(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import graphgp.benchmarking.cases,sys,json; "
             "print(json.dumps({name:name in sys.modules for name in "
             "['torch','numpy','scipy','gpytorch','graphgp.models.factory']}))"],
            check=True, text=True, capture_output=True,
            cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(json.loads(result.stdout), {
            "torch": False, "numpy": False, "scipy": False,
            "gpytorch": False, "graphgp.models.factory": False})

    def test_every_case_has_complete_coordinates_or_explicit_temporal_factor_failure(self):
        for case in primary_cases():
            with self.subTest(case=case.case_id):
                if case.num_times == 32:
                    # The prescribed long-grid factor is numerically singular
                    # in float64 on the measured platform. Check the installed
                    # factorization rather than assuming identical rounding on
                    # all platforms. Preserve any failure without fallback.
                    times = np.arange(32, dtype=np.float64) / 31
                    distance = times[:, None] - times[None, :]
                    temporal = np.exp(-(distance * distance) / (2 * LENGTHSCALE**2))
                    try:
                        np.linalg.cholesky(temporal)
                    except np.linalg.LinAlgError:
                        with self.assertRaises(np.linalg.LinAlgError):
                            make_problem(case)
                        continue
                problem = make_problem(case)
                self.assertEqual(problem["L"].shape, (case.n, case.n))
                self.assertEqual(problem["L"].dtype, np.float64)
                self.assertTrue(scipy.sparse.issparse(problem["L"]))
                self.assertEqual(problem["L"].format, "csr")
                self.assertEqual(problem["graph_scale"], 4.0)
                self.assertEqual(problem["train_pairs"].shape, (case.m, 2))
                self.assertEqual(problem["target_pairs"].shape, (case.q, 2))
                self.assertEqual(problem["train_y"].shape, (case.m,))
                self.assertEqual(problem["latent"].shape, (case.nT,))
                self.assertEqual(problem["solve_rhs"].shape, (case.m, 3))
                self.assertEqual(problem["mvp_rhs"].shape, (case.nT, 8))
                for name in ("train_pairs", "target_pairs", "train_y", "latent",
                             "noise", "y", "solve_rhs", "mvp_rhs"):
                    self.assertEqual(problem[name].dtype, np.float64)
                    self.assertTrue(np.isfinite(problem[name]).all())
                self.assertEqual(len(np.unique(problem["train_ids"])), case.m)
                self.assertEqual(len(np.unique(problem["target_ids"])), case.q)
                self.assertEqual(len(np.intersect1d(problem["train_ids"],
                                                   problem["target_ids"])), 0)
                np.testing.assert_array_equal(problem["time_grid"],
                                              np.arange(case.num_times) / (case.num_times - 1))
                np.testing.assert_array_equal(problem["train_pairs"],
                                              problem["all_pairs"][problem["train_ids"]])
                np.testing.assert_array_equal(problem["target_pairs"],
                                              problem["all_pairs"][problem["target_ids"]])
                np.testing.assert_array_equal(problem["train_y"],
                                              problem["y"][problem["train_ids"]])
                np.testing.assert_array_equal(problem["solve_rhs"][:, 0], problem["train_y"])
                # A random n×T RHS has rank >1: a separable vector would not
                # exercise the prescribed general Kronecker product action.
                self.assertGreater(np.linalg.matrix_rank(
                    problem["mvp_rhs"][:, 0].reshape(case.n, case.num_times)), 1)

    def test_masks_are_prefix_nested_and_same_case_is_backend_paired(self):
        family = [case for case in primary_cases() if (case.n, case.num_times) == (100, 8)]
        by_m = {case.m: make_problem(case) for case in family if "M" in case.groups}
        for m in (80, 200, 400):
            np.testing.assert_array_equal(by_m[m]["train_ids"], by_m[400]["train_ids"][:m])
            np.testing.assert_array_equal(by_m[m]["target_ids"], by_m[400]["target_ids"])
            for field in ("latent", "noise", "y", "mvp_rhs"):
                np.testing.assert_array_equal(by_m[m][field], by_m[400][field])
            np.testing.assert_array_equal(by_m[m]["solve_rhs"], by_m[400]["solve_rhs"][:m])
        by_q = {case.q: make_problem(case) for case in family if "Q" in case.groups}
        for q in (16, 64, 256):
            np.testing.assert_array_equal(by_q[q]["target_ids"], by_q[256]["target_ids"][:q])
            np.testing.assert_array_equal(by_q[q]["train_ids"], by_q[256]["train_ids"])
        first, repeated = make_problem(family[0]), make_problem(family[0])
        self.assertEqual(first["fingerprint"], repeated["fingerprint"])
        for field in ("train_pairs", "target_pairs", "latent", "noise", "y",
                      "solve_rhs", "mvp_rhs"):
            np.testing.assert_array_equal(first[field], repeated[field])
        self.assertNotEqual(first["fingerprint"], by_q[16]["fingerprint"])

    def test_seedsequence_streams_and_smoke_share_canonical_tiny_field(self):
        case = primary_cases()[0]
        problem = make_problem(case)
        latent_rng = np.random.default_rng(np.random.SeedSequence(
            MASTER_SEED, spawn_key=(case.stream_case_id, 0)))
        noise_rng = np.random.default_rng(np.random.SeedSequence(
            MASTER_SEED, spawn_key=(case.stream_case_id, 1)))
        expected_graph = expm(-TAU * problem["L"].toarray() / 2)
        times = problem["time_grid"]
        temporal = np.exp(-0.5 * ((times[:, None] - times[None, :]) / LENGTHSCALE)**2)
        expected_latent = (expected_graph @ latent_rng.standard_normal((case.n, case.num_times))
                           @ np.linalg.cholesky(temporal).T).reshape(-1)
        np.testing.assert_allclose(problem["latent"], expected_latent, rtol=1e-13, atol=1e-14)
        np.testing.assert_array_equal(problem["noise"],
                                      np.sqrt(NOISE_VARIANCE) * noise_rng.standard_normal(case.nT))
        expected_seed = np.random.SeedSequence(
            MASTER_SEED, spawn_key=(case.stream_case_id, 3, 301)).generate_state(1, dtype=np.uint32)[0]
        self.assertEqual(probe_seed(case, 301), int(expected_seed))
        self.assertEqual(len({probe_seed(case, label) for label in (301, 302, 303)}), 3)
        smoke = make_problem(smoke_case())
        for field in ("latent", "noise", "y", "train_ids"):
            np.testing.assert_array_equal(smoke[field], problem[field])
        np.testing.assert_array_equal(smoke["target_ids"], problem["target_ids"][:8])

    def test_sparse_generator_never_calls_dense_graph_or_full_grid_methods(self):
        with (patch.object(ChebyshevFunctionLinearOperator, "to_dense",
                           side_effect=AssertionError("dense graph covariance")),
              patch.object(scipy.sparse.csr_matrix, "toarray",
                           side_effect=AssertionError("dense Laplacian")),
              patch("numpy.kron", side_effect=AssertionError("full covariance")),
              patch("torch.kron", side_effect=AssertionError("full covariance")),
              patch("scipy.linalg.expm", side_effect=AssertionError("dense graph expm")),
              patch("numpy.linalg.eigh", side_effect=AssertionError("graph eig")),
              patch("torch.linalg.eigh", side_effect=AssertionError("graph eig"))):
            problem = make_problem(primary_cases()[-1])
        self.assertEqual(problem["latent"].shape, (800,))

    def test_temporal_factor_failure_propagates_without_jitter_or_retry(self):
        L, _ = make_laplacian("grid", n=4)
        times = np.arange(3, dtype=np.float64) / 2
        distance = times[:, None] - times[None, :]
        expected = np.exp(-(distance * distance) / (2 * LENGTHSCALE**2))
        with patch("numpy.linalg.cholesky", side_effect=np.linalg.LinAlgError("sentinel")) as factor:
            with self.assertRaisesRegex(np.linalg.LinAlgError, "sentinel"):
                _approximate_field(L, times, np.zeros((4, 3), dtype=np.float64))
        factor.assert_called_once()
        np.testing.assert_array_equal(factor.call_args.args[0], expected)

    def test_small_sampling_operator_covariance_matches_independent_exact_reference(self):
        L, _ = make_laplacian("grid", n=4)
        times = np.arange(3, dtype=np.float64) / 2
        # Explicitly construct only a tiny reference sampling map by feeding
        # standard-basis white fields through the production sampling route.
        transform = np.column_stack([
            _approximate_field(L, times, basis.reshape(4, 3))
            for basis in np.eye(12, dtype=np.float64)])
        expected = dense_prior(L, times, TAU, LENGTHSCALE)
        np.testing.assert_allclose(transform @ transform.T, expected, rtol=1e-13, atol=1e-14)
        draws = 5000
        samples = np.random.default_rng(51604).standard_normal((draws, 12)) @ transform.T
        sample_covariance = np.cov(samples, rowvar=False, ddof=1)
        # Gaussian covariance sampling variance, with a six-SE simultaneous
        # check over the 12×12 covariance rather than an arbitrary tolerance.
        sampling_variance = (np.outer(np.diag(expected), np.diag(expected)) + expected**2) / (draws - 1)
        self.assertLess(float(np.max(np.abs(sample_covariance - expected)
                                     / np.sqrt(sampling_variance))), 6)
        self.assertEqual(DEGREE, 32)

    def test_predeclared_profiles_and_tiny_exhaustive_limit(self):
        expected = {
            "fast_probe": (1e-3, 200, 8, 16),
            "balanced_probe": (1e-4, 500, 16, 32),
            "strict_probe": (1e-6, 500, 16, 32),
            "exhaustive_tiny": (1e-8, 500, 32, 32),
        }
        for name, config in profiles(32).items():
            self.assertEqual((config.cg_tolerance, config.max_cg_iterations,
                              config.trace_samples, config.lanczos_iterations), expected[name])
            self.assertEqual(config.eval_cg_tolerance, config.cg_tolerance)
            self.assertEqual(config.max_preconditioner_size, 0)
            self.assertIs(config.terminate_cg_by_size, False)
        self.assertNotIn("exhaustive_tiny", profiles(98))
        for invalid in (0, True, 32.0):
            with self.assertRaises(ValueError):
                profiles(invalid)


if __name__ == "__main__":
    unittest.main()
