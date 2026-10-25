"""Independent graph-time covariance, conditioning and sampling contracts."""

import unittest

import numpy as np
from scipy.linalg import cholesky, expm

from graphgp.analysis.graph_time_oracle import (
    dense_prior, exact_nll, flat_pair_indices, observed_covariance,
    posterior_marginals, sample_problem,
)
from graphgp.utils.graphs import make_laplacian


class GraphTimeOracleTests(unittest.TestCase):
    def setUp(self):
        self.L, _ = make_laplacian("path", n=4)
        self.times = np.arange(3, dtype=np.float64) / 2
        self.K = dense_prior(self.L, self.times, 1.2, 0.24)

    def test_prior_matches_independent_entry_formula_and_vertex_major_action(self):
        for family, n, total_times in (("path", 9, 4), ("grid", 16, 7)):
            L, _ = make_laplacian(family, n=n)
            times = np.arange(total_times, dtype=np.float64) / (total_times - 1)
            for tau, ell in ((0.7, 0.18), (2.1, 0.52)):
                with self.subTest(family=family, tau=tau, ell=ell):
                    K = dense_prior(L, times, tau, ell)
                    KG = expm(-tau * L.toarray())
                    KT = np.exp(-0.5 * ((times[:, None] - times[None, :]) / ell)**2)
                    pairs = np.array([(v, j) for v in range(n)
                                      for j in range(total_times)], dtype=np.float64)
                    ids = flat_pair_indices(pairs, n, total_times)
                    np.testing.assert_array_equal(ids, np.arange(n * total_times))
                    entries = np.array([[KG[int(v), int(w)] * KT[int(j), int(k)]
                                         for w, k in pairs] for v, j in pairs])
                    np.testing.assert_allclose(K, entries, rtol=1e-14, atol=1e-15)
                    np.testing.assert_allclose(K, K.T, rtol=0, atol=1e-14)
                    self.assertGreaterEqual(np.linalg.eigvalsh(K)[0], -1e-12)
                    X = np.random.default_rng(518).normal(size=(n, total_times))
                    np.testing.assert_allclose(K @ X.reshape(-1),
                                               (KG @ X @ KT.T).reshape(-1),
                                               rtol=1e-13, atol=1e-14)

    def test_flat_indices_preserve_unsorted_repeated_pairs_and_validate_inputs(self):
        pairs = np.array([[3, 2], [0, 1], [3, 2], [1, 0]], dtype=np.float64)
        np.testing.assert_array_equal(flat_pair_indices(pairs, 4, 3), [11, 1, 11, 3])
        malformed = [np.empty((0, 2), dtype=np.float64), np.ones((2, 3)),
                     np.ones((1, 2, 2)), np.array([[0, 0]], dtype=np.int64),
                     np.array([[0, 0]], dtype=np.float32), np.array([[0.5, 0]]),
                     np.array([[0, np.nan]]), np.array([[0, np.inf]]),
                     np.array([[-1.0, 0]]), np.array([[4.0, 0]]),
                     np.array([[0, 3.0]]), np.array([[2.0**54, 0]])]
        for invalid in malformed:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    flat_pair_indices(invalid, 4, 3)
        for n, total_times in ((4, 1), (True, 3), (4, 3.0), (2**54, 3)):
            with self.assertRaises(ValueError):
                flat_pair_indices(pairs, n, total_times)

    def test_selected_covariance_adds_noise_after_repeated_selection(self):
        pairs = np.array([[3, 2], [0, 1], [3, 2], [1, 0]], dtype=np.float64)
        ids = np.array([11, 1, 11, 3])
        actual = observed_covariance(self.K, pairs, noise_variance=0.06, num_times=3)
        expected = self.K[np.ix_(ids, ids)] + 0.06 * np.eye(4)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual[0, 2], self.K[11, 11])
        self.assertAlmostEqual(actual[0, 0] - actual[0, 2], 0.06)
        self.assertGreater(np.linalg.eigvalsh(actual)[0], 0)
        with self.assertRaises(ValueError):
            observed_covariance(self.K, pairs, noise_variance=-0.1, num_times=3)
        with self.assertRaises(ValueError):
            observed_covariance(self.K, pairs, noise_variance=0.06, num_times=5)

    def test_posterior_and_normalized_nll_match_direct_gaussian_conditioning(self):
        pairs = np.array([[3, 0], [0, 1], [3, 0], [1, 0]], dtype=np.float64)
        targets = np.array([[2, 2], [3, 2], [2, 2]], dtype=np.float64)
        train_ids, target_ids = np.array([9, 1, 9, 3]), np.array([8, 11, 8])
        y = np.array([0.4, -1.1, 0.7, -0.2], dtype=np.float64)
        noise = 0.06
        train = self.K[np.ix_(train_ids, train_ids)] + noise * np.eye(4)
        cross = self.K[np.ix_(target_ids, train_ids)]
        expected_mean = cross @ np.linalg.solve(train, y)
        expected_cov = (self.K[np.ix_(target_ids, target_ids)]
                        - cross @ np.linalg.solve(train, cross.T))
        actual = posterior_marginals(self.K, pairs, y, targets, noise, num_times=3)
        for field, expected in (("mean", expected_mean),
                                ("latent_var", np.diag(expected_cov)),
                                ("observation_var", np.diag(expected_cov) + noise)):
            np.testing.assert_allclose(actual[field], expected, rtol=1e-13, atol=1e-14)
            self.assertEqual(actual[field].shape, (3,))
            self.assertEqual(actual[field][0], actual[field][2])
        expected_nll = (np.linalg.slogdet(train)[1] + y @ np.linalg.solve(train, y)
                        + len(y) * np.log(2 * np.pi)) / (2 * len(y))
        self.assertAlmostEqual(exact_nll(self.K, pairs, y, noise, num_times=3),
                               expected_nll, places=14)
        for bad_y in (y[:, None], y[:-1], np.array([1, 2, np.nan, 4])):
            with self.assertRaises(ValueError):
                posterior_marginals(self.K, pairs, bad_y, targets, noise, num_times=3)
        # Identical noiseless recordings are singular: the oracle must not add jitter.
        repeated = np.array([[1, 0], [1, 0]], dtype=np.float64)
        with self.assertRaises(np.linalg.LinAlgError):
            posterior_marginals(np.eye(12), repeated, np.ones(2), targets, 0, num_times=3)

    def test_prior_rejects_invalid_grid_laplacian_and_parameters(self):
        for times in (np.arange(3), np.array([0.0, 0.4, 1.0]),
                      np.array([0.0, np.nan, 1.0]), np.array([0.0])):
            with self.assertRaises(ValueError):
                dense_prior(self.L, times, 1.2, 0.24)
        for tau, ell in ((0, 0.24), (1.2, -1), (np.nan, 0.24), (1.2, np.inf)):
            with self.assertRaises(ValueError):
                dense_prior(self.L, self.times, tau, ell)
        with self.assertRaises(ValueError):
            dense_prior(-np.eye(4), self.times, 1.2, 0.24)


class GraphTimeSamplingTests(unittest.TestCase):
    def test_canonical_masks_no_leakage_and_shared_fields_for_paired_tasks(self):
        for seed in (11, 22, 33, 44):
            interpolation = sample_problem(seed, "interpolation")
            forecast = sample_problem(seed, "forecast")
            for key in ("all_pairs", "time_grid", "latent", "y", "noise"):
                np.testing.assert_array_equal(interpolation[key], forecast[key])
            for problem, counts in ((interpolation, (64, 96)), (forecast, (48, 64))):
                self.assertEqual((len(problem["train_ids"]), len(problem["test_ids"])), counts)
                self.assertEqual(len(np.unique(problem["train_ids"])), counts[0])
                self.assertEqual(len(np.intersect1d(problem["train_ids"], problem["test_ids"])), 0)
                np.testing.assert_array_equal(problem["train_pairs"],
                                              problem["all_pairs"][problem["train_ids"]])
                np.testing.assert_array_equal(problem["test_pairs"],
                                              problem["all_pairs"][problem["test_ids"]])
            self.assertTrue(np.all(forecast["train_pairs"][:, 1] < 6))
            self.assertTrue(np.all(forecast["test_pairs"][:, 1] >= 6))
            np.testing.assert_array_equal(np.unique(forecast["test_pairs"][:, 1]),
                                          [6, 7, 8, 9])
            self.assertNotEqual(set(interpolation["train_ids"]), set(forecast["train_ids"]))
        first = sample_problem(11, "interpolation")
        second = sample_problem(22, "interpolation")
        self.assertFalse(np.array_equal(first["latent"], second["latent"]))
        self.assertFalse(np.array_equal(first["noise"], second["noise"]))

    def test_exact_seed_generator_invariant_and_independent_numeric_streams(self):
        problem = sample_problem(11, "interpolation")
        KG = expm(-1.2 * problem["L"].toarray())
        times = np.arange(10, dtype=np.float64) / 9
        distance = times[:, None] - times[None, :]
        KT = np.exp(-(distance * distance) / (2 * 0.24**2))
        latent_rng = np.random.default_rng(np.random.SeedSequence(11, spawn_key=(0,)))
        noise_rng = np.random.default_rng(np.random.SeedSequence(11, spawn_key=(1,)))
        split_rng = np.random.default_rng(np.random.SeedSequence(11, spawn_key=(2, 0)))
        expected_latent = (cholesky(KG, lower=True) @ latent_rng.standard_normal((16, 10))
                           @ cholesky(KT, lower=True).T).reshape(-1)
        expected_noise = np.sqrt(0.06) * noise_rng.standard_normal(160)
        np.testing.assert_array_equal(problem["latent"], expected_latent)
        np.testing.assert_array_equal(problem["noise"], expected_noise)
        np.testing.assert_array_equal(problem["y"], problem["latent"] + expected_noise)
        np.testing.assert_array_equal(problem["train_ids"],
                                      split_rng.choice(np.arange(160), size=64, replace=False))
        repeat = sample_problem(11, "interpolation")
        for field in ("latent", "noise", "y", "train_ids", "test_ids"):
            np.testing.assert_array_equal(problem[field], repeat[field])
        forecast = sample_problem(11, "forecast")
        forecast_rng = np.random.default_rng(np.random.SeedSequence(11, spawn_key=(2, 1)))
        historical = np.array([v * 10 + j for v in range(16) for j in range(6)])
        np.testing.assert_array_equal(forecast["train_ids"],
                                      forecast_rng.choice(historical, size=48, replace=False))

    def test_tiny_sampling_covariance_with_calculated_gaussian_sampling_tolerance(self):
        draws = 1200
        fields = []
        for seed in range(draws):
            problem = sample_problem(seed, "interpolation", graph_kind="path", n=4,
                                     num_times=3, tau=1.1, ell=0.22)
            fields.append(np.concatenate((problem["latent"], problem["noise"])))
        actual = np.cov(np.array(fields), rowvar=False, ddof=1)
        K = dense_prior(problem["L"], problem["time_grid"], 1.1, 0.22)
        expected = np.zeros((24, 24))
        expected[:12, :12] = K
        expected[12:, 12:] = 0.06 * np.eye(12)
        # For Gaussian draws, Var(sample_cov_ij)=(C_ii*C_jj+C_ij**2)/(R-1).
        # Six standard errors per entry allow for the 24×24 simultaneous checks.
        variance = (np.outer(np.diag(expected), np.diag(expected)) + expected**2) / (draws - 1)
        error_in_standard_errors = np.abs(actual - expected) / np.sqrt(variance)
        self.assertLess(float(error_in_standard_errors.max()), 6.0)
        # This also checks independent latent/noise streams, not merely latent covariance.
        self.assertGreater(np.max(np.abs(actual[:12, :12] - np.diag(np.diag(actual[:12, :12])))), 0.1)

    def test_sampler_small_options_and_validation(self):
        problem = sample_problem(7, "forecast", graph_kind="path", n=4, num_times=4,
                                 train_count=3, forecast_start=2)
        self.assertEqual(problem["latent"].shape, (16,))
        self.assertEqual(problem["train_pairs"].shape, (3, 2))
        self.assertEqual(problem["test_pairs"].shape, (8, 2))
        for options in (dict(split="unknown"), dict(split="interpolation", train_count=160),
                        dict(split="forecast", forecast_start=10),
                        dict(split="forecast", train_count=97)):
            with self.assertRaises(ValueError):
                sample_problem(11, **options)


if __name__ == "__main__":
    unittest.main()
