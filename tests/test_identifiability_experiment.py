"""Independent sampling-unit and failure-accounting checks for report summaries."""
import importlib.util
from pathlib import Path
import unittest


_path = Path(__file__).parents[1]/"experiments/02_heat_identifiability.py"
_spec = importlib.util.spec_from_file_location("heat_identifiability_experiment", _path)
experiment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment)


class IdentifiabilitySummaryTests(unittest.TestCase):
    def row(self, pool, replicates, tau, *, valid=True, mode="joint"):
        return dict(family="path", n=49, regime=0, observation_design="partial",
            observed_fraction=36/49, n_observed=36, replicates=replicates,
            fit_mode=mode, pool_seed=pool, true_tau=0.6, true_noise_variance=0.03,
            fit_tau=tau, fit_noise_variance=0.04, valid=valid,
            optimizer_success=False, stationary=False, boundary_flags='["noise_lower"]',
            tau_near_boundary=False, noise_near_boundary=True, start_disagreement=False,
            tau_relative_error=abs(tau-0.6)/0.6, noise_relative_error=1/3 if mode=="joint" else "")

    def test_summary_keeps_valid_nonconvergence_and_counts_failures(self):
        rows = [self.row(0, 1, 0.3), self.row(1, 1, 0.9), self.row(2, 1, 100, valid=False)]
        summary, = experiment.summary_rows(rows)
        self.assertEqual((summary['pools_attempted'], summary['pools_valid'], summary['pools_failed']), (3, 2, 1))
        self.assertAlmostEqual(summary['tau_estimate_median'], 0.6)
        self.assertAlmostEqual(summary['tau_relative_error_median'], 0.5)
        self.assertEqual(summary['optimizer_success_count'], 0)
        self.assertEqual(summary['boundary_count'], 2)
        self.assertEqual(summary['noise_boundary_count'], 2)
        known, = experiment.summary_rows([self.row(0, 1, 0.5, mode='noise_known')])
        self.assertFalse(any(key.startswith('noise_relative_error') for key in known))
        self.assertNotIn('noise_rmse', known)

    def test_paired_comparisons_match_pools_and_preserve_worsening(self):
        rows = [self.row(0, 1, 0.3), self.row(0, 5, 0.6),
                self.row(1, 1, 0.6), self.row(1, 5, 1.2)]
        paired, = experiment.paired_rows(rows, [1, 5])
        self.assertEqual(paired['paired_pools'], 2)
        self.assertEqual(paired['tau_improved_fraction'], 0.5)
        self.assertAlmostEqual(paired['tau_error_delta_median'], 0.25)

    def test_canonical_design_has_correct_number_of_pooled_fits(self):
        for config, expected in [(experiment.configuration(), 1440),
                                 (experiment.configuration(secondary=True), 160),
                                 (experiment.configuration(smoke=True), 16)]:
            count = (len(config['families'])*len(config['regimes'])*
                len(config['observation_designs'])*len(config['replicates'])*
                len(config['pool_seeds'])*len(config['fit_modes']))
            self.assertEqual(count, expected)
        self.assertEqual(experiment.configuration()['representative'], dict(
            family='grid', n=49, regime=0, observation_design='partial', pool_seed=0, replicates=1))
        self.assertEqual(experiment.configuration(graphs=['path', 'path'])['families'], ['path'])


if __name__ == '__main__':
    unittest.main()
