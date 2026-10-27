"""The Stage 04 plot entry point consumes evidence without a GP import."""
import csv
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


PLOT_SCRIPT = Path(__file__).resolve().parents[1] / "experiments/04_plot_scaling.py"


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class TestStage04PlotOnly(unittest.TestCase):
    def test_solver_figure_excludes_primary_case_checks_from_probe_screens(self):
        specification = importlib.util.spec_from_file_location("stage04_plot_fixture", PLOT_SCRIPT)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        row = dict(profile="balanced_probe", screen_status="pass", nll_abs_error=.01,
                   heat_time_raw_gradient_abs_error=.002, lengthscale_raw_gradient_abs_error=.003,
                   posterior_mean_max_abs_error=.0001, latent_variance_max_abs_error=.0001,
                   maximum_relative_residual=.00001)
        rows = [{**row, "screening_scope": "profile_seed"},
                {**row, "screening_scope": "case_check", "screen_status": "fail", "nll_abs_error": 1000}]
        with patch.object(module, "save") as saved:
            module.solver_plot([], rows, Path("unused"))
        figure = saved.call_args.args[0]
        self.assertIn("1/1 screens pass", figure._suptitle.get_text())
        self.assertLess(figure.axes[1].get_ylim()[1], 1)
        module.plt.close(figure)

    def test_fixture_generates_four_plots_without_loading_gp_stack(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / "evidence", Path(temporary) / "plots"
            source.mkdir()
            summary = []
            for n, m, q in ((16, 32, 64), (100, 200, 16), (100, 200, 64), (100, 200, 256), (225, 450, 64)):
                for backend, profile in (("dense_exact", "dense"), ("sparse_chebyshev", "balanced_probe")):
                    if n > 100 and backend == "dense_exact":
                        continue
                    for operation in ("construct", "mvp", "mll", "train3", "mean", "marginal"):
                        failure = n == 225 and operation == "marginal"
                        summary.append(dict(
                            campaign="scaling", case_id=f"n{n}_t8_m{m}_q{q}",
                            n=n, num_times=8, nT=n*8, m=m, q=q, operation=operation,
                            backend=backend, profile=profile, rhs_width=8 if operation == "mvp" else "",
                            attempted=2, success=0 if failure else 2, timeout=2 if failure else 0, failed=0,
                            seconds_median="" if failure else .02 * q,
                            seconds_minimum="" if failure else .018 * q, seconds_maximum="" if failure else .025 * q,
                            rss_peak_mib_median="" if failure else 250 + n,
                        ))
            write_csv(source / "summary.csv", summary)
            checks, accuracy = [], []
            for profile, tolerance in (("fast_probe", .001), ("balanced_probe", .0001), ("strict_probe", .000001)):
                for seed in (301, 302, 303):
                    checks.append(dict(case_id="n16_t8_m32_q64", profile=profile, probe_seed=seed,
                                       requested_tolerance=tolerance, relative_residual=2e-5,
                                       warning_count=int(seed == 302), status="ok"))
                    accuracy.append(dict(case_id="n16_t8_m32_q64", n=16, num_times=8, m=32,
                        profile=profile, probe_seed=seed, nll_abs_error=.03 if seed == 301 else .07,
                        heat_time_raw_gradient_abs_error=.002, lengthscale_raw_gradient_abs_error=.005,
                        posterior_mean_max_abs_error=2e-5, latent_variance_max_abs_error=8e-6,
                        maximum_relative_residual=2e-5, screen_status="pass" if seed == 301 else "fail"))
            write_csv(source / "solver_checks.csv", checks)
            write_csv(source / "accuracy.csv", accuracy)
            code = (
                "import runpy,sys; "
                "sys.argv=[sys.argv[1], '--input',sys.argv[2],'--output',sys.argv[3]]; "
                "runpy.run_path(sys.argv[0],run_name='__main__'); "
                "assert 'torch' not in sys.modules; assert 'gpytorch' not in sys.modules; "
                "assert 'graphgp.benchmarking.worker' not in sys.modules"
            )
            result = subprocess.run([sys.executable, "-c", code, str(PLOT_SCRIPT), str(source), str(output)],
                                    check=True, capture_output=True, text=True)
            names = {f"scaling_stage04_{name}.png" for name in ("runtime", "memory", "solver", "prediction")}
            self.assertEqual({path.name for path in output.iterdir()}, names)
            for name in names:
                data = (output / name).read_bytes()
                self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertGreater(len(data), 10_000)
            self.assertIn("scaling_stage04_runtime.png", result.stdout)


if __name__ == "__main__":
    unittest.main()
