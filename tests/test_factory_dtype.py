"""Physical scalar values must not round through GPyTorch's default dtype."""
import unittest

import torch

from graphgp.models.factory import make_heat_gp
from graphgp.utils.graphs import make_laplacian


class FactoryDtypeTests(unittest.TestCase):
    def test_heat_factory_preserves_double_noise_and_amplitude(self):
        laplacian, _ = make_laplacian("path", n=9)
        ids = torch.tensor([0, 2, 5])
        y = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64)
        noise, amplitude = 0.123456789012345, 1.23456789012345
        for method in ("A", "B", "C"):
            with self.subTest(method=method):
                model = make_heat_gp(laplacian, ids, y, method,
                                    noise=noise, amplitude=amplitude)
                self.assertEqual(model.likelihood.noise.dtype, torch.float64)
                self.assertEqual(model.covar_module.outputscale.dtype, torch.float64)
                self.assertAlmostEqual(model.likelihood.noise.item(), noise, delta=1e-14)
                self.assertAlmostEqual(model.covar_module.outputscale.item(), amplitude, delta=1e-14)


if __name__ == "__main__":
    unittest.main()
