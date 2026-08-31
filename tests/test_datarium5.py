import unittest

import numpy as np

from experiments.datarium5_internal_field import (
    Config,
    InternalField,
    grow_body,
    morphology_variant,
    morphology_metrics,
)


class Datarium5LocalMechanicsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(
            n=24,
            agents=12,
            builder_steps=10,
            phase_steps=10,
            body_growth_steps=40,
            develop_steps=60,
            probe_steps=20,
            probe_tail=5,
            probe_trials=8,
        )
        yy, xx = np.mgrid[:24, :24]
        self.phi = 1.0 / (
            1.0 + np.exp((np.hypot(xx - 12, yy - 12) - 6.0) / 0.8)
        )

    def test_local_growth_expands_existing_phase_without_new_seed(self):
        start = float(np.sum(self.phi))
        grown = grow_body(self.phi, self.cfg)
        self.assertGreater(float(np.sum(grown)), start)
        self.assertGreater(float(np.max(grown)), 0.9)

    def test_field_activity_writes_slow_internal_material(self):
        world = InternalField(
            self.phi,
            self.cfg,
            feedback=True,
            plastic=True,
        )
        for _ in range(140):
            world.step(np.ones(4))
        self.assertGreater(float(np.sum(world.fibre)), 0.01)
        self.assertGreater(
            float(np.sum(np.hypot(world.q1, world.q2))),
            0.001,
        )

    def test_erased_morphology_is_exact_zero(self):
        fibre = np.full_like(self.phi, 0.3)
        q1 = 0.2 * fibre
        q2 = -0.1 * fibre
        ef, e1, e2 = morphology_variant(
            fibre, q1, q2, "erased", 0
        )
        self.assertEqual(float(np.max(ef)), 0.0)
        self.assertEqual(float(np.max(np.abs(e1))), 0.0)
        self.assertEqual(float(np.max(np.abs(e2))), 0.0)

    def test_scramble_preserves_complete_local_tensor_histogram(self):
        rng = np.random.default_rng(4)
        fibre = rng.uniform(0.0, 1.0, self.phi.shape)
        q1 = rng.uniform(-0.2, 0.2, self.phi.shape)
        q2 = rng.uniform(-0.2, 0.2, self.phi.shape)
        sf, s1, s2 = morphology_variant(
            fibre, q1, q2, "scrambled", 7
        )
        original = np.stack(
            [fibre.ravel(), q1.ravel(), q2.ravel()],
            axis=1,
        )
        shuffled = np.stack(
            [sf.ravel(), s1.ravel(), s2.ravel()],
            axis=1,
        )
        original = original[np.lexsort(original.T[::-1])]
        shuffled = shuffled[np.lexsort(shuffled.T[::-1])]
        self.assertTrue(np.allclose(original, shuffled))

    def test_isotropic_preserves_fibre_amount_but_removes_direction(self):
        fibre = np.full_like(self.phi, 0.25)
        q1 = np.full_like(self.phi, 0.1)
        q2 = np.full_like(self.phi, -0.07)
        ff, f1, f2 = morphology_variant(
            fibre, q1, q2, "isotropic", 0
        )
        self.assertTrue(np.allclose(ff, fibre))
        self.assertEqual(float(np.max(np.abs(f1))), 0.0)
        self.assertEqual(float(np.max(np.abs(f2))), 0.0)

    def test_morphology_metrics_do_not_call_fractal_dimension(self):
        fibre = np.zeros_like(self.phi)
        fibre[8:16, 11:13] = 0.6
        q1 = 0.2 * fibre
        q2 = np.zeros_like(fibre)
        metrics = morphology_metrics(
            self.phi, fibre, q1, q2, self.cfg
        )
        self.assertIn("fibre_area", metrics)
        self.assertIn("active_spectral_octaves", metrics)
        self.assertNotIn("fractal_dimension", metrics)


if __name__ == "__main__":
    unittest.main()
