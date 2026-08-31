import unittest

import numpy as np

from experiments.datarium4_body import BodyWorld, Config as BodyConfig
from experiments.datarium4b_builder_body import (
    Config,
    phase_from_scaffold,
    phase_metrics,
    scaffold_variant,
)


class Datarium4BLocalConversionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(
            n=32,
            agents=24,
            builder_steps=10,
            phase_steps=120,
            move_steps=80,
            cargo_steps=80,
            source_radius=8.0,
            source_sigma=3.5,
        )

    def test_erased_scaffold_cannot_spawn_a_fallback_body(self):
        phase = phase_from_scaffold(np.zeros((32, 32)), self.cfg)
        self.assertEqual(float(np.max(phase)), 0.0)
        metrics = phase_metrics(phase, self.cfg.phase_threshold)
        self.assertEqual(metrics["phase_mass"], 0.0)
        self.assertEqual(metrics["component_count"], 0)

    def test_local_written_patch_nucleates_a_second_phase(self):
        scaffold = np.zeros((32, 32))
        scaffold[12:20, 12:20] = 0.12
        phase = phase_from_scaffold(scaffold, self.cfg)
        metrics = phase_metrics(phase, self.cfg.phase_threshold)
        self.assertGreater(metrics["phase_mass"], 20.0)
        self.assertGreater(metrics["largest_component_cells"], 20)
        self.assertLess(metrics["component_count"], 5)

    def test_scramble_preserves_scaffold_histogram_but_not_geometry(self):
        scaffold = np.zeros((32, 32))
        scaffold[8:24, 14:18] = np.linspace(0.03, 0.14, 16)[:, None]
        q1 = 0.5 * scaffold
        q2 = -0.25 * scaffold
        scrambled, sq1, sq2 = scaffold_variant(
            scaffold, q1, q2, "scrambled", 4
        )
        self.assertTrue(
            np.allclose(np.sort(scaffold.ravel()), np.sort(scrambled.ravel()))
        )
        self.assertTrue(
            np.allclose(np.sort(q1.ravel()), np.sort(sq1.ravel()))
        )
        self.assertTrue(
            np.allclose(np.sort(q2.ravel()), np.sort(sq2.ravel()))
        )
        self.assertFalse(np.allclose(scaffold, scrambled))

    def test_mean_field_preserves_only_total_amount(self):
        rng = np.random.default_rng(3)
        scaffold = rng.uniform(0.0, 0.1, (32, 32))
        q1 = 0.3 * scaffold
        q2 = 0.1 * scaffold
        mean_field, mq1, mq2 = scaffold_variant(
            scaffold, q1, q2, "mean_field", 0
        )
        self.assertAlmostEqual(
            float(np.mean(mean_field)), float(np.mean(scaffold))
        )
        self.assertAlmostEqual(float(np.std(mean_field)), 0.0)
        self.assertEqual(float(np.max(np.abs(mq1))), 0.0)
        self.assertEqual(float(np.max(np.abs(mq2))), 0.0)


class Datarium4BBodyHandoffTests(unittest.TestCase):
    def test_body_world_accepts_a_non_disk_builder_phase(self):
        n = 32
        phi = np.zeros((n, n))
        yy, xx = np.mgrid[:n, :n]
        # An off-center, elongated body: deliberately not the authored D4A disk.
        phi[((xx - 11.0) / 5.0) ** 2 + ((yy - 19.0) / 3.0) ** 2 <= 1.0] = 0.9
        world = BodyWorld(
            BodyConfig(
                n=n,
                radius=4.0,
                source_radius=8.0,
                source_sigma=3.5,
                movement_steps=40,
                retention_steps=40,
            ),
            source_angle=0.2,
            initial_phi=phi,
        )
        start = world.center().copy()
        self.assertLess(abs(start[0] - 11.0), 0.5)
        self.assertLess(abs(start[1] - 19.0), 0.5)
        for _ in range(40):
            world.step()
        self.assertTrue(np.isfinite(world.phi).all())
        self.assertGreater(float(np.sum(world.phi)), 1.0)


if __name__ == "__main__":
    unittest.main()
