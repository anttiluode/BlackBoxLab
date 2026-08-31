import unittest

import numpy as np

from datarium.layers import CoherentAssemblyTracker
from experiments.datarium3_layers import (
    Config,
    MaterialWorld,
    _material_variant,
)


class Datarium3AssemblyObserverTests(unittest.TestCase):
    def test_membership_overlap_preserves_measured_identity(self):
        tracker = CoherentAssemblyTracker(
            agent_count=4,
            period=20.0,
            link_radius=3.0,
        )
        positions = np.asarray(
            [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [15.0, 15.0]]
        )
        headings = np.zeros(4)
        phases = np.zeros(4)
        first = tracker.update(positions, headings, phases)
        second = tracker.update(
            (positions + np.asarray([0.2, 0.0])) % 20.0,
            headings,
            phases,
        )
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].track_id, second[0].track_id)
        self.assertEqual(
            tracker.tracks[first[0].track_id].lifetime_samples,
            2,
        )

    def test_observer_is_not_exposed_by_world(self):
        world = MaterialWorld(Config(n=20, agents=4), seed=1)
        self.assertFalse(hasattr(world, "assembly_tracker"))
        self.assertFalse(hasattr(world, "fitness"))
        self.assertFalse(hasattr(world, "roles"))
        self.assertTrue(np.all(world.frequencies == world.frequencies[0]))


class Datarium3MaterialTests(unittest.TestCase):
    def _coincident_world(self) -> MaterialWorld:
        config = Config(
            n=20,
            agents=4,
            local_smoothing_steps=2,
            build_steps=20,
            probe_steps=20,
        )
        world = MaterialWorld(config, seed=2, quiet_field=True)
        world.positions[:] = np.asarray([10.25, 10.25])
        world.headings[:] = 0.0
        world.phases[:] = 0.0
        # A positive local wave-energy gradient supplies the mechanical gate
        # required for materialization.
        world.envelope[:] = 0.1 + 0.01 * np.arange(config.n)[None, :]
        world.wave_re[:] = 0.05 * np.arange(config.n)[None, :]
        return world

    def test_polymerization_requires_local_phase_and_motion_coherence(self):
        aligned = self._coincident_world()
        aligned_map = aligned.polymerization_map(aligned._local_maps())

        cancelled = self._coincident_world()
        cancelled.phases[:] = np.asarray([0.0, np.pi, 0.0, np.pi])
        cancelled.headings[:] = np.asarray([0.0, np.pi, 0.0, np.pi])
        cancelled_map = cancelled.polymerization_map(cancelled._local_maps())

        self.assertGreater(float(np.max(aligned_map)), 0.05)
        self.assertLess(float(np.max(cancelled_map)), 1e-10)

    def test_traffic_without_a_wave_cannot_write_material(self):
        config = Config(
            n=20,
            agents=4,
            wave_source=0.0,
            build_steps=20,
            probe_steps=20,
        )
        world = MaterialWorld(config, seed=12, quiet_field=True)
        world.positions[:] = np.asarray([10.25, 10.25])
        world.headings[:] = 0.0
        world.phases[:] = 0.0
        for _ in range(20):
            world.step("tensor")
        self.assertEqual(float(np.max(world.matrix)), 0.0)

    def test_write_only_records_history_without_affecting_fast_dynamics(self):
        a = self._coincident_world()
        b = self._coincident_world()
        for _ in range(25):
            a.step("no_memory")
            b.step("write_only")

        self.assertTrue(np.allclose(a.positions, b.positions))
        self.assertTrue(np.allclose(a.headings, b.headings))
        self.assertTrue(np.allclose(a.wave_re, b.wave_re))
        self.assertTrue(np.allclose(a.wave_im, b.wave_im))
        self.assertEqual(float(np.max(a.matrix)), 0.0)
        self.assertGreater(float(np.max(b.matrix)), 0.0)
        self.assertAlmostEqual(
            float(np.mean(b.precursor + b.matrix)),
            1.0,
            places=10,
        )

    def test_mean_field_removes_spatial_and_directional_feedback(self):
        world = self._coincident_world()
        world.matrix[3, 4] = 0.8
        world.q1[3, 4] = 0.4
        world.q2[3, 4] = -0.2
        material, q1, q2 = world.feedback_fields("mean_field")
        self.assertIsInstance(material, float)
        self.assertAlmostEqual(material, float(np.mean(world.matrix)))
        self.assertEqual(q1, 0.0)
        self.assertEqual(q2, 0.0)

    def test_probe_interventions_preserve_or_destroy_the_intended_information(self):
        world = self._coincident_world()
        rng = np.random.default_rng(4)
        world.matrix[:] = rng.uniform(0.0, 0.8, world.matrix.shape)
        angle = rng.uniform(0.0, 2.0 * np.pi, world.matrix.shape)
        world.q1[:] = 0.5 * world.matrix * np.cos(angle)
        world.q2[:] = 0.5 * world.matrix * np.sin(angle)

        intact = _material_variant(world, "intact", seed=8)
        patchwork = _material_variant(world, "patchwork", seed=8)
        scrambled = _material_variant(world, "scrambled", seed=8)
        rotated = _material_variant(world, "rotated", seed=8)
        erased = _material_variant(world, "erased", seed=8)

        self.assertTrue(np.allclose(intact[0], world.matrix))
        self.assertTrue(
            np.allclose(
                np.sort(patchwork[0].ravel()),
                np.sort(world.matrix.ravel()),
            )
        )
        self.assertFalse(np.allclose(patchwork[0], world.matrix))
        self.assertTrue(
            np.allclose(
                np.sort(scrambled[0].ravel()),
                np.sort(world.matrix.ravel()),
            )
        )
        self.assertFalse(np.allclose(scrambled[0], world.matrix))
        self.assertTrue(np.allclose(rotated[1], -world.q1))
        self.assertTrue(np.allclose(rotated[2], -world.q2))
        self.assertEqual(float(np.max(erased[0])), 0.0)


if __name__ == "__main__":
    unittest.main()
