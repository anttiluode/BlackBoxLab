import unittest

import numpy as np

from datarium.thinker import (
    ACTION_NAMES,
    SENSOR_NAMES,
    Thinker,
    TinyController,
    pairwise_behavior_divergence,
)


class Datarium2ThinkerTests(unittest.TestCase):
    def test_controller_shapes_and_bounds(self):
        rng = np.random.default_rng(1)
        controller = TinyController.random(rng)
        hidden = np.zeros(controller.hidden_size)
        action, hidden2 = controller.step(
            np.zeros(len(SENSOR_NAMES)),
            hidden,
        )
        self.assertEqual(action.shape, (len(ACTION_NAMES),))
        self.assertEqual(hidden2.shape, hidden.shape)
        self.assertTrue(np.all(np.abs(action) <= 1.0))
        self.assertTrue(np.all(np.abs(hidden2) <= 1.0))

    def test_mutation_changes_but_preserves_shape(self):
        rng = np.random.default_rng(2)
        controller = TinyController.random(rng)
        child = controller.mutated(rng, sigma=0.02)
        self.assertEqual(controller.A.shape, child.A.shape)
        self.assertEqual(controller.B.shape, child.B.shape)
        self.assertEqual(controller.C.shape, child.C.shape)
        self.assertGreater(
            np.linalg.norm(controller.flat() - child.flat()),
            0.0,
        )

    def test_blend_lies_between_parents(self):
        rng = np.random.default_rng(3)
        a = TinyController.random(rng)
        b = TinyController.random(rng)
        child = TinyController.blend([a, b], np.asarray([0.25, 0.75]))
        expected = 0.25 * a.flat() + 0.75 * b.flat()
        self.assertTrue(np.allclose(child.flat(), expected))

    def test_behavior_probe_detects_divergent_controllers(self):
        rng = np.random.default_rng(4)
        a = TinyController.random(rng)
        b = TinyController.random(rng)
        # Flip the output map to guarantee a distinct behavior fingerprint.
        b.C = -a.C.copy()
        b.d = -a.d.copy()
        b.A = a.A.copy()
        b.B = a.B.copy()
        b.b = a.b.copy()
        self.assertGreater(
            pairwise_behavior_divergence([a, b]),
            0.1,
        )

    def test_thinker_behavior_vector_is_measured_not_weight_based(self):
        rng = np.random.default_rng(5)
        controller = TinyController.random(rng)
        thinker = Thinker(
            track_id=1,
            controller=controller,
            hidden=np.zeros(controller.hidden_size),
            scout_pos=np.zeros(2),
            scout_vel=np.zeros(2),
            parents=(),
            generation=0,
            born_t=0.0,
        )
        thinker.control_steps = 2
        thinker.action_abs_sum[:] = 1.0
        thinker.sample_sum[:] = 2.0
        vector = thinker.behavior_vector()
        self.assertEqual(vector.shape, (9,))
        self.assertTrue(np.allclose(vector[:5], 0.5))
        self.assertTrue(np.allclose(vector[5:], 1.0))


if __name__ == "__main__":
    unittest.main()
