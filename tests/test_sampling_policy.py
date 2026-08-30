import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "organogenesis",
    ROOT / "experiments" / "sampling_policy_organogenesis.py",
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(mod)


class SamplingPolicyTests(unittest.TestCase):
    def test_world_contains_four_simultaneous_ecologies(self):
        world = mod.make_world(128, 0)
        self.assertEqual(world.shape, (128, 4))

    def test_yoked_history_keeps_identical_computation(self):
        old_steps, old_organs = mod.STEPS, mod.ORGANS
        try:
            mod.STEPS = 800
            mod.ORGANS = 4
            row = mod.run("yoked", 0)
        finally:
            mod.STEPS, mod.ORGANS = old_steps, old_organs
        self.assertEqual(row["coverage"], 1)
        self.assertLess(row["computation_divergence"], 1e-12)

    def test_specialization_is_bounded(self):
        self.assertAlmostEqual(
            mod.normalized_specialization(
                __import__("numpy").array([10.0, 0.0, 0.0, 0.0])
            ),
            1.0,
        )
        self.assertAlmostEqual(
            mod.normalized_specialization(
                __import__("numpy").array([10.0, 10.0, 10.0, 10.0])
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
