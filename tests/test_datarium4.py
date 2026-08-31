import unittest

from experiments.datarium4_body import Config, retention_trial, run_movement_arm


class Datarium4BodyPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config(
            n=32,
            radius=4.0,
            source_radius=8.0,
            source_sigma=3.5,
            movement_steps=250,
            retention_steps=250,
        )
        cls.angle = 0.37

    def test_asymmetric_local_chemistry_can_move_the_phase_toward_source(self):
        row = run_movement_arm("active", self.angle, self.cfg)
        self.assertGreater(row["toward_source"], 2.0)
        self.assertLess(row["perpendicular"], 0.2)
        self.assertGreater(row["chemistry_polarity"], 0.05)

    def test_chemistry_without_active_stress_does_not_translate_body(self):
        row = run_movement_arm("no_stress", self.angle, self.cfg)
        self.assertGreater(row["chemistry_polarity"], 0.05)
        self.assertLess(abs(row["toward_source"]), 1e-3)

    def test_uniform_food_removes_preferred_direction(self):
        row = run_movement_arm("uniform_food", self.angle, self.cfg)
        self.assertLess(abs(row["toward_source"]), 1e-3)
        self.assertLess(abs(row["chemistry_polarity"]), 0.01)

    def test_pinned_phase_cannot_translate_even_when_chemistry_polarizes(self):
        row = run_movement_arm("pinned", self.angle, self.cfg)
        self.assertGreater(row["chemistry_polarity"], 0.05)
        self.assertLess(abs(row["toward_source"]), 1e-3)

    def test_selective_interface_retains_more_passive_cargo(self):
        selective = retention_trial(self.cfg, self.cfg.membrane_barrier)
        open_control = retention_trial(self.cfg, 0.0)
        self.assertGreater(
            selective["retained_inside_fraction"],
            open_control["retained_inside_fraction"] + 0.15,
        )
        self.assertAlmostEqual(selective["cargo_conservation"], 1.0, places=10)
        self.assertAlmostEqual(open_control["cargo_conservation"], 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
