import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "stigmergy",
    ROOT / "experiments" / "stigmergic_field.py",
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


class StigmergicFieldTests(unittest.TestCase):
    def test_mixing_is_continuous_and_mixed(self):
        a = mod.mixing_matrix(64)
        self.assertEqual(a.shape, (64, 4))
        self.assertTrue((a > 0).all())
        self.assertGreater((a > 0.15).sum(axis=1).mean(), 1.0)

    def test_no_source_labels_enter_policy(self):
        agent = mod.Agent.make(1)
        candidates = mod.candidate_positions(agent.position)
        self.assertEqual(len(candidates), 2 * mod.MOVE_RADIUS + 1)
        self.assertEqual(agent.values.shape, (mod.CELLS,))

    def test_trace_is_local_and_persistent(self):
        trace = __import__("numpy").zeros(mod.CELLS)
        mod.write_trace(trace, 0)
        self.assertGreater(trace[0], trace[8])
        self.assertGreater(trace.sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
