import unittest

import numpy as np

from datarium.thinker import Thinker, TinyController
from experiments.datarium2_thinkers import create_thinker


def parent(track_id: int, generation: int, score: float, seed: int) -> Thinker:
    rng = np.random.default_rng(seed)
    controller = TinyController.random(rng)
    t = Thinker(
        track_id=track_id,
        controller=controller,
        hidden=np.zeros(controller.hidden_size),
        scout_pos=np.asarray([10.0, 10.0]),
        scout_vel=np.zeros(2),
        parents=(),
        generation=generation,
        born_t=0.0,
    )
    t.control_steps = 10
    t.selection_score = score * 10.0
    return t


class Datarium2EvolutionSemanticsTests(unittest.TestCase):
    def test_split_is_the_only_reproductive_generation_step(self):
        p = parent(1, generation=7, score=0.5, seed=1)
        thinkers = {1: p}
        child = create_thinker(
            track_id=2,
            parents=(1,),
            event_type="split",
            domain_center=np.asarray([12.0, 10.0]),
            t=1.0,
            mode="evolve",
            thinkers=thinkers,
            rng=np.random.default_rng(22),
        )
        self.assertEqual(child.generation, 8)
        self.assertGreater(
            np.linalg.norm(child.controller.flat() - p.controller.flat()),
            0.0,
        )

    def test_merge_selects_parent_program_without_new_generation(self):
        strong = parent(1, generation=9, score=10.0, seed=2)
        weak = parent(2, generation=4, score=-10.0, seed=3)
        thinkers = {1: strong, 2: weak}
        child = create_thinker(
            track_id=3,
            parents=(1, 2),
            event_type="merge",
            domain_center=np.asarray([11.0, 10.0]),
            t=2.0,
            mode="evolve",
            thinkers=thinkers,
            rng=np.random.default_rng(44),
        )
        self.assertEqual(child.generation, strong.generation)
        self.assertTrue(
            np.allclose(child.controller.flat(), strong.controller.flat())
        )

    def test_random_mode_breaks_program_heredity(self):
        p = parent(1, generation=3, score=1.0, seed=4)
        thinkers = {1: p}
        child = create_thinker(
            track_id=2,
            parents=(1,),
            event_type="split",
            domain_center=np.asarray([10.0, 12.0]),
            t=3.0,
            mode="random",
            thinkers=thinkers,
            rng=np.random.default_rng(55),
        )
        self.assertEqual(child.generation, 4)
        self.assertFalse(
            np.allclose(child.controller.flat(), p.controller.flat())
        )


if __name__ == "__main__":
    unittest.main()
