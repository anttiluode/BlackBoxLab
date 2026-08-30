import unittest

import numpy as np

from datarium.lineage import (
    LineageTracker,
    periodic_components,
    synthetic_torus_blob,
)


class DatariumLineageTests(unittest.TestCase):
    def test_periodic_labelling_joins_the_seam(self):
        mask = np.zeros((16, 16), dtype=bool)
        mask[7:10, 0:2] = True
        mask[7:10, -2:] = True
        comps = periodic_components(mask)
        self.assertEqual(len(comps), 1)
        self.assertEqual(len(comps[0]), 12)

    def test_seam_crossing_keeps_identity(self):
        tracker = LineageTracker(
            (32, 32),
            min_area=5,
            min_mass=1.0,
            min_overlap=0.08,
        )
        ids = []
        for step, x in enumerate((27, 29, 31, 1, 3, 5)):
            phi = synthetic_torus_blob((32, 32), (x, 16), sigma=3.0)
            domains = tracker.update(phi, float(step))
            self.assertEqual(len(domains), 1)
            ids.append(domains[0].track_id)

        self.assertEqual(len(set(ids)), 1)
        self.assertGreaterEqual(tracker.seam_continuations, 1)

    def test_hysteresis_holds_a_real_domain_below_high_threshold(self):
        tracker = LineageTracker(
            (32, 32),
            high_threshold=0.30,
            low_threshold=0.20,
            min_area=5,
            min_mass=1.0,
            min_overlap=0.08,
        )
        first = tracker.update(
            synthetic_torus_blob((32, 32), (16, 16), sigma=4.0, amplitude=1.0),
            0.0,
        )
        second = tracker.update(
            synthetic_torus_blob((32, 32), (16, 16), sigma=4.0, amplitude=0.27),
            1.0,
        )
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(first[0].track_id, second[0].track_id)

    def test_mass_floor_rejects_threshold_dust(self):
        tracker = LineageTracker(
            (24, 24),
            min_area=8,
            min_mass=2.0,
        )
        phi = synthetic_torus_blob((24, 24), (12, 12), sigma=3.0)
        phi[2, 2] = 0.9
        phi[2, 3] = 0.8
        domains = tracker.update(phi, 0.0)
        self.assertEqual(len(domains), 1)
        self.assertGreater(tracker.raw_periodic_count, len(domains))

    def test_split_creates_children_with_parent(self):
        tracker = LineageTracker(
            (48, 48),
            min_area=8,
            min_mass=2.0,
            min_overlap=0.08,
        )

        fields = [
            synthetic_torus_blob((48, 48), (24, 24), sigma=3.0),
            synthetic_torus_blob((48, 48), (22, 24), sigma=3.0)
            + synthetic_torus_blob((48, 48), (26, 24), sigma=3.0),
            synthetic_torus_blob((48, 48), (18, 24), sigma=3.0)
            + synthetic_torus_blob((48, 48), (30, 24), sigma=3.0),
        ]

        parent_id = tracker.update(fields[0], 0.0)[0].track_id
        tracker.update(fields[1], 1.0)
        children = tracker.update(fields[2], 2.0)

        splits = [e for e in tracker.events if e["type"] == "split"]
        self.assertEqual(len(splits), 1)
        self.assertEqual(splits[0]["parents"], [parent_id])
        self.assertEqual(len(children), 2)
        self.assertTrue(all(parent_id in tracker.tracks[d.track_id].parents for d in children))

    def test_merge_creates_child_with_two_parents(self):
        tracker = LineageTracker(
            (48, 48),
            min_area=8,
            min_mass=2.0,
            min_overlap=0.08,
        )

        separated = (
            synthetic_torus_blob((48, 48), (17, 24), sigma=3.0)
            + synthetic_torus_blob((48, 48), (31, 24), sigma=3.0)
        )
        closer = (
            synthetic_torus_blob((48, 48), (20, 24), sigma=3.0)
            + synthetic_torus_blob((48, 48), (28, 24), sigma=3.0)
        )
        joined = synthetic_torus_blob((48, 48), (24, 24), sigma=5.0)

        parents = tracker.update(separated, 0.0)
        self.assertEqual(len(parents), 2)
        parent_ids = sorted(d.track_id for d in parents)
        tracker.update(closer, 1.0)
        child = tracker.update(joined, 2.0)

        merges = [e for e in tracker.events if e["type"] == "merge"]
        self.assertEqual(len(merges), 1)
        self.assertEqual(merges[0]["parents"], parent_ids)
        self.assertEqual(len(child), 1)
        self.assertEqual(sorted(tracker.tracks[child[0].track_id].parents), parent_ids)


if __name__ == "__main__":
    unittest.main()
