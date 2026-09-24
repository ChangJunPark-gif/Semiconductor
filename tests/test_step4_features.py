import unittest

import numpy as np

from src.features.step4_features import assign_lot_split, extract_features


class FeatureExtractionTest(unittest.TestCase):
    def test_edge_failure_has_positive_edge_enrichment(self) -> None:
        wafer = np.ones((9, 9), dtype=np.uint8)
        wafer[0, :] = 2
        features = extract_features(wafer, moran_i=0.2, fail_join_excess_per_edge=0.1)
        self.assertGreater(features["edge_enrichment"], 0)
        self.assertLess(features["center_enrichment"], 0)
        self.assertEqual(features["component_count"], 1)
        self.assertEqual(features["largest_component_size"], 9)

    def test_two_disconnected_fail_dies_make_two_components(self) -> None:
        wafer = np.ones((7, 7), dtype=np.uint8)
        wafer[1, 1] = 2
        wafer[5, 5] = 2
        features = extract_features(wafer, moran_i=0, fail_join_excess_per_edge=0)
        self.assertEqual(features["component_count"], 2)
        self.assertEqual(features["largest_component_fail_share"], 0.5)

    def test_split_assigns_each_lot_once_and_is_repeatable(self) -> None:
        lots = np.array([f"lot{i}" for i in range(100)])
        first = assign_lot_split(lots, seed=42)
        self.assertEqual(first, assign_lot_split(lots, seed=42))
        self.assertEqual(len(first), 100)
        self.assertEqual(set(first.values()), {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
