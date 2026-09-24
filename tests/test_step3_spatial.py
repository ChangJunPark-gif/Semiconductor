import unittest

import numpy as np

from src.spatial.step3_moran import permutation_test, spatial_statistics


class SpatialStatisticsTest(unittest.TestCase):
    def test_checkerboard_has_negative_unit_moran(self) -> None:
        checkerboard = np.array([[1, 2], [2, 1]], dtype=np.uint8)
        stats = spatial_statistics(checkerboard)
        self.assertAlmostEqual(stats["moran_i"], -1)
        self.assertEqual(stats["fail_fail_edges"], 0)

    def test_cluster_has_positive_moran(self) -> None:
        clustered = np.array(
            [[2, 2, 1, 1]] * 4, dtype=np.uint8
        )
        self.assertGreater(spatial_statistics(clustered)["moran_i"], 0)

    def test_constant_map_is_undefined(self) -> None:
        self.assertTrue(np.isnan(spatial_statistics(np.ones((3, 3)))["moran_i"]))

    def test_permutation_keeps_mask_and_is_reproducible(self) -> None:
        wafer = np.array(
            [[0, 1, 1, 0], [1, 2, 2, 1], [1, 2, 1, 1], [0, 1, 1, 0]],
            dtype=np.uint8,
        )
        first = permutation_test(wafer, permutations=99, seed=7)
        second = permutation_test(wafer, permutations=99, seed=7)
        np.testing.assert_array_equal(first[3], second[3])
        self.assertGreaterEqual(first[0], 0.01)
        self.assertLessEqual(first[0], 1)


if __name__ == "__main__":
    unittest.main()
