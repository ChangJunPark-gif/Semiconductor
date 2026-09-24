import unittest

import numpy as np
import pandas as pd

from src.clustering.step5_cluster import choose_dbscan, cluster_metrics


class ClusterSummaryTest(unittest.TestCase):
    def test_noise_and_largest_cluster_exclude_noise_from_denominator(self):
        result = cluster_metrics(np.array([-1, -1, 0, 0, 0, 1]))
        self.assertEqual(result["clusters"], 2)
        self.assertAlmostEqual(result["noise_fraction"], 2 / 6)
        self.assertAlmostEqual(result["largest_cluster_fraction"], 3 / 4)

    def test_dbscan_choice_filters_single_cluster(self):
        candidates = pd.DataFrame({"clusters": [1, 2], "noise_fraction": [0.1, 0.3],
                                   "largest_cluster_fraction": [1.0, 0.6]})
        self.assertEqual(choose_dbscan(candidates), 1)


if __name__ == "__main__":
    unittest.main()
