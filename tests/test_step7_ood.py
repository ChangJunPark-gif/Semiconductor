import unittest

import numpy as np

from src.evaluation.step7_ood import class_mahalanobis, score_metrics


class OODScoreTest(unittest.TestCase):
    def test_class_mahalanobis_prefers_near_known_cluster(self):
        train = np.array([[0.0, 0.0], [0.1, -0.1], [4.0, 4.0], [4.1, 3.9]])
        labels = np.array(["a", "a", "b", "b"])
        scores = class_mahalanobis(train, labels, np.array([[0.0, 0.0], [12.0, 12.0]]))
        self.assertLess(scores[0], scores[1])

    def test_threshold_uses_known_validation_only(self):
        result = score_metrics(np.array([0.0, 1.0, 2.0, 3.0]),
                               np.array([0.0, 1.0, 4.0, 5.0]),
                               np.array([False, False, True, True]))
        self.assertAlmostEqual(result["validation_threshold"], 2.85)
        self.assertEqual(result["test_known_false_alarm_rate"], 0)
        self.assertEqual(result["test_heldout_detection_rate"], 1)


if __name__ == "__main__":
    unittest.main()
