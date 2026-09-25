import unittest

import numpy as np

from src.models.step6_classify import cross_split_keep, map_digest, resize_map


class MapPreparationTest(unittest.TestCase):
    def test_resize_preserves_states_and_padding(self):
        wafer = np.array([[1, 2, 1], [0, 1, 0]], dtype=np.uint8)
        resized = resize_map(wafer, size=6)
        self.assertEqual(resized.shape, (6, 6))
        self.assertTrue(set(np.unique(resized)).issubset({0, 1, 2}))

    def test_digest_includes_shape(self):
        self.assertNotEqual(map_digest(np.ones((2, 3), dtype=np.uint8)),
                            map_digest(np.ones((3, 2), dtype=np.uint8)))

    def test_cross_split_duplicate_removal(self):
        hashes = np.array(["a", "a", "b", "a", "b", "c"])
        splits = np.array(["train", "train", "validation", "validation", "test", "test"])
        keep, removed = cross_split_keep(hashes, splits)
        np.testing.assert_array_equal(keep, [True, True, True, False, False, True])
        self.assertEqual(removed, {"validation": 1, "test": 1})


if __name__ == "__main__":
    unittest.main()
