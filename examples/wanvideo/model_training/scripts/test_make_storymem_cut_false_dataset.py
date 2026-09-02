#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("make_storymem_cut_false_dataset.py")
SPEC = importlib.util.spec_from_file_location("make_storymem_cut_false_dataset", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class StoryMemTimelineTest(unittest.TestCase):
    def test_memory_samples_are_strictly_before_boundary(self):
        times = MODULE.interval_center_times(0.0, 0.6, 3)
        for actual, expected in zip(times, [0.1, 0.3, 0.5]):
            self.assertAlmostEqual(actual, expected)
        self.assertTrue(all(0.0 <= value < 0.6 for value in times))

    def test_first_supervised_frame_is_after_boundary(self):
        current_start = MODULE.validate_continuous_timeline(0.0, 0.5, 0.5, 4.0, 16.0)
        self.assertEqual(current_start, 0.5625)

    def test_non_continuous_boundary_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same boundary frame"):
            MODULE.validate_continuous_timeline(0.0, 0.5, 0.7, 4.0, 16.0)

    def test_empty_memory_interval_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "greater than --memory-start"):
            MODULE.validate_continuous_timeline(0.5, 0.5, 0.5, 4.0, 16.0)

    def test_target_shot_memory_time_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the memory-only interval"):
            MODULE.validate_memory_times([0.25, 1.25], 0.0, 1.0)

    def test_memory_image_count_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            MODULE.interval_center_times(0.0, 1.0, 0)


if __name__ == "__main__":
    unittest.main()
