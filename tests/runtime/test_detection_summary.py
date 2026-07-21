from __future__ import annotations

import unittest

from wrc_park_vision.runtime.config import ReviewSettings
from wrc_park_vision.runtime.detection_summary import build_detection_summary
from wrc_park_vision.runtime.fusion import merge_and_mark_conflicts

from .helpers import make_observation


class DetectionSummaryTests(unittest.TestCase):
    def test_summary_is_deterministic_and_contains_review_context(self) -> None:
        observations = merge_and_mark_conflicts(
            [
                make_observation("prohibited_items", "prohibited", 0, "spray_can", 0.9, (10, 10, 60, 60)),
                make_observation("garbage", "garbage", 3, "plastic_bottle", 0.3, (10, 10, 60, 60)),
            ],
            0.75,
        )

        summary = build_detection_summary(observations, ReviewSettings())

        self.assertEqual(summary.total_detections, 2)
        self.assertEqual(summary.counts_by_task_group, {"garbage": 1, "prohibited_items": 1})
        self.assertEqual([item.observation_id for item in summary.detections], ["obs-0001", "obs-0002"])
        garbage = summary.detections[0]
        self.assertEqual(garbage.review_reasons, ["low_confidence", "cross_model_overlap"])
        self.assertEqual(garbage.conflict_observation_ids, ["obs-0002"])

    def test_empty_summary_is_valid(self) -> None:
        summary = build_detection_summary([], ReviewSettings())
        self.assertEqual(summary.total_detections, 0)
        self.assertEqual(summary.detections, [])


if __name__ == "__main__":
    unittest.main()
