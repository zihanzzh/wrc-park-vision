from __future__ import annotations

import unittest

from wrc_park_vision.runtime.fusion import bbox_iou, fuse_review_results, merge_and_mark_conflicts
from wrc_park_vision.runtime.schemas import ReviewSummary, VLMFinding, VLMReviewDecision

from .helpers import make_observation


class FusionTests(unittest.TestCase):
    def test_cross_task_overlap_is_preserved_and_marked(self) -> None:
        first = make_observation("prohibited_items", "prohibited", 0, "spray_can", 0.9, (10, 10, 60, 60))
        second = make_observation("garbage", "garbage", 3, "plastic_bottle", 0.8, (10, 10, 60, 60))
        merged = merge_and_mark_conflicts([first, second], 0.75)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].id, "obs-0001")
        self.assertEqual(merged[1].id, "obs-0002")
        self.assertEqual(merged[0].conflicts[0].observation_id, merged[1].id)
        self.assertEqual(merged[1].conflicts[0].observation_id, merged[0].id)

    def test_same_task_group_is_not_cross_model_conflict(self) -> None:
        first = make_observation("garbage", "one", 1, "box", 0.8, (10, 10, 60, 60))
        second = make_observation("garbage", "two", 2, "bottle", 0.9, (10, 10, 60, 60))
        merged = merge_and_mark_conflicts([second, first], 0.1)
        self.assertTrue(all(not observation.conflicts for observation in merged))
        self.assertEqual([observation.class_name for observation in merged], ["box", "bottle"])

    def test_iou(self) -> None:
        first = make_observation("a", "a", 0, "a", 1.0, (0, 0, 50, 50)).geometry
        second = make_observation("b", "b", 0, "b", 1.0, (25, 25, 75, 75)).geometry
        self.assertAlmostEqual(bbox_iou(first, second), 625 / 4375)

    def test_review_fusion_applies_rejection_and_correction(self) -> None:
        observations = merge_and_mark_conflicts(
            [
                make_observation("garbage", "garbage", 0, "paper", 0.7, (10, 10, 30, 30)),
                make_observation("prohibited", "prohibited", 0, "spray", 0.8, (40, 10, 60, 30)),
            ],
            0.75,
        )
        review = ReviewSummary(
            attempted=True,
            status="completed",
            decisions=[
                VLMReviewDecision(
                    observation_id="obs-0001",
                    verdict="rejected",
                    confidence=0.91,
                    reasoning="not garbage",
                ),
                VLMReviewDecision(
                    observation_id="obs-0002",
                    verdict="corrected",
                    corrected_task_group="prohibited",
                    corrected_class_id=1,
                    corrected_class_name="stove",
                    confidence=0.88,
                ),
            ],
            findings=[VLMFinding(id="vlm-0001", task_group="garbage", class_name="bottle")],
        )

        finalized, fusion = fuse_review_results(observations, review)

        self.assertEqual([item.class_name for item in finalized], ["stove"])
        self.assertEqual(finalized[0].task_group, "prohibited")
        self.assertEqual(finalized[0].class_id, 1)
        self.assertEqual(finalized[0].confidence, 0.8)
        self.assertEqual(
            finalized[0].geometry.bbox_xyxy,
            observations[1].geometry.bbox_xyxy,
        )
        self.assertEqual(finalized[0].metadata["original_class_name"], "spray")
        self.assertEqual(finalized[0].metadata["vlm_review_confidence"], 0.88)
        self.assertEqual(
            [item.action for item in fusion.decisions],
            ["reject_yolo", "correct_yolo", "add_vlm_finding"],
        )
        self.assertEqual(fusion.decisions[1].final_class_name, "stove")
        self.assertEqual(fusion.decisions[0].yolo_confidence, 0.7)
        self.assertEqual(fusion.decisions[0].vlm_confidence, 0.91)
        self.assertEqual(fusion.decisions[1].yolo_confidence, 0.8)
        self.assertEqual(fusion.decisions[1].vlm_confidence, 0.88)
        self.assertEqual(fusion.decisions[2].geometry_source, "none")

    def test_uncertain_is_kept_and_flagged_by_default(self) -> None:
        observations = merge_and_mark_conflicts(
            [make_observation("garbage", "garbage", 0, "paper", 0.4, (10, 10, 30, 30))],
            0.75,
        )
        review = ReviewSummary(
            attempted=True,
            status="completed",
            decisions=[
                VLMReviewDecision(
                    observation_id="obs-0001",
                    verdict="uncertain",
                    confidence=0.35,
                )
            ],
        )

        finalized, fusion = fuse_review_results(observations, review)

        self.assertEqual(len(finalized), 1)
        self.assertEqual(finalized[0].review.status, "pending")
        self.assertIn("vlm_uncertain", finalized[0].review.reasons)
        self.assertEqual(fusion.decisions[0].action, "keep_uncertain")

    def test_confirmed_keeps_yolo_geometry_and_both_confidences(self) -> None:
        observations = merge_and_mark_conflicts(
            [make_observation("garbage", "garbage", 0, "paper", 0.76, (10, 10, 30, 30))],
            0.75,
        )
        review = ReviewSummary(
            attempted=True,
            status="completed",
            decisions=[
                VLMReviewDecision(
                    observation_id="obs-0001",
                    verdict="confirmed",
                    confidence=0.92,
                )
            ],
        )

        finalized, fusion = fuse_review_results(observations, review)

        self.assertEqual(finalized[0].geometry, observations[0].geometry)
        self.assertEqual(finalized[0].confidence, 0.76)
        self.assertEqual(fusion.decisions[0].action, "keep_yolo")
        self.assertEqual(fusion.decisions[0].yolo_confidence, 0.76)
        self.assertEqual(fusion.decisions[0].vlm_confidence, 0.92)

    def test_uncertain_can_be_dropped_by_policy(self) -> None:
        observations = merge_and_mark_conflicts(
            [make_observation("garbage", "garbage", 0, "paper", 0.4, (10, 10, 30, 30))],
            0.75,
        )
        review = ReviewSummary(
            attempted=True,
            status="completed",
            uncertain_policy="drop",
            decisions=[
                VLMReviewDecision(
                    observation_id="obs-0001",
                    verdict="uncertain",
                    confidence=0.35,
                )
            ],
        )

        finalized, fusion = fuse_review_results(observations, review)

        self.assertEqual(finalized, [])
        self.assertEqual(fusion.decisions[0].action, "drop_uncertain")

    def test_review_failure_defaults_to_keep_flagged(self) -> None:
        observations = merge_and_mark_conflicts(
            [make_observation("garbage", "garbage", 0, "paper", 0.4, (10, 10, 30, 30))],
            0.75,
        )
        review = ReviewSummary(
            required=True,
            attempted=True,
            status="failed",
        )

        finalized, fusion = fuse_review_results(observations, review)

        self.assertEqual(review.review_failure_policy, "keep_flagged")
        self.assertEqual(len(finalized), 1)
        self.assertEqual(finalized[0].review.status, "pending")
        self.assertIn("review_item_missing_or_failed", finalized[0].review.reasons)
        self.assertEqual(fusion.decisions[0].action, "keep_review_failed")


if __name__ == "__main__":
    unittest.main()
