from __future__ import annotations

import unittest

from wrc_park_vision.runtime.config import ReviewSettings
from wrc_park_vision.runtime.detection_summary import build_detection_summary
from wrc_park_vision.runtime.fusion import merge_and_mark_conflicts
from wrc_park_vision.runtime.review import ReviewCoordinator, ReviewPolicy, ReviewProvider
from wrc_park_vision.runtime.schemas import (
    DetectionSummary,
    ModuleSummary,
    Observation,
    VLMFinding,
    VLMReviewDecision,
    VLMReviewResult,
    ValidatedImage,
)

from .helpers import make_observation


class ReviewTests(unittest.TestCase):
    def test_low_confidence_and_overlap_review(self) -> None:
        low = make_observation("garbage", "garbage", 1, "box", 0.3, (10, 10, 60, 60))
        other = make_observation("prohibited", "prohibited", 0, "spray", 0.9, (10, 10, 60, 60))
        merged = merge_and_mark_conflicts([low, other], 0.75)
        modules = [ModuleSummary(module_id="garbage", task_group="garbage", status="success", duration_ms=1)]
        reviewed, summary = ReviewPolicy(ReviewSettings()).apply(merged, modules)
        low_result = next(item for item in reviewed if item.class_name == "box")
        self.assertEqual(low_result.review.status, "pending")
        self.assertEqual(low_result.review.reasons, ["low_confidence", "cross_model_overlap"])
        self.assertEqual(summary.reasons, ["low_confidence", "cross_model_overlap"])

    def test_module_failure_only_changes_top_review(self) -> None:
        observation = make_observation("garbage", "garbage", 1, "box", 0.9, (10, 10, 20, 20))
        modules = [
            ModuleSummary(module_id="garbage", task_group="garbage", status="success", duration_ms=1),
            ModuleSummary(module_id="prohibited", task_group="prohibited", status="failure", duration_ms=1, error="x"),
        ]
        reviewed, summary = ReviewPolicy(ReviewSettings()).apply([observation], modules)
        self.assertEqual(reviewed[0].review.status, "not_required")
        self.assertEqual(summary.reasons, ["module_failure"])

    def test_review_provider_is_an_optional_extension_point(self) -> None:
        class FakeReviewProvider(ReviewProvider):
            def __init__(self) -> None:
                self.calls = 0

            def review(self, image: ValidatedImage, summary: DetectionSummary) -> VLMReviewResult:
                self.calls += 1
                return VLMReviewResult(
                    provider="fake",
                    model_id="fake-vl",
                    duration_ms=1,
                    decisions=[
                        VLMReviewDecision(observation_id="obs-0001", verdict="confirmed"),
                    ],
                )

        provider = FakeReviewProvider()
        coordinator = ReviewCoordinator(ReviewSettings(low_confidence_threshold=0.5), provider)
        observation = make_observation("garbage", "garbage", 1, "box", 0.3, (10, 10, 20, 20))
        modules = [ModuleSummary(module_id="garbage", task_group="garbage", status="success", duration_ms=1)]
        image = ValidatedImage("image.jpg", object(), 100, 80)

        prepared = merge_and_mark_conflicts([observation], 0.75)
        reviewed, summary = coordinator.apply(
            image,
            prepared,
            modules,
            build_detection_summary(prepared, ReviewSettings(low_confidence_threshold=0.5)),
        )

        self.assertEqual(provider.calls, 1)
        self.assertEqual(reviewed[0].review.status, "confirmed")
        self.assertTrue(summary.attempted)

    def test_provider_reviews_full_image_even_without_detections(self) -> None:
        class FakeReviewProvider(ReviewProvider):
            def __init__(self) -> None:
                self.image = None
                self.calls = 0

            def review(self, image: ValidatedImage, summary: DetectionSummary) -> VLMReviewResult:
                self.image = image
                self.calls += 1
                return VLMReviewResult(
                    provider="fake",
                    model_id="fake-vl",
                    duration_ms=1,
                    findings=[
                        VLMFinding(
                            id="vlm-0001",
                            task_group="garbage",
                            class_name="box",
                            reasoning="YOLO missed it",
                        )
                    ],
                )

        provider = FakeReviewProvider()
        coordinator = ReviewCoordinator(ReviewSettings(), provider)
        image = ValidatedImage("image.jpg", object(), 100, 80)
        reviewed, summary = coordinator.apply(
            image,
            [],
            [ModuleSummary(module_id="garbage", task_group="garbage", status="success", duration_ms=1)],
            DetectionSummary(total_detections=0),
        )

        self.assertEqual(provider.calls, 1)
        self.assertEqual(reviewed, [])
        self.assertIs(provider.image, image)
        self.assertTrue(summary.attempted)
        self.assertEqual(summary.findings[0].geometry, None)


if __name__ == "__main__":
    unittest.main()
