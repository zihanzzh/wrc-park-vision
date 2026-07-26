from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from wrc_park_vision.runtime.competition import build_competition_response
from wrc_park_vision.runtime.fusion import fuse_review_results
from wrc_park_vision.runtime.schemas import (
    BBoxGeometry,
    Observation,
    ObservationSource,
    RequestContext,
    ReviewSummary,
    VLMFinding,
    VLMReviewDecision,
)

from .helpers import make_observation, make_response, write_test_image


class CompetitionResponseTests(unittest.TestCase):
    def test_adapter_uses_final_fused_objects_and_frame_metadata(self) -> None:
        detector = make_observation(
            "prohibited_items", "detector", 0, "spray_can", 0.61, (10, 10, 30, 40)
        )
        detector.id = "obs-0001"
        rejected = make_observation(
            "prohibited_items", "detector", 1, "speaker", 0.55, (40, 10, 70, 40)
        )
        rejected.id = "obs-0002"
        review = ReviewSummary(
            attempted=True,
            status="completed",
            provider="fake_vlm",
            model_id="fake-vl",
            decisions=[
                VLMReviewDecision(
                    observation_id="obs-0001",
                    verdict="corrected",
                    corrected_task_group="garbage",
                    corrected_class_id=3,
                    corrected_class_name="plastic_drink_bottle",
                    confidence=0.92,
                ),
                VLMReviewDecision(
                    observation_id="obs-0002",
                    verdict="rejected",
                    confidence=0.88,
                ),
            ],
            findings=[
                VLMFinding(
                    id="vlm-multi-0001",
                    task_group="garbage",
                    class_id=2,
                    class_name="empty_cigarette_box",
                    confidence=0.81,
                    bbox_normalized_xyxy=(0.7, 0.5, 0.9, 0.8),
                    review_pass="multi_image",
                    geometry_source="vlm_multi_image",
                    geometry=BBoxGeometry.from_xyxy((70, 40, 90, 64), 100, 80),
                )
            ],
        )
        observations, fusion = fuse_review_results([detector, rejected], review)

        with tempfile.TemporaryDirectory() as directory:
            response = make_response(
                observations,
                write_test_image(Path(directory) / "image.jpg"),
            )
        response.review = review
        response.fusion = fusion
        response.input.context = RequestContext(
            frame_index=42,
            timestamp=datetime(2026, 7, 25, tzinfo=timezone.utc),
        )

        sdk = build_competition_response(response)

        self.assertEqual(sdk.frame.frame_id, 42)
        self.assertEqual(len(sdk.objects), 2)
        corrected = sdk.objects[0]
        self.assertEqual(corrected.class_name, "plastic_drink_bottle")
        self.assertEqual(corrected.review_status, "corrected")
        self.assertEqual(corrected.source, "detector_vlm_fused")
        self.assertEqual(corrected.confidence, 0.92)
        self.assertEqual(corrected.bbox_xyxy, (10.0, 10.0, 30.0, 40.0))
        self.assertEqual(sdk.objects[1].source, "vlm_finding")
        self.assertNotIn("speaker", [item.class_name for item in sdk.objects])

    def test_detector_only_behavior_and_degraded_response(self) -> None:
        detector = make_observation(
            "garbage",
            "garbage",
            3,
            "plastic_drink_bottle",
            0.87,
            (10, 10, 30, 40),
        )
        detector.id = "obs-0001"
        behavior = Observation(
            id="behavior-0001",
            kind="behavior",
            task_group="uncivilized_behavior",
            class_id=0,
            class_name="trampling_grass",
            confidence=0.9,
            source=ObservationSource(
                module_id="behavior_pipeline",
                backend="fake_vlm",
                model_id="fake-vl",
            ),
            geometry=BBoxGeometry.from_xyxy((20, 10, 60, 70), 100, 80),
            evidence_observation_ids=["obs-0001"],
        )
        with tempfile.TemporaryDirectory() as directory:
            response = make_response(
                [detector, behavior],
                write_test_image(Path(directory) / "image.jpg"),
            )
        response.status = "partial_success"

        sdk = build_competition_response(response)

        self.assertTrue(sdk.degraded)
        self.assertEqual(sdk.status, "partial_success")
        self.assertEqual(sdk.objects[0].confidence, 0.87)
        self.assertEqual(sdk.objects[0].review_status, "not_required")
        self.assertEqual(sdk.objects[0].source, "detector")
        self.assertEqual(sdk.behaviors[0].evidence_object_ids, ["obs-0001"])
        self.assertEqual(
            set(sdk.behaviors[0].model_dump()),
            {
                "class_id",
                "class_name",
                "confidence",
                "evidence_object_ids",
            },
        )

    def test_review_statuses_and_confidence_rules_are_explicit(self) -> None:
        confirmed = make_observation(
            "garbage",
            "garbage",
            3,
            "plastic_drink_bottle",
            0.71,
            (10, 10, 30, 40),
        )
        confirmed.id = "obs-confirmed"
        confirmed.review.status = "confirmed"
        confirmed.metadata["vlm_review_confidence"] = 0.91

        uncertain = make_observation(
            "prohibited_items",
            "detector",
            0,
            "spray_can",
            0.42,
            (35, 10, 55, 40),
        )
        uncertain.id = "obs-uncertain"
        uncertain.review.required = True
        uncertain.review.status = "pending"
        uncertain.review.reasons = ["vlm_uncertain"]
        uncertain.metadata["vlm_review_confidence"] = 0.51

        review_failed = make_observation(
            "prohibited_items",
            "detector",
            1,
            "speaker",
            0.36,
            (60, 10, 80, 40),
        )
        review_failed.id = "obs-review-failed"
        review_failed.review.required = True
        review_failed.review.status = "failed"
        review_failed.review.reasons = ["review_item_missing_or_failed"]

        with tempfile.TemporaryDirectory() as directory:
            response = make_response(
                [confirmed, uncertain, review_failed],
                write_test_image(Path(directory) / "image.jpg"),
            )

        sdk = build_competition_response(response)
        by_id = {item.object_id: item for item in sdk.objects}

        self.assertEqual(by_id["obs-confirmed"].review_status, "confirmed")
        self.assertEqual(by_id["obs-confirmed"].confidence, 0.91)
        self.assertEqual(
            by_id["obs-confirmed"].source,
            "detector_vlm_fused",
        )
        self.assertEqual(by_id["obs-uncertain"].review_status, "uncertain")
        self.assertEqual(by_id["obs-uncertain"].confidence, 0.42)
        self.assertEqual(
            by_id["obs-review-failed"].review_status,
            "review_failed",
        )
        self.assertEqual(
            by_id["obs-review-failed"].confidence,
            0.36,
        )


if __name__ == "__main__":
    unittest.main()
