from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from wrc_park_vision.runtime.backends.base import BackendDetection
from wrc_park_vision.runtime.config import RuntimeConfig
from wrc_park_vision.runtime.modules.detection import DetectionModule
from wrc_park_vision.runtime.pipeline import RuntimePipeline
from wrc_park_vision.runtime.review import ReviewProvider
from wrc_park_vision.runtime.schemas import (
    DetectionSummary,
    VLMBehaviorDecision,
    VLMFinding,
    VLMReviewDecision,
    VLMReviewResult,
    ValidatedImage,
)

from .helpers import FakeBackend, write_test_image


BEHAVIOR_CLASSES = [
    {
        "class_id": 0,
        "class_name": "trampling_grass",
        "required_object_classes": ["person", "grass"],
    },
    {
        "class_id": 1,
        "class_name": "smoking",
        "required_object_classes": ["person", "cigarette"],
    },
    {
        "class_id": 2,
        "class_name": "blocking_fire_lane",
        "required_object_classes": ["vehicle"],
    },
    {
        "class_id": 3,
        "class_name": "standing_or_lying_on_bench",
        "required_object_classes": ["person", "bench"],
    },
]


def make_behavior_config() -> RuntimeConfig:
    return RuntimeConfig.model_validate(
        {
            "modules": [
                {
                    "id": "world",
                    "enabled": True,
                    "type": "detection",
                    "task_group": "object_detection",
                    "backend": "ultralytics",
                    "model_path": Path("world.pt"),
                    "model_id": "world",
                    "expected_class_names": ["object"],
                }
            ],
            "behavior": {
                "enabled": True,
                "module_id": "behavior_pipeline",
                "task_group": "uncivilized_behavior",
                "object_task_group": "uncivilized_behavior",
                "classes": BEHAVIOR_CLASSES,
            },
        }
    )


def behavior_detection(class_id: int, class_name: str, bbox: tuple[int, int, int, int]) -> BackendDetection:
    return BackendDetection(
        class_id,
        class_name,
        0.9,
        bbox,
        task_group="uncivilized_behavior",
    )


class CapturingBehaviorProvider(ReviewProvider):
    def __init__(self, behavior_factory, findings: list[VLMFinding] | None = None) -> None:
        self.behavior_factory = behavior_factory
        self.findings = findings or []
        self.calls = 0
        self.summaries: list[DetectionSummary] = []

    def review(self, image: ValidatedImage, summary: DetectionSummary) -> VLMReviewResult:
        self.calls += 1
        self.summaries.append(summary)
        return VLMReviewResult(
            provider="fake_qwen",
            model_id="fake-qwen-vl",
            duration_ms=1,
            decisions=[
                VLMReviewDecision(observation_id=item.observation_id, verdict="confirmed")
                for item in summary.detections
            ],
            findings=self.findings,
            behaviors=self.behavior_factory(summary),
        )


class BehaviorPipelineTests(unittest.TestCase):
    def run_pipeline(
        self,
        detections: list[BackendDetection],
        provider: ReviewProvider | None,
    ):
        backend = FakeBackend("world", detections)
        module = DetectionModule("world", "object_detection", "world", backend)
        with tempfile.TemporaryDirectory() as directory:
            image_path = write_test_image(Path(directory) / "image.jpg")
            response = RuntimePipeline(
                make_behavior_config(),
                modules=[module],
                review_provider=provider,
            ).process(image_path)
        return response

    def test_person_and_grass_candidate_rejected_does_not_create_behavior(self) -> None:
        def reject_candidate(summary: DetectionSummary) -> list[VLMBehaviorDecision]:
            candidate = summary.behavior_candidates[0]
            return [
                VLMBehaviorDecision(
                    id="behavior-review-0001",
                    candidate_id=candidate.id,
                    class_id=candidate.class_id,
                    class_name=candidate.class_name,
                    verdict="rejected",
                    evidence_observation_ids=candidate.evidence_observation_ids,
                    reasoning="person is next to the lawn, not on it",
                )
            ]

        provider = CapturingBehaviorProvider(reject_candidate)
        response = self.run_pipeline(
            [
                behavior_detection(0, "person", (10, 10, 30, 70)),
                behavior_detection(2, "grass", (0, 50, 100, 80)),
            ],
            provider,
        )

        self.assertEqual(provider.calls, 1)
        assert response.detection_summary is not None
        self.assertEqual(response.detection_summary.behavior_candidates[0].class_name, "trampling_grass")
        self.assertEqual(response.review.behaviors[0].verdict, "rejected")
        self.assertFalse(any(item.kind == "behavior" for item in response.observations))

    def test_confirmed_candidate_creates_behavior_observation(self) -> None:
        def confirm_candidate(summary: DetectionSummary) -> list[VLMBehaviorDecision]:
            candidate = summary.behavior_candidates[0]
            return [
                VLMBehaviorDecision(
                    id="behavior-review-0001",
                    candidate_id=candidate.id,
                    class_id=candidate.class_id,
                    class_name=candidate.class_name,
                    verdict="confirmed",
                    confidence=0.87,
                    evidence_observation_ids=candidate.evidence_observation_ids,
                    reasoning="person is visibly holding and smoking the cigarette",
                )
            ]

        response = self.run_pipeline(
            [
                behavior_detection(0, "person", (10, 10, 40, 70)),
                behavior_detection(3, "cigarette", (30, 20, 35, 25)),
            ],
            CapturingBehaviorProvider(confirm_candidate),
        )

        behavior = next(item for item in response.observations if item.kind == "behavior")
        self.assertEqual(behavior.class_name, "smoking")
        self.assertEqual(behavior.task_group, "uncivilized_behavior")
        self.assertEqual(behavior.confidence, 0.87)
        self.assertIsNone(behavior.geometry)
        self.assertEqual(behavior.source.module_id, "behavior_pipeline")
        self.assertEqual(response.fusion.decisions[-1].action, "add_behavior")
        serialized = response.model_dump(mode="json")
        serialized_behavior = next(item for item in serialized["observations"] if item["kind"] == "behavior")
        self.assertIsNone(serialized_behavior["geometry"])
        self.assertEqual(serialized_behavior["evidence_observation_ids"], behavior.evidence_observation_ids)

    def test_full_image_vlm_can_find_behavior_without_object_detections(self) -> None:
        def full_image_finding(summary: DetectionSummary) -> list[VLMBehaviorDecision]:
            self.assertEqual(summary.behavior_candidates, [])
            return [
                VLMBehaviorDecision(
                    id="behavior-review-0001",
                    candidate_id=None,
                    class_id=2,
                    class_name="blocking_fire_lane",
                    verdict="confirmed",
                    confidence=0.79,
                    reasoning="vehicle blocks a clearly marked fire lane",
                )
            ]

        provider = CapturingBehaviorProvider(full_image_finding)
        response = self.run_pipeline([], provider)

        self.assertEqual(provider.calls, 1)
        behavior = next(item for item in response.observations if item.kind == "behavior")
        self.assertEqual(behavior.class_name, "blocking_fire_lane")
        self.assertEqual(behavior.evidence_observation_ids, [])

    def test_one_vlm_call_can_return_objects_and_multiple_behaviors(self) -> None:
        def behaviors(summary: DetectionSummary) -> list[VLMBehaviorDecision]:
            by_class = {item.class_name: item for item in summary.behavior_candidates}
            return [
                VLMBehaviorDecision(
                    id=f"behavior-review-{index:04d}",
                    candidate_id=by_class[class_name].id,
                    class_id=by_class[class_name].class_id,
                    class_name=class_name,
                    verdict="confirmed",
                    evidence_observation_ids=by_class[class_name].evidence_observation_ids,
                )
                for index, class_name in enumerate(
                    ["trampling_grass", "standing_or_lying_on_bench"],
                    1,
                )
            ]

        provider = CapturingBehaviorProvider(
            behaviors,
            findings=[
                VLMFinding(
                    id="vlm-0001",
                    task_group="garbage",
                    class_id=0,
                    class_name="plastic_drink_bottle",
                    bbox_normalized_xyxy=(0.6, 0.1, 0.8, 0.4),
                )
            ],
        )
        response = self.run_pipeline(
            [
                behavior_detection(0, "person", (10, 10, 40, 70)),
                behavior_detection(1, "bench", (5, 45, 80, 75)),
                behavior_detection(2, "grass", (0, 50, 100, 80)),
            ],
            provider,
        )

        self.assertEqual(provider.calls, 1)
        self.assertEqual(len(response.review.findings), 1)
        self.assertEqual(
            {item.class_name for item in response.observations if item.kind == "behavior"},
            {"trampling_grass", "standing_or_lying_on_bench"},
        )

    def test_vlm_failure_preserves_detections_and_adds_no_behavior(self) -> None:
        class FailingProvider(ReviewProvider):
            def __init__(self) -> None:
                self.calls = 0

            def review(self, image: ValidatedImage, summary: DetectionSummary) -> VLMReviewResult:
                self.calls += 1
                raise TimeoutError("VLM timed out")

        provider = FailingProvider()
        response = self.run_pipeline(
            [
                behavior_detection(0, "person", (10, 10, 40, 70)),
                behavior_detection(2, "grass", (0, 50, 100, 80)),
            ],
            provider,
        )

        self.assertEqual(provider.calls, 1)
        self.assertEqual(response.status, "partial_success")
        self.assertEqual(len(response.observations), 2)
        self.assertFalse(any(item.kind == "behavior" for item in response.observations))
        self.assertTrue(any(error.code == "review_failure" for error in response.errors))

    def test_provider_disabled_keeps_candidate_pending_without_behavior(self) -> None:
        response = self.run_pipeline(
            [
                behavior_detection(0, "person", (10, 10, 40, 70)),
                behavior_detection(2, "grass", (0, 50, 100, 80)),
            ],
            None,
        )

        self.assertEqual(response.status, "success")
        self.assertEqual(response.review.status, "pending")
        self.assertIn("behavior_candidate", response.review.reasons)
        self.assertFalse(any(item.kind == "behavior" for item in response.observations))


if __name__ == "__main__":
    unittest.main()
